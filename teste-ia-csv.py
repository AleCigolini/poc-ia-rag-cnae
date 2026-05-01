# =================================
# 1. IMPORTS
# =================================
import csv
import hashlib
import io
import json
import os

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =================================
# 2. GENERAL SETTINGS
# =================================
BASE_PERSIST_DIRECTORY = "./chroma_cnae_csv"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CSV_FILE_PATH = "./cnae_exemplo.csv"


@st.cache_data
def carregar_documentos_csv(csv_bytes):
    """
    Load CSV rows and convert each row into one semantic chunk.
    For tabular files, row-level chunks preserve column meaning.
    """
    decoded = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))

    documentos = []
    for idx, row in enumerate(reader, start=2):
        cnae = (row.get("CNAE") or "").strip()
        descricao = (row.get("Descricao") or row.get("Descrição") or "").strip()
        anexo = (row.get("Anexo") or "").strip()
        fator_r = (row.get("Fator R") or "").strip()
        aliquota = (row.get("Aliquota") or row.get("Alíquota") or "").strip()
        contabilizei = (row.get("Contabilizei") or "").strip()

        texto_linha = (
            f"CNAE: {cnae} | "
            f"Descricao: {descricao} | "
            f"Anexo: {anexo} | "
            f"Fator R: {fator_r} | "
            f"Aliquota: {aliquota} | "
            f"Contabilizei: {contabilizei}"
        )

        metadata = {
            "linha_csv": idx,
            "cnae": cnae,
            "descricao": descricao,
            "anexo": anexo,
            "fator_r": fator_r,
            "aliquota": aliquota,
            "contabilizei": contabilizei,
            "categoria": "cnae",
        }

        documentos.append(Document(page_content=texto_linha, metadata=metadata))

    return documentos


def gerar_chunks(documentos_csv):
    """
    CSV-aware chunking: one row = one chunk.
    This avoids breaking columns across arbitrary text splits.
    """
    return documentos_csv


def gerar_dataset_id(csv_bytes):
    return hashlib.sha256(csv_bytes).hexdigest()[:12]


@st.cache_resource
def criar_vectorstore(_chunks, dataset_id):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_directory = os.path.join(BASE_PERSIST_DIRECTORY, dataset_id)

    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

    return vectorstore


def rerank_documentos(pergunta, documentos, llm):
    """
    Reorder retrieved rows with LLM-based semantic scoring.
    """
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""Avalie a relevancia da linha abaixo para responder a pergunta.

Pergunta: {pergunta}

Linha CSV:
{texto}

Responda SOMENTE com um numero decimal entre 0 e 10 (10 = altamente relevante, 0 = irrelevante).""",
    )

    documentos_com_score = []
    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(pergunta=pergunta, texto=doc.page_content)
        ).content

        try:
            score = float(str(score).replace(",", "."))
        except ValueError:
            score = 0.0

        documentos_com_score.append((score, doc))

    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda item: item[0],
        reverse=True,
    )
    return [doc for _, doc in documentos_ordenados]


def documento_para_json(doc):
    return {
        "CNAE": doc.metadata.get("cnae", ""),
        "Descricao": doc.metadata.get("descricao", ""),
        "Anexo": doc.metadata.get("anexo", ""),
        "Fator R": doc.metadata.get("fator_r", ""),
        "Aliquota": doc.metadata.get("aliquota", ""),
        "Contabilizei": doc.metadata.get("contabilizei", ""),
    }


def responder_pergunta(pergunta, vectorstore):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    documentos_recuperados = vectorstore.similarity_search(pergunta, k=10)

    documentos_rerankeados = rerank_documentos(pergunta, documentos_recuperados, llm)
    contexto_final = documentos_rerankeados[:10]

    linhas_relevantes = [documento_para_json(doc) for doc in contexto_final]
    contexto_json = json.dumps(linhas_relevantes, ensure_ascii=False, indent=2)

    prompt_final = f"""Voce e um assistente especializado em consulta de CNAE.
Responda com base EXCLUSIVAMENTE nos itens JSON fornecidos abaixo.
Se a informacao nao estiver nos itens, retorne um array vazio: []

Retorne SOMENTE JSON valido, sem markdown e sem texto adicional.

Regras obrigatorias:
- Se houver exatamente 1 item correspondente, retorne um unico objeto JSON.
- Se houver 2 ou mais itens correspondentes, retorne um array de objetos JSON.
- Cada objeto deve conter EXATAMENTE estas chaves:
  "CNAE", "Descricao", "Anexo", "Fator R", "Aliquota", "Contabilizei"
- Nao invente valores e nao altere os valores dos itens fornecidos.
- Retorne apenas os itens que realmente respondem a pergunta.

Itens disponiveis:
{contexto_json}

Pergunta: {pergunta}
"""

    resposta = llm.invoke(prompt_final).content.strip()
    try:
        resposta_json = json.loads(resposta)
    except json.JSONDecodeError:
        if len(linhas_relevantes) == 1:
            resposta_json = linhas_relevantes[0]
        else:
            resposta_json = linhas_relevantes

    return resposta_json, contexto_final


st.set_page_config(page_title="CHAT CNAE com RAG (CSV)", layout="wide")
st.title("CHAT CNAE - RAG com CSV")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Defina a variavel de ambiente OPENAI_API_KEY antes de usar o app.")

csv_disponivel = os.path.exists(CSV_FILE_PATH)
if not csv_disponivel:
    st.error(f"Arquivo CSV nao encontrado em: {CSV_FILE_PATH}")

st.caption(f"CSV utilizado automaticamente: {CSV_FILE_PATH}")
pergunta = st.text_input("Digite sua pergunta sobre CNAE:")

if csv_disponivel and pergunta and api_key:
    with st.spinner("Processando CSV e consultando dados..."):
        with open(CSV_FILE_PATH, "rb") as arquivo_csv:
            csv_bytes = arquivo_csv.read()
        documentos = carregar_documentos_csv(csv_bytes)
        chunks = gerar_chunks(documentos)
        dataset_id = gerar_dataset_id(csv_bytes)
        vectorstore = criar_vectorstore(chunks, dataset_id)

        resposta, fontes = responder_pergunta(pergunta, vectorstore)

    st.subheader("Resposta")
    st.json(resposta)

    st.subheader("Fontes utilizadas")
    for i, doc in enumerate(fontes, start=1):
        st.markdown(f"**Linha relevante {i}**")
        st.write(f"Linha no CSV: {doc.metadata.get('linha_csv')}")
        st.write(f"CNAE: {doc.metadata.get('cnae')}")
        st.write(f"Descricao: {doc.metadata.get('descricao')}")
        st.write(f"Anexo: {doc.metadata.get('anexo')}")
        st.write(f"Fator R: {doc.metadata.get('fator_r')}")
        st.write(f"Aliquota: {doc.metadata.get('aliquota')}")
        st.write(f"Contabilizei: {doc.metadata.get('contabilizei')}")
        st.write(doc.page_content)
        st.divider()

elif pergunta and not csv_disponivel:
    st.info("Adicione o arquivo cnae_exemplo.csv no diretorio do projeto para habilitar a consulta.")
