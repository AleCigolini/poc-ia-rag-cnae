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
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Injeta a chave como variável de ambiente
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =================================
# 2. GENERAL SETTINGS
# =================================
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CSV_FILE_PATH = "./cnae-exemplo-2.csv"


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
        item_lc116 = (row.get("Item LC 116") or "").strip()
        descricao_item = (row.get("Descrição Item") or "").strip()
        codigo_nacional = (row.get("CÓDIGO NACIONAL") or "").strip()
        cnae = (row.get("CNAE") or "").strip()
        nbs = (row.get("NBS") or "").strip()
        descricao_nbs = (row.get("DESCRIÇÃO NBS") or "").strip()
        cclass_trib = (row.get("cClassTrib") or "").strip()
        nome_cclass_trib = (row.get("nome cClassTrib") or "").strip()
        onda_cadastro = (row.get("ONDA DE CADASTRO") or "").strip()

        texto_linha = (
            f"Item LC 116: {item_lc116} | "
            f"Descrição Item: {descricao_item} | "
            f"CÓDIGO NACIONAL: {codigo_nacional} | "
            f"CNAE: {cnae} | "
            f"NBS: {nbs} | "
            f"DESCRIÇÃO NBS: {descricao_nbs} | "
            f"cClassTrib: {cclass_trib} | "
            f"nome cClassTrib: {nome_cclass_trib} | "
            f"ONDA DE CADASTRO: {onda_cadastro}"
        )

        metadata = {
            "linha_csv": idx,
            "item_lc116": item_lc116,
            "descricao_item": descricao_item,
            "codigo_nacional": codigo_nacional,
            "cnae": cnae,
            "nbs": nbs,
            "descricao_nbs": descricao_nbs,
            "cclass_trib": cclass_trib,
            "nome_cclass_trib": nome_cclass_trib,
            "onda_cadastro": onda_cadastro,
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

    vectorstore = FAISS.from_documents(
        documents=_chunks,
        embedding=embeddings,
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
        "Item LC 116": doc.metadata.get("item_lc116", ""),
        "Descrição Item": doc.metadata.get("descricao_item", ""),
        "CÓDIGO NACIONAL": doc.metadata.get("codigo_nacional", ""),
        "CNAE": doc.metadata.get("cnae", ""),
        "NBS": doc.metadata.get("nbs", ""),
        "DESCRIÇÃO NBS": doc.metadata.get("descricao_nbs", ""),
        "cClassTrib": doc.metadata.get("cclass_trib", ""),
        "nome cClassTrib": doc.metadata.get("nome_cclass_trib", ""),
        "ONDA DE CADASTRO": doc.metadata.get("onda_cadastro", ""),
    }


def responder_pergunta(pergunta, vectorstore):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    documentos_recuperados = vectorstore.similarity_search(pergunta, k=10)

    documentos_rerankeados = rerank_documentos(pergunta, documentos_recuperados, llm)
    contexto_final = documentos_rerankeados[:10]

    linhas_relevantes = [documento_para_json(doc) for doc in contexto_final]
    contexto_json = json.dumps(linhas_relevantes, ensure_ascii=False, indent=2)

    prompt_final = f"""Voce e um assistente especializado em consulta de CNAE e tributacao (IBS/CBS).
Responda com base EXCLUSIVAMENTE nos itens JSON fornecidos abaixo.
Se a informacao nao estiver nos itens, retorne um array vazio: []

Retorne SOMENTE JSON valido, sem markdown e sem texto adicional.

Regras obrigatorias:
- Se houver exatamente 1 item correspondente, retorne um unico objeto JSON.
- Se houver 2 ou mais itens correspondentes, retorne um array de objetos JSON.
- Cada objeto deve conter EXATAMENTE estas chaves:
  "Item LC 116", "Descrição Item", "CÓDIGO NACIONAL", "CNAE", "NBS", "DESCRIÇÃO NBS", "cClassTrib", "nome cClassTrib", "ONDA DE CADASTRO"
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


st.set_page_config(page_title="CHAT CNAE com RAG (CSV v2)", layout="wide")
st.title("CHAT CNAE - RAG com CSV v2 (LC 116 / NBS / IBS / CBS)")

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
        st.write(f"Item LC 116: {doc.metadata.get('item_lc116')}")
        st.write(f"Descrição Item: {doc.metadata.get('descricao_item')}")
        st.write(f"CÓDIGO NACIONAL: {doc.metadata.get('codigo_nacional')}")
        st.write(f"CNAE: {doc.metadata.get('cnae')}")
        st.write(f"NBS: {doc.metadata.get('nbs')}")
        st.write(f"DESCRIÇÃO NBS: {doc.metadata.get('descricao_nbs')}")
        st.write(f"cClassTrib: {doc.metadata.get('cclass_trib')}")
        st.write(f"nome cClassTrib: {doc.metadata.get('nome_cclass_trib')}")
        st.write(f"ONDA DE CADASTRO: {doc.metadata.get('onda_cadastro')}")
        st.write(doc.page_content)
        st.divider()

elif pergunta and not csv_disponivel:
    st.info("Adicione o arquivo cnae-exemplo-2.csv no diretorio do projeto para habilitar a consulta.")
