# =================================
# 1. IMPORTS
# =================================
import json
import os

import streamlit as st
from langchain_openai import ChatOpenAI

# Injeta a chave como variável de ambiente
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = "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"

# =================================
# 2. GENERAL SETTINGS
# =================================
LLM_MODEL = "gpt-4.1-mini"

# =================================
# 3. DADOS EXTERNOS — JSON
# =================================
JSON_FILE_PATH = "./cnae-dados.json"


@st.cache_data
def carregar_dados() -> list:
    with open(JSON_FILE_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def listar_cnaes(dados: list) -> list:
    seen = set()
    result = []
    for r in dados:
        cnae = r.get("CNAE", "").strip()
        item = r.get("Item LC 116", "").strip()
        descricao = r.get("Descricao Item", "").strip()
        key = (cnae, item)
        if cnae and key not in seen:
            seen.add(key)
            result.append({"cnae": cnae, "item": item, "descricao": descricao})
    return sorted(result, key=lambda x: (x["cnae"], x["item"]))


# =================================
# 4. LLM FUNCTIONS
# =================================

def avaliar_e_responder(pergunta: str, dados_json: list, llm) -> dict:
    """
    LLM Call 1 — avalia se a pergunta é suficientemente específica.

    Retorna um dict com uma de duas estruturas:
      {"resolved": True,  "data": [...]}           — resultado direto
      {"resolved": False, "topics": ["...", ...]}  — 5 tópicos de clarificação
    """
    dados_str = json.dumps(dados_json, ensure_ascii=False, indent=2)

    prompt = f"""Voce e um assistente especializado em classificacao tributaria de servicos baseado no CNAE do cliente.

Abaixo estao TODOS os registros disponiveis para o CNAE informado:

{dados_str}

Pergunta do usuario (contem o CNAE e a descricao da atividade):
{pergunta}

Sua tarefa:
1. Analise se a descricao fornecida pelo usuario é especifica o suficiente para identificar UMA ou POUCAS linhas claramente correspondentes.
2. Se conseguir determinar com clareza quais registros correspondem, retorne:
   {{"resolved": true, "data": [<objeto(s) JSON dos registros correspondentes>]}}
3. Se a descricao for ambigua ou generica demais para decidir entre multiplas opcoes, retorne EXATAMENTE 5 topicos curtos e objetivos que o usuario possa selecionar para refinar a busca:
   {{"resolved": false, "topics": ["Topico 1", "Topico 2", "Topico 3", "Topico 4", "Topico 5"]}}

Regras obrigatorias:
- Retorne SOMENTE JSON valido, sem markdown, sem texto adicional.
- Em "data", inclua apenas registros realmente correspondentes. Nao invente valores.
- Em "topics", os topicos devem ser frases curtas (ate 8 palavras) que descrevam subtipos distintos de servico dentro do CNAE.
- Cada objeto em "data" deve conter EXATAMENTE estas chaves:
  "Item LC 116", "Descricao Item", "CODIGO NACIONAL", "CNAE", "NBS", "DESCRICAO NBS", "cClassTrib", "nome cClassTrib", "ONDA DE CADASTRO"
"""

    resposta_bruta = llm.invoke(prompt).content.strip()

    try:
        return json.loads(resposta_bruta)
    except json.JSONDecodeError:
        # Fallback seguro: trata como ambíguo sem tópicos
        return {"resolved": False, "topics": []}


def responder_com_contexto_adicional(pergunta: str, topico: str, dados_json: list, llm) -> object:
    """
    LLM Call 2 — usa a pergunta original + tópico selecionado para determinar o(s) registro(s) final(is).

    Retorna um dict ou lista de dicts correspondendo aos registros encontrados.
    """
    dados_str = json.dumps(dados_json, ensure_ascii=False, indent=2)

    prompt = f"""Voce e um assistente especializado em classificacao tributaria de servicos baseado no CNAE do cliente.

Abaixo estao TODOS os registros disponiveis para o CNAE informado:

{dados_str}

Pergunta original do usuario:
{pergunta}

O usuario refinou a busca selecionando o seguinte topico:
{topico}

Com base na pergunta original E no topico selecionado, identifique e retorne SOMENTE os registros que correspondam a essa combinacao.

Se nenhum item for compativel, retorne um array vazio: []

Retorne SOMENTE JSON valido, sem markdown e sem texto adicional.

Regras obrigatorias:
- Se houver exatamente 1 item correspondente, retorne um unico objeto JSON.
- Se houver 2 ou mais itens correspondentes, retorne um array de objetos JSON.
- Cada objeto deve conter EXATAMENTE estas chaves:
  "Item LC 116", "Descrição Item", "CÓDIGO NACIONAL", "CNAE", "NBS", "DESCRIÇÃO NBS", "cClassTrib", "nome cClassTrib", "ONDA DE CADASTRO"
- Nao invente valores e nao altere os valores dos itens fornecidos.
"""

    resposta_bruta = llm.invoke(prompt).content.strip()

    try:
        return json.loads(resposta_bruta)
    except json.JSONDecodeError:
        return []


# =================================
# 5. STREAMLIT UI
# =================================

st.set_page_config(page_title="CHAT CNAE — Fluxo Interativo", layout="wide")
st.title("CHAT CNAE — Classificação Tributária com Refinamento Interativo")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Defina a variável de ambiente OPENAI_API_KEY antes de usar o app.")
    st.stop()

# --- Inicialização do session_state ---
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""
if "topicos" not in st.session_state:
    st.session_state.topicos = []
if "resposta_final" not in st.session_state:
    st.session_state.resposta_final = None
if "dados_filtrado" not in st.session_state:
    st.session_state.dados_filtrado = []

# =================================
# STAGE: INPUT
# =================================
if st.session_state.stage == "input":
    todos_dados = carregar_dados()
    cnaes_unicos = listar_cnaes(todos_dados)

    st.subheader("Consulta")

    cnae_selecionado = st.selectbox(
        "Selecione o CNAE do cliente:",
        options=[None] + cnaes_unicos,
        format_func=lambda x: "— selecione um CNAE —" if x is None else f"{x['cnae']} - {x['item']} - {x['descricao']}",
        key="select_cnae",
    )

    descricao = st.text_area(
        "Descreva a atividade do cliente:",
        placeholder="Ex: desenvolvo aplicativos mobile personalizados para empresas",
        disabled=(cnae_selecionado is None),
        key="input_descricao",
    )

    if st.button(
        "Consultar",
        type="primary",
        disabled=(cnae_selecionado is None or not descricao.strip()),
    ):
        dados_filtrado = [
            r for r in todos_dados
            if r.get("CNAE", "").strip() == cnae_selecionado["cnae"]
            and r.get("Item LC 116", "").strip() == cnae_selecionado["item"]
        ]
        st.session_state.dados_filtrado = dados_filtrado

        with st.spinner(f"Analisando {len(dados_filtrado)} registro(s) para CNAE {cnae_selecionado['cnae']}..."):
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
            resultado = avaliar_e_responder(descricao.strip(), dados_filtrado, llm)

        st.session_state.pergunta = f"CNAE {cnae_selecionado['cnae']} ({cnae_selecionado['item']} - {cnae_selecionado['descricao']}) — {descricao.strip()}"

        if resultado.get("resolved"):
            st.session_state.resposta_final = resultado.get("data")
            st.session_state.stage = "result"
            st.rerun()
        else:
            topicos = resultado.get("topics", [])
            if topicos:
                st.session_state.topicos = topicos
                st.session_state.stage = "clarification"
                st.rerun()
            else:
                st.error(
                    "Não foi possível determinar a classificação e nenhum tópico de refinamento foi gerado. "
                    "Tente descrever a atividade com mais detalhes."
                )

# =================================
# STAGE: CLARIFICATION
# =================================
elif st.session_state.stage == "clarification":
    st.subheader("Refinamento necessário")
    st.info(f"**Pergunta original:** {st.session_state.pergunta}")
    st.markdown(
        "A descrição informada é abrangente demais para determinar uma classificação única. "
        "Selecione abaixo o tópico que melhor descreve a atividade do cliente:"
    )

    topico_selecionado = st.radio(
        "Tópicos disponíveis:",
        options=st.session_state.topicos,
        key="radio_topico",
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("Confirmar seleção", type="primary"):
            with st.spinner("Buscando classificação..."):
                llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
                resposta = responder_com_contexto_adicional(
                    st.session_state.pergunta,
                    topico_selecionado,
                    st.session_state.dados_filtrado,
                    llm,
                )
            st.session_state.resposta_final = resposta
            st.session_state.stage = "result"
            st.rerun()
    with col2:
        if st.button("Voltar"):
            st.session_state.stage = "input"
            st.session_state.topicos = []
            st.rerun()

# =================================
# STAGE: RESULT
# =================================
elif st.session_state.stage == "result":
    st.subheader("Resultado da Classificação")
    st.info(f"**Pergunta:** {st.session_state.pergunta}")

    if st.session_state.resposta_final == [] or st.session_state.resposta_final is None:
        st.warning("Nenhum registro correspondente foi encontrado para a atividade informada.")
    else:
        st.json(st.session_state.resposta_final)

    if st.button("Nova consulta"):
        st.session_state.stage = "input"
        st.session_state.pergunta = ""
        st.session_state.topicos = []
        st.session_state.resposta_final = None
        st.session_state.dados_filtrado = []
        st.rerun()
