"""
Interface Streamlit para o Chat com Busca Semântica em Manuais
"""

import streamlit as st
import chromadb
from chromadb.config import Settings
from rag import (
    get_openai_client,
    process_pdf_pipeline,
    ask_question_pipeline,
    get_or_create_collection
)

# ============== CONFIGURAÇÃO DA PÁGINA ==============

st.set_page_config(
    page_title="Chat com Manuais",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Chat com Manuais")
st.caption("Faça perguntas sobre seus manuais técnicos")


# ============== INICIALIZAÇÃO DO ESTADO ==============

def init_session_state():
    """Inicializa variáveis de sessão."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False
        ))

    if "collection" not in st.session_state:
        st.session_state.collection = get_or_create_collection(
            st.session_state.chroma_client,
            "manuals"
        )


init_session_state()


# ============== SIDEBAR - CONFIGURAÇÃO ==============

with st.sidebar:
    st.header("⚙️ Configuração")

    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Sua chave da API OpenAI",
        value=st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    )

    if api_key:
        st.success("✓ API Key configurada")
    else:
        st.warning("⚠️ Insira sua API Key")

    st.divider()

    # Upload de PDF
    st.header("📄 Upload de Manuais")

    uploaded_files = st.file_uploader(
        "Selecione PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Faça upload de 1 ou mais manuais em PDF"
    )

    if uploaded_files and api_key:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Processando {uploaded_file.name}..."):
                    try:
                        client = get_openai_client(api_key)
                        num_chunks = process_pdf_pipeline(
                            uploaded_file,
                            client,
                            st.session_state.collection,
                            uploaded_file.name
                        )
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.success(f"✓ {uploaded_file.name} ({num_chunks} chunks)")
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")

    # Lista de arquivos processados
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📁 Manuais carregados")
        for file in st.session_state.processed_files:
            st.text(f"• {file}")

        if st.button("🗑️ Limpar tudo", type="secondary"):
            st.session_state.messages = []
            st.session_state.processed_files = []
            st.session_state.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False
            ))
            st.session_state.collection = get_or_create_collection(
                st.session_state.chroma_client,
                "manuals"
            )
            st.rerun()


# ============== ÁREA PRINCIPAL - CHAT ==============

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Mostra fontes se houver
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Trechos relevantes"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Trecho {i}:**")
                    st.text(source[:500] + "..." if len(source) > 500 else source)
                    st.divider()

# Input do usuário
if prompt := st.chat_input("Faça uma pergunta sobre o manual..."):
    # Validações
    if not api_key:
        st.error("⚠️ Configure sua API Key na barra lateral")
        st.stop()

    if not st.session_state.processed_files:
        st.error("⚠️ Faça upload de pelo menos um manual PDF")
        st.stop()

    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera resposta
    with st.chat_message("assistant"):
        with st.spinner("Buscando no manual..."):
            try:
                client = get_openai_client(api_key)

                # Prepara histórico (últimas 5 mensagens)
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                    if m["role"] in ["user", "assistant"]
                ]

                # Faz a pergunta
                answer, sources = ask_question_pipeline(
                    prompt,
                    client,
                    st.session_state.collection,
                    n_results=3,
                    chat_history=chat_history[:-1]  # Exclui a pergunta atual
                )

                st.markdown(answer)

                # Mostra fontes
                if sources:
                    with st.expander("📖 Trechos relevantes"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Trecho {i}:**")
                            st.text(source[:500] + "..." if len(source) > 500 else source)
                            st.divider()

                # Salva no histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Erro ao processar: {str(e)}")


# ============== FOOTER ==============

st.divider()
st.caption("💡 Dica: Quanto mais específica a pergunta, melhor a resposta!")
