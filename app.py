"""
Interface Streamlit para o Chat com Busca Semântica em Manuais
"""

import streamlit as st
import traceback
from rag import (
    get_openai_client,
    extract_text_from_pdf,
    split_text_into_chunks,
    generate_embeddings,
    ask_question_pipeline,
    SimpleVectorStore
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

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()


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

    use_ocr = st.checkbox(
        "Usar OCR (para PDFs escaneados)",
        value=False,
        help="Ative se o PDF for uma imagem escaneada"
    )

    uploaded_files = st.file_uploader(
        "Selecione PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Faça upload de 1 ou mais manuais em PDF"
    )

    if uploaded_files and api_key:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                status = st.empty()
                progress_bar = st.progress(0)

                try:
                    # Etapa 1: Extrair texto
                    if use_ocr:
                        status.info("📄 Extraindo texto com OCR (pode demorar)...")
                    else:
                        status.info("📄 Extraindo texto do PDF...")
                    progress_bar.progress(10)

                    text = extract_text_from_pdf(uploaded_file, use_ocr=use_ocr)

                    if not text or len(text.strip()) < 50:
                        status.error("PDF sem texto extraível")
                        progress_bar.empty()
                        continue

                    status.info(f"📄 Texto extraído: {len(text)} caracteres")
                    progress_bar.progress(30)

                    # Etapa 2: Dividir em chunks
                    status.info("✂️ Dividindo em chunks...")
                    try:
                        chunks = split_text_into_chunks(text)
                    except Exception as chunk_err:
                        status.error(f"Erro no chunking: {chunk_err}")
                        progress_bar.empty()
                        continue

                    if not chunks:
                        status.error("Nenhum chunk gerado")
                        progress_bar.empty()
                        continue

                    status.info(f"✂️ {len(chunks)} chunks criados")
                    progress_bar.progress(50)

                    # Etapa 3: Gerar embeddings
                    status.info("🧠 Gerando embeddings...")
                    client = get_openai_client(api_key)

                    # Processa em batches de 20
                    all_embeddings = []
                    batch_size = 20
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        batch_embeddings = generate_embeddings(batch, client)
                        all_embeddings.extend(batch_embeddings)

                        # Atualiza progresso
                        pct = 50 + int((i / len(chunks)) * 40)
                        progress_bar.progress(min(pct, 90))
                        status.info(f"🧠 Embeddings: {len(all_embeddings)}/{len(chunks)}")

                    # Etapa 4: Salvar no vector store
                    status.info("💾 Salvando...")
                    st.session_state.vector_store.add(chunks, all_embeddings, uploaded_file.name)

                    progress_bar.progress(100)
                    st.session_state.processed_files.append(uploaded_file.name)

                    status.empty()
                    progress_bar.empty()
                    st.success(f"✓ {uploaded_file.name} ({len(chunks)} chunks)")

                except Exception as e:
                    status.empty()
                    progress_bar.empty()
                    st.error(f"Erro: {str(e)}")
                    with st.expander("Detalhes"):
                        st.code(traceback.format_exc())

    # Lista de arquivos processados
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📁 Manuais carregados")
        for file in st.session_state.processed_files:
            st.text(f"• {file}")

        if st.button("🗑️ Limpar tudo", type="secondary"):
            st.session_state.messages = []
            st.session_state.processed_files = []
            st.session_state.vector_store = SimpleVectorStore()
            st.rerun()


# ============== ÁREA PRINCIPAL - CHAT ==============

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Trechos relevantes"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Trecho {i}:**")
                    st.text(source[:500] + "..." if len(source) > 500 else source)
                    st.divider()

# Input do usuário
if prompt := st.chat_input("Faça uma pergunta sobre o manual..."):
    if not api_key:
        st.error("⚠️ Configure sua API Key na barra lateral")
        st.stop()

    if not st.session_state.processed_files:
        st.error("⚠️ Faça upload de pelo menos um manual PDF")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando no manual..."):
            try:
                client = get_openai_client(api_key)

                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                    if m["role"] in ["user", "assistant"]
                ]

                answer, sources = ask_question_pipeline(
                    prompt,
                    client,
                    st.session_state.vector_store,
                    n_results=5,
                    chat_history=chat_history[:-1]
                )

                st.markdown(answer)

                if sources:
                    with st.expander("📖 Trechos relevantes"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Trecho {i}:**")
                            st.text(source[:500] + "..." if len(source) > 500 else source)
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Erro: {str(e)}")

st.divider()
st.caption("💡 Dica: Quanto mais específica a pergunta, melhor a resposta!")
