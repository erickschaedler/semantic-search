"""
Chat com Busca Semântica em Documentos
"""

import streamlit as st
import traceback
import json
from datetime import datetime
from rag import (
    get_openai_client,
    extract_text_by_page,
    split_pages_into_chunks,
    generate_embeddings,
    ask_question_pipeline,
    SimpleVectorStore
)

# ============== CONFIGURAÇÃO DA PÁGINA ==============

st.set_page_config(
    page_title="Chat com Documentos",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d5a87;
        margin: 0.5rem 0;
    }
    .page-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 Chat com Documentos</h1>
    <p>Faça perguntas sobre seus manuais e documentos técnicos</p>
</div>
""", unsafe_allow_html=True)


# ============== INICIALIZAÇÃO ==============

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()

init_session_state()


# ============== SIDEBAR ==============

with st.sidebar:
    st.header("⚙️ Configuração")

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

    # Upload
    st.header("📄 Upload de Documentos")

    use_ocr = st.checkbox(
        "Usar OCR (PDFs escaneados)",
        value=False,
        help="Ative para documentos que são imagens"
    )

    uploaded_files = st.file_uploader(
        "Selecione PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and api_key:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                status = st.empty()
                progress_bar = st.progress(0)

                try:
                    # Etapa 1: Extrair texto por página
                    status.info("📄 Extraindo texto..." + (" (OCR)" if use_ocr else ""))
                    progress_bar.progress(10)

                    pages_data = extract_text_by_page(uploaded_file, use_ocr=use_ocr)
                    total_text = sum(len(p["text"]) for p in pages_data)

                    if total_text < 50:
                        status.error("PDF sem texto extraível")
                        progress_bar.empty()
                        continue

                    progress_bar.progress(30)

                    # Etapa 2: Dividir em chunks
                    status.info("✂️ Processando...")
                    try:
                        chunks = split_pages_into_chunks(pages_data)
                    except Exception as e:
                        status.error(f"Erro no processamento: {e}")
                        progress_bar.empty()
                        continue

                    if not chunks:
                        status.error("Falha ao processar")
                        progress_bar.empty()
                        continue

                    progress_bar.progress(50)

                    # Etapa 3: Embeddings
                    status.info("🧠 Gerando embeddings...")
                    client = get_openai_client(api_key)

                    all_embeddings = []
                    batch_size = 20
                    chunk_texts = [c["text"] for c in chunks]

                    for i in range(0, len(chunk_texts), batch_size):
                        batch = chunk_texts[i:i+batch_size]
                        batch_emb = generate_embeddings(batch, client)
                        all_embeddings.extend(batch_emb)
                        pct = 50 + int((i / len(chunk_texts)) * 40)
                        progress_bar.progress(min(pct, 90))

                    # Etapa 4: Salvar
                    status.info("💾 Salvando...")
                    st.session_state.vector_store.add(chunks, all_embeddings, uploaded_file.name)

                    progress_bar.progress(100)
                    st.session_state.processed_files.append(uploaded_file.name)

                    status.empty()
                    progress_bar.empty()

                    num_pages = pages_data[0]["total_pages"] if pages_data else 0
                    st.success(f"✓ {uploaded_file.name}")
                    st.caption(f"   {num_pages} páginas, {len(chunks)} chunks")

                except Exception as e:
                    status.empty()
                    progress_bar.empty()
                    st.error(f"Erro: {str(e)}")
                    with st.expander("Detalhes"):
                        st.code(traceback.format_exc())

    # Estatísticas
    if st.session_state.processed_files:
        st.divider()
        stats = st.session_state.vector_store.get_stats()

        st.markdown(f"""
        <div class="stat-box">
            <strong>📊 Documentos carregados</strong><br>
            📁 {len(st.session_state.processed_files)} arquivo(s)<br>
            📄 {stats['total_pages']} página(s)<br>
            🧩 {stats['total_chunks']} chunks
        </div>
        """, unsafe_allow_html=True)

        for file in st.session_state.processed_files:
            st.text(f"• {file}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Limpar", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.processed_files = []
                st.session_state.vector_store = SimpleVectorStore()
                st.rerun()

        with col2:
            if st.session_state.messages:
                # Exportar conversa
                export_data = {
                    "exported_at": datetime.now().isoformat(),
                    "documents": st.session_state.processed_files,
                    "conversation": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                }
                st.download_button(
                    "📥 Exportar",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )


# ============== CHAT ==============

# Histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📖 Fontes utilizadas"):
                for i, (source, meta) in enumerate(zip(message["sources"], message.get("metadatas", [])), 1):
                    page_num = meta.get("page_num", "?") if meta else "?"
                    st.markdown(f'**Trecho {i}** <span class="page-badge">Página {page_num}</span>', unsafe_allow_html=True)
                    st.text(source[:400] + "..." if len(source) > 400 else source)
                    if i < len(message["sources"]):
                        st.divider()

# Input
if prompt := st.chat_input("Faça uma pergunta sobre o documento..."):
    if not api_key:
        st.error("⚠️ Configure sua API Key na barra lateral")
        st.stop()

    if not st.session_state.processed_files:
        st.error("⚠️ Faça upload de pelo menos um documento PDF")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando informações..."):
            try:
                client = get_openai_client(api_key)

                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                    if m["role"] in ["user", "assistant"]
                ]

                answer, sources, metadatas = ask_question_pipeline(
                    prompt,
                    client,
                    st.session_state.vector_store,
                    n_results=5,
                    chat_history=chat_history[:-1]
                )

                st.markdown(answer)

                if sources:
                    with st.expander("📖 Fontes utilizadas"):
                        for i, (source, meta) in enumerate(zip(sources, metadatas), 1):
                            page_num = meta.get("page_num", "?")
                            st.markdown(f'**Trecho {i}** <span class="page-badge">Página {page_num}</span>', unsafe_allow_html=True)
                            st.text(source[:400] + "..." if len(source) > 400 else source)
                            if i < len(sources):
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metadatas": metadatas
                })

            except Exception as e:
                st.error(f"Erro: {str(e)}")

# Footer
st.divider()
st.caption("💡 Dica: Perguntas específicas geram respostas melhores!")
