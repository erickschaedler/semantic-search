"""
Módulo RAG (Retrieval-Augmented Generation)
Contém toda a lógica de processamento de PDF, embeddings e busca semântica.
"""

import os
import tempfile
from typing import List, Tuple
from pypdf import PdfReader
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# OCR desabilitado temporariamente
OCR_AVAILABLE = False


def check_ocr_available():
    """OCR desabilitado."""
    return False


# Inicializa cliente OpenAI
def get_openai_client(api_key: str = None) -> OpenAI:
    """Retorna cliente OpenAI configurado."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY não configurada")
    return OpenAI(api_key=key)


# ============== PROCESSAMENTO DE PDF ==============

def extract_text_from_pdf(pdf_file, use_ocr: bool = False) -> str:
    """
    Extrai texto de um arquivo PDF.

    Se use_ocr=True ou se o texto extraído for muito curto, usa OCR.
    """
    # Primeiro tenta extração normal
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Se extraiu pouco texto e OCR está disponível, tenta OCR
    if (use_ocr or len(text.strip()) < 100) and check_ocr_available():
        text = extract_text_with_ocr(pdf_file)

    return text


def extract_text_with_ocr(pdf_file) -> str:
    """Extrai texto de PDF usando OCR (para PDFs escaneados)."""
    if not check_ocr_available():
        raise RuntimeError("OCR não disponível. Instale pdf2image e pytesseract.")

    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()

    # Converte PDF para imagens (DPI menor = menos memória)
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
    except Exception as e:
        raise RuntimeError(f"Erro ao converter PDF para imagens: {str(e)}")

    # Extrai texto de cada página com OCR
    text = ""
    for i, image in enumerate(images):
        try:
            # Tenta português, se falhar usa inglês
            try:
                page_text = pytesseract.image_to_string(image, lang='por')
            except:
                page_text = pytesseract.image_to_string(image)
            text += f"\n--- Página {i+1} ---\n{page_text}"
        except Exception as e:
            text += f"\n--- Página {i+1} ---\n[Erro no OCR: {str(e)}]"
        finally:
            # Libera memória da imagem
            del image

    return text


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Divide o texto em chunks menores com overlap.

    Args:
        text: Texto completo
        chunk_size: Tamanho máximo de cada chunk em caracteres
        overlap: Quantidade de caracteres de sobreposição entre chunks

    Returns:
        Lista de chunks de texto
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Se não é o último chunk, tenta quebrar em um espaço
        if end < text_length:
            # Procura o último espaço dentro do chunk
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_length else text_length

    return chunks


# ============== EMBEDDINGS ==============

def generate_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos usando OpenAI.

    Args:
        texts: Lista de textos
        client: Cliente OpenAI

    Returns:
        Lista de embeddings (vetores)
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def generate_single_embedding(text: str, client: OpenAI) -> List[float]:
    """Gera embedding para um único texto."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


# ============== VECTOR STORE (ChromaDB) ==============

def get_chroma_client(persist_directory: str = "./chroma_db") -> chromadb.Client:
    """Retorna cliente ChromaDB com persistência."""
    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))


def get_or_create_collection(client: chromadb.Client, name: str = "manuals"):
    """Obtém ou cria uma coleção no ChromaDB."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def add_documents_to_collection(
    collection,
    chunks: List[str],
    embeddings: List[List[float]],
    source: str = "manual"
):
    """
    Adiciona documentos à coleção do ChromaDB.

    Args:
        collection: Coleção do ChromaDB
        chunks: Lista de chunks de texto
        embeddings: Lista de embeddings correspondentes
        source: Nome da fonte (ex: nome do PDF)
    """
    ids = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )


def search_similar(
    collection,
    query_embedding: List[float],
    n_results: int = 3
) -> Tuple[List[str], List[dict]]:
    """
    Busca documentos similares no ChromaDB.

    Args:
        collection: Coleção do ChromaDB
        query_embedding: Embedding da query
        n_results: Número de resultados

    Returns:
        Tupla com (documentos, metadados)
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    return documents, metadatas


# ============== CHAT COM CONTEXTO ==============

def build_context_prompt(question: str, relevant_chunks: List[str]) -> str:
    """
    Constrói o prompt com contexto para a LLM.

    Args:
        question: Pergunta do usuário
        relevant_chunks: Chunks relevantes encontrados

    Returns:
        Prompt formatado
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    return f"""Você é um assistente especializado em responder perguntas sobre manuais técnicos.
Use APENAS as informações do contexto abaixo para responder. Se a informação não estiver no contexto, diga que não encontrou essa informação no manual.

CONTEXTO:
{context}

---

PERGUNTA: {question}

RESPOSTA:"""


def chat_with_context(
    question: str,
    relevant_chunks: List[str],
    client: OpenAI,
    chat_history: List[dict] = None
) -> str:
    """
    Gera resposta usando GPT com contexto dos chunks relevantes.

    Args:
        question: Pergunta do usuário
        relevant_chunks: Chunks relevantes do manual
        client: Cliente OpenAI
        chat_history: Histórico de mensagens (opcional)

    Returns:
        Resposta gerada
    """
    system_prompt = build_context_prompt(question, relevant_chunks)

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ============== PIPELINE COMPLETO ==============

def process_pdf_pipeline(
    pdf_file,
    openai_client: OpenAI,
    collection,
    source_name: str = "manual"
) -> int:
    """
    Pipeline completo: PDF -> Chunks -> Embeddings -> ChromaDB

    Args:
        pdf_file: Arquivo PDF
        openai_client: Cliente OpenAI
        collection: Coleção ChromaDB
        source_name: Nome identificador do documento

    Returns:
        Número de chunks processados
    """
    # Extrai texto
    text = extract_text_from_pdf(pdf_file)

    # Divide em chunks
    chunks = split_text_into_chunks(text)

    if not chunks:
        return 0

    # Gera embeddings
    embeddings = generate_embeddings(chunks, openai_client)

    # Salva no ChromaDB
    add_documents_to_collection(collection, chunks, embeddings, source_name)

    return len(chunks)


def ask_question_pipeline(
    question: str,
    openai_client: OpenAI,
    collection,
    n_results: int = 3,
    chat_history: List[dict] = None
) -> Tuple[str, List[str]]:
    """
    Pipeline completo de pergunta: Query -> Busca -> Resposta

    Args:
        question: Pergunta do usuário
        openai_client: Cliente OpenAI
        collection: Coleção ChromaDB
        n_results: Número de chunks a recuperar
        chat_history: Histórico do chat

    Returns:
        Tupla com (resposta, chunks_usados)
    """
    # Gera embedding da pergunta
    query_embedding = generate_single_embedding(question, openai_client)

    # Busca chunks similares
    relevant_chunks, _ = search_similar(collection, query_embedding, n_results)

    if not relevant_chunks:
        return "Não encontrei informações relevantes no manual para responder essa pergunta.", []

    # Gera resposta
    answer = chat_with_context(question, relevant_chunks, openai_client, chat_history)

    return answer, relevant_chunks
