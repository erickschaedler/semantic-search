"""
Módulo RAG (Retrieval-Augmented Generation) - Versão Simplificada
Usa apenas numpy para busca vetorial em memória.
"""

import os
from typing import List, Tuple
import pdfplumber
from openai import OpenAI
import numpy as np


def get_openai_client(api_key: str = None) -> OpenAI:
    """Retorna cliente OpenAI configurado."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY não configurada")
    return OpenAI(api_key=key)


# ============== PROCESSAMENTO DE PDF ==============

def extract_text_from_pdf(pdf_file, use_ocr: bool = False) -> str:
    """Extrai texto de um arquivo PDF."""
    pdf_file.seek(0)

    # Primeiro tenta extração normal
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Se use_ocr ou pouco texto extraído, tenta OCR
    if use_ocr or len(text.strip()) < 100:
        pdf_file.seek(0)
        ocr_text = extract_text_with_ocr(pdf_file)
        if ocr_text and len(ocr_text) > len(text):
            text = ocr_text

    return text


def extract_text_with_ocr(pdf_file) -> str:
    """Extrai texto usando OCR (para PDFs escaneados)."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        return ""

    try:
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()

        # Converte PDF para imagens
        images = convert_from_bytes(pdf_bytes, dpi=150)

        text = ""
        for i, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(image, lang='por')
            except:
                page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"

        return text
    except Exception:
        return ""


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Divide o texto em chunks menores - versão simplificada."""
    if not text or len(text) < 10:
        return []

    # Limpa o texto
    text = text.strip()

    # Divide de forma simples
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 20:
            chunks.append(chunk)

    return chunks


# ============== EMBEDDINGS ==============

def generate_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Gera embeddings para uma lista de textos usando OpenAI."""
    if not texts:
        return []

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


# ============== VECTOR STORE (Em Memória com Numpy) ==============

class SimpleVectorStore:
    """Armazenamento vetorial simples em memória."""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []

    def add(self, documents: List[str], embeddings: List[List[float]], source: str = "manual"):
        """Adiciona documentos ao store."""
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            self.documents.append(doc)
            self.embeddings.append(emb)
            self.metadatas.append({"source": source, "chunk_index": i})

    def search(self, query_embedding: List[float], n_results: int = 3) -> Tuple[List[str], List[dict]]:
        """Busca documentos similares usando similaridade de cosseno."""
        if not self.embeddings:
            return [], []

        # Converte para numpy
        query = np.array(query_embedding)
        embeddings = np.array(self.embeddings)

        # Calcula similaridade de cosseno
        query_norm = query / np.linalg.norm(query)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm)

        # Pega os top N
        top_indices = np.argsort(similarities)[::-1][:n_results]

        docs = [self.documents[i] for i in top_indices]
        metas = [self.metadatas[i] for i in top_indices]

        return docs, metas

    def clear(self):
        """Limpa o store."""
        self.documents = []
        self.embeddings = []
        self.metadatas = []


# ============== CHAT COM CONTEXTO ==============

def build_context_prompt(question: str, relevant_chunks: List[str]) -> str:
    """Constrói o prompt com contexto para a LLM."""
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
    """Gera resposta usando GPT com contexto dos chunks relevantes."""
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
    vector_store: SimpleVectorStore,
    source_name: str = "manual"
) -> int:
    """Pipeline completo: PDF -> Chunks -> Embeddings -> Store"""
    # Extrai texto
    text = extract_text_from_pdf(pdf_file)

    # Divide em chunks
    chunks = split_text_into_chunks(text)

    if not chunks:
        return 0

    # Gera embeddings
    embeddings = generate_embeddings(chunks, openai_client)

    # Salva no store
    vector_store.add(chunks, embeddings, source_name)

    return len(chunks)


def ask_question_pipeline(
    question: str,
    openai_client: OpenAI,
    vector_store: SimpleVectorStore,
    n_results: int = 3,
    chat_history: List[dict] = None
) -> Tuple[str, List[str]]:
    """Pipeline completo de pergunta: Query -> Busca -> Resposta"""
    # Gera embedding da pergunta
    query_embedding = generate_single_embedding(question, openai_client)

    # Busca chunks similares
    relevant_chunks, _ = vector_store.search(query_embedding, n_results)

    if not relevant_chunks:
        return "Não encontrei informações relevantes no manual para responder essa pergunta.", []

    # Gera resposta
    answer = chat_with_context(question, relevant_chunks, openai_client, chat_history)

    return answer, relevant_chunks
