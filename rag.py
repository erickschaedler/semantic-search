"""
Módulo RAG (Retrieval-Augmented Generation) - Versão Melhorada
"""

import os
from typing import List, Tuple, Dict
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

def extract_text_by_page(pdf_file, use_ocr: bool = False, max_pages: int = 100) -> List[Dict]:
    """Extrai texto de cada página do PDF com metadados."""
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    pages_data = []

    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = min(len(pdf.pages), max_pages)
            for i in range(total_pages):
                try:
                    page = pdf.pages[i]
                    page_text = page.extract_text() or ""
                    pages_data.append({
                        "page_num": i + 1,
                        "total_pages": total_pages,
                        "text": page_text
                    })
                except Exception:
                    pages_data.append({
                        "page_num": i + 1,
                        "total_pages": total_pages,
                        "text": ""
                    })
    except Exception as e:
        raise RuntimeError(f"Erro ao abrir PDF: {str(e)}")

    # Se use_ocr ou pouco texto extraído, tenta OCR
    total_text = sum(len(p["text"]) for p in pages_data)
    if use_ocr or total_text < 100:
        try:
            ocr_pages = extract_pages_with_ocr(io.BytesIO(pdf_bytes))
            if ocr_pages:
                ocr_total = sum(len(p["text"]) for p in ocr_pages)
                if ocr_total > total_text:
                    pages_data = ocr_pages
        except Exception:
            pass

    return pages_data


def extract_pages_with_ocr(pdf_file) -> List[Dict]:
    """Extrai texto usando OCR por página."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        return []

    try:
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        images = convert_from_bytes(pdf_bytes, dpi=150)

        pages_data = []
        for i, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(image, lang='por')
            except:
                page_text = pytesseract.image_to_string(image)
            pages_data.append({
                "page_num": i + 1,
                "total_pages": len(images),
                "text": page_text
            })

        return pages_data
    except Exception:
        return []


def split_pages_into_chunks(pages_data: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Divide páginas em chunks mantendo informação da página."""
    chunks = []

    for page in pages_data:
        text = page["text"].strip()
        if len(text) < 20:
            continue

        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size].strip()
            if len(chunk_text) > 20:
                chunks.append({
                    "text": chunk_text,
                    "page_num": page["page_num"],
                    "total_pages": page["total_pages"]
                })

    return chunks


# ============== EMBEDDINGS ==============

def generate_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Gera embeddings para uma lista de textos."""
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


# ============== VECTOR STORE ==============

class SimpleVectorStore:
    """Armazenamento vetorial simples em memória."""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []

    def add(self, chunks: List[Dict], embeddings: List[List[float]], source: str = "manual"):
        """Adiciona documentos ao store com metadados."""
        for chunk, emb in zip(chunks, embeddings):
            self.documents.append(chunk["text"])
            self.embeddings.append(emb)
            self.metadatas.append({
                "source": source,
                "page_num": chunk.get("page_num", 0),
                "total_pages": chunk.get("total_pages", 0)
            })

    def search(self, query_embedding: List[float], n_results: int = 5) -> Tuple[List[str], List[dict]]:
        """Busca documentos similares."""
        if not self.embeddings:
            return [], []

        query = np.array(query_embedding)
        embeddings = np.array(self.embeddings)

        query_norm = query / np.linalg.norm(query)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm)

        top_indices = np.argsort(similarities)[::-1][:n_results]

        docs = [self.documents[i] for i in top_indices]
        metas = [self.metadatas[i] for i in top_indices]

        return docs, metas

    def clear(self):
        """Limpa o store."""
        self.documents = []
        self.embeddings = []
        self.metadatas = []

    def get_stats(self) -> Dict:
        """Retorna estatísticas do store."""
        if not self.metadatas:
            return {"total_chunks": 0, "total_pages": 0, "sources": []}

        sources = list(set(m["source"] for m in self.metadatas))
        total_pages = max((m["total_pages"] for m in self.metadatas), default=0)

        return {
            "total_chunks": len(self.documents),
            "total_pages": total_pages,
            "sources": sources
        }


# ============== CHAT COM CONTEXTO ==============

def build_context_prompt(question: str, relevant_chunks: List[str], metadatas: List[dict]) -> str:
    """Constrói o prompt com contexto estruturado."""
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(relevant_chunks, metadatas), 1):
        page_info = f"(Página {meta.get('page_num', '?')})" if meta.get('page_num') else ""
        context_parts.append(f"[Trecho {i}] {page_info}\n{chunk}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""Você é um assistente especializado em responder perguntas sobre documentos técnicos e manuais.

INSTRUÇÕES:
- Use APENAS as informações do contexto abaixo para responder
- Cite a página quando possível (ex: "Conforme a página X...")
- Se a informação não estiver no contexto, diga claramente que não encontrou
- Seja direto e objetivo nas respostas
- Use formatação markdown quando apropriado (listas, negrito, etc.)

CONTEXTO DO DOCUMENTO:
{context}

---

PERGUNTA DO USUÁRIO: {question}"""


def chat_with_context(
    question: str,
    relevant_chunks: List[str],
    metadatas: List[dict],
    client: OpenAI,
    chat_history: List[dict] = None
) -> str:
    """Gera resposta usando GPT-4 com contexto."""
    system_prompt = build_context_prompt(question, relevant_chunks, metadatas)

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content


# ============== PIPELINE ==============

def ask_question_pipeline(
    question: str,
    openai_client: OpenAI,
    vector_store: SimpleVectorStore,
    n_results: int = 5,
    chat_history: List[dict] = None
) -> Tuple[str, List[str], List[dict]]:
    """Pipeline de pergunta com metadados."""
    query_embedding = generate_single_embedding(question, openai_client)
    relevant_chunks, metadatas = vector_store.search(query_embedding, n_results)

    if not relevant_chunks:
        return "Não encontrei informações relevantes no documento para responder essa pergunta.", [], []

    answer = chat_with_context(question, relevant_chunks, metadatas, openai_client, chat_history)

    return answer, relevant_chunks, metadatas
