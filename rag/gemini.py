import requests

from .config import (
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
)


def _post(path: str, payload: dict) -> dict:
    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}{path}",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        result = _post("/api/embed", {
            "model": OLLAMA_EMBEDDING_MODEL,
            "input": text,
        })
        vectors.extend(result.get("embeddings", []))
    if vectors and len(vectors[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vectors[0])}"
        )
    return vectors


def embed_query(text: str) -> list[float]:
    vectors = embed_texts([text])
    if not vectors:
        raise RuntimeError("Ollama returned no embedding.")
    return vectors[0]


def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a grounded RAG assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
- If the context does not contain enough information, say that the information was not found in the provided sources.
- Do not invent facts, citations, page numbers, URLs, or quotations.
- Clearly distinguish explicit source information from reasonable inference.
- Keep the answer concise and useful.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""
    result = _post("/api/generate", {
        "model": OLLAMA_CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
    })
    return result.get("response", "")
