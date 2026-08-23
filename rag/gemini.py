from google import genai
from google.genai import types

from .config import GEMINI_CHAT_MODEL, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIM, required_env

_client = None

def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=required_env("GEMINI_API_KEY"))
    return _client

def embed_texts(texts: list[str]) -> list[list[float]]:
    result = client().models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    return [item.values for item in result.embeddings]

def embed_query(text: str) -> list[float]:
    result = client().models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return result.embeddings[0].values

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
    response = client().models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
    )
    return response.text or ""
