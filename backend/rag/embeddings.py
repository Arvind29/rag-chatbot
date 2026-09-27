from google import genai
from google.genai import types

from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, GEMINI_API_KEY

_client = None


def client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    result = client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return [list(item.values) for item in result.embeddings]


def embed_query(text: str) -> list[float]:
    result = client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return list(result.embeddings[0].values)
