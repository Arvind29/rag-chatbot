from .embeddings import embed_query
from .vector_store import VectorStore


def retrieve(question: str, top_k: int, threshold: float) -> list[dict]:
    vector = embed_query(question)
    return VectorStore().search(vector, top_k, threshold)
