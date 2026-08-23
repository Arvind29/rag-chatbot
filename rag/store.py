from dataclasses import dataclass
import numpy as np

@dataclass
class Chunk:
    id: str
    content: str
    embedding: list[float]
    metadata: dict

class VectorStore:
    """Prototype-only in-memory vector store.

    This is intentionally replaceable. The production version will use
    a hosted persistent vector database behind the same interface.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]):
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int, threshold: float):
        if not self.chunks:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)

        scored = []
        for chunk in self.chunks:
            v = np.asarray(chunk.embedding, dtype=np.float32)
            denom = q_norm * np.linalg.norm(v)
            score = float(np.dot(q, v) / denom) if denom else 0.0
            if score >= threshold:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
