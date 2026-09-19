from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import QDRANT_COLLECTION, QDRANT_PATH


@dataclass
class Chunk:
    id: str
    content: str
    embedding: list[float]
    metadata: dict


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if any(c.name == QDRANT_COLLECTION for c in collections):
            return

        self.client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            ),
        )

    def add(self, chunks: list[Chunk]):
        if not chunks:
            return

        document_hash = chunks[0].metadata.get("document_hash")
        if document_hash:
            self.client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector={"filter": {"must": [{"key": "document_hash", "match": {"value": document_hash}}]}},
            )

        points = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["content"] = chunk.content.replace("\x00", "")
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload=metadata,
                )
            )

        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    def search(self, query_embedding: list[float], top_k: int, threshold: float):
        result = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            score_threshold=threshold,
            with_payload=True,
        )

        matches = []
        for point in result.points:
            payload = point.payload or {}
            metadata = {
                "document_name": payload.get("document_name"),
                "source_type": payload.get("source_type"),
                "source_url": payload.get("source_url"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "document_hash": payload.get("document_hash"),
            }
            matches.append(
                (
                    float(point.score),
                    Chunk(
                        id=str(point.id),
                        content=str(payload.get("content", "")),
                        embedding=[],
                        metadata=metadata,
                    ),
                )
            )
        return matches
