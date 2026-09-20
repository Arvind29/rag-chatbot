from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from .config import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_PATH


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
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    def add(self, chunks: list[Chunk]):
        if not chunks:
            return
        document_hash = chunks[0].metadata.get("document_hash")
        if document_hash:
            self.delete_document(document_hash=document_hash)
        points = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["content"] = chunk.content.replace("\x00", "")
            points.append(PointStruct(id=chunk.id, vector=chunk.embedding, payload=metadata))
        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    @staticmethod
    def _metadata(payload: dict) -> dict:
        return {
            "document_name": payload.get("document_name"),
            "document_type": payload.get("document_type"),
            "source_type": payload.get("source_type"),
            "source_url": payload.get("source_url"),
            "category": payload.get("category"),
            "page_number": payload.get("page_number"),
            "chunk_index": payload.get("chunk_index"),
            "document_hash": payload.get("document_hash"),
        }

    def search(self, query_embedding: list[float], top_k: int, threshold: float, category: str | None = None):
        query_filter = None
        if category:
            query_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
        result = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            score_threshold=threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            (float(point.score), Chunk(id=str(point.id), content=str((point.payload or {}).get("content", "")), embedding=[], metadata=self._metadata(point.payload or {})))
            for point in result.points
        ]

    def count(self) -> int:
        return int(self.client.count(collection_name=QDRANT_COLLECTION, exact=True).count)

    def all_chunks(self, limit: int = 2000) -> list[Chunk]:
        points, _ = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            Chunk(
                id=str(point.id),
                content=str((point.payload or {}).get("content", "")),
                embedding=[],
                metadata=self._metadata(point.payload or {}),
            )
            for point in points
        ]

    def list_documents(self) -> list[dict]:
        documents = {}
        for chunk in self.all_chunks():
            metadata = chunk.metadata
            key = metadata.get("document_hash") or metadata.get("document_name")
            if not key:
                continue
            if key not in documents:
                documents[key] = {
                    "document_name": metadata.get("document_name") or "Unknown",
                    "document_type": metadata.get("document_type") or "unknown",
                    "source_type": metadata.get("source_type") or "unknown",
                    "source_url": metadata.get("source_url"),
                    "category": metadata.get("category") or "reference",
                    "document_hash": metadata.get("document_hash"),
                    "chunks": 0,
                }
            documents[key]["chunks"] += 1
        return sorted(documents.values(), key=lambda item: item["document_name"].lower())

    def delete_document(self, document_hash: str | None = None, document_name: str | None = None) -> int:
        if not document_hash and not document_name:
            raise ValueError("document_hash or document_name is required")
        key = "document_hash" if document_hash else "document_name"
        value = document_hash or document_name
        before = self.count()
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))]),
        )
        return max(0, before - self.count())
