from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

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
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_hash",
                            match=MatchValue(value=document_hash),
                        )
                    ]
                ),
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

    def count(self) -> int:
        return int(
            self.client.count(
                collection_name=QDRANT_COLLECTION,
                exact=True,
            ).count
        )

    def all_chunks(self, limit: int = 2000) -> list[Chunk]:
        points, _ = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        chunks = []
        for point in points:
            payload = point.payload or {}
            chunks.append(
                Chunk(
                    id=str(point.id),
                    content=str(payload.get("content", "")),
                    embedding=[],
                    metadata={
                        "document_name": payload.get("document_name"),
                        "source_type": payload.get("source_type"),
                        "source_url": payload.get("source_url"),
                        "page_number": payload.get("page_number"),
                        "chunk_index": payload.get("chunk_index"),
                        "document_hash": payload.get("document_hash"),
                    },
                )
            )
        return chunks

    def list_documents(self) -> list[dict]:
        chunks = self.all_chunks()
        documents = {}

        for chunk in chunks:
            metadata = chunk.metadata
            key = metadata.get("document_hash") or metadata.get("document_name")
            if not key:
                continue

            if key not in documents:
                documents[key] = {
                    "document_name": metadata.get("document_name") or "Unknown",
                    "source_type": metadata.get("source_type") or "unknown",
                    "source_url": metadata.get("source_url"),
                    "document_hash": metadata.get("document_hash"),
                    "chunks": 0,
                }

            documents[key]["chunks"] += 1

        return sorted(
            documents.values(),
            key=lambda item: item["document_name"].lower(),
        )
