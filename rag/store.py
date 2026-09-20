from dataclasses import dataclass
from supabase import create_client, Client
from .config import SUPABASE_URL, SUPABASE_SECRET_KEY, DOCUMENT_MAX_CHUNKS

@dataclass
class Chunk:
    id: str
    content: str
    embedding: list[float]
    metadata: dict

class VectorStore:
    def __init__(self):
        if not SUPABASE_URL: raise RuntimeError("SUPABASE_URL is missing.")
        if not SUPABASE_SECRET_KEY: raise RuntimeError("SUPABASE_SECRET_KEY is missing.")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

    def add(self, chunks: list[Chunk]):
        if not chunks: return
        document_hash = chunks[0].metadata.get("document_hash")
        if document_hash:
            self.client.table("documents").delete().eq("document_hash", document_hash).execute()
        rows = []
        for chunk in chunks:
            m = chunk.metadata
            rows.append({"content": chunk.content.replace("\x00", ""), "document_name": m.get("document_name", "Unknown"), "source_type": m.get("source_type", "unknown"), "source_url": m.get("source_url"), "page_number": m.get("page_number"), "chunk_index": m.get("chunk_index", 0), "document_hash": m.get("document_hash"), "embedding": chunk.embedding})
        self.client.table("documents").insert(rows).execute()

    def search(self, query_embedding: list[float], top_k: int, threshold: float):
        result = self.client.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": threshold, "match_count": top_k}).execute()
        return [self._row_to_match(row) for row in (result.data or [])]

    def list_documents(self) -> list[dict]:
        result = self.client.table("documents").select("document_name,source_type,source_url,document_hash,chunk_index,page_number").order("document_name").order("chunk_index").execute()
        grouped = {}
        for row in result.data or []:
            key = row.get("document_hash") or row.get("document_name") or "unknown"
            item = grouped.setdefault(key, {"document_name": row.get("document_name") or "Unknown", "source_type": row.get("source_type") or "unknown", "source_url": row.get("source_url"), "document_hash": row.get("document_hash"), "chunk_count": 0, "page_count": 0})
            item["chunk_count"] += 1
            if row.get("page_number"): item["page_count"] = max(item["page_count"], int(row["page_number"]))
        return list(grouped.values())

    def get_document_chunks(self, document_hash: str, limit: int | None = None) -> list[Chunk]:
        result = self.client.table("documents").select("id,content,embedding,document_name,source_type,source_url,page_number,chunk_index,document_hash").eq("document_hash", document_hash).order("chunk_index").limit(limit or DOCUMENT_MAX_CHUNKS).execute()
        return [self._row_to_chunk(row) for row in (result.data or [])]

    def delete_document(self, document_hash: str) -> int:
        result = self.client.table("documents").delete().eq("document_hash", document_hash).execute()
        return len(result.data or [])

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        return Chunk(id=str(row.get("id")), content=row.get("content") or "", embedding=row.get("embedding") or [], metadata={"document_name": row.get("document_name"), "source_type": row.get("source_type"), "source_url": row.get("source_url"), "page_number": row.get("page_number"), "chunk_index": row.get("chunk_index"), "document_hash": row.get("document_hash")})

    @classmethod
    def _row_to_match(cls, row: dict):
        return float(row.get("similarity", 0.0)), cls._row_to_chunk(row)
