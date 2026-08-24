from dataclasses import dataclass

from supabase import create_client, Client

from .config import SUPABASE_URL, SUPABASE_SECRET_KEY


@dataclass
class Chunk:
    id: str
    content: str
    embedding: list[float]
    metadata: dict


class VectorStore:

    def __init__(self):
        if not SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL is missing.")

        if not SUPABASE_SECRET_KEY:
            raise RuntimeError("SUPABASE_SECRET_KEY is missing.")

        self.client: Client = create_client(
            SUPABASE_URL,
            SUPABASE_SECRET_KEY,
        )

    def add(self, chunks: list[Chunk]):
        if not chunks:
            return

        # All chunks from one ingestion operation have
        # the same document_hash.
        document_hash = chunks[0].metadata.get(
            "document_hash"
        )

        # Remove the previous version of this document.
        if document_hash:
            self.client.table("documents").delete().eq(
                "document_hash",
                document_hash,
            ).execute()

        rows = []

        for chunk in chunks:
            metadata = chunk.metadata

            # PostgreSQL text fields cannot contain NUL characters.
            clean_content = chunk.content.replace(
                "\x00",
                "",
            )

            rows.append({
                "content": clean_content,

                "document_name": metadata.get(
                    "document_name",
                    "Unknown",
                ),

                "source_type": metadata.get(
                    "source_type",
                    "unknown",
                ),

                "source_url": metadata.get(
                    "source_url",
                ),

                "page_number": metadata.get(
                    "page_number",
                ),

                "chunk_index": metadata.get(
                    "chunk_index",
                    0,
                ),

                "document_hash": metadata.get(
                    "document_hash",
                ),

                "embedding": chunk.embedding,
            })

        self.client.table(
            "documents"
        ).insert(rows).execute()

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        threshold: float,
    ):
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": top_k,
            },
        ).execute()

        matches = []

        for row in result.data or []:

            metadata = {
                "document_name": row.get(
                    "document_name"
                ),
                "source_type": row.get(
                    "source_type"
                ),
                "source_url": row.get(
                    "source_url"
                ),
                "page_number": row.get(
                    "page_number"
                ),
                "chunk_index": row.get(
                    "chunk_index"
                ),
                "document_hash": row.get(
                    "document_hash"
                ),
            }

            chunk = Chunk(
                id=str(row["id"]),
                content=row["content"],
                embedding=[],
                metadata=metadata,
            )

            matches.append(
                (
                    float(row["similarity"]),
                    chunk,
                )
            )

        return matches