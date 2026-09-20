from .config import SIMILARITY_THRESHOLD, TOP_K
from .ollama import embed_query, generate_answer
from .store import VectorStore


class RAGPipeline:
    def __init__(self, store: VectorStore):
        self.store = store

    def answer(self, question: str, category: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        query_embedding = embed_query(question)
        matches = self.store.search(
            query_embedding,
            top_k=TOP_K,
            threshold=SIMILARITY_THRESHOLD,
            category=category,
        )
        if not matches:
            return {"answer": "I could not find enough relevant information in the provided sources.", "sources": []}

        context_parts = []
        sources = []
        for score, chunk in matches:
            metadata = chunk.metadata
            document_name = metadata.get("document_name") or "Unknown"
            page_number = metadata.get("page_number")
            context_parts.append(
                f"Source: {document_name}\n"
                f"Category: {metadata.get('category') or 'reference'}\n"
                f"Page: {page_number or 'N/A'}\n"
                f"Chunk: {metadata.get('chunk_index') or 'N/A'}\n"
                f"URL: {metadata.get('source_url') or 'N/A'}\n"
                f"Content:\n{chunk.content}"
            )
            sources.append({
                "document_name": document_name,
                "document_type": metadata.get("document_type"),
                "category": metadata.get("category"),
                "page_number": page_number,
                "chunk_index": metadata.get("chunk_index"),
                "source_url": metadata.get("source_url"),
                "score": round(float(score), 3),
            })

        answer = generate_answer(question, "\n\n---\n\n".join(context_parts))
        return {"answer": answer, "sources": sources}
