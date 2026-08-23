from .config import SIMILARITY_THRESHOLD, TOP_K
from .gemini import embed_query, generate_answer
from .store import VectorStore

class RAGPipeline:
    def __init__(self, store: VectorStore):
        self.store = store

    def answer(self, question: str) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding = embed_query(question)
        matches = self.store.search(
            query_embedding,
            top_k=TOP_K,
            threshold=SIMILARITY_THRESHOLD,
        )

        if not matches:
            return {
                "answer": "I could not find enough relevant information in the provided sources.",
                "sources": [],
            }

        context_parts = []
        sources = []

        for score, chunk in matches:
            metadata = chunk.metadata
            context_parts.append(
                f"Source: {metadata.get('document_name', 'Unknown')}\n"
                f"Page: {metadata.get('page_number', 'N/A')}\n"
                f"URL: {metadata.get('source_url', 'N/A')}\n"
                f"Content:\n{chunk.content}"
            )
            sources.append(
                f"{metadata.get('document_name', 'Unknown')} "
                f"(score={score:.3f})"
            )

        answer = generate_answer(question, "\n\n---\n\n".join(context_parts))
        return {"answer": answer, "sources": sources}
