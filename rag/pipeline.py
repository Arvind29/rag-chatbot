from .config import SIMILARITY_THRESHOLD, TOP_K
from .ollama import embed_query, generate_answer
from .store import VectorStore


SUMMARY_WORDS = (
    "summarize",
    "summarise",
    "summary",
    "give me a summary",
    "give a summary",
    "summarize it",
    "summarise it",
)


class RAGPipeline:
    def __init__(self, store: VectorStore):
        self.store = store

    @staticmethod
    def _is_summary_request(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        return any(phrase in normalized for phrase in SUMMARY_WORDS)

    def _summary_answer(self, question: str, category: str | None = None) -> dict | None:
        """For summary requests, retrieve the document itself rather than
        embedding the generic word 'summarize'. Semantic search is appropriate
        for factual questions, but it is the wrong retrieval strategy for a
        document-wide summary request.
        """
        chunks = self.store.all_chunks(limit=2000)
        if category:
            chunks = [c for c in chunks if c.metadata.get("category") == category]
        if not chunks:
            return None

        # If exactly one document is indexed, treat "summarize it" as a
        # request for that document. This is the normal single-document RAG
        # workflow and avoids irrelevant vector matches.
        documents = {}
        for chunk in chunks:
            key = chunk.metadata.get("document_hash") or chunk.metadata.get("document_name")
            documents.setdefault(key, []).append(chunk)

        if len(documents) != 1:
            return None

        selected = next(iter(documents.values()))
        selected.sort(key=lambda c: (
            c.metadata.get("page_number") or 0,
            c.metadata.get("chunk_index") or 0,
        ))

        context_parts = []
        source_map = {}
        for chunk in selected:
            metadata = chunk.metadata
            document_name = metadata.get("document_name") or "Unknown"
            page_number = metadata.get("page_number")
            context_parts.append(
                f"Source: {document_name}\n"
                f"Page: {page_number or 'N/A'}\n"
                f"Chunk: {metadata.get('chunk_index') or 'N/A'}\n"
                f"Content:\n{chunk.content}"
            )
            source_map[(document_name, page_number)] = {
                "document_name": document_name,
                "document_type": metadata.get("document_type"),
                "category": metadata.get("category"),
                "page_number": page_number,
                "chunk_index": metadata.get("chunk_index"),
                "source_url": metadata.get("source_url"),
            }

        summary_prompt = f"""Create a concise but useful summary of the uploaded document.

The user asked: {question}

Use ONLY the document content below.
- Identify the main topic/purpose.
- Summarize the major sections, concepts, facts, or conclusions.
- Preserve important technical terms and numbers when present.
- Do not claim information that is not in the document.
- Do not say that the query word 'summary' is missing from the source; the task is to summarize the supplied document.
- Use clear headings and bullet points where useful.

DOCUMENT CONTENT:
{chr(10).join(context_parts)}
"""
        answer = generate_answer(question, summary_prompt)
        return {"answer": answer, "sources": list(source_map.values())}

    def answer(self, question: str, category: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self._is_summary_request(question):
            summary = self._summary_answer(question, category=category)
            if summary is not None:
                return summary

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
