from .config import SIMILARITY_THRESHOLD, TOP_K
from .ollama import embed_query, generate_answer
from .store import VectorStore


SUMMARY_WORDS = ("summarize", "summarise", "summary", "give me a summary", "give a summary", "summarize it", "summarise it")
GLOBAL_WORDS = ("all documents", "all docs", "every document", "everything i uploaded", "my documents", "my docs")


class RAGPipeline:
    def __init__(self, store: VectorStore):
        self.store = store

    @staticmethod
    def _is_summary_request(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        return any(phrase in normalized for phrase in SUMMARY_WORDS)

    @staticmethod
    def _is_global_request(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        return any(phrase in normalized for phrase in GLOBAL_WORDS)

    def _build_context(self, chunks):
        parts = []
        sources = []
        seen = set()
        for chunk in chunks:
            m = chunk.metadata
            name = m.get("document_name") or "Unknown"
            page = m.get("page_number")
            parts.append(
                f"Source: {name}\nCategory: {m.get('category') or 'reference'}\n"
                f"Page: {page or 'N/A'}\nChunk: {m.get('chunk_index') or 'N/A'}\nContent:\n{chunk.content}"
            )
            key = (name, page, m.get("chunk_index"))
            if key not in seen:
                sources.append({
                    "document_name": name,
                    "document_type": m.get("document_type"),
                    "category": m.get("category"),
                    "page_number": page,
                    "chunk_index": m.get("chunk_index"),
                    "source_url": m.get("source_url"),
                })
                seen.add(key)
        return "\n\n---\n\n".join(parts), sources

    def _document_summary(self, question: str, category: str | None = None):
        chunks = self.store.all_chunks(limit=2000)
        if category:
            chunks = [c for c in chunks if c.metadata.get("category") == category]
        if not chunks:
            return None

        documents = {}
        for chunk in chunks:
            key = chunk.metadata.get("document_hash") or chunk.metadata.get("document_name")
            documents.setdefault(key, []).append(chunk)

        # If the user explicitly asks for all documents, summarize each document
        # independently and then combine the results. This avoids cross-document
        # blending and keeps provenance visible.
        if self._is_global_request(question):
            document_outputs = []
            all_sources = []
            for name, selected in documents.items():
                selected.sort(key=lambda c: (c.metadata.get("page_number") or 0, c.metadata.get("chunk_index") or 0))
                context, sources = self._build_context(selected)
                prompt = f"""You are summarizing one private document. Use ONLY the supplied content.\n\nUser request: {question}\n\nProduce a concise summary with the document name as a heading. Include major topics, facts, conclusions, and important numbers only when explicitly present. Do not invent or merge facts with other documents.\n\nDOCUMENT:\n{context}"""
                document_outputs.append(generate_answer(question, prompt))
                all_sources.extend(sources)
            combined = "\n\n---\n\n".join(document_outputs)
            final_prompt = f"""Combine the following independently generated document summaries into a concise overview of the user's uploaded knowledge base. Keep each document distinguishable. Use ONLY the supplied summaries. Do not invent facts.\n\n{combined}"""
            return {"answer": generate_answer(question, final_prompt), "sources": all_sources, "mode": "global_summary"}

        if len(documents) != 1:
            return None
        selected = next(iter(documents.values()))
        selected.sort(key=lambda c: (c.metadata.get("page_number") or 0, c.metadata.get("chunk_index") or 0))
        context, sources = self._build_context(selected)
        prompt = f"""Create a concise but useful summary of the supplied document. Use ONLY the document content. Preserve important technical terms and numbers. Do not invent information.\n\nUser request: {question}\n\nDOCUMENT:\n{context}"""
        return {"answer": generate_answer(question, prompt), "sources": sources, "mode": "document_summary"}

    def answer(self, question: str, category: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self._is_summary_request(question):
            summary = self._document_summary(question, category)
            if summary is not None:
                return summary

        query_embedding = embed_query(question)
        matches = self.store.search(query_embedding, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD, category=category)
        if not matches:
            return {"answer": "I could not find enough relevant information in the provided sources.", "sources": [], "mode": "no_match"}
        chunks = [chunk for _, chunk in matches]
        context, sources = self._build_context(chunks)
        for source, (score, _) in zip(sources, matches):
            source["score"] = round(float(score), 3)
        answer = generate_answer(question, context)
        return {"answer": answer, "sources": sources, "mode": "rag"}
