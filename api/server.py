from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import tempfile

from rag.ingest import ingest_pdf, ingest_url
from rag.ollama import generate_answer
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

app = FastAPI(title="Local RAG Assistant")


class ChatRequest(BaseModel):
    question: str


class UrlRequest(BaseModel):
    url: str


def store():
    return VectorStore()


def _analysis_context(s: VectorStore, max_chunks: int = 120) -> str:
    chunks = s.all_chunks(limit=max_chunks)
    parts = []
    for chunk in chunks:
        m = chunk.metadata
        parts.append(
            f"SOURCE: {m.get('document_name') or 'Unknown'}\n"
            f"PAGE: {m.get('page_number') or 'N/A'}\n"
            f"CHUNK: {m.get('chunk_index') or 'N/A'}\n"
            f"CONTENT:\n{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


def _structured_analysis(s: VectorStore, category: str, instruction: str) -> list[dict]:
    context = _analysis_context(s)
    if not context:
        return []

    prompt = f"""Analyze the supplied document excerpts for: {category}.

{instruction}

Return ONLY a JSON array. Each item must have exactly these fields:
{{"text":"concise finding","document_name":"source filename","page_number":1,"evidence":"short supporting excerpt"}}

If there are no supported findings, return []. Do not invent information.

DOCUMENT EXCERPTS:
{context}
"""

    raw = generate_answer("Produce the requested structured document analysis.", prompt)
    try:
        import json
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "local-rag"}


@app.get("/api/documents")
def documents():
    s = store()
    return {"documents": s.list_documents(), "total_chunks": s.count()}


@app.get("/api/dashboard")
def dashboard():
    s = store()
    docs = s.list_documents()
    if not docs:
        return {"documents": 0, "chunks": 0, "facts": [], "errors": [], "deadlines": [], "events": []}

    instructions = {
        "facts": "Extract only explicit important facts.",
        "errors": "Identify only explicit errors, contradictions, conflicts, missing information, or inconsistencies supported by the text.",
        "deadlines": "Extract only explicit dates, deadlines, expirations, due dates, renewal dates, or time-sensitive commitments.",
        "events": "Extract only important explicit events, actions, decisions, changes, or upcoming activities.",
    }

    results = {
        key: _structured_analysis(s, key, instruction)
        for key, instruction in instructions.items()
    }

    return {"documents": len(docs), "chunks": s.count(), **results}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question is required.")
    return RAGPipeline(store()).answer(request.question.strip())


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        temp.write(data)
        path = temp.name

    try:
        count = ingest_pdf(path, store(), document_name=os.path.basename(file.filename))
    finally:
        if os.path.exists(path):
            os.remove(path)

    return {"success": True, "filename": file.filename, "chunks": count}


@app.post("/api/add-url")
def add_url(request: UrlRequest):
    count = ingest_url(request.url.strip(), store())
    return {"success": True, "url": request.url, "chunks": count}
