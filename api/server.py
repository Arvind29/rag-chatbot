from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tempfile
import os

from rag.ingest import ingest_pdf, ingest_url
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

app = FastAPI(title="Local RAG Assistant")


class ChatRequest(BaseModel):
    question: str


class UrlRequest(BaseModel):
    url: str


def store():
    return VectorStore()


def analyze(s: VectorStore, prompt: str) -> str:
    result = RAGPipeline(s).answer(prompt)
    return result.get("answer", "").strip()


def as_items(text: str) -> list[str]:
    lines = [line.strip(" -*•\t") for line in text.splitlines() if line.strip()]
    return [line for line in lines if line and not line.lower().startswith(("i could", "no relevant", "not found"))][:8]


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

    prompts = {
        "facts": "Extract important explicit facts from the provided documents. Return only concise bullet points.",
        "errors": "Find explicit errors, contradictions, conflicts, missing information, or suspicious inconsistencies in the provided documents. Return only concise bullet points. If none are found, say none found.",
        "deadlines": "Find dates, deadlines, expirations, due dates, renewal dates, or time-sensitive commitments in the provided documents. Return only concise bullet points.",
        "events": "Find important events, actions, decisions, changes, or upcoming activities mentioned in the provided documents. Return only concise bullet points.",
    }

    results = {}
    for key, prompt in prompts.items():
        results[key] = as_items(analyze(s, prompt))

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
        os.remove(path)

    return {"success": True, "filename": file.filename, "chunks": count}


@app.post("/api/add-url")
def add_url(request: UrlRequest):
    count = ingest_url(request.url.strip(), store())
    return {"success": True, "url": request.url, "chunks": count}
