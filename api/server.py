from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import threading
import requests

from rag.config import DOCUMENT_CATEGORIES, MAX_FILE_BYTES
from rag.ingest import SUPPORTED_EXTENSIONS, ingest_file, ingest_url
from rag.ollama import generate_answer
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

app = FastAPI(title="Local RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    category: str | None = None

class UrlRequest(BaseModel):
    url: str
    category: str = "reference"

# Qdrant Local locks its on-disk directory. Do not create a new client per
# request, and do not rely on lru_cache for singleton initialization: the
# cache can execute the wrapped function more than once during a concurrent
# cache miss. Initialize exactly once during FastAPI startup instead.
_store: VectorStore | None = None
_store_lock = threading.Lock()

@app.on_event("startup")
def initialize_store():
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()

@app.on_event("shutdown")
def shutdown_store():
    global _store
    with _store_lock:
        if _store is not None:
            try:
                _store.client.close()
            except Exception:
                pass
            _store = None

def store() -> VectorStore:
    if _store is None:
        raise RuntimeError("Vector store is not initialized. Restart FastAPI.")
    return _store

@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    # Always return JSON so the browser never tries to parse a plain-text
    # "Internal Server Error" response as JSON.
    print(f"Unhandled API error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check the FastAPI console."})

def _analysis_context(s: VectorStore, max_chunks: int = 120) -> str:
    parts = []
    for chunk in s.all_chunks(limit=max_chunks):
        m = chunk.metadata
        parts.append(f"SOURCE: {m.get('document_name') or 'Unknown'}\nTYPE: {m.get('document_type') or 'unknown'}\nCATEGORY: {m.get('category') or 'reference'}\nPAGE: {m.get('page_number') or 'N/A'}\nCHUNK: {m.get('chunk_index') or 'N/A'}\nCONTENT:\n{chunk.content}")
    return "\n\n---\n\n".join(parts)

def _structured_analysis(s: VectorStore, category: str, instruction: str) -> list[dict]:
    context = _analysis_context(s)
    if not context:
        return []
    prompt = f"""Analyze the supplied document excerpts for: {category}.\n\n{instruction}\n\nReturn ONLY a JSON array. Each item must have exactly: {{\"text\":\"concise finding\",\"document_name\":\"source filename\",\"page_number\":1,\"evidence\":\"short supporting excerpt\"}}\n\nIf there are no supported findings, return []. Never invent information.\n\nDOCUMENT EXCERPTS:\n{context}"""
    raw = generate_answer("Produce the requested structured document analysis.", prompt)
    try:
        import json
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "local-rag", "qdrant": _store is not None}

@app.get("/api/categories")
def categories():
    return {"categories": list(DOCUMENT_CATEGORIES), "formats": sorted(SUPPORTED_EXTENSIONS)}

@app.get("/api/documents")
def documents():
    try:
        s = store()
        return {"documents": s.list_documents(), "total_chunks": s.count()}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

@app.get("/api/dashboard")
def dashboard():
    try:
        s = store()
        docs = s.list_documents()
        if not docs:
            return {"documents": 0, "chunks": 0, "facts": [], "errors": [], "deadlines": [], "events": []}
        instructions = {"facts": "Extract only explicit important facts.", "errors": "Identify only explicit errors, contradictions, conflicts, missing information, or inconsistencies supported by the text.", "deadlines": "Extract only explicit dates, deadlines, expirations, due dates, renewal dates, or time-sensitive commitments.", "events": "Extract only important explicit events, actions, decisions, changes, or upcoming activities."}
        return {"documents": len(docs), "chunks": s.count(), **{key: _structured_analysis(s, key, instruction) for key, instruction in instructions.items()}}
    except Exception as exc:
        raise HTTPException(503, f"Dashboard unavailable: {exc}") from exc

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question is required.")
    if request.category and request.category not in DOCUMENT_CATEGORIES:
        raise HTTPException(400, "Invalid category.")
    try:
        return RAGPipeline(store()).answer(request.question.strip(), category=request.category)
    except requests.RequestException as exc:
        raise HTTPException(503, "Cannot reach Ollama. Make sure Ollama is running and the configured models are available.") from exc
    except Exception as exc:
        print(f"Chat error: {exc}")
        raise HTTPException(500, f"Chat failed: {exc}") from exc

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), category: str = "reference"):
    if category not in DOCUMENT_CATEGORIES:
        raise HTTPException(400, "Invalid category.")
    if not file.filename:
        raise HTTPException(400, "Filename is required.")
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "File exceeds configured size limit.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(data)
        path = temp.name
    try:
        count = ingest_file(path, store(), category=category, document_name=os.path.basename(file.filename))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(503, "Cannot reach Ollama for embeddings. Make sure Ollama is running and the embedding model is available.") from exc
    finally:
        if os.path.exists(path):
            os.remove(path)
    return {"success": True, "filename": file.filename, "category": category, "chunks": count}

@app.post("/api/add-url")
def add_url(request: UrlRequest):
    if request.category not in DOCUMENT_CATEGORIES:
        raise HTTPException(400, "Invalid category.")
    try:
        count = ingest_url(request.url.strip(), store(), category=request.category)
    except (ValueError, requests.RequestException) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"success": True, "url": request.url, "category": request.category, "chunks": count}
