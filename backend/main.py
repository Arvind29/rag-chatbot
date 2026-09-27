import os
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag.chunker import make_chunks
from .rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    FRONTEND_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_PDF_MB,
    MIN_SIMILARITY,
    TOP_K,
)
from .rag.document_loader import extract_pages
from .rag.embeddings import embed_texts
from .rag.gemini import answer_question
from .rag.retriever import retrieve
from .rag.vector_store import VectorStore

app = FastAPI(title="PDF RAG Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

STORE = VectorStore()
DEPLOYED = os.getenv("VERCEL", "") == "1"


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


def safe_filename(name: str) -> str:
    cleaned = Path(name or "document.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", cleaned).strip(" .")
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:180]


def chunk_count() -> int:
    return len(STORE.load()["chunks"])


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "vercel-read-only" if DEPLOYED else "local-indexing",
        "gemini": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "documents": len(STORE.documents()),
        "chunks": chunk_count(),
    }


@app.get("/api/documents")
def documents():
    return {"documents": STORE.documents()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if DEPLOYED:
        raise HTTPException(403, "Document upload is disabled on Vercel. Index documents locally and deploy the generated vector store.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    raw = await file.read()
    max_bytes = MAX_PDF_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(413, f"PDF exceeds the {MAX_PDF_MB} MB limit.")

    name = safe_filename(file.filename)
    path = DOCUMENTS_DIR / f"{uuid.uuid4().hex}_{name}"
    try:
        path.write_bytes(raw)
        pages = extract_pages(path)
        chunks = make_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP, name)
        if not chunks:
            raise HTTPException(400, "No readable text was found in this PDF. Scanned/image-only PDFs need OCR before indexing.")

        vectors = []
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            vectors.extend(embed_texts([row["text"] for row in batch]))
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count did not match chunk count")
        for row, vector in zip(chunks, vectors):
            row["vector"] = vector
        STORE.add(chunks, EMBEDDING_MODEL)
        return {"status": "indexed", "document": name, "pages": len(pages), "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Indexing failed: {type(exc).__name__}") from exc
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@app.delete("/api/documents/{document_name}")
def delete_document(document_name: str):
    if DEPLOYED:
        raise HTTPException(403, "Document deletion is disabled on Vercel read-only mode.")
    deleted = STORE.delete_document(document_name)
    if not deleted:
        raise HTTPException(404, "Document not found.")
    return {"status": "deleted", "document": document_name, "chunks_removed": deleted}


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not configured on the server.")
    if not STORE.load()["chunks"]:
        return {"answer": "Please index a PDF first.", "sources": []}

    try:
        chunks = retrieve(question, TOP_K, MIN_SIMILARITY)
        if not chunks:
            return {
                "answer": "I couldn't find that information in the uploaded document.",
                "sources": [],
            }
        answer = answer_question(question, chunks)
        sources = [
            {"document": row["document"], "page": row["page"], "score": row["score"]}
            for row in chunks
        ]
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        raise HTTPException(503, f"RAG service error: {type(exc).__name__}") from exc
