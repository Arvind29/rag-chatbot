import json
import math
import re
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
STORE = DATA / "vector_store.json"
FRONTEND = ROOT / "index.html"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
MAX_PDF_BYTES = 15 * 1024 * 1024
TOP_K = 5
MIN_SIMILARITY = 0.30
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Simple RAG Chatbot", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
_model = None

def model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def load_store():
    if not STORE.exists(): return {"version": 1, "chunks": []}
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("chunks"), list) else {"version": 1, "chunks": []}
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "chunks": []}

def save_store(store):
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STORE)

def clean(text): return re.sub(r"\s+", " ", text or "").strip()

def split_chunks(text, size=900, overlap=120):
    text = clean(text); out=[]; start=0
    while start < len(text):
        end=min(len(text), start+size); part=text[start:end]
        if part: out.append(part)
        if end >= len(text): break
        start=max(end-overlap, start+1)
    return out

def cosine(a,b):
    if len(a) != len(b): return 0.0
    da=math.sqrt(sum(x*x for x in a)); db=math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/(da*db) if da and db else 0.0

class ChatRequest(BaseModel): question: str

@app.get("/")
def home(): return FileResponse(FRONTEND)

@app.get("/health")
def health():
    ollama=False
    try: ollama=requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException: pass
    store=load_store()
    return {"status":"ok", "chunks":len(store["chunks"]), "ollama":ollama, "model":OLLAMA_MODEL}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400,"Only PDF files are supported")
    raw=await file.read()
    if len(raw) > MAX_PDF_BYTES: raise HTTPException(413,"PDF is larger than 15 MB")
    safe_name=Path(file.filename).name
    path=UPLOADS / f"{uuid.uuid4().hex}_{safe_name}"
    path.write_bytes(raw)
    try:
        reader=PdfReader(str(path)); store=load_store()
        store["chunks"]=[x for x in store["chunks"] if x.get("document") != safe_name]
        records=[]; texts=[]
        for page_no,page in enumerate(reader.pages,1):
            for part in split_chunks(page.extract_text() or ""):
                texts.append(part); records.append({"id":uuid.uuid4().hex,"document":safe_name,"page":page_no,"text":part})
        if not texts: raise HTTPException(400,"No readable text found in PDF")
        vectors=model().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()
        for r,v in zip(records,vectors): r["vector"]=v
        store["chunks"].extend(records); save_store(store)
        return {"status":"indexed","document":safe_name,"pages":len(reader.pages),"chunks":len(records)}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,f"Indexing failed: {e}")

def ollama_answer(question, ranked):
    context="\n\n".join(f"SOURCE {i}: {r['document']} | page {r['page']}\n{r['text']}" for i,r in enumerate(ranked,1))
    prompt=f'''You answer questions about uploaded PDF documents.
RULES:
1. Use ONLY the supplied SOURCE text.
2. Do not use outside knowledge.
3. Do not guess or invent missing information.
4. If the sources do not contain the answer, reply exactly: "I couldn't find that information in the uploaded PDF."
5. Answer the question directly and concisely.
6. When useful, mention the source page in the answer.

SOURCE TEXT:
{context}

QUESTION:
{question}

ANSWER:'''
    try:
        r=requests.post(f"{OLLAMA_URL}/api/generate", json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False,"options":{"temperature":0,"num_ctx":4096}}, timeout=180)
        if not r.ok: raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
        answer=(r.json().get("response") or "").strip()
        if not answer: raise RuntimeError("Ollama returned an empty response")
        return answer
    except requests.RequestException as e:
        raise RuntimeError("Ollama is not running. Start Ollama and run: ollama pull llama3.2:3b") from e

@app.post("/chat")
def chat(req: ChatRequest):
    question=clean(req.question)
    if not question: raise HTTPException(400,"Question is required")
    store=load_store()
    if not store["chunks"]: return {"answer":"Please upload a PDF first.","sources":[]}
    qvec=model().encode([question], normalize_embeddings=True)[0].tolist()
    scored=sorted(((cosine(qvec,x.get("vector",[])),x) for x in store["chunks"]), key=lambda z:z[0], reverse=True)
    relevant=[x for score,x in scored[:TOP_K] if score >= MIN_SIMILARITY]
    if not relevant:
        return {"answer":"I couldn't find that information in the uploaded PDF.","sources":[]}
    try: answer=ollama_answer(question,relevant)
    except RuntimeError as e: raise HTTPException(503,str(e))
    return {"answer":answer,"sources":[{"document_name":r["document"],"page_number":r["page"]} for r in relevant]}
