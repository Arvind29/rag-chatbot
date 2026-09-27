import os
import json
import math
import re
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
STORE = DATA / "vector_store.json"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Simple RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

_model = None

def model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def load_store():
    if not STORE.exists():
        return {"chunks": []}
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {"chunks": []}

def save_store(store):
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STORE)

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def chunks(text, size=900, overlap=120):
    text = clean(text)
    out=[]
    start=0
    while start < len(text):
        end=min(len(text), start+size)
        part=text[start:end]
        if part: out.append(part)
        if end >= len(text): break
        start=max(end-overlap, start+1)
    return out

def cosine(a,b):
    da=math.sqrt(sum(x*x for x in a)); db=math.sqrt(sum(x*x for x in b))
    if not da or not db: return 0.0
    return sum(x*y for x,y in zip(a,b))/(da*db)

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status":"ok", "chunks":len(load_store()["chunks"])}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400,"Only PDF files are supported")
    raw=await file.read()
    path=UPLOADS / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    path.write_bytes(raw)
    try:
        reader=PdfReader(str(path))
        store=load_store()
        store["chunks"]=[x for x in store["chunks"] if x.get("document") != file.filename]
        records=[]; texts=[]
        for page_no,page in enumerate(reader.pages,1):
            for part in chunks(page.extract_text() or ""):
                texts.append(part)
                records.append({"id":uuid.uuid4().hex,"document":file.filename,"page":page_no,"text":part})
        if not texts:
            raise HTTPException(400,"No readable text found in PDF")
        vectors=model().encode(texts, normalize_embeddings=True).tolist()
        for r,v in zip(records,vectors): r["vector"]=v
        store["chunks"].extend(records)
        save_store(store)
        return {"status":"indexed","document":file.filename,"chunks":len(records)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500,str(e))

@app.post("/chat")
def chat(req: ChatRequest):
    question=clean(req.question)
    if not question: raise HTTPException(400,"Question is required")
    store=load_store()
    if not store["chunks"]:
        return {"answer":"Please upload a PDF first.","sources":[]}
    qvec=model().encode([question], normalize_embeddings=True)[0].tolist()
    ranked=sorted(store["chunks"], key=lambda x: cosine(qvec,x["vector"]), reverse=True)[:5]
    context="\n\n".join(f"[{r['document']} page {r['page']}] {r['text']}" for r in ranked)
    # Deliberately simple: extractive answer. No paid LLM/API key is required.
    terms=[x.lower() for x in re.findall(r"\b\w+\b",question) if len(x)>2]
    sentences=re.split(r"(?<=[.!?])\s+", context)
    selected=[]
    for s in sentences:
        score=sum(1 for t in terms if t in s.lower())
        if score: selected.append((score,s))
    selected=[s for _,s in sorted(selected,reverse=True)[:6]]
    answer=" ".join(selected) if selected else "I found relevant document sections, but I could not extract a direct answer. Please try a more specific question."
    return {"answer":answer,"sources":[{"document_name":r["document"],"page_number":r["page"]} for r in ranked]}
