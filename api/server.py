from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64,binascii,os,tempfile
from rag.ingest import ingest_pdf, ingest_url
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

app=FastAPI(title="Private RAG Assistant",version="2.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:3005","http://127.0.0.1:3005"],allow_methods=["*"],allow_headers=["*"])
class ChatRequest(BaseModel): question:str=Field(min_length=1,max_length=12000); history:list[dict]=Field(default_factory=list)
class UrlRequest(BaseModel): url:str=Field(min_length=1,max_length=4000)
class DeleteRequest(BaseModel): document_hash:str=Field(min_length=1,max_length=128)
class PdfRequest(BaseModel): filename:str; content_base64:str
@app.get("/api/health")
def health():
 from rag.config import GEMINI_CHAT_MODEL,GEMINI_EMBEDDING_MODEL
 return {"status":"ok","service":"private-rag-assistant","chat_model":GEMINI_CHAT_MODEL,"embedding_model":GEMINI_EMBEDDING_MODEL}
@app.get("/api/documents")
def documents():
 try:
  docs=VectorStore().list_documents(); return {"documents":docs,"count":len(docs)}
 except Exception as e: raise HTTPException(500,detail=str(e)) from e
@app.post("/api/chat")
def chat(req:ChatRequest):
 try:return RAGPipeline(VectorStore()).answer(req.question,req.history[-6:])
 except ValueError as e:raise HTTPException(400,detail=str(e)) from e
 except Exception as e:raise HTTPException(500,detail=str(e)) from e
@app.post("/api/add-url")
def add_url(req:UrlRequest):
 try:return {"success":True,"chunks":ingest_url(req.url,VectorStore()),"url":req.url}
 except ValueError as e:raise HTTPException(400,detail=str(e)) from e
 except Exception as e:raise HTTPException(500,detail=str(e)) from e
@app.delete("/api/document")
def delete(req:DeleteRequest):
 try:return {"success":True,"deleted":VectorStore().delete_document(req.document_hash)}
 except Exception as e:raise HTTPException(500,detail=str(e)) from e
@app.post("/api/upload-pdf")
def upload(req:PdfRequest):
 from rag.config import MAX_PDF_BYTES
 try:
  encoded=req.content_base64.split(',',1)[-1]
  try:data=base64.b64decode(encoded,validate=True)
  except (binascii.Error,ValueError) as e:raise HTTPException(400,detail="Invalid base64 PDF data.") from e
  if not data:raise HTTPException(400,detail="PDF is empty.")
  if len(data)>MAX_PDF_BYTES:raise HTTPException(413,detail="PDF exceeds configured size limit.")
  name=os.path.basename(req.filename.strip()) or "document.pdf"
  if not name.lower().endswith('.pdf'):raise HTTPException(400,detail="Only PDF files are supported.")
  path=None
  try:
   with tempfile.NamedTemporaryFile(suffix='.pdf',delete=False) as f:f.write(data);path=f.name
   count=ingest_pdf(path,VectorStore(),document_name=name)
  finally:
   if path and os.path.exists(path):os.remove(path)
  return {"success":True,"chunks":count,"filename":name}
 except HTTPException:raise
 except Exception as e:raise HTTPException(500,detail=str(e)) from e
