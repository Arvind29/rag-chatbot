from http.server import BaseHTTPRequestHandler
import base64,binascii,json,os,tempfile
from urllib.parse import urlparse,parse_qs

def _json(h,status,data):
    body=json.dumps(data,ensure_ascii=False).encode('utf-8'); h.send_response(status); h.send_header('Content-Type','application/json; charset=utf-8'); h.send_header('Content-Length',str(len(body))); h.send_header('Access-Control-Allow-Origin','*'); h.send_header('Access-Control-Allow-Methods','GET, POST, DELETE, OPTIONS'); h.send_header('Access-Control-Allow-Headers','Content-Type'); h.end_headers(); h.wfile.write(body)
class handler(BaseHTTPRequestHandler):
    def _route(self): return parse_qs(urlparse(self.path).query).get('route',[''])[0]
    def _body(self):
        n=int(self.headers.get('Content-Length','0'))
        if n<=0: raise ValueError('Request body is required.')
        return json.loads(self.rfile.read(n).decode('utf-8'))
    def do_OPTIONS(self): _json(self,200,{'status':'ok'})
    def do_GET(self):
        try:
            route=self._route()
            if route=='health':
                from rag.config import GEMINI_CHAT_MODEL,GEMINI_EMBEDDING_MODEL
                _json(self,200,{'status':'ok','service':'private-rag-assistant','chat_model':GEMINI_CHAT_MODEL,'embedding_model':GEMINI_EMBEDDING_MODEL}); return
            if route=='documents':
                from rag.store import VectorStore
                docs=VectorStore().list_documents(); _json(self,200,{'documents':docs,'count':len(docs)}); return
            _json(self,404,{'error':'Endpoint not found.'})
        except Exception as exc: _json(self,500,{'error':'Request failed.','detail':str(exc)})
    def do_DELETE(self):
        try:
            if self._route()!='document': _json(self,404,{'error':'Endpoint not found.'}); return
            data=self._body(); h=str(data.get('document_hash','')).strip()
            if not h: _json(self,400,{'error':'document_hash is required.'}); return
            from rag.store import VectorStore
            _json(self,200,{'success':True,'deleted':VectorStore().delete_document(h)})
        except Exception as exc: _json(self,500,{'error':'Delete failed.','detail':str(exc)})
    def do_POST(self):
        try:
            route=self._route(); data=self._body()
            if route=='chat':
                from rag.pipeline import RAGPipeline
                from rag.store import VectorStore
                q=str(data.get('question','')).strip()
                if not q: _json(self,400,{'error':'Question is required.'}); return
                _json(self,200,RAGPipeline(VectorStore()).answer(q,data.get('history',[]) if isinstance(data.get('history',[]),list) else [])); return
            if route=='url':
                from rag.ingest import ingest_url
                from rag.store import VectorStore
                url=str(data.get('url','')).strip()
                if not url: _json(self,400,{'error':'URL is required.'}); return
                _json(self,200,{'success':True,'chunks':ingest_url(url,VectorStore()),'url':url}); return
            if route=='pdf':
                from rag.config import MAX_PDF_BYTES
                from rag.ingest import ingest_pdf
                from rag.store import VectorStore
                filename=os.path.basename(str(data.get('filename','document.pdf')).strip()) or 'document.pdf'
                if not filename.lower().endswith('.pdf'): _json(self,400,{'error':'Only PDF files are supported.'}); return
                encoded=data.get('content_base64')
                if not encoded: _json(self,400,{'error':'content_base64 is required.'}); return
                if ',' in encoded: encoded=encoded.split(',',1)[1]
                try: pdf_bytes=base64.b64decode(encoded,validate=True)
                except (binascii.Error,ValueError): _json(self,400,{'error':'Invalid base64 PDF data.'}); return
                if not pdf_bytes: _json(self,400,{'error':'PDF is empty.'}); return
                if len(pdf_bytes)>MAX_PDF_BYTES: _json(self,413,{'error':'PDF exceeds configured size limit.'}); return
                path=None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.pdf',delete=False) as f: f.write(pdf_bytes); path=f.name
                    count=ingest_pdf(path,VectorStore(),document_name=filename)
                finally:
                    if path and os.path.exists(path): os.remove(path)
                _json(self,200,{'success':True,'chunks':count,'filename':filename}); return
            _json(self,404,{'error':'Endpoint not found.'})
        except json.JSONDecodeError: _json(self,400,{'error':'Invalid JSON request body.'})
        except ValueError as exc: _json(self,400,{'error':str(exc)})
        except Exception as exc: _json(self,500,{'error':'Request failed.','detail':str(exc)})
