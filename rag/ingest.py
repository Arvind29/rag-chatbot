import hashlib,ipaddress,socket,uuid
from pathlib import Path
from urllib.parse import urlparse
import requests,trafilatura
from pypdf import PdfReader
from .chunking import chunk_text
from .config import MAX_PDF_BYTES,MAX_URL_BYTES,URL_TIMEOUT_SECONDS
from .gemini import embed_texts
from .store import Chunk,VectorStore

def _validate_public_url(url):
    p=urlparse(url)
    if p.scheme not in {'http','https'} or not p.hostname: raise ValueError('Only public HTTP(S) URLs are allowed.')
    host=p.hostname.lower()
    if host in {'localhost','localhost.localdomain'}: raise ValueError('Localhost URLs are blocked.')
    try: addresses=socket.getaddrinfo(host,None)
    except socket.gaierror as exc: raise ValueError('Unable to resolve URL hostname.') from exc
    blocked=[ipaddress.ip_network(x) for x in ['127.0.0.0/8','10.0.0.0/8','172.16.0.0/12','192.168.0.0/16','169.254.0.0/16','::1/128','fc00::/7','fe80::/10']]
    for entry in addresses:
        ip=ipaddress.ip_address(entry[4][0])
        if any(ip in net for net in blocked): raise ValueError('Private/internal network addresses are blocked.')
    return url

def _hash_text(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()
def _hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def _build_records(items):
    items=[x for x in items if x[0].strip()]
    if not items: raise ValueError('No extractable text found.')
    records=[]; idx=0
    for start in range(0,len(items),32):
        batch=items[start:start+32]; embeddings=embed_texts([x[0] for x in batch])
        for (text,meta),embedding in zip(batch,embeddings):
            records.append(Chunk(id=str(uuid.uuid4()),content=text,embedding=embedding,metadata={**meta,'chunk_index':idx})); idx+=1
    return records

def ingest_pdf(path,store,document_name=None):
    file=Path(path)
    if not file.is_file(): raise FileNotFoundError(path)
    if file.stat().st_size>MAX_PDF_BYTES: raise ValueError('PDF exceeds configured size limit.')
    if file.suffix.lower()!='.pdf': raise ValueError('Only PDF files are supported.')
    name=document_name or file.name; doc_hash=_hash_file(file); items=[]
    for page_number,page in enumerate(PdfReader(str(file)).pages,1):
        text=(page.extract_text() or '').strip()
        for chunk in chunk_text(text): items.append((chunk,{'document_name':name,'source_type':'pdf','document_hash':doc_hash,'page_number':page_number}))
    records=_build_records(items); store.add(records); return len(records)

def ingest_url(url,store):
    url=_validate_public_url(url); response=requests.get(url,timeout=URL_TIMEOUT_SECONDS,headers={'User-Agent':'Private-RAG-Assistant/2.0'},stream=True); response.raise_for_status()
    length=response.headers.get('Content-Length')
    if length and int(length)>MAX_URL_BYTES: raise ValueError('Web response exceeds configured size limit.')
    raw=response.content
    if len(raw)>MAX_URL_BYTES: raise ValueError('Web response exceeds configured size limit.')
    text=trafilatura.extract(raw,include_comments=False,include_tables=True)
    if not text: raise ValueError('Could not extract readable webpage content.')
    meta=trafilatura.extract_metadata(raw); title=meta.title if meta else url; doc_hash=_hash_text(url)
    records=_build_records([(chunk,{'document_name':title or url,'source_type':'web','source_url':url,'document_hash':doc_hash}) for chunk in chunk_text(text)])
    store.add(records); return len(records)
