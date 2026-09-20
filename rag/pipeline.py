import math, re
from .config import CONTEXT_MAX_CHARS, GLOBAL_CHUNKS_PER_DOCUMENT, MEMORY_TURNS, SIMILARITY_THRESHOLD, TOP_K
from .gemini import embed_query, generate_answer
from .store import Chunk, VectorStore

GLOBAL_PATTERNS=(r"\ball\s+(my\s+)?docs?\b",r"\ball\s+(my\s+)?documents?\b",r"\beverything\s+(in|inside)\b",r"\boverview\s+of\s+(all|my|the)\b",r"\bsummar(?:ize|ise)\s+(all|my|the)\b",r"\bcompare\s+(all|my|the)\s+documents?\b")
INVENTORY_PATTERNS=(r"\bwhat\s+(documents|docs)\s+do\s+i\s+have\b",r"\blist\s+(my\s+)?documents?\b",r"\bshow\s+(my\s+)?documents?\b")

def _intent(question, documents):
    q=question.lower().strip()
    if any(re.search(p,q) for p in INVENTORY_PATTERNS): return "inventory",None
    if any(re.search(p,q) for p in GLOBAL_PATTERNS): return "global",None
    for d in documents:
        name=d["document_name"]; stem=re.sub(r"\.[a-z0-9]+$","",name,flags=re.I)
        if name.lower() in q or stem.lower() in q: return "document",d
    return "qa",None

def _cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def _representative(chunks,n):
    if len(chunks)<=n:return chunks
    idx=[round(i*(len(chunks)-1)/(n-1)) for i in range(n)] if n>1 else [0]
    return [chunks[i] for i in sorted(set(idx))]

def _context(matches,max_chars):
    parts=[]; sources=[]; used=0; seen=set()
    for score,chunk in matches:
        m=chunk.metadata; key=(m.get("document_hash"),m.get("chunk_index"))
        if key in seen or not chunk.content.strip(): continue
        block=f"Source: {m.get('document_name') or 'Unknown'}\nPage: {m.get('page_number') or 'N/A'}\nURL: {m.get('source_url') or 'N/A'}\nContent:\n{chunk.content.strip()}"
        if used and used+len(block)>max_chars: break
        parts.append(block); used+=len(block); seen.add(key)
        sources.append({"document_name":m.get("document_name") or "Unknown","page_number":m.get("page_number"),"source_url":m.get("source_url"),"score":round(float(score),3)})
    return "\n\n---\n\n".join(parts),sources

class RAGPipeline:
    def __init__(self,store:VectorStore): self.store=store
    def answer(self,question,history=None):
        question=question.strip()
        if not question: raise ValueError("Question cannot be empty.")
        history=(history or [])[-MEMORY_TURNS:]; docs=self.store.list_documents(); mode,selected=_intent(question,docs)
        if not docs:return {"answer":"No documents are indexed yet. Upload a PDF or add a web source first.","sources":[],"mode":"empty","documents":[]}
        if mode=="inventory":
            lines=[f"{i}. {d['document_name']} — {d['chunk_count']} chunks"+(f", {d['page_count']} pages" if d.get('page_count') else "") for i,d in enumerate(docs,1)]
            return {"answer":"I have these indexed documents:\n\n"+"\n".join(lines),"sources":[],"mode":mode,"documents":docs}
        if mode=="global":
            matches=[]
            for d in docs:
                if d.get("document_hash"):
                    for c in _representative(self.store.get_document_chunks(d["document_hash"]),GLOBAL_CHUNKS_PER_DOCUMENT): matches.append((1.0,c))
            context,sources=_context(matches,CONTEXT_MAX_CHARS)
            if not context:return {"answer":"The document index exists, but no readable content was found.","sources":[],"mode":mode,"documents":docs}
            return {"answer":generate_answer(question,context,history,mode),"sources":sources,"mode":mode,"documents":docs}
        q=embed_query(question)
        if mode=="document" and selected and selected.get("document_hash"):
            chunks=self.store.get_document_chunks(selected["document_hash"])
            matches=sorted(((_cosine(q,c.embedding),c) for c in chunks),key=lambda x:x[0],reverse=True)[:TOP_K]
            matches=[x for x in matches if x[0]>=SIMILARITY_THRESHOLD]
        else: matches=self.store.search(q,TOP_K,SIMILARITY_THRESHOLD)
        context,sources=_context(matches,CONTEXT_MAX_CHARS)
        if not context:return {"answer":"I could not find enough relevant information in the indexed documents.","sources":[],"mode":mode,"documents":docs}
        return {"answer":generate_answer(question,context,history,mode),"sources":sources,"mode":mode,"documents":docs}
