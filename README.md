# PDF RAG Assistant — Gemini

A simple, deployment-ready PDF RAG application built with FastAPI, vanilla HTML/CSS/JavaScript, Gemini embeddings, Gemini generation, and a file-based vector store.

## Architecture

```text
PDF
 ↓
pypdf page extraction
 ↓
page-aware chunking
 ↓
Gemini Embedding (gemini-embedding-001)
 ↓
backend/data/vector_store/index.json
 ↓
question embedding
 ↓
cosine similarity + relevance threshold
 ↓
Top-K document chunks
 ↓
Gemini 3.8 Flash
 ↓
Grounded answer + source pages
```

Google's current Gemini documentation lists `gemini-3.8-flash` as a production-ready stable Flash model and documents `gemini-embedding-001` for semantic search and RAG. citeturn0search0turn4search1

## Project structure

```text
rag-chatbot/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── backend/
│   ├── main.py
│   ├── rag/
│   │   ├── config.py
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── gemini.py
│   ├── data/
│   │   ├── documents/
│   │   └── vector_store/index.json
│   ├── requirements.txt
│   └── .env.example
├── api/index.py
├── scripts/index_documents.py
├── requirements.txt
├── vercel.json
└── README.md
```

## 1. Local Windows setup

From the repository root:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy backend\.env.example backend\.env
```

Edit `backend\.env`:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768
TOP_K=5
MIN_SIMILARITY=0.38
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MAX_PDF_MB=15
```

Start locally:

```cmd
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

Health check:

```text
http://127.0.0.1:8080/api/health
```

## 2. Local RAG test

1. Open the application.
2. Upload a text-based PDF.
3. Wait for the indexing confirmation.
4. Ask a question whose answer is definitely in the PDF.
5. Verify the answer and page source.
6. Ask something that is not in the PDF.
7. The assistant should say it could not find the information instead of inventing an answer.

Example:

```text
What is Power Automate?
```

## 3. File-based vector store

The vector database is simply:

```text
backend/data/vector_store/index.json
```

It contains chunk metadata and vectors. There is no Supabase, Pinecone, Qdrant, Redis, MongoDB, or PostgreSQL dependency.

The API can index a PDF directly in local mode:

```http
POST /api/upload
```

It also exposes:

```http
GET    /api/health
GET    /api/documents
POST   /api/chat
DELETE /api/documents/{document_name}
```

## 4. Build the deployable knowledge base

For Vercel, do not depend on runtime filesystem writes. Vercel Functions have a read-only filesystem with temporary `/tmp` storage, so this project deliberately uses a read-only vector store in deployed mode. citeturn6search0

Put PDFs into:

```text
backend/data/documents/
```

Then run:

```cmd
python scripts\index_documents.py
```

This creates:

```text
backend/data/vector_store/index.json
```

The generated vector file must be committed to GitHub for the Vercel deployment.

Because it is intentionally ignored by `.gitignore`, use:

```cmd
git add -f backend/data/vector_store/index.json
git commit -m "Add indexed RAG knowledge base"
git push origin rag-gemini-production
```

Do not commit the source PDFs unless you intentionally want them public.

## 5. Vercel deployment

Vercel supports FastAPI directly as a Python Function and can detect an application exported from `api/index.py`. citeturn5search0turn5search2

1. Push the branch to GitHub.
2. Import the repository into Vercel.
3. Select branch `rag-gemini-production`.
4. Add this environment variable in Vercel:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

5. Also set, if desired:

```text
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768
TOP_K=5
MIN_SIMILARITY=0.38
```

6. Deploy.
7. Test:

```text
https://YOUR-APP.vercel.app/api/health
```

## 6. Important Vercel behavior

### Local

```text
Upload PDF
   ↓
Index
   ↓
Write vector_store/index.json
   ↓
Chat
```

### Vercel

```text
Committed vector_store/index.json
   ↓
Read-only retrieval
   ↓
Gemini embedding for question
   ↓
Gemini 3.8 Flash
   ↓
Answer
```

Runtime upload/delete is intentionally disabled on Vercel. This avoids pretending that a serverless filesystem is a persistent database.

If you later need arbitrary users to upload PDFs in production, add a persistent storage/vector service behind the existing `VectorStore` abstraction rather than modifying the RAG logic.

## 7. Security

- Gemini key is server-side only.
- `.env` is ignored.
- Frontend never receives the API key.
- PDF filenames are sanitized.
- PDF size is limited.
- Only PDFs are accepted.
- Answers are instructed to stay inside retrieved context.
- Source pages come from stored chunk metadata.
- Raw server exceptions are not returned to users.

## 8. Notes

This implementation targets text-based PDFs. Scanned/image-only PDFs require OCR before indexing.

The file vector store is intentionally simple and suitable for a small personal knowledge base. For a large multi-user production system, replace only the vector-store/storage layer.
