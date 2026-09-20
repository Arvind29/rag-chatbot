# Private RAG Assistant

Personal knowledge assistant with document-aware retrieval, global cross-document summarization, document inventory, follow-up context, source grounding, and controlled local development.

## Architecture

Next.js :3005 -> `/api/*` development proxy -> FastAPI :8005 -> Query Router -> Supabase pgvector -> Gemini.

Retrieval modes:
- normal semantic question
- document-specific retrieval
- cross-document/global retrieval
- document inventory

Global requests such as `Summarize all docs` deliberately sample representative chunks from every indexed document instead of asking the vector search for only the five most similar chunks.

## Local run

Backend:
```powershell
cd C:\Users\arvin\local-rag
.\venv\Scripts\Activate.ps1
uvicorn api.server:app --reload --port 8005
```

Frontend:
```powershell
cd frontend
npm install
npm run dev -- --port 3005
```

The frontend development proxy points `/api/*` to `http://127.0.0.1:8005`.

## Configuration

Copy `.env.example` to `.env` and configure Gemini and Supabase credentials. RAG tuning variables control top-k retrieval, global sampling, context size, memory turns, and chunking.

## Safety boundary

The assistant is currently knowledge-only. Computer-changing tools and automations are intentionally outside the LLM control path until an explicit approval/audit layer is added.
