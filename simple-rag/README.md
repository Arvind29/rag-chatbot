# Simple RAG Chatbot

A deliberately small RAG chatbot prototype on the `simple-rag-html` branch.

## Architecture

Browser HTML/CSS/JS → FastAPI → local file-based vector store → embeddings/LLM.

## Recommended backend

Use a simple FastAPI backend with `sentence-transformers` for embeddings and a JSON/NumPy file for vectors. No Supabase, Redis, Kubernetes, or external vector database is required.

Required endpoints:

- `POST /upload` — multipart PDF upload and indexing
- `POST /chat` — JSON `{ "question": "..." }`
- `GET /health` — health check

The frontend currently points to `http://localhost:8000` for safe local testing. Change the `API` constant in `index.html` only when the backend is deployed.
