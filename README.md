# RAG Chatbot — Python Validation Prototype

This repository is Phase 1 of a production-oriented RAG chatbot.

It validates the core pipeline before the Next.js/Tailwind frontend is built:

PDF/URL → extraction → cleaning → chunking → Gemini embeddings → vector retrieval → Gemini answer.

## Current scope

- PDF text extraction
- Public webpage extraction
- Gemini embeddings
- In-memory cosine-similarity retrieval
- Similarity threshold
- Grounded Gemini answer generation
- Basic SSRF protection for URL ingestion

## Important

The vector store is intentionally in-memory for validation. It will be replaced by a persistent hosted vector database before production deployment.

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key.

## Test Gemini

```bash
python app.py --question "Hello"
```

For RAG testing, ingest a PDF:

```bash
python app.py --pdf data/sample.pdf --question "What is the main topic of this document?"
```

Or a public webpage:

```bash
python app.py --url "https://example.com" --question "What is this page about?"
```

## Phase 2

After this prototype passes:

1. Move the backend into the deployable application architecture.
2. Add persistent vector storage.
3. Build the Next.js + TypeScript + Tailwind frontend.
4. Add the mobile-first responsive chat UI.
5. Push to GitHub.
6. Deploy/check on Vercel.
7. Run functional and responsive tests.
