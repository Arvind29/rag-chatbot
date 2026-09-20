# Local RAG Assistant

A local-first document RAG assistant using **Next.js + FastAPI + Ollama + Qdrant Local**.

No Gemini API, cloud vector database, or Vercel deployment is required for the local application.

## Architecture

```text
Browser :3005
    |
    | Next.js /api proxy
    v
FastAPI :8005
    |
    +--> Ollama :11434
    |      +--> llama3.2:3b       (chat)
    |      +--> nomic-embed-text  (embeddings)
    |
    +--> Qdrant Local
          ./data/qdrant
```

The application supports local ingestion and grounded questions over:

- PDF
- DOCX
- TXT
- Markdown
- CSV
- Public URLs

Documents are assigned a category (`work`, `learning`, `finance`, `personal`, `reference`) and the category is stored in Qdrant metadata and used for filtered retrieval.

## Windows setup

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Make sure Ollama is installed and the required models exist:

```powershell
ollama list
```

Expected models:

```text
llama3.2:3b
nomic-embed-text:latest
```

## Start the backend

Use **one and only one** FastAPI process because Qdrant Local locks `./data/qdrant`.

```powershell
.\venv\Scripts\Activate.ps1
uvicorn api.server:app --reload --port 8005
```

Health check:

```powershell
curl.exe http://127.0.0.1:8005/api/health
```

Expected:

```json
{"status":"ok","service":"local-rag"}
```

## Start the frontend

In a second PowerShell:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on:

```text
http://localhost:3005
```

The Next.js proxy forwards `/api/*` to FastAPI on `127.0.0.1:8005`, so the browser does not need direct cross-origin access to FastAPI.

## Important Qdrant Local rule

Do not start a second Uvicorn process against the same `./data/qdrant` directory. If an old Python/Uvicorn process is running, stop it before starting the backend.

To find Python processes:

```powershell
Get-Process python -ErrorAction SilentlyContinue
```

To stop an unwanted process:

```powershell
Stop-Process -Id <PID> -Force
```

Do **not** delete `data/qdrant` just to fix a lock unless you intentionally want to remove the indexed data.

## Application flow

```text
Upload document
   -> extract text
   -> normalize
   -> chunk
   -> Ollama embedding
   -> Qdrant Local
   -> category + source metadata

Question
   -> Ollama embedding
   -> category-aware Qdrant retrieval
   -> similarity threshold
   -> grounded context
   -> Ollama answer
   -> source metadata returned to UI
```

The UI shows upload/indexing status, indexed documents, categories, chat responses, and retrieved sources.

## Local-only principle

The application is designed for local use. Ollama and Qdrant Local run on the workstation; uploaded documents are processed locally by the application. URL ingestion is the only feature that intentionally retrieves an external webpage.
