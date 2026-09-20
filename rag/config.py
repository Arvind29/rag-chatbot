import os
from dotenv import load_dotenv

load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.7-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
GLOBAL_CHUNKS_PER_DOCUMENT = int(os.getenv("RAG_GLOBAL_CHUNKS_PER_DOCUMENT", "8"))
DOCUMENT_MAX_CHUNKS = int(os.getenv("RAG_DOCUMENT_MAX_CHUNKS", "500"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.30"))
CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "30000"))
MEMORY_TURNS = int(os.getenv("RAG_MEMORY_TURNS", "6"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "2200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "300"))
URL_TIMEOUT_SECONDS = int(os.getenv("URL_TIMEOUT_SECONDS", "15"))
MAX_URL_BYTES = int(os.getenv("MAX_URL_BYTES", "5000000"))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", "20000000"))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")
