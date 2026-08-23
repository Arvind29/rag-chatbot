import os

def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.7-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.35"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "3500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "500"))
URL_TIMEOUT_SECONDS = int(os.getenv("URL_TIMEOUT_SECONDS", "15"))
MAX_URL_BYTES = int(os.getenv("MAX_URL_BYTES", "5000000"))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", "20000000"))
