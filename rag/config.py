import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.35"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
URL_TIMEOUT_SECONDS = int(os.getenv("URL_TIMEOUT_SECONDS", "15"))
MAX_URL_BYTES = int(os.getenv("MAX_URL_BYTES", "5000000"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", "20000000"))

DOCUMENT_CATEGORIES = (
    "work",
    "learning",
    "finance",
    "personal",
    "reference",
)
