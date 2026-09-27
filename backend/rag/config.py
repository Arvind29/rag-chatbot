import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DIR = DATA_DIR / "vector_store"
VECTOR_FILE = VECTOR_DIR / "index.json"
FRONTEND_DIR = ROOT_DIR / "frontend"

load_dotenv(BASE_DIR / ".env")
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.38"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "15"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

for directory in (DOCUMENTS_DIR, VECTOR_DIR):
    directory.mkdir(parents=True, exist_ok=True)
