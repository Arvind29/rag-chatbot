import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rag.chunker import make_chunks
from backend.rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR, EMBEDDING_MODEL
from backend.rag.document_loader import extract_pages
from backend.rag.embeddings import embed_texts
from backend.rag.vector_store import VectorStore


def main() -> None:
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {DOCUMENTS_DIR}")

    store = VectorStore()
    total = 0
    for pdf in pdfs:
        pages = extract_pages(pdf)
        chunks = make_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP, pdf.name)
        if not chunks:
            print(f"SKIP {pdf.name}: no extractable text")
            continue
        vectors = []
        for start in range(0, len(chunks), 32):
            vectors.extend(embed_texts([x["text"] for x in chunks[start:start + 32]]))
        for row, vector in zip(chunks, vectors):
            row["vector"] = vector
        store.add(chunks, EMBEDDING_MODEL)
        total += len(chunks)
        print(f"INDEXED {pdf.name}: {len(pages)} pages, {len(chunks)} chunks")

    print(f"DONE: {total} chunks in {store.path}")


if __name__ == "__main__":
    main()
