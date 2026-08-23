import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rag.ingest import ingest_pdf, ingest_url
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="RAG chatbot Python prototype")
    parser.add_argument("--pdf", help="Path to a PDF to ingest")
    parser.add_argument("--url", help="Public HTTP(S) URL to ingest")
    parser.add_argument("--question", help="Question to ask after ingestion")
    args = parser.parse_args()

    store = VectorStore()
    pipeline = RAGPipeline(store)

    if args.pdf:
        print(f"[1/3] Ingesting PDF: {args.pdf}")
        count = ingest_pdf(args.pdf, store)
        print(f"      Stored {count} chunks.")

    if args.url:
        print(f"[1/3] Ingesting URL: {args.url}")
        count = ingest_url(args.url, store)
        print(f"      Stored {count} chunks.")

    if not args.question:
        print("[2/3] Ingestion check complete.")
        print("[3/3] No question supplied. Add --question to test retrieval + Gemini.")
        return

    print("[2/3] Running retrieval...")
    result = pipeline.answer(args.question)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")
    for source in result["sources"]:
        print(f"- {source}")

if __name__ == "__main__":
    main()
