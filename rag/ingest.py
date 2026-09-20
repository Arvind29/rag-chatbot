import hashlib
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse
import uuid

import requests
import trafilatura
from pypdf import PdfReader
from docx import Document

from .chunking import chunk_text
from .config import MAX_FILE_BYTES, MAX_URL_BYTES, URL_TIMEOUT_SECONDS
from .ollama import embed_texts
from .store import Chunk, VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed.")
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are blocked.")
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError("Unable to resolve URL hostname.") from exc
    blocked_ranges = [
        ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"), ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"), ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"), ipaddress.ip_network("fe80::/10"),
    ]
    for entry in addresses:
        ip = ipaddress.ip_address(entry[4][0])
        if any(ip in network for network in blocked_ranges):
            raise ValueError("Private/internal network addresses are blocked.")
    return url


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_file(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _extract_file(file_path: Path) -> list[tuple[str, int | None]]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return [(page.extract_text() or "", page_number) for page_number, page in enumerate(reader.pages, 1)]
    if suffix == ".docx":
        document = Document(str(file_path))
        return [("\n".join(p.text for p in document.paragraphs), None)]
    if suffix in {".txt", ".md", ".csv"}:
        return [(file_path.read_text(encoding="utf-8", errors="replace"), None)]
    raise ValueError(f"Unsupported file type: {suffix}")


def _make_chunks(text: str, metadata: dict, page_number: int | None = None) -> list[Chunk]:
    chunks = chunk_text(text)
    if not chunks:
        return []
    embeddings = embed_texts(chunks)
    return [
        Chunk(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata={**metadata, "chunk_index": index, "page_number": page_number},
        )
        for index, (content, embedding) in enumerate(zip(chunks, embeddings))
    ]


def ingest_file(path: str, store: VectorStore, category: str = "reference", document_name: str | None = None) -> int:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(path)
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if file_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds configured size limit.")

    document_hash = _hash_file(file_path)
    display_name = document_name or file_path.name
    base_metadata = {
        "document_name": display_name,
        "document_type": file_path.suffix.lower().lstrip("."),
        "source_type": "upload",
        "category": category,
        "document_hash": document_hash,
    }

    records: list[Chunk] = []
    for text, page_number in _extract_file(file_path):
        if text.strip():
            records.extend(_make_chunks(text, base_metadata, page_number))
    if not records:
        raise ValueError("No extractable text found.")
    store.add(records)
    return len(records)


def ingest_pdf(path: str, store: VectorStore, document_name: str | None = None, category: str = "reference") -> int:
    return ingest_file(path, store, category=category, document_name=document_name)


def ingest_url(url: str, store: VectorStore, category: str = "reference") -> int:
    url = _validate_public_url(url)
    response = requests.get(url, timeout=URL_TIMEOUT_SECONDS, headers={"User-Agent": "Local-RAG/1.0"}, stream=True)
    response.raise_for_status()
    raw = response.content
    if len(raw) > MAX_URL_BYTES:
        raise ValueError("Web response exceeds configured size limit.")
    downloaded = trafilatura.extract(raw, include_comments=False, include_tables=True)
    if not downloaded:
        raise ValueError("Could not extract readable webpage content.")
    metadata = trafilatura.extract_metadata(raw)
    title = metadata.title if metadata else url
    records = _make_chunks(downloaded, {
        "document_name": title or url,
        "document_type": "web",
        "source_type": "web",
        "source_url": url,
        "category": category,
        "document_hash": _hash_text(url),
    })
    if not records:
        raise ValueError("No extractable text found.")
    store.add(records)
    return len(records)
