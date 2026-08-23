import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse
import uuid

import requests
import trafilatura
from pypdf import PdfReader

from .chunking import chunk_text
from .config import MAX_PDF_BYTES, MAX_URL_BYTES, URL_TIMEOUT_SECONDS
from .gemini import embed_texts
from .store import Chunk, VectorStore

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
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
        ipaddress.ip_network("fe80::/10"),
    ]

    for entry in addresses:
        ip = ipaddress.ip_address(entry[4][0])
        if any(ip in network for network in blocked_ranges):
            raise ValueError("Private/internal network addresses are blocked.")

    return url

def _make_chunks(text: str, metadata: dict, store: VectorStore) -> int:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No extractable text found.")

    embeddings = embed_texts(chunks)
    records = [
        Chunk(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata={**metadata, "chunk_index": index},
        )
        for index, (content, embedding) in enumerate(zip(chunks, embeddings))
    ]
    store.add(records)
    return len(records)

def ingest_pdf(path: str, store: VectorStore) -> int:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(path)

    if file_path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds configured size limit.")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(file_path))
    all_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            all_text.append(f"[Page {page_number}]\n{text}")

    return _make_chunks(
        "\n\n".join(all_text),
        {"document_name": file_path.name, "source_type": "pdf"},
        store,
    )

def ingest_url(url: str, store: VectorStore) -> int:
    url = _validate_public_url(url)

    response = requests.get(
        url,
        timeout=URL_TIMEOUT_SECONDS,
        headers={"User-Agent": "RAG-Prototype/1.0"},
        stream=True,
    )
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_URL_BYTES:
        raise ValueError("Web response exceeds configured size limit.")

    raw = response.content
    if len(raw) > MAX_URL_BYTES:
        raise ValueError("Web response exceeds configured size limit.")

    downloaded = trafilatura.extract(raw, include_comments=False, include_tables=True)
    if not downloaded:
        raise ValueError("Could not extract readable webpage content.")

    title = trafilatura.extract_metadata(raw).title if trafilatura.extract_metadata(raw) else url

    return _make_chunks(
        downloaded,
        {"document_name": title or url, "source_type": "web", "source_url": url},
        store,
    )
