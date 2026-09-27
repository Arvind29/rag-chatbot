import re


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_page(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + int(size * 0.65):
                end = boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) >= 40:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def make_chunks(pages: list[dict], size: int, overlap: int, document: str) -> list[dict]:
    chunks = []
    for page in pages:
        for index, text in enumerate(split_page(page["text"], size, overlap), start=1):
            chunks.append({
                "id": f"{document}:{page['page']}:{index}",
                "document": document,
                "page": page["page"],
                "text": text,
            })
    return chunks
