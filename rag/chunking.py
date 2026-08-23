from .config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary

        chunks.append(text[start:end].strip())

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return [c for c in chunks if c]
