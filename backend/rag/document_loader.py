from pathlib import Path

from pypdf import PdfReader


def extract_pages(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ").strip()
        pages.append({"page": number, "text": text})
    return pages
