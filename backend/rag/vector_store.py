import json
import math
import os
from pathlib import Path

from .config import VECTOR_FILE


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorStore:
    def __init__(self, path: Path = VECTOR_FILE):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "embedding_model": None, "chunks": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
                raise ValueError("invalid vector store")
            return data
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Vector store is invalid or unreadable: {exc}") from exc

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, self.path)

    def add(self, records: list[dict], embedding_model: str) -> None:
        data = self.load()
        document_names = {r["document"] for r in records}
        data["chunks"] = [r for r in data["chunks"] if r.get("document") not in document_names]
        data["chunks"].extend(records)
        data["embedding_model"] = embedding_model
        self.save(data)

    def delete_document(self, document: str) -> int:
        data = self.load()
        before = len(data["chunks"])
        data["chunks"] = [r for r in data["chunks"] if r.get("document") != document]
        self.save(data)
        return before - len(data["chunks"])

    def documents(self) -> list[dict]:
        data = self.load()
        grouped = {}
        for row in data["chunks"]:
            name = row.get("document", "unknown")
            item = grouped.setdefault(name, {"document": name, "pages": set(), "chunks": 0})
            item["pages"].add(row.get("page"))
            item["chunks"] += 1
        return [
            {"document": x["document"], "pages": len([p for p in x["pages"] if p is not None]), "chunks": x["chunks"]}
            for x in sorted(grouped.values(), key=lambda v: v["document"].lower())
        ]

    def search(self, query_vector: list[float], top_k: int, threshold: float) -> list[dict]:
        rows = self.load()["chunks"]
        scored = []
        for row in rows:
            score = _cosine(query_vector, row.get("vector", []))
            if score >= threshold:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{**row, "score": round(score, 4)} for score, row in scored[:top_k]]
