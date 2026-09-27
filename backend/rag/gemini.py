from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, GEMINI_MODEL

_client = None

SYSTEM_INSTRUCTION = """You are a document-grounded assistant.
Answer ONLY from the supplied document context.
Do not use outside knowledge. Do not guess or invent facts.
If the context does not contain the answer, say exactly: I couldn't find that information in the uploaded document.
Give a concise direct answer. Use bullets when useful.
Do not claim a fact is in the document unless the supplied context supports it.
"""


def client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def answer_question(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[SOURCE {i}] {row['document']} | page {row['page']} | relevance {row['score']}\n{row['text']}"
        for i, row in enumerate(chunks, start=1)
    )
    prompt = f"""{SYSTEM_INSTRUCTION}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    response = client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=1200,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty answer")
    return text
