from google import genai
from google.genai import types
from .config import GEMINI_CHAT_MODEL, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIM, required_env

_client = None
SYSTEM_PROMPT = """You are a private, grounded knowledge assistant.
Use the supplied source context as the authority for document facts.
Never invent facts, document names, page numbers, quotations, or citations.
For a specific document, prioritize that document. For global questions, synthesize across supplied documents.
Use conversation history only to resolve follow-ups; it is not source evidence.
If sources genuinely lack the answer, say so clearly. Prefer a direct answer first, then concise structure.
"""

def client():
    global _client
    if _client is None: _client = genai.Client(api_key=required_env("GEMINI_API_KEY"))
    return _client

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts: return []
    result = client().models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=texts, config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM, task_type="RETRIEVAL_DOCUMENT"))
    return [item.values for item in result.embeddings]

def embed_query(text: str) -> list[float]:
    result = client().models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=text, config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM, task_type="RETRIEVAL_QUERY"))
    return result.embeddings[0].values

def generate_answer(question: str, context: str, history: list[dict] | None = None, mode: str = "qa") -> str:
    history = history or []
    history_text = "\n".join(f"{x.get('role','user').upper()}: {str(x.get('content',''))[:2500]}" for x in history[-6:] if x.get('content'))
    mode_hint = {"global":"Synthesize across the supplied documents.", "document":"Stay focused on the identified document.", "inventory":"Describe the indexed document set directly.", "qa":"Answer the grounded question."}.get(mode, "Answer the grounded question.")
    prompt = f"{SYSTEM_PROMPT}\n\nMODE: {mode_hint}\n\nRECENT CONVERSATION:\n{history_text or '(none)'}\n\nCURRENT QUESTION:\n{question}\n\nSOURCE CONTEXT:\n{context}\n\nReturn only the answer for the user."
    response = client().models.generate_content(model=GEMINI_CHAT_MODEL, contents=prompt)
    return response.text or "I could not generate an answer from the available sources."
