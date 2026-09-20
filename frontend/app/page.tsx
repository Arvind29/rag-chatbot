"use client";

import { DragEvent, FormEvent, useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8005";
const CATEGORIES = ["work", "learning", "finance", "personal", "reference"];
const EXTENSIONS = ".pdf,.docx,.txt,.md,.csv";

type Source = { document_name: string; page_number: number | null; source_url: string | null; score?: number; category?: string };
type Message = { role: "user" | "assistant"; content: string };
type Document = { document_name: string; source_type: string; chunks: number; category?: string; document_type?: string; document_hash?: string };

async function readApiResponse(response: Response) {
  const raw = await response.text();
  let data: any = {};
  try { data = raw ? JSON.parse(raw) : {}; } catch { data = { detail: raw || `HTTP ${response.status}` }; }
  if (!response.ok) throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  return data;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [category, setCategory] = useState("reference");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  async function refreshDocuments() {
    try {
      const response = await fetch(`${API_URL}/api/documents`, { cache: "no-store" });
      const data = await readApiResponse(response);
      setDocuments(data.documents || []);
      setError("");
    } catch (err) { console.error("Document refresh failed", err); setError(err instanceof Error ? err.message : "Cannot connect to local API on port 8005."); }
  }
  useEffect(() => { refreshDocuments(); }, []);

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || loading) return;
    setQuestion(""); setError(""); setLoading(true);
    const nextMessages = [...messages, { role: "user" as const, content: text }];
    setMessages(nextMessages);
    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, history: nextMessages.slice(-7, -1) }),
      });
      const data = await readApiResponse(response);
      setMessages(current => [...current, { role: "assistant", content: data.answer || "No answer returned." }]);
      setSources(data.sources || []);
    } catch (err) { setError(err instanceof Error ? err.message : "Chat failed."); }
    finally { setLoading(false); }
  }

  async function upload(file: File) {
    const allowed = [".pdf", ".docx", ".txt", ".md", ".csv"];
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(extension)) { setError("Supported files: PDF, DOCX, TXT, MD and CSV."); return; }
    setUploading(true); setError(""); setUploadStatus(`Uploading ${file.name}...`);
    try {
      const form = new FormData(); form.append("file", file); form.append("category", category);
      setUploadStatus(`Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`);
      const response = await fetch(`${API_URL}/api/upload`, { method: "POST", body: form });
      const data = await readApiResponse(response);
      setUploadStatus(`✓ Indexed: ${file.name} • ${data.chunks ?? 0} chunks • ${data.category || category}`);
      await refreshDocuments();
    } catch (err) { setUploadStatus(""); setError(err instanceof Error ? err.message : "Upload failed. Check FastAPI on port 8005."); }
    finally { setUploading(false); }
  }

  async function removeDocument(doc: Document) {
    if (!doc.document_hash || !confirm(`Remove ${doc.document_name} from the local knowledge base?`)) return;
    try {
      const response = await fetch(`${API_URL}/api/documents/${encodeURIComponent(doc.document_hash)}`, { method: "DELETE" });
      await readApiResponse(response); await refreshDocuments(); setSources([]);
    } catch (err) { setError(err instanceof Error ? err.message : "Delete failed."); }
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault(); setDragging(false);
    const file = event.dataTransfer.files?.[0]; if (file) upload(file);
  }

  return <main className="flex h-screen flex-col bg-slate-50 text-slate-900">
    <header className="flex h-16 shrink-0 items-center justify-between border-b bg-white px-5">
      <div><h1 className="text-xl font-semibold">Private Knowledge Assistant</h1><p className="text-xs text-slate-500">Ollama + Qdrant • Local only • Grounded RAG</p></div>
      <div className="flex gap-2"><a href="/dashboard" className="border px-4 py-2 text-sm">Dashboard</a><button onClick={() => { setMessages([]); setSources([]); setError(""); }} className="bg-slate-200 px-4 py-2 text-sm font-semibold">New Chat</button></div>
    </header>
    <div className="flex min-h-0 flex-1">
      <aside className="hidden w-80 shrink-0 overflow-y-auto border-r bg-white p-5 md:block">
        <h2 className="font-semibold">Knowledge base</h2>
        <label className="mt-3 block text-xs font-medium text-slate-500">Category</label>
        <select value={category} onChange={e => setCategory(e.target.value)} className="mt-1 w-full border px-3 py-2 text-sm">{CATEGORIES.map(item => <option key={item} value={item}>{item}</option>)}</select>
        <div onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop} className={`mt-3 border-2 border-dashed p-6 text-center ${dragging ? "border-slate-700 bg-slate-100" : "border-slate-300"}`}>
          <div className="text-sm font-medium">{uploading ? "Indexing document..." : "Drag & drop document"}</div><div className="my-2 text-xs text-slate-500">PDF • DOCX • TXT • MD • CSV</div>
          <label className="inline-block cursor-pointer bg-slate-200 px-4 py-2 text-sm font-medium">Choose file<input type="file" accept={EXTENSIONS} className="hidden" disabled={uploading} onChange={e => { const file = e.target.files?.[0]; if (file) upload(file); e.currentTarget.value = ""; }} /></label>
        </div>
        {uploadStatus && <div className="mt-3 border border-slate-300 bg-slate-50 p-3 text-xs font-medium text-slate-700">{uploadStatus}</div>}
        <h2 className="mt-7 font-semibold">Indexed documents ({documents.length})</h2>
        <div className="mt-3 space-y-2">{documents.length ? documents.map(doc => <div key={`${doc.document_hash || doc.document_name}-${doc.category}`} className="border p-3"><div className="truncate text-sm font-medium">{doc.document_name}</div><div className="mt-1 text-xs text-slate-500">{doc.chunks} chunks • {doc.category || "reference"} • {doc.document_type || doc.source_type}</div>{doc.document_hash && <button onClick={() => removeDocument(doc)} className="mt-2 text-xs text-red-600">Remove</button>}</div>) : <p className="text-sm text-slate-400">No documents indexed.</p>}</div>
      </aside>
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto"><div className="mx-auto max-w-4xl space-y-5 p-5 sm:p-8">
          {!messages.length && <div className="py-20 text-center"><h2 className="text-3xl font-semibold">Ask your private knowledge base</h2><p className="mx-auto mt-3 max-w-xl text-slate-500">Upload documents and ask grounded questions, summaries, comparisons, or “summarize all my documents”.</p><div className="mx-auto mt-6 grid max-w-2xl gap-2 sm:grid-cols-2">{["Summarize all my documents", "What are the key facts?", "Find important deadlines", "What information is missing?"] .map(prompt => <button key={prompt} onClick={() => setQuestion(prompt)} className="border bg-white p-3 text-left text-sm hover:bg-slate-50">{prompt}</button>)}</div></div>}
          {messages.map((message, index) => <div key={index} className={message.role === "user" ? "ml-auto max-w-2xl border bg-slate-100 p-4" : "max-w-3xl border bg-white p-5"}><div className="mb-2 text-xs font-semibold uppercase text-slate-400">{message.role === "user" ? "You" : "Assistant"}</div><div className="whitespace-pre-wrap leading-7">{message.content}</div></div>)}
          {loading && <div className="border bg-white p-5 text-slate-500">Searching your local knowledge base...</div>}
          {sources.length > 0 && <div><h3 className="font-semibold">Sources</h3><div className="mt-2 grid gap-2 sm:grid-cols-2">{sources.map((source, index) => <div key={index} className="border bg-white p-3 text-sm"><div className="font-medium">{source.document_name}</div><div className="text-slate-500">{source.category || "reference"}{source.page_number ? ` • Page ${source.page_number}` : ""}</div>{source.score !== undefined && <div className="text-xs text-slate-400">Similarity {source.score}</div>}</div>)}</div></div>}
          {error && <div className="border border-red-300 bg-red-50 p-3 text-sm text-red-800">{error}</div>}
        </div></div>
        <form onSubmit={ask} className="border-t bg-white p-4"><div className="mx-auto flex max-w-4xl gap-2"><textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} rows={1} placeholder="Ask your knowledge base..." className="min-h-12 flex-1 resize-none border px-4 py-3 outline-none" /><button disabled={loading || !question.trim()} className="bg-slate-200 px-5 font-semibold disabled:opacity-40">Send</button></div></form>
      </section>
    </div>
  </main>;
}
