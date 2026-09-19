"use client";

import { DragEvent, FormEvent, useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Source = { document_name: string; page_number: number | null; source_url: string | null; score?: number };
type Message = { role: "user" | "assistant"; content: string };
type Document = { document_name: string; source_type: string; chunks: number };

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  async function refreshDocuments() {
    try {
      const response = await fetch(`${API_URL}/api/documents`);
      if (response.ok) setDocuments((await response.json()).documents || []);
    } catch {}
  }

  useEffect(() => { refreshDocuments(); }, []);

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || loading) return;
    setQuestion(""); setError(""); setLoading(true);
    setMessages(current => [...current, { role: "user", content: text }]);
    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Chat failed.");
      setMessages(current => [...current, { role: "assistant", content: data.answer || "No answer returned." }]);
      setSources(data.sources || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed.");
    } finally { setLoading(false); }
  }

  async function upload(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) { setError("Only PDF files are supported."); return; }
    setUploading(true); setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_URL}/api/upload-pdf`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Upload failed.");
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally { setUploading(false); }
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault(); setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) upload(file);
  }

  return (
    <main className="flex h-screen flex-col bg-slate-50 text-slate-900">
      <header className="flex h-16 shrink-0 items-center justify-between border-b bg-white px-5">
        <div><h1 className="text-xl font-semibold">Local RAG Assistant</h1><p className="text-xs text-slate-500">Ollama + Qdrant • Local only</p></div>
        <div className="flex gap-2"><a href="/dashboard" className="border px-4 py-2 text-sm">Dashboard</a><button onClick={() => { setMessages([]); setSources([]); setError(""); }} className="bg-slate-200 px-4 py-2 text-sm font-semibold">New Chat</button></div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-80 shrink-0 overflow-y-auto border-r bg-white p-5 md:block">
          <UploadPanel uploading={uploading} dragging={dragging} setDragging={setDragging} onDrop={drop} upload={upload} />
          <h2 className="mt-7 font-semibold">Indexed documents</h2>
          <div className="mt-3 space-y-2">
            {documents.length ? documents.map(doc => <div key={doc.document_name} className="border p-3"><div className="truncate text-sm font-medium">{doc.document_name}</div><div className="mt-1 text-xs text-slate-500">{doc.chunks} chunks • {doc.source_type}</div></div>) : <p className="text-sm text-slate-400">No documents indexed.</p>}
          </div>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto"><div className="mx-auto max-w-4xl space-y-5 p-5 sm:p-8">
            {!messages.length && <div className="py-20 text-center"><h2 className="text-3xl font-semibold">Ask your documents</h2><p className="mx-auto mt-3 max-w-xl text-slate-500">Upload PDFs, index them locally, and ask grounded questions.</p></div>}
            {messages.map((message, index) => <div key={index} className={message.role === "user" ? "ml-auto max-w-2xl border bg-slate-100 p-4" : "max-w-3xl border bg-white p-5"}><div className="mb-2 text-xs font-semibold uppercase text-slate-400">{message.role === "user" ? "You" : "Assistant"}</div><div className="whitespace-pre-wrap leading-7">{message.content}</div></div>)}
            {loading && <div className="border bg-white p-5 text-slate-500">Searching Qdrant and asking Ollama...</div>}
            {sources.length > 0 && <div><h3 className="font-semibold">Sources</h3><div className="mt-2 grid gap-2 sm:grid-cols-2">{sources.map((source, index) => <div key={index} className="border bg-white p-3 text-sm"><div className="font-medium">{source.document_name}</div>{source.page_number && <div className="text-slate-500">Page {source.page_number}</div>}{source.score !== undefined && <div className="text-xs text-slate-400">Similarity {source.score}</div>}</div>)}</div></div>}
            {error && <div className="border border-red-300 bg-red-50 p-3 text-sm text-red-800">{error}</div>}
          </div></div>

          <form onSubmit={ask} className="border-t bg-white p-4"><div className="mx-auto flex max-w-4xl gap-2"><textarea value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} rows={1} placeholder="Ask a question..." className="min-h-12 flex-1 resize-none border px-4 py-3 outline-none" /><button disabled={loading || !question.trim()} className="bg-slate-200 px-5 font-semibold disabled:opacity-40">Send</button></div></form>
        </section>
      </div>
    </main>
  );
}

function UploadPanel({ uploading, dragging, setDragging, onDrop, upload }: { uploading: boolean; dragging: boolean; setDragging: (value: boolean) => void; onDrop: (event: DragEvent<HTMLDivElement>) => void; upload: (file: File) => void }) {
  return <div><h2 className="font-semibold">Add documents</h2><div onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop} className={`mt-3 border-2 border-dashed p-6 text-center ${dragging ? "border-slate-700 bg-slate-100" : "border-slate-300"}`}><div className="text-sm font-medium">{uploading ? "Indexing PDF..." : "Drag & drop PDF"}</div><div className="my-2 text-xs text-slate-500">or</div><label className="inline-block cursor-pointer bg-slate-200 px-4 py-2 text-sm font-medium">Choose PDF<input type="file" accept="application/pdf" className="hidden" disabled={uploading} onChange={event => { const file = event.target.files?.[0]; if (file) upload(file); event.currentTarget.value = ""; }} /></label></div></div>;
}
