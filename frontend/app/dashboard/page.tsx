"use client";

import { useEffect, useState } from "react";
const API = "http://127.0.0.1:8005";

type Finding = { text: string; document_name?: string; page_number?: number | null; evidence?: string };
type Dashboard = { documents: number; chunks: number; facts: Finding[]; errors: Finding[]; deadlines: Finding[]; events: Finding[] };

export default function Dashboard() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");
  async function load() {
    try { const response = await fetch(`${API}/api/dashboard`, { cache: "no-store" }); const raw = await response.text(); let body: any = {}; try { body = raw ? JSON.parse(raw) : {}; } catch { body = { detail: raw }; } if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`); setData(body); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "Dashboard API is not reachable."); }
  }
  useEffect(() => { load(); }, []);
  if (error) return <main className="min-h-screen p-8 text-red-700">{error}</main>;
  if (!data) return <main className="min-h-screen p-8">Loading dashboard...</main>;
  const cards = [["Documents", data.documents], ["Chunks", data.chunks], ["Facts", data.facts.length], ["Errors", data.errors.length], ["Deadlines", data.deadlines.length], ["Events", data.events.length]];
  return <main className="min-h-screen bg-slate-50 text-slate-900">
    <header className="border-b bg-white px-6 py-5"><div className="mx-auto flex max-w-6xl items-center justify-between"><div><h1 className="text-2xl font-semibold">Local Intelligence</h1><p className="text-sm text-slate-500">Evidence-backed findings from your local knowledge base</p></div><div className="flex gap-2"><button onClick={load} className="border px-4 py-2 text-sm">Refresh</button><a href="/" className="border px-4 py-2 text-sm">Open Chat</a></div></div></header>
    <section className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">{cards.map(([label, value]) => <div key={label} className="border bg-white p-5"><div className="text-sm text-slate-500">{label}</div><div className="mt-2 text-3xl font-semibold">{value}</div></div>)}</div>
      <div className="grid gap-5 md:grid-cols-2"><Panel title="Important Facts" items={data.facts} empty="No supported facts found."/><Panel title="Errors / Conflicts" items={data.errors} empty="No supported errors found."/><Panel title="Deadlines" items={data.deadlines} empty="No supported deadlines found."/><Panel title="Important Events" items={data.events} empty="No supported events found."/></div>
    </section>
  </main>;
}

function Panel({ title, items, empty }: { title: string; items: Finding[]; empty: string }) {
  return <section className="border bg-white p-5"><h2 className="font-semibold">{title}</h2>{items.length ? <div className="mt-4 space-y-4">{items.map((item, i) => <article key={i} className="border-l-2 border-slate-300 pl-3"><div className="text-sm font-medium">{item.text}</div>{item.document_name && <div className="mt-1 text-xs text-slate-500">{item.document_name}{item.page_number ? ` • Page ${item.page_number}` : ""}</div>}{item.evidence && <div className="mt-2 text-xs text-slate-600">Evidence: {item.evidence}</div>}</article>)}</div> : <p className="mt-4 text-sm text-slate-400">{empty}</p>}</section>;
}
