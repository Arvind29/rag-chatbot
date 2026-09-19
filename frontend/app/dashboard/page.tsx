"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Dashboard = {
  documents: number;
  chunks: number;
  facts: string[];
  errors: string[];
  deadlines: string[];
  events: string[];
};

export default function Dashboard() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/dashboard`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError("Dashboard API is not reachable."));
  }, []);

  if (error) return <main className="min-h-screen p-8 text-red-700">{error}</main>;
  if (!data) return <main className="min-h-screen p-8">Loading dashboard...</main>;

  const cards = [
    ["Documents", data.documents],
    ["Chunks", data.chunks],
    ["Facts", data.facts.length],
    ["Errors", data.errors.length],
    ["Deadlines", data.deadlines.length],
    ["Events", data.events.length],
  ];

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white px-6 py-5">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div><h1 className="text-2xl font-semibold">Local Intelligence</h1><p className="text-sm text-slate-500">Facts, errors, deadlines and important events from your documents</p></div>
          <a href="/" className="border px-4 py-2 text-sm">Open Chat</a>
        </div>
      </header>
      <section className="mx-auto max-w-6xl space-y-6 p-6">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          {cards.map(([label, value]) => <div key={label} className="border bg-white p-5"><div className="text-sm text-slate-500">{label}</div><div className="mt-2 text-3xl font-semibold">{value}</div></div>)}
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          <Panel title="Important Facts" items={data.facts} empty="No facts extracted yet." />
          <Panel title="Errors / Conflicts" items={data.errors} empty="No errors detected yet." />
          <Panel title="Deadlines" items={data.deadlines} empty="No deadlines detected yet." />
          <Panel title="Important Events" items={data.events} empty="No events detected yet." />
        </div>
        <div className="border bg-white p-5 text-sm text-slate-600">Next stage: predefined Ollama prompts will analyze the indexed chunks and populate these four panels automatically.</div>
      </section>
    </main>
  );
}

function Panel({ title, items, empty }: { title: string; items: string[]; empty: string }) {
  return <section className="border bg-white p-5"><h2 className="font-semibold">{title}</h2>{items.length ? <ul className="mt-4 space-y-3">{items.map((x, i) => <li key={i} className="border-l-2 border-slate-300 pl-3 text-sm">{x}</li>)}</ul> : <p className="mt-4 text-sm text-slate-400">{empty}</p>}</section>;
}
