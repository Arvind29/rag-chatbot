const $ = (id) => document.getElementById(id);
const input = $("pdfInput");
const uploadLabel = $("uploadLabel");
const uploadStatus = $("uploadStatus");
const modeBadge = $("modeBadge");
const documents = $("documents");
const messages = $("messages");
const form = $("chatForm");
const question = $("question");
const send = $("send");

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const raw = await response.text();
  let data;
  try { data = JSON.parse(raw); } catch { throw new Error(`Server returned non-JSON (${response.status})`); }
  if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status})`);
  return data;
}

function renderDocuments(list) {
  if (!list.length) {
    documents.textContent = "No indexed documents.";
    return;
  }
  documents.innerHTML = "";
  list.forEach((doc) => {
    const row = document.createElement("div");
    row.className = "doc";
    const name = document.createElement("strong");
    name.textContent = doc.document;
    const meta = document.createElement("span");
    meta.textContent = `${doc.pages} pages • ${doc.chunks} chunks`;
    row.append(name, meta);
    documents.appendChild(row);
  });
}

async function refresh() {
  try {
    const health = await api("/api/health");
    const deployed = health.mode === "vercel-read-only";
    modeBadge.textContent = deployed ? "VERCEL • READ ONLY" : "LOCAL • INDEXING ENABLED";
    modeBadge.style.background = deployed ? "#315a80" : "#236b4d";
    uploadLabel.style.display = deployed ? "none" : "block";
    uploadStatus.textContent = health.gemini
      ? `Gemini ready • ${health.chunks} chunks`
      : "Gemini API key is not configured";
    uploadStatus.className = health.gemini ? "status ok" : "status err";
    const data = await api("/api/documents");
    renderDocuments(data.documents || []);
  } catch (error) {
    modeBadge.textContent = "BACKEND OFFLINE";
    uploadStatus.textContent = error.message;
    uploadStatus.className = "status err";
  }
}

input.addEventListener("change", async () => {
  const file = input.files[0];
  if (!file) return;
  uploadStatus.textContent = "Uploading, extracting and embedding...";
  uploadStatus.className = "status";
  try {
    const body = new FormData();
    body.append("file", file);
    const result = await api("/api/upload", { method: "POST", body });
    uploadStatus.textContent = `Indexed ${result.chunks} chunks from ${result.pages} pages`;
    uploadStatus.className = "status ok";
    addMessage(`Indexed: ${result.document}\n${result.pages} pages • ${result.chunks} chunks`, "bot");
    await refresh();
  } catch (error) {
    uploadStatus.textContent = error.message;
    uploadStatus.className = "status err";
  } finally {
    input.value = "";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const q = question.value.trim();
  if (!q) return;
  question.value = "";
  addMessage(q, "user");
  const pending = addMessage("Searching the document and generating a grounded answer...", "bot");
  send.disabled = true;
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    pending.textContent = result.answer || "No answer returned.";
    if (result.sources?.length) {
      const source = document.createElement("div");
      source.className = "source";
      source.textContent = "Sources: " + result.sources.map(s => `${s.document} — page ${s.page}`).join(" • ");
      pending.appendChild(source);
    }
    messages.scrollTop = messages.scrollHeight;
  } catch (error) {
    pending.textContent = `Error: ${error.message}`;
  } finally {
    send.disabled = false;
    question.focus();
  }
});

$("refresh").addEventListener("click", refresh);
refresh();
