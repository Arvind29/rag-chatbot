"use client";

import {
  ChangeEvent,
  FormEvent,
  useState,
} from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://rag-chatbot-eight-phi.vercel.app";

type Source = {
  document_name: string;
  page_number: number | null;
  source_url: string | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [documents, setDocuments] = useState<string[]>([]);
  const [url, setUrl] = useState("");

  const [drawerOpen, setDrawerOpen] = useState(false);

  const [chatLoading, setChatLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [urlLoading, setUrlLoading] = useState(false);

  const [error, setError] = useState("");

  function newChat() {
    setMessages([]);
    setSources([]);
    setQuestion("");
    setError("");
    setDrawerOpen(false);
  }

  async function askQuestion(event: FormEvent) {
    event.preventDefault();

    const text = question.trim();

    if (!text || chatLoading) {
      return;
    }

    setQuestion("");
    setError("");

    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: text,
      },
    ]);

    setChatLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: text,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.error ||
            "Unable to process your question."
        );
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            data.answer ||
            "No answer was returned.",
        },
      ]);

      setSources(data.sources || []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setChatLoading(false);
    }
  }

  async function uploadPdf(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (
      file.type !== "application/pdf" &&
      !file.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("Please select a PDF file.");
      return;
    }

    setError("");
    setUploadLoading(true);

    try {
      const buffer = await file.arrayBuffer();
      const bytes = new Uint8Array(buffer);

      let binary = "";
      const chunkSize = 8192;

      for (
        let index = 0;
        index < bytes.length;
        index += chunkSize
      ) {
        binary += String.fromCharCode(
          ...bytes.subarray(
            index,
            Math.min(
              index + chunkSize,
              bytes.length
            )
          )
        );
      }

      const contentBase64 = btoa(binary);

      const response = await fetch(
        `${API_URL}/api/upload-pdf`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename: file.name,
            content_base64: contentBase64,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.error ||
            "PDF upload failed."
        );
      }

      setDocuments((current) => [
        ...current.filter(
          (name) => name !== file.name
        ),
        file.name,
      ]);

      setDrawerOpen(false);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "PDF upload failed."
      );
    } finally {
      setUploadLoading(false);
      event.target.value = "";
    }
  }

  async function addUrl(event: FormEvent) {
    event.preventDefault();

    const targetUrl = url.trim();

    if (!targetUrl || urlLoading) {
      return;
    }

    setError("");
    setUrlLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/add-url`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            url: targetUrl,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.error ||
            "Web page could not be added."
        );
      }

      let label = targetUrl;

      try {
        label = new URL(targetUrl).hostname;
      } catch {
        // Keep the original URL.
      }

      setDocuments((current) => [
        ...current.filter(
          (name) => name !== label
        ),
        label,
      ]);

      setUrl("");
      setDrawerOpen(false);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Web page could not be added."
      );
    } finally {
      setUrlLoading(false);
    }
  }

  return (
    <main className="flex h-[100dvh] w-full flex-col overflow-hidden bg-slate-50 text-slate-900">
      {/* HEADER */}
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          {/* Mobile menu */}
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="flex h-11 w-11 shrink-0 items-center justify-center border border-slate-300 bg-white text-lg text-slate-700 md:hidden"
            aria-label="Open documents"
          >
            ☰
          </button>

          <div className="min-w-0">
            <h1 className="truncate text-xl font-semibold tracking-tight">
              RAG Chatbot
            </h1>

            <p className="hidden text-sm text-slate-500 sm:block">
              Document question answering
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={newChat}
          className="h-11 bg-slate-200 px-4 text-base font-semibold text-slate-900 hover:bg-slate-300 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2"
        >
          <span className="sm:hidden">
            + New
          </span>

          <span className="hidden sm:inline">
            + New Chat
          </span>
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* DESKTOP SIDEBAR */}
        <aside className="hidden w-80 shrink-0 overflow-y-auto border-r border-slate-200 bg-white md:block">
          <div className="p-6">
            <DocumentPanel
              documents={documents}
              url={url}
              setUrl={setUrl}
              uploadPdf={uploadPdf}
              addUrl={addUrl}
              uploadLoading={uploadLoading}
              urlLoading={urlLoading}
            />
          </div>
        </aside>

        {/* MOBILE DRAWER */}
        {drawerOpen && (
          <div className="fixed inset-0 z-50 md:hidden">
            <button
              type="button"
              onClick={() => setDrawerOpen(false)}
              className="absolute inset-0 bg-slate-900/30"
              aria-label="Close documents"
            />

            <aside className="relative h-full w-[88%] max-w-sm overflow-y-auto border-r border-slate-200 bg-white p-5 shadow-lg">
              <div className="mb-6 flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-semibold">
                    Documents
                  </h2>

                  <p className="mt-1 text-sm text-slate-500">
                    Add PDF or web sources
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="flex h-11 w-11 items-center justify-center border border-slate-300 text-xl text-slate-700"
                  aria-label="Close documents"
                >
                  ×
                </button>
              </div>

              <DocumentPanel
                documents={documents}
                url={url}
                setUrl={setUrl}
                uploadPdf={uploadPdf}
                addUrl={addUrl}
                uploadLoading={uploadLoading}
                urlLoading={urlLoading}
              />
            </aside>
          </div>
        )}

        {/* CHAT AREA */}
        <section className="flex min-w-0 flex-1 flex-col">
          {/* MESSAGE AREA */}
          <div className="min-h-0 flex-1 overflow-y-auto">
            <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
              {messages.length === 0 &&
                !chatLoading && (
                  <div className="mx-auto max-w-2xl py-10 text-center sm:py-16">
                    <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                      Ask your documents
                    </h2>

                    <p className="mt-4 text-base leading-7 text-slate-600 sm:text-lg">
                      Upload a PDF or add a web page,
                      then ask questions about the
                      information in your sources.
                    </p>
                  </div>
                )}

              <div className="space-y-6">
                {/* MESSAGES */}
                {messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={
                      message.role === "user"
                        ? "ml-auto w-full max-w-[90%] sm:max-w-[75%]"
                        : "w-full max-w-[95%] sm:max-w-[90%]"
                    }
                  >
                    <div
                      className={
                        message.role === "user"
                          ? "border border-slate-300 bg-slate-200 px-4 py-4 text-base leading-7 text-slate-900 sm:px-5"
                          : "border border-slate-300 bg-white px-4 py-5 text-[18px] leading-8 text-slate-800 sm:px-6"
                      }
                    >
                      <div
                        className={
                          message.role === "user"
                            ? "mb-2 text-xs font-semibold uppercase tracking-wide text-slate-900"
                            : "mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500"
                        }
                      >
                        {message.role === "user"
                          ? "You"
                          : "Answer"}
                      </div>

                      <div className="break-words whitespace-pre-wrap [overflow-wrap:anywhere]">
                        {message.content}
                      </div>
                    </div>
                  </div>
                ))}

                {/* LOADING */}
                {chatLoading && (
                  <div className="w-full max-w-[95%]">
                    <div
                      className="border border-slate-300 bg-white px-4 py-5 text-base text-slate-500"
                      aria-live="polite"
                    >
                      Processing your question...
                    </div>
                  </div>
                )}

                {/* SOURCES */}
                {sources.length > 0 && (
                  <div className="pt-2">
                    <h2 className="text-base font-semibold text-slate-900">
                      Sources
                    </h2>

                    <p className="mt-1 text-sm text-slate-500">
                      Information used for the answer
                    </p>

                    <div className="mt-4 space-y-3">
                      {sources.map(
                        (source, index) => (
                          <div
                            key={`${source.document_name}-${index}`}
                            className="border border-slate-300 bg-white p-4"
                          >
                            <div className="break-words text-base font-medium text-slate-900 [overflow-wrap:anywhere]">
                              {source.document_name}
                            </div>

                            {source.page_number && (
                              <div className="mt-2 text-sm text-slate-500">
                                Page{" "}
                                {source.page_number}
                              </div>
                            )}

                            {source.source_url && (
                              <div className="mt-2 break-all text-sm leading-6 text-slate-700">
                                {source.source_url}
                              </div>
                            )}
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {/* ERROR */}
                {error && (
                  <div
                    className="border border-red-300 bg-red-50 px-4 py-3 text-base leading-7 text-red-800"
                    role="alert"
                  >
                    {error}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* INPUT */}
          <div className="shrink-0 border-t border-slate-200 bg-white px-3 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] sm:px-5 sm:py-4">
            <form
              onSubmit={askQuestion}
              className="mx-auto flex w-full max-w-4xl items-end gap-2"
            >
              <textarea
                value={question}
                onChange={(event) =>
                  setQuestion(event.target.value)
                }
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey
                  ) {
                    event.preventDefault();

                    if (question.trim()) {
                      event.currentTarget.form?.requestSubmit();
                    }
                  }
                }}
                rows={1}
                placeholder="Ask a question..."
                disabled={chatLoading}
                className="min-h-12 min-w-0 flex-1 resize-none overflow-y-auto border border-slate-300 bg-white px-4 py-3 text-base leading-7 text-slate-900 outline-none placeholder:text-slate-400 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 disabled:bg-slate-100"
                aria-label="Ask a question"
              />

              <button
                type="submit"
                disabled={
                  chatLoading ||
                  !question.trim()
                }
                className="h-12 shrink-0 bg-slate-200 px-5 text-base font-semibold text-slate-900 hover:bg-slate-300 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send
              </button>
            </form>

            <p className="mx-auto mt-2 max-w-4xl text-center text-xs text-slate-400">
              Enter to send · Shift + Enter for a new line
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}

function DocumentPanel({
  documents,
  url,
  setUrl,
  uploadPdf,
  addUrl,
  uploadLoading,
  urlLoading,
}: {
  documents: string[];
  url: string;
  setUrl: (value: string) => void;
  uploadPdf: (
    event: ChangeEvent<HTMLInputElement>
  ) => void;
  addUrl: (
    event: FormEvent
  ) => void;
  uploadLoading: boolean;
  urlLoading: boolean;
}) {
  return (
    <div className="space-y-7">
      {/* TITLE */}
      <div>
        <h2 className="text-xl font-semibold">
          Documents
        </h2>

        <p className="mt-2 text-sm leading-6 text-slate-500">
          Add PDFs or web pages to your sources.
        </p>
      </div>

      {/* PDF UPLOAD */}
      <div>
        <label
          htmlFor="pdf-upload"
          className="flex min-h-12 cursor-pointer items-center justify-center border border-slate-300 bg-white px-4 text-base font-semibold text-slate-800 hover:bg-slate-50 focus-within:ring-2 focus-within:ring-emerald-400 focus-within:ring-offset-2"
        >
          {uploadLoading
            ? "Uploading PDF..."
            : "Upload PDF"}
        </label>

        <input
          id="pdf-upload"
          type="file"
          accept=".pdf,application/pdf"
          className="sr-only"
          disabled={uploadLoading}
          onChange={uploadPdf}
        />
      </div>

      {/* WEB PAGE */}
      <form
        onSubmit={addUrl}
        className="space-y-3"
      >
        <label
          htmlFor="web-url"
          className="block text-base font-semibold text-slate-800"
        >
          Web Page URL
        </label>

        <input
          id="web-url"
          type="url"
          value={url}
          onChange={(event) =>
            setUrl(event.target.value)
          }
          placeholder="https://example.com"
          className="min-h-12 w-full border border-slate-300 bg-white px-3 text-base text-slate-900 outline-none placeholder:text-slate-400 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
        />

        <button
          type="submit"
          disabled={urlLoading || !url.trim()}
          className="min-h-12 w-full bg-slate-200 px-4 text-base font-semibold text-slate-900 hover:bg-slate-300 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {urlLoading
            ? "Adding Web Page..."
            : "Add Web Page"}
        </button>
      </form>

      {/* DOCUMENT LIST */}
      <div className="border-t border-slate-200 pt-6">
        <h3 className="text-base font-semibold">
          Added Documents
        </h3>

        <p className="mt-1 text-sm leading-6 text-slate-500">
          Sources currently available
        </p>

        <div className="mt-4 space-y-2">
          {documents.length === 0 ? (
            <div className="border border-dashed border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-500">
              No documents added yet.
            </div>
          ) : (
            documents.map(
              (document, index) => (
                <div
                  key={`${document}-${index}`}
                  className="border border-slate-300 bg-white p-3"
                >
                  <div className="break-words text-base leading-6 text-slate-800 [overflow-wrap:anywhere]">
                    {document}
                  </div>
                </div>
              )
            )
          )}
        </div>
      </div>
    </div>
  );
}