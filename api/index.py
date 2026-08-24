from http.server import BaseHTTPRequestHandler
import base64
import binascii
import json
import os
import tempfile
from urllib.parse import urlparse, parse_qs


class handler(BaseHTTPRequestHandler):

    def _send_json(self, status_code, data):
        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header(
            "Content-Length",
            str(len(body))
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(
            200,
            {"status": "ok"}
        )

    def _get_route(self):
        query = parse_qs(
            urlparse(self.path).query
        )

        return query.get(
            "route",
            [""]
        )[0]

    def do_GET(self):
        route = self._get_route()

        if route == "health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "rag-chatbot"
                }
            )
            return

        self._send_json(
            404,
            {
                "error": "Endpoint not found."
            }
        )

    def do_POST(self):
        try:
            route = self._get_route()

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            if content_length <= 0:
                self._send_json(
                    400,
                    {
                        "error": "Request body is required."
                    }
                )
                return

            raw_body = self.rfile.read(
                content_length
            )

            data = json.loads(
                raw_body.decode("utf-8")
            )

            # ==========================
            # CHAT
            # ==========================

            if route == "chat":
                from rag.pipeline import RAGPipeline
                from rag.store import VectorStore

                question = str(
                    data.get(
                        "question",
                        ""
                    )
                ).strip()

                if not question:
                    self._send_json(
                        400,
                        {
                            "error": "Question is required."
                        }
                    )
                    return

                store = VectorStore()

                result = RAGPipeline(
                    store
                ).answer(question)

                self._send_json(
                    200,
                    result
                )
                return

            # ==========================
            # URL INGESTION
            # ==========================

            if route == "url":
                from rag.ingest import ingest_url
                from rag.store import VectorStore

                url = str(
                    data.get(
                        "url",
                        ""
                    )
                ).strip()

                if not url:
                    self._send_json(
                        400,
                        {
                            "error": "URL is required."
                        }
                    )
                    return

                store = VectorStore()

                count = ingest_url(
                    url,
                    store
                )

                self._send_json(
                    200,
                    {
                        "success": True,
                        "message": (
                            "Web page indexed successfully."
                        ),
                        "chunks": count,
                        "url": url
                    }
                )
                return

            # ==========================
            # PDF INGESTION
            # ==========================

            if route == "pdf":
                from rag.config import MAX_PDF_BYTES
                from rag.ingest import ingest_pdf
                from rag.store import VectorStore

                filename = str(
                    data.get(
                        "filename",
                        "document.pdf"
                    )
                ).strip()

                if not filename.lower().endswith(
                    ".pdf"
                ):
                    self._send_json(
                        400,
                        {
                            "error": (
                                "Only PDF files are supported."
                            )
                        }
                    )
                    return

                encoded = data.get(
                    "content_base64"
                )

                if not encoded:
                    self._send_json(
                        400,
                        {
                            "error": (
                                "content_base64 is required."
                            )
                        }
                    )
                    return

                # Accept an optional data URL prefix.
                if "," in encoded:
                    encoded = encoded.split(
                        ",",
                        1
                    )[1]

                try:
                    pdf_bytes = base64.b64decode(
                        encoded,
                        validate=True
                    )
                except (binascii.Error, ValueError):
                    self._send_json(
                        400,
                        {
                            "error": (
                                "Invalid base64 PDF data."
                            )
                        }
                    )
                    return

                if not pdf_bytes:
                    self._send_json(
                        400,
                        {
                            "error": "PDF is empty."
                        }
                    )
                    return

                if len(pdf_bytes) > MAX_PDF_BYTES:
                    self._send_json(
                        413,
                        {
                            "error": (
                                "PDF exceeds configured size limit."
                            )
                        }
                    )
                    return

                safe_name = os.path.basename(
                    filename
                )

                if not safe_name:
                    safe_name = "document.pdf"

                store = VectorStore()

                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False
                    ) as temp_file:
                        temp_file.write(
                            pdf_bytes
                        )
                        temp_path = temp_file.name

                    # ingest_pdf uses the local filename
                    # as document_name.
                    count = ingest_pdf(
                        temp_path,
                        store
                    )

                finally:
                    if temp_path and os.path.exists(
                        temp_path
                    ):
                        os.remove(temp_path)

                self._send_json(
                    200,
                    {
                        "success": True,
                        "message": (
                            "PDF indexed successfully."
                        ),
                        "chunks": count,
                        "filename": safe_name
                    }
                )
                return

            self._send_json(
                404,
                {
                    "error": "Endpoint not found."
                }
            )

        except json.JSONDecodeError:
            self._send_json(
                400,
                {
                    "error": "Invalid JSON."
                }
            )

        except Exception as exc:
            self._send_json(
                500,
                {
                    "error": "Request failed.",
                    "detail": str(exc)
                }
            )