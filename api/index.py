from http.server import BaseHTTPRequestHandler
import json
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
        parsed = urlparse(self.path)

        query = parse_qs(
            parsed.query
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

                pipeline = RAGPipeline(
                    store
                )

                result = pipeline.answer(
                    question
                )

                self._send_json(
                    200,
                    result
                )

                return

            # ==========================
            # ADD URL
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
            # UNKNOWN ROUTE
            # ==========================

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