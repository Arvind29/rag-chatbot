from http.server import BaseHTTPRequestHandler
import json

from rag.pipeline import RAGPipeline
from rag.store import VectorStore


class handler(BaseHTTPRequestHandler):

    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json"
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
            "POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_POST(self):

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            raw_body = self.rfile.read(
                content_length
            )

            data = json.loads(
                raw_body.decode("utf-8")
            )

            question = data.get(
                "question",
                ""
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
            pipeline = RAGPipeline(store)

            result = pipeline.answer(
                question
            )

            self._send_json(
                200,
                result
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
                    "error": str(exc)
                }
            )