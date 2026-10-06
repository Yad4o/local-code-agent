import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class StubServer:
    """A real local HTTP server that imitates Ollama / OpenAI-style endpoints."""

    def __init__(self):
        self.requests = []          # (method, path, parsed_json_or_None)
        self.responses = []         # queue of (status, body_dict_or_str); last one repeats
        self.delay = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                outer.requests.append((self.command, self.path, body))
                if outer.delay:
                    time.sleep(outer.delay)
                queue = outer.responses
                status, payload = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else (200, {}))
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _serve

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stub():
    s = StubServer()
    yield s
    s.close()
