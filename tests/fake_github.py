"""A tiny local stand-in for api.github.com plus bare git repositories, so GitHub code can be tested offline."""
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FakeGitHub:
    def __init__(self, root: Path, token: str = "tok_test_1234567890", login: str = "octo", name: str = "Octo Cat",
                 user_id: int = 4242, email: str | None = None, scopes: str = "repo"):
        self.root, self.token, self.login, self.name, self.user_id, self.email = root, token, login, name, user_id, email
        self.scopes = scopes
        self.repos: dict[str, Path] = {}
        self.requests: list[tuple[str, str, dict | None]] = []
        self.refuse_create = 0                      # answer 403 to this many create calls
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):               # quiet
                pass

            def _send(self, status, body, headers=None):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _auth(self):
                if self.headers.get("Authorization") != f"Bearer {outer.token}":
                    self._send(401, {"message": "Bad credentials"})
                    return False
                return True

            def do_GET(self):
                outer.requests.append(("GET", self.path, None))
                if not self._auth():
                    return
                if self.path == "/user":
                    self._send(200, {"login": outer.login, "id": outer.user_id, "name": outer.name, "email": outer.email},
                               {"X-OAuth-Scopes": outer.scopes})
                else:
                    self._send(404, {"message": "Not Found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append(("POST", self.path, body))
                if not self._auth():
                    return
                if self.path != "/user/repos":
                    return self._send(404, {"message": "Not Found"})
                if outer.refuse_create > 0:
                    outer.refuse_create -= 1
                    return self._send(403, {"message": "Resource not accessible by personal access token"})
                name = body["name"]
                if name in outer.repos:
                    return self._send(422, {"message": "Repository creation failed.",
                                            "errors": [{"message": "name already exists on this account"}]})
                path = outer.root / outer.login / f"{name}.git"
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(path)], check=True)
                outer.repos[name] = path
                self._send(201, {"full_name": f"{outer.login}/{name}", "name": name, "private": body.get("private"),
                                 "html_url": f"https://github.com/{outer.login}/{name}"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def git_template(self) -> str:
        return f"file://{self.root}/{{owner}}/{{repo}}.git"

    def env(self) -> dict[str, str]:
        return {"GITHUB_TOKEN": self.token, "GITHUB_API_URL": self.api_url, "GITHUB_GIT_URL_TEMPLATE": self.git_template}

    def branches(self, repo: str) -> list[str]:
        out = subprocess.run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], cwd=self.repos[repo],
                             capture_output=True, text=True).stdout
        return sorted(out.split())
