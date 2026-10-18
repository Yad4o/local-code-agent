import http.client
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent import ProjectError, ensure_repo, has_app_functions, main, render_app, write_app

MINI = '''
import html


def initial_state():
    return {"items": ["first"]}


def render_page(state):
    if state.get("crash"):
        raise RuntimeError("boom")
    if state.get("not_text"):
        return 5
    return "<!doctype html><title>Mini</title><ul>" + "".join("<li>" + html.escape(i) + "</li>" for i in state["items"]) + "</ul>"


def handle_add(state, form):
    if form["text"].strip() == "":
        raise ValueError("empty")
    return {"items": state["items"] + [form["text"]]}


def handle_nothing(state, form):
    return None


def handle_weird(state, form):
    return {"x": {1, 2}}


def handle_crash(state, form):
    return {"items": [], "crash": True}


def handle_not_text(state, form):
    return {"items": [], "not_text": True}


def handle_missing(state, form):
    raise KeyError("no such thing")
'''


@pytest.fixture
def app(tmp_path):
    pkg = tmp_path / "mini"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .main import *\n", encoding="utf-8")
    (pkg / "main.py").write_text(MINI, encoding="utf-8")
    (tmp_path / "app.py").write_text(render_app("mini", "Mini <App>"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("app_under_test", tmp_path / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    servers = []

    def start():
        server = mod.make_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    port = start()
    yield type("App", (), {"mod": mod, "port": port, "tmp": tmp_path, "start": staticmethod(start)})()
    for s in servers:
        s.shutdown()
        s.server_close()
    for name in [n for n in sys.modules if n == "mini" or n.startswith("mini.")]:
        del sys.modules[name]
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def request(app, path, data=None, headers=None, port=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port or app.port}{path}", data=data, method="POST" if data is not None else "GET", headers=headers or {})
    try:
        with urllib.request.build_opener(NoRedirect).open(req) as r:
            return r.status, r.read().decode(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e.headers


def test_the_page_comes_from_render_page_with_the_state_and_a_strict_policy(app):
    status, body, headers = request(app, "/")
    assert status == 200 and "<li>first</li>" in body                       # initial_state() was used
    csp = headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "form-action 'self'" in csp and "script-src" not in csp and "'unsafe-eval'" not in csp
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["Cache-Control"] == "no-store"


def test_a_posted_form_runs_handle_action_saves_the_state_and_redirects_home(app):
    status, _, headers = request(app, "/add", b"text=hello+world%21")
    assert status == 303 and headers["Location"] == "/"
    assert json.loads((app.tmp / "data.json").read_text()) == {"items": ["first", "hello world!"]}
    assert "<li>hello world!</li>" in request(app, "/")[1]


def test_the_state_survives_a_restart(app):
    request(app, "/add", b"text=kept")
    port2 = app.start()                                                      # a fresh server reads data.json again
    assert "<li>kept</li>" in request(app, "/", port=port2)[1]


def test_what_a_handler_raises_is_shown_as_a_plain_error_page_and_nothing_is_saved(app):
    status, body, _ = request(app, "/add", b"text=+++")
    assert status == 400 and "Could not do that" in body and "ValueError: empty" in body and "href='/'" in body
    status, body, _ = request(app, "/missing", b"")
    assert status == 400 and "KeyError: no such thing" in body
    status, body, _ = request(app, "/add", b"")                              # the field is missing
    assert status == 400 and "KeyError: text" in body
    assert not (app.tmp / "data.json").exists()


def test_a_handler_that_returns_nothing_or_a_state_that_is_not_json_is_refused(app):
    status, body, _ = request(app, "/nothing", b"")
    assert status == 400 and "returned nothing" in body
    status, body, _ = request(app, "/weird", b"")
    assert status == 400 and "not JSON serializable" in body and not (app.tmp / "data.json").exists()


def test_a_page_that_cannot_be_drawn_is_explained_not_a_stack_trace(app):
    request(app, "/crash", b"")
    status, body, _ = request(app, "/")
    assert status == 500 and "The page could not be drawn" in body and "RuntimeError: boom" in body and "Traceback" not in body
    request(app, "/not_text", b"")
    status, body, _ = request(app, "/")
    assert status == 500 and "must return the page as text" in body


def test_a_form_that_does_not_come_from_the_page_is_refused(app):
    for headers in ({"Origin": "http://evil.example.com"}, {"Origin": f"http://127.0.0.1:{app.port + 1}"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
        assert request(app, "/add", b"text=hack", headers)[0] == 403, headers
    assert request(app, "/add", b"text=ok", {"Origin": f"http://127.0.0.1:{app.port}"})[0] == 303
    assert request(app, "/add", b"text=ok2", {"Sec-Fetch-Site": "same-origin"})[0] == 303
    assert request(app, "/add", b"text=ok3", {"Origin": f"http://localhost:{app.port}"})[0] == 303
    assert json.loads((app.tmp / "data.json").read_text())["items"] == ["first", "ok", "ok2", "ok3"]


def test_another_host_name_is_refused_even_for_reading(app):
    for method in ("GET", "POST"):
        conn = http.client.HTTPConnection("127.0.0.1", app.port)
        conn.putrequest(method, "/" if method == "GET" else "/add", skip_host=True)
        conn.putheader("Host", "evil.example.com")
        conn.putheader("Content-Length", "8")
        conn.endheaders(b"text=bad" if method == "POST" else b"")
        assert conn.getresponse().status == 403


def test_unknown_pages_and_actions_and_huge_forms(app):
    assert request(app, "/nope")[0] == 404
    assert request(app, "/nope", b"")[0] == 404
    assert request(app, "/add", b"text=" + b"x" * 200_000)[0] == 413
    assert request(app, "/render_page", b"")[0] == 404                       # only handle_* functions are actions


def test_the_title_in_the_generated_file_cannot_inject_code():
    title = 'x"""; import os; os.system("echo pwned") #\n\'\'\''
    src = render_app("mini", title)
    compile(src, "app.py", "exec")
    import ast
    tree = ast.parse(src)
    assert "Only this computer can connect." in ast.get_docstring(tree)          # the docstring was not closed early
    assert [type(n).__name__ for n in tree.body[:3]] == ["Expr", "Import", "Import"]    # no statement slipped in before the imports
    ns = {"__file__": "app.py"}
    exec(src.split("pkg = importlib.import_module(PACKAGE)")[0], ns)
    assert ns["TITLE"] == title and ns["PACKAGE"] == "mini"


def test_it_only_listens_on_this_computer(app):
    assert "0.0.0.0" not in (app.tmp / "app.py").read_text(encoding="utf-8")
    assert app.mod.make_server("127.0.0.1", 0).server_address[0] == "127.0.0.1"


def test_app_py_is_only_written_for_a_package_that_has_a_page(tmp_path):
    pkg = tmp_path / "plain"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "m.py").write_text("def f():\n    return 1\n")
    assert has_app_functions(tmp_path, "plain") is False
    with pytest.raises(ProjectError, match="no render_page"):
        write_app(tmp_path, "plain", "Plain", commit=False)
    (pkg / "page.py").write_text("def render_page(state):\n    return ''\n")
    assert has_app_functions(tmp_path, "plain") is True


def test_the_app_command_writes_and_commits_app_py(tmp_path):
    ensure_repo(tmp_path)
    pkg = tmp_path / "mini"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "m.py").write_text(MINI)
    lines = []
    assert main(["app", str(tmp_path), "--title", "Mini"], out=lines.append) == 0
    assert (tmp_path / "app.py").exists() and "python app.py" in "\n".join(lines)
    import subprocess
    assert subprocess.run(["git", "log", "-1", "--format=%s"], cwd=tmp_path, capture_output=True, text=True).stdout.strip() == "feat: add the web app (app.py)"
    out = []
    (tmp_path / "mini" / "m.py").write_text("def f():\n    return 1\n")
    assert main(["app", str(tmp_path)], out=out.append) == 2 and "no render_page" in out[0]
