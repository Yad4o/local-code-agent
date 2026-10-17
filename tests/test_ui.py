import http.client
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent import Config, Runner, Workspace, load_plan, main, render_ui, ui_examples
from tests.test_runner import FakeGuard, FakeLLM, block
from tests.test_todo_example import BODIES

ROOT = Path(__file__).parent.parent

CORE = '''
from os.path import join


def add(a, b=1):
    """Add two numbers."""
    return a + b


def unique(items):
    return set(items)


def pair(x):
    return (x, x)


def boom(x):
    raise ValueError("bad value")


def silent(x):
    raise KeyError


def _private():
    return 1
'''


@pytest.fixture
def ui(tmp_path):
    pkg = tmp_path / "shop"
    pkg.mkdir()
    (pkg / "core.py").write_text(CORE, encoding="utf-8")
    (pkg / "__init__.py").write_text("from .core import add, unique, pair, boom, silent, join\n", encoding="utf-8")
    examples = {"add": [{"args": {"a": 1, "b": 2}, "expect": 3}], "boom": [{"args": {"x": 1}, "raises": "ValueError"}]}
    path = tmp_path / "ui.py"
    path.write_text(render_ui("shop", "Shop <Demo> & co", examples), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ui_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    server = mod.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield type("UI", (), {"mod": mod, "port": port, "path": path, "tmp": tmp_path})()
    server.shutdown()
    server.server_close()
    for name in [n for n in sys.modules if n == "shop" or n.startswith("shop.")]:
        del sys.modules[name]
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def get(ui, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{ui.port}{path}") as r:
        return r.status, r.read().decode()


def post(ui, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{ui.port}{path}", data=json.dumps(body).encode() if not isinstance(body, bytes) else body,
                                 method="POST", headers={"Content-Type": "application/json", "X-UI": "1", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_page_is_served_with_the_title_escaped(ui):
    status, html = get(ui, "/")
    assert status == 200 and "<title>Shop &lt;Demo&gt; &amp; co</title>" in html and "<script>" in html


def test_only_the_packages_own_public_functions_are_listed_with_params_defaults_and_examples(ui):
    _, body = get(ui, "/api/functions")
    funcs = {f["name"]: f for f in json.loads(body)}
    assert sorted(funcs) == ["add", "boom", "pair", "silent", "unique"]               # not _private, not the imported join
    assert funcs["add"]["params"] == [{"name": "a", "default": None}, {"name": "b", "default": "1"}]
    assert funcs["add"]["doc"] == "Add two numbers."
    assert funcs["add"]["examples"] == [{"args": {"a": 1, "b": 2}, "expect": 3}]
    assert funcs["boom"]["examples"][0]["raises"] == "ValueError" and funcs["pair"]["examples"] == []


def test_a_call_runs_the_function_and_the_default_is_used_when_the_argument_is_left_out(ui):
    assert post(ui, "/api/call", {"name": "add", "args": {"a": 2, "b": 5}}) == (200, {"ok": True, "result": 7, "saved": None})
    assert post(ui, "/api/call", {"name": "add", "args": {"a": 2}})[1]["result"] == 3


def test_errors_are_shown_not_raised(ui):
    assert post(ui, "/api/call", {"name": "boom", "args": {"x": 1}})[1] == {"ok": False, "error": "ValueError: bad value"}
    assert post(ui, "/api/call", {"name": "silent", "args": {"x": 1}})[1] == {"ok": False, "error": "KeyError"}
    err = post(ui, "/api/call", {"name": "add", "args": {}})[1]
    assert err["ok"] is False and "TypeError" in err["error"] and "missing" in err["error"]
    err = post(ui, "/api/call", {"name": "add", "args": {"a": 1, "zzz": 2}})[1]
    assert err["ok"] is False and "unexpected keyword" in err["error"]


def test_sets_and_tuples_come_back_as_json_lists(ui):
    assert post(ui, "/api/call", {"name": "unique", "args": {"items": [3, 1, 3, 2]}})[1]["result"] == [1, 2, 3]
    assert post(ui, "/api/call", {"name": "pair", "args": {"x": "a"}})[1]["result"] == ["a", "a"]


def test_a_result_can_be_saved_and_used_in_another_call_as_a_variable(ui):
    assert post(ui, "/api/call", {"name": "pair", "args": {"x": 4}, "save_as": "p"})[1]["saved"] == "p"
    assert json.loads(get(ui, "/api/vars")[1]) == {"p": [4, 4]}
    assert post(ui, "/api/call", {"name": "unique", "args": {"items": "$p"}})[1]["result"] == [4]
    assert post(ui, "/api/call", {"name": "pair", "args": {"x": ["$p", {"k": "$p"}]}})[1]["result"] == [[[4, 4], {"k": [4, 4]}]] * 2   # inside lists and objects too
    assert post(ui, "/api/call", {"name": "add", "args": {"a": "$missing"}})[1] == {"ok": False, "error": "KeyError: no saved variable $missing"}
    assert post(ui, "/api/delete_var", {"name": "p"})[1] == {}


def test_a_saved_variable_is_a_copy_so_a_function_cannot_change_it_behind_your_back(ui):
    post(ui, "/api/call", {"name": "pair", "args": {"x": [1]}, "save_as": "v"})
    post(ui, "/api/call", {"name": "unique", "args": {"items": "$v"}})
    assert json.loads(get(ui, "/api/vars")[1]) == {"v": [[1], [1]]}


def test_a_bad_variable_name_and_bad_bodies_are_rejected(ui):
    assert post(ui, "/api/call", {"name": "add", "args": {"a": 1}, "save_as": "not valid!"})[0] == 400
    assert post(ui, "/api/call", {"name": "add", "args": [1]})[0] == 400
    assert post(ui, "/api/call", b"{not json")[0] == 400
    assert post(ui, "/api/call", [1, 2])[0] == 400
    assert post(ui, "/api/call", {"name": "nope", "args": {}})[0] == 404
    assert post(ui, "/api/nothing", {})[0] == 404


def test_a_request_without_the_ui_header_or_from_another_host_is_refused(ui):
    req = urllib.request.Request(f"http://127.0.0.1:{ui.port}/api/call", data=b'{"name": "add", "args": {"a": 1}}', method="POST",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 403
    for method, path in (("GET", "/"), ("GET", "/api/functions"), ("POST", "/api/call")):
        conn = http.client.HTTPConnection("127.0.0.1", ui.port)
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", "evil.example.com")
        conn.putheader("X-UI", "1")
        conn.putheader("Content-Length", "2")
        conn.endheaders(b"{}")
        assert conn.getresponse().status == 403, (method, path)


def test_it_only_listens_on_this_computer(ui):
    assert ui.mod.make_server("127.0.0.1", 0).server_address[0] == "127.0.0.1"
    assert "0.0.0.0" not in ui.path.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_javascript_of_the_page_is_valid(ui, tmp_path):
    js = re.search(r"<script>(.*?)</script>", get(ui, "/")[1], re.S).group(1)
    (tmp_path / "page.js").write_text(js, encoding="utf-8")
    assert subprocess.run(["node", "--check", str(tmp_path / "page.js")], capture_output=True, text=True).returncode == 0


def test_a_hostile_title_cannot_inject_code_into_the_generated_file(tmp_path):
    title = 'x"""; import os; os.system("echo pwned") #\n\'\'\''
    src = render_ui("shop", title, {})
    compile(src, "ui.py", "exec")
    assert "pwned" not in re.search(r'^TITLE = (.*)$', src, re.M).group(0).replace(repr(title), "")
    ns = {"__file__": "ui.py"}
    code = src.split("pkg = importlib.import_module(PACKAGE)")[0]
    exec(code, ns)
    assert ns["TITLE"] == title and ns["PACKAGE"] == "shop"


def test_examples_are_named_by_parameter_with_keyword_arguments_merged_and_capped_at_three(tmp_path):
    plan = load_plan(ROOT / "tests" / "examples" / "todo_core.json")
    ex = ui_examples([plan])
    assert ex["add_task"][0] == {"args": {"tasks": [], "text": "milk"}, "expect": [{"id": 1, "text": "milk", "done": False}]}
    assert ex["add_task"][2] == {"args": {"tasks": [], "text": "   "}, "raises": "ValueError"}
    assert all(len(v) <= 3 for v in ex.values()) and len(ex) == 7
    cart = load_plan(ROOT / "tests" / "examples" / "cart_split.json")
    assert ui_examples([cart])["add_item"][0]["args"] == {"cart": [], "name": "apple", "price": 1.5}      # the default quantity is left out


# ------------------------------------------------------------------ the command, on a really built project

@pytest.fixture
def todo_workspace(tmp_path):
    plan = load_plan(ROOT / "tests" / "examples" / "todo_core.json")
    replies = [block(s.files[1], textwrap.dedent(BODIES[s.function])) for s in plan.steps]
    cfg = Config(step_cooldown_s=0, rescue=False)
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    Runner(cfg, plan, ws, FakeLLM(*replies), guard=FakeGuard(), log=lambda m: None, sleep=lambda s: None,
           should_stop=lambda: False).run()
    return tmp_path / "proj", ROOT / "tests" / "examples" / "todo_core.json"


def _load_ui(path: Path):
    spec = importlib.util.spec_from_file_location("ui_cmd_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_ui_command_writes_commits_and_the_result_really_calls_the_projects_functions(todo_workspace):
    ws, plan = todo_workspace
    lines = []
    assert main(["ui", str(ws), "--plans", str(plan)], out=lines.append) == 0
    assert (ws / "ui.py").exists() and "python ui.py" in "\n".join(lines)
    assert subprocess.run(["git", "log", "-1", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout.strip() == "feat: add a small local UI (ui.py)"
    mod = _load_ui(ws / "ui.py")
    try:
        names = [f["name"] for f in mod.describe()]
        assert names == sorted(["add_task", "complete_task", "filter_tasks", "format_task", "next_id", "remove_task", "summarize"])
        out = mod.call("add_task", {"tasks": [], "text": " milk "}, save_as="t")
        assert out["result"] == [{"id": 1, "text": "milk", "done": False}]
        assert mod.call("format_task", {"task": "$t"})["ok"] is False                      # a list is not a task: shown as an error
        assert mod.call("summarize", {"tasks": "$t"}) == {"ok": True, "result": "1 open, 0 done", "saved": None}
    finally:
        for n in [n for n in sys.modules if n == "todo" or n.startswith("todo.")]:
            del sys.modules[n]
        if str(ws) in sys.path:
            sys.path.remove(str(ws))


def test_the_package_is_found_by_itself_and_no_commit_leaves_the_tree_alone(todo_workspace):
    ws, _ = todo_workspace
    before = subprocess.run(["git", "log", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout
    assert main(["ui", str(ws), "--no-commit"], out=lambda m: None) == 0
    assert subprocess.run(["git", "log", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout == before
    assert "PACKAGE = 'todo'" in (ws / "ui.py").read_text(encoding="utf-8")


def test_two_packages_without_a_hint_is_explained(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("")
    lines = []
    assert main(["ui", str(tmp_path)], out=lines.append) == 2 and "--package" in lines[0]
