import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent import SYSTEM_RESCUE, load_plan, main
from tests.fake_github import FakeGitHub
from tests.test_runner import block
from tests.test_todo_example import BODIES
from tests.todo_ui_refs import UI_BODIES

ROOT = Path(__file__).parent.parent
FAKE_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2


class BuildLLM:
    """Answers by step title. `replies[title]` is a body (str) or a list used in order (the last one repeats)."""
    def __init__(self, plans, replies, repair=None, bodies=None):
        self.steps = {s.title: s for p in plans for s in p.steps}
        self.replies, self.repair, self.count, self.calls = replies, repair or {}, {}, []
        self.bodies = BODIES if bodies is None else bodies

    def chat(self, messages, **kw):
        user = messages[1]["content"]
        title = next(t for t in self.steps if f"Task: {t}\n" in user)
        step = self.steps[title]
        table = self.repair if messages[0]["content"] == SYSTEM_RESCUE else self.replies
        key = (title, messages[0]["content"] == SYSTEM_RESCUE)
        n = self.count[key] = self.count.get(key, 0) + 1
        self.calls.append(key)
        body = table.get(title, self.bodies.get(step.function, "return None"))
        if isinstance(body, list):
            body = body[min(n - 1, len(body) - 1)]
        return block(step.files[1], body)


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEAGENT_LESSONS", str(tmp_path / "global-lessons.json"))
    monkeypatch.setenv("CODEAGENT_STEP_COOLDOWN_S", "0")
    monkeypatch.setenv("CODEAGENT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("CODEAGENT_RESCUE_ATTEMPTS", "1")
    for k in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_API_URL", "GITHUB_GIT_URL_TEMPLATE", "CODEAGENT_ENV"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    with FakeGitHub(tmp_path / "remote") as gh:
        envfile = tmp_path / "token.env"
        envfile.write_text("".join(f"{k}={v}\n" for k, v in gh.env().items()), encoding="utf-8")
        yield type("Env", (), {"tmp": tmp_path, "gh": gh, "envfile": envfile})()


def make_project(tmp, plans=("todo_core",), name="Todo List", groups=None, description="A tiny todo list core"):
    proj = tmp / "proj"
    proj.mkdir(exist_ok=True)
    files = []
    for p in plans:
        dest = proj / f"{p}.json"
        if not dest.exists():
            shutil.copy(ROOT / "tests" / "examples" / f"{p}.json", dest)
        files.append(dest.name)
    spec = {"name": name, "description": description, "plans": files}
    if groups:
        (proj / "groups.json").write_text(json.dumps(groups))
        spec["groups"] = "groups.json"
    (proj / "project.json").write_text(json.dumps(spec))
    return proj / "project.json"


def run(argv, llm=None, out=None):
    lines = [] if out is None else out
    code = main(argv, llm_factory=lambda cfg: llm, guard_factory=None, out=lines.append)
    return code, "\n".join(lines)


def build(env, project, llm, *extra):
    return run(["build", str(project), "--workspace", str(env.tmp / "ws"), "--env", str(env.envfile), *extra], llm)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def plans_of(project):
    return [load_plan(project.parent / p) for p in json.loads(project.read_text())["plans"]]


def test_a_project_is_built_consolidated_and_published_as_the_user_to_a_private_repo(env):
    project = make_project(env.tmp)
    code, text = build(env, project, BuildLLM(plans_of(project), {}))
    assert code == 0, text
    assert "GitHub: octo   commits will be made as Octo Cat <4242+octo@users.noreply.github.com>" in text
    assert "created the private repository octo/todo-list" in text and "DONE: 7 steps built" in text
    assert "modules 7 -> 1" in text and "https://github.com/octo/todo-list" in text
    method, path, body = [r for r in env.gh.requests if r[0] == "POST"][0]
    assert body["private"] is True and body["name"] == "todo-list" and body["description"] == "A tiny todo list core"
    assert env.gh.branches("todo-list") == ["agent/todo-list", "main"]
    bare = env.gh.repos["todo-list"]
    assert git(bare, "rev-parse", "main") == git(bare, "rev-parse", "agent/todo-list")      # main fast-forwarded
    authors = set(git(bare, "log", "main", "--format=%an <%ae>|%cn <%ce>").splitlines())
    assert authors == {"Octo Cat <4242+octo@users.noreply.github.com>|Octo Cat <4242+octo@users.noreply.github.com>"}
    files = git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert ".gitignore" in files and "README.md" in files and "todo/todo_core.py" in files and "tests/test_todo_core.py" in files
    assert "ui.py" in files and "wrote ui.py" in text and "python ui.py" in text
    assert not any(f.startswith(".codeagent") or f == ".env" for f in files)
    subjects = git(bare, "log", "main", "--format=%s").splitlines()
    assert subjects[-1] == "chore: start Todo List" and "refactor: consolidate 7 modules into 1" in subjects
    assert FAKE_TOKEN not in text and env.gh.token not in text


def test_the_token_is_nowhere_in_the_workspace_or_the_logs(env):
    project = make_project(env.tmp)
    code, text = build(env, project, BuildLLM(plans_of(project), {}))
    assert code == 0
    for p in (env.tmp / "ws").rglob("*"):
        if p.is_file() and ".git/objects" not in str(p):
            try:
                assert env.gh.token not in p.read_text(encoding="utf-8"), p
            except UnicodeDecodeError:
                pass
    assert env.gh.token not in (env.tmp / "ws" / ".git" / "config").read_text()


def test_building_again_changes_nothing_and_creates_no_second_repo(env):
    project = make_project(env.tmp)
    llm = BuildLLM(plans_of(project), {})
    assert build(env, project, llm)[0] == 0
    posts = [r for r in env.gh.requests if r[0] == "POST"]
    code, text = build(env, project, llm)
    assert code == 0 and "already built and consolidated" in text
    assert [r for r in env.gh.requests if r[0] == "POST"] == posts
    assert env.gh.branches("todo-list") == ["agent/todo-list", "main"]


def test_without_a_token_it_says_how_to_set_one_up_and_touches_nothing(env):
    project = make_project(env.tmp)
    code, text = run(["build", str(project), "--workspace", str(env.tmp / "ws")], BuildLLM(plans_of(project), {}))
    assert code == 2 and "codeagent setup" in text and "--no-github" in text
    assert not (env.tmp / "ws").exists()


def test_a_wrong_token_stops_before_anything_is_built(env):
    project = make_project(env.tmp)
    env.envfile.write_text(env.envfile.read_text().replace(env.gh.token, "wrong_token_value"))
    code, text = build(env, project, BuildLLM(plans_of(project), {}))
    assert code == 2 and "rejected the token" in text and not (env.tmp / "ws").exists()


def test_no_github_builds_locally_and_still_merges_main(env):
    project = make_project(env.tmp)
    code, text = run(["build", str(project), "--workspace", str(env.tmp / "ws"), "--no-github"], BuildLLM(plans_of(project), {}))
    assert code == 0 and "DONE" in text and not env.gh.requests
    assert git(env.tmp / "ws", "rev-parse", "main") == git(env.tmp / "ws", "rev-parse", "agent/todo-list")


def test_no_main_leaves_main_at_the_first_commit(env):
    project = make_project(env.tmp)
    assert build(env, project, BuildLLM(plans_of(project), {}), "--no-main")[0] == 0
    bare = env.gh.repos["todo-list"]
    assert git(bare, "log", "main", "--format=%s") == "chore: start Todo List"
    assert "agent/todo-list" in env.gh.branches("todo-list")


def test_a_taken_repository_name_gets_a_suffix_and_the_user_is_told(env):
    project = make_project(env.tmp)
    from agent import GitHub
    GitHub(env.gh.token, env.gh.api_url).create_repo("todo-list")
    code, text = build(env, project, BuildLLM(plans_of(project), {}))
    assert code == 0 and "octo/todo-list-2" in text and "the name todo-list was taken" in text


def test_a_secret_in_generated_code_is_never_committed_or_pushed(env):
    project = make_project(env.tmp)
    leaky = BODIES["todo.fmt:format_task"] + f"\n# TOKEN = '{FAKE_TOKEN}'"
    leaky = "TOKEN = '" + FAKE_TOKEN + "'\n" + BODIES["todo.fmt:format_task"]
    llm = BuildLLM(plans_of(project), {"format_task prints one task as a line of text": leaky})
    code, text = build(env, project, llm)
    assert code == 1 and "refusing to commit" in text
    bare = env.gh.repos["todo-list"]
    for ref in env.gh.branches("todo-list"):
        assert FAKE_TOKEN not in git(bare, "log", ref, "-p", "--format=")
    assert "main" in env.gh.branches("todo-list")


def test_a_workspace_that_is_not_ours_is_not_used(env):
    project = make_project(env.tmp)
    (env.tmp / "ws").mkdir()
    (env.tmp / "ws" / "precious.txt").write_text("mine")
    code, text = build(env, project, BuildLLM(plans_of(project), {}))
    assert code == 2 and "was not made by `codeagent build`" in text and (env.tmp / "ws" / "precious.txt").exists()


def test_dry_run_prints_the_plan_and_creates_nothing(env):
    project = make_project(env.tmp)
    code, text = build(env, project, None, "--dry-run")
    assert code == 0 and "[dry run]" in text and "Todo Core: 7 steps" in text and not env.gh.requests and not (env.tmp / "ws").exists()


def test_project_file_mistakes_are_explained(env):
    bad = env.tmp / "p.json"
    for content, needle in [("{", "not valid JSON"), ('{"plans": ["a.json"]}', "\"name\" is required"),
                            ('{"name": "x", "plans": []}', "\"plans\" must list"), ('{"name": "x", "plans": ["nope.json"]}', "plan file not found"),
                            ('{"name": "x", "plans": ["nope.json"], "private": false}', "allowed keys")]:
        bad.write_text(content)
        code, text = run(["build", str(bad)], None)
        assert code == 2 and needle in text, (content, text)


# --------------------------------------------------------------------- a failing step: park it, report it, fix it, go on

def failing_plan_project(env):
    """4 steps: double (easy), triple (English recipe the model never gets), inc (independent), six (needs triple)."""
    def step(title, desc, mod, fn, sig, examples, context=()):
        return {"title": title, "description": desc, "commit": f"feat: add {fn}", "tests": [f"tests/test_{mod}.py"],
                "files": ["calc/__init__.py", f"calc/{mod}.py"], "context": [f"calc/{c}.py" for c in context],
                "function": f"calc.{mod}:{fn}", "signature": sig, "examples": examples}
    plan = {"title": "Calc", "test_command": "python -m pytest -q", "steps": [
        step("double doubles", "Create double(x) in calc/dbl.py.\n1. return x * 2", "dbl", "double", "double(x)", [{"args": [2], "expect": 4}]),
        step("triple triples", "Create triple(x) in calc/tri.py. Give back the number multiplied by three.", "tri", "triple", "triple(x)",
             [{"args": [2], "expect": 6}, {"args": [0], "expect": 0}]),
        step("inc adds one", "Create inc(x) in calc/inc.py.\n1. return x + 1", "inc", "inc", "inc(x)", [{"args": [1], "expect": 2}]),
        step("six uses triple", "Create six(x) in calc/six.py. Use triple from calc/tri.py (already written, shown below).\n1. tripled = triple(x)\n2. return tripled * 2",
             "six", "six", "six(x)", [{"args": [1], "expect": 6}], ["tri"]),
    ]}
    proj = env.tmp / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "calc.json").write_text(json.dumps(plan))
    (proj / "project.json").write_text(json.dumps({"name": "Calc", "plans": ["calc.json"], "workspace": str(env.tmp / "ws")}))
    return proj / "project.json"


CALC_BODIES = {"double doubles": "return x * 2", "inc adds one": "return x + 1", "six uses triple": "tripled = triple(x)\nreturn tripled * 2"}


def test_a_step_that_cannot_be_built_is_parked_reported_and_the_rest_is_built(env):
    project = failing_plan_project(env)
    llm = BuildLLM(plans_of(project), {**CALC_BODIES, "triple triples": "return x"}, repair={"triple triples": "return x"})
    code, text = build(env, project, llm)
    assert code == 1
    assert "1 step could not be built" in text and "(and 1 waiting for them)" in text and "  - triple triples" in text
    assert f"codeagent fix fix.json --workspace {env.tmp / 'ws'}" in text and "FAILURES.md" in text
    ws = env.tmp / "ws"
    assert (ws / "calc" / "dbl.py").exists() and (ws / "calc" / "inc.py").exists()          # the independent steps were built
    assert not (ws / "calc" / "six.py").exists()                                              # the dependent one waited
    report = (ws / ".codeagent" / "FAILURES.md").read_text(encoding="utf-8")
    assert "triple triples" in report and "return x" in report and "six uses triple" in report
    assert "main" in env.gh.branches("todo-list".replace("todo-list", "calc")) and "agent/calc" in env.gh.branches("calc")
    bare = env.gh.repos["calc"]
    assert git(bare, "log", "main", "--format=%s") == "chore: start Calc"                    # main untouched while steps are open


def test_the_fix_pack_from_an_ai_chat_fixes_the_step_and_the_next_build_finishes(env):
    project = failing_plan_project(env)
    llm = BuildLLM(plans_of(project), {**CALC_BODIES, "triple triples": "return x"}, repair={"triple triples": "return x"})
    assert build(env, project, llm)[0] == 1
    ws = env.tmp / "ws"
    reg = json.loads((ws / ".codeagent" / "failures.json").read_text())
    (fid,) = reg
    reply = ("Here you go:\n```json\n" + json.dumps({"fixes": [{"failure": fid, "plan_step": {
        "description": "Create triple(x) in calc/tri.py.\n1. return x * 3"}, "lessons": ["Write each step as code."]}]}) + "\n```")
    (env.tmp / "fix.json").write_text(reply)
    code, text = run(["fix", str(env.tmp / "fix.json"), "--workspace", str(ws)])
    assert code == 0 and "rewrote description" in text and "added 1 project lesson" in text
    assert "x * 3" in (project.parent / "calc.json").read_text()
    code, text = build(env, project, llm)
    assert code == 0, text
    assert "needed help: triple triples (recipe)" in text and "DONE: 4 steps built" in text
    assert not (ws / ".codeagent" / "FAILURES.md").exists()
    bare = env.gh.repos["calc"]
    assert git(bare, "rev-parse", "main") == git(bare, "rev-parse", "agent/calc")
    assert "calc/calc.py" in git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()    # consolidated into one module


def test_a_fix_can_be_pasted_into_the_terminal(env, monkeypatch):
    project = failing_plan_project(env)
    llm = BuildLLM(plans_of(project), {**CALC_BODIES, "triple triples": "return x"}, repair={"triple triples": "return x"})
    build(env, project, llm)
    (fid,) = json.loads((env.tmp / "ws" / ".codeagent" / "failures.json").read_text())
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"failure": fid, "body": "return x * 3"})))
    code, text = run(["fix", "-", "--workspace", str(env.tmp / "ws")])
    assert code == 0 and "saved the body" in text
    code, text = build(env, project, llm)
    assert code == 0 and "triple triples (your fix)" in text


def test_a_wrong_fix_pack_is_refused_and_changes_nothing(env):
    project = failing_plan_project(env)
    build(env, project, BuildLLM(plans_of(project), {**CALC_BODIES, "triple triples": "return x"}, repair={"triple triples": "return x"}))
    before = (project.parent / "calc.json").read_text()
    (env.tmp / "fix.json").write_text('{"fixes": [{"failure": "does-not-exist", "lessons": ["x"]}]}')
    code, text = run(["fix", str(env.tmp / "fix.json"), "--project", str(project)])
    assert code == 2 and "unknown failure id" in text and (project.parent / "calc.json").read_text() == before


# --------------------------------------------------------------------- setup and whoami

def test_setup_checks_the_token_and_saves_it_privately(env, monkeypatch):
    target = env.tmp / "saved.env"
    monkeypatch.setenv("GITHUB_API_URL", env.gh.api_url)
    monkeypatch.setattr(sys, "stdin", io.StringIO(env.gh.token + "\n"))
    code, text = run(["setup", "--env", str(target)])
    assert code == 0 and "signed in as octo" in text and env.gh.token not in text
    assert target.read_text() == f"GITHUB_TOKEN={env.gh.token}\n"
    if os.name == "posix":
        assert oct(target.stat().st_mode & 0o777) == "0o600"


def test_setup_refuses_a_bad_token_and_saves_nothing(env, monkeypatch):
    target = env.tmp / "saved.env"
    monkeypatch.setenv("GITHUB_API_URL", env.gh.api_url)
    monkeypatch.setattr(sys, "stdin", io.StringIO("nope_nope_nope_nope_nope_nope\n"))
    code, text = run(["setup", "--env", str(target)])
    assert code == 2 and "rejected the token" in text and not target.exists()


def test_whoami_shows_the_account_and_the_commit_identity(env):
    code, text = run(["whoami", "--env", str(env.envfile)])
    assert code == 0 and "signed in as octo" in text and "Octo Cat <4242+octo@users.noreply.github.com>" in text
    assert "token scopes: repo" in text and env.gh.token not in text


def test_fix_with_the_project_file_finds_the_workspace_named_in_it(env):
    project = failing_plan_project(env)
    build(env, project, BuildLLM(plans_of(project), {**CALC_BODIES, "triple triples": "return x"}, repair={"triple triples": "return x"}))
    (fid,) = json.loads((env.tmp / "ws" / ".codeagent" / "failures.json").read_text())
    (env.tmp / "fix.json").write_text(json.dumps({"failure": fid, "lessons": ["Keep steps short."]}))
    code, text = run(["fix", str(env.tmp / "fix.json"), "--project", str(project)])
    assert code == 0 and "added 1 project lesson" in text


def test_fix_in_a_folder_with_no_recorded_failures_explains_what_to_do(env):
    (env.tmp / "empty").mkdir()
    (env.tmp / "fix.json").write_text('{"lessons": ["x"]}')
    code, text = run(["fix", str(env.tmp / "fix.json"), "--workspace", str(env.tmp / "empty")])
    assert code == 2 and "nothing to fix" in text and "--workspace" in text


def test_the_whole_expense_manager_builds_as_one_project_with_one_command(env):
    import textwrap
    from tests.expense_refs import REF
    bodies = {fn: textwrap.dedent("\n".join(src.strip("\n").splitlines()[1:])) for fn, src in REF.items()}
    project = ROOT / "examples" / "expense_project.json"
    code, text = build(env, project, BuildLLM(plans_of(project), {}, bodies=bodies))
    assert code == 0, text
    assert "DONE: 38 steps built" in text and "modules 38 -> 6" in text and "132 passed before, 132 passed after" in text
    assert env.gh.branches("expense-manager") == ["agent/expense-manager", "main"]
    bare = env.gh.repos["expense-manager"]
    files = git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert sorted(f for f in files if f.startswith("expense_manager/")) == [
        "expense_manager/__init__.py", "expense_manager/budgets.py", "expense_manager/filters.py",
        "expense_manager/formatting.py", "expense_manager/records.py", "expense_manager/reports.py", "expense_manager/validation.py"]
    assert len(git(bare, "log", "main", "--format=%s").splitlines()) == 41                  # start + 38 steps + the merge + the UI
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=env.tmp / "ws", capture_output=True, text=True)
    assert result.returncode == 0 and "132 passed" in result.stdout


def test_setup_says_what_it_received_and_refuses_a_one_letter_paste_without_calling_github(env, monkeypatch):
    # real report: only "i" arrived, and the old error message turned every "i" in its own text into ***
    target = env.tmp / "saved.env"
    monkeypatch.setenv("GITHUB_API_URL", env.gh.api_url)
    monkeypatch.setattr(sys, "stdin", io.StringIO("i\n"))
    code, text = run(["setup", "--env", str(target)])
    assert code == 2 and "received 1 character, starting with 'i'" in text and "far too short" in text
    assert "codeagent setup --visible" in text and "GitHub" in text and "***" not in text
    assert not target.exists() and not env.gh.requests


def test_a_rejected_token_message_is_never_garbled(env, monkeypatch):
    target = env.tmp / "saved.env"
    monkeypatch.setenv("GITHUB_API_URL", env.gh.api_url)
    monkeypatch.setattr(sys, "stdin", io.StringIO("ghp_" + "x" * 36 + "\n"))
    code, text = run(["setup", "--env", str(target)])
    assert code == 2 and "GitHub rejected the token (401 bad credentials)" in text and "***" not in text


@pytest.mark.parametrize("pasted", [
    "\x1b[200~@@T@@\x1b[201~", "GITHUB_TOKEN=@@T@@", "export GITHUB_TOKEN='@@T@@'", '  "@@T@@"  ', "^[[200~@@T@@^[[201~",
])
def test_setup_cleans_what_a_terminal_paste_adds_around_the_token(env, monkeypatch, pasted):
    target = env.tmp / "saved.env"
    monkeypatch.setenv("GITHUB_API_URL", env.gh.api_url)
    monkeypatch.setattr(sys, "stdin", io.StringIO(pasted.replace("@@T@@", env.gh.token) + "\n"))
    code, text = run(["setup", "--env", str(target)])
    assert code == 0 and target.read_text() == f"GITHUB_TOKEN={env.gh.token}\n"


def test_setup_visible_reads_with_input(env, monkeypatch):
    target = env.tmp / "saved.env"
    monkeypatch.setenv("GITHUB_API_URL", env.gh.api_url)
    monkeypatch.setattr("builtins.input", lambda prompt="": env.gh.token)
    code, text = run(["setup", "--visible", "--env", str(target)])
    assert code == 0 and "signed in as octo" in text and target.exists()


def test_a_short_token_never_garbles_error_messages_from_github(env):
    from agent import GitHub, GitHubError
    with pytest.raises(GitHubError) as err:
        GitHub("i", env.gh.api_url).whoami()
    assert "GitHub rejected the token" in str(err.value)


def test_no_ui_leaves_ui_py_out(env):
    project = make_project(env.tmp)
    code, text = build(env, project, BuildLLM(plans_of(project), {}), "--no-ui")
    assert code == 0 and "ui.py" not in text
    assert "ui.py" not in git(env.gh.repos["todo-list"], "ls-tree", "-r", "--name-only", "main").splitlines()


def test_the_generated_ui_of_a_built_project_calls_its_real_functions(env):
    project = make_project(env.tmp)
    assert build(env, project, BuildLLM(plans_of(project), {}))[0] == 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("built_ui", env.tmp / "ws" / "ui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        assert mod.call("add_task", {"tasks": [], "text": "milk"})["result"] == [{"id": 1, "text": "milk", "done": False}]
        assert mod.TITLE == "Todo List"
    finally:
        for n in [n for n in sys.modules if n == "todo" or n.startswith("todo.")]:
            del sys.modules[n]
        if str(env.tmp / "ws") in sys.path:
            sys.path.remove(str(env.tmp / "ws"))


def test_a_model_built_web_app_is_written_committed_and_pushed_with_the_project(env):
    project = ROOT / "examples" / "todo_app_project.json"
    llm = BuildLLM(plans_of(project), {}, bodies={**BODIES, **UI_BODIES})
    code, text = build(env, project, llm)
    assert code == 0, text
    assert "DONE: 14 steps built" in text and "wrote app.py" in text and "python app.py" in text and "modules 14 -> 2" in text
    bare = env.gh.repos["todo-app"]
    files = git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert {"app.py", "ui.py", "todo/core.py", "todo/ui.py", ".gitignore"} <= set(files)
    assert "data.json" not in files and "data.json" in git(bare, "show", "main:.gitignore").splitlines()
    assert git(bare, "rev-parse", "main") == git(bare, "rev-parse", "agent/todo-app")
    subjects = git(bare, "log", "main", "--format=%s").splitlines()
    assert "feat: add the web app (app.py)" in subjects and "feat: add a small local UI (ui.py)" in subjects


def test_no_ui_also_leaves_out_the_web_app(env):
    project = ROOT / "examples" / "todo_app_project.json"
    code, text = build(env, project, BuildLLM(plans_of(project), {}, bodies={**BODIES, **UI_BODIES}), "--no-ui")
    assert code == 0 and "app.py" not in text
    files = git(env.gh.repos["todo-app"], "ls-tree", "-r", "--name-only", "main").splitlines()
    assert "app.py" not in files and "ui.py" not in files


def test_a_plan_with_a_wrong_context_path_fails_the_build_instead_of_saying_done(env):
    # real bug: context ["ui_button"] left six steps "waiting" for a file nobody makes, yet the build printed DONE
    proj = env.tmp / "proj"
    proj.mkdir()
    shutil.copy(ROOT / "tests" / "examples" / "todo_core.json", proj / "todo_core.json")
    ui = json.loads((ROOT / "tests" / "examples" / "todo_ui.json").read_text())
    next(st for st in ui["steps"] if st["function"].endswith("render_task"))["context"] = ["ui_button"]
    (proj / "todo_ui.json").write_text(json.dumps(ui))
    (proj / "project.json").write_text(json.dumps({"name": "Todo App", "plans": ["todo_core.json", "todo_ui.json"]}))
    project = proj / "project.json"
    code, text = build(env, project, BuildLLM(plans_of(project), {}, bodies={**BODIES, **UI_BODIES}))
    assert code == 1 and "DONE:" not in text and "steps built" not in text and "could not be built" in text and "render_task shows one task as a list item" in text
    assert not (env.tmp / "ws" / "ui.py").exists() and not (env.tmp / "ws" / "app.py").exists()
    report = (env.tmp / "ws" / ".codeagent" / "FAILURES.md").read_text(encoding="utf-8")
    assert "ui_button" in report and "render_task" in report
    bare = env.gh.repos["todo-app"]
    assert git(bare, "log", "main", "--format=%s") == "chore: start Todo App"            # main is not touched
