"""Builds the whole Todo app (core + model-built UI plans) and uses it in a real headless browser."""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from agent import load_plan, main
from tests.test_build import BuildLLM
from tests.test_todo_example import BODIES
from tests.todo_ui_refs import UI_BODIES

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).parent.parent


def _chromium_available():
    try:
        with playwright.sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _chromium_available(), reason="no headless chromium")


@pytest.fixture
def built_app(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEAGENT_LESSONS", str(tmp_path / "none.json"))
    monkeypatch.setenv("CODEAGENT_STEP_COOLDOWN_S", "0")
    monkeypatch.setenv("CODEAGENT_MAX_ATTEMPTS", "2")
    monkeypatch.chdir(tmp_path)
    project = ROOT / "examples" / "todo_app_project.json"
    plans = [load_plan(project.parent / p) for p in json.loads(project.read_text())["plans"]]
    ws = tmp_path / "ws"
    code = main(["build", str(project), "--workspace", str(ws), "--no-github"],
                llm_factory=lambda cfg: BuildLLM(plans, {}, bodies={**BODIES, **UI_BODIES}), guard_factory=None, out=lambda m: None)
    assert code == 0 and (ws / "app.py").exists()
    return ws


def _start(ws):
    srv = subprocess.Popen([sys.executable, "app.py", "--port", "8793"], cwd=ws, stdout=subprocess.PIPE, text=True)
    return srv, re.search(r"http://127\.0\.0\.1:\d+", srv.stdout.readline()).group(0)


def test_a_user_can_add_finish_and_remove_tasks_in_the_model_built_app(built_app):
    srv, url = _start(built_app)
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            problems = []
            page.on("pageerror", lambda e: problems.append(str(e)))
            page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            page.goto(url)
            assert "0 open, 0 done" in page.inner_text("body") and "No tasks yet." in page.inner_text("body")
            for text in ("buy tea", "<b>call mum</b>", "pay rent"):
                page.fill("input[name=text]", text)
                page.click("text=Add")
            body = page.inner_text("body")
            assert "3 open, 0 done" in body and "buy tea" in body and "<b>call mum</b>" in body       # shown as text
            assert page.locator("li b").count() == 0                                                 # not interpreted as HTML
            page.locator("li", has_text="buy tea").get_by_role("button", name="Done").click()
            assert "2 open, 1 done" in page.inner_text("body") and page.locator("li s").inner_text() == "buy tea"
            page.locator("li", has_text="pay rent").get_by_role("button", name="Remove").click()
            assert "pay rent" not in page.inner_text("body") and "1 open, 1 done" in page.inner_text("body")
            assert problems == []
            browser.close()
        saved = json.loads((built_app / "data.json").read_text())
        assert [t["text"] for t in saved] == ["buy tea", "<b>call mum</b>"] and saved[0]["done"] is True
    finally:
        srv.terminate()
        srv.wait(timeout=10)
    srv2, url2 = _start(built_app)                                                                  # the data is still there
    try:
        assert "&lt;b&gt;call mum&lt;/b&gt;" in urllib.request.urlopen(url2).read().decode()
    finally:
        srv2.terminate()
        srv2.wait(timeout=10)
