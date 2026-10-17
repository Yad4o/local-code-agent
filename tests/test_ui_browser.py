"""Drives the generated ui.py in a real headless browser. Skipped when playwright/chromium are not installed."""
import re
import subprocess
import sys

import pytest

from agent import main
from tests.test_ui import todo_workspace  # noqa: F401  (a fixture: a really built Todo project)

playwright = pytest.importorskip("playwright.sync_api")


def _chromium_available():
    try:
        with playwright.sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _chromium_available(), reason="no headless chromium")


def test_a_user_can_run_functions_chain_results_through_variables_and_see_errors(todo_workspace):  # noqa: F811
    ws, plan = todo_workspace
    assert main(["ui", str(ws), "--plans", str(plan), "--title", "Todo", "--no-commit"], out=lambda m: None) == 0
    srv = subprocess.Popen([sys.executable, "ui.py", "--port", "8791"], cwd=ws, stdout=subprocess.PIPE, text=True)
    try:
        url = re.search(r"http://127\.0\.0\.1:\d+", srv.stdout.readline()).group(0)
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            problems = []
            page.on("pageerror", lambda e: problems.append(str(e)))
            page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            page.goto(url)
            page.wait_for_selector("#list button")
            names = page.eval_on_selector_all("#list button", "els => els.map(e => e.textContent)")
            assert names == sorted(["add_task", "complete_task", "filter_tasks", "format_task", "next_id", "remove_task", "summarize"])
            page.fill("#q", "sum")                                              # the search box filters the list
            assert page.eval_on_selector_all("#list button", "els => els.map(e => e.textContent)") == ["summarize"]
            page.fill("#q", "")
            page.click("#list >> text=add_task")
            assert page.input_value("#p_text") == '"milk"'                      # prefilled from the plan's first example
            page.fill("#p_tasks", "[]")
            page.fill("#p_text", "buy tea")                                     # not JSON: sent as a plain string
            page.fill("#save", "t")
            page.click("button.run")
            page.wait_for_function("document.querySelector('#out').textContent.includes('saved as')")
            assert '"text": "buy tea"' in page.inner_text("#out") and page.get_attribute("#out", "class") == "ok"
            assert "$t" in page.inner_text("#vars")
            page.click("#list >> text=summarize")
            page.fill("#p_tasks", "$t")
            page.click("button.run")
            page.wait_for_function("document.querySelector('#out').className === 'ok'")      # not the example hint, the result
            assert page.inner_text("#out").strip() == '"1 open, 0 done"'
            page.click("#list >> text=remove_task")
            page.fill("#p_tasks", "$t")
            page.fill("#p_task_id", "9")
            page.click("button.run")
            page.wait_for_function("document.querySelector('#out').className === 'err'")
            assert page.inner_text("#out") == "KeyError"
            page.click("#vars button")                                          # delete the variable
            page.wait_for_function("!document.querySelector('#vars').textContent.includes('$t =')")
            assert problems == []
            browser.close()
    finally:
        srv.terminate()
        srv.wait(timeout=10)
