"""The Todo example (plan + hand-written glue) works end to end with reference bodies standing in for the model."""
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from agent import Config, Runner, Workspace, load_plan
from tests.test_runner import FakeGuard, FakeLLM, block

ROOT = Path(__file__).parent.parent
BODIES = {
    "todo.ids:next_id": "if len(tasks) == 0:\n    return 1\nbiggest = 0\nfor task in tasks:\n    if task['id'] > biggest:\n        biggest = task['id']\nreturn biggest + 1",
    "todo.add:add_task": "if len(text.strip()) == 0:\n    raise ValueError\nnew_task = {'id': next_id(tasks), 'text': text.strip(), 'done': False}\nreturn [*tasks, new_task]",
    "todo.complete:complete_task": "result = []\nfound = False\nfor task in tasks:\n    if task['id'] == task_id:\n        new_task = dict(task)\n        new_task['done'] = True\n        result.append(new_task)\n        found = True\n    else:\n        result.append(task)\nif found == False:\n    raise KeyError\nreturn result",
    "todo.remove:remove_task": "result = []\nfor task in tasks:\n    if task['id'] != task_id:\n        result.append(task)\nif len(result) == len(tasks):\n    raise KeyError\nreturn result",
    "todo.filt:filter_tasks": "if status == 'all':\n    return list(tasks)\nif status == 'done':\n    return [task for task in tasks if task['done']]\nif status == 'open':\n    return [task for task in tasks if not task['done']]\nraise ValueError",
    "todo.fmt:format_task": "mark = ' '\nif task['done']:\n    mark = 'x'\nreturn '[' + mark + '] ' + str(task['id']) + ' ' + task['text']",
    "todo.summary:summarize": "done = 0\nfor task in tasks:\n    if task['done']:\n        done = done + 1\nreturn str(len(tasks) - done) + ' open, ' + str(done) + ' done'",
}


def test_the_todo_plan_builds_and_the_hand_written_glue_works_on_top(tmp_path):
    plan = load_plan(ROOT / "tests" / "examples" / "todo_core.json")
    replies = [block(step.files[1], textwrap.dedent(BODIES[step.function])) for step in plan.steps]
    cfg = Config()
    root = tmp_path / "proj"
    runner = Runner(cfg, plan, Workspace(root, cfg.max_file_bytes), FakeLLM(*replies), guard=FakeGuard(),
                    log=lambda m: None, sleep=lambda s: None, should_stop=lambda: False)
    report = runner.run()
    assert [s.status for s in report.results] == ["done"] * 7 and all(s.attempts == 1 for s in report.results)

    glue = ROOT / "examples" / "todo"
    for name in ("store.py", "cli.py", "__main__.py"):
        shutil.copy(glue / name, root / "todo" / name)
    shutil.copy(glue / "test_cli.py", root / "tests" / "test_cli.py")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = subprocess.run([sys.executable, "-m", "todo", "add", "buy milk"], cwd=root, capture_output=True, text=True)
    assert out.stdout.strip() == "added [ ] 1 buy milk"
