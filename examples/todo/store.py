"""Hand-written glue: load and save the task list as a JSON file."""
import json
from pathlib import Path


def load_tasks(path):
    file = Path(path)
    if not file.exists():
        return []
    return json.loads(file.read_text(encoding="utf-8"))


def save_tasks(path, tasks):
    Path(path).write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
