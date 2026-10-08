"""Prompts for a very small model: short, rigid, one job at a time.

The harness (not the model) decides what happens next; the model only ever
answers "write these specific files", in one fixed output format.
"""
from codeagent.plan import Plan, Step

SYSTEM = """You are a careful programmer. You write small, correct, working code.

Reply with ONLY file blocks, in exactly this format, and nothing else:

FILE: path/to/file.py
```python
<the COMPLETE content of the file>
```

Rules:
- Always write the COMPLETE file, never a diff or a snippet.
- Write only the files you are asked to write, with exactly those paths.
- No explanations, no text outside the file blocks.
- No comments and no docstrings: write only the code, as short as possible.
- An empty file is a block with nothing inside the fence."""


def _fit(files: dict[str, str | None], budget: int) -> str:
    """Render files as FILE blocks, sharing `budget` characters between them."""
    present = {p: c for p, c in files.items() if c is not None}
    if not present:
        return "(none)"
    share = max(200, budget // len(present))
    out = []
    for path, content in present.items():
        if len(content) > share:
            content = content[:share].rstrip() + "\n# ... (truncated)\n"
        out.append(f"FILE: {path}\n```\n{content}{'' if content.endswith(chr(10)) or not content else chr(10)}```")
    return "\n\n".join(out)


def _header(plan: Plan, step: Step) -> str:
    return f"Project: {plan.title}\nTask: {step.title}\n\n{step.description.strip()}\n"


def build_tests_prompt(plan: Plan, step: Step, existing: dict[str, str | None], budget: int,
                   feedback: str | None) -> list[dict]:
    parts = [
        _header(plan, step),
        "Write the TEST file(s) first (test-driven development).",
        "The tests must FAIL until the feature is implemented, so test the real behaviour with "
        "concrete examples and assert exact results.",
        "Every test is a plain function with NO parameters and the input written inside it, like: "
        "def test_lower():\n    assert f('A') == 'a'. Write 3 to 6 short tests, no fixtures, no loops.",
        f"Test file(s) to write: {', '.join(step.tests)}",
        f"The code under test will be written later in: {', '.join(step.files)}",
    ]
    ref = {p: existing.get(p) for p in (*step.files, *step.context)}
    if any(v is not None for v in ref.values()):
        parts.append("Existing files for reference (read-only):\n" + _fit(ref, budget))
    if feedback:
        parts.append("Problem with your previous answer, fix it:\n" + feedback)
    parts.append(f"Reply with ONLY the file block(s) for: {', '.join(step.tests)}. "
                 f"Do NOT write {', '.join(step.files)} now: the implementation comes later.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def build_impl_prompt(plan: Plan, step: Step, tests: dict[str, str], existing: dict[str, str | None],
                  budget: int, test_output: str | None, feedback: str | None) -> list[dict]:
    parts = [
        _header(plan, step),
        "These tests must pass. Do NOT change the tests:\n" + _fit(dict(tests), budget // 2),
        f"Write the implementation file(s): {', '.join(step.files)}",
    ]
    if step.function:
        module, name = step.function.split(":")
        parts.append(f"The tests call `{name}` imported from {module}: define exactly `def {name}(...)` "
                     f"at the top level of {module.replace('.', '/')}.py.")
    cur = {p: existing.get(p) for p in (*step.files, *step.context)}
    if any(v is not None for v in cur.values()):
        parts.append("Current content of the files:\n" + _fit(cur, budget // 2))
    if test_output:
        parts.append("Result of running the tests (fix what fails):\n" + test_output.strip())
    if feedback:
        parts.append("Problem with your previous answer, fix it:\n" + feedback)
    parts.append(f"Reply with ONLY the file block(s) for: {', '.join(step.files)}. "
                 f"Do NOT write or change {', '.join(step.tests)}: the tests are already final.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
