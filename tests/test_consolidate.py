import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import agent
from agent import (Config, ConsolidateError, Runner, Workspace, consolidate_sources, load_groups, load_plan, main,
                   suggest_groups)
from tests.expense_refs import REF
from tests.test_runner import FakeGuard, FakeLLM, block

ROOT = Path(__file__).parent.parent
PKG = "pkg"
MODS = {
    "a": "import re\n\n\ndef fa(x):\n    return re.sub('a', 'b', x)\n",
    "b": "from pkg.a import fa\n\n\ndef fb(x):\n    unused = fa(x)\n    return fa(x) + 'b'\n",
    "c": "import re\nfrom pkg.b import fb\n\n\ndef fc(x):\n    return fb(x) + re.escape(x)\n",
}
TESTS = {
    "a": "import pytest\nfrom pkg.a import fa\n\n\ndef test_example_1():\n    assert fa('a') == 'b'\n",
    "b": "import pytest\nfrom pkg.b import fb\n\n\ndef test_example_1():\n    assert fb('c') == 'cb'\n",
    "c": "import pytest\nfrom pkg.c import fc\n\n\ndef test_example_1():\n    assert fc('x') == 'xbx'\n",
}
INIT = "from .a import fa\nfrom .b import fb\nfrom .c import fc\n"


def merge(groups, mods=MODS, tests=TESTS, init=INIT, **kw):
    return consolidate_sources(PKG, groups, mods, tests, init, **kw)


def test_modules_are_merged_with_imports_rewritten_and_deduplicated():
    r = merge({"low": ["a", "b"], "high": ["c"]})
    low, high = r.writes["pkg/low.py"], r.writes["pkg/high.py"]
    assert low.startswith("import re\n") and "from pkg.a" not in low and "def fa(x):" in low and "def fb(x):" in low
    assert "import re\n\nfrom pkg.low import fb\n" in high
    assert r.writes["pkg/__init__.py"] == "from .low import fa, fb\nfrom .high import fc\n"
    assert sorted(r.deletes) == ["pkg/a.py", "pkg/b.py", "pkg/c.py", "tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_unused_call_assignments_are_tidied_unless_asked_not_to():
    assert "unused = " not in merge({"low": ["a", "b"], "high": ["c"]}).writes["pkg/low.py"]
    assert "    fa(x)\n" in merge({"low": ["a", "b"], "high": ["c"]}).writes["pkg/low.py"]
    assert "unused = fa(x)" in merge({"low": ["a", "b"], "high": ["c"]}, tidy=False).writes["pkg/low.py"]


def test_test_files_are_merged_and_repeated_test_names_get_the_old_module_name():
    r = merge({"low": ["a", "b"], "high": ["c"]})
    t = r.writes["tests/test_low.py"]
    assert "from pkg.low import fa, fb" in t or "from pkg.low import fa" in t
    assert "def test_a_example_1():" in t and "def test_b_example_1():" in t and "def test_example_1" not in t
    assert t.count("import pytest") == 1
    assert "def test_example_1():" in r.writes["tests/test_high.py"]            # not repeated there, so not renamed


def test_a_circle_between_new_modules_is_refused():
    with pytest.raises(ConsolidateError, match="circle"):
        merge({"one": ["a", "c"], "two": ["b"]})        # one -> two (c uses b), two -> one (b uses a)


def test_unknown_module_duplicate_module_and_clashing_name_are_refused(tmp_path):
    with pytest.raises(ConsolidateError, match="does not exist"):
        merge({"low": ["a", "zzz"]})
    with pytest.raises(ConsolidateError, match="already a module"):
        merge({"b": ["a"], "other": ["c"]})
    bad = tmp_path / "g.json"
    bad.write_text(json.dumps({"package": "pkg", "modules": {"x": ["a"], "y": ["a"]}}))
    with pytest.raises(ConsolidateError, match="both"):
        load_groups(bad)


def test_two_functions_with_the_same_name_in_one_group_are_refused():
    mods = {**MODS, "d": "def fa(x):\n    return x\n"}
    with pytest.raises(ConsolidateError, match="more than one"):
        merge({"low": ["a", "d"], "rest": ["b", "c"]}, mods=mods)


def test_modules_imported_by_alias_are_rewritten_to_direct_names():
    # real run: the model wrote `from pkg import a as am` and called am.fa(...)
    mods = {"a": MODS["a"], "e": "from pkg import a as am\n\n\ndef fe(x):\n    return am.fa(x)\n"}
    same = merge({"low": ["a", "e"]}, mods=mods, tests={}, init="from .a import fa\nfrom .e import fe\n")
    assert "am" not in same.writes["pkg/low.py"] and "return fa(x)" in same.writes["pkg/low.py"]
    apart = merge({"one": ["a"], "two": ["e"]}, mods=mods, tests={}, init="from .a import fa\nfrom .e import fe\n")
    assert "from pkg.one import fa" in apart.writes["pkg/two.py"] and "am." not in apart.writes["pkg/two.py"]
    dotted = {"a": MODS["a"], "e": "import pkg.a as am\n\n\ndef fe(x):\n    return am.fa(x)\n"}
    assert "return fa(x)" in merge({"low": ["a", "e"]}, mods=dotted, tests={}, init=None).writes["pkg/low.py"]


def test_a_module_alias_used_on_its_own_is_refused():
    mods = {"a": MODS["a"], "e": "from pkg import a as am\n\n\ndef fe(x):\n    return list(map(am.fa, x)), am\n"}
    with pytest.raises(ConsolidateError, match="on its own"):
        merge({"low": ["a", "e"]}, mods=mods, tests={}, init=None)


def test_modules_not_in_any_group_are_left_alone_and_keep_their_init_lines():
    mods = {**MODS, "store": "def save():\n    return 1\n"}
    r = merge({"low": ["a", "b"], "high": ["c"]}, mods=mods, init=INIT + "from .store import save\n")
    assert r.untouched == ["store"] and "from .store import save" in r.writes["pkg/__init__.py"]


def test_suggest_groups_makes_one_module_per_plan_named_from_the_titles():
    plans = [load_plan(ROOT / "tests" / "examples" / f"{n}.json") for n in ("expense_core", "expense_reports", "expense_editing")]
    g = suggest_groups(plans)
    assert g["package"] == "expense_manager" and list(g["modules"]) == ["core", "reports", "editing_and_display"]
    assert len(g["modules"]["core"]) == 13 and g["modules"]["core"][0] == "amounts"
    assert list(suggest_groups(plans[:1])["modules"]) == ["expense_manager_core"]


# --------------------------------------------------------------- the whole command, on a real built workspace

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("built") / "proj"
    cfg = Config()
    ws = Workspace(root, cfg.max_file_bytes)
    for name in ("expense_core", "expense_reports", "expense_editing"):
        plan = load_plan(ROOT / "tests" / "examples" / f"{name}.json")
        replies = [block(s.files[1], textwrap.dedent("\n".join(REF[s.function].strip("\n").splitlines()[1:]))) for s in plan.steps]
        Runner(cfg, plan, ws, FakeLLM(*replies), guard=FakeGuard(), log=lambda m: None, sleep=lambda s: None,
               should_stop=lambda: False).run()
    return root


def _copy(built, tmp_path):
    import shutil
    dest = tmp_path / "proj"
    shutil.copytree(built, dest)
    return dest


def test_the_command_merges_a_real_project_and_the_tests_still_pass(built, tmp_path):
    ws = _copy(built, tmp_path)
    lines: list[str] = []
    code = main(["consolidate", str(ROOT / "examples" / "expense_groups.json"), "--workspace", str(ws)], out=lines.append)
    assert code == 0, lines
    text = "\n".join(lines)
    assert "modules 38 -> 6" in text and "132 passed before, 132 passed after" in text
    assert sorted(p.name for p in (ws / "expense_manager").glob("*.py")) == [
        "__init__.py", "budgets.py", "filters.py", "formatting.py", "records.py", "reports.py", "validation.py"]
    assert len(list((ws / "tests").glob("test_*.py"))) == 6
    assert "amount_ok" not in (ws / "expense_manager" / "validation.py").read_text(encoding="utf-8")
    assert subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ws, capture_output=True, text=True).returncode == 0
    assert "refactor: consolidate 38 modules into 6" in subprocess.run(["git", "log", "-1", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout
    assert (ws / ".codeagent" / "consolidated.json").exists()
    out2: list[str] = []
    assert main(["run", str(ROOT / "tests" / "examples" / "expense_core.json"), "--workspace", str(ws)], out=out2.append) == 2
    assert "was consolidated" in out2[0]
    assert "import expense_manager" not in text


def test_dry_run_changes_nothing(built, tmp_path):
    ws = _copy(built, tmp_path)
    before = sorted(p.name for p in (ws / "expense_manager").iterdir())
    lines: list[str] = []
    assert main(["consolidate", str(ROOT / "examples" / "expense_groups.json"), "--workspace", str(ws), "--dry-run"], out=lines.append) == 0
    assert "nothing was written" in lines[-1] and sorted(p.name for p in (ws / "expense_manager").iterdir()) == before


def test_if_the_tests_change_everything_is_put_back(built, tmp_path, monkeypatch):
    ws = _copy(built, tmp_path)
    before = {p.name: p.read_text(encoding="utf-8") for p in (ws / "expense_manager").glob("*.py")}
    calls = iter([(132, 0, "132 passed"), (130, 2, "2 failed, 130 passed")])
    monkeypatch.setattr(agent, "_pytest_passed", lambda root: next(calls))
    lines: list[str] = []
    assert main(["consolidate", str(ROOT / "examples" / "expense_groups.json"), "--workspace", str(ws)], out=lines.append) == 2
    assert "Nothing was changed" in lines[-1]
    assert {p.name: p.read_text(encoding="utf-8") for p in (ws / "expense_manager").glob("*.py")} == before
    assert not (ws / ".codeagent" / "consolidated.json").exists()


def test_uncommitted_changes_and_failing_tests_stop_it_before_anything_is_touched(built, tmp_path, monkeypatch):
    ws = _copy(built, tmp_path)
    (ws / "notes.txt").write_text("hello", encoding="utf-8")
    lines: list[str] = []
    assert main(["consolidate", str(ROOT / "examples" / "expense_groups.json"), "--workspace", str(ws)], out=lines.append) == 2
    assert "uncommitted changes" in lines[-1]
    (ws / "notes.txt").unlink()
    monkeypatch.setattr(agent, "_pytest_passed", lambda root: (3, 1, "1 failed, 3 passed"))
    assert main(["consolidate", str(ROOT / "examples" / "expense_groups.json"), "--workspace", str(ws)], out=lines.append) == 2
    assert "do not pass before" in lines[-1]
    assert len(list((ws / "expense_manager").glob("*.py"))) == 39


def test_suggest_prints_a_groups_file(capsys):
    lines: list[str] = []
    ex = ROOT / "tests" / "examples"
    assert main(["consolidate", "--suggest", str(ex / "expense_core.json"), str(ex / "expense_reports.json")], out=lines.append) == 0
    assert list(json.loads("\n".join(lines))["modules"]) == ["core", "reports"]
