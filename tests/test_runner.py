import os
import subprocess

import pytest

from codeagent import git
from codeagent.config import Config
from codeagent.llm import LLMError
from codeagent.plan import PlanError, parse_plan
from codeagent.runner import Runner
from codeagent.state import WorkspaceBusy, WorkspaceLock
from codeagent.workspace import Workspace


# ----------------------------------------------------------------- helpers

def block(path, code):
    body = f"{code}\n" if code else ""
    return f"FILE: {path}\n```python\n{body}```\n"


TEST_CORE = block("tests/test_core.py",
                  "from slugify.core import slugify\n\n"
                  "def test_basic():\n    assert slugify('Hello, World!') == 'hello-world'\n")
TEST_VACUOUS = block("tests/test_core.py", "def test_nothing():\n    assert True\n")
INIT = block("slugify/__init__.py", "")
CORE_OK = block("slugify/core.py",
                "import re\n\ndef slugify(t):\n    return re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')\n")
CORE_BAD = block("slugify/core.py", "def slugify(t):\n    return t\n")

TEST_EXTRA = block("tests/test_extra.py",
                   "from slugify.extra import shout\n\ndef test_shout():\n    assert shout('hi') == 'HI!'\n")
EXTRA_OK = block("slugify/extra.py", "def shout(t):\n    return t.upper() + '!'\n")


def plan_text(*, two=False, test_command="python -m pytest -q"):
    text = f'''
title = "Slugify utility"
test_command = "{test_command}"

[[step]]
title = "core function"
description = "Create slugify(text) in slugify/core.py. 'Hello, World!' -> 'hello-world'."
commit = "feat: add slugify core function"
tests = ["tests/test_core.py"]
files = ["slugify/__init__.py", "slugify/core.py"]
'''
    if two:
        text += '''
[[step]]
title = "shout helper"
description = "Create shout(text) in slugify/extra.py returning the text upper-cased plus '!'."
commit = "feat: add shout helper"
tests = ["tests/test_extra.py"]
files = ["slugify/extra.py"]
'''
    return text


class Cut(str):
    """A reply that hit the token limit (like Reply(truncated=True))."""

    truncated = True


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.kwargs = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        self.kwargs.append(kw)
        assert self.replies, "unexpected extra LLM call"
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeGuard:
    def __init__(self, ok=True):
        self.ok = ok

    def wait_until_ok(self, **kw):
        return self.ok


@pytest.fixture(autouse=True)
def isolated_git(tmp_path_factory, monkeypatch):
    cfg = tmp_path_factory.mktemp("gitcfg") / "gitconfig"
    cfg.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("HOME", str(cfg.parent))


@pytest.fixture
def cfg():
    return Config(step_cooldown_s=0, max_attempts=3, test_timeout=60)


def make_runner(tmp_path, llm, cfg, *, two=False, guard=None, sleep=None, stop=None, log=None, **plan_kw):
    plan = parse_plan(plan_text(two=two, **plan_kw))
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    return Runner(cfg, plan, ws, llm, guard=guard or FakeGuard(), log=log or (lambda m: None),
                  sleep=sleep or (lambda s: None), should_stop=stop or (lambda: False)), ws


def log_subjects(root):
    out = subprocess.run(["git", "log", "--format=%s"], cwd=root, capture_output=True, text=True).stdout
    return out.split("\n")[:-1]


# ----------------------------------------------------------------- happy path

def test_happy_path_writes_tests_then_code_then_commits(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    report = r.run()
    assert report.outcome == "done"
    assert [x.status for x in report.results] == ["done"]
    assert report.results[0].commit
    assert ws.read("slugify/core.py").startswith("import re")
    assert ws.read("tests/test_core.py").startswith("from slugify.core")
    assert len(llm.calls) == 2
    assert log_subjects(ws.root) == ["feat: add slugify core function"]
    assert git.current_branch(ws.root) == "agent/slugify-utility"


def test_commit_contains_exactly_the_step_files(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    (ws.root / "unrelated.txt").write_text("mine")
    r.run()
    files = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"], cwd=ws.root,
                           capture_output=True, text=True).stdout.split()
    assert sorted(files) == ["slugify/__init__.py", "slugify/core.py", "tests/test_core.py"]


def test_tests_prompt_comes_first_and_impl_prompt_contains_the_tests(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    make_runner(tmp_path, llm, cfg)[0].run()
    first = llm.calls[0][1]["content"]
    second = llm.calls[1][1]["content"]
    assert "FAIL" in first.upper() and "tests/test_core.py" in first
    assert "assert slugify('Hello, World!') == 'hello-world'" in second


def test_two_steps_two_commits(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK, TEST_EXTRA, EXTRA_OK)
    r, ws = make_runner(tmp_path, llm, cfg, two=True)
    report = r.run()
    assert [x.status for x in report.results] == ["done", "done"]
    assert log_subjects(ws.root) == ["feat: add shout helper", "feat: add slugify core function"]


# ----------------------------------------------------------------- retries

def test_failing_implementation_is_retried_with_the_test_output(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_BAD, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    report = r.run()
    assert report.outcome == "done"
    assert report.results[0].attempts == 3
    retry_prompt = llm.calls[2][1]["content"]
    assert "AssertionError" in retry_prompt               # real pytest output fed back
    assert "return t" in retry_prompt                      # and the code it wrote last time


def test_reply_without_file_blocks_gets_format_feedback_then_recovers(tmp_path, cfg):
    llm = FakeLLM("Sure, I would write some tests for you!", TEST_CORE, INIT + CORE_OK)
    report = make_runner(tmp_path, llm, cfg)[0].run()
    assert report.outcome == "done"
    assert "FILE" in llm.calls[1][1]["content"]
    assert "no FILE blocks" in llm.calls[1][1]["content"] or "missing" in llm.calls[1][1]["content"]


def test_syntax_error_is_caught_before_running_and_not_written(tmp_path, cfg):
    broken = block("slugify/core.py", "def slugify(:\n    pass\n")
    llm = FakeLLM(TEST_CORE, INIT + broken, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert "syntax error" in llm.calls[2][1]["content"].lower()


def test_vacuous_tests_that_pass_without_implementation_are_rejected(tmp_path, cfg):
    llm = FakeLLM(TEST_VACUOUS, TEST_CORE, INIT + CORE_OK)
    report = make_runner(tmp_path, llm, cfg)[0].run()
    assert report.outcome == "done"
    assert "PASS" in llm.calls[1][1]["content"]
    assert llm.calls[2][1]["content"].count("test_nothing") == 0   # replaced tests are used


def test_missing_init_file_is_created_automatically(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, CORE_OK)                     # model forgot slugify/__init__.py
    r, ws = make_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert ws.exists("slugify/__init__.py")


# ----------------------------------------------------------------- safety

def test_files_outside_the_step_are_ignored(tmp_path, cfg):
    sneaky = block("setup.py", "import os\nos.system('echo pwned')\n") + block("../escape.py", "x = 1\n")
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK + sneaky)
    r, ws = make_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert not ws.exists("setup.py")
    assert not (ws.root.parent / "escape.py").exists()


def test_model_cannot_rewrite_the_tests_while_implementing(tmp_path, cfg):
    cheat = block("tests/test_core.py", "def test_basic():\n    assert True\n")
    llm = FakeLLM(TEST_CORE, INIT + CORE_BAD + cheat, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert "slugify('Hello, World!') == 'hello-world'" in ws.read("tests/test_core.py")


def test_failed_step_is_rolled_back_completely(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_BAD, INIT + CORE_BAD, INIT + CORE_BAD)
    r, ws = make_runner(tmp_path, llm, cfg)
    ws.write("slugify/core.py", "# my own earlier work\n")
    report = r.run()
    assert report.outcome == "failed"
    assert report.results[0].status == "failed"
    assert "attempts" in report.results[0].reason
    assert ws.read("slugify/core.py") == "# my own earlier work\n"      # restored, not deleted
    assert not ws.exists("tests/test_core.py")                           # new file removed
    assert not ws.exists("slugify/__init__.py")
    assert log_subjects(ws.root) == [] or "feat: add slugify core function" not in log_subjects(ws.root)


def test_tests_phase_gives_up_after_max_attempts(tmp_path, cfg):
    llm = FakeLLM("nope", "still nope", "no blocks again")
    report = make_runner(tmp_path, llm, cfg)[0].run()
    assert report.outcome == "failed"
    assert "tests" in report.results[0].reason.lower()
    assert len(llm.calls) == cfg.max_attempts


def test_disallowed_test_command_is_refused_up_front(tmp_path, cfg):
    with pytest.raises(PlanError, match="allowed"):
        make_runner(tmp_path, FakeLLM(), cfg, test_command="make test")


def test_already_satisfied_step_skips_implementation(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE)                                # only the tests are requested
    r, ws = make_runner(tmp_path, llm, cfg)
    ws.write("slugify/__init__.py", "")
    ws.write("slugify/core.py", "import re\ndef slugify(t):\n    return re.sub(r'[^a-z0-9]+','-',t.lower()).strip('-')\n")
    report = r.run()
    assert report.outcome == "done" and len(llm.calls) == 1


# ----------------------------------------------------------------- failure policy

def test_stop_on_failure_true_does_not_attempt_later_steps(tmp_path, cfg):
    llm = FakeLLM("no", "no", "no")
    r, ws = make_runner(tmp_path, llm, cfg, two=True)
    report = r.run()
    assert [x.status for x in report.results] == ["failed"]


def test_stop_on_failure_false_continues_with_later_steps(tmp_path, cfg):
    cfg.stop_on_failure = False
    llm = FakeLLM("no", "no", "no", TEST_EXTRA, EXTRA_OK)
    r, ws = make_runner(tmp_path, llm, cfg, two=True)
    report = r.run()
    assert [x.status for x in report.results] == ["failed", "done"]
    assert report.outcome == "failed"


# ----------------------------------------------------------------- pause / resume

def test_model_server_down_pauses_and_rolls_back(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, LLMError("cannot reach the model server"))
    r, ws = make_runner(tmp_path, llm, cfg)
    report = r.run()
    assert report.outcome == "paused"
    assert "cannot reach" in report.results[0].reason
    assert not ws.exists("tests/test_core.py")


def test_resources_unavailable_pauses_without_calling_the_model(tmp_path, cfg):
    llm = FakeLLM()
    r, ws = make_runner(tmp_path, llm, cfg, guard=FakeGuard(ok=False))
    report = r.run()
    assert report.outcome == "paused"
    assert llm.calls == []


def test_stop_request_pauses_before_next_step(tmp_path, cfg):
    llm = FakeLLM()
    r, ws = make_runner(tmp_path, llm, cfg, stop=lambda: True)
    report = r.run()
    assert report.outcome == "paused" and llm.calls == []


def test_resume_skips_steps_that_are_already_done(tmp_path, cfg):
    cfg.stop_on_failure = True
    llm1 = FakeLLM(TEST_CORE, INIT + CORE_OK, "x", "y", "z")      # step 2 fails
    r1, ws = make_runner(tmp_path, llm1, cfg, two=True)
    assert [s.status for s in r1.run().results] == ["done", "failed"]

    llm2 = FakeLLM(TEST_EXTRA, EXTRA_OK)                           # only step 2 needs the model
    r2, _ = make_runner(tmp_path, llm2, cfg, two=True)
    report = r2.run()
    assert [s.status for s in report.results] == ["skipped", "done"]
    assert len(llm2.calls) == 2


def test_cooldown_sleep_happens_after_each_model_call(tmp_path, cfg):
    cfg.step_cooldown_s = 1.5
    slept = []
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    make_runner(tmp_path, llm, cfg, sleep=slept.append)[0].run()
    assert slept == [1.5, 1.5]


def test_second_agent_on_the_same_workspace_is_refused(tmp_path, cfg):
    r, ws = make_runner(tmp_path, FakeLLM(), cfg)
    with WorkspaceLock(ws.root):
        with pytest.raises(WorkspaceBusy):
            r.run()


def test_report_summary_is_readable(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    report = make_runner(tmp_path, llm, cfg)[0].run()
    s = report.summary()
    assert "Slugify utility" in s and "core function" in s and "done" in s.lower()


# ----------------------------------------------------------------- stop while waiting / wrong files

def test_stop_request_during_a_resource_wait_pauses_promptly(tmp_path, cfg):
    stop = {"flag": False}

    class WaitingGuard:
        def wait_until_ok(self, *, should_stop=None, **kw):
            stop["flag"] = True                    # user presses Ctrl+C while we wait
            assert should_stop is not None and should_stop() is True
            return False                           # guard gives up because of the stop

    llm = FakeLLM()
    r, ws = make_runner(tmp_path, llm, cfg, guard=WaitingGuard(), stop=lambda: stop["flag"])
    report = r.run()
    assert report.outcome == "paused"
    assert "stop requested" in report.results[0].reason
    assert llm.calls == []


def test_model_that_writes_the_wrong_files_is_told_exactly_which_to_write(tmp_path, cfg):
    wrong = INIT + CORE_OK                           # wrote the implementation instead of the tests
    llm = FakeLLM(wrong, TEST_CORE, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    feedback = llm.calls[1][1]["content"]
    assert "slugify/core.py" in feedback and "does not allow" in feedback
    assert "ONLY" in feedback and "tests/test_core.py" in feedback


# ----------------------------------------------------------------- reply log + timing

def test_every_raw_reply_is_kept_on_disk_and_timing_is_logged(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    lines = []
    r, ws = make_runner(tmp_path, llm, cfg, log=lines.append)
    assert r.run().outcome == "done"
    d = ws.root / ".codeagent" / "replies"
    names = sorted(p.name for p in d.iterdir())
    assert len(names) == 2
    assert "tests" in names[0] and "impl" in names[1]
    assert "from slugify.core import slugify" in (d / names[0]).read_text()
    assert any("model replied" in l and "s" in l for l in lines)


def test_replies_are_never_committed_to_the_project(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    r.run()
    tracked = subprocess.run(["git", "ls-files"], cwd=ws.root, capture_output=True, text=True).stdout
    assert ".codeagent" not in tracked


# ----------------------------------------------------------------- cut-off / forgotten-fence replies

def test_reply_cut_off_at_the_token_limit_is_salvaged_instead_of_rejected(tmp_path, cfg):
    # exactly the failure seen on a real 0.8B model: the test file never ends
    rambling = ("FILE: tests/test_core.py\n```python\nfrom slugify.core import slugify\n\n"
                "def test_basic():\n    assert slugify('Hello, World!') == 'hello-world'\n\n"
                "def test_again():\n    assert slugify(")
    llm = FakeLLM(Cut(rambling), INIT + CORE_OK)
    lines = []
    r, ws = make_runner(tmp_path, llm, cfg, log=lines.append)
    assert r.run().outcome == "done"
    assert len(llm.calls) == 2                           # no wasted retry
    assert "def test_again" not in ws.read("tests/test_core.py")
    assert any("cut off" in l for l in lines)


def test_unusable_cut_off_reply_gets_a_short_file_instruction(tmp_path, cfg):
    useless = Cut("FILE: tests/test_core.py\n```python\ndef test_x(:\n")
    llm = FakeLLM(useless, TEST_CORE, INIT + CORE_OK)
    assert make_runner(tmp_path, llm, cfg)[0].run().outcome == "done"
    assert "SHORTER" in llm.calls[1][1]["content"]


def test_model_that_forgot_the_closing_fence_is_accepted(tmp_path, cfg):
    no_fence = TEST_CORE.rstrip("\n").removesuffix("```").rstrip("\n") + "\n"
    llm = FakeLLM(no_fence, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert len(llm.calls) == 2


# ----------------------------------------------------------------- retries must not repeat themselves

def test_first_attempt_uses_the_configured_temperature_and_retries_run_hotter(tmp_path, cfg):
    cfg.temperature = 0.2
    llm = FakeLLM("no blocks", "still none", TEST_CORE, INIT + CORE_OK)
    make_runner(tmp_path, llm, cfg)[0].run()
    temps = [k["temperature"] for k in llm.kwargs]
    assert temps[0] == 0.2
    assert temps[1] > temps[0] and temps[2] > temps[1]    # deterministic loops get broken up


def test_retry_temperature_is_capped(tmp_path):
    cfg = Config(step_cooldown_s=0, max_attempts=12, temperature=0.9, test_timeout=60)
    llm = FakeLLM(*["nope"] * 12)
    make_runner(tmp_path, llm, cfg)[0].run()
    assert max(k["temperature"] for k in llm.kwargs) <= 1.0


def test_test_writing_gets_the_smaller_token_budget_and_implementation_the_normal_one(tmp_path, cfg):
    cfg.num_predict, cfg.num_predict_tests = 1500, 600
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    make_runner(tmp_path, llm, cfg)[0].run()
    assert llm.kwargs[0]["num_predict"] == 600            # tests phase
    assert llm.kwargs[1]["num_predict"] == 1500            # implementation phase


def test_test_budget_never_exceeds_the_general_budget(tmp_path, cfg):
    cfg.num_predict, cfg.num_predict_tests = 300, 900
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    make_runner(tmp_path, llm, cfg)[0].run()
    assert llm.kwargs[0]["num_predict"] == 300


def test_the_temperature_actually_used_is_recorded_with_the_saved_reply(tmp_path, cfg):
    cfg.temperature = 0.2
    llm = FakeLLM("garbage", TEST_CORE, INIT + CORE_OK)
    r, ws = make_runner(tmp_path, llm, cfg)
    r.run()
    names = sorted(p.name for p in (ws.root / ".codeagent" / "replies").iterdir())
    second = (ws.root / ".codeagent" / "replies" / names[1]).read_text()
    assert "attempt: 2" in second and "temperature: 0.25" in second


# ----------------------------------------------------------------- broken tests are sent back to the test phase

TEST_PARAMS = block("tests/test_core.py",
                    "from slugify.core import slugify\n\n"
                    "def test_basic(text: str) -> None:\n    assert slugify(text) == 'hello-world'\n")
TEST_FIXTURE_MISSING = block("tests/test_core.py",
                             "import pytest\n\n@pytest.fixture\ndef a(missing):\n    return 1\n\n"
                             "def test_x(a):\n    assert a == 2\n")
TEST_FIXTURE_MISSING_LATE = block("tests/test_core.py",
                                  "import pytest\nfrom slugify.core import slugify\n\n"
                                  "@pytest.fixture\ndef a(missing):\n    return 1\n\n"
                                  "def test_x(a):\n    assert slugify('A') == 'a'\n")


def test_tests_with_parameters_nothing_provides_are_rejected_before_running(tmp_path, cfg):
    llm = FakeLLM(TEST_PARAMS, TEST_CORE, INIT + CORE_OK)
    lines = []
    report = make_runner(tmp_path, llm, cfg, log=lines.append)[0].run()
    assert report.outcome == "done"
    assert len(llm.calls) == 3                            # no implementation call wasted on bad tests
    assert "NO parameters" in llm.calls[1][1]["content"]
    assert any("tests attempt 1" in l and "text" in l for l in lines)


def test_tests_that_are_red_because_pytest_cannot_run_them_are_rewritten(tmp_path, cfg):
    llm = FakeLLM(TEST_FIXTURE_MISSING, TEST_CORE, INIT + CORE_OK)
    report = make_runner(tmp_path, llm, cfg)[0].run()
    assert report.outcome == "done"
    assert len(llm.calls) == 3
    assert "fixture 'missing'" in llm.calls[1][1]["content"]


def test_tests_found_broken_while_implementing_are_rewritten_not_retried(tmp_path, cfg):
    llm = FakeLLM(TEST_FIXTURE_MISSING_LATE, INIT + CORE_OK, TEST_CORE, INIT + CORE_OK)
    lines = []
    r, ws = make_runner(tmp_path, llm, cfg, log=lines.append)
    report = r.run()
    assert report.outcome == "done"
    assert len(llm.calls) == 4                            # one impl try, not max_attempts of them
    assert "fixture 'missing'" in llm.calls[2][1]["content"]
    assert ws.read("tests/test_core.py").startswith("from slugify.core")
    assert any("rewriting the tests" in l for l in lines)


def test_tests_that_stay_broken_fail_the_step_and_roll_back(tmp_path, cfg):
    cfg.max_test_rewrites = 1
    llm = FakeLLM(*[TEST_FIXTURE_MISSING_LATE, INIT + CORE_OK] * 2)
    r, ws = make_runner(tmp_path, llm, cfg)
    report = r.run()
    assert report.outcome == "failed"
    assert "tests are broken" in report.results[0].reason
    assert not ws.exists("slugify/core.py") and not ws.exists("tests/test_core.py")


def test_zero_test_rewrites_fails_immediately(tmp_path, cfg):
    cfg.max_test_rewrites = 0
    llm = FakeLLM(TEST_FIXTURE_MISSING_LATE, INIT + CORE_OK)
    assert make_runner(tmp_path, llm, cfg)[0].run().outcome == "failed"
    assert len(llm.calls) == 2


# ----------------------------------------------------------------- retries stay calm

def test_default_retries_never_run_hot(tmp_path):
    cfg = Config(step_cooldown_s=0, max_attempts=12, test_timeout=60)
    llm = FakeLLM(*["nope"] * 12)
    make_runner(tmp_path, llm, cfg)[0].run()
    temps = [k["temperature"] for k in llm.kwargs]
    assert temps[0] == 0.2 and max(temps) <= 0.4


def test_retry_temperature_step_and_cap_are_configurable(tmp_path):
    cfg = Config(step_cooldown_s=0, max_attempts=5, temperature=0.1, retry_temperature_step=0.1,
                 max_retry_temperature=0.25, test_timeout=60)
    llm = FakeLLM(*["nope"] * 5)
    make_runner(tmp_path, llm, cfg)[0].run()
    assert [round(k["temperature"], 2) for k in llm.kwargs] == [0.1, 0.2, 0.25, 0.25, 0.25]


def test_system_prompt_forbids_comments_and_docstrings_and_tests_must_take_no_parameters(tmp_path, cfg):
    llm = FakeLLM(TEST_CORE, INIT + CORE_OK)
    make_runner(tmp_path, llm, cfg)[0].run()
    assert "No comments and no docstrings" in llm.calls[0][0]["content"]
    assert "NO parameters" in llm.calls[0][1]["content"]


# ----------------------------------------------------------------- the plan's examples become the tests

EXAMPLES_PLAN = '''
title = "Slugify utility"
test_command = "python -m pytest -q"

[[step]]
title = "core function"
description = "Create slugify(text) in slugify/core.py."
commit = "feat: add slugify core function"
tests = ["tests/test_core.py"]
files = ["slugify/__init__.py", "slugify/core.py"]
function = "slugify.core:slugify"

[[step.examples]]
args = ["Hello, World!"]
expect = "hello-world"

[[step.examples]]
args = ["  A  B "]
expect = "a-b"

[[step.examples]]
args = [""]
expect = ""
'''


def make_examples_runner(tmp_path, llm, cfg, log=None):
    plan = parse_plan(EXAMPLES_PLAN)
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    return Runner(cfg, plan, ws, llm, guard=FakeGuard(), log=log or (lambda m: None),
                  sleep=lambda s: None, should_stop=lambda: False), ws


def test_examples_are_turned_into_tests_without_asking_the_model(tmp_path, cfg):
    llm = FakeLLM(INIT + CORE_OK)
    lines = []
    r, ws = make_examples_runner(tmp_path, llm, cfg, log=lines.append)
    assert r.run().outcome == "done"
    assert len(llm.calls) == 1                                  # only the implementation was generated
    tests = ws.read("tests/test_core.py")
    assert "from slugify.core import slugify" in tests and "== 'hello-world'" in tests
    assert any("generated from 3 example(s)" in l for l in lines)


def test_model_never_sees_a_request_to_write_tests_when_examples_exist(tmp_path, cfg):
    llm = FakeLLM(INIT + CORE_OK)
    make_examples_runner(tmp_path, llm, cfg)[0].run()
    assert "def slugify(...)" in llm.calls[0][1]["content"]
    assert "Do NOT change the tests" in llm.calls[0][1]["content"]


def test_wrong_implementation_is_retried_against_the_same_generated_tests(tmp_path, cfg):
    llm = FakeLLM(INIT + CORE_BAD, INIT + CORE_OK)
    r, ws = make_examples_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert len(llm.calls) == 2
    assert "failed" in llm.calls[1][1]["content"]               # the real pytest output goes back to the model


def test_implementation_that_does_not_define_the_function_is_sent_back_before_running(tmp_path, cfg):
    wrong = block("slugify/core.py", "def make_slug(t):\n    return t\n")
    llm = FakeLLM(INIT + wrong, INIT + CORE_OK)
    r, ws = make_examples_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert "does not define `slugify`" in llm.calls[1][1]["content"]


def test_step_whose_implementation_already_passes_the_examples_is_skipped(tmp_path, cfg):
    llm = FakeLLM()
    r, ws = make_examples_runner(tmp_path, llm, cfg)
    ws.write("slugify/__init__.py", "")
    ws.write("slugify/core.py", "import re\n\ndef slugify(t):\n    return re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')\n")
    r.run()                                                     # already implemented: nothing to do
    assert llm.calls == []
