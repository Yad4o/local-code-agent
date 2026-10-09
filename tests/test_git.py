import os
import subprocess

import pytest

import agent


def sh(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(autouse=True)
def no_user_git_config(tmp_path_factory, monkeypatch):
    """Isolate from the developer's real git config (identity, signing, hooks)."""
    empty = tmp_path_factory.mktemp("home") / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("HOME", str(empty.parent))


@pytest.fixture
def repo(tmp_path):
    agent.ensure_repo(tmp_path)
    return tmp_path


def test_ensure_repo_initialises_on_main(tmp_path):
    assert agent.ensure_repo(tmp_path) is True
    assert (tmp_path / ".git").is_dir()
    assert agent.current_branch(tmp_path) == "main"


def test_ensure_repo_is_idempotent(tmp_path):
    agent.ensure_repo(tmp_path)
    assert agent.ensure_repo(tmp_path) is False


def test_codeagent_dir_is_excluded_without_touching_gitignore(repo):
    (repo / ".codeagent").mkdir()
    (repo / ".codeagent" / "state.json").write_text("{}")
    assert ".codeagent/" in (repo / ".git" / "info" / "exclude").read_text()
    assert sh(repo, "status", "--porcelain") == ""
    assert not (repo / ".gitignore").exists()
    agent.ensure_repo(repo)                     # no duplicate line
    assert (repo / ".git" / "info" / "exclude").read_text().count(".codeagent/") == 1


def test_test_run_noise_is_excluded_too(repo):
    (repo / "pkg" / "__pycache__").mkdir(parents=True)
    (repo / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"0")
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "v").write_text("1")
    assert sh(repo, "status", "--porcelain") == ""


def test_checkout_branch_creates_then_reuses(repo):
    agent.checkout_branch(repo, "agent/my-plan")
    assert agent.current_branch(repo) == "agent/my-plan"
    (repo / "a.txt").write_text("1")
    agent.commit_paths(repo, ["a.txt"], "feat: a")
    sh(repo, "checkout", "-q", "-b", "other")
    agent.checkout_branch(repo, "agent/my-plan")       # exists: just switch
    assert agent.current_branch(repo) == "agent/my-plan"


def test_commit_paths_commits_only_the_listed_files(repo):
    (repo / "mine.py").write_text("x = 1\n")
    (repo / "other.py").write_text("y = 2\n")
    sha = agent.commit_paths(repo, ["mine.py"], "feat: add mine")
    assert sha and len(sha) >= 7
    assert sh(repo, "log", "-1", "--format=%s") == "feat: add mine"
    assert sh(repo, "show", "--name-only", "--format=", "HEAD") == "mine.py"
    assert "other.py" in sh(repo, "status", "--porcelain")      # left alone


def test_commit_paths_ignores_other_staged_changes(repo):
    (repo / "mine.py").write_text("1")
    (repo / "staged.py").write_text("2")
    sh(repo, "add", "staged.py")
    agent.commit_paths(repo, ["mine.py"], "feat: mine")
    assert sh(repo, "show", "--name-only", "--format=", "HEAD") == "mine.py"


def test_commit_paths_returns_none_when_nothing_changed(repo):
    (repo / "a.py").write_text("1")
    assert agent.commit_paths(repo, ["a.py"], "feat: a")
    assert agent.commit_paths(repo, ["a.py"], "feat: a again") is None
    assert sh(repo, "rev-list", "--count", "HEAD") == "1"


def test_commit_works_without_any_git_identity(repo):
    (repo / "a.py").write_text("1")
    agent.commit_paths(repo, ["a.py"], "feat: a")
    assert sh(repo, "log", "-1", "--format=%an") == "codeagent"


def test_configured_identity_is_respected(repo):
    sh(repo, "config", "user.name", "Real Person")
    sh(repo, "config", "user.email", "real@example.com")
    (repo / "a.py").write_text("1")
    agent.commit_paths(repo, ["a.py"], "feat: a")
    assert sh(repo, "log", "-1", "--format=%an <%ae>") == "Real Person <real@example.com>"


def test_gpg_signing_cannot_hang_an_unattended_commit(repo):
    sh(repo, "config", "commit.gpgsign", "true")
    sh(repo, "config", "gpg.program", "/bin/false")
    (repo / "a.py").write_text("1")
    assert agent.commit_paths(repo, ["a.py"], "feat: a")


def test_user_hooks_do_not_run(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    (repo / "a.py").write_text("1")
    assert agent.commit_paths(repo, ["a.py"], "feat: a")


def test_errors_are_wrapped(tmp_path):
    with pytest.raises(agent.GitError):
        agent.commit_paths(tmp_path / "missing", ["a.py"], "feat: x")
