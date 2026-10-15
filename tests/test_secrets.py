import os
import subprocess

import pytest

from agent import (GITIGNORE, GitError, Secrets, blocked_name, commit_paths, ensure_repo, load_secrets, parse_dotenv,
                   scan_files, scan_history, scan_text)

FAKE_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2          # looks like a classic GitHub token; not a real one


def test_parse_dotenv_handles_comments_quotes_and_export():
    text = "# comment\nGITHUB_TOKEN=abc123\nexport GIT_AUTHOR_NAME=\"Om Yadao\"\nX='single'  \nY=plain # trailing\nbad line\n=novalue\n"
    assert parse_dotenv(text) == {"GITHUB_TOKEN": "abc123", "GIT_AUTHOR_NAME": "Om Yadao", "X": "single", "Y": "plain"}


def test_load_secrets_priority_explicit_file_then_environment_then_folder_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".codeagent.env").write_text("GITHUB_TOKEN=from_home\nGITHUB_USER=homeuser\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".env").write_text("GITHUB_TOKEN=from_project\n")
    monkeypatch.chdir(tmp_path)
    s = load_secrets(project_dir=proj, environ={}, home=home)
    assert s.token == "from_project" and s.get("GITHUB_USER") == "homeuser"
    assert load_secrets(project_dir=proj, environ={"GITHUB_TOKEN": "from_env"}, home=home).token == "from_env"
    explicit = tmp_path / "x.env"
    explicit.write_text("GITHUB_TOKEN=from_explicit\n")
    assert load_secrets(explicit, project_dir=proj, environ={"GITHUB_TOKEN": "from_env"}, home=home).token == "from_explicit"
    assert load_secrets(environ={}, home=tmp_path / "nohome").token is None


def test_the_token_is_never_put_into_the_process_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GITHUB_TOKEN=supersecrettoken123\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    load_secrets(environ={}, home=tmp_path / "nohome")
    assert "GITHUB_TOKEN" not in os.environ


@pytest.mark.skipif(os.name != "posix", reason="file modes")
def test_a_world_readable_env_file_gets_a_warning(tmp_path):
    f = tmp_path / ".codeagent.env"
    f.write_text("GITHUB_TOKEN=abcdefgh1234\n")
    f.chmod(0o644)
    assert any("chmod 600" in w for w in load_secrets(environ={}, home=tmp_path).warnings)
    f.chmod(0o600)
    assert not load_secrets(environ={}, home=tmp_path).warnings


def test_scrub_removes_the_token_from_messages():
    s = Secrets(values={"GITHUB_TOKEN": "supersecrettoken123"})
    assert s.scrub("push failed for supersecrettoken123 at x") == "push failed for *** at x"
    assert Secrets().scrub("nothing to hide") == "nothing to hide"


@pytest.mark.parametrize("text", [
    FAKE_TOKEN,
    "token = 'github_pat_" + "A1b2C3d4" * 6 + "'",
    "AKIAABCDEFGHIJKLMNOP",
    "-----BEGIN RSA PRIVATE KEY-----",
    "key = 'sk-ant-" + "abcdef0123456789" * 2 + "'",
    'password = "hunter2hunter2hunter2x1"',
])
def test_secret_looking_text_is_found(text):
    assert scan_text(text)


@pytest.mark.parametrize("text", [
    "def check_args(price, quantity):\n    return True",
    "password = 'short'",
    "token = 'abcdefghijklmnopqrstuvwxyz'",                  # long but no digit: not secret-like
    "ghp_ is the prefix of a GitHub token",
    "x = 'sk-learn'",
])
def test_ordinary_code_is_not_flagged(text):
    assert scan_text(text) == []


def test_the_users_own_token_is_found_wherever_it_hides():
    assert scan_text("note = 'use abcdef123456 here'", literals=("abcdef123456",)) == [(1, "your own token")]


@pytest.mark.parametrize("name,blocked", [
    (".env", True), ("config/.env", True), (".env.local", True), (".env.example", False), ("server.pem", True),
    ("id_rsa", True), ("credentials.json", True), (".npmrc", True), ("app/models.py", False), ("environment.py", False),
])
def test_file_names_that_hold_secrets_are_blocked(name, blocked):
    assert blocked_name(name) is blocked


def test_scan_files_reports_file_and_line_but_not_the_secret(tmp_path):
    (tmp_path / "a.py").write_text(f"x = 1\nTOKEN = '{FAKE_TOKEN}'\n")
    (tmp_path / ".env").write_text("A=1\n")
    problems = scan_files(tmp_path, ["a.py", ".env", "missing.py"])
    assert any(p.startswith("a.py:2:") for p in problems) and any(p.startswith(".env:") for p in problems)
    assert all(FAKE_TOKEN not in p for p in problems)


def test_a_commit_with_a_secret_is_refused_and_nothing_is_committed(tmp_path):
    ensure_repo(tmp_path)
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text(f"T = '{FAKE_TOKEN}'\n")
    (tmp_path / ".env").write_text("GITHUB_TOKEN=whatever12345\n")
    assert commit_paths(tmp_path, ["ok.py"], "feat: ok")
    with pytest.raises(GitError, match="looks like a secret"):
        commit_paths(tmp_path, ["bad.py"], "feat: bad")
    with pytest.raises(GitError, match="secrets"):
        commit_paths(tmp_path, [".env"], "feat: env")
    log = subprocess.run(["git", "log", "--format=%s"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert log.split() == ["feat:", "ok"]


def test_scan_history_finds_a_secret_that_was_committed_and_deleted_again(tmp_path):
    ensure_repo(tmp_path)
    for cmd in (["git", "config", "user.name", "t"], ["git", "config", "user.email", "t@t"]):
        subprocess.run(cmd, cwd=tmp_path, check=True)
    (tmp_path / "x.py").write_text(f"T = '{FAKE_TOKEN}'\n")
    subprocess.run(["git", "add", "x.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "oops"], cwd=tmp_path, check=True)
    (tmp_path / "x.py").write_text("T = None\n")
    subprocess.run(["git", "commit", "-qam", "fix"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("A=1\n")
    subprocess.run(["git", "add", "-f", ".env"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "env"], cwd=tmp_path, check=True)
    found = scan_history(tmp_path)
    assert any("x.py" in f and "history" in f for f in found) and any(f.startswith(".env") for f in found)
    assert all(FAKE_TOKEN not in f for f in found)


def test_a_clean_history_has_no_findings(tmp_path):
    ensure_repo(tmp_path)
    (tmp_path / "x.py").write_text("x = 1\n")
    assert commit_paths(tmp_path, ["x.py"], "feat: x")
    assert scan_history(tmp_path) == []


def test_the_gitignore_keeps_env_and_keys_out():
    for line in (".env", ".env.*", "*.pem", "id_rsa*", ".codeagent/", "__pycache__/"):
        assert line in GITIGNORE.splitlines()
