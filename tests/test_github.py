import subprocess

import pytest

from agent import (GitHub, GitHubError, GitError, Secrets, commit_paths, ensure_repo, preflight_push, push_ref,
                   remote_url, repo_name, set_identity)
from tests.fake_github import FakeGitHub

FAKE_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2


@pytest.fixture
def gh_server(tmp_path):
    with FakeGitHub(tmp_path / "remote") as srv:
        yield srv


def client(srv, token=None):
    return GitHub(token or srv.token, srv.api_url)


def test_whoami_returns_login_name_scopes_and_a_noreply_email_when_none_is_public(gh_server):
    me = client(gh_server).whoami()
    assert me["login"] == "octo" and me["name"] == "Octo Cat" and me["scopes"] == ["repo"]
    assert me["email"] == "4242+octo@users.noreply.github.com"
    gh_server.email = "octo@example.com"
    assert client(gh_server).whoami()["email"] == "octo@example.com"


def test_a_bad_token_and_an_unreachable_server_give_clear_errors(gh_server):
    with pytest.raises(GitHubError, match="rejected the token"):
        client(gh_server, "wrong").whoami()
    with pytest.raises(GitHubError, match="cannot reach GitHub"):
        GitHub("x", "http://127.0.0.1:1", timeout=2).whoami()


def test_the_repository_is_created_private_and_without_auto_init(gh_server):
    data = client(gh_server).create_repo("expense-manager", "A tracker")
    assert data["full_name"] == "octo/expense-manager" and data["private"] is True
    method, path, body = gh_server.requests[-1]
    assert (method, path) == ("POST", "/user/repos")
    assert body == {"name": "expense-manager", "description": "A tracker", "private": True, "auto_init": False}


def test_a_taken_name_gets_a_numbered_suffix_and_never_touches_the_existing_repo(gh_server):
    c = client(gh_server)
    names = [c.create_unique_repo("todo")["name"] for _ in range(3)]
    assert names == ["todo", "todo-2", "todo-3"]


def test_a_token_without_the_repo_scope_is_explained(gh_server):
    gh_server.refuse_create = 1
    with pytest.raises(GitHubError, match="`repo` scope"):
        client(gh_server).create_repo("x")


@pytest.mark.parametrize("name,expected", [
    ("Expense Manager", "expense-manager"), ("  my_app!! ", "my_app"), ("todo.cli", "todo.cli"), ("###", "project"),
    ("a" * 200, "a" * 90),
])
def test_repo_names_are_clean(name, expected):
    assert repo_name(name) == expected


def test_commits_are_made_as_the_user_in_local_config_only(tmp_path):
    ensure_repo(tmp_path)
    set_identity(tmp_path, "Om Yadao", "1+Yad4o@users.noreply.github.com")
    (tmp_path / "a.py").write_text("x = 1\n")
    assert commit_paths(tmp_path, ["a.py"], "feat: a")
    who = subprocess.run(["git", "log", "-1", "--format=%an <%ae> | %cn <%ce>"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert who == "Om Yadao <1+Yad4o@users.noreply.github.com> | Om Yadao <1+Yad4o@users.noreply.github.com>"
    assert subprocess.run(["git", "config", "--local", "commit.gpgsign"], cwd=tmp_path, capture_output=True, text=True).stdout.strip() == "false"


def test_push_sends_the_branch_to_the_remote_without_leaving_the_token_anywhere(gh_server, tmp_path):
    repo = client(gh_server).create_repo("proj")
    work = tmp_path / "work"
    work.mkdir()
    ensure_repo(work)
    set_identity(work, "Octo Cat", "o@c")
    (work / "a.py").write_text("x = 1\n")
    commit_paths(work, ["a.py"], "feat: a")
    secrets = Secrets(values={"GITHUB_TOKEN": gh_server.token, "GITHUB_GIT_URL_TEMPLATE": gh_server.git_template})
    url = remote_url(secrets, "octo", "proj")
    push_ref(work, url, "main", secrets, "octo")
    assert gh_server.branches("proj") == ["main"]
    assert gh_server.token not in (work / ".git" / "config").read_text()
    assert gh_server.token not in url and repo["private"]


def test_a_failed_push_never_prints_the_token(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ensure_repo(work)
    set_identity(work, "a", "a@a")
    (work / "a.py").write_text("x = 1\n")
    commit_paths(work, ["a.py"], "feat: a")
    secrets = Secrets(values={"GITHUB_TOKEN": "supersecrettoken123456"})
    with pytest.raises(GitError) as err:
        push_ref(work, str(tmp_path / "does-not-exist.git"), "main", secrets, "octo")
    assert "supersecrettoken123456" not in str(err.value)


def test_preflight_refuses_a_history_with_a_secret(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ensure_repo(work)
    set_identity(work, "a", "a@a")
    (work / "a.py").write_text("x = 1\n")
    commit_paths(work, ["a.py"], "feat: a")
    preflight_push(work)
    (work / "a.py").write_text(f"T = '{FAKE_TOKEN}'\n")
    subprocess.run(["git", "commit", "-qam", "oops"], cwd=work, check=True)
    with pytest.raises(GitError, match="not pushing"):
        preflight_push(work)
