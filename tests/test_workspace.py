import os

import pytest

from codeagent.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "proj", max_file_bytes=1000)


def test_root_is_created(tmp_path):
    w = Workspace(tmp_path / "a" / "b", max_file_bytes=1000)
    assert w.root.is_dir()


def test_write_then_read_roundtrip_and_creates_parents(ws):
    ws.write("pkg/sub/mod.py", "x = 1\n")
    assert ws.read("pkg/sub/mod.py") == "x = 1\n"
    assert (ws.root / "pkg" / "sub" / "mod.py").is_file()


def test_read_missing_returns_none(ws):
    assert ws.read("nope.py") is None
    assert ws.exists("nope.py") is False


def test_overwrite(ws):
    ws.write("a.py", "1\n")
    ws.write("a.py", "2\n")
    assert ws.read("a.py") == "2\n"


@pytest.mark.parametrize(
    "bad", ["/etc/passwd", "../x.py", "a/../../x.py", ".git/config", ".codeagent/state.json", "", "  "]
)
def test_unsafe_paths_are_rejected(ws, bad):
    with pytest.raises(WorkspaceError):
        ws.write(bad, "x")
    with pytest.raises(WorkspaceError):
        ws.read(bad)


def test_symlink_escape_is_rejected(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, ws.root / "link")
    with pytest.raises(WorkspaceError, match="outside"):
        ws.write("link/evil.py", "x")
    assert not (outside / "evil.py").exists()


def test_size_cap_on_write(ws):
    with pytest.raises(WorkspaceError, match="too large"):
        ws.write("big.py", "x" * 1001)
    assert not ws.exists("big.py")


def test_size_cap_on_read(ws):
    (ws.root / "big.txt").write_text("x" * 1001)
    with pytest.raises(WorkspaceError, match="too large"):
        ws.read("big.txt")


def test_binary_file_cannot_be_read_as_text(ws):
    (ws.root / "b.bin").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(WorkspaceError, match="not valid UTF-8"):
        ws.read("b.bin")


def test_write_leaves_no_temp_files(ws):
    ws.write("a.py", "1\n")
    assert sorted(p.name for p in ws.root.iterdir()) == ["a.py"]


def test_delete(ws):
    ws.write("a.py", "1\n")
    ws.delete("a.py")
    assert not ws.exists("a.py")
    ws.delete("a.py")  # deleting a missing file is fine


def test_snapshot_and_restore_undo_everything(ws):
    ws.write("keep.py", "original\n")
    ws.write("other.py", "untouched\n")
    snap = ws.snapshot(["keep.py", "new.py"])
    assert snap == {"keep.py": "original\n", "new.py": None}
    ws.write("keep.py", "changed\n")
    ws.write("new.py", "created\n")
    ws.restore(snap)
    assert ws.read("keep.py") == "original\n"
    assert not ws.exists("new.py")
    assert ws.read("other.py") == "untouched\n"
