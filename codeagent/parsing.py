"""Parse the model's reply. A tiny model follows a simple, rigid format far
better than JSON/tool-calls:

    FILE: path/to/file.py
    ```python
    ...code...
    ```
"""
import ast
import re
from dataclasses import dataclass, field

HEADER = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:file|path)\s*:\s*`?([^`*\n]+?)`?(?:\*\*)?\s*$", re.I
)
INLINE_PATH = re.compile(r"^\s*(?:#|//|--|/\*)\s*(?:file|path)\s*:\s*(\S+?)\s*(?:\*/)?\s*$", re.I)
FENCE_OPEN = re.compile(r"^\s*```[\w+.\-]*\s*$")
FENCE_CLOSE = re.compile(r"^\s*```\s*$")


@dataclass(frozen=True)
class FileBlock:
    path: str
    content: str


@dataclass
class ParseResult:
    blocks: list[FileBlock] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # things we quietly fixed (worth logging)
    anonymous: list[str] = field(default_factory=list)  # fenced blocks that came without any file name


def strip_thinking(text: str) -> str:
    """Remove Qwen-style <think>...</think> reasoning (closed, unclosed or half-open)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:                       # opening tag was cut off
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:                        # reasoning never finished
        text = text.split("<think>", 1)[0]
    return text.strip()


def _norm_path(p: str) -> str:
    p = p.strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _longest_valid_python_prefix(lines: list[str]) -> list[str] | None:
    """Drop trailing lines until what is left parses as Python (None if nothing does)."""
    for k in range(len(lines), 0, -1):
        chunk = lines[:k]
        if not any(l.strip() for l in chunk):
            return None
        try:
            ast.parse("\n".join(chunk))
        except SyntaxError:
            continue
        while chunk and not chunk[-1].strip():
            chunk = chunk[:-1]
        return chunk
    return None


def parse_file_blocks(text: str, *, truncated: bool = False) -> ParseResult:
    """`truncated=True` means the model hit its token limit, so an unclosed block was CUT OFF
    (repair it if it is Python). Otherwise an unclosed block just means it forgot the fence."""
    result = ParseResult()
    found: dict[str, str] = {}
    pending: str | None = None      # path waiting for its opening fence
    collecting = False              # inside a fenced block
    block_path: str | None = None   # path for the block being collected (None = anonymous)
    lines_acc: list[str] = []

    def finish(path: str | None, lines: list[str]) -> None:
        if path is None and lines:
            m = INLINE_PATH.match(lines[0])
            if m:
                path, lines = m.group(1), lines[1:]
        if path is None:
            if any(l.strip() for l in lines):
                result.anonymous.append("".join(l + "\n" for l in lines))
            return
        found[_norm_path(path)] = "".join(l + "\n" for l in lines)

    for line in strip_thinking(text).replace("\r\n", "\n").split("\n"):
        if collecting:
            if FENCE_CLOSE.match(line):
                finish(block_path, lines_acc)
                collecting, block_path, lines_acc = False, None, []
            else:
                lines_acc.append(line)
            continue
        if pending is not None:
            if not line.strip():
                continue
            if FENCE_OPEN.match(line):
                collecting, block_path, lines_acc, pending = True, pending, [], None
                continue
            pending = None                       # header without a fence: forget it
        m = HEADER.match(line)
        if m:
            pending = m.group(1)
        elif FENCE_OPEN.match(line):
            collecting, block_path, lines_acc = True, None, []

    if collecting and block_path is None and not truncated:
        finish(None, lines_acc)                  # an unclosed block with no name: still counts as anonymous
    if collecting and block_path is not None:
        path = _norm_path(block_path)
        if not truncated:
            finish(block_path, lines_acc)
            result.notes.append(f"{path}: the closing ``` was missing; accepted the code anyway")
        else:
            kept = _longest_valid_python_prefix(lines_acc) if path.endswith(".py") else None
            if kept:
                found[path] = "".join(l + "\n" for l in kept)
                result.notes.append(f"{path}: reply was cut off at the token limit; kept the first "
                                    f"{len(kept)} of {len(lines_acc)} lines that form valid Python")
            else:
                result.problems.append(
                    f"{path}: your reply was cut off because it was too long. Write a much SHORTER file: "
                    "no repeated code, only a few small functions, and close the code block with ```")

    result.blocks = [FileBlock(p, c) for p, c in found.items()]
    return result
