from codeagent.parsing import FileBlock, parse_file_blocks, strip_thinking


def blocks(text):
    return parse_file_blocks(text).blocks


def test_single_block():
    r = parse_file_blocks("FILE: a.py\n```python\nx = 1\n```\n")
    assert r.blocks == [FileBlock("a.py", "x = 1\n")]
    assert r.problems == []


def test_multiple_blocks_with_prose_around():
    text = "Sure! Here you go.\n\nFILE: a.py\n```python\nA = 1\n```\n\nand\n\nFILE: pkg/b.py\n```\nB = 2\n```\nDone."
    assert [b.path for b in blocks(text)] == ["a.py", "pkg/b.py"]
    assert blocks(text)[1].content == "B = 2\n"


def test_header_variants():
    for header in ["### FILE: a.py", "**FILE: a.py**", "File: `a.py`", "file: a.py", "PATH: a.py", "  FILE:   a.py  "]:
        assert [b.path for b in blocks(f"{header}\n```python\nx\n```")] == ["a.py"], header


def test_dot_slash_prefix_is_stripped():
    assert blocks("FILE: ./a.py\n```\nx\n```")[0].path == "a.py"


def test_blank_line_between_header_and_fence_is_ok():
    assert blocks("FILE: a.py\n\n```python\nx\n```")[0].content == "x\n"


def test_fence_without_header_but_with_comment_path():
    r = parse_file_blocks("```python\n# FILE: a.py\nx = 1\n```")
    assert r.blocks == [FileBlock("a.py", "x = 1\n")]


def test_anonymous_fence_is_ignored():
    assert blocks("```python\nprint('hi')\n```") == []


def test_header_text_inside_code_does_not_start_new_block():
    text = 'FILE: a.py\n```python\ns = """\nFILE: b.py\n"""\n```'
    assert [b.path for b in blocks(text)] == ["a.py"]


def test_truncated_block_is_reported_and_excluded():
    r = parse_file_blocks("FILE: a.py\n```python\nA = 1\n```\nFILE: b.py\n```python\ndef f(:\n")
    assert [b.path for b in r.blocks] == ["a.py"]
    assert any("b.py" in p and "truncated" in p for p in r.problems)


def test_duplicate_paths_last_one_wins():
    r = blocks("FILE: a.py\n```\n1\n```\nFILE: a.py\n```\n2\n```")
    assert [(b.path, b.content) for b in r] == [("a.py", "2\n")]


def test_crlf_line_endings():
    assert blocks("FILE: a.py\r\n```python\r\nx = 1\r\n```\r\n")[0].content == "x = 1\n"


def test_empty_block_gives_empty_content():
    assert blocks("FILE: __init__.py\n```python\n```")[0].content == ""


def test_no_blocks():
    r = parse_file_blocks("I cannot do that.")
    assert r.blocks == [] and r.problems == []


def test_thinking_is_stripped():
    assert strip_thinking("<think>plan...</think>\nanswer") == "answer"
    assert strip_thinking("<think>\nmulti\nline\n</think>FILE: a") == "FILE: a"
    assert strip_thinking("no tags") == "no tags"


def test_unclosed_thinking_drops_the_rest():
    assert strip_thinking("<think>rambling forever") == ""


def test_stray_closing_tag_keeps_text_after_it():
    assert strip_thinking("hidden reasoning</think>\nreal answer") == "real answer"


def test_file_block_inside_thinking_is_ignored():
    text = "<think>FILE: x.py\n```\nbad\n```</think>\nFILE: a.py\n```\nok\n```"
    assert [b.path for b in blocks(text)] == ["a.py"]
