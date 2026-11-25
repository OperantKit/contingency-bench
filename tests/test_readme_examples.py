"""Execute every ```python block in the READMEs and compare what it prints with its comments.

Blocks of one README run in order in a shared namespace (later blocks reuse the
imports of earlier ones) inside a temporary directory (examples write files).
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READMES = ("README.md", "README.ja.md")
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks(name: str) -> list[str]:
    return _BLOCK.findall((ROOT / name).read_text(encoding="utf-8"))


def _documented_output(code: str) -> list[str]:
    """Output lines a README block documents in the comments of its ``print`` lines."""

    def clean(comment: str) -> str:
        return comment.split(" — ")[0].rstrip()  # drop a trailing " — note"

    expected: list[str] = []
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("print("):
            continue
        _, same_line, comment = line.partition("  # ")
        if same_line:  # print(x)  # X
            expected.append(clean(comment))
            continue
        for nxt in lines[i + 1 :]:  # comment lines directly below the print
            if nxt.startswith("#  ") and expected:  # "#" + 2 spaces: continues the line above
                expected[-1] += " " + clean(nxt[1:].strip())
            elif nxt.startswith("# "):
                expected.append(clean(nxt[2:]))
            else:
                break
    return expected


@pytest.mark.parametrize("name", READMES)
def test_readme_examples_run(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _blocks(name)
    assert blocks
    expected = [line for block in blocks for line in _documented_output(block)]
    assert expected, f"{name} documents no output"
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for index, block in enumerate(blocks):
            exec(compile(block, f"{name}#{index}", "exec"), namespace)
    assert out.getvalue().splitlines() == expected
