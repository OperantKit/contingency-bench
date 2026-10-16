"""Execute every ```python block in the READMEs so the documented examples keep working.

Blocks of one README run in order in a shared namespace (later blocks reuse the
imports of earlier ones) inside a temporary directory (examples write files).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READMES = ("README.md", "README.ja.md")
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks(name: str) -> list[str]:
    return _BLOCK.findall((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", READMES)
def test_readme_examples_run(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _blocks(name)
    assert blocks
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        exec(compile(block, f"{name}#{index}", "exec"), namespace)
