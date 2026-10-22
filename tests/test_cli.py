"""Tests for the CLI defaults."""

from __future__ import annotations

import re
from pathlib import Path

from contingency_bench import benchmark, cli

# APILLMClient sends the name unchanged to the Anthropic Messages API, so the
# defaults must be Anthropic model IDs (dated snapshot or current alias).
_ANTHROPIC_ID = re.compile(r"^claude-(haiku|sonnet|opus|fable)-\d+(-\d+)?(-\d{8})?$")


def test_default_models_are_anthropic_ids() -> None:
    assert cli.DEFAULT_MODELS
    for model in cli.DEFAULT_MODELS:
        assert _ANTHROPIC_ID.match(model), model


def test_mock_betas_cover_exactly_the_default_models() -> None:
    assert set(cli.DEFAULT_MOCK_BETAS) == set(cli.DEFAULT_MODELS)


def test_help_lists_the_default_models(capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("COLUMNS", "400")  # argparse wraps at hyphens otherwise
    try:
        cli.main(["--help"])
    except SystemExit:
        pass
    out = " ".join(capsys.readouterr().out.split())
    assert ", ".join(cli.DEFAULT_MODELS) in out


def test_docstring_example_model_is_an_anthropic_id() -> None:
    doc = benchmark.run_benchmark.__doc__ or ""
    examples = re.findall(r'``"([^"]+)"``', doc)
    assert examples
    for model in examples:
        assert _ANTHROPIC_ID.match(model), model


def test_readme_cli_examples_use_anthropic_ids() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("README.md", "README.ja.md"):
        text = (root / name).read_text(encoding="utf-8")
        for model in re.findall(r"--model\s+(\S+)", text):
            assert _ANTHROPIC_ID.match(model), f"{name}: {model}"
