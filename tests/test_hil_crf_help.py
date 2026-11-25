"""``contingency-bench-hil-crf --help`` against what the tool does.

The schedules the device decides are read from ``SUPPORTED_ON_DEVICE``, so the
help cannot fall behind it. These tests build the parser only; no port is opened.
"""

from __future__ import annotations

import argparse
import re
import sys
from types import ModuleType

import pytest

from contingency_bench.hil_crf import _build_parser, main


def _help_text(capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    return " ".join(capsys.readouterr().out.split())


def _decision_help() -> str:
    (action,) = [a for a in _build_parser()._actions if a.dest == "decision"]
    assert action.help is not None
    return action.help


def _names(text: str, names: tuple[str, ...]) -> set[str]:
    return {name for name in names if re.search(rf"\b{name}\b", text)}


def _fake_device_module(supported: tuple[str, ...], seeded: tuple[str, ...]) -> ModuleType:
    module = ModuleType("contingency_bench.hil_device")
    module.SUPPORTED_ON_DEVICE = supported  # type: ignore[attr-defined]
    module.SEEDED_ON_DEVICE = seeded  # type: ignore[attr-defined]
    return module


def test_the_decision_help_names_every_schedule_the_device_decides(
    capsys: pytest.CaptureFixture[str],
):
    pytest.importorskip("experiment_io.device")
    from contingency_bench.hil_device import SUPPORTED_ON_DEVICE

    printed = _help_text(capsys)

    assert _names(_decision_help(), SUPPORTED_ON_DEVICE) == set(SUPPORTED_ON_DEVICE)
    assert ", ".join(SUPPORTED_ON_DEVICE) in printed
    assert "CRF and FR only" not in printed


def test_the_decision_help_names_the_schedules_that_need_a_seed():
    pytest.importorskip("experiment_io.device")
    from contingency_bench.hil_device import SEEDED_ON_DEVICE

    assert f"{', '.join(SEEDED_ON_DEVICE)} need a seed" in _decision_help()


def test_the_decision_help_follows_the_table(monkeypatch: pytest.MonkeyPatch):
    fake = _fake_device_module(("AAA", "BBB"), ("BBB",))
    monkeypatch.setitem(sys.modules, "contingency_bench.hil_device", fake)

    text = _decision_help()

    assert "AAA, BBB" in text
    assert "BBB need a seed" in text
    assert _names(text, ("CRF", "FR", "VI")) == set()


def test_help_is_printed_without_the_hil_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setitem(sys.modules, "contingency_bench.hil_device", None)

    printed = _help_text(capsys)

    assert "--decision {host,device}" in printed
    assert "hil extra" in _decision_help()


def test_the_decision_help_says_realtime_cannot_be_combined_with_device():
    assert "cannot be combined with --realtime" in _decision_help()


def test_every_option_has_a_help_text():
    missing = [
        "/".join(action.option_strings)
        for action in _build_parser()._actions
        if not action.help or action.help is argparse.SUPPRESS
    ]
    assert missing == []
