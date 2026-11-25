"""The "Host-side run" section of both READMEs against ``contingency-bench-hil-crf``.

The options come from the argparse parser and the schedules from
``SUPPORTED_ON_DEVICE``, so a README that falls behind either fails here. The
option test builds the parser only and needs no hil extra.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from contingency_bench.hil_crf import _build_parser

ROOT = Path(__file__).resolve().parent.parent

# README -> (heading of the section, header cell of its option table, label of
# the schedules row of its host/device table)
LAYOUT = {
    "README.md": ("### Host-side run", "Option", "Schedules"),
    "README.ja.md": ("### ホスト側の実行", "オプション", "スケジュール"),
}
_FENCE = re.compile(r"^```.*?^```[^\n]*\n", re.DOTALL | re.MULTILINE)
_OPTION = re.compile(r"`(--[a-z][a-z-]*)`")
_SCHEDULE = re.compile(r"\b[A-Z]{2,}\b", re.ASCII)


def _section(text: str, heading: str) -> list[str]:
    """Lines under ``heading`` up to the next heading of its level or above."""
    lines = _FENCE.sub("", text).splitlines()
    assert heading in lines, f"no heading {heading!r}"
    body = lines[lines.index(heading) + 1 :]
    level = len(heading) - len(heading.lstrip("#"))
    end = next(
        (i for i, line in enumerate(body) if re.match(rf"#{{1,{level}}} ", line)),
        len(body),
    )
    return body[:end]


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(lines: list[str]) -> list[list[list[str]]]:
    """Each Markdown table as rows of cells; the header row first, the rule row dropped."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*lines, ""]:
        if line.startswith("|"):
            cells = _cells(line)
            if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    return tables


def _read(name: str) -> list[str]:
    return _section((ROOT / name).read_text(encoding="utf-8"), LAYOUT[name][0])


def _documented_options(name: str) -> set[str]:
    header = LAYOUT[name][1]
    tables = [t for t in _tables(_read(name)) if t[0][0] == header]
    assert len(tables) == 1, f"{name}: {len(tables)} tables headed {header!r}"
    return {option for row in tables[0][1:] for option in _OPTION.findall(row[0])}


def _documented_schedules(name: str) -> set[str]:
    """Schedule names in the device column of the host/device table."""
    label = LAYOUT[name][2]
    rows = [
        (table[0], row)
        for table in _tables(_read(name))
        for row in table[1:]
        if row[0] == label and "`--decision device`" in table[0]
    ]
    assert len(rows) == 1, f"{name}: {len(rows)} rows {label!r} in a host/device table"
    header, row = rows[0]
    return set(_SCHEDULE.findall(row[header.index("`--decision device`")]))


def _parser_options() -> set[str]:
    return {
        option
        for action in _build_parser()._actions
        if action.dest != "help"
        for option in action.option_strings
        if option.startswith("--")
    }


@pytest.mark.parametrize("name", LAYOUT)
def test_the_option_table_lists_the_options_of_the_parser(name: str) -> None:
    documented, parsed = _documented_options(name), _parser_options()

    missing, unknown = sorted(parsed - documented), sorted(documented - parsed)
    assert not missing, f"{name} lacks the options {', '.join(missing)}"
    assert not unknown, f"{name} lists options the tool does not have: {', '.join(unknown)}"


@pytest.mark.parametrize("name", LAYOUT)
def test_the_device_schedules_are_those_the_device_decides(name: str) -> None:
    pytest.importorskip("experiment_io.device")
    from contingency_bench.hil_device import SUPPORTED_ON_DEVICE

    documented, supported = _documented_schedules(name), set(SUPPORTED_ON_DEVICE)

    missing, unknown = sorted(supported - documented), sorted(documented - supported)
    assert not missing, f"{name} lacks the schedules {', '.join(missing)}"
    assert not unknown, f"{name} lists schedules the device does not decide: {', '.join(unknown)}"
