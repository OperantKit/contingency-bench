"""The law suite runs to the end and writes a well-formed result."""

from __future__ import annotations

import json
from pathlib import Path

from contingency_bench.controls import RandomSubject
from contingency_bench.laws import LAW_IDS, main, run_law_suite


def test_law_suite_completes_for_random_subject(tmp_path: Path) -> None:
    result = run_law_suite(
        "random", lambda seed: RandomSubject(seed=seed), seeds=range(2), steps=40
    )

    assert set(result["laws"]) == set(LAW_IDS)
    for law in result["laws"].values():
        assert law["verdict"] in ("pass", "fail")
        assert len(law["values"]) == 2
        assert law["k"] == len(law["passed_seeds"])
    # The procedure invariant must hold for any subject.
    assert result["laws"]["I1"]["verdict"] == "pass"

    out = tmp_path / "random.json"
    assert main(["--subject", "random", "--seeds", "2", "--steps", "40", "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["subject"] == "random"
    assert written["kind"] == "control"
    assert set(written["laws"]) == set(LAW_IDS)
