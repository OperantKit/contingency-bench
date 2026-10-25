"""Tests for the logic-analyzer latency analysis (VCD -> latency table).

The fixtures are two CRF 200 runs recorded on a Teensy 4.1 fake chamber with
sigrok-cli at 1 MHz (D0 = feeder, D1 = response marker). ``crf200`` was taken
with a firmware bug that switched the feeder off about 1 us after switching it
on, so only 30 feeder pulses survived the 1 MHz sampling; ``crf200-fixed`` is
the same run with the corrected firmware (50 ms pulses).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contingency_bench import hil_latency
from contingency_bench.hil_latency import (
    analyze,
    load_device_report,
    parse_vcd,
    rising_edges,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hil"
FIXED_VCD = FIXTURES / "2026-10-25_crf200-fixed.vcd"
FIXED_HOST = FIXTURES / "2026-10-25_crf200-fixed_host.json"
BROKEN_VCD = FIXTURES / "2026-10-25_crf200.vcd"
BROKEN_HOST = FIXTURES / "2026-10-25_crf200_host.json"


# --- VCD parsing -----------------------------------------------------------


def test_timescale_1us() -> None:
    vcd = parse_vcd(FIXED_VCD.read_text())
    assert vcd.timescale_s == pytest.approx(1e-6)
    assert set(vcd.signals) == {"D0", "D1"}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("$timescale 10 ns $end", 10e-9),
        ("$timescale 1us $end", 1e-6),
        ("$timescale\n  100 ps\n$end", 100e-12),
        ("$timescale 1 ms $end", 1e-3),
    ],
)
def test_timescale_units(header: str, expected: float) -> None:
    text = (
        f"{header}\n$scope module m $end\n$var wire 1 ! D0 $end\n$upscope $end\n"
        "$enddefinitions $end\n#0 0!\n#25 1!\n#40 0!\n"
    )
    vcd = parse_vcd(text)
    assert vcd.timescale_s == pytest.approx(expected)
    assert rising_edges(vcd.signals["D0"]) == [25]


def test_multi_char_identifiers_and_same_line_changes() -> None:
    text = (
        "$timescale 1 us $end\n$var wire 1 !# D0 $end\n$var wire 1 a D1 $end\n"
        "$enddefinitions $end\n#0 0!# 0a\n#5 1!# 1a\n#9 0a\n#12 0!#\n#20\n"
    )
    vcd = parse_vcd(text)
    assert vcd.signals["D0"] == [(0, 0), (5, 1), (12, 0)]
    assert vcd.signals["D1"] == [(0, 0), (5, 1), (9, 0)]


def test_parse_rejects_missing_timescale() -> None:
    with pytest.raises(ValueError, match="timescale"):
        parse_vcd("$var wire 1 ! D0 $end\n$enddefinitions $end\n#0 0!\n")


# --- latency table on the fixed run ---------------------------------------


def test_fixed_run_matches_expected_table() -> None:
    device = load_device_report(FIXED_HOST)
    result = analyze(parse_vcd(FIXED_VCD.read_text()), device_latencies_us=device)

    assert result.n_responses == 200
    assert result.n_latencies == 200
    assert result.n_unpaired_responses == 0
    stats = result.latency_stats_us
    assert stats.median == pytest.approx(2287.5)
    assert stats.minimum == pytest.approx(616.0)
    assert stats.maximum == pytest.approx(86996.0)
    assert stats.p95 == pytest.approx(19783.0, abs=1.0)
    assert stats.p99 == pytest.approx(84313.0, abs=1.0)

    assert result.width_stats_us is not None
    assert result.width_stats_us.median == pytest.approx(50000.0)
    assert result.width_stats_us.minimum >= 49999.0
    assert result.width_stats_us.maximum <= 50001.0

    diffs = [d for d in result.device_diff_us if d is not None]
    assert len(diffs) == 200
    assert min(diffs) == pytest.approx(2.0)
    assert max(diffs) == pytest.approx(9.0)


def test_fixed_run_has_no_width_warning_against_command_duration() -> None:
    result = analyze(parse_vcd(FIXED_VCD.read_text()), expected_width_s=0.05)
    assert result.width_mismatches == []
    assert result.warnings == []


# --- width warning on the buggy run ----------------------------------------


def test_broken_run_warns_about_pulse_width() -> None:
    device = load_device_report(BROKEN_HOST)
    result = analyze(
        parse_vcd(BROKEN_VCD.read_text()),
        device_latencies_us=device,
        expected_width_s=0.05,
    )
    assert result.n_responses == 200
    assert result.n_latencies == 30
    assert result.n_unpaired_responses == 170
    assert len(result.width_mismatches) == 30
    assert all(m.width_us <= 1.0 for m in result.width_mismatches)
    assert len(result.warnings) == 2
    width_warning = next(w for w in result.warnings if "width" in w)
    assert "30 of 30" in width_warning
    assert "firmware" in width_warning
    unpaired_warning = next(w for w in result.warnings if "no feeder" in w)
    assert "170" in unpaired_warning


def test_device_diff_is_aligned_by_response_index() -> None:
    device = load_device_report(BROKEN_HOST)
    result = analyze(parse_vcd(BROKEN_VCD.read_text()), device_latencies_us=device)
    assert len(result.device_diff_us) == 200
    paired = [
        (lat, dev) for lat, dev in zip(result.latencies_us, device, strict=True) if lat is not None
    ]
    assert len(paired) == 30
    diffs = [d for d in result.device_diff_us if d is not None]
    assert diffs == [pytest.approx(lat - dev) for lat, dev in paired]


def test_device_report_length_mismatch_warns() -> None:
    device = load_device_report(FIXED_HOST)[:-5]
    result = analyze(parse_vcd(FIXED_VCD.read_text()), device_latencies_us=device)
    assert any("device report has 195" in w for w in result.warnings)


def test_load_device_report_accepts_bare_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"kind": "report", "n": 2, "lat_us": [10, 20]}))
    assert load_device_report(path) == [10.0, 20.0]


# --- CLI -------------------------------------------------------------------


def test_cli_prints_table(capsys: pytest.CaptureFixture[str]) -> None:
    code = hil_latency.main(
        [str(FIXED_VCD), "--device-report", str(FIXED_HOST), "--expected-width", "0.05"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2287.5" in out
    assert "200" in out
    assert "device" in out


def test_cli_json_and_warning_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    code = hil_latency.main([str(BROKEN_VCD), "--expected-width", "0.05", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert code == 1
    assert data["n_latencies"] == 30
    assert data["latency_us"]["count"] == 30
    assert len(data["warnings"]) == 2
    assert "firmware" in captured.err
