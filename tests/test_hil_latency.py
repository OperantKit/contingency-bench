"""Tests for the logic-analyzer latency analysis (VCD -> latency table).

The fixtures are two CRF 200 runs recorded on a Teensy 4.1 fake chamber with
sigrok-cli at 1 MHz (D0 = feeder, D1 = response marker). ``crf200`` was taken
with a firmware bug that switched the feeder off about 1 us after switching it
on, so only 30 feeder pulses survived the 1 MHz sampling; ``crf200-fixed`` is
the same run with the corrected firmware (50 ms pulses).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, norm

from contingency_bench import hil_latency
from contingency_bench.hil_latency import (
    analyze,
    load_device_report,
    min_sample_size,
    parse_vcd,
    quantile_interval,
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


# --- confidence intervals of p95 / p99 -------------------------------------


@pytest.mark.parametrize(
    ("n", "p", "level", "lower_rank", "upper_rank", "coverage"),
    [
        # Le Boudec (2010), Table A.3: sample sizes where the upper rank is the maximum.
        (59, 0.95, 0.95, 50, 59, 0.951),
        (60, 0.95, 0.95, 52, 60, 0.951),
        (65, 0.95, 0.95, 57, 65, 0.959),
        (71, 0.95, 0.95, 63, 71, 0.965),
        (90, 0.95, 0.99, 76, 90, 0.990),
        (100, 0.95, 0.99, 88, 100, 0.993),
        # At most 2.5 % on each side: qbinom(0.025) and qbinom(0.975) + 1.
        (200, 0.95, 0.95, 184, 197, 0.967),
    ],
)
def test_interval_ranks(
    n: int, p: float, level: float, lower_rank: int, upper_rank: int, coverage: float
) -> None:
    interval = quantile_interval(range(1, n + 1), p, level)
    assert interval is not None
    assert (interval.lower_rank, interval.upper_rank) == (lower_rank, upper_rank)
    assert (interval.lower, interval.upper) == (lower_rank, upper_rank)
    assert interval.level == level
    assert interval.coverage == pytest.approx(coverage, abs=5.1e-4)


@pytest.mark.parametrize(
    ("p", "level", "expected"),
    [(0.95, 0.95, 59), (0.99, 0.95, 299), (0.95, 0.99, 90), (0.99, 0.99, 459)],
)
def test_interval_exists_from_the_minimum_sample_size(
    p: float, level: float, expected: int
) -> None:
    assert min_sample_size(p, level) == expected
    assert quantile_interval(range(expected - 1), p, level) is None
    assert quantile_interval(range(expected), p, level) is not None


@pytest.mark.parametrize("p", [0.95, 0.99])
def test_interval_satisfies_the_binomial_condition_for_every_sample_size(p: float) -> None:
    for n in range(min_sample_size(p, 0.95), 700):
        interval = quantile_interval(range(1, n + 1), p)
        assert interval is not None
        j, k = interval.lower_rank, interval.upper_rank
        coverage = binom.cdf(k - 1, n, p) - binom.cdf(j - 1, n, p)
        assert 1 <= j < k <= n
        assert coverage >= 0.95
        assert interval.coverage == pytest.approx(coverage)
        point_rank = math.ceil((n - 1) * p) + 1
        assert j <= point_rank <= k


def test_interval_leaves_at_most_half_the_error_on_each_side() -> None:
    for n in (72, 200, 368, 1000):
        interval = quantile_interval(range(1, n + 1), 0.95)
        assert interval is not None
        assert binom.cdf(interval.lower_rank - 1, n, 0.95) <= 0.025
        assert 1 - binom.cdf(interval.upper_rank - 1, n, 0.95) <= 0.025


def test_interval_of_a_low_quantile_mirrors_the_high_one() -> None:
    for n in (59, 65, 100, 200):
        high = quantile_interval(range(1, n + 1), 0.95)
        low = quantile_interval(range(1, n + 1), 0.05)
        assert high is not None and low is not None
        assert low.lower_rank == n + 1 - high.upper_rank
        assert low.upper_rank == n + 1 - high.lower_rank


@pytest.mark.parametrize(
    ("n", "p"), [(59, 0.95), (200, 0.95), (1000, 0.95), (299, 0.99), (1000, 0.99)]
)
def test_interval_contains_the_true_quantile_at_the_stated_rate(n: int, p: float) -> None:
    """Right-skewed samples of a known distribution (lognormal)."""
    rng = np.random.default_rng(20250101)
    truth = math.exp(norm.ppf(p))
    n_samples = 4000
    hits = 0
    coverage = 0.0
    for sample in rng.lognormal(size=(n_samples, n)):
        interval = quantile_interval(sample, p)
        assert interval is not None
        coverage = interval.coverage
        hits += interval.lower <= truth <= interval.upper
    assert coverage >= 0.95
    tolerance = 4 * math.sqrt(coverage * (1 - coverage) / n_samples)
    assert hits / n_samples == pytest.approx(coverage, abs=tolerance)


@pytest.mark.parametrize(("p", "level"), [(0.0, 0.95), (1.0, 0.95), (0.95, 0.0), (0.95, 1.0)])
def test_interval_rejects_probabilities_outside_the_open_unit_range(p: float, level: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantile_interval(range(100), p, level)
    with pytest.raises(ValueError, match="between 0 and 1"):
        min_sample_size(p, level)


def test_fixed_run_has_a_p95_interval_and_warns_about_p99() -> None:
    result = analyze(parse_vcd(FIXED_VCD.read_text()), expected_width_s=0.05)
    stats = result.latency_stats_us
    assert stats is not None and stats.p95_ci is not None
    ordered = sorted(v for v in result.latencies_us if v is not None)
    assert stats.p95_ci.lower == pytest.approx(ordered[183])
    assert stats.p95_ci.upper == pytest.approx(ordered[196])
    assert stats.p95_ci.lower <= stats.p95 <= stats.p95_ci.upper
    assert stats.p99_ci is None

    assert result.warnings == []
    assert len(result.interval_warnings) == 1
    assert "p99" in result.interval_warnings[0]
    assert "299" in result.interval_warnings[0]
    assert "response -> feeder n=200" in result.interval_warnings[0]


def test_broken_run_has_no_interval_at_all() -> None:
    result = analyze(parse_vcd(BROKEN_VCD.read_text()))
    stats = result.latency_stats_us
    assert stats is not None
    assert stats.p95_ci is None and stats.p99_ci is None
    assert [w for w in result.interval_warnings if "p95" in w and "n=30" in w]
    assert [w for w in result.interval_warnings if "p99" in w and "n=30" in w]


def test_no_interval_warning_without_values() -> None:
    text = (
        "$timescale 1 us $end\n$var wire 1 ! D0 $end\n$var wire 1 a D1 $end\n"
        "$enddefinitions $end\n#0 0! 0a\n#20\n"
    )
    assert analyze(parse_vcd(text)).interval_warnings == []


# --- CLI -------------------------------------------------------------------


def test_cli_shows_intervals_and_keeps_the_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = hil_latency.main([str(FIXED_VCD), "--expected-width", "0.05"])
    captured = capsys.readouterr()
    assert code == 0
    assert "p95 95% CI" in captured.out
    assert "p99 95% CI" in captured.out
    row = next(line for line in captured.out.splitlines() if line.startswith("response ->"))
    assert row.count("[") == 1
    assert "warning: no 95% confidence interval for p99" in captured.err


def test_cli_json_has_intervals(capsys: pytest.CaptureFixture[str]) -> None:
    code = hil_latency.main([str(FIXED_VCD), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    ci = data["latency_us"]["p95_ci"]
    assert set(ci) == {"lower", "upper", "level", "coverage", "lower_rank", "upper_rank"}
    assert (ci["lower_rank"], ci["upper_rank"]) == (184, 197)
    assert data["latency_us"]["p99_ci"] is None
    assert len(data["interval_warnings"]) == 1


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
