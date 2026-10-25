"""Latency table from a logic-analyzer recording (VCD) of a hardware-in-the-loop run.

The device under test marks each response on one digital line and drives the
feeder (reinforcer) on another. For every response marker rising edge, the
latency is the time to the next feeder rising edge, provided that edge comes
before the next response marker. The analysis reports the median, p95, p99,
maximum and count of these latencies, compares them one by one with the
latencies the device measured itself (its ``report``), and checks every feeder
pulse width against the commanded reinforcer duration.

The VCD is the format written by ``sigrok-cli -O vcd``. The time unit is read
from ``$timescale`` (sigrok writes ``1 us`` at 1 MHz and ``10 ns`` at 4 MHz),
so recordings at different sample rates give latencies in the same unit.

Default channels follow the Teensy 4.1 fake chamber in this repository:
sigrok ``D0`` (analyzer CH1) is the feeder and ``D1`` (CH2) the response marker.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_FEEDER_SIGNAL = "D0"
DEFAULT_RESPONSE_SIGNAL = "D1"
DEFAULT_WIDTH_TOLERANCE_S = 1e-4
_EXAMPLES_IN_WARNING = 3

_UNIT_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}
_TIMESCALE = re.compile(r"\$timescale\s+(\d+)\s*([munpf]?s)\s+\$end", re.DOTALL)
_VAR = re.compile(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[[^\]]*\])?\s+\$end", re.DOTALL)

Changes = list[tuple[int, int]]


@dataclass(frozen=True)
class Vcd:
    """Value changes of the 1-bit signals in a VCD, keyed by signal name.

    Each change is ``(tick, value)`` with value 0 or 1; one tick lasts
    ``timescale_s`` seconds. ``x`` / ``z`` values are dropped.
    """

    timescale_s: float
    signals: dict[str, Changes]


@dataclass(frozen=True)
class Stats:
    """Summary of a sample. ``p95`` / ``p99`` are recorded values (numpy's
    ``higher`` method), so a tail is never interpolated below an observation."""

    count: int
    median: float
    p95: float
    p99: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class WidthMismatch:
    rise_us: float
    width_us: float


@dataclass(frozen=True)
class LatencyAnalysis:
    """Result of :func:`analyze`. Times are in microseconds.

    ``latencies_us`` and ``device_diff_us`` have one entry per response marker
    (``None`` when that response has no feeder edge or no device value).
    """

    n_responses: int
    latencies_us: list[float | None]
    latency_stats_us: Stats | None
    n_feeder_pulses: int
    width_stats_us: Stats | None
    width_mismatches: list[WidthMismatch]
    device_diff_us: list[float | None]
    device_diff_stats_us: Stats | None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_latencies(self) -> int:
        return sum(1 for v in self.latencies_us if v is not None)

    @property
    def n_unpaired_responses(self) -> int:
        return self.n_responses - self.n_latencies


def parse_vcd(text: str) -> Vcd:
    """Parse the 1-bit signals of a VCD file."""
    header, sep, body = text.partition("$enddefinitions")
    if not sep:
        raise ValueError("VCD has no $enddefinitions")
    match = _TIMESCALE.search(header)
    if match is None:
        raise ValueError("VCD has no $timescale")
    timescale_s = int(match.group(1)) * _UNIT_SECONDS[match.group(2)]

    names: dict[str, str] = {}
    for width, ident, name in _VAR.findall(header):
        if width == "1":
            names[ident] = name
    signals: dict[str, Changes] = {name: [] for name in names.values()}

    body = body.split("$end", 1)[1] if "$end" in body else body
    tick = 0
    for token in body.split():
        if token.startswith("#"):
            tick = int(token[1:])
        elif token[0] in "01" and token[1:] in names:
            signals[names[token[1:]]].append((tick, int(token[0])))
    return Vcd(timescale_s=timescale_s, signals=signals)


def rising_edges(changes: Changes) -> list[int]:
    """Ticks where the signal goes from 0 to 1."""
    edges: list[int] = []
    prev: int | None = None
    for tick, value in changes:
        if prev == 0 and value == 1:
            edges.append(tick)
        prev = value
    return edges


def pulses(changes: Changes) -> list[tuple[int, int]]:
    """``(rise, fall)`` ticks of every complete high pulse."""
    out: list[tuple[int, int]] = []
    prev: int | None = None
    rise: int | None = None
    for tick, value in changes:
        if prev == 0 and value == 1:
            rise = tick
        elif prev == 1 and value == 0 and rise is not None:
            out.append((rise, tick))
            rise = None
        prev = value
    return out


def load_device_report(path: Path) -> list[float]:
    """Read the device's own latencies (``lat_us``) from a report JSON.

    Accepts the bare report line ``{"kind": "report", "lat_us": [...]}`` or a
    host summary that holds it under ``"device"``.
    """
    data = json.loads(Path(path).read_text())
    report = data.get("device", data) if isinstance(data, dict) else None
    if not isinstance(report, dict) or "lat_us" not in report:
        raise ValueError(f"{path}: no device report with 'lat_us'")
    return [float(v) for v in report["lat_us"]]


def _stats(values: Sequence[float]) -> Stats | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return Stats(
        count=len(arr),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95, method="higher")),
        p99=float(np.percentile(arr, 99, method="higher")),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
    )


def _pair_latencies(responses: list[int], feeders: list[int]) -> list[int | None]:
    """For each response, ticks to the first feeder edge before the next response."""
    out: list[int | None] = []
    j = 0
    for i, r in enumerate(responses):
        while j < len(feeders) and feeders[j] < r:
            j += 1
        limit = responses[i + 1] if i + 1 < len(responses) else None
        if j < len(feeders) and (limit is None or feeders[j] < limit):
            out.append(feeders[j] - r)
        else:
            out.append(None)
    return out


def _signal(vcd: Vcd, name: str) -> Changes:
    if name not in vcd.signals:
        raise ValueError(f"signal {name!r} not in VCD (have: {', '.join(sorted(vcd.signals))})")
    return vcd.signals[name]


def analyze(
    vcd: Vcd,
    *,
    response_signal: str = DEFAULT_RESPONSE_SIGNAL,
    feeder_signal: str = DEFAULT_FEEDER_SIGNAL,
    device_latencies_us: Sequence[float] | None = None,
    expected_width_s: float | None = None,
    width_tolerance_s: float = DEFAULT_WIDTH_TOLERANCE_S,
) -> LatencyAnalysis:
    """Response-to-feeder latencies, feeder widths and the diff to the device report."""
    us_per_tick = vcd.timescale_s * 1e6
    responses = rising_edges(_signal(vcd, response_signal))
    feeder_changes = _signal(vcd, feeder_signal)
    ticks = _pair_latencies(responses, rising_edges(feeder_changes))
    latencies = [None if t is None else t * us_per_tick for t in ticks]
    warnings: list[str] = []

    n_unpaired = sum(1 for v in latencies if v is None)
    if n_unpaired:
        warnings.append(
            f"{n_unpaired} of {len(responses)} response markers have no feeder rising edge "
            "before the next response marker"
        )

    feeder_pulses = pulses(feeder_changes)
    widths = [(rise * us_per_tick, (fall - rise) * us_per_tick) for rise, fall in feeder_pulses]
    mismatches: list[WidthMismatch] = []
    if expected_width_s is not None:
        expected_us = expected_width_s * 1e6
        tolerance_us = width_tolerance_s * 1e6
        mismatches = [
            WidthMismatch(rise_us=rise, width_us=width)
            for rise, width in widths
            if abs(width - expected_us) > tolerance_us
        ]
        if mismatches:
            examples = ", ".join(
                f"{m.width_us:g} us at {m.rise_us / 1e6:.6f} s"
                for m in mismatches[:_EXAMPLES_IN_WARNING]
            )
            warnings.append(
                f"{len(mismatches)} of {len(widths)} feeder pulse widths differ from the "
                f"commanded duration {expected_us:g} us by more than {tolerance_us:g} us "
                f"(e.g. {examples}). Check the firmware output first (how it times the end "
                "of the pulse) before suspecting the wiring."
            )

    diffs: list[float | None] = [None] * len(latencies)
    if device_latencies_us is not None:
        if len(device_latencies_us) != len(latencies):
            warnings.append(
                f"device report has {len(device_latencies_us)} latencies but the VCD has "
                f"{len(latencies)} response markers; compared the first "
                f"{min(len(device_latencies_us), len(latencies))} by index"
            )
        for i, (lat, dev) in enumerate(zip(latencies, device_latencies_us, strict=False)):
            if lat is not None:
                diffs[i] = lat - float(dev)

    return LatencyAnalysis(
        n_responses=len(responses),
        latencies_us=latencies,
        latency_stats_us=_stats([v for v in latencies if v is not None]),
        n_feeder_pulses=len(feeder_pulses),
        width_stats_us=_stats([w for _, w in widths]),
        width_mismatches=mismatches,
        device_diff_us=diffs,
        device_diff_stats_us=_stats([d for d in diffs if d is not None]),
        warnings=warnings,
    )


def _format_row(label: str, stats: Stats | None) -> str:
    if stats is None:
        return f"{label:<28} {'-':>8}"
    return (
        f"{label:<28} {stats.count:>8} {stats.median:>12.1f} {stats.p95:>12.1f} "
        f"{stats.p99:>12.1f} {stats.minimum:>12.1f} {stats.maximum:>12.1f}"
    )


def format_table(result: LatencyAnalysis) -> str:
    """Plain-text table of the analysis (times in microseconds)."""
    head = (
        f"{'(us)':<28} {'count':>8} {'median':>12} {'p95':>12} {'p99':>12} {'min':>12} {'max':>12}"
    )
    lines = [
        f"response markers: {result.n_responses}, with feeder edge: {result.n_latencies}, "
        f"feeder pulses: {result.n_feeder_pulses}",
        head,
        _format_row("response -> feeder", result.latency_stats_us),
        _format_row("feeder pulse width", result.width_stats_us),
        _format_row("analyzer - device report", result.device_diff_stats_us),
    ]
    return "\n".join(lines)


def to_json(result: LatencyAnalysis) -> dict[str, object]:
    """JSON-serialisable summary (per-response values included)."""

    def stats(s: Stats | None) -> dict[str, float] | None:
        return None if s is None else asdict(s)

    return {
        "n_responses": result.n_responses,
        "n_latencies": result.n_latencies,
        "n_feeder_pulses": result.n_feeder_pulses,
        "latency_us": stats(result.latency_stats_us),
        "width_us": stats(result.width_stats_us),
        "device_diff_us": stats(result.device_diff_stats_us),
        "latencies_us": result.latencies_us,
        "device_diffs_us": result.device_diff_us,
        "width_mismatches": [asdict(m) for m in result.width_mismatches],
        "warnings": result.warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: print the latency table; exit 1 when there are warnings."""
    parser = argparse.ArgumentParser(
        prog="contingency-bench-hil-latency",
        description="Response-to-feeder latency table from a sigrok VCD recording.",
    )
    parser.add_argument("vcd", type=Path, help="VCD written by sigrok-cli -O vcd.")
    parser.add_argument(
        "--device-report",
        type=Path,
        help="JSON with the device's own latencies (lat_us), bare or under 'device'.",
    )
    parser.add_argument(
        "--expected-width",
        type=float,
        help="Commanded reinforcer duration in seconds; feeder pulses are checked against it.",
    )
    parser.add_argument(
        "--width-tolerance",
        type=float,
        default=DEFAULT_WIDTH_TOLERANCE_S,
        help="Allowed feeder width error in seconds (default: %(default)s).",
    )
    parser.add_argument("--response-signal", default=DEFAULT_RESPONSE_SIGNAL)
    parser.add_argument("--feeder-signal", default=DEFAULT_FEEDER_SIGNAL)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args(argv)

    try:
        device = load_device_report(args.device_report) if args.device_report else None
        result = analyze(
            parse_vcd(args.vcd.read_text()),
            response_signal=args.response_signal,
            feeder_signal=args.feeder_signal,
            device_latencies_us=device,
            expected_width_s=args.expected_width,
            width_tolerance_s=args.width_tolerance,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(to_json(result)) if args.json else format_table(result))
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 1 if result.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
