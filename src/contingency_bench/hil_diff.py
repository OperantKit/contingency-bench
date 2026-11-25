"""Compare the decisions of a host-decided and a device-decided hardware-in-the-loop run.

``contingency-bench-hil-crf`` can let the host (contingency-py) or the device
(contingency-rs on the board) decide which response is reinforced. The board's
simulated subject responds on a fixed period but not while the feeder is on, so
both runs see the same response sequence as long as no feeder operation
overlaps a scheduled response, and their OKL logs can be compared response by
response.

The two logs do not share a clock or an origin:

- host log (``decision = "host"``): row times are the host's monotonic clock,
  and a response row carries the device's time in ``device_time``. The origin
  is the ``state_change IDLE RUNNING`` row (the time sent to the device by
  ``sync``), not the header's ``session_start``. The reinforcer row has the
  same time as the response row, so the response-to-reinforcer latency cannot
  be read from this log; the device's report (``lat_us`` in the summary JSON)
  holds it.
- device log (``decision = "device"``): row times are the device's clock and
  the origin is the header's ``session_start``.

Responses are matched by their ``id``. A response is "reinforced" when a
``reinforcer_start`` row follows it before the next response. The comparison
first checks that the response sequences agree (same count, and each
response's time from the origin within a threshold); where they stop agreeing,
the decisions after that point are not a comparison of the same input, so the
decision agreement and the reinforcer time differences are computed only on
the responses before it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLD_MS = 1.0


@dataclass(frozen=True)
class OklRun:
    """The parts of one OKL log the comparison uses. Times are seconds from the origin."""

    path: str
    decision: str
    dsl: str
    response_ids: list[int]
    response_times: list[float]
    reinforced_ids: list[int]
    reinforcer_times: list[float]
    reinforcer_durations: list[float]
    response_to_reinforcer_us: list[float] | None


@dataclass(frozen=True)
class Stats:
    """Summary of a sample. ``p95`` is a recorded value (numpy's ``method="higher"``)."""

    count: int
    minimum: float
    median: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class DiffResult:
    n_responses: tuple[int, int]
    n_reinforcers: tuple[int, int]
    threshold_ms: float
    response_time_diff_ms: Stats | None
    divergence_at: int | None
    compared_responses: int
    agreement: tuple[int, int]
    mismatched_ids: list[int]
    reinforced_ids: tuple[list[int], list[int]]
    reinforcer_time_diff_ms: Stats | None
    host_latency_us: Stats | None
    device_latency_us: Stats | None
    reinforcer_duration_ms: tuple[Stats | None, Stats | None]

    @property
    def ok(self) -> bool:
        return (
            self.n_responses[0] == self.n_responses[1]
            and self.n_reinforcers[0] == self.n_reinforcers[1]
            and self.divergence_at is None
            and not self.mismatched_ids
        )


def stats(values: Sequence[float]) -> Stats | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return Stats(
        count=int(arr.size),
        minimum=float(arr.min()),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95, method="higher")),
        maximum=float(arr.max()),
    )


def _header_value(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"'):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip('"')
        return str(value)
    return raw


def load_okl(path: Path) -> OklRun:
    """Read an OKL v1 log written by ``contingency-bench-hil-crf``."""
    header: dict[str, str] = {}
    rows: list[tuple[float, str, list[str]]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            if " = " in line:
                key, value = line[1:].split(" = ", 1)
                header[key.strip()] = _header_value(value)
            continue
        fields = line.split("\t")
        try:
            rows.append((float(fields[0]), fields[1], fields[2:]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: not an OKL event row: {line!r}") from exc
    for key in ("decision", "session_start"):
        if key not in header:
            raise ValueError(f"{path}: header has no {key!r}")
    decision = header["decision"]
    origin = float(header["session_start"])
    if decision == "host":
        origin = next(
            (t for t, kind, f in rows if kind == "state_change" and f[:2] == ["IDLE", "RUNNING"]),
            origin,
        )

    response_ids: list[int] = []
    response_times: list[float] = []
    response_rows: list[float] = []
    reinforced: list[int] = []
    rf_times: list[float] = []
    rf_rows: list[float] = []
    rf_ends: list[float] = []
    for t, kind, f in rows:
        if kind == "response":
            response_ids.append(int(f[0]))
            device_time = f[2] if len(f) > 2 else "-"
            stamp = float(device_time) if device_time not in ("", "-") else t
            response_times.append(stamp - origin)
            response_rows.append(t)
        elif kind == "reinforcer_start":
            reinforced.append(response_ids[-1] if response_ids else 0)
            rf_times.append(t - origin)
            rf_rows.append(t)
        elif kind == "reinforcer_end":
            rf_ends.append(t)

    latency: list[float] | None = None
    if decision == "device":
        index = dict(zip(response_ids, response_rows, strict=True))
        latency = [
            (t - index[rid]) * 1e6
            for rid, t in zip(reinforced, rf_rows, strict=True)
            if rid in index
        ]
    return OklRun(
        path=str(path),
        decision=decision,
        dsl=header.get("dsl", ""),
        response_ids=response_ids,
        response_times=response_times,
        reinforced_ids=reinforced,
        reinforcer_times=rf_times,
        reinforcer_durations=[(e - s) for s, e in zip(rf_rows, rf_ends, strict=False)],
        response_to_reinforcer_us=latency,
    )


def load_lat_us(path: Path) -> list[float]:
    """Read ``lat_us`` from a summary JSON (bare report or under ``"device"``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    report = data.get("device", data) if isinstance(data, dict) else None
    lat = report.get("lat_us") if isinstance(report, dict) else None
    if not isinstance(lat, list):
        raise ValueError(f"{path}: no lat_us list (bare or under 'device')")
    return [float(v) for v in lat]


def _first_times(ids: Sequence[int], times: Sequence[float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for rid, t in zip(ids, times, strict=True):
        out.setdefault(rid, t)
    return out


def compare(
    host: OklRun,
    device: OklRun,
    *,
    threshold_ms: float = DEFAULT_THRESHOLD_MS,
    host_summary: Sequence[float] | None = None,
    device_summary: Sequence[float] | None = None,
) -> DiffResult:
    """Compare two runs. ``*_summary`` are the device-reported ``lat_us`` values."""
    n = min(len(host.response_times), len(device.response_times))
    diffs = [
        (a - b) * 1e3
        for a, b in zip(host.response_times[:n], device.response_times[:n], strict=True)
    ]
    divergence: int | None = next(
        (k + 1 for k, d in enumerate(diffs) if abs(d) > threshold_ms), None
    )
    if divergence is None and len(host.response_times) != len(device.response_times):
        divergence = n + 1
    compared = n if divergence is None else divergence - 1

    host_set = {r for r in host.reinforced_ids if r <= compared}
    device_set = {r for r in device.reinforced_ids if r <= compared}
    mismatched = sorted(host_set ^ device_set)

    host_rf = _first_times(host.reinforced_ids, host.reinforcer_times)
    device_rf = _first_times(device.reinforced_ids, device.reinforcer_times)
    rf_diff = [(host_rf[r] - device_rf[r]) * 1e3 for r in sorted(host_set & device_set)]

    device_latency = (
        device_summary if device_summary is not None else device.response_to_reinforcer_us
    )
    return DiffResult(
        n_responses=(len(host.response_times), len(device.response_times)),
        n_reinforcers=(len(host.reinforced_ids), len(device.reinforced_ids)),
        threshold_ms=threshold_ms,
        response_time_diff_ms=stats(diffs[:compared]),
        divergence_at=divergence,
        compared_responses=compared,
        agreement=(compared - len(mismatched), compared),
        mismatched_ids=mismatched,
        reinforced_ids=(sorted(set(host.reinforced_ids)), sorted(set(device.reinforced_ids))),
        reinforcer_time_diff_ms=stats(rf_diff),
        host_latency_us=stats(host_summary) if host_summary is not None else None,
        device_latency_us=stats(device_latency) if device_latency is not None else None,
        reinforcer_duration_ms=(
            stats([d * 1e3 for d in host.reinforcer_durations]),
            stats([d * 1e3 for d in device.reinforcer_durations]),
        ),
    )


def _fmt(s: Stats | None, digits: int) -> str:
    if s is None:
        return "-"
    return (
        f"n={s.count} min={s.minimum:.{digits}f} median={s.median:.{digits}f} "
        f"p95={s.p95:.{digits}f} max={s.maximum:.{digits}f}"
    )


def _same(a: object, b: object) -> str:
    return "match" if a == b else "DIFFER"


def format_table(result: DiffResult) -> str:
    h_n, d_n = result.n_responses
    h_r, d_r = result.n_reinforcers
    agree, total = result.agreement
    seq = (
        "match" if result.divergence_at is None else f"DIFFER from response {result.divergence_at}"
    )
    host_lat = (
        _fmt(result.host_latency_us, 0)
        if result.host_latency_us is not None
        else "not in the OKL (pass --host-summary)"
    )
    rows = [
        ("axis", "host", "device", "result"),
        ("A responses", str(h_n), str(d_n), _same(h_n, d_n)),
        (
            f"A response time diff (ms, threshold {result.threshold_ms:g})",
            "",
            "",
            f"{seq}; {_fmt(result.response_time_diff_ms, 3)}",
        ),
        (
            "B decision agreement",
            "",
            "",
            f"{agree}/{total}"
            + (f"; differ at responses {result.mismatched_ids}" if result.mismatched_ids else ""),
        ),
        (
            "B reinforced responses",
            ",".join(map(str, result.reinforced_ids[0])),
            ",".join(map(str, result.reinforced_ids[1])),
            _same(*result.reinforced_ids),
        ),
        ("C reinforcers", str(h_r), str(d_r), _same(h_r, d_r)),
        (
            "D reinforcer time diff, host - device (ms)",
            "",
            "",
            _fmt(result.reinforcer_time_diff_ms, 3),
        ),
        ("E response to reinforcer (us)", host_lat, _fmt(result.device_latency_us, 0), ""),
        (
            "F reinforcer duration (ms)",
            _fmt(result.reinforcer_duration_ms[0], 3),
            _fmt(result.reinforcer_duration_ms[1], 3),
            "",
        ),
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [" | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contingency-bench-hil-diff",
        description="Compare the OKL logs of a host-decided and a device-decided run.",
    )
    parser.add_argument("host_okl", type=Path)
    parser.add_argument("device_okl", type=Path)
    parser.add_argument("--host-summary", type=Path, help="Summary JSON of the host run (lat_us).")
    parser.add_argument("--device-summary", type=Path, help="Summary JSON of the device run.")
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=DEFAULT_THRESHOLD_MS,
        help="Largest response time difference counted as the same response (default 1 ms).",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args(argv)

    try:
        host = load_okl(args.host_okl)
        device = load_okl(args.device_okl)
        host_lat = load_lat_us(args.host_summary) if args.host_summary else None
        device_lat = load_lat_us(args.device_summary) if args.device_summary else None
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if host.decision != "host" or device.decision != "device":
        print(
            f"error: decision is {host.decision!r} / {device.decision!r}; "
            "pass the host log first and the device log second",
            file=sys.stderr,
        )
        return 2
    if host.dsl != device.dsl:
        print(f"error: dsl differs: {host.dsl!r} vs {device.dsl!r}", file=sys.stderr)
        return 2

    result = compare(
        host,
        device,
        threshold_ms=args.threshold_ms,
        host_summary=host_lat,
        device_summary=device_lat,
    )
    if args.json:
        payload = asdict(result)
        payload["ok"] = result.ok
        print(json.dumps(payload, indent=1))
    else:
        print(format_table(result))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
