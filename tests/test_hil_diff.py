"""Tests for the host-vs-device decision diff of two OKL logs.

The synthetic logs copy the shape of a real pair taken on a Teensy 4.1 fake
chamber (VI 30 s, seed 7): in the host log a response carries the device's
time in ``device_time`` and its reinforcer row has the same timestamp as the
response row; the host log's origin is the ``state_change IDLE RUNNING`` row,
not the header's ``session_start``. In the device log the operandum is ``0``,
``device_time`` is ``-``, and the origin is ``session_start``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contingency_bench import hil_diff
from contingency_bench.hil_diff import compare, load_okl

DSL = '@algorithm(\\"fleshler-hoffman\\", seed=7)\\nVI 30s'
EVENTS = """# events:
#   response          : id:int operandum:int? device_time:float?
#   reinforcer_start  : id:int potency:float operandum:int?
#   reinforcer_end    : id:int operandum:int?
#   state_change      : from:str to:str
# ---
"""
PERIOD = 0.3


def host_okl(
    n: int,
    reinforced: set[int],
    *,
    host_delay: float = 0.003,
    shift: dict[int, float] | None = None,
    decision: str = "host",
    dsl: str = DSL,
) -> str:
    # device_time is the device's clock after ``sync`` set it to the host's time at run0.
    start, run0 = 1000.0, 1000.2
    shift = shift or {}
    lines = [
        "# OKL v1",
        '# session_name = "hil-crf"',
        '# clock_type = "monotonic"',
        f"# session_start = {start}",
        "# meta:",
        f'#   dsl = "{dsl}"',
        f'#   decision = "{decision}"',
        EVENTS.rstrip("\n"),
        f"{run0}\tstate_change\tIDLE\tRUNNING",
    ]
    rid = 0
    for k in range(1, n + 1):
        rel = k * PERIOD + shift.get(k, 0.0)
        t = run0 + rel + host_delay
        lines.append(f"{t:.6f}\tresponse\t{k}\t-\t{run0 + rel:.6f}")
        if k in reinforced:
            rid += 1
            lines.append(f"{t:.6f}\treinforcer_start\t{rid}\t1.0\t-")
            lines.append(f"{t:.6f}\tstate_change\tRUNNING\tREINFORCING")
            lines.append(f"{t + 0.051:.6f}\treinforcer_end\t{rid}\t-")
            lines.append(f"{t + 0.051:.6f}\tstate_change\tREINFORCING\tRUNNING")
    return "\n".join(lines) + "\n"


def device_okl(n: int, reinforced: set[int], *, decision: str = "device", dsl: str = DSL) -> str:
    start = 2000.0
    lines = [
        "# OKL v1",
        '# session_name = "hil-crf"',
        '# clock_type = "device"',
        f"# session_start = {start}",
        "# meta:",
        f'#   dsl = "{dsl}"',
        f'#   decision = "{decision}"',
        EVENTS.rstrip("\n"),
    ]
    rid = 0
    for k in range(1, n + 1):
        t = start + k * PERIOD
        lines.append(f"{t:.6f}\tresponse\t{k}\t0\t-")
        if k in reinforced:
            rid += 1
            lines.append(f"{t + 0.000003:.6f}\treinforcer_start\t{rid}\t1.0\t0")
            lines.append(f"{t + 0.050003:.6f}\treinforcer_end\t{rid}\t0")
    return "\n".join(lines) + "\n"


def write_pair(tmp_path: Path, host: str, device: str) -> tuple[Path, Path]:
    h, d = tmp_path / "host.okl", tmp_path / "device.okl"
    h.write_text(host, encoding="utf-8")
    d.write_text(device, encoding="utf-8")
    return h, d


REINFORCED = {1, 7, 12, 30}


def test_load_okl_puts_both_logs_on_the_same_origin(tmp_path: Path) -> None:
    h, d = write_pair(tmp_path, host_okl(40, REINFORCED), device_okl(40, REINFORCED))
    host, device = load_okl(h), load_okl(d)
    assert host.decision == "host" and device.decision == "device"
    assert host.response_times[0] == pytest.approx(0.3)
    assert device.response_times[0] == pytest.approx(0.3)
    assert host.reinforced_ids == [1, 7, 12, 30]
    assert device.reinforced_ids == [1, 7, 12, 30]
    assert host.reinforcer_times[0] == pytest.approx(0.303)
    assert device.reinforcer_times[0] == pytest.approx(0.300003)


def test_matching_pair_agrees_and_reports_time_differences(tmp_path: Path) -> None:
    h, d = write_pair(tmp_path, host_okl(40, REINFORCED), device_okl(40, REINFORCED))
    result = compare(load_okl(h), load_okl(d))
    assert result.n_responses == (40, 40)
    assert result.n_reinforcers == (4, 4)
    assert result.divergence_at is None
    assert result.compared_responses == 40
    assert result.agreement == (40, 40)
    assert result.mismatched_ids == []
    assert result.reinforcer_time_diff_ms is not None
    assert result.reinforcer_time_diff_ms.count == 4
    assert result.reinforcer_time_diff_ms.median == pytest.approx(2.997, abs=1e-3)
    assert result.reinforcer_time_diff_ms.maximum == pytest.approx(2.997, abs=1e-3)
    assert result.device_latency_us is not None
    assert result.device_latency_us.median == pytest.approx(3.0, abs=0.5)
    assert result.ok


def test_one_different_decision_is_found(tmp_path: Path) -> None:
    h, d = write_pair(tmp_path, host_okl(40, REINFORCED), device_okl(40, REINFORCED - {12} | {13}))
    result = compare(load_okl(h), load_okl(d))
    assert result.n_reinforcers == (4, 4)
    assert result.agreement == (38, 40)
    assert result.mismatched_ids == [12, 13]
    # The time difference is taken only on responses both sides reinforced.
    assert result.reinforcer_time_diff_ms is not None
    assert result.reinforcer_time_diff_ms.count == 3
    assert not result.ok


def test_reinforcer_count_difference_is_not_ok(tmp_path: Path) -> None:
    h, d = write_pair(tmp_path, host_okl(40, REINFORCED), device_okl(40, REINFORCED - {30}))
    result = compare(load_okl(h), load_okl(d))
    assert result.n_reinforcers == (4, 3)
    assert result.mismatched_ids == [30]
    assert not result.ok


def test_response_sequence_divergence_limits_the_compared_range(tmp_path: Path) -> None:
    host = host_okl(40, REINFORCED, shift=dict.fromkeys(range(20, 41), 0.25))
    h, d = write_pair(tmp_path, host, device_okl(40, REINFORCED))
    result = compare(load_okl(h), load_okl(d))
    assert result.divergence_at == 20
    assert result.compared_responses == 19
    assert result.agreement == (19, 19)
    assert result.reinforcer_time_diff_ms is not None
    assert result.reinforcer_time_diff_ms.count == 3  # responses 1, 7, 12
    assert not result.ok


def test_response_count_difference_is_not_ok(tmp_path: Path) -> None:
    h, d = write_pair(tmp_path, host_okl(38, REINFORCED - {30}), device_okl(40, REINFORCED - {30}))
    result = compare(load_okl(h), load_okl(d))
    assert result.n_responses == (38, 40)
    assert result.divergence_at == 39
    assert result.compared_responses == 38
    assert not result.ok


def test_host_latency_comes_from_the_summary(tmp_path: Path) -> None:
    h, d = write_pair(tmp_path, host_okl(40, REINFORCED), device_okl(40, REINFORCED))
    summary = tmp_path / "host.json"
    summary.write_text(
        json.dumps({"device": {"kind": "report", "lat_us": [900, 2500, 4200, 3000]}})
    )
    result = compare(load_okl(h), load_okl(d), host_summary=hil_diff.load_lat_us(summary))
    assert result.host_latency_us is not None
    assert result.host_latency_us.median == pytest.approx(2750)
    assert result.host_latency_us.maximum == pytest.approx(4200)


def test_main_prints_table_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    h, d = write_pair(tmp_path, host_okl(40, REINFORCED), device_okl(40, REINFORCED))
    assert hil_diff.main([str(h), str(d)]) == 0
    out = capsys.readouterr().out
    assert "decision agreement" in out
    assert "40/40" in out
    assert "not in the OKL" in out  # host latency without --host-summary

    d.write_text(device_okl(40, REINFORCED - {12} | {13}), encoding="utf-8")
    assert hil_diff.main([str(h), str(d), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["mismatched_ids"] == [12, 13]


def test_main_rejects_swapped_or_different_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    h, d = write_pair(tmp_path, host_okl(10, {1}), device_okl(10, {1}))
    assert hil_diff.main([str(d), str(h)]) == 2
    assert "decision" in capsys.readouterr().err

    d.write_text(device_okl(10, {1}, dsl="CRF"), encoding="utf-8")
    assert hil_diff.main([str(h), str(d)]) == 2
    assert "dsl" in capsys.readouterr().err
