"""Host-side CRF run against the Teensy fake chamber, with a simulated port.

The fake port behaves like the firmware in ``firmware/teensy41``: after
``sync`` it responds every 300 ms on its own clock (offset from the host's),
does not respond while the feeder is on, logs the latency from each response
to the reinforcer command, and answers ``report``. Its lines become readable
a few milliseconds after the device time they carry, as over USB serial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("experiment_io")
pytest.importorskip("serial")

from experiment_io.hw.backends.virtual import ManualClock  # noqa: E402

from contingency_bench.hil_crf import main, run_crf  # noqa: E402

_DELAYS = (0.001, 0.00125, 0.003, 0.007)
_IRT = 0.3


class _FakeChamberPort:
    """pyserial-like port (``timeout=0``) speaking the fake chamber's protocol."""

    def __init__(self, clock: ManualClock, device_offset: float = 1234.5) -> None:
        self.clock = clock
        self.device_offset = device_offset
        self._pending: list[tuple[float, bytes]] = []
        self._buffer = b""
        self._next_response: float | None = None
        self._last_response = 0.0
        self._feeder_off = 0.0
        self._seq = 0
        self.lat_us: list[int] = []
        self.commands: list[dict[str, object]] = []
        self.opened = 1
        self.closed = 0

    def _emit_due(self) -> None:
        now = self.clock()
        while self._next_response is not None and self._next_response <= now:
            t = self._next_response
            delay = _DELAYS[self._seq % len(_DELAYS)]
            line = {"op": 0, "ts": t + self.device_offset, "kind": "response", "seq": self._seq}
            self._pending.append((t + delay, (json.dumps(line) + "\n").encode()))
            self._seq += 1
            self._last_response = t
            self._next_response = t + _IRT

    def readline(self) -> bytes:
        self._emit_due()
        now = self.clock()
        if self._pending and self._pending[0][0] <= now:
            return self._pending.pop(0)[1]
        return b""

    def write(self, data: bytes) -> int:
        if not isinstance(data, bytes):
            raise TypeError("unicode strings are not supported, please encode to bytes")
        self._buffer += data
        while b"\n" in self._buffer:
            raw, self._buffer = self._buffer.split(b"\n", 1)
            if raw.strip():
                self._command(json.loads(raw))
        return len(data)

    def _command(self, cmd: dict[str, object]) -> None:
        self._emit_due()
        self.commands.append(cmd)
        now = self.clock()
        kind = cmd["cmd"]
        if kind == "sync":
            self.device_offset = float(cmd["t"]) - now  # type: ignore[arg-type]
            self.lat_us.clear()
            self._seq = 0
            self._next_response = now + _IRT
        elif kind == "stop":
            self._next_response = None
        elif kind == "reinforcer":
            self.lat_us.append(round((now - self._last_response) * 1e6))
            duration = float(cmd.get("duration", 0.5))  # type: ignore[arg-type]
            self._feeder_off = now + duration
            self._next_response = self._feeder_off + _IRT
        elif kind == "report":
            line = {"kind": "report", "n": len(self.lat_us), "lat_us": list(self.lat_us)}
            self._pending.insert(0, (now, (json.dumps(line) + "\n").encode()))

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self.closed += 1


def _opener(port: _FakeChamberPort):
    def open_port() -> _FakeChamberPort:
        return port

    return open_port


def test_crf_run_reaches_the_reinforcer_count_and_fetches_the_device_report(tmp_path: Path):
    clock = ManualClock()
    port = _FakeChamberPort(clock)
    okl = tmp_path / "run.okl"

    run = run_crf(
        _opener(port), reinforcers=20, reinforcer_duration=0.05, clock=clock, okl_path=okl
    )

    assert run.finished
    assert run.n_reinforcers == 20
    assert run.n_responses == 20
    assert run.device is not None and run.device["n"] == 20
    assert len(run.device["lat_us"]) == 20
    reinforcer_cmds = [c for c in port.commands if c["cmd"] == "reinforcer"]
    assert len(reinforcer_cmds) == 20
    assert all(c["duration"] == 0.05 for c in reinforcer_cmds)
    # The device clock is never aligned with the host: the receipt minus the
    # device time is the transmission delay plus the poll granularity, in ms.
    assert len(run.host_minus_device_ms) == 20
    assert all(0.9 <= lag <= 8.5 for lag in run.host_minus_device_ms)
    kinds = [
        line.split("\t")[1] for line in okl.read_text().splitlines() if not line.startswith("#")
    ]
    assert kinds.count("response") == 20
    assert kinds.count("reinforcer_start") == 20
    assert kinds.count("reinforcer_end") == 20


def test_the_device_is_stopped_before_the_report_and_the_port_is_reopened():
    clock = ManualClock()
    port = _FakeChamberPort(clock)

    run_crf(_opener(port), reinforcers=3, reinforcer_duration=0.05, clock=clock)

    kinds = [c["cmd"] for c in port.commands]
    assert kinds[0] == "stop"
    assert kinds[-2:] == ["stop", "report"]
    assert "sync" in kinds
    assert port.closed >= 2


def test_a_ratio_schedule_from_the_dsl_counts_responses(tmp_path: Path):
    clock = ManualClock()
    port = _FakeChamberPort(clock)

    run = run_crf(_opener(port), dsl="FR 3", reinforcers=4, reinforcer_duration=0.05, clock=clock)

    assert run.finished
    assert run.n_reinforcers == 4
    assert run.n_responses == 12
    assert run.device is not None and run.device["n"] == 4


def test_a_silent_device_ends_at_max_duration_without_finishing():
    clock = ManualClock()
    port = _FakeChamberPort(clock)
    port.write = lambda data: len(data)  # type: ignore[method-assign]

    run = run_crf(_opener(port), reinforcers=5, clock=clock, max_duration=2.0, report_timeout=0.05)

    assert not run.finished
    assert run.n_reinforcers == 0
    assert run.device is None


def test_the_cli_writes_the_summary_json_that_the_latency_analysis_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from contingency_bench import hil_crf
    from contingency_bench.hil_latency import load_device_report

    clock = ManualClock()
    port = _FakeChamberPort(clock)
    monkeypatch.setattr(hil_crf, "_open_serial", lambda path, baud: port)
    monkeypatch.setattr(hil_crf, "_default_clock", lambda: clock)
    summary = tmp_path / "host.json"

    code = main(
        [
            "--port",
            "/dev/fake",
            "--reinforcers",
            "5",
            "--duration",
            "0.05",
            "--okl",
            str(tmp_path / "run.okl"),
            "--summary",
            str(summary),
        ]
    )

    assert code == 0
    data = json.loads(summary.read_text())
    assert data["n_reinforcers"] == 5
    assert data["dsl"] == "CRF"
    assert len(load_device_report(summary)) == 5


def test_the_cli_exits_1_when_the_session_does_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from contingency_bench import hil_crf

    clock = ManualClock()
    port = _FakeChamberPort(clock)
    port.write = lambda data: len(data)  # type: ignore[method-assign]
    monkeypatch.setattr(hil_crf, "_open_serial", lambda path, baud: port)
    monkeypatch.setattr(hil_crf, "_default_clock", lambda: clock)

    code = main(
        [
            "--port",
            "/dev/fake",
            "--reinforcers",
            "5",
            "--max-duration",
            "1",
            "--report-timeout",
            "0.05",
        ]
    )

    assert code == 1
