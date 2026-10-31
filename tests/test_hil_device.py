"""Choosing where the contingency is decided: on the host (default) or on the device.

The fake device port behaves like ``firmware/teensy41`` in device mode: it
accepts ``load`` for CRF and FR only, answers ``probe`` without real time,
and after ``start`` presses every 300 ms on its own clock, decides each
press itself and reports ``response`` / ``reinforcer_start`` /
``reinforcer_end`` / ``stopped`` with a ``seq`` that goes up by one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("experiment_io")
pytest.importorskip("serial")

from experiment_io.hw.backends.virtual import ManualClock  # noqa: E402

from contingency_bench import hil_crf  # noqa: E402
from contingency_bench.hil_device import (  # noqa: E402
    DeviceUnsupportedError,
    check_decision_clock,
    device_schedule,
    probe_fixture,
    run_device,
)

from .test_hil_crf import _FakeChamberPort  # noqa: E402

_IRT = 0.3
_DEVICE_EPOCH = 500.0


class _FakeDevicePort:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.commands: list[dict[str, object]] = []
        self._out: list[tuple[float, bytes]] = []
        self._buffer = b""
        self._seq = 0
        self._n = 0
        self._count = 0
        self._probe_count = 0
        self._loaded = False
        self._running = False
        self._duration = 0.05
        self._stop_reinforcers = -1
        self._next_press: float | None = None
        self._feeder_off: float | None = None
        self.n_responses = 0
        self.n_reinforcers = 0
        self.closed = 0

    def _ts(self, t: float) -> float:
        return round(t + _DEVICE_EPOCH, 6)

    def _say(self, t: float, obj: dict[str, object], with_seq: bool = True) -> None:
        if with_seq:
            obj = {**obj, "seq": self._seq}
            self._seq += 1
        self._out.append((t + 0.002, (json.dumps(obj) + "\n").encode()))

    def _advance(self) -> None:
        now = self.clock()
        while self._running:
            due = [t for t in (self._next_press, self._feeder_off) if t is not None]
            if not due or min(due) > now:
                break
            t = min(due)
            if self._feeder_off is not None and t == self._feeder_off:
                self._feeder_off = None
                self._say(t, {"kind": "reinforcer_end", "op": 0, "ts": self._ts(t)})
                if 0 < self._stop_reinforcers <= self.n_reinforcers:
                    self._stop(t)
                continue
            self._count += 1
            self.n_responses += 1
            reinforced = self._count >= self._n
            self._say(t, {"kind": "response", "op": 0, "ts": self._ts(t), "reinforced": reinforced})
            if reinforced:
                self._count = 0
                self.n_reinforcers += 1
                self._say(t, {"kind": "reinforcer_start", "op": 0, "ts": self._ts(t)})
                self._feeder_off = t + self._duration
                self._next_press = t + self._duration + _IRT
            else:
                self._next_press = t + _IRT

    def _stop(self, t: float) -> None:
        self._running = False
        self._loaded = False
        self._next_press = None
        self._say(
            t,
            {
                "kind": "stopped",
                "ts": self._ts(t),
                "n_responses": self.n_responses,
                "n_reinforcers": self.n_reinforcers,
            },
        )

    def readline(self) -> bytes:
        self._advance()
        now = self.clock()
        ready = sorted((x for x in self._out if x[0] <= now), key=lambda x: x[0])
        self._out = [x for x in self._out if x[0] > now]
        for _, raw in ready:
            self._buffer += raw
        if b"\n" not in self._buffer:
            return b""
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line + b"\n"

    def write(self, data: bytes) -> int:
        now = self.clock()
        cmd = json.loads(data.decode())
        self.commands.append(cmd)
        kind = cmd.get("cmd")
        if kind == "load":
            sched = cmd["schedule"]
            assert isinstance(sched, dict)
            if sched["type"] == "CRF":
                self._n = 1
            elif sched["type"] == "FR":
                self._n = int(sched["params"]["n"])
            else:
                self._say(now, {"kind": "error", "code": "unsupported_schedule", "msg": "no"})
                return len(data)
            reinforcer = cmd.get("reinforcer") or {}
            self._duration = float(reinforcer.get("duration", 0.05))  # type: ignore[union-attr]
            stop_after = cmd.get("stop_after") or {}
            self._stop_reinforcers = int(stop_after.get("reinforcers", -1))  # type: ignore[union-attr]
            self._loaded = True
            self._probe_count = 0
            self._say(
                now,
                {"kind": "loaded", "firmware": "fake-device", "supported": ["CRF", "FR"]},
                with_seq=False,
            )
        elif kind == "probe":
            reinforced = False
            if cmd.get("event") is not None:
                self._probe_count += 1
                if self._probe_count >= self._n:
                    self._probe_count = 0
                    reinforced = True
            out: dict[str, object] = {"kind": "outcome", "reinforced": reinforced}
            if reinforced:
                out["reinforcer"] = {"time": cmd["now"], "magnitude": 1.0, "label": "SR+"}
            self._say(now, out, with_seq=False)
        elif kind == "start":
            self._seq = 0
            self._running = True
            self._say(now, {"kind": "started", "ts": self._ts(now)})
            self._next_press = now + _IRT
        elif kind == "stop":
            if self._running:
                self._stop(now)
        elif kind == "report":
            self._say(now, {"kind": "report", "n": self.n_reinforcers, "lat_us": [1] * 3}, False)
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._advance()
        self._out = []
        self._buffer = b""

    def close(self) -> None:
        self.closed += 1


def _header(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.startswith("#")]


# (a) no --decision: the host decides; the device gets no load.
def test_without_decision_the_host_decides_and_no_load_is_sent(tmp_path, monkeypatch):
    clock = ManualClock()
    port = _FakeChamberPort(clock)
    monkeypatch.setattr(hil_crf, "_open_serial", lambda path, baud: port)
    monkeypatch.setattr(hil_crf, "_default_clock", lambda: clock)
    okl = tmp_path / "host.okl"

    code = hil_crf.main(["--port", "/dev/fake", "--reinforcers", "3", "--okl", str(okl)])

    assert code == 0
    header = _header(okl)
    assert any(line.replace(" ", "") == '#decision="host"' for line in header), header
    assert not any(c.get("cmd") == "load" for c in port.commands)


# (b) --decision device with a schedule the device cannot decide: exit 2 before anything.
def test_device_decision_with_vi_stops_before_the_session(tmp_path, monkeypatch, capsys):
    clock = ManualClock()
    port = _FakeDevicePort(clock)
    monkeypatch.setattr(hil_crf, "_open_serial", lambda path, baud: port)
    monkeypatch.setattr(hil_crf, "_default_clock", lambda: clock)
    okl = tmp_path / "device.okl"

    code = hil_crf.main(
        ["--port", "/dev/fake", "--dsl", "VI 30", "--decision", "device", "--okl", str(okl)]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "VI 30" in err
    assert "--decision host" in err
    assert not okl.exists()
    assert not any(c.get("cmd") == "start" for c in port.commands)


def test_a_load_error_from_the_device_also_stops_before_start(tmp_path):
    clock = ManualClock()
    port = _FakeDevicePort(clock)
    okl = tmp_path / "device.okl"

    def refuse(data: bytes) -> int:
        cmd = json.loads(data.decode())
        port.commands.append(cmd)
        if cmd.get("cmd") == "load":
            port._out.append((clock(), b'{"kind":"error","code":"bad_params","msg":"x"}\n'))
        return len(data)

    port.write = refuse  # type: ignore[method-assign]
    with pytest.raises(DeviceUnsupportedError, match="--decision host"):
        run_device(lambda: port, dsl="FR 5", reinforcers=3, clock=clock, okl_path=okl)
    assert not okl.exists()
    assert not any(c.get("cmd") == "start" for c in port.commands)


# (c) --decision device with CRF: the header says the device decided and the device clock.
def test_device_decision_with_crf_writes_device_header_and_device_times(tmp_path, monkeypatch):
    clock = ManualClock()
    port = _FakeDevicePort(clock)
    monkeypatch.setattr(hil_crf, "_open_serial", lambda path, baud: port)
    monkeypatch.setattr(hil_crf, "_default_clock", lambda: clock)
    okl = tmp_path / "device.okl"
    summary = tmp_path / "summary.json"

    code = hil_crf.main(
        [
            "--port",
            "/dev/fake",
            "--dsl",
            "CRF",
            "--decision",
            "device",
            "--reinforcers",
            "4",
            "--okl",
            str(okl),
            "--summary",
            str(summary),
        ]
    )

    assert code == 0
    header = [line.replace(" ", "") for line in _header(okl)]
    assert '#decision="device"' in header
    assert '#clock_type="device"' in header
    assert '#device_firmware="fake-device"' in header
    body = [line for line in okl.read_text().splitlines() if line and not line.startswith("#")]
    times = [float(line.split("\t")[0]) for line in body if "\t" in line]
    assert times and min(times) >= _DEVICE_EPOCH
    assert times == sorted(times)
    data = json.loads(summary.read_text())
    assert data["decision"] == "device"
    assert data["n_reinforcers"] == 4
    assert data["finished"] is True
    assert data["device"]["kind"] == "report"
    load = next(c for c in port.commands if c.get("cmd") == "load")
    assert load["schedule"] == {"type": "CRF", "params": {}}
    assert load["stop_after"] == {"reinforcers": 4}


def test_device_run_counts_fr_and_checks_seq(tmp_path):
    clock = ManualClock()
    port = _FakeDevicePort(clock)

    run = run_device(lambda: port, dsl="FR 3", reinforcers=2, clock=clock)

    assert run.finished
    assert run.n_reinforcers == 2
    assert run.n_responses == 6
    assert run.seq_gaps == 0


def test_device_schedule_maps_only_crf_and_fr():
    assert device_schedule("CRF") == {"type": "CRF", "params": {}}
    assert device_schedule("FR 5") == {"type": "FR", "params": {"n": 5}}
    for dsl in ("VI 30", "FI 10", "EXT", "Conc(VI 30, VI 60)"):
        with pytest.raises(DeviceUnsupportedError):
            device_schedule(dsl)


def test_decision_and_clock_must_agree():
    check_decision_clock("device", "device")
    check_decision_clock("host", "monotonic")
    with pytest.raises(ValueError):
        check_decision_clock("device", "monotonic")
    with pytest.raises(ValueError):
        check_decision_clock("host", "device")
    with pytest.raises(ValueError):
        check_decision_clock("auto", "device")


@pytest.mark.parametrize("name", ["fr_basic", "crf_basic"])
def test_probe_matches_the_conformance_fixture(name):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "conformance" / f"{name}.json").read_text()
    )
    clock = ManualClock()
    port = _FakeDevicePort(clock)

    result = probe_fixture(port, fixture, clock=clock)

    assert result.mismatches == []
    assert result.n_steps == len(fixture["steps"])
