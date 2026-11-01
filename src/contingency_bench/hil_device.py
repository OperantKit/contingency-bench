"""Run a session whose contingency is decided on the Teensy (``--decision device``).

Where the contingency is decided is chosen by the researcher and only by the
researcher: ``host`` (the default, ``hil_crf.run_crf``) or ``device`` (this
module). Nothing switches between them on its own. A schedule the device
cannot decide stops the run before it starts — before the OKL file is
created and before ``start`` is sent — with a message that says how to run
it on the host instead.

The firmware in ``firmware/teensy41`` decides every atomic schedule
(:data:`SUPPORTED_ON_DEVICE`); a random one needs a seed in the DSL
(``@algorithm("fleshler-hoffman", seed=N)``). The host sends
``load`` (``schedule: {type, params}`` in the conformance fixture form, the
reinforcer duration and ``stop_after``), then ``start``, and turns the
device's events into OKL lines. Session time is the device clock: the
header says ``clock_type = "device"``, ``session_start`` is the ``ts`` of
``started`` and every line carries the device's ``ts``. The header's
``meta`` always says ``decision``; :func:`check_decision_clock` refuses a
pair of decision and clock type that do not belong together.

``probe`` sends one conformance step at a time without real time, so a
fixture's ``steps`` can be checked against the device (:func:`probe_fixture`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DECISIONS = ("host", "device")
DEVICE_CLOCK_TYPE = "device"
SUPPORTED_ON_DEVICE = ("CRF", "EXT", "FR", "VR", "RR", "FI", "VI", "RI", "FT", "VT", "RT")
SEEDED_ON_DEVICE = ("VR", "RR", "VI", "RI", "VT", "RT")
_REPLY_TIMEOUT_S = 2.0
_POLL_S = 0.001
_SETTLE_S = 0.2

Port = Any
Clock = Callable[[], float]


class DeviceUnsupportedError(ValueError):
    """The schedule cannot be decided on the device; nothing was started."""


def _host_hint(dsl: str, reason: str) -> str:
    supported = ", ".join(SUPPORTED_ON_DEVICE)
    return (
        f'schedule "{dsl}" cannot run on the device ({reason}; supported: {supported}); '
        "run it on the host with --decision host"
    )


def check_decision_clock(decision: str, clock_type: str) -> None:
    """Refuse a header whose decision and clock type do not belong together."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, not {decision!r}")
    if decision == "device" and clock_type != DEVICE_CLOCK_TYPE:
        raise ValueError('decision = "device" needs clock_type = "device"')
    if decision == "host" and clock_type == DEVICE_CLOCK_TYPE:
        raise ValueError('decision = "host" runs on the host clock, not clock_type = "device"')


def _parse_spec(dsl: str) -> dict[str, Any]:
    from contingency.bridge.spec import to_spec
    from contingency_dsl import parse
    from contingency_dsl.annotations.apparatus.extension import ApparatusExtension
    from contingency_dsl.annotations.measurement.extension import MeasurementExtension
    from contingency_dsl.annotations.stimulus.extension import StimulusExtension
    from contingency_dsl.annotations.subjects.extension import SubjectsExtension
    from contingency_dsl.annotations.temporal.extension import TemporalExtension
    from contingency_dsl.extensions import ExtensionRegistry

    registry = ExtensionRegistry.build(
        ApparatusExtension(),
        MeasurementExtension(),
        StimulusExtension(),
        SubjectsExtension(),
        TemporalExtension(),
    )
    spec = to_spec(parse(dsl, extensions=registry))
    if not isinstance(spec, dict):
        raise TypeError(f"to_spec returned {type(spec).__name__}, not a dict")
    return spec


def device_schedule(dsl: str) -> dict[str, Any]:
    """Map a contingency-dsl expression to the ``load`` schedule, or refuse it.

    Every atomic schedule maps to its conformance-fixture spec. A random one
    (:data:`SEEDED_ON_DEVICE`) needs a seed, because the device has no
    entropy source: without ``@algorithm(..., seed=N)`` it is refused here,
    before anything starts, rather than by the device.
    """
    try:
        spec = _parse_spec(dsl)
    except Exception as exc:  # the parser's and to_spec's own errors carry the detail
        raise DeviceUnsupportedError(_host_hint(dsl, f"not mapped: {exc}")) from exc
    kind = spec.get("type")
    if kind not in SUPPORTED_ON_DEVICE:
        raise DeviceUnsupportedError(
            _host_hint(dsl, f"{kind} is not an atomic schedule this firmware decides")
        )
    params = dict(spec.get("params", {}))
    if kind == "FR":
        n = float(params.get("n", 0))
        if not (n >= 1 and n.is_integer()):
            raise DeviceUnsupportedError(_host_hint(dsl, "FR needs an integer n >= 1"))
        params["n"] = int(n)
    if kind in SEEDED_ON_DEVICE and params.get("seed") is None:
        raise DeviceUnsupportedError(
            f'schedule "{dsl}" draws random numbers and the device needs a seed for it; '
            'add @algorithm("fleshler-hoffman", seed=N) to the schedule, '
            "or run it on the host with --decision host"
        )
    return {"type": kind, "params": params}


def _send(port: Port, command: dict[str, Any]) -> None:
    port.write((json.dumps(command) + "\n").encode())
    port.flush()


def _wait(clock: Clock, seconds: float) -> None:
    advance = getattr(clock, "advance", None)
    if callable(advance):
        advance(seconds)
    else:
        time.sleep(seconds)


def _read_json(port: Port) -> dict[str, Any] | None:
    raw = port.readline()
    text = (raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)).strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _await(port: Port, clock: Clock, kinds: set[str], timeout_s: float) -> dict[str, Any] | None:
    deadline = clock() + timeout_s
    while clock() < deadline:
        data = _read_json(port)
        if data is None:
            _wait(clock, _POLL_S)
            continue
        if data.get("kind") in kinds:
            return data
    return None


def _quiet(port: Port, clock: Clock) -> None:
    _send(port, {"cmd": "stop"})
    _wait(clock, _SETTLE_S)
    reset = getattr(port, "reset_input_buffer", None)
    if callable(reset):
        reset()


def _load(port: Port, clock: Clock, dsl: str, command: dict[str, Any]) -> dict[str, Any]:
    _send(port, command)
    reply = _await(port, clock, {"loaded", "error"}, _REPLY_TIMEOUT_S)
    if reply is None:
        raise OSError("the device did not answer load")
    if reply["kind"] == "error":
        raise DeviceUnsupportedError(_host_hint(dsl, f"device error {reply.get('code')}"))
    return reply


@dataclass(frozen=True)
class DeviceRun:
    """Outcome of one run decided on the device."""

    dsl: str
    target_reinforcers: int
    reinforcer_duration_s: float
    finished: bool
    n_reinforcers: int
    n_responses: int
    elapsed_s: float
    seq_gaps: int
    decision: str = "device"
    firmware: str | None = None
    session_start: float | None = None
    host_minus_device_ms: list[float] = field(default_factory=list)
    device: dict[str, Any] | None = None


def run_device(
    open_port: Callable[[], Port],
    *,
    dsl: str = "CRF",
    reinforcers: int = 200,
    reinforcer_duration: float = 0.05,
    clock: Clock | None = None,
    okl_path: Path | None = None,
    max_duration: float = 600.0,
    report_timeout: float = 5.0,
) -> DeviceRun:
    """Load ``dsl`` on the device, run until ``reinforcers``, and fetch its report.

    Raises :class:`DeviceUnsupportedError` before anything starts when the
    device cannot decide the schedule.
    """
    from experiment_core import ReinforcerEndEvent, ReinforcerStartEvent, ResponseEvent
    from session_recorder import OklSink

    if reinforcers < 1:
        raise ValueError("reinforcers must be >= 1")
    clock = clock or time.monotonic
    schedule = device_schedule(dsl)  # refuses before the port is touched
    check_decision_clock("device", DEVICE_CLOCK_TYPE)

    port = open_port()
    try:
        _quiet(port, clock)
        loaded = _load(
            port,
            clock,
            dsl,
            {
                "cmd": "load",
                "schedule": schedule,
                "reinforcer": {"op": 0, "duration": reinforcer_duration},
                "stop_after": {"reinforcers": reinforcers},
            },
        )
        firmware = loaded.get("firmware")
        _send(port, {"cmd": "start"})
        host_start = clock()
        started = _await(port, clock, {"started"}, _REPLY_TIMEOUT_S)
        if started is None:
            raise OSError("the device did not answer start")
        session_start = float(started["ts"])
        last_seq = int(started.get("seq", 0))

        okl = None
        if okl_path is not None:
            okl = OklSink(okl_path)
            okl.write_header(
                session_name="hil-crf",
                clock_type=DEVICE_CLOCK_TYPE,
                session_start=session_start,
                metadata={
                    "dsl": dsl,
                    "backend": "serial",
                    "decision": "device",
                    "device_firmware": str(firmware),
                    "reinforcer_duration_s": reinforcer_duration,
                },
            )

        n_resp = n_sr = gaps = 0
        lags: list[float] = []
        stopped = False
        deadline = host_start + max_duration
        while clock() < deadline:
            data = _read_json(port)
            if data is None:
                _wait(clock, _POLL_S)
                continue
            kind = data.get("kind")
            seq = data.get("seq")
            if isinstance(seq, int):
                gaps += max(0, seq - last_seq - 1)
                last_seq = seq
            ts = data.get("ts")
            if kind == "response":
                n_resp += 1
                lags.append(round((clock() - host_start - (float(ts) - session_start)) * 1e3, 3))
                if okl is not None:
                    okl.emit(ResponseEvent(id=n_resp, timestamp=float(ts), operandum=0))
            elif kind == "reinforcer_start":
                n_sr += 1
                if okl is not None:
                    okl.emit(ReinforcerStartEvent(id=n_sr, timestamp=float(ts), operandum=0))
            elif kind == "reinforcer_end":
                if okl is not None:
                    okl.emit(ReinforcerEndEvent(id=n_sr, timestamp=float(ts), operandum=0))
                if n_sr >= reinforcers:
                    _send(port, {"cmd": "stop"})  # the device stops itself too
            elif kind == "stopped":
                stopped = True
                break
        elapsed = clock() - host_start
        if not stopped:
            _send(port, {"cmd": "stop"})
        _send(port, {"cmd": "report"})
        report = _await(port, clock, {"report"}, report_timeout)
    finally:
        port.close()

    return DeviceRun(
        dsl=dsl,
        target_reinforcers=reinforcers,
        reinforcer_duration_s=reinforcer_duration,
        finished=n_sr >= reinforcers,
        n_reinforcers=n_sr,
        n_responses=n_resp,
        elapsed_s=round(elapsed, 3),
        seq_gaps=gaps,
        firmware=None if firmware is None else str(firmware),
        session_start=session_start,
        host_minus_device_ms=lags,
        device=report,
    )


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one conformance fixture sent step by step with ``probe``."""

    name: str
    n_steps: int
    mismatches: list[dict[str, Any]]


def probe_fixture(
    port: Port, fixture: dict[str, Any], *, clock: Clock | None = None
) -> ProbeResult:
    """Load the fixture's schedule and compare each ``probe`` outcome with ``expect``."""
    clock = clock or time.monotonic
    name = str(fixture.get("name", "?"))
    _quiet(port, clock)
    _load(port, clock, name, {"cmd": "load", "schedule": fixture["schedule"]})
    mismatches: list[dict[str, Any]] = []
    for i, step in enumerate(fixture["steps"]):
        _send(port, {"cmd": "probe", "now": step["now"], "event": step["event"]})
        reply = _await(port, clock, {"outcome", "error"}, _REPLY_TIMEOUT_S)
        got = None if reply is None else {k: v for k, v in reply.items() if k != "kind"}
        if got != step["expect"]:
            mismatches.append({"step": i, "expect": step["expect"], "got": got})
    return ProbeResult(name=name, n_steps=len(fixture["steps"]), mismatches=mismatches)
