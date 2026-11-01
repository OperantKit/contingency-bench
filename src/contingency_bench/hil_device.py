"""Run a session whose contingency is decided on the Teensy (``--decision device``).

Where the contingency is decided is chosen by the researcher and only by the
researcher: ``host`` (the default, ``hil_crf.run_crf``) or ``device`` (this
module). Nothing switches between them on its own. A schedule the device
cannot decide stops the run before it starts — before the OKL file is
created and before ``start`` is sent — with a message that says how to run
it on the host instead.

Driving the device is experiment-io's (``experiment_io.device``); this module
turns a contingency-dsl expression into the ``load`` schedule and writes the
device's events as OKL lines. The firmware in ``firmware/teensy41`` decides
every atomic schedule (:data:`SUPPORTED_ON_DEVICE`); a random one needs a
seed in the DSL (``@algorithm("fleshler-hoffman", seed=N)``). Session time
is the device clock: the header says ``clock_type = "device"``,
``session_start`` is the ``ts`` of ``started`` and every line carries the
device's ``ts``. The header's ``meta`` always says ``decision``;
:func:`check_decision_clock` refuses a pair of decision and clock type that
do not belong together.

``probe`` sends one conformance step at a time without real time, so a
fixture's ``steps`` can be checked against the device (:func:`probe_fixture`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiment_io.device import (
    DECISIONS,
    DEVICE_CLOCK_TYPE,
    SEEDED_ON_DEVICE,
    SUPPORTED_ON_DEVICE,
    DeviceEvent,
    DeviceUnsupportedError,
    ProbeResult,
    check_decision_clock,
    device_schedule_from_spec,
    drive_device,
    probe_fixture,
)

__all__ = [
    "DECISIONS",
    "DEVICE_CLOCK_TYPE",
    "SEEDED_ON_DEVICE",
    "SUPPORTED_ON_DEVICE",
    "DeviceRun",
    "DeviceUnsupportedError",
    "ProbeResult",
    "check_decision_clock",
    "device_schedule",
    "probe_fixture",
    "run_device",
]

Port = Any
Clock = Callable[[], float]


def _host_hint(dsl: str, reason: str) -> str:
    supported = ", ".join(SUPPORTED_ON_DEVICE)
    return (
        f'schedule "{dsl}" cannot run on the device ({reason}; supported: {supported}); '
        "run it on the host with --decision host"
    )


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
    return device_schedule_from_spec(spec, dsl)


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
    schedule = device_schedule(dsl)  # refuses before the port is touched
    sink: list[OklSink] = []

    def on_start(session_start: float, firmware: str | None) -> None:
        if okl_path is None:
            return
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
        sink.append(okl)

    def on_event(event: DeviceEvent) -> None:
        if not sink:
            return
        if event.kind == "response":
            sink[0].emit(ResponseEvent(id=event.index, timestamp=event.ts, operandum=0))
        elif event.kind == "reinforcer_start":
            sink[0].emit(ReinforcerStartEvent(id=event.index, timestamp=event.ts, operandum=0))
        else:
            sink[0].emit(ReinforcerEndEvent(id=event.index, timestamp=event.ts, operandum=0))

    run = drive_device(
        open_port,
        schedule,
        reinforcers=reinforcers,
        reinforcer_duration=reinforcer_duration,
        clock=clock or time.monotonic,
        on_start=on_start,
        on_event=on_event,
        max_duration=max_duration,
        report_timeout=report_timeout,
        label=dsl,
    )
    return DeviceRun(
        dsl=dsl,
        target_reinforcers=run.target_reinforcers,
        reinforcer_duration_s=run.reinforcer_duration_s,
        finished=run.finished,
        n_reinforcers=run.n_reinforcers,
        n_responses=run.n_responses,
        elapsed_s=run.elapsed_s,
        seq_gaps=run.seq_gaps,
        decision=run.decision,
        firmware=run.firmware,
        session_start=run.session_start,
        host_minus_device_ms=run.host_minus_device_ms,
        device=run.device,
    )
