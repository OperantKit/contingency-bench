"""Host side of a hardware-in-the-loop run against the Teensy 4.1 fake chamber.

The session uses the same OperantKit parts as ``operant run``: a
contingency-dsl expression bridged to a contingency-py schedule, a
session-runner ``SessionRunner`` writing an OKL log through session-recorder,
and experiment-io ``drive`` with the serial backend on a real port and the
host's monotonic clock. The pyserial port is handed to ``SerialOperandum`` and
``SerialReinforcer`` as it is.

Session time is the host clock. The device's own time for each response is
kept beside the host's receipt time (``host_minus_device_ms``); the two clocks
are not aligned beyond the ``sync`` command, so that difference holds the
transmission delay plus a fixed offset of a few milliseconds at most.

After the session the device is stopped and its ``report`` (the latency from
each response to the arrival of its reinforcer command, measured inside the
Teensy) is fetched. ``drive`` closes the port when it returns, so the port is
opened again for that. The summary JSON holds the report under ``"device"``,
which is what ``contingency-bench-hil-latency --device-report`` reads.

Requires the ``hil`` extra (pyserial and the OperantKit packages).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BAUD = 115200
DEFAULT_REINFORCER_S = 0.05
DEFAULT_MAX_DURATION_S = 600.0
DEFAULT_REPORT_TIMEOUT_S = 5.0
_SETTLE_S = 0.2

Port = Any
Clock = Callable[[], float]


@dataclass(frozen=True)
class HilRun:
    """Outcome of one host-side run."""

    dsl: str
    target_reinforcers: int
    reinforcer_duration_s: float
    finished: bool
    n_reinforcers: int
    n_responses: int
    elapsed_s: float
    host_minus_device_ms: list[float] = field(default_factory=list)
    device: dict[str, Any] | None = None


def _default_clock() -> Clock:
    return time.monotonic


def _open_serial(path: str, baud: int) -> Port:
    import serial

    return serial.Serial(path, baud, timeout=0)


def _build_schedule(source: str) -> Any:
    from contingency.bridge import from_dsl
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
    return from_dsl(parse(source, extensions=registry))


def _send(port: Port, command: dict[str, Any]) -> None:
    port.write((json.dumps(command) + "\n").encode())
    port.flush()


def _wait(clock: Clock, seconds: float) -> None:
    advance = getattr(clock, "advance", None)
    if callable(advance):
        advance(seconds)
    else:
        time.sleep(seconds)


def _read_line(port: Port) -> str:
    raw = port.readline()
    return raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)


def _fetch_report(port: Port, timeout_s: float) -> dict[str, Any] | None:
    """Stop the device and read lines until its report arrives or time runs out."""
    _send(port, {"cmd": "stop"})
    _send(port, {"cmd": "report"})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = _read_line(port).strip()
        if not line:
            time.sleep(0.001)
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("kind") == "report":
            return data
    return None


def run_crf(
    open_port: Callable[[], Port],
    *,
    dsl: str = "CRF",
    reinforcers: int = 200,
    reinforcer_duration: float = DEFAULT_REINFORCER_S,
    clock: Clock | None = None,
    okl_path: Path | None = None,
    poll_interval: float = 0.001,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    report_timeout: float = DEFAULT_REPORT_TIMEOUT_S,
) -> HilRun:
    """Run a session on the device behind ``open_port`` until ``reinforcers``.

    ``open_port`` is called twice: once for the session (``drive`` closes it)
    and once to stop the device and fetch its report.
    """
    from experiment_core import ReinforcementCountExit
    from experiment_io import drive
    from experiment_io.hw.backends.serial import SerialOperandum, SerialReinforcer
    from experiment_io.hw.protocols import ResponseSample
    from session_recorder import OklSink
    from session_runner import SessionConfig, SessionRunner, tee

    if reinforcers < 1:
        raise ValueError("reinforcers must be >= 1")
    clock = clock or _default_clock()
    schedule = _build_schedule(dsl)
    metadata = {"dsl": dsl, "backend": "serial", "reinforcer_duration_s": reinforcer_duration}

    sinks = []
    if okl_path is not None:
        okl = OklSink(okl_path)
        okl.write_header(
            session_name="hil-crf", clock_type="monotonic", session_start=clock(), metadata=metadata
        )
        sinks.append(okl)
    runner = SessionRunner(
        SessionConfig(
            name="hil-crf",
            schedule=schedule,
            exit_condition=ReinforcementCountExit(count=reinforcers),
            metadata=metadata,
        ),
        sink=tee(*sinks),
        clock=clock,
    )

    port = open_port()
    _send(port, {"cmd": "stop"})
    _wait(clock, _SETTLE_S)
    reset = getattr(port, "reset_input_buffer", None)
    if callable(reset):
        reset()

    lags_ms: list[float] = []

    def record(sample: ResponseSample, session_time: float) -> None:
        lags_ms.append(round((session_time - sample.timestamp) * 1e3, 3))

    t0 = clock()
    runner.start(start_time=t0)
    _send(port, {"cmd": "sync", "t": t0})
    drive(
        runner,
        operanda=[SerialOperandum(port, 0)],
        reinforcers=[SerialReinforcer(port, 0)],
        clock=clock,
        reinforcer_duration=reinforcer_duration,
        max_steps=2**62,
        max_duration=max_duration,
        poll_interval=poll_interval,
        on_response=record,
    )
    elapsed = clock() - t0

    report_port = open_port()
    try:
        device = _fetch_report(report_port, report_timeout)
    finally:
        report_port.close()

    return HilRun(
        dsl=dsl,
        target_reinforcers=reinforcers,
        reinforcer_duration_s=reinforcer_duration,
        finished=runner.is_finished,
        n_reinforcers=runner.reinforcement_count,
        n_responses=len(lags_ms),
        elapsed_s=round(elapsed, 3),
        host_minus_device_ms=lags_ms,
        device=device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the session, write the summary JSON; exit 1 if it did not finish."""
    parser = argparse.ArgumentParser(
        prog="contingency-bench-hil-crf",
        description="Run an OperantKit session against the Teensy fake chamber over serial.",
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbmodem…")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--dsl", default="CRF", help="contingency-dsl expression.")
    parser.add_argument("--reinforcers", type=int, default=200, help="Stop after this many.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_REINFORCER_S,
        help="Reinforcer duration in seconds sent to the device (default: %(default)s).",
    )
    parser.add_argument("--okl", type=Path, help="Write the session's OKL log here.")
    parser.add_argument("--summary", type=Path, help="Write the summary JSON here.")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--report-timeout", type=float, default=DEFAULT_REPORT_TIMEOUT_S)
    args = parser.parse_args(argv)

    try:
        run = run_crf(
            lambda: _open_serial(args.port, args.baud),
            dsl=args.dsl,
            reinforcers=args.reinforcers,
            reinforcer_duration=args.duration,
            clock=_default_clock(),
            okl_path=args.okl,
            max_duration=args.max_duration,
            report_timeout=args.report_timeout,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = asdict(run)
    if args.summary is not None:
        args.summary.write_text(json.dumps(summary) + "\n")
    device_n = run.device.get("n") if run.device else None
    print(
        f"finished={run.finished} reinforcers={run.n_reinforcers}/{run.target_reinforcers} "
        f"responses={run.n_responses} elapsed={run.elapsed_s} s device_report_n={device_n}"
    )
    if not run.finished:
        print("warning: the session did not reach the reinforcer count", file=sys.stderr)
    if run.device is None:
        print("warning: no report from the device", file=sys.stderr)
    return 0 if run.finished and run.device is not None else 1


if __name__ == "__main__":
    sys.exit(main())
