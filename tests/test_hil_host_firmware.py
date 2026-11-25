"""The device decision against the real firmware, built for the host and put behind a pty.

Unlike ``test_hil_device.py``, the port is not a fake: ``main.cpp`` and
bench-decide run in a process of their own (``host_firmware_port`` in
``conftest.py``), and the host opens the pty with pyserial, as it opens the
board. The session runs in real time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("experiment_io")
pytest.importorskip("serial")

from contingency_bench import hil_crf  # noqa: E402
from contingency_bench.hil_device import run_device  # noqa: E402

# kFirmwareId in main.cpp; the fake device of test_hil_device.py says "fake-device".
_FIRMWARE = "contingency-bench-teensy41-contingency-rs"


def test_device_run_counts_fr_and_checks_seq(host_firmware_port):
    run = run_device(
        lambda: hil_crf._open_serial(host_firmware_port, 115200),
        dsl="FR 3",
        reinforcers=2,
        max_duration=60.0,
    )

    assert run.firmware == _FIRMWARE
    assert run.finished
    assert run.n_reinforcers == 2
    assert run.n_responses == 6
    assert run.seq_gaps == 0
