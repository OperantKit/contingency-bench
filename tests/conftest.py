"""Fixtures that run the Teensy firmware on the host (``firmware/teensy41/host``)."""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

_FIRMWARE_DIR = Path(__file__).parent.parent / "firmware" / "teensy41"
_READY_TIMEOUT_S = 30.0
_EXIT_TIMEOUT_S = 10.0


def _why_not_runnable(tool: str, cwd: Path) -> str | None:
    """Why ``tool`` cannot be used, or None. A version manager's shim without a version counts."""
    try:
        run = subprocess.run(
            [tool, "--version"], cwd=cwd, capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError:
        return f"{tool} not found"
    if run.returncode != 0:
        said = run.stderr.strip().splitlines() or ["no message"]
        return f"{tool} does not run: {said[0]}"
    return None


@pytest.fixture(scope="session")
def host_firmware() -> Path:
    """The firmware built for the host by ``build.sh``.

    Skips where the tools to build it are missing. With the tools there, a
    build that fails is a failure, not a skip.
    """
    for tool in ("bash", "cargo", "rustc", os.environ.get("CXX") or "c++"):
        reason = _why_not_runnable(tool, _FIRMWARE_DIR / "rust")
        if reason is not None:
            pytest.skip(f"cannot build the firmware for the host: {reason}")
    build = subprocess.run(
        ["bash", str(_FIRMWARE_DIR / "host" / "build.sh")], capture_output=True, text=True
    )
    if build.returncode != 0:
        pytest.fail(f"host/build.sh failed:\n{build.stderr[-4000:]}", pytrace=False)
    return Path(build.stdout.strip())


def _wait_until_it_answers(fd: int, firmware: subprocess.Popen[bytes]) -> None:
    """Ask for a report and read it, so that a test's own timeouts do not include the start."""
    os.write(fd, b'{"cmd":"report"}\n')
    deadline = time.monotonic() + _READY_TIMEOUT_S
    got = b""
    while not got.endswith(b"\n"):
        left = deadline - time.monotonic()
        if left <= 0 or firmware.poll() is not None:
            pytest.fail(f"the host firmware did not answer report: {got!r}, {firmware.poll()=}")
        if select.select([fd], [], [], min(left, 0.1))[0]:
            got += os.read(fd, 4096)
    assert json.loads(got)["kind"] == "report"


@pytest.fixture
def host_firmware_port(host_firmware: Path) -> Iterator[str]:
    """A serial port path with the host firmware (real clock) at the other end.

    The firmware's standard input and output are the master of a pty; the
    path is the slave, which pyserial opens like the board's port. The slave
    is also kept open here, because the firmware ends once nobody holds it:
    a test may close the port and open it again.
    """
    pty = pytest.importorskip("pty")
    tty = pytest.importorskip("tty")
    master, slave = pty.openpty()
    try:
        tty.setraw(slave)  # no echo, and lines pass through unchanged
        firmware = subprocess.Popen([str(host_firmware)], stdin=master, stdout=master)
    except BaseException:
        os.close(slave)
        raise
    finally:
        os.close(master)
    try:
        _wait_until_it_answers(slave, firmware)
        yield os.ttyname(slave)
    finally:
        os.close(slave)  # end of input: the firmware ends by itself
        try:
            firmware.wait(timeout=_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            firmware.kill()
            firmware.wait()
