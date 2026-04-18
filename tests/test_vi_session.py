"""ConcurrentVISession invariants."""

from __future__ import annotations

import numpy as np
import pytest

from contingency_bench.vi_session import ConcurrentVISession


def test_constructor_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError):
        ConcurrentVISession((1.5, 0.5))
    with pytest.raises(ValueError):
        ConcurrentVISession((-0.1, 0.5))


def test_step_rejects_invalid_action() -> None:
    session = ConcurrentVISession((0.1, 0.1))
    with pytest.raises(ValueError):
        session.step(2)


def test_response_count_increments() -> None:
    session = ConcurrentVISession((0.0, 0.0))
    for _ in range(10):
        session.step(0)
    assert session.responses == [10, 0]
    assert session.reinforcements == [0, 0]


def test_reinforcement_only_when_armed() -> None:
    """With rate=1.0, a key is always armed; with rate=0.0, never."""
    np.random.seed(0)
    session = ConcurrentVISession((1.0, 0.0))
    reinforced = [session.step(0) for _ in range(5)]
    assert reinforced[0] is True  # First step: armed -> reinforced
    # After consumption, a new step re-arms with prob 1.0, so all True
    assert all(reinforced)
    assert session.reinforcements == [5, 0]

    session2 = ConcurrentVISession((0.0, 0.0))
    reinforced2 = [session2.step(1) for _ in range(5)]
    assert not any(reinforced2)
    assert session2.reinforcements == [0, 0]
