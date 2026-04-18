"""Training-step equalisation (Johnson et al., 2022)."""

from __future__ import annotations

import pytest

from contingency_bench.training import equate_training_steps


def test_returns_step_count_unchanged_for_valid_input() -> None:
    assert equate_training_steps([(0.1, 0.9), (0.5, 0.5)], 100) == 100


def test_rejects_zero_or_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps_per_condition must be > 0"):
        equate_training_steps([(0.1, 0.9)], 0)
    with pytest.raises(ValueError, match="steps_per_condition must be > 0"):
        equate_training_steps([(0.1, 0.9)], -5)


def test_rejects_empty_conditions() -> None:
    with pytest.raises(ValueError, match="conditions list must not be empty"):
        equate_training_steps([], 50)
