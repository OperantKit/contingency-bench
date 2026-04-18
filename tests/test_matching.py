"""GML estimator: synthesized data -> fit -> verify recovery."""

from __future__ import annotations

import math

import numpy as np
import pytest

from contingency_bench.matching import (
    MatchingParameters,
    estimate_matching_parameters,
    generalized_matching_law,
)


def test_generalized_matching_law_is_linear_in_log_space() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = generalized_matching_law(x, a=0.8, log_b=0.1)
    assert isinstance(y, np.ndarray)
    np.testing.assert_allclose(y, 0.8 * x + 0.1)


def test_estimate_matching_parameters_recovers_true_params() -> None:
    """Synth: with a=0.8, log_b=0.05, recover within tight tolerance."""
    true_a = 0.8
    true_log_b = 0.05
    reinforcement_ratios = [0.111, 0.428, 1.0, 2.333, 9.0]  # R1/R2 from VI-VI
    log_r = np.log(reinforcement_ratios)
    log_b = true_a * log_r + true_log_b
    response_ratios = list(np.exp(log_b))

    params = estimate_matching_parameters(response_ratios, reinforcement_ratios)
    assert params.sensitivity == pytest.approx(true_a, abs=1e-6)
    assert params.bias == pytest.approx(true_log_b, abs=1e-6)
    assert params.r_squared == pytest.approx(1.0, abs=1e-6)


def test_estimate_handles_insufficient_data() -> None:
    params = estimate_matching_parameters([], [])
    assert params == MatchingParameters(sensitivity=0.0, bias=0.0, r_squared=0.0)

    params = estimate_matching_parameters([1.0], [1.0])
    assert params == MatchingParameters(sensitivity=0.0, bias=0.0, r_squared=0.0)


def test_estimate_filters_nonpositive_ratios() -> None:
    """Zero or negative ratios are stripped before the log transform."""
    response_ratios = [0.0, 1.0, 2.0, -1.0, 4.0]
    reinforcement_ratios = [0.5, 1.0, 2.0, 3.0, 4.0]
    params = estimate_matching_parameters(response_ratios, reinforcement_ratios)
    # 4 valid points (response_ratios 1.0, 2.0, 4.0; index 3 dropped); negative dropped too
    assert params.sensitivity > 0.0
    assert math.isfinite(params.bias)
