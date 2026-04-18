"""End-to-end benchmark on MockLLMClient (small-scale smoke + invariants)."""

from __future__ import annotations

import numpy as np
import pytest

from contingency_bench.benchmark import (
    BenchmarkResult,
    estimate_beta_from_sensitivity,
    run_benchmark,
    run_condition,
)
from contingency_bench.llm_client import MockLLMClient


def test_run_condition_emits_consistent_counts() -> None:
    np.random.seed(42)
    client = MockLLMClient("mock", inherent_beta=0.3)
    responses, reinforcements = run_condition(client, (0.3, 0.7), steps=30)
    assert sum(responses) == 30
    for i in range(2):
        assert reinforcements[i] <= responses[i]


def test_run_benchmark_mock_returns_one_result_per_model() -> None:
    np.random.seed(123)
    results = run_benchmark(
        models=["alpha", "beta"],
        use_mock=True,
        steps_per_condition=20,
        mock_betas={"alpha": 0.1, "beta": 0.4},
    )
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results)
    assert {r.model for r in results} == {"alpha", "beta"}
    for r in results:
        # Even small samples must yield finite stats
        assert np.isfinite(r.sensitivity_unreinforced)
        assert np.isfinite(r.bias)
        assert 0.0 <= r.r_squared <= 1.0 + 1e-9 or r.r_squared == 0.0


def test_run_benchmark_rejects_zero_steps() -> None:
    with pytest.raises(ValueError):
        run_benchmark(models=["x"], use_mock=True, steps_per_condition=0)


def test_estimate_beta_from_sensitivity_returns_value_in_grid_range() -> None:
    np.random.seed(7)
    # Use a tiny step budget to keep the test fast; we only verify the contract.
    beta = estimate_beta_from_sensitivity(target_a=0.5, steps=50)
    assert 0.0 < beta <= 1.0
