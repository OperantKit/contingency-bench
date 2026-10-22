"""LLM behavioral-characteristics benchmark on concurrent VI-VI sessions.

Drives one or more :class:`~contingency_bench.llm_client.BaseLLMClient`
subjects through a fixed grid of VI-rate conditions, fits the generalized
matching law to the resulting choice allocations, and reverse-estimates the
inherent policy-compression ``beta`` that best reproduces the observed
sensitivity.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from contingency_bench.llm_client import BaseLLMClient, MockLLMClient
from contingency_bench.matching import estimate_matching_parameters
from contingency_bench.policy_compression import PolicyCompressionAgent
from contingency_bench.training import equate_training_steps
from contingency_bench.vi_session import ConcurrentVISession

logger = logging.getLogger(__name__)

DEFAULT_CONDITIONS: tuple[tuple[float, float], ...] = (
    (0.1, 0.9),
    (0.3, 0.7),
    (0.5, 0.5),
    (0.7, 0.3),
    (0.9, 0.1),
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated GML-fit result for a single LLM subject."""

    model: str
    sensitivity_unreinforced: float
    sensitivity_total: float
    bias: float
    r_squared: float
    estimated_beta: float


def run_condition(
    client: BaseLLMClient,
    vi_rates: tuple[float, float],
    steps: int = 50,
) -> tuple[list[int], list[int]]:
    """Run one VI-VI condition; return per-key (responses, reinforcements)."""
    session = ConcurrentVISession(vi_rates)

    if isinstance(client, MockLLMClient):
        client.agent.reset()

    for i in range(steps):
        prompt = (
            f"試行 {i + 1}/{steps}: \n"
            f"これまでのあなたの選択数: ボタン1={session.responses[0]}, "
            f"ボタン2={session.responses[1]}\n"
            f"これまでの当たり回数: ボタン1={session.reinforcements[0]}, "
            f"ボタン2={session.reinforcements[1]}\n"
            f"どちらのボタンを押しますか？"
        )

        action = client.get_action(prompt)
        reinforced = session.step(action)

        if isinstance(client, MockLLMClient):
            client.update(action, 1.0 if reinforced else 0.0)

    return session.responses, session.reinforcements


def estimate_beta_from_sensitivity(
    target_a: float,
    num_simulations: int = 10,  # noqa: ARG001 - retained for API compatibility
    steps: int = 1000,
    conditions: Sequence[tuple[float, float]] = DEFAULT_CONDITIONS,
) -> float:
    """Reverse-estimate the policy-compression ``beta`` matching ``target_a``.

    Sweeps a 20-point grid of ``beta`` values, runs each through the same VI
    grid, fits a GML, and returns the ``beta`` whose fitted sensitivity is
    closest to ``target_a``.
    """
    beta_candidates = np.linspace(0.01, 1.0, 20)
    best_beta = 0.1
    min_diff = float("inf")

    for beta in beta_candidates:
        agent = PolicyCompressionAgent(n_actions=2, beta=float(beta), learning_rate=0.05)
        b_ratios: list[float] = []
        r_ratios: list[float] = []

        for r1, r2 in conditions:
            agent.reset()
            session = ConcurrentVISession((r1, r2))
            for _ in range(steps):
                act = agent.choose_action()
                reinf = session.step(act)
                agent.update(act, 1.0 if reinf else 0.0)

            unrein = [session.responses[i] - session.reinforcements[i] for i in range(2)]
            if unrein[1] > 0 and session.reinforcements[1] > 0:
                b_ratios.append(unrein[0] / unrein[1])
                r_ratios.append(session.reinforcements[0] / session.reinforcements[1])

        if len(b_ratios) < 2:
            continue
        params = estimate_matching_parameters(b_ratios, r_ratios)
        diff = abs(params.sensitivity - target_a)

        if diff < min_diff:
            min_diff = diff
            best_beta = float(beta)

    return best_beta


def run_benchmark(
    models: Sequence[str],
    use_mock: bool = False,
    steps_per_condition: int = 50,
    conditions: Sequence[tuple[float, float]] = DEFAULT_CONDITIONS,
    mock_betas: dict[str, float] | None = None,
) -> list[BenchmarkResult]:
    """Run the policy-compression benchmark across one or more LLM subjects.

    Parameters
    ----------
    models:
        Anthropic model IDs (e.g. ``"claude-haiku-4-5-20251001"``).
    use_mock:
        If True, use :class:`MockLLMClient` (no API calls). If False, use
        :class:`APILLMClient`, which requires the ``[llm]`` extra.
    steps_per_condition:
        Trials per condition. Equated across conditions per Johnson et al.
        (2022).
    conditions:
        VI-rate tuples to evaluate.
    mock_betas:
        Optional mapping from model name to inherent ``beta`` for
        :class:`MockLLMClient`. Models without an entry receive ``beta=0.2``.
    """
    logger.info("=" * 60)
    logger.info("LLM Behavioral Characteristics (Policy Compression) Benchmark")
    logger.info("Mode: %s", "Mock (Simulation)" if use_mock else "Live API")
    logger.info("Steps per condition: %d", steps_per_condition)
    logger.info("=" * 60)

    steps_per_condition = equate_training_steps(conditions, steps_per_condition)
    if mock_betas is None:
        mock_betas = {}

    results: list[BenchmarkResult] = []
    for model_name in models:
        logger.info("Evaluating: %s", model_name)

        client: BaseLLMClient
        if use_mock:
            client = MockLLMClient(model_name, inherent_beta=mock_betas.get(model_name, 0.2))
        else:
            from contingency_bench.llm_client import APILLMClient

            client = APILLMClient(model_name)

        # Davison (2021) showed unreinforced responses give an unbiased
        # sensitivity estimate; total responses bias upward at high reinforcer
        # rates. Both variants are reported for comparison. Each variant
        # maintains its own aligned (b, r) pair list so the strict-zip contract
        # of estimate_matching_parameters is upheld.
        unrein_pairs: list[tuple[float, float]] = []
        total_pairs: list[tuple[float, float]] = []

        for r1, r2 in conditions:
            b, r = run_condition(client, (r1, r2), steps=steps_per_condition)
            if r[1] <= 0:
                continue
            r_ratio = r[0] / r[1]
            unrein = [b[i] - r[i] for i in range(2)]
            if unrein[1] > 0:
                unrein_pairs.append((unrein[0] / unrein[1], r_ratio))
            if b[1] > 0:
                total_pairs.append((b[0] / b[1], r_ratio))

        params = estimate_matching_parameters(
            [p[0] for p in unrein_pairs], [p[1] for p in unrein_pairs]
        )
        params_total = estimate_matching_parameters(
            [p[0] for p in total_pairs], [p[1] for p in total_pairs]
        )
        est_beta = estimate_beta_from_sensitivity(params.sensitivity, steps=1000)

        result = BenchmarkResult(
            model=model_name,
            sensitivity_unreinforced=params.sensitivity,
            sensitivity_total=params_total.sensitivity,
            bias=params.bias,
            r_squared=params.r_squared,
            estimated_beta=est_beta,
        )
        results.append(result)
        logger.info(
            "  -> a (unrein)=%.4f  a (total)=%.4f  bias=%.4f  R^2=%.4f  est_beta=%.3f",
            result.sensitivity_unreinforced,
            result.sensitivity_total,
            result.bias,
            result.r_squared,
            result.estimated_beta,
        )

    return results
