"""Benchmark runner for OperantKit experimental kit.

Provides timing precision benchmarks, LLM-as-subject performance evaluation,
and matching law parameter recovery on simulated concurrent VI-VI sessions.
"""

from contingency_bench.benchmark import (
    BenchmarkResult,
    estimate_beta_from_sensitivity,
    run_benchmark,
    run_condition,
)
from contingency_bench.llm_client import (
    APILLMClient,
    BaseLLMClient,
    MockLLMClient,
)
from contingency_bench.matching import (
    MatchingParameters,
    estimate_matching_parameters,
    generalized_matching_law,
)
from contingency_bench.policy_compression import PolicyCompressionAgent
from contingency_bench.training import equate_training_steps
from contingency_bench.vi_session import ConcurrentVISession

__all__ = [
    "APILLMClient",
    "BaseLLMClient",
    "BenchmarkResult",
    "ConcurrentVISession",
    "MatchingParameters",
    "MockLLMClient",
    "PolicyCompressionAgent",
    "equate_training_steps",
    "estimate_beta_from_sensitivity",
    "estimate_matching_parameters",
    "generalized_matching_law",
    "run_benchmark",
    "run_condition",
]
