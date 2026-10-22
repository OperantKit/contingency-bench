"""Command-line entry point for ``contingency-bench``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from contingency_bench.benchmark import BenchmarkResult, run_benchmark

# Anthropic model IDs: APILLMClient passes the name to the Messages API as is.
DEFAULT_MODELS: tuple[str, ...] = (
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-opus-5-5",
)

DEFAULT_MOCK_BETAS: dict[str, float] = {
    "claude-haiku-4-5-20251001": 0.45,
    "claude-sonnet-5": 0.25,
    "claude-opus-5-5": 0.15,
}


def _print_summary(results: Sequence[BenchmarkResult]) -> None:
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print(
        f"{'Model':<20} | {'a (unrein)':<12} | {'a (total)':<12} | "
        f"{'Bias (log b)':<13} | {'Est. Beta':<10}"
    )
    print("-" * 80)
    for res in results:
        print(
            f"{res.model:<20} | {res.sensitivity_unreinforced:<12.4f} | "
            f"{res.sensitivity_total:<12.4f} | {res.bias:<13.4f} | "
            f"{res.estimated_beta:<10.3f}"
        )
    print("=" * 80)
    print("a (unrein): GML sensitivity using unreinforced responses (Davison, 2021).")
    print("a (total) : GML sensitivity using total responses (biased upward; legacy).")
    print("Note: Lower a (unrein) -> higher estimated beta -> stronger behavioral inertia.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contingency-bench",
        description="LLM behavioral-characteristics benchmark on concurrent VI-VI sessions.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMClient (no API calls; CI-safe).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Steps per condition (equated across all conditions).",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Repeatable. Defaults to " + ", ".join(DEFAULT_MODELS) + ".",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit benchmark progress logs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    models = tuple(args.models) if args.models else DEFAULT_MODELS
    results = run_benchmark(
        models=models,
        use_mock=args.mock,
        steps_per_condition=args.steps,
        mock_betas=DEFAULT_MOCK_BETAS,
    )
    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
