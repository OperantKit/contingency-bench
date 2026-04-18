"""Smoke tests: imports and surface-level invariants."""

from __future__ import annotations


def test_package_imports() -> None:
    import contingency_bench

    assert hasattr(contingency_bench, "run_benchmark")
    assert hasattr(contingency_bench, "MockLLMClient")
    assert hasattr(contingency_bench, "PolicyCompressionAgent")
    assert hasattr(contingency_bench, "estimate_matching_parameters")


def test_cli_module_importable() -> None:
    from contingency_bench import cli

    assert hasattr(cli, "main")
