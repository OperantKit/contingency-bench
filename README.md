# contingency-bench

:jp: [日本語版 README](README.ja.md)

Benchmark runner for the OperantKit experimental kit: timing precision,
LLM-as-subject performance, and matching-law parameter recovery.

## Strategy A: Python-only port

This package is the Python-only port of the original mixed Rust + Python
benchmark project. Strategy A (per the porting plan) keeps the LLM
behavioral-characteristics runner here and defers the Rust hardware-in-the-loop
(HIL) host_runner / Teensy firmware to a separate package. Until that handoff
ships, the Rust workspace and the Teensy ground-truth firmware remain in the
frozen v0 archive (read-only reference, not part of this distribution).

## Public API

```python
from contingency_bench import (
    BaseLLMClient,
    APILLMClient,
    MockLLMClient,
    ConcurrentVISession,
    PolicyCompressionAgent,
    MatchingParameters,
    estimate_matching_parameters,
    estimate_beta_from_sensitivity,
    equate_training_steps,
    run_benchmark,
    run_condition,
)
```

CLI:

```bash
contingency-bench --mock --steps 50 --verbose
contingency-bench --mock --model claude-3-5-sonnet --model gpt-4o --steps 100
```

## 30-second example

```bash
pip install git+https://github.com/OperantKit/contingency-bench
```

Run the benchmark on a simulated subject (a policy-compression Q-learner with `beta = 0.2`) instead of an LLM, so no API key is needed:

```python
import numpy as np
from contingency_bench import run_benchmark

np.random.seed(0)  # the simulated sessions draw from numpy's global generator
results = run_benchmark(["mock-subject"], use_mock=True, steps_per_condition=200,
                        mock_betas={"mock-subject": 0.2})
r = results[0]
print(round(r.sensitivity_unreinforced, 2), round(r.sensitivity_total, 2), round(r.bias, 2), round(r.estimated_beta, 2))
# 0.38 0.82 -0.18 0.17
```

The printed values are the generalized-matching sensitivity from unreinforced responses, the sensitivity from total responses, the bias (log b), and the `beta` recovered from the sensitivity; the recovered 0.17 is close to the simulated 0.2. With `use_mock=False` the same procedure presents each trial to a Claude model through the Anthropic API (`pip install 'contingency-bench[llm]'`, `ANTHROPIC_API_KEY`).

## Why not a 50-line hand-written LLM benchmark?

- **Matching-law fit with the less biased response measure.** Sensitivity is estimated from unreinforced responses as well as from total responses (total responses bias the estimate upward at high reinforcer rates; Davison, 2021), across five reinforcement-ratio conditions.
- **Equated exposure.** `equate_training_steps` gives every condition the same number of trials (Johnson et al., 2022), so differences in sensitivity are not differences in training.
- **A reference model to interpret the result.** `estimate_beta_from_sensitivity` maps a fitted sensitivity onto the `beta` of a policy-compression agent (Lai & Gershman, 2021), so a model's matching can be stated as one parameter.
- **Runs without an API.** `MockLLMClient` and the `--mock` CLI flag run the whole pipeline offline for tests and CI; the Anthropic SDK is imported only when a live client is created.

## Installation

```bash
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

For live API benchmarks (Anthropic SDK, lazy-loaded):

```bash
.venv/bin/python -m pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Local dependency notes

The Q-learning agent and the GML estimator originate in sibling packages
in the same monorepo but are not yet published to PyPI. To keep
`contingency-bench` installable as a standalone package, both modules are
**inlined** here:

- `contingency_bench.policy_compression.PolicyCompressionAgent`
- `contingency_bench.matching.estimate_matching_parameters`
  (mirrors `session_analyzer.analytics.matching`)

When the upstream modules are published, the inlined copies should be
replaced with re-exports. The public symbols stay the same.

## LLM dependency notes

The v0 driver supported multiple model vendors via `litellm`. This port adopts
the single-vendor `anthropic` SDK to match the new repository's standard LLM
dependency. The SDK is imported **lazily** in
`contingency_bench.llm_client.APILLMClient.__init__`; importing the module
itself never requires the optional `[llm]` extra. Re-introducing multi-vendor
support is a follow-up.

## Future Rust workspace

The Rust HIL harness (`host_runner` binary, `protocol`, `scenarios`, `metrics`)
and the Teensy 4.1 ground-truth firmware live in the v0 archive and are
expected to be ported into a separate package alongside the `contingency-rs`
handoff. They are intentionally out of scope for this Python-only release.

## References

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias
  and undermatching. *Journal of the Experimental Analysis of Behavior*, 22,
  231-242. https://doi.org/10.1901/jeab.1974.22-231
- Davison, M. (2021). What is the value of the generalized matching law?
  *Journal of the Experimental Analysis of Behavior*, 116, 114-123.
  https://doi.org/10.1002/jeab.694
- Johnson, J. E., Kangas, B. D., Bergman, J., & Rice, K. C. (2022).
  Comparison of fixed- and variable-ratio schedules of reinforcement on
  progressive-ratio responding. *Journal of the Experimental Analysis of
  Behavior*, 117, 320-330. https://doi.org/10.1002/jeab.755
- Lai, L., & Gershman, S. J. (2021). Policy compression: An information
  bottleneck in action selection. *Psychology of Learning and Motivation*, 74,
  195-232. https://doi.org/10.1016/bs.plm.2021.02.004
