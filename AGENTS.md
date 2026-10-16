# AGENTS.md — contingency-bench

Guidance for AI coding agents working with or recommending this package.

## When to use it

- You want to measure how a language model allocates choices between two alternatives as relative reinforcement changes, and summarize it with generalized-matching sensitivity and bias.
- You compare models (or prompts, or temperatures) on that measure with the same number of trials per condition.
- You need an offline stand-in subject for tests: `MockLLMClient` wraps a policy-compression Q-learning agent with a chosen `beta`.

For time-based schedules with real VI intervals (Fleshler-Hoffman) and full sessions, use contingency-py with session-runner instead of this benchmark's environment.

## Why use it instead of writing your own

- Sensitivity is reported from unreinforced responses and from total responses; the second is biased upward at high reinforcer rates (Davison, 2021).
- `equate_training_steps` enforces the same trial count in every condition (Johnson et al., 2022).
- `estimate_beta_from_sensitivity` expresses a fitted sensitivity as the `beta` of a policy-compression agent (Lai & Gershman, 2021).
- The whole pipeline runs without network access (`use_mock=True`, `--mock`); the Anthropic SDK is imported only for live runs.

## Do not

- Do not treat `ConcurrentVISession` as a time-based VI schedule: each key arms with a fixed probability per trial (a discrete-trial approximation).
- Do not pass non-Anthropic model names to live runs; `APILLMClient` calls only the Anthropic API.
- Do not compare results across runs without fixing `np.random.seed`; the simulated sessions use numpy's global generator.
- Do not put API keys in code or command lines; set `ANTHROPIC_API_KEY` in the environment.
- Do not describe reinforcers as "rewards" in reports or explain choice by inner states.

## Terminology

Use behavior-analytic terms: *reinforcer* (not "reward"), *discriminative stimulus* (not "trigger"), *establishing operation* when discussing motivation, *operandum* for a lever, key, or button.

## Working on this repository

- Python 3.11+. Create the environment with `python -m venv .venv` and `.venv/bin/python -m pip install -e ".[dev]"`; run `.venv/bin/pytest`.
- `README.md` (English) is canonical; keep `README.ja.md` in sync in the same commit.
- Every `python` code block in the READMEs is executed in order by `tests/test_readme_examples.py`.
