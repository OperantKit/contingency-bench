# contingency-bench

:jp: [日本語版 README](README.ja.md)

contingency-bench measures how a large language model (LLM) chooses between two options on concurrent variable-interval (VI) schedules, and summarizes its choices with the generalized matching law. It is for researchers who want to describe an LLM's choice behavior in the terms used for animals: sensitivity to reinforcement, bias, and one parameter of a reference learning model. Within [OperantKit](https://github.com/OperantKit), it is the benchmark for LLMs as subjects; it does not measure hardware timing.

- **Sensitivity from the less biased measure.** Fitted to unreinforced responses as well as to total responses.
- **Every condition gets the same number of trials,** so differences are not differences in exposure.
- **Runs offline.** A simulated subject stands in for the LLM, so you can check the pipeline without an API key.

## Installation

Python 3.11 or newer. contingency-bench is not on PyPI yet; install it from GitHub. Its dependencies, numpy and scipy, come from PyPI.

```bash
pip install git+https://github.com/OperantKit/contingency-bench
```

To query Claude models, also install the Anthropic SDK and set `ANTHROPIC_API_KEY`:

```bash
pip install anthropic
```

## Example

Run the whole procedure on the simulated subject (a policy-compression learner with `beta = 0.2`) instead of an LLM, so no API key is needed:

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

The four values are the generalized-matching sensitivity fitted to unreinforced responses, the sensitivity fitted to total responses, the bias (log b), and the `beta` recovered from the first sensitivity. The recovered 0.17 is close to the simulated 0.2.

## How it works

1. **Session.** Two buttons. On every trial each button that is not already set up becomes set up with its own probability; a response on a set-up button is reinforced and resets it. Five conditions are run, with set-up probabilities (0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), and (0.9, 0.1).
2. **Subject.** On each trial an LLM gets a short Japanese prompt with the trial number and its responses and reinforcers so far on each button, and answers `1` or `2`. The simulated subject is a policy-compression learner (Lai & Gershman, 2021).
3. **Fit.** The generalized matching law (Baum, 1974) is fitted across the five conditions twice: to the ratio of unreinforced responses and to the ratio of total responses. Total responses bias the sensitivity upward when reinforcers are frequent (Davison, 2021).
4. **Reference model.** `estimate_beta_from_sensitivity` finds the `beta` of the policy-compression learner whose sensitivity is closest to the fitted one, so the result can be stated as one parameter.

| Name | What it is |
|---|---|
| `run_benchmark`, `run_condition` | Run all conditions for one or more subjects, or one condition |
| `ConcurrentVISession` | The two-button session |
| `MockLLMClient`, `APILLMClient`, `BaseLLMClient` | The simulated subject, a Claude model through the Anthropic API, and the base class for your own client |
| `PolicyCompressionAgent` | The reference learner |
| `estimate_matching_parameters`, `estimate_beta_from_sensitivity` | The matching-law fit and the `beta` lookup |
| `equate_training_steps` | Gives every condition the same number of trials |

The command line runs the same procedure: `contingency-bench --mock --steps 50`. Without `--mock` it calls the API; `--model` (repeatable) chooses the Claude model IDs. With `--mock`, the default model names are only labels for simulated subjects with different `beta` values.

## Limitations

- **The session is in trials, not in time.** "VI" here means a fixed set-up probability per trial; there are no inter-reinforcement intervals in seconds, no changeover delay, and no Fleshler–Hoffman values. It is not a model of a chamber session.
- **Only two buttons and five conditions,** fixed in the code unless you pass your own `conditions`.
- **Only Anthropic models** are built in. Other models need a subclass of `BaseLLMClient`.
- **Failed calls become random choices.** If the API call fails (a warning is logged), or the answer contains both `1` and `2` or neither (nothing is logged), the trial is answered at random. The number of such trials is not reported.
- **It does not measure timing.** There is no hardware-in-the-loop (HIL) timing benchmark in this package.

## For AI coding agents

Guidance for AI coding agents is in [AGENTS.md](AGENTS.md).

## Development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

Every `python` block in both READMEs is run by `tests/test_readme_examples.py`.

## References

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias and undermatching. *Journal of the Experimental Analysis of Behavior, 22*(1), 231–242. https://doi.org/10.1901/jeab.1974.22-231
- Davison, M. (2021). Concurrent schedules: How responses per reinforcer affects estimates of sensitivity to reinforcement. *Journal of the Experimental Analysis of Behavior, 116*(1), 114–123. https://doi.org/10.1002/jeab.694
- Lai, L., & Gershman, S. J. (2021). Policy compression: An information bottleneck in action selection. *Psychology of Learning and Motivation, 74*, 195–232. https://doi.org/10.1016/bs.plm.2021.02.004

## License

MIT. See [LICENSE](LICENSE).
