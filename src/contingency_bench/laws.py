"""Pass/fail law suite for subjects on the concurrent VI-VI procedure.

Each law is a predicate on one run (all conditions, one seed). A subject
passes a law only if it passes on every seed. The predicates are defined by a
test on the fitted generalized matching line, not by threshold values taken
from the literature:

- ``I1`` Procedure invariant: in every condition the responses sum to the
  number of trials, and on each button reinforcers do not exceed responses.
  Every subject, including negative controls, should pass it; a failure
  points to the procedure, not the subject.
- ``L1`` Sensitivity to reinforcement (total responses): the slope ``a`` of
  log(B1/B2) on log(R1/R2) is greater than 0 by a one-sided t test.
- ``L2`` Sensitivity to reinforcement (unreinforced responses): the same test
  on the ratio of unreinforced responses.
- ``L3`` Not overmatching: the lower end of the 95% confidence interval of the
  total-response slope does not exceed 1.
- ``L4`` No bias: the 95% confidence interval of the total-response intercept
  (log b) contains 0.

Pairs are built as in :func:`~contingency_bench.benchmark.run_benchmark`
(conditions with no reinforcer on button 2 are dropped), and only pairs with
both ratios greater than 0 enter the log-log fit. With fewer than three
usable pairs, or with all log(R1/R2) equal, the fit cannot be made and every
law that needs it fails ("cannot be evaluated" is never a pass). alpha = 0.05.

Run from the command line::

    python -m contingency_bench.laws --subject random --out result.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from contingency_bench.benchmark import DEFAULT_CONDITIONS, run_condition
from contingency_bench.controls import RandomSubject
from contingency_bench.llm_client import BaseLLMClient

LAW_IDS: tuple[str, ...] = ("I1", "L1", "L2", "L3", "L4")
LAW_NAMES: dict[str, str] = {
    "I1": "procedure invariant",
    "L1": "sensitivity to reinforcement (total responses)",
    "L2": "sensitivity to reinforcement (unreinforced responses)",
    "L3": "not overmatching",
    "L4": "no bias",
}
ALPHA = 0.05
MIN_PAIRS = 3
DEFAULT_SEEDS = 10
DEFAULT_STEPS = 1000

SubjectFactory = Callable[[int], BaseLLMClient]

# Subjects that the command line can run: name -> (kind, factory).
SUBJECTS: dict[str, tuple[str, SubjectFactory]] = {
    "random": ("control", lambda seed: RandomSubject(seed=seed)),
}


@dataclass(frozen=True)
class LineFit:
    """A log-log line fitted with standard errors."""

    slope: float
    slope_stderr: float
    intercept: float
    intercept_stderr: float
    n_pairs: int

    @property
    def df(self) -> int:
        return self.n_pairs - 2

    def t_crit(self) -> float:
        return float(stats.t.ppf(1 - ALPHA / 2, self.df))


def fit_line(pairs: Sequence[tuple[float, float]]) -> LineFit | None:
    """Fit log(behavior ratio) on log(reinforcer ratio); None if it cannot be fitted."""
    usable = [(b, r) for b, r in pairs if b > 0 and r > 0]
    if len(usable) < MIN_PAIRS:
        return None
    y = np.log([p[0] for p in usable])
    x = np.log([p[1] for p in usable])
    if np.ptp(x) == 0:
        return None
    res = stats.linregress(x, y)
    return LineFit(
        slope=float(res.slope),
        slope_stderr=float(res.stderr),
        intercept=float(res.intercept),
        intercept_stderr=float(res.intercept_stderr),
        n_pairs=len(usable),
    )


def _slope_positive(fit: LineFit | None) -> bool:
    if fit is None:
        return False
    if fit.slope_stderr == 0:
        return fit.slope > 0
    p = float(stats.t.sf(fit.slope / fit.slope_stderr, fit.df))
    return fit.slope > 0 and p < ALPHA


def _not_overmatching(fit: LineFit | None) -> bool:
    if fit is None:
        return False
    return not (fit.slope - fit.t_crit() * fit.slope_stderr > 1.0)


def _no_bias(fit: LineFit | None) -> bool:
    if fit is None:
        return False
    if math.isclose(fit.intercept, 0.0, abs_tol=1e-12):
        return True
    return abs(fit.intercept) <= fit.t_crit() * fit.intercept_stderr


def _fit_values(fit: LineFit | None) -> dict[str, Any]:
    if fit is None:
        return {"fitted": False}
    return {
        "fitted": True,
        "slope": fit.slope,
        "slope_stderr": fit.slope_stderr,
        "intercept": fit.intercept,
        "intercept_stderr": fit.intercept_stderr,
        "n_pairs": fit.n_pairs,
    }


def evaluate_run(
    client: BaseLLMClient,
    steps: int,
    conditions: Sequence[tuple[float, float]] = DEFAULT_CONDITIONS,
) -> dict[str, tuple[bool, dict[str, Any]]]:
    """Run every condition once and return ``law id -> (passed, values)``."""
    unrein_pairs: list[tuple[float, float]] = []
    total_pairs: list[tuple[float, float]] = []
    invariant = True
    for vi_rates in conditions:
        b, r = run_condition(client, vi_rates, steps=steps)
        invariant = invariant and sum(b) == steps and all(r[i] <= b[i] for i in range(2))
        if r[1] <= 0:
            continue
        r_ratio = r[0] / r[1]
        unrein = [b[i] - r[i] for i in range(2)]
        if unrein[1] > 0:
            unrein_pairs.append((unrein[0] / unrein[1], r_ratio))
        if b[1] > 0:
            total_pairs.append((b[0] / b[1], r_ratio))

    fit_total = fit_line(total_pairs)
    fit_unrein = fit_line(unrein_pairs)
    total_values = _fit_values(fit_total)
    return {
        "I1": (invariant, {}),
        "L1": (_slope_positive(fit_total), total_values),
        "L2": (_slope_positive(fit_unrein), _fit_values(fit_unrein)),
        "L3": (_not_overmatching(fit_total), total_values),
        "L4": (_no_bias(fit_total), total_values),
    }


def run_law_suite(
    subject: str,
    make_subject: SubjectFactory,
    seeds: Iterable[int] = range(DEFAULT_SEEDS),
    steps: int = DEFAULT_STEPS,
    conditions: Sequence[tuple[float, float]] = DEFAULT_CONDITIONS,
    kind: str = "control",
) -> dict[str, Any]:
    """Run the suite on one subject over several seeds.

    Before each seed, numpy's global generator (used by the session and by
    the simulated learner) is seeded with the same value that is passed to
    ``make_subject``. A law's verdict is ``"pass"`` only if it passed on every
    seed.
    """
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    if steps <= 0:
        raise ValueError("steps must be positive")

    per_law: dict[str, dict[str, Any]] = {
        law: {"name": LAW_NAMES[law], "passed_seeds": [], "values": []} for law in LAW_IDS
    }
    for seed in seed_list:
        np.random.seed(seed)
        outcome = evaluate_run(make_subject(seed), steps, conditions)
        for law_id in LAW_IDS:
            passed, values = outcome[law_id]
            per_law[law_id]["values"].append({"seed": seed, "passed": passed, **values})
            if passed:
                per_law[law_id]["passed_seeds"].append(seed)

    for entry in per_law.values():
        entry["k"] = len(entry["passed_seeds"])
        entry["verdict"] = "pass" if entry["k"] == len(seed_list) else "fail"

    return {
        "subject": subject,
        "kind": kind,
        "steps": steps,
        "seeds": seed_list,
        "conditions": [list(c) for c in conditions],
        "alpha": ALPHA,
        "laws": per_law,
    }


def summarize(result: dict[str, Any]) -> str:
    """One line per law, then the count of laws passed (``I1`` not counted)."""
    lines = [f"subject={result['subject']} steps={result['steps']} seeds={len(result['seeds'])}"]
    n = len(result["seeds"])
    for law_id in LAW_IDS:
        law = result["laws"][law_id]
        lines.append(f"  {law_id} {law['verdict']:<4} {law['k']}/{n}  {law['name']}")
    laws = [law_id for law_id in LAW_IDS if law_id != "I1"]
    passed = sum(result["laws"][law_id]["verdict"] == "pass" for law_id in laws)
    lines.append(f"  laws passed: {passed}/{len(laws)} (I1 not counted)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m contingency_bench.laws",
        description="Run the pass/fail law suite on one subject and write the result as JSON.",
    )
    parser.add_argument("--subject", choices=sorted(SUBJECTS), required=True)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds 0..N-1")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="trials per condition")
    parser.add_argument("--out", type=Path, required=True, help="path of the JSON result")
    args = parser.parse_args(argv)

    kind, factory = SUBJECTS[args.subject]
    result = run_law_suite(
        args.subject, factory, seeds=range(args.seeds), steps=args.steps, kind=kind
    )
    result["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summarize(result))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
