"""Law suite: the 26 Tier B respondent procedures under compound-CS R–W.

Each procedure is read from the ``expected_ast`` of its Tier B conformance
fixture (``contingency_respondent_dsl.from_dict``), expanded into trials by
contingency-respondent-py and run through session-analyzer's compound-CS
Rescorla–Wagner model. The phenomenon is then judged by inequalities on
associative strength (V) only -- no mapping from V to a conditioned response
is assumed.

Needs the ``respondent`` extra (contingency-respondent-py and
session-analyzer); the module is skipped without it.

Outcome per row:

- ``PREDICTS``  -- R–W gives the direction of the phenomenon: plain assert.
- ``PARTIAL``   -- the direction comes out under a plain extension (context as
  an element, one system per US), but the mechanism argued in the source does
  not: assert, with the note in ``MECHANISM_NOTES``.
- ``NO``        -- R–W does not give the phenomenon: the assertion states the
  phenomenon and is marked ``xfail(strict=True)`` (the mismatch is expected).
- ``UNSUPPORTED`` -- within-CS time is needed; ``expand`` must refuse.

Fixture location: ``$RESPONDENT_CONFORMANCE_ROOT`` if set, else the first
ancestor holding ``contingency-respondent-dsl/conformance/tier-b`` or
``apps/core/contingency-respondent-dsl/conformance/tier-b``; skipped when none
is found.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("contingency_respondent")
pytest.importorskip("session_analyzer.analytics")

from contingency_respondent import Expansion, expand, expand_control  # noqa: E402
from contingency_respondent_dsl import from_dict  # noqa: E402
from session_analyzer.analytics import (  # noqa: E402
    CompoundRescorlaWagner,
    CompoundTrace,
    CompoundTrial,
    simulate_compound_rescorla_wagner,
)

ENV_ROOT = "RESPONDENT_CONFORMANCE_ROOT"
_CANDIDATES = (
    Path("contingency-respondent-dsl") / "conformance" / "tier-b",
    Path("apps") / "core" / "contingency-respondent-dsl" / "conformance" / "tier-b",
)
EPS = 1e-3


def _find_root() -> Path | None:
    explicit = os.environ.get(ENV_ROOT)
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        for rel in _CANDIDATES:
            if (ancestor / rel).is_dir():
                return ancestor / rel
    return None


ROOT = _find_root()


def _cases(name: str) -> list[Any]:
    if ROOT is None:
        pytest.skip("Tier B conformance fixtures not found")
    data = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
    return [from_dict(c["expected_ast"]) for c in data]


MODEL = CompoundRescorlaWagner(
    {},
    lambda_by_us={
        "shock": 1.0,
        "food": 1.0,
        "footshock": 1.0,
        "licl_injection": 1.0,
        "lithium": 1.0,
    },
)


def run(e: Expansion, model: CompoundRescorlaWagner) -> CompoundTrace:
    """Run ``model`` over the trials of ``e`` (context summed in when the expansion says so)."""
    if e.unsupported is not None:
        raise ValueError(f"cannot run an unsupported expansion: {e.unsupported}")
    cae = e.context_as_element
    trials = [CompoundTrial(t.elements(cae), t.us, t.stage) for t in e.trials]
    return simulate_compound_rescorla_wagner(trials, model)


def _run(e: Expansion | None) -> CompoundTrace:
    if e is None or e.unsupported is not None:
        raise RuntimeError(f"expansion not runnable: {e}")
    return run(e, MODEL)


def _pair(node: Any, d: int = 20) -> tuple[CompoundTrace, CompoundTrace | None]:
    c = expand_control(node, default_trials=d)
    return _run(expand(node, default_trials=d)), (_run(c) if c is not None else None)


def _ctl(t: CompoundTrace | None) -> CompoundTrace:
    assert t is not None, "control expansion missing"
    return t


def sum_at_start(t: CompoundTrace, stage: str) -> float:
    """Summed V of the first trial of ``stage``, before the trial."""
    r = next(r for r in t.records if r.stage == stage)
    return r.sum_v_before


def sum_at_end(t: CompoundTrace, stage: str) -> float:
    """Summed V of the last trial's elements of ``stage``, after the trial."""
    r = [r for r in t.records if r.stage == stage][-1]
    return sum(r.v_after[s] for s in r.elements)


def v_start(t: CompoundTrace, stage: str, cs: str, *, us: str | None = None) -> float:
    return t.v_at_stage_start(stage, us=us)[cs]


def v_end(t: CompoundTrace, stage: str, cs: str) -> float:
    r = [r for r in t.records if r.stage == stage][-1]
    return r.v_after[cs]


def _after_k(t: CompoundTrace, stage: str, k: int) -> dict[str, float]:
    """V after the ``k``-th trial of ``stage``."""
    return [r for r in t.records if r.stage == stage][k - 1].v_after


# --------------------------------------------------------------------------- #
# Judgements (one per fixture file)
# --------------------------------------------------------------------------- #


def j_higher_order(node: Any) -> None:
    t, _ = _pair(node)
    assert t.final()["light"] > EPS, "second-order excitation of X"


def j_sensory_preconditioning(node: Any) -> None:
    t, _ = _pair(node)
    assert v_start(t, "test", "light") > EPS, "X acquires strength through A"


def j_counterconditioning(node: Any) -> None:
    t, _ = _pair(node)
    before = v_start(t, "phase2", "tone", us="shock")
    after = t.final(us="shock")["tone"]
    assert after < before
    assert t.final(us="food")["tone"] > 0.5


def j_blocking(node: Any) -> None:
    t, c = _pair(node)
    assert v_start(t, "test", "x") < v_start(_ctl(c), "test", "x") - 0.1


def j_overshadowing(node: Any) -> None:
    model = CompoundRescorlaWagner(
        {"loud_tone": 0.5, "dim_light": 0.2}, lambda_by_us={"shock": 1.0}
    )
    e, c = expand(node), expand_control(node)
    assert c is not None
    t, tc = run(e, model), run(c, model)
    assert v_start(t, "test_b", "dim_light") < v_start(tc, "test_b", "dim_light")
    assert v_start(t, "test_a", "loud_tone") > v_start(t, "test_a", "dim_light")


def j_overexpectation(node: Any) -> None:
    t, c = _pair(node)
    assert v_start(t, "test", "a") < v_start(t, "phase2", "a") - 0.1
    assert abs(v_start(_ctl(c), "test", "a") - v_end(_ctl(c), "phase1", "a")) < EPS


def j_super_conditioning(node: Any) -> None:
    t, c = _pair(node)
    assert v_start(t, "phase2", "i") < 0
    assert v_start(t, "test", "x") > v_start(_ctl(c), "test", "x") + 0.05


def j_retrospective_revaluation(nodes: list[Any]) -> None:
    backward_blocking, unovershadowing = nodes
    t1, c1 = _pair(backward_blocking)
    t2, c2 = _pair(unovershadowing)
    assert v_start(t1, "test", "x") < v_start(_ctl(c1), "test", "x") - EPS, "backward blocking"
    assert v_start(t2, "test", "x") > v_start(_ctl(c2), "test", "x") + EPS, "recovery"


def j_conditioned_inhibition(node: Any) -> None:
    t, _ = _pair(node)
    v = t.v_at_stage_start("test")
    assert v["i"] < 0
    assert abs(v["a"] + v["i"]) < EPS


def j_occasion_setting(node: Any) -> None:
    t, _ = _pair(node, d=80)
    v = t.v_at_stage_start("test")
    if node.mode == "FeaturePositive":
        assert v[node.feature] > 0.5
        assert abs(v[node.target]) < 0.05
    else:
        assert v[node.target] > 0.5
        assert v[node.feature] < -0.5


def j_unsupported(node: Any) -> None:
    e = expand(node)
    assert e.unsupported is not None
    assert e.trials == ()
    assert expand_control(node) is None


def j_summation_retardation(nodes: list[Any]) -> None:
    for node in nodes:
        t, c = _pair(node, d=80)
        tc = _ctl(c)
        if node.mode == "Summation":
            assert sum_at_start(t, "summation_test") < sum_at_start(tc, "summation_test") - 0.1
        else:
            # Retardation is a rate effect: compare after the first few trials.
            k = 5
            treated = _after_k(t, "retardation", k)[node.inhibitor]
            ctl = _after_k(tc, "retardation", k)
            probe = next(s for s in ctl if s not in (node.excitor, node.inhibitor))
            assert treated < ctl[probe] - 0.1


def j_latent_inhibition(node: Any) -> None:
    t, c = _pair(node)
    assert v_start(t, "test", "tone") < v_start(_ctl(c), "test", "tone") - EPS


def j_us_preexposure(node: Any) -> None:
    t, c = _pair(node)
    assert v_start(t, "test", "tone") < v_start(_ctl(c), "test", "tone") - 0.1


def j_renewal(node: Any) -> None:
    t, c = _pair(node)
    end_ext = sum_at_end(t, "extinction")
    assert sum_at_start(t, "test") > end_ext + 0.1
    assert abs(sum_at_start(_ctl(c), "test") - end_ext) < EPS


def j_reinstatement(node: Any) -> None:
    t, c = _pair(node)
    end_ext = sum_at_end(t, "extinction")
    assert sum_at_start(t, "test") > end_ext + 0.1
    assert abs(sum_at_start(_ctl(c), "test") - end_ext) < EPS


def j_spontaneous_recovery(node: Any) -> None:
    t, _ = _pair(node)
    assert v_start(t, "test", "tone") > v_end(t, "extinction", "tone") + EPS


def j_latent_extinction(node: Any) -> None:
    t, c = _pair(node)
    assert sum_at_start(t, "test") < sum_at_start(_ctl(c), "test") - 0.1


def j_pree(node: Any) -> None:
    t, c = _pair(node)
    assert v_start(t, "test", "tone") > v_start(_ctl(c), "test", "tone") + EPS, "PREE"


def j_contextual_extinction(node: Any) -> None:
    t, c = _pair(node)
    assert sum_at_start(t, "test") > sum_at_end(t, "extinction") + 0.1
    assert abs(sum_at_start(_ctl(c), "test")) < 0.05


def j_reconsolidation(node: Any) -> None:
    t, c = _pair(node)
    tc = _ctl(c)
    # Phenomenon: recovery after plain extinction, none after retrieval-extinction.
    assert v_start(tc, "test", "tone") > v_end(tc, "extinction", "tone") + EPS
    assert v_start(t, "test", "tone") <= v_end(t, "extinction", "tone")


def j_cta(node: Any) -> None:
    t, _ = _pair(node)
    assert len(t.records) == node.training_trials == 1
    assert t.final()[node.taste_cs] > 0


def j_generalization(node: Any) -> None:
    t, _ = _pair(node)
    near = v_start(t, "test_tone_750hz", "tone_750hz")
    far = v_start(t, "test_tone_500hz", "tone_500hz")
    assert near > far > EPS, "graded generalization"


def j_contextual_fear(node: Any) -> None:
    t, _ = _pair(node)
    assert len(t.records) == 1
    assert t.final()[node.context] > 0


def j_mediated(node: Any) -> None:
    t, _ = _pair(node)
    assert v_start(t, "test", "flavor", us="lithium") > EPS, "mediated aversion"


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

PREDICTS, PARTIAL, NO, UNSUPPORTED = "predicts", "partial", "no", "unsupported"

MECHANISM_NOTES: dict[str, str] = {
    "counterconditioning": "one R–W system per US; no reciprocal inhibition between US systems",
    "occasion-setting": "discrimination only; occasion setting proper is not elemental",
    "us-preexposure": "context blocking only; US habituation is not modelled",
    "renewal": "context B ends inhibitory, unlike the summation result for renewal",
    "reinstatement": "context excitation only; CS-specific part is not modelled",
    "latent-extinction": "only the context's share of V is lost",
    "contextual-extinction": "same mechanism as renewal (negative V of the extinction context)",
    "conditioned-taste-aversion": "one-trial acquisition only; CS-US delay is dropped",
    "contextual-fear-conditioning": "acquisition only; placement-to-shock interval is dropped",
}

Judge = Callable[[Any], None]

ROWS: list[tuple[int, str, str, Judge, bool]] = [
    # (#, fixture, outcome, judge, judge takes all cases)
    (1, "higher-order-conditioning", NO, j_higher_order, False),
    (2, "sensory-preconditioning", NO, j_sensory_preconditioning, False),
    (3, "counterconditioning", PARTIAL, j_counterconditioning, False),
    (4, "blocking", PREDICTS, j_blocking, False),
    (5, "overshadowing", PREDICTS, j_overshadowing, False),
    (6, "overexpectation", PREDICTS, j_overexpectation, False),
    (7, "super-conditioning", PREDICTS, j_super_conditioning, False),
    (8, "retrospective-revaluation", NO, j_retrospective_revaluation, True),
    (9, "conditioned-inhibition", PREDICTS, j_conditioned_inhibition, False),
    (10, "occasion-setting", PARTIAL, j_occasion_setting, False),
    (11, "inhibition-of-delay", UNSUPPORTED, j_unsupported, False),
    (12, "summation-retardation-test", PREDICTS, j_summation_retardation, True),
    (13, "latent-inhibition", NO, j_latent_inhibition, False),
    (14, "us-preexposure", PARTIAL, j_us_preexposure, False),
    (15, "renewal", PARTIAL, j_renewal, False),
    (16, "reinstatement", PARTIAL, j_reinstatement, False),
    (17, "spontaneous-recovery", NO, j_spontaneous_recovery, False),
    (18, "latent-extinction", PARTIAL, j_latent_extinction, False),
    (19, "pavlovian-pree", NO, j_pree, False),
    (20, "contextual-extinction", PARTIAL, j_contextual_extinction, False),
    (21, "reconsolidation-interference", NO, j_reconsolidation, False),
    (22, "conditioned-taste-aversion", PARTIAL, j_cta, False),
    (23, "stimulus-generalization", NO, j_generalization, False),
    (24, "contextual-fear-conditioning", PARTIAL, j_contextual_fear, False),
    (25, "peak-procedure", UNSUPPORTED, j_unsupported, False),
    (26, "mediated-conditioning", NO, j_mediated, False),
]


def _param(row: tuple[int, str, str, Judge, bool]) -> Any:
    num, name, outcome, _, _ = row
    marks = []
    if outcome == NO:
        marks.append(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="R–W does not predict this phenomenon (expected mismatch)",
            )
        )
    return pytest.param(row, id=f"{num:02d}-{name}", marks=marks)


def test_table_covers_every_fixture() -> None:
    if ROOT is None:
        pytest.skip("Tier B conformance fixtures not found")
    assert sorted(p.stem for p in ROOT.glob("*.json")) == sorted(r[1] for r in ROWS)
    assert [r[0] for r in ROWS] == list(range(1, 27))
    assert set(MECHANISM_NOTES) == {r[1] for r in ROWS if r[2] == PARTIAL}


@pytest.mark.parametrize("row", [_param(r) for r in ROWS])
def test_phenomenon(row: tuple[int, str, str, Judge, bool]) -> None:
    _, name, _, judge, all_cases = row
    cases = _cases(name)
    if all_cases:
        judge(cases)
    else:
        for node in cases:
            judge(node)
