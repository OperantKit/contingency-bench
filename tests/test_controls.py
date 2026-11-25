"""Negative-control subjects: reproducibility from the seed alone."""

from __future__ import annotations

from contingency_bench.controls import RandomSubject


def _actions(subject: RandomSubject, n: int = 200) -> list[int]:
    return [subject.get_action("ignored") for _ in range(n)]


def test_random_subject_same_seed_reproduces_and_other_seed_differs() -> None:
    first = _actions(RandomSubject(seed=7))
    second = _actions(RandomSubject(seed=7))
    other = _actions(RandomSubject(seed=8))

    assert first == second
    assert first != other
    assert set(first) == {0, 1}
