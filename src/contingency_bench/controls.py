"""Negative-control subjects for the law suite.

A negative control is a subject whose choices do not depend on the
reinforcement it receives. A law that such a subject also passes cannot tell a
subject that is sensitive to reinforcement from one that is not.

Each control owns its own random generator, so its response sequence is fixed
by its seed alone and does not depend on how many numbers the session draws
from numpy's global generator.
"""

from __future__ import annotations

import numpy as np

from contingency_bench.llm_client import BaseLLMClient


class RandomSubject(BaseLLMClient):
    """Chooses button 1 or button 2 with probability 0.5 on every trial.

    The prompt is ignored, so the choice does not depend on the condition or
    on the history of responses and reinforcers.
    """

    def __init__(self, seed: int | None = None, model_name: str = "random") -> None:
        super().__init__(model_name)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def get_action(self, prompt: str) -> int:  # noqa: ARG002 - prompt is ignored by design
        return int(self._rng.integers(0, 2))
