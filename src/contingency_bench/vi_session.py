"""Concurrent VI-VI session simulator used by the LLM benchmark.

A simplified two-key concurrent variable-interval environment: each step,
each key independently arms with probability ``rate_i``; the chosen key
delivers reinforcement iff it is currently armed.
"""

from __future__ import annotations

import numpy as np


class ConcurrentVISession:
    """Two-key concurrent VI environment with Bernoulli arming per step."""

    def __init__(self, rates: tuple[float, float]) -> None:
        if len(rates) != 2:
            raise ValueError(f"rates must be a 2-tuple, got length {len(rates)}")
        if any(r < 0.0 or r > 1.0 for r in rates):
            raise ValueError(f"rates must lie in [0, 1], got {rates}")
        self.rates = rates
        self.available: list[bool] = [False, False]
        self.responses: list[int] = [0, 0]
        self.reinforcements: list[int] = [0, 0]

    def step(self, action: int) -> bool:
        """Take one response, advance arming, return True iff reinforced."""
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action}")
        for i in range(2):
            if not self.available[i] and np.random.random() < self.rates[i]:
                self.available[i] = True

        self.responses[action] += 1
        if self.available[action]:
            self.reinforcements[action] += 1
            self.available[action] = False
            return True
        return False
