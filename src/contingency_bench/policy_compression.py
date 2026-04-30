"""Policy-Compression reinforcement learning agent (information-theoretic regularization).

Maximizes ``E[Reward] - beta * KL(pi || pi_0)`` where ``pi_0`` is a default
(prior) policy. The complexity penalty pulls the current policy towards this
prior, producing the undermatching that is empirically observed in many
biological subjects on concurrent VI schedules.

This module is inlined here (rather than imported from a sibling package)
so that ``contingency-bench`` has zero hard runtime dependencies on other
in-monorepo packages — see README "Local dependency notes" for the swap-in
path once upstream modules are published.

References
----------
Lai & Gershman (2021). Policy compression: An information bottleneck in action
selection. Psychology of Learning and Motivation, 74, 195-232.
https://doi.org/10.1016/bs.plm.2021.02.004
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PolicyCompressionAgent:
    """Two-or-more action policy-compression RL agent.

    Parameters
    ----------
    n_actions:
        Number of discrete actions available.
    beta:
        Complexity penalty / inverse temperature. Larger ``beta`` weights the
        prior policy more heavily and produces stronger undermatching.
    learning_rate:
        Step size for the incremental Q-value update.
    prior_policy:
        Optional categorical prior ``pi_0``. Defaults to uniform.
    """

    def __init__(
        self,
        n_actions: int = 2,
        beta: float = 0.1,
        learning_rate: float = 0.1,
        prior_policy: NDArray[np.float64] | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.beta = beta
        self.learning_rate = learning_rate
        self.q_values: NDArray[np.float64] = np.zeros(n_actions, dtype=np.float64)
        if prior_policy is None:
            self.prior_policy: NDArray[np.float64] = (
                np.ones(n_actions, dtype=np.float64) / n_actions
            )
        else:
            self.prior_policy = np.asarray(prior_policy, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset Q-values while keeping the prior policy fixed."""
        self.q_values = np.zeros(self.n_actions, dtype=np.float64)

    def get_policy(self) -> NDArray[np.float64]:
        """Compute ``pi(a) = pi_0(a) * exp(Q_a / beta) / Z``.

        Returns a numerically-stable categorical distribution. When ``beta`` is
        effectively zero the agent acts greedily.
        """
        if self.beta < 1e-8:
            policy = np.zeros(self.n_actions, dtype=np.float64)
            policy[int(np.argmax(self.q_values))] = 1.0
            return policy

        max_q = float(np.max(self.q_values))
        weighted_exp = self.prior_policy * np.exp((self.q_values - max_q) / self.beta)
        sum_exp = float(np.sum(weighted_exp))
        if sum_exp < 1e-15:
            return np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        return weighted_exp / sum_exp

    def choose_action(self) -> int:
        """Sample an action from the current policy."""
        policy = self.get_policy()
        return int(np.random.choice(self.n_actions, p=policy))

    def update(self, action: int, reward: float) -> None:
        """Incremental update of the Q-value for the chosen action."""
        self.q_values[action] += self.learning_rate * (reward - self.q_values[action])
