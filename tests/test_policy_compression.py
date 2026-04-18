"""Policy-compression agent: shape, normalization, greedy-limit invariants."""

from __future__ import annotations

import numpy as np
import pytest

from contingency_bench.policy_compression import PolicyCompressionAgent


def test_policy_is_normalized_distribution() -> None:
    agent = PolicyCompressionAgent(n_actions=3, beta=0.2)
    policy = agent.get_policy()
    assert policy.shape == (3,)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0.0).all()


def test_uniform_prior_yields_uniform_policy_at_init() -> None:
    agent = PolicyCompressionAgent(n_actions=4, beta=0.5)
    policy = agent.get_policy()
    np.testing.assert_allclose(policy, np.full(4, 0.25))


def test_greedy_limit_when_beta_is_effectively_zero() -> None:
    agent = PolicyCompressionAgent(n_actions=3, beta=1e-12)
    agent.q_values = np.array([0.1, 5.0, 0.2])
    policy = agent.get_policy()
    assert policy.argmax() == 1
    assert policy[1] == pytest.approx(1.0)


def test_choose_action_returns_valid_index() -> None:
    agent = PolicyCompressionAgent(n_actions=2, beta=0.3)
    for _ in range(20):
        action = agent.choose_action()
        assert action in (0, 1)


def test_update_moves_q_value_toward_reward() -> None:
    agent = PolicyCompressionAgent(n_actions=2, beta=0.3, learning_rate=0.5)
    agent.update(action=0, reward=1.0)
    assert agent.q_values[0] == pytest.approx(0.5)
    assert agent.q_values[1] == pytest.approx(0.0)


def test_reset_clears_q_values_but_keeps_prior() -> None:
    prior = np.array([0.1, 0.9])
    agent = PolicyCompressionAgent(n_actions=2, beta=0.3, prior_policy=prior)
    agent.update(0, 1.0)
    agent.update(1, 1.0)
    agent.reset()
    np.testing.assert_allclose(agent.q_values, np.zeros(2))
    np.testing.assert_allclose(agent.prior_policy, prior)
