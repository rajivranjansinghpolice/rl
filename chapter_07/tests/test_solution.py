"""
Tests for Chapter 7: n-step Bootstrapping.

Reference: Sutton & Barto (2nd Ed), Chapter 7.

Tests for: the n-step TD prediction box (Sec. 7.1, Example 7.1),
n-step Sarsa (Sec. 7.2), n-step Tree Backup (Sec. 7.5),
SARSA(lambda) control (Sec. 12.2-style).
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_envs.random_walk import RandomWalkEnv
from chapter_07.solution import (
    n_step_sarsa,
    n_step_td_prediction,
    run_n_step_prediction_experiment,
    sarsa_lambda,
    tree_backup,
)


class TestNStepSarsa:
    """Tests for n-step Sarsa (Sec. 7.2)."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, _ = n_step_sarsa(env, n=4, n_episodes=100, epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = n_step_sarsa(env, n=4, n_episodes=10, epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * 2

    def test_learns_right_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = n_step_sarsa(env, n=4, n_episodes=5000, epsilon=0.1, seed=0)
        assert q_table[("E", "right")] > q_table[("E", "left")]

    def test_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            n_step_sarsa(RandomWalkEnv(seed=0), n=0)

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            n_step_sarsa(RandomWalkEnv(seed=0), epsilon=0.0)

    def test_one_step_is_sarsa_zero_behavior(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, q_table = n_step_sarsa(env, n=1, n_episodes=100, epsilon=0.1, seed=0)
        assert len(policy) == len(RandomWalkEnv.STATES)


class TestTreeBackup:
    """Tests for n-step Tree Backup (Sec. 7.5)."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, _ = tree_backup(env, n=4, n_episodes=100, epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = tree_backup(env, n=4, n_episodes=10, epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * 2

    def test_learns_right_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = tree_backup(env, n=4, n_episodes=5000, epsilon=0.1, seed=0)
        assert q_table[("E", "right")] > q_table[("E", "left")]

    def test_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            tree_backup(RandomWalkEnv(seed=0), n=0)

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            tree_backup(RandomWalkEnv(seed=0), epsilon=0.0)

    def test_converged_values_finite(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = tree_backup(env, n=2, n_episodes=500, epsilon=0.3, seed=0)
        assert all(np.isfinite(v) for v in q_table.values())


class TestSarsaLambda:
    """Tests for SARSA(lambda) control (Sec. 12.2-style)."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, _ = sarsa_lambda(env, lam=0.9, n_episodes=100, epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa_lambda(env, lam=0.9, n_episodes=10, epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * 2

    def test_learns_right_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa_lambda(env, lam=0.9, n_episodes=5000, epsilon=0.1, seed=0)
        assert q_table[("E", "right")] > q_table[("E", "left")]

    def test_lambda_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            sarsa_lambda(RandomWalkEnv(seed=0), lam=1.5)

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            sarsa_lambda(RandomWalkEnv(seed=0), epsilon=0.0)

    def test_converged_values_finite(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa_lambda(env, lam=0.5, n_episodes=500, epsilon=0.1, seed=0)
        assert all(np.isfinite(v) for v in q_table.values())


class TestNStepTDPrediction:
    """n-step TD prediction (Sec. 7.1, Example 7.1). Book Sec. 7.1."""

    def test_returns_all_19_states(self) -> None:
        env = RandomWalkEnv(n_states=19, left_reward=-1.0, right_reward=1.0, seed=0)
        values = n_step_td_prediction(env, n=4, alpha=0.4, n_episodes=3, seed=0)
        assert set(values) == set(env.STATES)

    def test_prediction_reduces_error(self) -> None:
        env = RandomWalkEnv(n_states=19, left_reward=-1.0, right_reward=1.0, seed=0)
        true_values = env.true_values()
        init_rms = float(np.sqrt(np.mean([(0.0 - v) ** 2 for v in true_values.values()])))
        values = n_step_td_prediction(env, n=4, alpha=0.4, n_episodes=30, seed=0)
        trained_rms = float(np.sqrt(np.mean([(values[s] - true_values[s]) ** 2 for s in values])))
        assert trained_rms < init_rms

    def test_monte_carlo_like_large_n(self) -> None:
        env = RandomWalkEnv(seed=0)
        values = n_step_td_prediction(env, n=100, alpha=0.1, n_episodes=40, seed=0)
        assert all(np.isfinite(v) for v in values.values())

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            n_step_td_prediction(RandomWalkEnv(seed=0), alpha=0.0)

    def test_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            n_step_td_prediction(RandomWalkEnv(seed=0), n=0)

    def test_experiment_output_shapes(self) -> None:
        result = run_n_step_prediction_experiment(ns=(1, 4), alphas=(0.1, 0.4), repeats=2, n_episodes=3, seed=0)
        assert result["ns"] == [1, 4]
        assert result["alphas"] == [0.1, 0.4]
        assert len(result["curves"][1]) == 2
        assert len(result["curves"][4]) == 2

    def test_intermediate_n_beats_extremes(self) -> None:
        # book finding: intermediate n works best
        result = run_n_step_prediction_experiment(
            ns=(1, 4, 512), alphas=(0.4,), repeats=10, n_episodes=10, seed=0
        )
        assert result["curves"][4][0] < result["curves"][1][0]
        assert result["curves"][4][0] < result["curves"][512][0]
