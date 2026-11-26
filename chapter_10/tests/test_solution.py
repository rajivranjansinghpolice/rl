"""Tests for Chapter 10 - On-policy Control with Approximation."""
from __future__ import annotations

import numpy as np
import pytest

from chapter_10.solution import (
    AccessControlQueue,
    DifferentialSarsaAgent,
    EpisodicSemiGradientSarsaAgent,
    mountain_car_features,
    run_access_control,
    run_mountain_car,
)


class TestMountainCarFeatures:
    def test_shape_and_binary(self) -> None:
        f = mountain_car_features(0.0, 0.0)
        assert f.shape == (8 * 8 * 8,)
        assert set(f) <= {0.0, 1.0}
        assert np.sum(f) == 8  # one tile per tiling


class TestEpisodicSemiGradientSarsa:
    def test_first_update_changes_weights_one_action(self) -> None:
        agent = EpisodicSemiGradientSarsaAgent(8, n_actions=3, alpha=0.1, seed=0)
        x = np.zeros(8)
        x[0] = 1.0
        # hand-computed update: delta = R + gamma * Q(s', a') - Q(s, a)
        # = 0 + 1.0 * 0 - 0 = 0 by default? No -- Q(s', a') = 0 initially.
        agent.update_sarsa(x, 0, reward=-1.0, next_x=None, next_action=None)
        expected = (0.1 / 8.0) * (-1.0) * x
        assert np.allclose(agent.weights[0], expected)
        assert np.allclose(agent.weights[1], np.zeros(8))

    def test_q_values_shape(self) -> None:
        agent = EpisodicSemiGradientSarsaAgent(512, n_actions=3, seed=0)
        f = mountain_car_features(0.0, 0.0)
        assert agent.q_values(f).shape == (3,)

    def test_act_returns_valid_action(self) -> None:
        agent = EpisodicSemiGradientSarsaAgent(512, n_actions=3, seed=0)
        f = mountain_car_features(0.0, 0.0)
        a = agent.act(f)
        assert a in (0, 1, 2)


class TestRunMountainCar:
    def test_returns_episode_lengths(self) -> None:
        _, lengths = run_mountain_car(episodes=10, seed=0, max_steps=50, epsilon=0.5)
        assert len(lengths) == 10
        assert all(0 < l <= 50 for l in lengths)

    def test_weights_finite(self) -> None:
        agent, _ = run_mountain_car(episodes=10, seed=0, max_steps=50, epsilon=0.5)
        assert np.all(np.isfinite(agent.weights))


class TestDifferentialSarsa:
    def test_average_reward_update_moves_r_bar(self) -> None:
        agent = DifferentialSarsaAgent(4, n_actions=2, alpha=0.1, alpha_bar=0.1,
                                       epsilon=0.0, seed=0)
        x = np.array([1.0, 0.0, 0.0, 1.0])
        before = agent.r_bar
        agent.update(x, 0, reward=1.0, next_x=x, next_action=0)
        after = agent.r_bar
        # delta = 1 - 0 + 0 - 0 = 1 ; r_bar += 0.1 * 1
        assert after == pytest.approx(before + 0.1 * 1.0)

    def test_weights_update_scales_by_alpha(self) -> None:
        agent = DifferentialSarsaAgent(4, n_actions=2, alpha=0.1, alpha_bar=0.1,
                                       epsilon=0.0, seed=0)
        x = np.array([1.0, 0.0, 0.0, 1.0])
        agent.update(x, 1, reward=2.0, next_x=x, next_action=0)
        assert agent.weights[1][0] != 0.0


class TestAccessControlQueue:
    def test_full_queue_rejects_arrival(self) -> None:
        env = AccessControlQueue(capacity=2, p_high=1.0, p_low=1.0, seed=0)
        env.h, env.l = 2, 2
        _, reward = env.step(0)
        assert env.h == 2 and env.l == 2
        assert reward == 0.0

    def test_serving_high_priority_pays_more(self) -> None:
        env = AccessControlQueue(capacity=5, p_high=0.0, p_low=0.0, seed=0)
        env.h, env.l = 2, 2
        _, reward_high = env.step(1)
        env.h, env.l = 2, 2
        _, reward_low = env.step(2)
        assert reward_high == 8.0
        assert reward_low == 1.0

    def test_state_features_bounded(self) -> None:
        env = AccessControlQueue(capacity=10)
        f = env.state_features((10, 10))
        assert f.shape == (6,)
        assert np.all(f <= 1.0)

    def test_run_access_control_finite(self) -> None:
        agent, moving = run_access_control(n_steps=300, seed=0, plot_window=50)
        assert np.isfinite(agent.r_bar)
        assert len(moving) == 300 - 50 + 1
