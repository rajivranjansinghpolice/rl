"""Tests for Chapter 11 - Off-policy Methods with Approximation."""
from __future__ import annotations

import numpy as np
import pytest

from chapter_11.solution import (
    BEHAVIOR,
    ACTION_A,
    ACTION_B,
    N_OUTER,
    N_STATES,
    SemiGradientOffPolicyAgent,
    baird_features,
    baird_transition_probs,
    behavior_stationary,
    mspbe,
    mspbe_gradient_descent,
    projected_bellman_solution,
    target_transition_matrix,
)


class TestBairdMDP:
    def test_features_shape(self) -> None:
        phi = baird_features()
        assert phi.shape == (N_STATES, N_OUTER)
        assert np.all(phi[0, :] == 2.0)
        for i in range(N_OUTER):
            assert phi[i + 1, i] == 1.0
            assert np.count_nonzero(phi[i + 1, :]) == 1

    def test_transitions_are_distributions(self) -> None:
        t = baird_transition_probs()
        for (s, a), outcomes in t.items():
            total = sum(p for _, p in outcomes)
            assert total == pytest.approx(1.0)

    def test_outer_a_goes_to_center(self) -> None:
        t = baird_transition_probs()
        for i in range(N_OUTER):
            s = i + 1
            assert t[(s, ACTION_A)] == [(0, 1.0)]

    def test_outer_b_hub_to_outer_1(self) -> None:
        t = baird_transition_probs()
        for i in range(N_OUTER):
            s = i + 1
            assert t[(s, ACTION_B)] == [(1, 1.0)]

    def test_center_uniform(self) -> None:
        t = baird_transition_probs()
        for a in (ACTION_A, ACTION_B):
            assert len({p for _, p in t[(0, a)]}) == 1
            assert t[(0, a)][0][1] == pytest.approx(1.0 / N_OUTER)


class TestBehaviorStationary:
    def test_sums_to_one(self) -> None:
        mu = behavior_stationary()
        assert mu.sum() == pytest.approx(1.0)
        assert np.all(mu >= 0.0)

    def test_concentrated_off_center(self) -> None:
        # hub dynamics: the behavior distribution concentrates on outer_1
        mu = behavior_stationary()
        assert mu[1] > 0.8
        assert mu[0] < 0.1


class TestExpectedUpdateMatrix:
    """The core counterexample: H is indefinite under the off-policy mu."""

    def test_h_has_negative_real_eigenvalue(self) -> None:
        phi = baird_features()
        mu = behavior_stationary()
        P = target_transition_matrix()
        H = phi.T @ np.diag(mu) @ (phi - 0.99 * P @ phi)
        real_parts = np.linalg.eigvals(H).real
        assert np.any(real_parts < 0.0)

    def test_policy_matrix_is_valid(self) -> None:
        P = target_transition_matrix()
        assert P.shape == (N_STATES, N_STATES)
        assert np.allclose(P.sum(axis=1), 1.0)


class TestProjectedBellman:
    def test_zero_reward_solution_is_zero(self) -> None:
        # reward is 0 everywhere, so the projected Bellman solution is 0
        w = projected_bellman_solution()
        assert np.allclose(w, np.zeros(N_OUTER), atol=1e-10)


class TestMSPBE:
    def test_nonnegative_and_zero_at_solution(self) -> None:
        assert mspbe(np.zeros(N_OUTER)) == pytest.approx(0.0)
        w = np.ones(N_OUTER)
        assert mspbe(w) >= 0.0

    def test_gradient_descent_reduces_mspbe(self) -> None:
        weights_traj, mspbe_traj = mspbe_gradient_descent(n_steps=100, seed=0)
        assert mspbe_traj[-1] < mspbe_traj[0]
        assert len(weights_traj) == 100

    def test_gradient_descent_weight_norm_decreases(self) -> None:
        weights_traj, _ = mspbe_gradient_descent(n_steps=200, seed=0)
        initial = np.linalg.norm(weights_traj[0])
        final = np.linalg.norm(weights_traj[-1])
        assert final < initial


class TestSemiGradientOffPolicy:
    def test_exact_one_step_update(self) -> None:
        agent = SemiGradientOffPolicyAgent(alpha=0.1, gamma=0.5, seed=0)
        agent.weights = np.zeros(N_OUTER)
        agent.weights[1] = 2.0  # v(outer_2) = 2
        before = agent.weights.copy()
        # transition outer_2 -> center, action A, reward 0
        agent.update(2, ACTION_A, 0.0, 0)
        phi = baird_features()[2]
        delta = 0.5 * float(baird_features()[0] @ agent.weights) - float(phi @ before)
        rho = 1.0 / BEHAVIOR[ACTION_A]
        expected = before + 0.1 * rho * delta * phi
        assert np.allclose(agent.weights, expected)

    def test_divergence_growth(self) -> None:
        # off-policy hub MDP: ||w|| must grow over training (fixed seed)
        agent = SemiGradientOffPolicyAgent(alpha=0.005, gamma=0.99, seed=0)
        agent.weights = np.random.default_rng(0).normal(0.0, 1.0, size=N_OUTER)
        traj = agent.run(1200)
        norms = np.linalg.norm(traj, axis=1)
        assert norms[-1] > norms[0] * 2.0
