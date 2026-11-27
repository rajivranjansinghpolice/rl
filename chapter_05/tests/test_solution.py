"""
Tests for Chapter 5: Monte Carlo Methods.

Reference: Sutton & Barto (2nd Ed), Chapter 5.

Tests for: Alg 5.1 (first-visit / every-visit MC prediction),
Alg 5.3 (on-policy MC control), Alg 5.4/5.5 (off-policy controls,
ordinary / weighted importance sampling), Exercise 5.1 (Blackjack).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_05.solution import (
    BlackjackEnv,
    _log_step_list,
    _one_state_is_samples,
    blackjack_stick_on_threshold,
    mc_control_epsilon_greedy,
    mc_control_off_policy,
    mc_every_visit_prediction,
    mc_exploring_starts,
    mc_first_visit_prediction,
    mc_off_policy_prediction,
)
from rl_envs.racetrack import RacetrackEnv


class TinyChainEnv:
    """Deterministic S --> A (r=+1) --> T (r=+1). gamma = 1 for all tests."""

    def __init__(self) -> None:
        self.state = None

    def reset(self):
        self.state = "S"
        return self.state

    def step(self, action: str):
        if self.state == "S":
            self.state = "A"
            return 1.0, "A", False
        assert self.state == "A"
        self.state = "T"
        return 1.0, "T", True


class InfiniteVarianceEnv:
    """One nonterminal state s. 'left': 0.9 -> s, 0.1 -> terminal (r=+1).
    'right': always terminal (r=0). Target policy: always left; behavior: 50/50."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        self.state = "s"
        return self.state

    def step(self, action: str):
        if action == "right":
            return 0.0, "T", True
        if self.rng.random() < 0.1:
            return 1.0, "T", True
        return 0.0, "s", False


def uniform_policy(state) -> np.ndarray:
    return np.array([0.5, 0.5])


class TestBlackjackEnv:
    """Tests for BlackjackEnv (Exercise 5.1 / Example 5.1). Book Sec 5.1."""

    def _scripted_env(self, script) -> BlackjackEnv:
        env = BlackjackEnv(seed=0)
        cards = iter(script)
        env._draw = lambda: next(cards)
        return env

    def test_reset_observation_shape(self) -> None:
        env = BlackjackEnv(seed=0)
        obs = env.reset()
        player_sum, dealer_showing, usable_ace = obs
        assert 2 <= player_sum <= 21
        assert 1 <= dealer_showing <= 10
        assert isinstance(usable_ace, bool)

    def test_usable_ace_counts_as_11(self) -> None:
        env = BlackjackEnv(seed=0)
        total, usable = env._hand_value([1, 10])
        assert total == 21
        assert usable is True

    def test_two_aces_collapse_to_12(self) -> None:
        env = BlackjackEnv(seed=0)
        total, usable = env._hand_value([1, 1, 10])
        assert total == 12
        assert usable is False

    def test_face_cards_are_ten(self) -> None:
        env = BlackjackEnv(seed=0)
        total, _ = env._hand_value([11, 12])
        assert total == 20

    def test_player_natural_wins(self) -> None:
        env = self._scripted_env([1, 11, 7, 9])
        env.reset()
        reward, _, done = env.step("hit")
        assert done is True
        assert reward == pytest.approx(1.0)

    def test_both_natural_draw(self) -> None:
        env = self._scripted_env([1, 11, 1, 10])
        env.reset()
        reward, _, done = env.step("hit")
        assert done is True
        assert reward == pytest.approx(0.0)

    def test_player_bust_loses(self) -> None:
        env = self._scripted_env([10, 6, 9, 7, 6])
        env.reset()
        reward, _, done = env.step("hit")
        assert done is True
        assert reward == pytest.approx(-1.0)

    def test_dealer_bust_player_wins(self) -> None:
        env = self._scripted_env([10, 7, 10, 6, 8])
        env.reset()
        reward, _, done = env.step("stick")
        assert done is True
        assert reward == pytest.approx(1.0)

    def test_draw_on_equal_sums(self) -> None:
        env = self._scripted_env([10, 7, 10, 7])
        env.reset()
        reward, _, done = env.step("stick")
        assert done is True
        assert reward == pytest.approx(0.0)

    def test_dealer_must_hit_below_17(self) -> None:
        env = self._scripted_env([10, 7, 10, 6, 9])
        env.reset()  # player 17, dealer 16 -> dealer must hit
        reward, _, done = env.step("stick")
        assert done is True
        assert reward == pytest.approx(1.0)  # 16+9=25, dealer busts -> player wins

    def test_invalid_action_raises(self) -> None:
        env = BlackjackEnv(seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step("double")


class TestPrediction:
    """Tests for MC prediction (Algorithm 5.1). Book Sec 5.1."""

    def test_first_visit_exact_on_deterministic_env(self) -> None:
        env = TinyChainEnv()
        values = mc_first_visit_prediction(env, uniform_policy, n_episodes=5, gamma=1.0)
        assert values["S"] == pytest.approx(2.0)
        assert values["A"] == pytest.approx(1.0)

    def test_every_visit_exact_on_deterministic_env(self) -> None:
        env = TinyChainEnv()
        values = mc_every_visit_prediction(env, uniform_policy, n_episodes=5, gamma=1.0)
        assert values["S"] == pytest.approx(2.0)
        assert values["A"] == pytest.approx(1.0)

    def test_policy_fn_must_be_callable(self) -> None:
        with pytest.raises(TypeError):
            mc_first_visit_prediction(TinyChainEnv(), None)

    def test_policy_fn_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            mc_first_visit_prediction(TinyChainEnv(), lambda s: np.array([0.5]), n_episodes=1)

    def test_policy_fn_probs_must_sum_to_one(self) -> None:
        def bad_policy(s):
            return np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            mc_first_visit_prediction(TinyChainEnv(), bad_policy, n_episodes=1)


class TestOnPolicyControl:
    """Tests for on-policy MC control (Algorithm 5.3). Book Sec 5.4."""

    def test_returns_policy_q_rewards(self) -> None:
        env = TinyChainEnv()
        policy, q_table, rewards = mc_control_epsilon_greedy(
            env, n_episodes=10, epsilon=0.2, seed=0
        )
        assert len(rewards) == 10
        assert set(policy) == set(q_table)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            mc_control_epsilon_greedy(TinyChainEnv(), epsilon=0.0)

    def test_learns_after_many_episodes(self) -> None:
        env = BlackjackEnv(seed=0)
        _, q_table, _ = mc_control_epsilon_greedy(env, n_episodes=5_000, epsilon=0.2, seed=0)
        assert len(q_table) > 0


class TestOffPolicyPrediction:
    """Tests for off-policy prediction (Algorithms 5.4/5.5). Book Sec 5.5."""

    def test_weighted_is_infinite_variance_example(self) -> None:
        env = InfiniteVarianceEnv(seed=0)
        def target(s):
            return np.array([1.0, 0.0])
        def behavior(s):
            return np.array([0.5, 0.5])
        values = mc_off_policy_prediction(
            env, target, behavior, n_episodes=100, gamma=1.0, weighted=True, seed=0
        )
        assert values["s"] == pytest.approx(1.0)

    def test_ordinary_is_returns_entries(self) -> None:
        env = InfiniteVarianceEnv(seed=0)
        def target(s):
            return np.array([1.0, 0.0])
        def behavior(s):
            return np.array([0.5, 0.5])
        values = mc_off_policy_prediction(
            env, target, behavior, n_episodes=100, gamma=1.0, weighted=False, seed=0
        )
        assert "s" in values
        assert np.isfinite(values["s"])

    def test_coverage_required(self) -> None:
        env = InfiniteVarianceEnv(seed=0)
        def target(s):
            return np.array([1.0, 0.0])
        def behavior(s):
            return np.array([0.0, 1.0])  # never takes 'left'
        values = mc_off_policy_prediction(
            env, target, behavior, n_episodes=10, gamma=1.0, weighted=True, seed=0
        )
        assert values.get("s", 0.0) == pytest.approx(0.0)


class TestOffPolicyControl:
    """Tests for off-policy MC control (Algorithms 5.4/5.5). Book Sec 5.7."""

    def test_returns_greedy_policy(self) -> None:
        env = TinyChainEnv()
        policy, q_table, rewards = mc_control_off_policy(env, n_episodes=10, epsilon=0.2, seed=0)
        assert len(rewards) == 10
        for state, probs in policy.items():
            assert probs.sum() == pytest.approx(1.0)
            q = q_table[state]
            best = np.flatnonzero(q == q.max())
            assert np.all(probs[best] == pytest.approx(1.0 / len(best)))

    def test_ordinary_is_control_runs(self) -> None:
        env = TinyChainEnv()
        policy, q_table, rewards = mc_control_off_policy(
            env, n_episodes=10, epsilon=0.2, weighted=False, seed=0
        )
        assert len(rewards) == 10

    def test_learning_high_sums_stick(self) -> None:
        env = BlackjackEnv(seed=0)
        _, q_table, _ = mc_control_off_policy(env, n_episodes=5_000, epsilon=0.3, seed=0)
        for dealer_showing in (1, 5, 10):
            key = (20, dealer_showing, False)
            if key in q_table:
                assert q_table[key][1] >= q_table[key][0]


def test_blackjack_stick_on_threshold() -> None:
    policy = blackjack_stick_on_threshold(20)
    assert policy((17, 10, False)).tolist() == [1.0, 0.0]
    assert policy((20, 10, False)).tolist() == [0.0, 1.0]
    assert policy((21, 10, True)).tolist() == [0.0, 1.0]


class TestExploringStarts:
    def test_tiny_chain_mechanics(self) -> None:
        env = TinyChainEnv()
        policy, q_table, rewards = mc_exploring_starts(env, n_episodes=50, seed=0)
        assert len(rewards) == 50
        assert rewards == [2.0] * 50  # deterministic chain: +1 + +1
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)
        assert "S" in q_table

    def test_blackjack_small_run(self) -> None:
        env = BlackjackEnv(seed=0)
        policy, q_table, rewards = mc_exploring_starts(env, n_episodes=3000, seed=0)
        assert len(rewards) == 3000
        for state, q in q_table.items():
            assert np.all(q >= -1.0) and np.all(q <= 1.0)
            greedy = np.flatnonzero(q == q.max())
            assert policy[state][greedy].sum() == pytest.approx(1.0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)


class TestOneStateIs:
    def test_samples_are_zero_or_powers_of_two(self) -> None:
        s = _one_state_is_samples(5_000, seed=0)
        assert np.all(s >= 0)
        nonzero = s[s > 0]
        assert nonzero.size > 0
        assert np.all(nonzero >= 2.0)
        log2 = np.log2(nonzero)
        assert np.allclose(log2, np.round(log2))

    def test_high_absorb_prob_produces_many_wins(self) -> None:
        s = _one_state_is_samples(200, seed=0, p_stay=0.1)  # absorb prob 0.9
        assert s.sum() > 100.0

    def test_log_step_list_exact(self) -> None:
        assert _log_step_list(10_000) == (
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
            + [10, 20, 30, 40, 50, 60, 70, 80, 90]
            + [100, 200, 300, 400, 500, 600, 700, 800, 900]
            + [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
            + [10_000]
        )
        assert _log_step_list(50)[-1] == 50
        assert len(set(_log_step_list(100_000))) == len(_log_step_list(100_000))


class TestRacetrackControl:
    def test_epsilon_greedy_control_runs_on_racetrack(self) -> None:
        env = RacetrackEnv(seed=0, noise_prob=0.0, max_steps=100)
        policy, q_table, rewards = mc_control_epsilon_greedy(
            env, n_episodes=60, epsilon=0.3, seed=0
        )
        assert len(rewards) == 60
        assert all(isinstance(r, float) for r in rewards)
        for probs in policy.values():
            assert probs.shape == (9,)
            assert probs.sum() == pytest.approx(1.0)
        for q in q_table.values():
            assert q.shape == (9,)
