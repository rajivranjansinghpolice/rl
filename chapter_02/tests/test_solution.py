"""
Tests for Chapter 2 Bandit Algorithms.

Reference: Sutton & Barto (2nd Ed), Chapter 2.

Tests for: Alg 2.1 (k-armed bandit), Alg 2.2 (action-value),
Alg 2.3 (epsilon-greedy), Alg 2.4 (UCB), Alg 2.5 (gradient bandit),
plus the exploration strategies (UCB, optimism in the face of
uncertainty, descending bonus, Q-learning with exploration).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chapter_02.solution import (
    ActionValueEstimator,
    BanditProblem,
    ContextualBandit,
    GradientBanditAgent,
    _ucb_experiment_selector,
    descending_bonus,
    descending_bonus_select,
    make_small_gridworld,
    q_learning_descending_bonus,
    q_learning_optimistic,
    q_learning_ucb,
    run_descending_bonus_bandit,
    run_epsilon_greedy_bandit,
    run_optimistic_greedy_bandit,
    run_ucb_bandit,
    select_epsilon_greedy,
    select_ucb,
    simulate_nonstationary,
    simulate_optimistic_initial_values,
    simulate_ucb_uncertainty,
    ucb_bonus,
)


class TestBanditProblem:
    """Tests for k-armed bandit environment (Algorithm 2.1). Book Sec 2.2."""

    def test_init_default_k(self) -> None:
        bandit = BanditProblem()
        assert bandit.k == 10

    def test_init_custom_k(self) -> None:
        bandit = BanditProblem(k=5)
        assert bandit.k == 5
        assert len(bandit.q_star) == 5

    def test_step_returns_float(self) -> None:
        bandit = BanditProblem(seed=42)
        reward = bandit.step(0)
        assert isinstance(reward, float)

    def test_step_returns_finite(self) -> None:
        bandit = BanditProblem(seed=42)
        for _ in range(100):
            reward = bandit.step(0)
            assert np.isfinite(reward)

    def test_best_action_exists(self) -> None:
        bandit = BanditProblem(seed=42)
        best = bandit.best_action()
        assert 0 <= best < bandit.k

    def test_step_reproducible_with_seed(self) -> None:
        b1 = BanditProblem(seed=123)
        b2 = BanditProblem(seed=123)
        assert b1.step(0) == b2.step(0)


class TestActionValueEstimator:
    """Tests for action-value estimation (Algorithm 2.2). Book Sec 2.3."""

    def test_init(self) -> None:
        est = ActionValueEstimator(k=5)
        assert est.k == 5
        assert np.all(est.Q == 0)
        assert np.all(est.N == 0)

    def test_update_changes_Q(self) -> None:
        est = ActionValueEstimator(k=3)
        est.update(0, 1.0)
        assert est.Q[0] == pytest.approx(1.0)

    def test_update_accumulates(self) -> None:
        est = ActionValueEstimator(k=3)
        est.update(0, 1.0)
        est.update(0, 3.0)
        assert est.Q[0] == pytest.approx(2.0)

    def test_N_increments(self) -> None:
        est = ActionValueEstimator(k=3)
        est.update(0, 1.0)
        assert est.N[0] == 1

    def test_select_action_returns_int(self) -> None:
        est = ActionValueEstimator(k=5)
        action = est.select_action()
        assert isinstance(action, int)


class TestEpsilonGreedy:
    """Tests for epsilon-greedy action selection (Algorithm 2.3). Book Sec 2.3."""

    def test_epsilon_1_returns_valid(self) -> None:
        Q = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(42)
        action = select_epsilon_greedy(Q, 1.0, rng)
        assert 0 <= action < len(Q)

    def test_epsilon_0_selects_max(self) -> None:
        Q = np.array([1.0, 5.0, 2.0])
        rng = np.random.default_rng(42)
        action = select_epsilon_greedy(Q, 0.0, rng)
        assert action == 1

    def test_returns_int(self) -> None:
        Q = np.array([1.0, 2.0])
        action = select_epsilon_greedy(Q, 0.5)
        assert isinstance(action, int)


class TestUCB:
    """Tests for upper-confidence bound (Algorithm 2.4). Book Sec 2.4."""

    def test_tries_untried_first(self) -> None:
        Q = np.array([0.0, 0.0, 0.0])
        N = np.array([0, 0, 0])
        action = select_ucb(Q, N, 1, 2.0)
        assert action == 0

    def test_returns_valid_action(self) -> None:
        Q = np.array([1.0, 2.0, 3.0])
        N = np.array([5, 5, 5])
        action = select_ucb(Q, N, 100, 2.0)
        assert 0 <= action < 3

    def test_exploration_increases_with_c(self) -> None:
        Q = np.array([1.0, 1.0, 1.0])
        N = np.array([100, 100, 1])
        action = select_ucb(Q, N, 200, 10.0)
        assert action == 2


class TestGradientBandit:
    """Tests for gradient bandit algorithm (Algorithm 2.5). Book Sec 2.5."""

    def test_init(self) -> None:
        agent = GradientBanditAgent(k=5, alpha=0.1)
        assert agent.k == 5
        assert agent.alpha == 0.1

    def test_select_returns_int(self) -> None:
        agent = GradientBanditAgent(k=5)
        action = agent.select_action()
        assert isinstance(action, int)
        assert 0 <= action < 5

    def test_update_no_error(self) -> None:
        agent = GradientBanditAgent(k=5)
        agent.select_action()
        agent.update(0, 1.0)
        assert np.isfinite(agent.H).all()

    def test_preferences_learn_best_action(self) -> None:
        agent = GradientBanditAgent(k=5, alpha=0.5)
        for _ in range(100):
            a = agent.select_action()
            r = 1.0 if a == 0 else 0.0
            agent.update(a, r)
        assert agent.H[0] > agent.H[1]


class TestNonstationary:
    """Tests for non-stationary bandits (Exercise 2.3)."""

    def test_returns_arrays(self) -> None:
        Q, rewards = simulate_nonstationary(k=5, steps=100)
        assert Q.shape == (5,)
        assert rewards.shape == (100,)

    def test_rewards_finite(self) -> None:
        _, rewards = simulate_nonstationary(k=5, steps=100)
        assert np.all(np.isfinite(rewards))


class TestOptimisticInitialValues:
    """Tests for optimistic initial values (Exercise 2.4)."""

    def test_returns_tuple(self) -> None:
        Q, rate = simulate_optimistic_initial_values()
        assert Q.shape == (10,)
        assert 0 <= rate <= 1.0

    def test_optimistic_exploration(self) -> None:
        _, rate = simulate_optimistic_initial_values(initial_value=5.0)
        assert rate > 0.5


class TestUCBExperimentSelector:
    """Regression tests for the Figure 2.4 experiment selector."""

    def test_untried_arms_chosen_uniformly(self) -> None:
        selector = _ucb_experiment_selector("ucb", 2.0, 0.1)
        Q = np.zeros(4)
        N = np.zeros(4, dtype=int)
        picks = {
            selector(Q, N, 1, np.random.default_rng(seed)) for seed in range(60)
        }
        assert picks == {0, 1, 2, 3}

    def test_uses_ucb_bonus_after_all_arms_tried(self) -> None:
        selector = _ucb_experiment_selector("ucb", 2.0, 0.1)
        Q = np.array([0.0, 1.0, 0.0, 0.0])
        N = np.array([1, 1, 1, 1])
        picks = {
            selector(Q, N, 10, np.random.default_rng(seed)) for seed in range(60)
        }
        assert picks == {1}


class TestUCBUncertainty:
    """Tests for UCB with uncertainty (Exercise 2.5)."""

    def test_returns_tuple(self) -> None:
        Q, rate = simulate_ucb_uncertainty()
        assert Q.shape == (10,)
        assert 0 <= rate <= 1.0

    def test_ucb_selects_best(self) -> None:
        _, rate = simulate_ucb_uncertainty(c=2.0)
        assert rate > 0.5


class TestContextualBandit:
    """Tests for contextual bandits (Exercise 2.6)."""

    def test_init(self) -> None:
        cb = ContextualBandit(num_contexts=3, k=5)
        assert cb.num_contexts == 3
        assert cb.k == 5

    def test_select_returns_valid_action(self) -> None:
        cb = ContextualBandit(num_contexts=3, k=5)
        action = cb.select_action(context=0)
        assert 0 <= action < 5

    def test_update_no_error(self) -> None:
        cb = ContextualBandit(num_contexts=3, k=5)
        cb.select_action(0)
        cb.update(0, 0, 1.0)
        assert np.isfinite(cb.H).all()


class TestUCBBonus:
    """The UCB1 bonus term. Book eq. (2.8); Algorithm 2.4."""

    def test_bonus_matches_book_formula(self) -> None:
        assert ucb_bonus(10, 1, 2.0) == pytest.approx(2.0 * math.sqrt(math.log(10)))
        assert ucb_bonus(100, 25, 1.5) == pytest.approx(1.5 * math.sqrt(math.log(100) / 25))

    def test_untried_action_has_infinite_bonus(self) -> None:
        assert ucb_bonus(5, 0, 2.0) == math.inf

    def test_bonus_decreases_with_count(self) -> None:
        b1 = ucb_bonus(100, 1, 2.0)
        b2 = ucb_bonus(100, 4, 2.0)
        b3 = ucb_bonus(100, 100, 2.0)
        assert b1 > b2 > b3 > 0.0

    def test_bonus_grows_unboundedly_with_time(self) -> None:
        assert ucb_bonus(10**8, 4, 2.0) > ucb_bonus(10, 4, 2.0)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            ucb_bonus(0, 0, 2.0)
        with pytest.raises(ValueError):
            ucb_bonus(5, -1, 2.0)
        with pytest.raises(ValueError):
            ucb_bonus(5, 1, -0.5)

    def test_select_tries_untried_actions_first(self) -> None:
        rng = np.random.default_rng(0)
        q_values = np.array([0.0, 5.0, 0.0, 0.0])
        counts = np.array([3, 3, 0, 0])
        # the untried actions both outrank the q=5 action via inf bonus
        picks = {select_ucb(q_values, counts, 4, 2.0, rng) for _ in range(50)}
        assert picks <= {2, 3}
        assert len(picks) == 2

    def test_select_prefers_close_estimate_with_high_uncertainty(self) -> None:
        rng = np.random.default_rng(0)
        q_values = np.array([1.0, 0.5, 0.5])
        counts = np.array([1000, 1000, 1])
        # action 2 is nearly as valuable but far more uncertain
        assert select_ucb(q_values, counts, 2001, 2.0, rng) == 2


class TestBanditProblemCounts:
    """Visit counts on the k-armed testbed (book Sec 2.3 / Sec 2.7)."""

    def test_step_increments_counts(self) -> None:
        bandit = BanditProblem(k=10, seed=0)
        bandit.step(3)
        bandit.step(3)
        bandit.step(7)
        assert bandit.counts[3] == 2
        assert bandit.counts[7] == 1

    def test_optimal_action_is_argmax(self) -> None:
        bandit = BanditProblem(k=10, seed=0)
        assert bandit.best_action() == int(np.argmax(bandit.q_star))

    def test_action_out_of_range_raises(self) -> None:
        bandit = BanditProblem(k=10, seed=0)
        with pytest.raises(ValueError):
            bandit.step(10)

    def test_reproducible_with_seed(self) -> None:
        b1 = BanditProblem(k=10, seed=42)
        b2 = BanditProblem(k=10, seed=42)
        assert np.array_equal(b1.q_star, b2.q_star)
        assert np.array_equal(b1.counts, b2.counts)


class TestDescendingBonus:
    """Optimistic bonus descending with visit counts. Exercise 11.1 (OFU)."""

    def test_bonus_is_maximal_for_untried(self) -> None:
        assert descending_bonus(0, 2.0) == pytest.approx(2.0)

    def test_bonus_descends_monotonically(self) -> None:
        bonuses = [descending_bonus(n, 2.0) for n in range(10)]
        assert all(b1 > b2 for b1, b2 in zip(bonuses, bonuses[1:]))
        assert bonuses[-1] < bonuses[0]

    def test_bonus_vanishes(self) -> None:
        assert descending_bonus(10**6, 2.0) < 1e-4

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            descending_bonus(-1)
        with pytest.raises(ValueError):
            descending_bonus(0, -0.5)

    def test_select_with_large_bonus_tries_all_arms_first(self) -> None:
        for seed in range(10):
            result = run_descending_bonus_bandit(steps=10, b0=10.0, seed=seed)
            assert min(result["counts"]) >= 1

    def test_descending_bonus_agent_reaches_goal(self) -> None:
        env = make_small_gridworld(seed=0)
        _, _, stats = q_learning_descending_bonus(
            env, n_episodes=15, b0=2.0, seed=0, start_state="0,0", max_steps=300,
        )
        assert min(stats["episode_rewards"][-3:]) > 0.0

    def test_descending_bonus_visits_all_start_actions_early(self) -> None:
        env = make_small_gridworld(seed=0)
        _, counts, _ = q_learning_descending_bonus(
            env, n_episodes=10, b0=2.0, seed=0, start_state="0,0", max_steps=300,
        )
        start_counts = [counts[("0,0", a)] for a in ("up", "down", "left", "right")]
        assert min(start_counts) >= 1

    def test_descending_bonus_select_prefers_untried(self) -> None:
        rng = np.random.default_rng(0)
        q_values = np.array([3.0, 0.0])
        counts = np.array([50, 0])
        assert descending_bonus_select(q_values, counts, b0=5.0, rng=rng) == 1


class TestBanditExplorationRunners:
    """Single-task bandit runners for exploration strategies."""

    def test_ucb_tries_every_arm_early(self) -> None:
        for seed in range(10):
            result = run_ucb_bandit(steps=10, seed=seed)
            assert min(result["counts"]) >= 1

    def test_ucb_beats_epsilon_greedy_on_average(self) -> None:
        ucb_rewards = [
            np.mean(run_ucb_bandit(steps=500, c=2.0, seed=seed)["rewards"][-100:])
            for seed in range(50)
        ]
        eps_rewards = [
            np.mean(
                run_epsilon_greedy_bandit(steps=500, epsilon=0.1, seed=seed)["rewards"]
                [-100:]
            )
            for seed in range(50)
        ]
        assert np.mean(ucb_rewards) > np.mean(eps_rewards)

    def test_ucb_optimal_action_frequency_high_late(self) -> None:
        result = run_ucb_bandit(steps=500, c=2.0, seed=0)
        late_optimal = np.mean(result["optimal"][-100:])
        assert late_optimal > 0.3

    def test_ucb_rewards_are_finite(self) -> None:
        result = run_ucb_bandit(steps=200, seed=0)
        assert all(np.isfinite(r) for r in result["rewards"])

    def test_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            run_ucb_bandit(steps=0)

    def test_optimistic_agent_tries_all_arms(self) -> None:
        for seed in range(20):
            result = run_optimistic_greedy_bandit(steps=40, init_q=5.0, seed=seed)
            assert min(result["counts"]) >= 1

    def test_zero_init_greedy_rarely_tries_all_arms(self) -> None:
        tried_all = 0
        for seed in range(20):
            result = run_epsilon_greedy_bandit(steps=40, epsilon=0.0, seed=seed)
            tried_all += min(result["counts"]) >= 1
        # unlike the optimistic agent, the zero-init greedy agent usually
        # locks onto one arm immediately and never explores (Fig 2.3)
        assert tried_all <= 5

    def test_optimistic_beats_zero_init_on_average(self) -> None:
        optimistic = [
            np.mean(run_optimistic_greedy_bandit(steps=400, init_q=5.0, seed=s)["rewards"][-100:])
            for s in range(20)
        ]
        zero_init = [
            np.mean(run_epsilon_greedy_bandit(steps=400, epsilon=0.0, seed=s)["rewards"][-100:])
            for s in range(20)
        ]
        assert np.mean(optimistic) > np.mean(zero_init)

    def test_optimistic_greedy_performance_approaches_optimal(self) -> None:
        result = run_optimistic_greedy_bandit(steps=600, init_q=5.0, seed=0)
        late_optimal = np.mean(result["optimal"][-100:])
        assert late_optimal > 0.4

    def test_steps_must_be_positive_optimistic(self) -> None:
        with pytest.raises(ValueError):
            run_optimistic_greedy_bandit(steps=0)


class TestQLearningExploration:
    """Q-learning with UCB / optimistic init / descending bonus exploration."""

    def test_ucb_agent_reaches_goal(self) -> None:
        env = make_small_gridworld(seed=0)
        _, _, stats = q_learning_ucb(env, n_episodes=30, seed=0, start_state="0,0")
        assert np.mean(stats["episode_steps"][-5:]) <= 8.0
        assert min(stats["episode_rewards"][-5:]) > 0.0

    def test_ucb_visits_all_start_actions(self) -> None:
        env = make_small_gridworld(seed=0)
        _, counts, _ = q_learning_ucb(env, n_episodes=30, seed=0, start_state="0,0")
        for action in ("up", "down", "left", "right"):
            assert counts[("0,0", action)] >= 1

    def test_ucb_explores_broadly_on_small_mdp(self) -> None:
        env = make_small_gridworld(seed=0)
        _, counts, _ = q_learning_ucb(env, n_episodes=30, seed=0, start_state="0,0")
        visited = sum(1 for v in counts.values() if v > 0)
        assert visited >= 20

    def test_reproducible_with_seed(self) -> None:
        _, _, s1 = q_learning_ucb(
            make_small_gridworld(seed=0), n_episodes=5, seed=7, start_state="0,0"
        )
        _, _, s2 = q_learning_ucb(
            make_small_gridworld(seed=0), n_episodes=5, seed=7, start_state="0,0"
        )
        assert s1["episode_steps"] == s2["episode_steps"]

    def test_optimistic_q_learning_balances_start_actions(self) -> None:
        env = make_small_gridworld(seed=0)
        _, counts, _ = q_learning_optimistic(
            env, n_episodes=10, initial_q=5.0, epsilon=0.0, seed=0,
            start_state="0,0", tie_break="first", max_steps=300,
        )
        # forced balanced exploration: every action from the start state
        # is tried a comparable number of times
        start_counts = [counts[("0,0", a)] for a in ("up", "down", "left", "right")]
        assert min(start_counts) >= 30
        assert max(start_counts) - min(start_counts) <= 10

    def test_zero_init_greedy_stuck_on_blocked_action(self) -> None:
        env = make_small_gridworld(seed=0)
        _, counts, stats = q_learning_optimistic(
            env, n_episodes=10, initial_q=0.0, epsilon=0.0, seed=0,
            start_state="0,0", tie_break="first", max_steps=300,
        )
        # deterministic first-index tie-breaking picks the blocked 'up'
        # forever: zero-init greedy never explores and never reaches the goal
        assert counts[("0,0", "up")] == 3000
        assert counts[("0,0", "right")] == 0
        assert sum(stats["episode_rewards"]) == pytest.approx(0.0)

    def test_optimistic_visits_every_start_action(self) -> None:
        for seed in range(5):
            env = make_small_gridworld(seed=seed)
            _, counts, _ = q_learning_optimistic(
                env, n_episodes=5, initial_q=5.0, epsilon=0.0, seed=seed,
                start_state="0,0", tie_break="first", max_steps=300,
            )
            assert all(counts[("0,0", a)] >= 1 for a in ("up", "down", "left", "right"))
