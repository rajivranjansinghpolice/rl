"""
Tests for Chapter 6: Temporal-Difference Learning.

Reference: Sutton & Barto (2nd Ed), Chapter 6.

Tests for: Alg 6.1 (Sarsa), Alg 6.2 (Q-learning), Alg 6.3 (TD(lambda)),
TD(0) prediction, Exercise 6.1 (random walk TD(0) vs MC comparison).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_06.solution import (
    CliffWalkingEnv,
    MaximizationBiasMDP,
    RandomWalkEnv,
    _walk_transitions,
    batch_mc_values,
    batch_td_fixed_point,
    double_q_learning,
    expected_sarsa_control,
    maximization_bias_experiment,
    mc_prediction_incremental,
    q_learning,
    q_learning_control,
    random_walk_true_values,
    run_batch_updating,
    run_random_walk_comparison,
    sarsa,
    sarsa_control,
    td0_prediction,
    td_lambda_prediction,
)
from rl_envs.windy_gridworld import WindyGridWorld


class TestRandomWalkEnv:
    """Tests for the random walk environment (Example 6.2). Book Sec 6.2."""

    def test_reset_returns_c(self) -> None:
        env = RandomWalkEnv(seed=0)
        assert env.reset() == "C"

    def test_step_before_reset_raises(self) -> None:
        env = RandomWalkEnv(seed=0)
        with pytest.raises(RuntimeError):
            env.step("left")

    def test_invalid_action_raises(self) -> None:
        env = RandomWalkEnv(seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step("up")

    def test_episode_terminates_at_ends(self) -> None:
        env = RandomWalkEnv(seed=0)
        env.reset()
        for _ in range(10):
            reward, obs, done = env.step("right")
            if done:
                assert obs == "right_end"
                assert reward == pytest.approx(1.0)
                break

    def test_left_end_reward(self) -> None:
        env = RandomWalkEnv(seed=0)
        env.reset()
        for _ in range(10):
            reward, obs, done = env.step("left")
            if done:
                assert obs == "left_end"
                assert reward == pytest.approx(0.0)
                break

    def test_true_values(self) -> None:
        expected = {"A": 1 / 6, "B": 2 / 6, "C": 3 / 6, "D": 4 / 6, "E": 5 / 6}
        assert random_walk_true_values() == pytest.approx(expected)


class TestTDPrediction:
    """Tests for TD(0) prediction (Algorithm 6.1). Book Sec 6.1."""

    def test_converges_toward_true_values(self) -> None:
        env = RandomWalkEnv(seed=0)
        values = td0_prediction(env, n_episodes=5000, seed=0)
        true_values = random_walk_true_values()
        errs = [abs(values[s] - true_values[s]) for s in true_values]
        assert max(errs) < 0.08

    def test_returns_all_states(self) -> None:
        env = RandomWalkEnv(seed=0)
        values = td0_prediction(env, n_episodes=100, seed=0)
        assert set(values) == set(RandomWalkEnv.STATES)

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            td0_prediction(RandomWalkEnv(seed=0), alpha=0.0)

    def test_reproducible_with_seed(self) -> None:
        v1 = td0_prediction(RandomWalkEnv(seed=42), n_episodes=50, seed=42)
        v2 = td0_prediction(RandomWalkEnv(seed=42), n_episodes=50, seed=42)
        assert v1 == pytest.approx(v2)

    def test_td0_beats_untrained(self) -> None:
        env = RandomWalkEnv(seed=0)
        values = td0_prediction(env, n_episodes=5000, seed=0)
        true_values = random_walk_true_values()
        initial = 0.0
        err_initial = max(abs(initial - true_values[s]) for s in true_values)
        err_trained = max(abs(values[s] - true_values[s]) for s in true_values)
        assert err_trained < err_initial


class TestTDLambda:
    """Tests for TD(lambda) prediction (Algorithm 6.3). Book Sec 6.3-style."""

    def test_lambda_zero_matches_td0(self) -> None:
        env = RandomWalkEnv(seed=0)
        lam_zero = td_lambda_prediction(env, lam=0.0, n_episodes=500, seed=0)
        td0 = td0_prediction(RandomWalkEnv(seed=0), n_episodes=500, seed=0)
        # lam=0 degenerates to TD(0) math; both runs consume the same stream.
        for s in RandomWalkEnv.STATES:
            assert lam_zero[s] == pytest.approx(td0[s], abs=1e-6)

    def test_high_lambda_close_to_mc(self) -> None:
        env = RandomWalkEnv(seed=0)
        values = td_lambda_prediction(env, lam=0.9, n_episodes=5000, seed=0)
        true_values = random_walk_true_values()
        errs = [abs(values[s] - true_values[s]) for s in true_values]
        assert max(errs) < 0.12

    def test_lam_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            td_lambda_prediction(RandomWalkEnv(seed=0), lam=1.5)


class TestSarsa:
    """Tests for Sarsa (Algorithm 6.1). Book Sec 6.4."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, q_table = sarsa(env, n_episodes=100, epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa(env, n_episodes=10, epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * 2

    def test_learns_right_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa(env, n_episodes=5000, epsilon=0.1, seed=0)
        # reward is 1 at the right end, so 'right' dominates at state E
        assert q_table[("E", "right")] > q_table[("E", "left")]

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            sarsa(RandomWalkEnv(seed=0), epsilon=0.0)


class TestQLearning:
    """Tests for Q-learning (Algorithm 6.2). Book Sec 6.5."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, q_table = q_learning(env, n_episodes=100, epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_learns_right_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = q_learning(env, n_episodes=5000, epsilon=0.1, seed=0)
        assert q_table[("E", "right")] > q_table[("E", "left")]

    def test_converges(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = q_learning(env, n_episodes=5000, epsilon=0.1, seed=0)
        assert all(np.isfinite(v) for v in q_table.values())


class TestExercise61:
    """Exercise 6.1: TD(0) vs Monte Carlo on the random walk. Book Sec 6.2."""

    def test_comparison_structure(self) -> None:
        result = run_random_walk_comparison(repeats=5, episodes=10, seed=0)
        assert set(result) == {"td0", "mc"}
        assert len(result["td0"]) == 10
        assert len(result["mc"]) == 10

    def test_td0_learned_faster_than_mc(self) -> None:
        result = run_random_walk_comparison(repeats=30, episodes=100, alpha=0.1, seed=0)
        early_td0 = np.mean(result["td0"][:20])
        early_mc = np.mean(result["mc"][:20])
        assert early_td0 < early_mc

    def test_td0_error_decreases(self) -> None:
        result = run_random_walk_comparison(repeats=30, episodes=100, alpha=0.1, seed=0)
        assert np.mean(result["td0"][-10:]) < np.mean(result["td0"][:10])

    def test_mc_incremental_estimation(self) -> None:
        env = RandomWalkEnv(seed=0)
        values = mc_prediction_incremental(env, n_episodes=5000, seed=0)
        true_values = random_walk_true_values()
        errs = [abs(values[s] - true_values[s]) for s in true_values]
        assert max(errs) < 0.08


class TestBatchUpdating:
    """Batch updating (Example 6.3, Figure 6.2). Book Sec. 6.3."""

    def test_fixed_point_residual(self) -> None:
        rng = np.random.default_rng(0)
        walks = [_walk_transitions(rng) for _ in range(30)]
        fixed = batch_td_fixed_point(walks)
        for state in RandomWalkEnv.STATES:
            total = 0.0
            count = 0
            for trajectory in walks:
                for st, reward, next_state, done in trajectory:
                    if st == state:
                        total += reward + (0.0 if done else fixed[next_state])
                        count += 1
            if count:
                assert fixed[state] * count - total == pytest.approx(0.0, abs=1e-8)

    def test_mc_values_manual_check(self) -> None:
        batch = [
            [
                ("A", 0.0, "B", False),
                ("B", 1.0, "right_end", True),
            ],
            [
                ("A", 0.0, "left_end", True),
            ],
        ]
        values = batch_mc_values(batch)
        assert values["A"] == pytest.approx(0.5)
        assert values["B"] == pytest.approx(1.0)
        assert values["C"] == pytest.approx(0.5)  # unvisited keeps the init value

    def test_empty_batch_keeps_initial_values(self) -> None:
        assert batch_td_fixed_point([]) == pytest.approx({s: 0.5 for s in RandomWalkEnv.STATES})
        assert batch_mc_values([]) == pytest.approx({s: 0.5 for s in RandomWalkEnv.STATES})

    def test_td_beats_mc_on_average(self) -> None:
        result = run_batch_updating(repeats=4, n_walks=40, seed=0)
        assert set(result) == {"walks", "td", "mc"}
        assert len(result["walks"]) == 40
        assert result["td"].mean() < result["mc"].mean()


class TestCliffWalking:
    """Cliff walking (Example 6.6, Figures 6.4 and 6.6). Book Sec. 6.5-6.6."""

    def test_reset_at_start(self) -> None:
        env = CliffWalkingEnv()
        assert env.reset() == (3, 0)

    def test_step_before_reset_raises(self) -> None:
        with pytest.raises(RuntimeError):
            CliffWalkingEnv().step("up")

    def test_safe_step_reward_minus_one(self) -> None:
        env = CliffWalkingEnv()
        env.reset()
        reward, obs, done = env.step("up")
        assert (reward, obs, done) == (-1.0, (2, 0), False)

    def test_falling_into_cliff_sends_back_to_start(self) -> None:
        env = CliffWalkingEnv()
        env.state = (2, 1)
        reward, obs, done = env.step("down")
        assert reward == -100.0
        assert obs == (3, 0)
        assert not done

    def test_start_to_cliff_cell_is_teleported(self) -> None:
        env = CliffWalkingEnv()
        env.reset()
        reward, obs, done = env.step("right")  # (3,1) is a cliff cell
        assert reward == -100.0
        assert obs == (3, 0)

    def test_reaching_goal_terminates(self) -> None:
        env = CliffWalkingEnv()
        env.state = (2, 11)
        reward, obs, done = env.step("down")
        assert done
        assert obs == (3, 11)
        assert reward == -1.0

    def test_boundary_clipped(self) -> None:
        env = CliffWalkingEnv()
        env.state = (0, 0)
        _, obs, _ = env.step("up")
        assert obs == (0, 0)

    def test_n_states_excludes_cliff(self) -> None:
        assert CliffWalkingEnv().n_states == 38


class TestControl:
    """Generic indexed-action control family (Sec. 6.4-6.6)."""

    def test_sarsa_smoke(self) -> None:
        _, steps, returns = sarsa_control(CliffWalkingEnv(), n_episodes=60, seed=0)
        assert len(steps) == len(returns) == 60
        assert all(np.isfinite(r) for r in returns)
        assert all(s >= 1 for s in steps)

    def test_q_learning_smoke(self) -> None:
        _, steps, returns = q_learning_control(CliffWalkingEnv(), n_episodes=60, seed=0)
        assert len(steps) == len(returns) == 60
        assert all(np.isfinite(r) for r in returns)

    def test_expected_sarsa_smoke(self) -> None:
        _, steps, returns = expected_sarsa_control(CliffWalkingEnv(), n_episodes=60, seed=0)
        assert len(steps) == len(returns) == 60
        assert all(np.isfinite(r) for r in returns)

    def test_windy_smoke(self) -> None:
        _, steps, _ = sarsa_control(WindyGridWorld(), n_episodes=20, seed=0)
        assert len(steps) == 20
        assert all(s >= 1 for s in steps)

    def test_max_steps_truncation_cap(self) -> None:
        _, steps, _ = sarsa_control(WindyGridWorld(), n_episodes=10, seed=0, max_steps=5)
        assert all(s <= 5 for s in steps)

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            sarsa_control(CliffWalkingEnv(), n_episodes=1, alpha=0.0)

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            q_learning_control(CliffWalkingEnv(), n_episodes=1, epsilon=0.0)


class TestMaximizationBias:
    """Maximization-bias toy MDP (Example 6.7, Sec. 6.7, Fig. 6.7)."""

    def test_reset_at_a(self) -> None:
        assert MaximizationBiasMDP(seed=0).reset() == "A"

    def test_left_goes_to_b(self) -> None:
        env = MaximizationBiasMDP(seed=0)
        env.reset()
        reward, obs, done = env.step("left")
        assert (reward, obs, done) == (0.0, "B", False)

    def test_b_actions_terminate_with_reward(self) -> None:
        env = MaximizationBiasMDP(seed=0)
        env.reset()
        env.step("left")
        reward, obs, done = env.step(3)
        assert done
        assert obs == "T"
        assert np.isfinite(reward)

    def test_right_is_safe_terminal(self) -> None:
        env = MaximizationBiasMDP(seed=0)
        env.reset()
        reward, obs, done = env.step("right")
        assert (reward, obs, done) == (0.0, "T", True)

    def test_actions_for_states(self) -> None:
        env = MaximizationBiasMDP()
        assert env.actions_for("A") == ("left", "right")
        assert env.actions_for("B") == tuple(range(10))

    def test_double_q_learning_runs(self) -> None:
        env = MaximizationBiasMDP(seed=0)
        q1, q2, first_actions = double_q_learning(env, n_episodes=60, seed=0)
        assert len(first_actions) == 60
        assert set(first_actions) <= {0, 1}
        assert all(np.isfinite(v) for v in q1.values())
        assert all(np.isfinite(v) for v in q2.values())

    def test_experiment_output_shapes(self) -> None:
        result = maximization_bias_experiment(n_episodes=30, repeats=2, seed=0)
        assert list(result["episodes"]) == list(range(1, 31))
        for key in ("q_learning_left", "double_q_learning_left"):
            values = result[key]
            assert len(values) == 30
            assert np.all(values >= 0.0) and np.all(values <= 100.0)


class TestWindy:
    """Windy gridworld (Example 6.5; Exercises 6.9-6.10). Book Sec. 6.4."""

    def test_standard_action_count(self) -> None:
        assert len(WindyGridWorld().ACTIONS) == 4

    def test_king_moves_eight_actions(self) -> None:
        assert len(WindyGridWorld(king_moves=True).ACTIONS) == 8

    def test_king_moves_with_stay_nine_actions(self) -> None:
        assert len(WindyGridWorld(king_moves=True, allow_stay=True).ACTIONS) == 9

    def test_stochastic_wind_moves_within_plus_minus_one(self) -> None:
        env = WindyGridWorld(king_moves=True, allow_stay=True, stochastic_wind=True, seed=0)
        rows = set()
        for _ in range(300):
            env.state = (3, 6)  # column wind 2, so effective wind in {1,2,3}
            _, obs, _ = env.step("down")  # +1 row minus wind
            rows.add(obs[0])
        assert rows <= {1, 2, 3}
        assert len(rows) == 3  # all three wind outcomes sampled

    def test_stochastic_wind_zero_column_stays_zero(self) -> None:
        env = WindyGridWorld(stochastic_wind=True, seed=0)
        env.state = (3, 0)  # column wind 0: stochastic branch never applies
        _, obs, _ = env.step("down")
        assert obs == (4, 0)
