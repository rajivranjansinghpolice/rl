"""
Tests for Chapter 17: Frontiers.

Reference: Sutton & Barto (2nd Ed), Chapter 17 (Frontiers).

Tests for: Alg 17.1 (options / hierarchical RL demo in a four-room
GridWorld, Section 17.2) and Alg 17.2 (intrinsic motivation via a
count-based novelty bonus, Section 17.4).

The novelty-bonus tests use the configuration the chapter documents:
gamma = 0 with optimistic initialization at q_init = beta, under which the
bonus is a clean one-step frontier signal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chapter_17.solution import (
    GridWorld,
    default_options,
    make_subgoal_option,
    novelty_bonus,
    primitive_q_learning,
    q_learning_novelty,
    rollout_primitive,
    rollout_smdp,
    run_option,
    smdp_q_learning,
)


class TestGridWorld:
    """Shared environment checks (used by both demos)."""

    def test_reset_and_step_bookkeeping(self) -> None:
        env = GridWorld()
        state = env.reset()
        assert env.cell(state) == (0, 0)
        state, reward, done = env.step(1)  # down, free cell
        assert env.cell(state) == (1, 0)
        assert reward == pytest.approx(env.step_cost)
        assert done is False

    def test_bump_into_wall_stays_in_place_penalized(self) -> None:
        env = GridWorld()
        env.reset()
        state, reward, done = env.step(0)  # up = wall at row -1
        assert env.cell(state) == (0, 0)
        assert reward == pytest.approx(env.wall_penalty)
        assert done is False

    def test_goal_is_terminal_and_rewarding(self) -> None:
        env = GridWorld()
        env.reset()
        env.state = env.state_id((6, 5))
        state, reward, done = env.step(3)  # right -> (6, 6)
        assert env.cell(state) == (6, 6)
        assert reward == pytest.approx(1.0)
        assert done is True

    def test_state_index_mapping_round_trip(self) -> None:
        env = GridWorld()
        for cell in [(0, 0), (3, 2), (6, 6)]:
            assert env.cell(env.state_id(cell)) == cell


class TestOptions:
    """Tests for Algorithm 17.1 — options / hierarchical RL (Section 17.2)."""

    def test_option_unrolls_and_terminates_at_subgoal(self) -> None:
        env = GridWorld()
        env.reset()
        goal_option = make_subgoal_option(None, (6, 6), "option_goal")
        result = run_option(env, goal_option, max_steps=50)
        assert result.done is True
        assert result.steps <= 20
        assert env.cell(env.state) == (6, 6)
        assert math.isfinite(result.discounted_return)

    def test_option_termination_is_subgoal_specific(self) -> None:
        env = GridWorld()
        env.reset()
        gap_left = make_subgoal_option(None, (3, 2), "option_gap_left")
        assert gap_left.terminates(env) is False
        env.state = env.state_id((3, 2))
        assert gap_left.terminates(env) is True

    def test_option_respects_step_cap(self) -> None:
        env = GridWorld()
        env.reset()
        goal_option = default_options()[-1]
        result = run_option(env, goal_option, max_steps=3)
        assert result.steps == 3
        assert env.cell(env.state) != (6, 6)

    def test_options_solve_in_few_macro_decisions(self) -> None:
        q_table, options = smdp_q_learning(n_episodes=300, seed=0)
        rollout = rollout_smdp(q_table, 4 + len(options), len(options), options=options)
        assert rollout["success"] is True
        assert rollout["option_decisions"] <= 3
        assert rollout["primitive_steps"] <= 20

    def test_options_agent_vs_primitive_only_agent(self) -> None:
        options_q, options = smdp_q_learning(n_episodes=300, seed=0)
        rollout = rollout_smdp(options_q, 4 + len(options), len(options), options=options)
        primitive_q = primitive_q_learning(n_episodes=300, seed=0)
        primitive = rollout_primitive(primitive_q)
        assert primitive["success"] is True
        # Same move count (both near-optimal in primitive steps), but the
        # options agent makes far fewer *decisions*.
        assert rollout["primitive_steps"] <= primitive["primitive_steps"]
        assert rollout["decisions"] < primitive["decisions"]

    def test_smdp_q_values_are_finite(self) -> None:
        q_table, options = smdp_q_learning(n_episodes=50, seed=1)
        assert len(q_table) == GridWorld().n_states * (4 + len(options))
        assert all(math.isfinite(v) for v in q_table.values())

    def test_smdp_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            smdp_q_learning(alpha=0.0)
        with pytest.raises(ValueError):
            smdp_q_learning(epsilon=0.0)
        with pytest.raises(ValueError):
            smdp_q_learning(gamma=1.5)
        with pytest.raises(ValueError):
            primitive_q_learning(alpha=0.0)


class TestNoveltyBonus:
    """Tests for Algorithm 17.2 — count-based novelty bonus (Section 17.4)."""

    def test_bonus_exact_known_answer(self) -> None:
        assert novelty_bonus(0, 0.5) == pytest.approx(0.5)
        assert novelty_bonus(3, 0.5) == pytest.approx(0.25)
        assert novelty_bonus(15, 0.5) == pytest.approx(0.125)

    def test_bonus_decreases_with_visits(self) -> None:
        seq = [novelty_bonus(n, 0.5) for n in [0, 1, 3, 10, 100]]
        assert all(a > b for a, b in zip(seq, seq[1:]))

    def test_counts_accumulate_across_run(self) -> None:
        result = q_learning_novelty(n_episodes=2, seed=0, max_steps=50)
        assert int(result["counts"].sum()) == result["steps"] + 2

    def test_unique_states_count_reporting(self) -> None:
        result = q_learning_novelty(n_episodes=2, seed=0, max_steps=50)
        assert result["unique_states"] == int(np.count_nonzero(result["counts"]))
        assert result["steps"] > 0

    def test_novelty_bonus_visits_more_distinct_states(self) -> None:
        novelty_total = sum(
            q_learning_novelty(
                n_episodes=8, beta=0.5, epsilon=0.1, alpha=1.0, seed=seed, max_steps=50,
            )["unique_states"]
            for seed in range(20)
        )
        zero_total = sum(
            q_learning_novelty(
                n_episodes=8, beta=0.0, epsilon=0.1, alpha=1.0, seed=seed, max_steps=50,
            )["unique_states"]
            for seed in range(20)
        )
        assert novelty_total > zero_total + 60  # clear margin, not a coin flip

    def test_zero_beta_collects_no_intrinsic_reward(self) -> None:
        result = q_learning_novelty(n_episodes=4, beta=0.0, seed=3, max_steps=50)
        assert result["total_intrinsic_reward"] == pytest.approx(0.0)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            novelty_bonus(-1, 0.5)
        with pytest.raises(ValueError):
            novelty_bonus(0, -0.1)
        with pytest.raises(ValueError):
            q_learning_novelty(beta=-1.0)
        with pytest.raises(ValueError):
            q_learning_novelty(alpha=0.0)
        with pytest.raises(ValueError):
            q_learning_novelty(epsilon=0.0)
