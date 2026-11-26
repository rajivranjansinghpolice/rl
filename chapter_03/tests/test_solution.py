"""
Tests for Chapter 3 Finite MDPs.

Reference: Sutton & Barto (2nd Ed), Chapter 3.

Tests for: Alg 3.1 (MDP definition and environment), Alg 3.2 (return /
reward-to-go), Alg 3.3 (softmax policy), Alg 3.4 (Bellman equations).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_03.solution import (
    FiniteMDP,
    SoftmaxPolicy,
    bellman_backup_expectation,
    bellman_backup_optimality,
    bellman_expectation_residual,
    bellman_optimality_residual,
    compute_return,
    compute_returns_to_go,
    example_3_5_gridworld,
    example_3_5_gridworld_random_policy_values,
    recycling_robot,
    solve_bellman_expectation,
    verify_bellman,
)


def make_lineworld(gamma: float = 0.9, seed: int | None = None) -> FiniteMDP:
    """A -> B -> (terminal). 'go' gives +1 into B, +2 into terminal."""
    transitions = {
        ("A", "go"): [(1.0, 1.0, "B")],
        ("B", "go"): [(1.0, 2.0, "T")],
    }
    return FiniteMDP(
        ["A", "B", "T"], ["go"], transitions, gamma=gamma, terminal_states=["T"], seed=seed
    )


def make_two_action_mdp() -> FiniteMDP:
    """S -> terminal. 'good' yields +2, 'bad' yields 0."""
    transitions = {
        ("S", "good"): [(1.0, 2.0, "T")],
        ("S", "bad"): [(1.0, 0.0, "T")],
    }
    return FiniteMDP(["S", "T"], ["good", "bad"], transitions, gamma=0.9, terminal_states=["T"])


def make_self_loop_mdp(gamma: float = 0.9, seed: int | None = None) -> FiniteMDP:
    """S --'chance'--> 50% r=2 -> T, 50% r=0 -> S (self loop)."""
    transitions = {
        ("S", "chance"): [(0.5, 2.0, "T"), (0.5, 0.0, "S")],
    }
    return FiniteMDP(
        ["S", "T"], ["chance"], transitions, gamma=gamma, terminal_states=["T"], seed=seed
    )


def single_action_policy(mdp: FiniteMDP) -> np.ndarray:
    return np.ones((mdp.n_states, mdp.n_actions))


class TestFiniteMDP:
    """Tests for FiniteMDP (Algorithm 3.1). Book Sec 3.1-3.2."""

    def test_init_basic(self) -> None:
        mdp = make_lineworld()
        assert mdp.n_states == 3
        assert mdp.n_actions == 1
        assert mdp.gamma == pytest.approx(0.9)

    def test_empty_states_raises(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP([], ["a"], {})

    def test_empty_actions_raises(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s"], [], {})

    def test_duplicate_states_raise(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s", "s"], ["a"], {})

    def test_duplicate_actions_raise(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s"], ["a", "a"], {})

    def test_gamma_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            make_lineworld(gamma=1.5)

    def test_unknown_terminal_state_raises(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s"], ["a"], {("s", "a"): [(1.0, 0.0, "s")]}, terminal_states=["nope"])

    def test_missing_transition_raises(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s"], ["a"], {}, terminal_states=[])

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s", "x"], ["a"], {("s", "a"): [(0.5, 0.0, "s")]})

    def test_negative_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s"], ["a"], {("s", "a"): [(-0.5, 0.0, "s"), (1.5, 0.0, "s")]})

    def test_unknown_next_state_raises(self) -> None:
        with pytest.raises(ValueError):
            FiniteMDP(["s"], ["a"], {("s", "a"): [(1.0, 0.0, "nope")]})

    def test_transition_probs(self) -> None:
        mdp = make_lineworld()
        outcomes = mdp.transition_probs("A", "go")
        assert outcomes == [(1.0, 1.0, "B")]

    def test_transition_probs_terminal_raises(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(RuntimeError):
            mdp.transition_probs("T", "go")

    def test_expected_reward(self) -> None:
        mdp = make_lineworld()
        assert mdp.expected_reward("A", "go") == pytest.approx(1.0)

    def test_is_terminal(self) -> None:
        mdp = make_lineworld()
        assert mdp.is_terminal("T")
        assert not mdp.is_terminal("A")

    def test_reset_and_step_episode(self) -> None:
        mdp = make_lineworld(seed=0)
        mdp.reset("A")
        reward, state, done = mdp.step("go")
        assert (reward, state, done) == (1.0, "B", False)
        reward, state, done = mdp.step("go")
        assert (reward, state, done) == (2.0, "T", True)

    def test_step_after_done_raises(self) -> None:
        mdp = make_lineworld(seed=0)
        mdp.reset("B")
        _, _, done = mdp.step("go")
        assert done
        with pytest.raises(RuntimeError):
            mdp.step("go")

    def test_step_before_reset_raises(self) -> None:
        mdp = make_lineworld(seed=0)
        with pytest.raises(RuntimeError):
            mdp.step("go")

    def test_step_reproducible_with_seed(self) -> None:
        mdp1 = make_self_loop_mdp(seed=42)
        mdp2 = make_self_loop_mdp(seed=42)
        mdp1.reset("S")
        mdp2.reset("S")
        for _ in range(5):
            if mdp1.done:
                mdp1.reset("S")
                mdp2.reset("S")
            assert mdp1.step("chance") == mdp2.step("chance")

    def test_recycling_robot_constructs(self) -> None:
        mdp = recycling_robot(seed=0)
        assert mdp.n_states == 2
        assert set(mdp.actions) == {"search", "wait", "recharge"}
        for outcome in mdp.transition_probs("low", "search"):
            assert outcome[0] > 0.0


class TestComputeReturn:
    """Tests for compute_return / compute_returns_to_go (Algorithm 3.2). Book Sec 3.3."""

    def test_empty(self) -> None:
        assert compute_return([]) == pytest.approx(0.0)

    def test_gamma_zero(self) -> None:
        assert compute_return([1.0, 2.0, 3.0], gamma=0.0) == pytest.approx(1.0)

    def test_gamma_one(self) -> None:
        assert compute_return([1.0, 2.0, 3.0], gamma=1.0) == pytest.approx(6.0)

    def test_discounted(self) -> None:
        assert compute_return([1.0, 1.0, 1.0], gamma=0.5) == pytest.approx(1.75)

    def test_gamma_must_be_valid(self) -> None:
        with pytest.raises(ValueError):
            compute_return([1.0], gamma=2.0)

    def test_returns_to_go(self) -> None:
        assert compute_returns_to_go([1.0, 2.0, 3.0], gamma=1.0) == pytest.approx([6.0, 5.0, 3.0])

    def test_returns_to_go_discounted(self) -> None:
        assert compute_returns_to_go([1.0, 2.0, 3.0], gamma=0.5) == pytest.approx([2.75, 3.5, 3.0])

    def test_returns_to_go_empty(self) -> None:
        assert compute_returns_to_go([], gamma=0.9) == []

    def test_returns_to_go_gamma_must_be_valid(self) -> None:
        with pytest.raises(ValueError):
            compute_returns_to_go([1.0], gamma=-0.1)


class TestSoftmaxPolicy:
    """Tests for SoftmaxPolicy (Algorithm 3.3). Book Sec 3.4."""

    def test_init_zero_preferences_uniform(self) -> None:
        policy = SoftmaxPolicy(n_states=2, n_actions=3)
        probs = policy.action_probs(0)
        assert probs == pytest.approx(np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_probs_sum_to_one(self) -> None:
        policy = SoftmaxPolicy(2, 4, seed=0)
        for s in range(2):
            assert policy.action_probs(s).sum() == pytest.approx(1.0)

    def test_init_invalid_dims_raise(self) -> None:
        with pytest.raises(ValueError):
            SoftmaxPolicy(0, 3)

    def test_preferences_shape_validated(self) -> None:
        with pytest.raises(ValueError):
            SoftmaxPolicy(2, 2, preferences=np.zeros((3, 3)))

    def test_greedy_action_is_argmax(self) -> None:
        policy = SoftmaxPolicy(1, 3, preferences=np.array([[0.0, 5.0, 1.0]]))
        assert policy.greedy_action(0) == 1

    def test_sample_action_in_range(self) -> None:
        policy = SoftmaxPolicy(1, 5, seed=0)
        for _ in range(20):
            assert 0 <= policy.sample_action(0) < 5

    def test_higher_preference_higher_probability(self) -> None:
        policy = SoftmaxPolicy(1, 2, preferences=np.array([[3.0, 0.0]]))
        probs = policy.action_probs(0)
        assert probs[0] > probs[1]

    def test_preferences_returns_copy(self) -> None:
        policy = SoftmaxPolicy(1, 2, preferences=np.array([[1.0, 2.0]]))
        copy = policy.preferences(0)
        copy[0] = 999.0
        assert policy.preferences(0)[0] == pytest.approx(1.0)

    def test_numerical_stability_extreme_logits(self) -> None:
        policy = SoftmaxPolicy(1, 2, preferences=np.array([[1000.0, 0.0]]))
        probs = policy.action_probs(0)
        assert np.isfinite(probs).all()
        assert probs[0] == pytest.approx(1.0)

    def test_update_increases_taken_action_probability(self) -> None:
        policy = SoftmaxPolicy(1, 2, seed=0)
        before = policy.action_probs(0)[0]
        policy.update(0, action=0, advantage=1.0, step_size=0.1)
        assert policy.action_probs(0)[0] > before

    def test_update_negative_advantage_decreases_probability(self) -> None:
        policy = SoftmaxPolicy(1, 2, seed=0)
        before = policy.action_probs(0)[0]
        policy.update(0, action=0, advantage=-1.0, step_size=0.1)
        assert policy.action_probs(0)[0] < before

    def test_update_zero_advantage_leaves_preferences_unchanged(self) -> None:
        policy = SoftmaxPolicy(1, 3, preferences=np.array([[0.5, 1.0, 1.5]]))
        before = policy.preferences(0)
        policy.update(0, action=1, advantage=0.0, step_size=0.5)
        assert policy.preferences(0) == pytest.approx(before)

    def test_update_step_size_must_be_non_negative(self) -> None:
        policy = SoftmaxPolicy(1, 2)
        with pytest.raises(ValueError):
            policy.update(0, action=0, advantage=1.0, step_size=-0.1)

    def test_update_action_out_of_range_raises(self) -> None:
        policy = SoftmaxPolicy(1, 2)
        with pytest.raises(ValueError):
            policy.update(0, action=5, advantage=1.0, step_size=0.1)


class TestBellman:
    """Tests for Bellman backup/solve/verify (Algorithm 3.4). Book Sec 3.5."""

    def test_backup_expectation_lineworld(self) -> None:
        mdp = make_lineworld()
        values = np.zeros(mdp.n_states)
        new_values = bellman_backup_expectation(mdp, single_action_policy(mdp), values)
        assert new_values[0] == pytest.approx(1.0)
        assert new_values[1] == pytest.approx(2.0)
        assert new_values[2] == pytest.approx(0.0)

    def test_backup_optimality_lineworld(self) -> None:
        mdp = make_lineworld()
        values = np.zeros(mdp.n_states)
        new_values = bellman_backup_optimality(mdp, values)
        assert new_values[0] == pytest.approx(1.0)
        assert new_values[1] == pytest.approx(2.0)
        assert new_values[2] == pytest.approx(0.0)

    def test_backup_optimality_converges(self) -> None:
        mdp = make_lineworld()
        values = np.zeros(mdp.n_states)
        for _ in range(100):
            values = bellman_backup_optimality(mdp, values)
        assert values[0] == pytest.approx(2.8, abs=1e-6)
        assert values[1] == pytest.approx(2.0, abs=1e-6)
        assert values[2] == pytest.approx(0.0, abs=1e-6)

    def test_solve_bellman_expectation_lineworld(self) -> None:
        mdp = make_lineworld()
        values = solve_bellman_expectation(mdp, single_action_policy(mdp))
        assert values[0] == pytest.approx(2.8, abs=1e-8)
        assert values[1] == pytest.approx(2.0, abs=1e-8)
        assert values[2] == pytest.approx(0.0, abs=1e-8)

    def test_solve_bellman_expectation_mixed_policy(self) -> None:
        mdp = make_two_action_mdp()
        policy = np.array([[0.5, 0.5], [0.0, 0.0]])
        values = solve_bellman_expectation(mdp, policy)
        assert values[0] == pytest.approx(1.0, abs=1e-8)

    def test_solve_bellman_expectation_self_loop(self) -> None:
        mdp = make_self_loop_mdp(gamma=0.9)
        policy = np.array([[1.0], [0.0]])
        values = solve_bellman_expectation(mdp, policy)
        assert values[0] == pytest.approx(1.0 / 0.55, abs=1e-8)

    def test_solve_validates_policy_shape(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(ValueError):
            solve_bellman_expectation(mdp, np.ones((2, 2)))

    def test_solve_validates_policy_rows_sum_to_one(self) -> None:
        mdp = make_two_action_mdp()
        policy = np.array([[0.3, 0.3], [0.0, 0.0]])
        with pytest.raises(ValueError):
            solve_bellman_expectation(mdp, policy)

    def test_values_shape_validated(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(ValueError):
            bellman_backup_optimality(mdp, np.zeros(2))

    def test_expectation_residual_zero_on_solution(self) -> None:
        mdp = make_lineworld()
        policy = single_action_policy(mdp)
        values = solve_bellman_expectation(mdp, policy)
        assert bellman_expectation_residual(mdp, policy, values) == pytest.approx(0.0, abs=1e-8)

    def test_optimality_residual_zero_on_fixed_point(self) -> None:
        mdp = make_lineworld()
        values = np.zeros(mdp.n_states)
        for _ in range(200):
            values = bellman_backup_optimality(mdp, values)
        assert bellman_optimality_residual(mdp, values) == pytest.approx(0.0, abs=1e-8)

    def test_expectation_residual_positive_for_arbitrary_values(self) -> None:
        mdp = make_lineworld()
        residual = bellman_expectation_residual(
            mdp, single_action_policy(mdp), np.zeros(mdp.n_states)
        )
        assert residual > 0.0

    def test_verify_bellman_expectation(self) -> None:
        mdp = make_two_action_mdp()
        policy = np.array([[0.5, 0.5], [0.0, 0.0]])
        assert verify_bellman(mdp, policy_probs=policy) is True

    def test_verify_bellman_optimality(self) -> None:
        mdp = make_lineworld()
        assert verify_bellman(mdp) is True

    def test_verify_bellman_false_for_bad_values(self) -> None:
        mdp = make_lineworld()
        assert verify_bellman(mdp, values=np.full(mdp.n_states, 123.0)) is False

    def test_terminal_values_forced_to_zero(self) -> None:
        mdp = make_lineworld()
        values = bellman_backup_optimality(mdp, np.full(mdp.n_states, 100.0))
        assert values[2] == pytest.approx(0.0)


class TestExample35Gridworld:
    """Example 3.5 gridworld (Section 3.5-3.6) reproduction."""

    def test_dimensions(self) -> None:
        mdp = example_3_5_gridworld(gamma=0.9)
        assert mdp.n_states == 25
        assert mdp.n_actions == 4
        assert mdp.gamma == 0.9
        assert mdp.terminal_states == set()

    def test_special_state_a_transition(self) -> None:
        mdp = example_3_5_gridworld(gamma=0.9)
        for action in mdp.actions:
            assert mdp.transition_probs((0, 1), action) == [(1.0, 10.0, (4, 1))]

    def test_special_state_b_transition(self) -> None:
        mdp = example_3_5_gridworld(gamma=0.9)
        for action in mdp.actions:
            assert mdp.transition_probs((0, 3), action) == [(1.0, 5.0, (2, 3))]

    def test_off_grid_move_stays_and_penalizes(self) -> None:
        mdp = example_3_5_gridworld(gamma=0.9)
        assert mdp.transition_probs((0, 0), "up") == [(1.0, -1.0, (0, 0))]

    def test_normal_cell_deterministic_move(self) -> None:
        mdp = example_3_5_gridworld(gamma=0.9)
        assert mdp.transition_probs((1, 1), "right") == [(1.0, 0.0, (1, 2))]

    def test_random_policy_a_value(self) -> None:
        v = example_3_5_gridworld_random_policy_values(gamma=0.9)
        assert v[0 * 5 + 1] == pytest.approx(8.789, abs=1e-3)  # book ~8.8

    def test_optimal_a_value(self) -> None:
        from chapter_03.solution import _optimal_values_and_policy
        mdp = example_3_5_gridworld(gamma=0.9)
        values, _ = _optimal_values_and_policy(mdp)
        assert values[0 * 5 + 1] == pytest.approx(24.4194, abs=1e-3)  # book: 24.4

    def test_optimal_policy_bellman_optimality_residual(self) -> None:
        import chapter_03.solution as sol
        mdp = example_3_5_gridworld(gamma=0.9)
        values, _ = sol._optimal_values_and_policy(mdp)
        assert sol.bellman_optimality_residual(mdp, values) < 1e-8

    def test_plot_example_3_5_gridworld_runs(self) -> None:
        import chapter_03.solution as sol
        out = sol.plot_example_3_5_gridworld(save_path=None, show=False)
        assert out is not None

