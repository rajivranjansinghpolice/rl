"""
Tests for Chapter 4: Dynamic Programming.

Reference: Sutton & Barto (2nd Ed), Chapter 4.

Tests for: Alg 4.1 (iterative policy evaluation), Alg 4.2 (policy improvement),
Alg 4.3 (policy iteration), Alg 4.4 (value iteration), Exercise 4.1
(gambler's problem, Example 4.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_03.solution import FiniteMDP, solve_bellman_expectation, verify_bellman
from chapter_04.solution import (
    JacksCarRentalMDP,
    gamblers_problem,
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)


def make_lineworld(gamma: float = 0.9) -> FiniteMDP:
    """A -> B -> terminal. 'go' yields +1 into B, +2 into terminal."""
    transitions = {
        ("A", "go"): [(1.0, 1.0, "B")],
        ("B", "go"): [(1.0, 2.0, "T")],
    }
    return FiniteMDP(
        ["A", "B", "T"], ["go"], transitions, gamma=gamma, terminal_states=["T"]
    )


def make_two_action_mdp() -> FiniteMDP:
    """S -> terminal. 'good' yields +2, 'bad' yields 0."""
    transitions = {
        ("S", "good"): [(1.0, 2.0, "T")],
        ("S", "bad"): [(1.0, 0.0, "T")],
    }
    return FiniteMDP(
        ["S", "T"], ["good", "bad"], transitions, gamma=0.9, terminal_states=["T"]
    )


def make_corridor_mdp(gamma: float = 0.9) -> FiniteMDP:
    """A --advance--> B --advance--> T(r=+1). 'stay' loops r=0."""
    transitions = {
        ("A", "stay"): [(1.0, 0.0, "A")],
        ("A", "advance"): [(1.0, 0.0, "B")],
        ("B", "stay"): [(1.0, 0.0, "B")],
        ("B", "advance"): [(1.0, 1.0, "T")],
    }
    return FiniteMDP(
        ["A", "B", "T"],
        ["stay", "advance"],
        transitions,
        gamma=gamma,
        terminal_states=["T"],
    )


def uniform_policy(mdp: FiniteMDP) -> np.ndarray:
    policy = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
    return policy


class TestPolicyEvaluation:
    """Tests for iterative policy evaluation (Algorithm 4.1). Book Sec 4.1."""

    def test_lineworld_converges_to_exact(self) -> None:
        mdp = make_lineworld(gamma=0.9)
        values = policy_evaluation(mdp, uniform_policy(mdp))
        assert values[0] == pytest.approx(2.8, abs=1e-6)
        assert values[1] == pytest.approx(2.0, abs=1e-6)
        assert values[2] == pytest.approx(0.0, abs=1e-6)

    def test_jacobi_matches_inplace(self) -> None:
        mdp = make_lineworld(gamma=0.9)
        assert np.allclose(
            policy_evaluation(mdp, uniform_policy(mdp), inplace=True),
            policy_evaluation(mdp, uniform_policy(mdp), inplace=False),
            atol=1e-8,
        )

    def test_matches_exact_solve(self) -> None:
        mdp = make_two_action_mdp()
        policy = uniform_policy(mdp)
        values = policy_evaluation(mdp, policy)
        assert np.allclose(values, solve_bellman_expectation(mdp, policy), atol=1e-6)

    def test_terminal_values_are_zero(self) -> None:
        mdp = make_lineworld()
        values = policy_evaluation(mdp, uniform_policy(mdp))
        assert values[2] == pytest.approx(0.0)

    def test_rejects_bad_theta(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(ValueError):
            policy_evaluation(mdp, uniform_policy(mdp), theta=0.0)

    def test_rejects_wrong_policy_shape(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(ValueError):
            policy_evaluation(mdp, np.ones((2, 2)))

    def test_policy_rows_must_sum_to_one(self) -> None:
        mdp = make_two_action_mdp()
        bad_policy = np.array([[0.4, 0.4], [0.0, 0.0]])
        with pytest.raises(ValueError):
            policy_evaluation(mdp, bad_policy)

    def test_partial_policy_evaluation_moves_values(self) -> None:
        mdp = make_lineworld(gamma=0.9)
        values = np.zeros(mdp.n_states)
        one_sweep = policy_evaluation(mdp, uniform_policy(mdp), theta=np.inf, max_iter=1)
        assert one_sweep[0] > values[0]


class TestPolicyImprovement:
    """Tests for policy improvement (Algorithm 4.2). Book Sec 4.2."""

    def test_greedy_policy_picks_best_action(self) -> None:
        mdp = make_two_action_mdp()
        values = np.array([1.0, 0.0])
        policy = policy_improvement(mdp, values)
        assert policy[0, 0] == pytest.approx(1.0)  # 'good' wins

    def test_ties_split_uniformly(self) -> None:
        mdp = FiniteMDP(
            ["S", "T"],
            ["a", "b"],
            {("S", "a"): [(1.0, 1.0, "T")], ("S", "b"): [(1.0, 1.0, "T")]},
            gamma=0.9,
            terminal_states=["T"],
        )
        values = np.zeros(mdp.n_states)
        policy = policy_improvement(mdp, values)
        assert policy[0] == pytest.approx(np.array([0.5, 0.5]))

    def test_lineworld_single_action(self) -> None:
        mdp = make_lineworld()
        policy = policy_improvement(mdp, np.array([2.0, 1.0, 0.0]))
        assert policy[:, 0] == pytest.approx(np.ones(3))

    def test_values_shape_validated(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(ValueError):
            policy_improvement(mdp, np.zeros(10))


class TestPolicyIteration:
    """Tests for policy iteration (Algorithm 4.3). Book Sec 4.3."""

    def test_finds_optimal_policy_corridor(self) -> None:
        mdp = make_corridor_mdp(gamma=0.9)
        values, policy = policy_iteration(mdp, theta=1e-10)
        assert values[0] == pytest.approx(0.9, abs=1e-6)
        assert values[1] == pytest.approx(1.0, abs=1e-6)
        assert np.argmax(policy[0]) == mdp.actions.index("advance")
        assert np.argmax(policy[1]) == mdp.actions.index("advance")

    def test_agrees_with_value_iteration(self) -> None:
        mdp = make_corridor_mdp(gamma=0.9)
        values_pi, _ = policy_iteration(mdp, theta=1e-10)
        values_vi, _ = value_iteration(mdp, theta=1e-10)
        assert np.allclose(values_pi, values_vi, atol=1e-6)

    def test_improves_over_uniform_start(self) -> None:
        mdp = make_two_action_mdp()
        values, policy = policy_iteration(mdp, theta=1e-10)
        assert values[0] == pytest.approx(2.0, abs=1e-6)
        assert np.argmax(policy[0]) == mdp.actions.index("good")

    def test_verify_bellman_after_convergence(self) -> None:
        mdp = make_corridor_mdp(gamma=0.9)
        _, policy = policy_iteration(mdp, theta=1e-10)
        assert verify_bellman(mdp, policy_probs=policy) is True


class TestValueIteration:
    """Tests for value iteration (Algorithm 4.4). Book Sec 4.4."""

    def test_lineworld_values(self) -> None:
        mdp = make_lineworld(gamma=0.9)
        values, _ = value_iteration(mdp, theta=1e-10)
        assert values[0] == pytest.approx(2.8, abs=1e-6)
        assert values[1] == pytest.approx(2.0, abs=1e-6)
        assert values[2] == pytest.approx(0.0, abs=1e-6)

    def test_greedy_policy_returned(self) -> None:
        mdp = make_corridor_mdp(gamma=0.9)
        _, policy = value_iteration(mdp, theta=1e-10)
        assert np.argmax(policy[0]) == mdp.actions.index("advance")
        assert np.argmax(policy[1]) == mdp.actions.index("advance")

    def test_rejects_bad_theta(self) -> None:
        mdp = make_lineworld()
        with pytest.raises(ValueError):
            value_iteration(mdp, theta=-1e-8)

    def test_terminal_values_stay_zero(self) -> None:
        mdp = make_lineworld()
        values, _ = value_iteration(mdp, theta=1e-10)
        assert values[2] == pytest.approx(0.0)


def _bellman_optimality_residual(mdp: FiniteMDP, values: np.ndarray) -> float:
    normalized = np.where(
        np.isin(np.arange(mdp.n_states), [mdp.state_index[s] for s in mdp.terminal_states]),
        0.0,
        values,
    )
    residual = 0.0
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        best = max(
            mdp.expected_reward(s, a)
            + mdp.gamma
            * sum(
                p * normalized[mdp.state_index[ns]]
                for p, _, ns in mdp.transition_probs(s, a)
            )
            for a in mdp.actions
        )
        residual = max(residual, abs(normalized[i] - best))
    return residual


class TestGamblersProblem:
    """Tests for gambler's problem (Exercise 4.1 / Example 4.3). Book Sec 4.3."""

    def test_terminal_boundaries(self) -> None:
        values, _ = gamblers_problem(p_h=0.4, theta=1e-8)
        assert values[0] == pytest.approx(0.0)
        assert values[100] == pytest.approx(1.0)

    def test_values_are_probabilities(self) -> None:
        values, _ = gamblers_problem(p_h=0.4, theta=1e-8)
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_policy_action_bounds(self) -> None:
        goal = 100
        _, policy = gamblers_problem(p_h=0.4, goal=goal, theta=1e-8)
        for s in range(1, goal):
            assert 1 <= policy[s] <= min(s, goal - s)

    def test_solution_satisfies_bellman_optimality(self) -> None:
        goal = 50
        p_h = 0.4
        values, _ = gamblers_problem(p_h=p_h, goal=goal, gamma=1.0, theta=1e-10)
        residual = 0.0
        for s in range(1, goal):
            best = -np.inf
            for stake in range(1, min(s, goal - s) + 1):
                win_state, lose_state = s + stake, s - stake
                q = p_h * values[win_state] + (1.0 - p_h) * values[lose_state]
                best = max(best, q)
            residual = max(residual, abs(values[s] - best))
        assert residual < 1e-8

    def test_solution_converges_tight(self) -> None:
        values, _ = gamblers_problem(p_h=0.4, theta=1e-12)
        assert np.all(np.isfinite(values))

    def test_policy_stakes_valid_for_p_h_025(self) -> None:
        goal = 100
        _, policy = gamblers_problem(p_h=0.25, goal=goal, theta=1e-8)
        for s in range(1, goal):
            assert 1 <= policy[s] <= min(s, goal - s)

    def test_policy_stakes_valid_for_p_h_055(self) -> None:
        goal = 100
        _, policy = gamblers_problem(p_h=0.55, goal=goal, theta=1e-8)
        for s in range(1, goal):
            assert 1 <= policy[s] <= min(s, goal - s)

# Jack's car rental (Example 4.2, Sec. 4.3) is expensive to build (~30 s),
# so tests share lazily-built MDP instances.
_JCR_INSTANCES: dict = {}


def _jcr_mdp(free_first_shuttle: bool = False, second_parking_fee=None):
    key = (free_first_shuttle, second_parking_fee)
    if key not in _JCR_INSTANCES:
        _JCR_INSTANCES[key] = JacksCarRentalMDP(
            free_first_shuttle=free_first_shuttle,
            second_parking_fee=second_parking_fee,
        )
    return _JCR_INSTANCES[key]


class TestJacksCarRental:
    """Tests for Jack's car rental (Example 4.2 / Exercise 4.5). Book Sec 4.3."""

    def test_location_outcomes_sum_to_one(self) -> None:
        from chapter_04.solution import _location_outcomes, _poisson_pmf

        req1, req2 = _poisson_pmf(3.0), _poisson_pmf(4.0)
        ret1, ret2 = _poisson_pmf(3.0), _poisson_pmf(2.0)
        for x in (0, 1, 5, 20):
            assert sum(p for _, _, p in _location_outcomes(x, req1, ret1)) == pytest.approx(1.0, abs=1e-9)
            assert sum(p for _, _, p in _location_outcomes(x, req2, ret2)) == pytest.approx(1.0, abs=1e-9)

    def test_expected_reward_zero_at_empty_state(self) -> None:
        mdp = _jcr_mdp()
        # (i=0, j=0): no cars can be rented or moved yet; returns bring no money.
        assert np.allclose(mdp.expected_reward[:, 0], 0.0, atol=1e-10)

    def test_transition_probabilities_row_sums_one(self) -> None:
        mdp = _jcr_mdp()
        assert np.allclose(mdp.P.sum(axis=2), 1.0, atol=1e-9)

    def test_move_and_cost_basic(self) -> None:
        mdp = _jcr_mdp()
        x1, x2, moved, cost = mdp._move_and_cost(20, 0, 5)
        assert (x1, x2, moved, cost) == (15, 5, 5, 10.0)
        # Opposite direction: no cars at loc 2.
        assert mdp._move_and_cost(20, 0, -5)[2] == 0
        # Can never move more cars than the source holds.
        assert mdp._move_and_cost(3, 0, 5)[2] == 3

    def test_free_first_shuttle_discounts_first_car_1_to_2(self) -> None:
        base = _jcr_mdp()
        mod = _jcr_mdp(free_first_shuttle=True)
        n_base = base._move_and_cost(20, 0, 3)[3]
        n_mod = mod._move_and_cost(20, 0, 3)[3]
        assert n_base - n_mod == pytest.approx(2.0)  # one car shuttled free
        # Opposite direction gets no discount.
        assert mod._move_and_cost(0, 20, -3)[3] == base._move_and_cost(0, 20, -3)[3]
        # A single car 1 -> 2 moved for free.
        assert mod._move_and_cost(10, 0, 1)[3] == 0.0

    def test_second_parking_fee_applies_per_overfull_location(self) -> None:
        mod = _jcr_mdp(second_parking_fee=4.0)
        # After the overnight move, one or both locations exceed 10 cars.
        assert mod._move_and_cost(20, 20, 0)[3] == pytest.approx(8.0)   # two lots
        assert mod._move_and_cost(11, 5, 0)[3] == pytest.approx(4.0)    # loc 1 only
        assert mod._move_and_cost(10, 10, 0)[3] == pytest.approx(0.0)   # not > 10
        # Move itself can push both locations over the threshold.
        assert mod._move_and_cost(12, 12, 1)[3] == pytest.approx(2.0 + 4.0 + 4.0)

    def test_move_into_full_location_disappears(self) -> None:
        mdp = _jcr_mdp()
        # Cars moved into a full location disappear (book rule, Sec. 4.3).
        x1, x2, moved, cost = mdp._move_and_cost(20, 20, 5)
        assert (x1, x2) == (15, 20)

    def test_policy_iteration_converges_and_is_directional(self) -> None:
        mdp = _jcr_mdp()
        v, policy, history = mdp.policy_iteration()
        assert 2 <= len(history) <= 8
        assert np.array_equal(history[0][0], np.zeros((21, 21)))
        assert policy.min() >= -5 and policy.max() <= 5
        # Full location 1, no cars at location 2: ship max cars to location 2.
        assert policy[20, 0] == 5
        # No cars at location 1, full at location 2: pull cars back (or 0), never send more.
        assert policy[0, 20] <= 0
        assert v[0] > 0.0
        for _, v_grid in history:
            assert np.all(np.isfinite(v_grid))

    def test_policy_iteration_improves_policy_on_first_round(self) -> None:
        mdp = _jcr_mdp()
        _, _, history = mdp.policy_iteration()
        # pi_0 (zero move) must not be identical to the greedy policy pi_1.
        assert not np.array_equal(history[0][0], history[1][0])

    def test_modifications_still_yield_valid_optimal_policy(self) -> None:
        # Exercise 4.5 (Programming): re-solve with free first shuttle and
        # parking fee; the result must be a valid net-move policy with finite values.
        mdp = _jcr_mdp(free_first_shuttle=True, second_parking_fee=4.0)
        v, policy, history = mdp.policy_iteration(max_rounds=20)
        assert policy.min() >= -5 and policy.max() <= 5
        assert np.all(np.isfinite(v))
        assert len(history) >= 2

    def test_policy_indices_roundtrip(self) -> None:
        mdp = _jcr_mdp()
        policy = np.array([-5, -1, 0, 1, 5], dtype=np.int64)
        assert list(mdp.policy_indices(policy)) == [0, 4, 5, 6, 10]
