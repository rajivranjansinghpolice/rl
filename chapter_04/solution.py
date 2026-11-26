"""Solution implementation for Chapter 4 - Dynamic Programming.

Reference: Sutton & Barto (2nd Ed), Chapter 4

Implements: Alg 4.1 (iterative policy evaluation), Alg 4.2 (policy improvement),
Alg 4.3 (policy iteration), Alg 4.4 (value iteration),
Exercise 4.1 (gambler's problem â€” Example 4.3).

Reuses the FiniteMDP dynamics model from chapter_03 (package `chapter_03`).
"""

from __future__ import annotations

import numpy as np

from chapter_03.solution import FiniteMDP


def _check_mdp(mdp: FiniteMDP) -> None:
    if not isinstance(mdp, FiniteMDP):
        raise TypeError("expected chapter_03.solution.FiniteMDP")


def _policy_probabilities(mdp: FiniteMDP, policy_probs: np.ndarray) -> np.ndarray:
    policy_probs = np.asarray(policy_probs, dtype=np.float64)
    expected = (mdp.n_states, mdp.n_actions)
    if policy_probs.shape != expected:
        raise ValueError(f"policy_probs must have shape {expected}")
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        row_sum = policy_probs[i].sum()
        if not np.isclose(row_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"policy row for state {s} sums to {row_sum}, expected 1.0"
            )
    return policy_probs


def policy_evaluation(
    mdp: FiniteMDP,
    policy_probs: np.ndarray,
    theta: float = 1e-8,
    max_iter: int = 10_000,
    inplace: bool = True,
) -> np.ndarray:
    """Iterative policy evaluation (Algorithm 4.1). Book Section 4.1.

    Estimates v_pi using repeated application of the expected Bellman
    expectation backup until the maximum value change in a sweep is < theta.

    Args:
        mdp: The finite MDP.
        policy_probs: Policy matrix of shape (n_states, n_actions).
        theta: Convergence threshold (small positive number).
        max_iter: Maximum number of sweeps.
        inplace: If True, use in-place (Gauss-Seidel) updates; otherwise use
            two arrays (Jacobi), matching the book's synchronous variant.

    Returns:
        Estimated state-value vector (terminal states always 0).
    """
    _check_mdp(mdp)
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    policy_probs = _policy_probabilities(mdp, policy_probs)
    values = np.zeros(mdp.n_states, dtype=np.float64)

    for _ in range(max_iter):
        delta = 0.0
        if inplace:
            new_values = values
        else:
            new_values = values.copy()
        for i, s in enumerate(mdp.states):
            if s in mdp.terminal_states:
                continue
            total = 0.0
            for j, a in enumerate(mdp.actions):
                weight = policy_probs[i, j]
                if weight == 0.0:
                    continue
                outcomes = mdp.transition_probs(s, a)
                expected = sum(p * r for p, r, _ in outcomes)
                backed = sum(
                    p * values[mdp.state_index[ns]] for p, _, ns in outcomes
                )
                total += weight * (expected + mdp.gamma * backed)
            old = new_values[i]
            delta = max(delta, abs(old - total))
            new_values[i] = total
        values = new_values
        if delta < theta:
            break
    return values


def policy_improvement(mdp: FiniteMDP, values: np.ndarray) -> np.ndarray:
    """Policy improvement (Algorithm 4.2). Book Section 4.2.

    Returns the greedy policy pi'(s) = argmax_a sum_{s',r} p(s',r|s,a)
    [r + gamma * v(s')]. Probability mass is apportioned uniformly among the
    maximizing actions; sub-maximal actions get zero probability.

    Args:
        mdp: The finite MDP.
        values: State-value vector (terminal values are forced to 0).

    Returns:
        Policy matrix of shape (n_states, n_actions).
    """
    _check_mdp(mdp)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mdp.n_states,):
        raise ValueError(f"values must have shape ({mdp.n_states},)")
    normalized = values.copy()
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            normalized[i] = 0.0

    policy = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            policy[i] = 1.0 / mdp.n_actions
            continue
        action_values = np.array(
            [
                mdp.expected_reward(s, a)
                + mdp.gamma * sum(
                    p * normalized[mdp.state_index[ns]]
                    for p, _, ns in mdp.transition_probs(s, a)
                )
                for a in mdp.actions
            ]
        )
        best = action_values.max()
        maximizing = np.flatnonzero(np.isclose(action_values, best))
        policy[i, maximizing] = 1.0 / len(maximizing)
    return policy


def policy_iteration(
    mdp: FiniteMDP,
    theta: float = 1e-8,
    max_iter: int = 1_000,
    inplace: bool = True,
    initial_policy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Policy iteration (Algorithm 4.3). Book Section 4.3.

    Alternates full policy evaluation (Algorithm 4.1) and policy improvement
    (Algorithm 4.2) until the policy is stable (greedy with respect to its
    own value function).

    Args:
        mdp: The finite MDP.
        theta: Convergence threshold for the evaluation step.
        max_iter: Maximum number of improvement sweeps.
        inplace: Passed through to policy_evaluation.
        initial_policy: Optional starting policy matrix; defaults to uniform.

    Returns:
        (values, policy_probs) approximating v* and pi*.
    """
    _check_mdp(mdp)
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if initial_policy is None:
        policy = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
        for i, s in enumerate(mdp.states):
            if s in mdp.terminal_states:
                policy[i] = 1.0 / mdp.n_actions
    else:
        policy = _policy_probabilities(mdp, initial_policy)

    for _ in range(max_iter):
        values = policy_evaluation(mdp, policy, theta=theta, inplace=inplace)
        old_action_sets = [
            np.flatnonzero(np.isclose(policy[i], policy[i].max()))
            for i in range(mdp.n_states)
        ]
        new_policy = policy_improvement(mdp, values)
        new_action_sets = [
            np.flatnonzero(np.isclose(new_policy[i], new_policy[i].max()))
            for i in range(mdp.n_states)
        ]
        if all(
            np.array_equal(old, new)
            for old, new in zip(old_action_sets, new_action_sets)
        ):
            policy = new_policy
            break
        policy = new_policy
    else:
        values = policy_evaluation(mdp, policy, theta=theta, inplace=inplace)
    return values, policy


def value_iteration(
    mdp: FiniteMDP,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration (Algorithm 4.4). Book Section 4.4.

    Combines in each sweep one expected (max) backup per state until the
    maximum value change in a sweep is < theta. Returns the greedy policy
    with respect to the final values.

    Args:
        mdp: The finite MDP.
        theta: Convergence threshold (small positive number).
        max_iter: Maximum number of sweeps.

    Returns:
        (values, policy_probs) approximating v* and pi*.
    """
    _check_mdp(mdp)
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    values = np.zeros(mdp.n_states, dtype=np.float64)
    for _ in range(max_iter):
        delta = 0.0
        for i, s in enumerate(mdp.states):
            if s in mdp.terminal_states:
                continue
            old = values[i]
            best = -np.inf
            for a in mdp.actions:
                outcomes = mdp.transition_probs(s, a)
                q = sum(
                    p * (r + mdp.gamma * values[mdp.state_index[ns]])
                    for p, r, ns in outcomes
                )
                best = max(best, q)
            values[i] = best
            delta = max(delta, abs(old - best))
        if delta < theta:
            break
    policy = policy_improvement(mdp, values)
    return values, policy


def gamblers_problem(
    p_h: float = 0.4,
    goal: int = 100,
    theta: float = 1e-8,
    gamma: float = 1.0,
    max_iter: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Gambler's problem (Example 4.3 / Exercise 4.1). Book Section 4.3.

    Undiscounted, episodic, finite MDP. States are capital 0..goal; 0 and
    `goal` are dummy terminals, giving them values 0 and 1 respectively
    (Exercise 4.9 remark). All rewards are 0 â€” the +1 outcome of reaching the
    goal is represented entirely by the terminal value 1 at `goal`.
    From s in 1..goal-1, actions are integer stakes 1..min(s, goal-s);
    winning a stake transitions to s + stake, losing to s - stake.

    Args:
        p_h: Probability of heads (0.4 in the book; 0.25 and 0.55 in Ex 4.9).
        goal: Capital goal that ends the episode (100 in the book).
        theta: Value-iteration convergence threshold.
        gamma: Discount factor (1.0 for undiscounted episodic tasks).
        max_iter: Maximum number of value-iteration sweeps.

    Returns:
        (values, policy): values[i] is the probability of winning from capital
        i; policy[i] is the integer stake staked at capital i (0 elsewhere).

    Tie-breaking: among optimal stakes, the SMALLEST stake wins ties. This
    reproduces the book's Figure 4.3 staircase/pyramid shape with stakes of
    12, 25, 50 and 25 at capitals 12/13, 25, 50 and 75. The optimal policy
    is highly non-unique (Exercise 4.9 remarks on this), so do not expect
    pixel-identity with any printed edition of the book's figure.
    """
    if not 0.0 <= p_h <= 1.0:
        raise ValueError("p_h must be in [0, 1]")
    if goal < 2:
        raise ValueError("goal must be at least 2")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if theta <= 0.0:
        raise ValueError("theta must be positive")

    values = np.zeros(goal + 1, dtype=np.float64)
    values[goal] = 1.0

    for _ in range(max_iter):
        delta = 0.0
        for s in range(1, goal):
            best = -np.inf
            for stake in range(1, min(s, goal - s) + 1):
                win_state = s + stake
                lose_state = s - stake
                q = p_h * gamma * values[win_state] + (1.0 - p_h) * gamma * values[
                    lose_state
                ]
                best = max(best, q)
            delta = max(delta, abs(values[s] - best))
            values[s] = best
        if delta < theta:
            break

    policy = np.zeros(goal + 1, dtype=int)
    for s in range(1, goal):
        best_stake = 0
        best_value = -np.inf
        for stake in range(1, min(s, goal - s) + 1):
            win_state = s + stake
            lose_state = s - stake
            q = p_h * gamma * values[win_state] + (1.0 - p_h) * gamma * values[lose_state]
            if q > best_value:
                best_value = q
                best_stake = stake
        policy[s] = best_stake
    return values, policy


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 4)
# ---------------------------------------------------------------------------

def plot_gamblers_problem(
    save_path: str | None = None,
    show: bool = False,
    p_h: float = 0.4,
    goal: int = 100,
):
    """Gambler's problem value function and policy (Example 4.3, Fig. 4.3).

    Sutton & Barto (2nd ed.), Example 4.3 / Sec. 4.4, Fig. 4.3: the gambler's
    problem solved by value iteration. States are capital 0..goal (goal=100 in
    book), actions are stakes; heads probability p_h (book shows p_h=0.4).
    Reward +1 only on reaching the goal (modeled via terminal values 0 and 1).
    Top panel: estimated value v*(s) = probability of reaching goal from s.
    Bottom panel: optimal stake (policy). The non-unique optimal-stake
    structure reflects multiple optimal actions at some states.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values, policy = gamblers_problem(p_h=p_h, goal=goal)
    states = np.arange(goal + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax1.plot(states, values, color="#1f77b4")
    ax1.set_ylabel("Value\n(prob. of winning)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(
        f"S&B (2nd ed.), Example 4.3, Fig. 4.3: Gambler's problem (p_h={p_h})"
    )

    ax2.bar(states[1:goal], policy[1:goal], color="#444444", width=0.8)
    ax2.set_xlabel("Capital")
    ax2.set_ylabel("Final policy\n(stake)")
    ax2.set_xlim(0, goal)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Jack's car rental (Sutton & Barto, 2nd ed., Example 4.2, Sec. 4.3)
# ---------------------------------------------------------------------------

_JCR_MAX_CARS = 20
_JCR_MAX_MOVE = 5
_JCR_GAMMA = 0.9
_JCR_RENTAL_REVENUE = 10.0
_JCR_MOVE_COST_PER_CAR = 2.0
_JCR_N = _JCR_MAX_CARS + 1


def _poisson_pmf(lam: float, max_k: int = _JCR_MAX_CARS) -> np.ndarray:
    """Exact Poisson pmf for k = 0..max_k (recurrence, stable)."""
    pmf = np.empty(max_k + 1, dtype=np.float64)
    pmf[0] = np.exp(-lam)
    for k in range(1, max_k + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def _capped_buckets(exact: np.ndarray, capped_bucket: int, eps: float = 1e-11) -> list[tuple[int, float]]:
    """Probability buckets for an unbounded Poisson X capped at ``capped_bucket``.

    Returns list of (value, prob) pairs with value in 0..capped_bucket. The
    capped bucket accumulates P(X >= capped_bucket); buckets whose mass is
    below ``eps`` are folded in as well, so probabilities sum to 1 exactly.
    """
    out = []
    for k in range(capped_bucket):
        p = exact[k]
        if p >= eps:
            out.append((k, float(p)))
    tail = 1.0 - sum(p for _, p in out)
    out.append((capped_bucket, tail))
    return out


def _location_outcomes(
    x: int, req_exact: np.ndarray, ret_exact: np.ndarray
) -> list[tuple[int, int, float]]:
    """Exact outcome buckets (end_cars, rented, prob) for one location.

    ``x`` is the number of cars on the lot at the start of the day (after the
    overnight move). Requests are Poisson (``req_exact``); only cars present
    can be rented. Returns are Poisson (``ret_exact``); cars beyond 20
    disappear. Probabilities sum to 1 exactly.
    """
    outcomes: dict[tuple[int, int], float] = {}
    for rented, p_req in _capped_buckets(req_exact, x):
        after = x - rented
        space = _JCR_MAX_CARS - after
        for m, p_ret in _capped_buckets(ret_exact, space):
            end = after + m
            key = (end, rented)
            outcomes[key] = outcomes.get(key, 0.0) + p_req * p_ret
    return [(end, rented, p) for (end, rented), p in outcomes.items()]


class JacksCarRentalMDP:
    """Example 4.2 Jack's car rental (2nd ed., Sec. 4.3) as an MDP.

    State: (i, j), i cars at location 1 and j cars at location 2,
    0 <= i, j <= _JCR_MAX_CARS (20).
    Action: integer a in [-5, +5]; a > 0 moves a cars from location 1 to
    location 2 overnight, a < 0 moves -a cars from location 2 to location 1.
    A location never supplies more cars than it holds; if fewer cars than
    requested are present, as many as possible are moved.
    Dynamics (one day-night cycle): first the overnight move ($2 per car
    actually moved), then customers arrive and rent (Poisson lambda = 3 at
    location 1, lambda = 4 at location 2; $10 revenue per car rented), then
    cars are returned (Poisson lambda = 3 at location 1, lambda = 2 at
    location 2). Cars beyond 20 at a location disappear (returned to the
    national company). gamma = 0.9. Continuing task, as in the book.

    Optional Exercise 4.5-style modifications (off by default, matching the
    base Example 4.2 dynamics):

    * ``free_first_shuttle``: the first car moved from location 1 to
      location 2 each night rides free; additional cars in that direction
      and all cars in the opposite direction still cost $2 each.
    * ``second_parking_fee``: if more than ``parking_threshold`` cars (10 in
      the exercise) are kept overnight at a location (measured after the
      overnight move), an additional flat cost of this amount ($4 in the
      exercise) is incurred per such location, independent of how many cars
      are kept there. ``None`` disables the fee.

    Policy vectors and the (21, 21) grid views use the same layout:
    ``policy[s]`` is the net number of cars moved (negative = 2 -> 1).
    """

    # Class-level cache of the (Poisson dynamics) transition builders.
    # The transition matrices P and the expected-rentals matrix depend only
    # on the Poisson lambdas; cost/revenue/move modifications only affect the
    # per-instance expected-reward vector. Sharing the heavy build keeps the
    # Exercise 4.5 modifications cheap.
    _DYNAMICS_CACHE: dict = {}

    def __init__(
        self,
        free_first_shuttle: bool = False,
        second_parking_fee: float | None = None,
        parking_threshold: int = 10,
        gamma: float = _JCR_GAMMA,
        move_cost: float = _JCR_MOVE_COST_PER_CAR,
        rental_revenue: float = _JCR_RENTAL_REVENUE,
        lambda_req1: float = 3.0,
        lambda_req2: float = 4.0,
        lambda_ret1: float = 3.0,
        lambda_ret2: float = 2.0,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if move_cost < 0.0 or rental_revenue < 0.0:
            raise ValueError("move_cost and rental_revenue must be non-negative")
        if parking_threshold < 0:
            raise ValueError("parking_threshold must be non-negative")
        self.gamma = float(gamma)
        self.move_cost = float(move_cost)
        self.rental_revenue = float(rental_revenue)
        self.free_first_shuttle = bool(free_first_shuttle)
        self.second_parking_fee = (
            None if second_parking_fee is None else float(second_parking_fee)
        )
        self.parking_threshold = int(parking_threshold)
        req = (lambda_req1, lambda_req2)
        ret = (lambda_ret1, lambda_ret2)
        key = (tuple(req), tuple(ret))
        dynamics = JacksCarRentalMDP._DYNAMICS_CACHE.get(key)
        if dynamics is None:
            dynamics = self._compute_dynamics(key)
            JacksCarRentalMDP._DYNAMICS_CACHE[key] = dynamics
        self.P, rentals = dynamics
        self.n_states = _JCR_N * _JCR_N
        self.n_actions = 2 * _JCR_MAX_MOVE + 1
        # Per-instance expected reward (depends on cost/revenue model only,
        # not on the Poisson dynamics): r(s,a) = revenue * E[rented] - cost.
        self.expected_reward = np.empty(
            (self.n_actions, self.n_states), dtype=np.float64
        )
        for s in range(self.n_states):
            i, j = divmod(s, _JCR_N)
            for ai, a in enumerate(range(-_JCR_MAX_MOVE, _JCR_MAX_MOVE + 1)):
                _, _, _, cost = self._move_and_cost(i, j, a)
                self.expected_reward[ai, s] = self.rental_revenue * rentals[ai, s] - cost

    @staticmethod
    def _compute_dynamics(
        key: tuple[tuple[float, float], tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        req = (_poisson_pmf(key[0][0]), _poisson_pmf(key[0][1]))
        ret = (_poisson_pmf(key[1][0]), _poisson_pmf(key[1][1]))
        loc = {}
        for x in range(_JCR_N):
            loc[(1, x)] = _location_outcomes(x, req[0], ret[0])
            loc[(2, x)] = _location_outcomes(x, req[1], ret[1])
        n_states = _JCR_N * _JCR_N
        n_actions = 2 * _JCR_MAX_MOVE + 1
        transition = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
        rentals = np.zeros((n_actions, n_states), dtype=np.float64)
        # A temporary MDP-like object is not needed: moves are lambda-free,
        # only lambdas were keyed. Reuse the base move rules via a bare
        # instance of the cost model.
        cost_mdp = JacksCarRentalMDP.__new__(JacksCarRentalMDP)
        cost_mdp.gamma = 0.0
        cost_mdp.move_cost = 0.0
        cost_mdp.rental_revenue = 0.0
        cost_mdp.free_first_shuttle = False
        cost_mdp.second_parking_fee = None
        cost_mdp.parking_threshold = _JCR_MAX_CARS + 1  # never triggers
        for s in range(n_states):
            i, j = divmod(s, _JCR_N)
            for ai, a in enumerate(range(-_JCR_MAX_MOVE, _JCR_MAX_MOVE + 1)):
                x1, x2, _, _ = cost_mdp._move_and_cost(i, j, a)
                joint: dict[int, float] = {}
                expected_rentals = 0.0
                for e1, r1, p1 in loc[(1, x1)]:
                    ns_base = e1 * _JCR_N
                    for e2, r2, p2 in loc[(2, x2)]:
                        p = p1 * p2
                        ns = ns_base + e2
                        joint[ns] = joint.get(ns, 0.0) + p
                        expected_rentals += p * (r1 + r2)
                for ns, p in joint.items():
                    transition[ai, s, ns] = p
                rentals[ai, s] = expected_rentals
        return transition, rentals

    def _move_and_cost(self, i: int, j: int, a: int) -> tuple[int, int, int, float]:
        """Overnight move outcome: (x1, x2, n_moved, dollar_cost).

        The number of cars moved is limited only by the source location
        (at most its current stock). As with returns, cars arriving beyond
        20 at the destination disappear (returned to the national company).
        """
        if a > 0:
            moved = min(a, i)
            signed = moved
        elif a < 0:
            moved = min(-a, j)
            signed = -moved
        else:
            moved = 0
            signed = 0
        x1 = min(_JCR_MAX_CARS, max(0, i - signed))
        x2 = min(_JCR_MAX_CARS, max(0, j + signed))
        cost = self.move_cost * moved
        if self.free_first_shuttle and a > 0 and moved >= 1:
            cost -= self.move_cost  # first car 1 -> 2 shuttled for free
        if self.second_parking_fee is not None:
            for held in (x1, x2):
                if held > self.parking_threshold:
                    cost += self.second_parking_fee
        return x1, x2, moved, cost

    def q_values(self, values: np.ndarray) -> np.ndarray:
        """Action-value table q(a, s) (shape (n_actions, n_states))."""
        values = np.asarray(values, dtype=np.float64).reshape(self.n_states)
        return self.expected_reward + self.gamma * np.einsum(
            "asn,n->as", self.P, values
        )

    def policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        """Solve v_pi exactly: v = r_pi + gamma P_pi v (441 x 441 solve)."""
        policy = np.asarray(policy, dtype=np.int64).reshape(self.n_states)
        if policy.min() < -_JCR_MAX_MOVE or policy.max() > _JCR_MAX_MOVE:
            raise ValueError(f"policy entries must lie in [{-_JCR_MAX_MOVE}, {_JCR_MAX_MOVE}]")
        idx = self.policy_indices(policy)
        p_pi = self.P[idx, np.arange(self.n_states)]
        r_pi = self.expected_reward[idx, np.arange(self.n_states)]
        return np.linalg.solve(np.eye(self.n_states) - self.gamma * p_pi, r_pi)

    @staticmethod
    def policy_indices(policy: np.ndarray) -> np.ndarray:
        """Map net-move policy values to action indices into P/expected_reward."""
        return np.asarray(policy, dtype=np.int64).reshape(-1) + _JCR_MAX_MOVE

    def policy_iteration(self, max_rounds: int = 100):
        """Policy iteration from the zero-move policy pi_0.

        Returns (values, policy_grid, history): ``values`` is the 441-vector
        of the final value function, ``policy_grid`` the (21, 21) int array
        of optimal net car moves, and ``history`` the list of
        ``(policy_grid, value_grid)`` snapshots after each evaluation round
        (pi_0 ... pi_N), matching the layout of book Figure 4.2.
        """
        policy = np.zeros(self.n_states, dtype=np.int64)
        history = []
        for _ in range(max_rounds):
            v = self.policy_evaluation(policy)
            history.append(
                (
                    policy.reshape(_JCR_N, _JCR_N).copy(),
                    v.reshape(_JCR_N, _JCR_N).copy(),
                )
            )
            q = self.q_values(v)
            new_policy = np.argmax(q, axis=0).astype(np.int64) - _JCR_MAX_MOVE
            if np.array_equal(new_policy, policy):
                return v, policy.reshape(_JCR_N, _JCR_N), history
            policy = new_policy
        v = self.policy_evaluation(policy)
        history.append(
            (
                policy.reshape(_JCR_N, _JCR_N).copy(),
                v.reshape(_JCR_N, _JCR_N).copy(),
            )
        )
        return v, policy.reshape(_JCR_N, _JCR_N), history


def plot_jacks_car_rental(
    save_path: str | None = None,
    show: bool = False,
    free_first_shuttle: bool = False,
    second_parking_fee: float | None = None,
):
    """Policy sequence and value function (Example 4.2, Sec. 4.3, Fig. 4.2).

    Sutton & Barto (2nd ed.), Example 4.2: the policy improvement steps of
    Jack's car rental solved by policy iteration, starting from the
    zero-move policy. Top row: policy grids pi_0 ... pi_N (-5 = five cars
    2 -> 1, +5 = five cars 1 -> 2). Bottom row: the value function of the
    final policy as a heatmap and as a 3D surface.

    Figure identity note: in the 2nd (2018) edition this is Figure 4.2
    ("Jack's car rental"); the in-progress 2nd-edition draft and the 1st
    edition number it differently. This reproduction is an illustrative
    demonstration of Example 4.2, not a pixel-identical redraw of any
    edition's print figure.

    Args:
        save_path: if set, save the figure to this path.
        show: if True, call plt.show().
        free_first_shuttle: pass-through to JacksCarRentalMDP (Ex 4.5 mod).
        second_parking_fee: pass-through to JacksCarRentalMDP (Ex 4.5 mod);
            None disables the fee.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mdp = JacksCarRentalMDP(
        free_first_shuttle=free_first_shuttle,
        second_parking_fee=second_parking_fee,
    )
    v, policy, history = mdp.policy_iteration()

    n = len(history)
    ncols = max(3, n + 1)
    fig = plt.figure(figsize=(4.2 * ncols, 8.6))

    for idx, (pi_grid, v_grid) in enumerate(history):
        ax = fig.add_subplot(2, ncols, idx + 1)
        ax.imshow(
            pi_grid,
            origin="lower",
            cmap="RdBu",
            vmin=-_JCR_MAX_MOVE,
            vmax=_JCR_MAX_MOVE,
        )
        ax.set_title(rf"$\pi_{idx}$ (moves)")
        ax.set_xlabel("j (cars at loc 2)")
        if idx == 0:
            ax.set_ylabel("i (cars at loc 1)")
        else:
            ax.set_yticks([])
    # Hide unused first-row slots.
    for k in range(n, ncols):
        ax = fig.add_subplot(2, ncols, k + 1)
        ax.axis("off")
    # Value heatmap panel.
    ax_val = fig.add_subplot(2, ncols, ncols + 1)
    im2 = ax_val.imshow(v.reshape(_JCR_N, _JCR_N), origin="lower", cmap="viridis")
    ax_val.set_title("v* heatmap")
    ax_val.set_xlabel("j (cars at loc 2)")
    ax_val.set_ylabel("i (cars at loc 1)")
    fig.colorbar(im2, ax=ax_val, shrink=0.8)
    # Value surface panel.
    ax_surf = fig.add_subplot(2, ncols, ncols + 2, projection="3d")
    xs = np.arange(_JCR_N)
    ys = np.arange(_JCR_N)
    xs, ys = np.meshgrid(xs, ys, indexing="ij")
    ax_surf.plot_surface(xs, ys, v.reshape(_JCR_N, _JCR_N), cmap="viridis", alpha=0.95)
    ax_surf.set_xlabel("i (loc 1)")
    ax_surf.set_ylabel("j (loc 2)")
    ax_surf.set_zlabel("v*")
    ax_surf.set_title("v* surface")
    fig.suptitle(
        "S&B (2nd ed.), Example 4.2 / Fig. 4.2: Jack's car rental "
        "(policy improvement; illustrative demonstration)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.45)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, history


