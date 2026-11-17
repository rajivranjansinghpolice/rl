"""Solution implementation for Chapter 4 - Dynamic Programming.

Reference: Sutton & Barto (2nd Ed), Chapter 4

Implements: Alg 4.1 (iterative policy evaluation), Alg 4.2 (policy improvement),
Alg 4.3 (policy iteration), Alg 4.4 (value iteration),
Exercise 4.1 (gambler's problem — Example 4.3).

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
    (Exercise 4.9 remark). All rewards are 0 — the +1 outcome of reaching the
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

    ax2.bar(states[1:goal], policy[1:goal], color="#444444", width=1.0)
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
