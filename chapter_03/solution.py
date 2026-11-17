"""Solution implementation for Chapter 3 - Finite MDPs.

Reference: Sutton & Barto (2nd Ed), Chapter 3

Implements: Alg 3.1 (MDP definition and environment),
Alg 3.2 (return and reward-to-go),
Alg 3.3 (softmax policy representation),
Alg 3.4 (Bellman equations, backups, and verification).
"""

from __future__ import annotations

import numpy as np


class FiniteMDP:
    """Finite Markov Decision Process (Algorithm 3.1). Book Section 3.1-3.2.

    A finite MDP is defined by the state space S, action space A, and
    dynamics p(s', r | s, a) = Pr{St+1=s', Rt+1=r | St=s, At=a}.

    Args:
        states: Non-empty sequence of hashable state labels.
        actions: Non-empty sequence of hashable action labels (shared by all states).
        transitions: Mapping from (state, action) to a list of
            (probability, reward, next_state) tuples for every non-terminal (state, action)
            pair. Probabilities per (state, action) must sum to 1 (within 1e-6).
        gamma: Discount factor in [0, 1].
        terminal_states: States in which the episode ends. Terminal states need no
            transitions and have value 0.
        seed: Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        states: list,
        actions: list,
        transitions: dict,
        gamma: float = 0.99,
        terminal_states: list | None = None,
        seed: int | None = None,
    ) -> None:
        if not states:
            raise ValueError("states must be non-empty")
        if not actions:
            raise ValueError("actions must be non-empty")
        if len(set(states)) != len(states):
            raise ValueError("states must be unique")
        if len(set(actions)) != len(actions):
            raise ValueError("actions must be unique")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

        self.states: list = list(states)
        self.actions: list = list(actions)
        self.gamma: float = float(gamma)
        self.terminal_states: set = set(terminal_states or [])
        self.state_index: dict = {s: i for i, s in enumerate(self.states)}
        self.action_index: dict = {a: i for i, a in enumerate(self.actions)}
        self.rng = np.random.default_rng(seed)

        unknown = self.terminal_states - set(self.states)
        if unknown:
            raise ValueError(f"terminal states not in states: {unknown}")

        self._validate_transitions(transitions)
        self.transitions: dict = {
            (s, a): [(float(p), float(r), ns) for p, r, ns in transitions[(s, a)]]
            for s in self.states
            for a in self.actions
            if (s, a) in transitions
        }
        self.current_state = None
        self.done: bool = False

    def _validate_transitions(self, transitions: dict) -> None:
        all_states = set(self.states)
        all_actions = set(self.actions)
        for (s, a), outcomes in transitions.items():
            if s not in all_states:
                raise ValueError(f"transition references unknown state: {s}")
            if a not in all_actions:
                raise ValueError(f"transition references unknown action: {a}")
            prob_sum = sum(p for p, _, _ in outcomes)
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                raise ValueError(f"probabilities for ({s}, {a}) sum to {prob_sum}, expected 1.0")
            for p, _, ns in outcomes:
                if p < 0.0:
                    raise ValueError(f"negative probability in transitions for ({s}, {a})")
                if ns not in all_states:
                    raise ValueError(f"transition references unknown next state: {ns}")
        for s in self.states:
            if s not in self.terminal_states:
                for a in self.actions:
                    if (s, a) not in transitions:
                        raise ValueError(f"missing transitions for non-terminal ({s}, {a})")

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def is_terminal(self, state) -> bool:
        """Return True if state is a terminal state."""
        return state in self.terminal_states

    def transition_probs(self, state, action) -> list:
        """Return p(s', r | s, a) as a list of (probability, reward, next_state).

        Book Section 3.2: the dynamics p(s', r | s, a).
        """
        if state not in self.transitions and self.is_terminal(state):
            raise RuntimeError(f"terminal state {state} has no transitions")
        if (state, action) not in self.transitions:
            raise KeyError(f"no transitions defined for ({state}, {action})")
        return list(self.transitions[(state, action)])

    def expected_reward(self, state, action) -> float:
        """Expected reward E[R_{t+1} | St=s, At=a]. Book Section 3.2."""
        return sum(p * r for p, r, _ in self.transition_probs(state, action))

    def reset(self, state=None) -> object:
        """Reset the MDP to a state. If none given, sample uniformly (seeded)."""
        if state is None:
            self.current_state = self.states[int(self.rng.integers(len(self.states)))]
        else:
            if state not in self.state_index:
                raise ValueError(f"unknown state: {state}")
            self.current_state = state
        self.done = self.is_terminal(self.current_state)
        return self.current_state

    def step(self, action) -> tuple:
        """Take action; returns (reward, next_state, done).

        Raises:
            RuntimeError: If the MDP is in a terminal state.
        """
        if self.current_state is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        outcomes = self.transition_probs(self.current_state, action)
        probs = np.array([p for p, _, _ in outcomes], dtype=np.float64)
        idx = int(self.rng.choice(len(outcomes), p=probs))
        _, reward, next_state = outcomes[idx]
        self.current_state = next_state
        self.done = self.is_terminal(next_state)
        return reward, next_state, self.done

    def _expected_next_value(self, state, action, values: np.ndarray) -> float:
        transitions = self.transition_probs(state, action)
        return sum(p * values[self.state_index[ns]] for p, _, ns in transitions)


def compute_return(rewards: list, gamma: float = 0.99) -> float:
    """Compute the discounted return G_t (Algorithm 3.2). Book Section 3.3.

    G_t = sum_{k=0}^{T-t-1} gamma^k * R_{t+k+1}  (episodic case).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * reward
        discount *= gamma
    return total


def compute_returns_to_go(rewards: list, gamma: float = 0.99) -> list:
    """Compute the reward-to-go G_t for each t (Algorithm 3.2). Book Section 3.3.

    Iterates backwards from the end of the episode:
        G_T = 0;  G_t = R_{t+1} + gamma * G_{t+1}.

    Returns a list with the same length as rewards: out[t] = G_t.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    returns = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


class SoftmaxPolicy:
    """Softmax (Gibbs/Boltzmann) action-selection policy (Algorithm 3.3). Book Section 3.4.

    pi(a|s) = exp(H(s,a)) / sum_b exp(H(s,b))

    Maintains a preference matrix H of shape (n_states, n_actions) and applies
    gradient ascent on the expected reward when updating preferences.

    Args:
        n_states: Number of states.
        n_actions: Number of actions.
        preferences: Optional initial preference matrix of shape (n_states, n_actions).
        seed: Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        preferences: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if preferences is None:
            self.H = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        else:
            preferences = np.asarray(preferences, dtype=np.float64)
            if preferences.shape != (self.n_states, self.n_actions):
                raise ValueError("preferences must have shape (n_states, n_actions)")
            self.H = preferences.copy()
        self.rng = np.random.default_rng(seed)

    def action_probs(self, state: int) -> np.ndarray:
        """Return pi(.|state) as a probability vector (numerically stable softmax)."""
        logits = self.H[state] - np.max(self.H[state])
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def greedy_action(self, state: int) -> int:
        """Return argmax_a pi(a|state) (ties broken uniformly at random)."""
        probs = self.action_probs(state)
        best = np.flatnonzero(probs == probs.max())
        return int(self.rng.choice(best))

    def sample_action(self, state: int) -> int:
        """Sample A ~ pi(.|state)."""
        probs = self.action_probs(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def preferences(self, state: int) -> np.ndarray:
        """Return a copy of the preference row H[state]."""
        return self.H[state].copy()

    def update(self, state: int, action: int, advantage: float, step_size: float) -> None:
        """Gradient-ascent update of preferences (Algorithm 3.3).

        H(s,a) <- H(s,a) + alpha * advantage * (onehot(a) - pi(.|s))

        Args:
            state: State index.
            action: Action taken (integer index).
            advantage: Signal proportional to how much better the action was than
                expected (e.g., R - V(s)). Positive reinforces, negative suppresses.
            step_size: Positive learning rate.
        """
        if step_size < 0.0:
            raise ValueError("step_size must be non-negative")
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        probs = self.action_probs(state)
        self.H[state] += step_size * advantage * (
            np.eye(self.n_actions)[action] - probs
        )


def _policy_matrix(mdp: FiniteMDP, policy_probs: np.ndarray) -> np.ndarray:
    policy_probs = np.asarray(policy_probs, dtype=np.float64)
    expected = (mdp.n_states, mdp.n_actions)
    if policy_probs.shape != expected:
        raise ValueError(f"policy_probs must have shape {expected}")
    row_sums = policy_probs.sum(axis=1)
    non_terminal_rows = [
        i for i, s in enumerate(mdp.states) if s not in mdp.terminal_states
    ]
    for i in non_terminal_rows:
        if not np.isclose(row_sums[i], 1.0, atol=1e-6):
            raise ValueError(
                f"policy row for state {mdp.states[i]} sums to {row_sums[i]}, expected 1.0"
            )
    return policy_probs


def _non_terminal_values(mdp: FiniteMDP, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mdp.n_states,):
        raise ValueError(f"values must have shape ({mdp.n_states},)")
    result = values.copy()
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            result[i] = 0.0
    return result


def bellman_backup_expectation(
    mdp: FiniteMDP, policy_probs: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """One expected Bellman backup (Algorithm 3.4, expectation equation). Book Section 3.5.

        v'(s) = sum_a pi(a|s) * sum_{s',r} p(s',r|s,a) * [r + gamma * v(s')]

    Terminal states back up to 0.
    """
    policy_probs = _policy_matrix(mdp, policy_probs)
    values = _non_terminal_values(mdp, values)
    new_values = np.zeros(mdp.n_states, dtype=np.float64)
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        total = 0.0
        for j, a in enumerate(mdp.actions):
            if policy_probs[i, j] == 0.0:
                continue
            total += policy_probs[i, j] * (
                mdp.expected_reward(s, a) + mdp.gamma * mdp._expected_next_value(s, a, values)
            )
        new_values[i] = total
    return new_values


def bellman_backup_optimality(mdp: FiniteMDP, values: np.ndarray) -> np.ndarray:
    """One optimality Bellman backup (Algorithm 3.4, optimality equation). Book Section 3.5.

        v'(s) = max_a sum_{s',r} p(s',r|s,a) * [r + gamma * v(s')]

    Terminal states back up to 0.
    """
    values = _non_terminal_values(mdp, values)
    new_values = np.zeros(mdp.n_states, dtype=np.float64)
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        best = -np.inf
        for a in mdp.actions:
            best = max(
                best,
                mdp.expected_reward(s, a) + mdp.gamma * mdp._expected_next_value(s, a, values),
            )
        new_values[i] = best
    return new_values


def solve_bellman_expectation(mdp: FiniteMDP, policy_probs: np.ndarray) -> np.ndarray:
    """Solve the Bellman expectation equations exactly: v_pi = (I - gamma P_pi)^{-1} r_pi.

    Restricted to non-terminal states (terminal values are 0). Book Section 3.5.
    """
    policy_probs = _policy_matrix(mdp, policy_probs)
    non_terminal = [i for i, s in enumerate(mdp.states) if s not in mdp.terminal_states]
    n = len(non_terminal)
    matrix = np.eye(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    for row, i in enumerate(non_terminal):
        s = mdp.states[i]
        rhs[row] = sum(
            policy_probs[i, j] * mdp.expected_reward(s, a)
            for j, a in enumerate(mdp.actions)
        )
        for col, j_state in enumerate(non_terminal):
            ns = mdp.states[j_state]
            coefficient = 0.0
            for j, a in enumerate(mdp.actions):
                if policy_probs[i, j] == 0.0:
                    continue
                coefficient += policy_probs[i, j] * sum(
                    p for p, _, s_next in mdp.transition_probs(s, a) if s_next == ns
                )
            matrix[row, col] -= mdp.gamma * coefficient
    solved = np.linalg.solve(matrix, rhs)
    values = np.zeros(mdp.n_states, dtype=np.float64)
    for row, i in enumerate(non_terminal):
        values[i] = solved[row]
    return values


def bellman_expectation_residual(
    mdp: FiniteMDP, policy_probs: np.ndarray, values: np.ndarray
) -> float:
    """Max absolute Bellman-expectation residual over non-terminal states.

    A residual of 0 means values satisfy the Bellman expectation equation exactly.
    """
    backed_up = bellman_backup_expectation(mdp, policy_probs, values)
    normalized = _non_terminal_values(mdp, values)
    residuals = [
        abs(normalized[i] - backed_up[i])
        for i, s in enumerate(mdp.states)
        if s not in mdp.terminal_states
    ]
    return max(residuals) if residuals else 0.0


def bellman_optimality_residual(mdp: FiniteMDP, values: np.ndarray) -> float:
    """Max absolute Bellman-optimality residual over non-terminal states."""
    backed_up = bellman_backup_optimality(mdp, values)
    normalized = _non_terminal_values(mdp, values)
    residuals = [
        abs(normalized[i] - backed_up[i])
        for i, s in enumerate(mdp.states)
        if s not in mdp.terminal_states
    ]
    return max(residuals) if residuals else 0.0


def verify_bellman(
    mdp: FiniteMDP,
    policy_probs: np.ndarray | None = None,
    values: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> bool:
    """Verify Bellman equations for the given policy/values (Algorithm 3.4).

    If `values` is None, the exact solution v_pi is computed via solve_bellman_expectation
    (for a policy) or by iterating Bellman-optimality backups until convergence
    (if policy_probs is None), then the residual is checked against `tolerance`.

    Args:
        mdp: The finite MDP.
        policy_probs: Optional policy matrix; verifies the expectation equation.
        values: Optional value vector; if given, checks these values directly.
        tolerance: Maximum allowed absolute residual.

    Returns:
        True if max residual <= tolerance.
    """
    if values is None:
        if policy_probs is not None:
            values = solve_bellman_expectation(mdp, policy_probs)
        else:
            values = np.zeros(mdp.n_states, dtype=np.float64)
            for _ in range(10_000):
                updated = bellman_backup_optimality(mdp, values)
                if np.max(np.abs(updated - values)) < tolerance:
                    values = updated
                    break
                values = updated
    if policy_probs is not None:
        return bool(bellman_expectation_residual(mdp, policy_probs, values) <= tolerance)
    return bool(bellman_optimality_residual(mdp, values) <= tolerance)


def recycling_robot(
    alpha: float = 0.7,
    beta: float = 0.4,
    gamma: float = 0.9,
    seed: int | None = None,
) -> FiniteMDP:
    """Recycling-robot MDP (Example 3.3), Book Section 3.4.

    States: 'high' (good charge), 'low' (battery alarm).
    Actions: 'search' (find cans, +1 reward), 'wait' (stay put, 0 reward),
    'recharge' (recharge to 'high', 0 reward).

    Dynamics parameters (modelling choices, not from the book):
        alpha: P(stay 'high' after searching from 'high').
        beta: P(stay 'low' after searching from 'low').
    Depleting the battery while searching (low -> high via rescue) costs -3.
    """
    states = ["high", "low"]
    actions = ["search", "wait", "recharge"]
    transitions = {
        ("high", "search"): [(alpha, 1.0, "high"), (1.0 - alpha, 1.0, "low")],
        ("high", "wait"): [(1.0, 0.0, "high")],
        ("high", "recharge"): [(1.0, 0.0, "high")],
        ("low", "search"): [(beta, 1.0, "low"), (1.0 - beta, -3.0, "high")],
        ("low", "wait"): [(1.0, 0.0, "low")],
        ("low", "recharge"): [(1.0, 0.0, "high")],
    }
    return FiniteMDP(states, actions, transitions, gamma=gamma, seed=seed)
