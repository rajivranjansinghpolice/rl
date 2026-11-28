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
        """init (reference implementation).


        Args:
            states: See the relevant book section and function body.
            actions: See the relevant book section and function body.
            transitions: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            terminal_states: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
        # initialize the parameters and state required by the algorithm (book Chapter 3).
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

        # self.states: list attribute/state used by the algorithm (book Chapter 3).
        self.states: list = list(states)
        # self.actions: list attribute/state used by the algorithm (book Chapter 3).
        self.actions: list = list(actions)
        # self.gamma: float attribute/state used by the algorithm (book Chapter 3).
        self.gamma: float = float(gamma)
        # self.terminal states: set attribute/state used by the algorithm (book Chapter 3).
        self.terminal_states: set = set(terminal_states or [])
        # self.state index: dict attribute/state used by the algorithm (book Chapter 3).
        self.state_index: dict = {s: i for i, s in enumerate(self.states)}
        # self.action index: dict attribute/state used by the algorithm (book Chapter 3).
        self.action_index: dict = {a: i for i, a in enumerate(self.actions)}
        self.rng = np.random.default_rng(seed)

        # unknown: value computed from its operands (book Chapter 3).
        unknown = self.terminal_states - set(self.states)
        if unknown:
            raise ValueError(f"terminal states not in states: {unknown}")

        self._validate_transitions(transitions)
        # self.transitions: dict attribute/state used by the algorithm (book Chapter 3).
        self.transitions: dict = {
            (s, a): [(float(p), float(r), ns) for p, r, ns in transitions[(s, a)]]
            for s in self.states
            for a in self.actions
            if (s, a) in transitions
        }
        self.current_state = None
        # self.done: bool attribute/state used by the algorithm (book Chapter 3).
        self.done: bool = False

    def _validate_transitions(self, transitions: dict) -> None:
        """validate transitions (reference implementation).


        Args:
            transitions: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
        # internal helper for the validate transitions step (book Chapter 3).
        # all states: result of set(...) (book Chapter 3).
        all_states = set(self.states)
        # all actions: result of set(...) (book Chapter 3).
        all_actions = set(self.actions)
        # Iterate over ((s, a), outcomes): each item in transitions.items() (book algorithm).
        for (s, a), outcomes in transitions.items():
            if s not in all_states:
                raise ValueError(f"transition references unknown state: {s}")
            if a not in all_actions:
                raise ValueError(f"transition references unknown action: {a}")
            # prob sum: sum of the values (book Chapter 3).
            prob_sum = sum(p for p, _, _ in outcomes)
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                raise ValueError(f"probabilities for ({s}, {a}) sum to {prob_sum}, expected 1.0")
            # Iterate over (p, _, ns): each item in outcomes (book algorithm).
            for p, _, ns in outcomes:
                if p < 0.0:
                    raise ValueError(f"negative probability in transitions for ({s}, {a})")
                if ns not in all_states:
                    raise ValueError(f"transition references unknown next state: {ns}")
        # Iterate over s: one pass per state (book algorithm).
        for s in self.states:
            if s not in self.terminal_states:
                # Iterate over a: one pass per action (book algorithm).
                for a in self.actions:
                    if (s, a) not in transitions:
                        raise ValueError(f"missing transitions for non-terminal ({s}, {a})")

    @property
    def n_states(self) -> int:
        """n states (reference implementation).


        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
        # carry out the n states step (book Chapter 3).
        return len(self.states)

    @property
    def n_actions(self) -> int:
        """n actions (reference implementation).


        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
        # carry out the n actions step (book Chapter 3).
        return len(self.actions)

    def is_terminal(self, state) -> bool:
        """Return True if state is a terminal state.

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

        Args:
            state: See Algorithm 3.1 and the implementation for its role.

        Returns:
            is terminal as defined in Algorithm 3.1."""
        # carry out the is terminal step (book Chapter 3).
        return state in self.terminal_states

    def transition_probs(self, state, action) -> list:
        """Return p(s', r | s, a) as a list of (probability, reward, next_state).

        Book Section 3.2: the dynamics p(s', r | s, a).

        Book: Sutton & Barto (2nd ed.), Section 3.2 (Chapter 3).

        Args:
            state: See Section 3.2 and the implementation for its role.
            action: See Section 3.2 and the implementation for its role.

        Returns:
            transition probs as defined in Section 3.2."""
        # carry out the transition probs step (book Chapter 3).
        if state not in self.transitions and self.is_terminal(state):
            raise RuntimeError(f"terminal state {state} has no transitions")
        if (state, action) not in self.transitions:
            raise KeyError(f"no transitions defined for ({state}, {action})")
        return list(self.transitions[(state, action)])

    def expected_reward(self, state, action) -> float:
        """Expected reward E[R_{t+1} | St=s, At=a]. Book Section 3.2.

        Book: Sutton & Barto (2nd ed.), Section 3.2 (Chapter 3).

        Args:
            state: See Section 3.2 and the implementation for its role.
            action: See Section 3.2 and the implementation for its role.

        Returns:
            expected reward as defined in Section 3.2."""
        # carry out the expected reward step (book Chapter 3).
        return sum(p * r for p, r, _ in self.transition_probs(state, action))

    def reset(self, state=None) -> object:
        """Reset to `state`, the configured `start_state`, or a uniform sample.

        A `start_state` attribute set by task factories (e.g. Chapter 8's
        gridworlds) takes precedence over uniform sampling when no state is
        passed explicitly.

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

        Args:
            state: See Algorithm 3.1 and the implementation for its role.

        Returns:
            reset as defined in Algorithm 3.1."""
        # reset internal state for a new episode (book Chapter 3).
        if state is None:
            # state: current state of the environment or MDP (book Chapter 3).
            state = getattr(self, "start_state", None)
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

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

        Args:
            action: See Algorithm 3.1 and the implementation for its role.

        Returns:
            step as defined in Algorithm 3.1."""
        # advance one step and return the observed transition (book Chapter 3).
        if self.current_state is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        # outcomes: result of self.transition_probs(...) (book Chapter 3).
        outcomes = self.transition_probs(self.current_state, action)
        # probs: action probabilities from the current policy (book Chapter 3).
        probs = np.array([p for p, _, _ in outcomes], dtype=np.float64)
        # idx: integer conversion (book Chapter 3).
        idx = int(self.rng.choice(len(outcomes), p=probs))
        # (_, reward, next_state): element selected from the indexed collection (book Chapter 3).
        _, reward, next_state = outcomes[idx]
        self.current_state = next_state
        self.done = self.is_terminal(next_state)
        return reward, next_state, self.done

    def _expected_next_value(self, state, action, values: np.ndarray) -> float:
        """expected next value (reference implementation).


        Args:
            state: See the relevant book section and function body.
            action: See the relevant book section and function body.
            values: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
        # internal helper for the expected next value step (book Chapter 3).
        # transitions: result of self.transition_probs(...) (book Chapter 3).
        transitions = self.transition_probs(state, action)
        return sum(p * values[self.state_index[ns]] for p, _, ns in transitions)


def compute_return(rewards: list, gamma: float = 0.99) -> float:
    """Compute the discounted return G_t (Algorithm 3.2). Book Section 3.3.

    G_t = sum_{k=0}^{T-t-1} gamma^k * R_{t+k+1}  (episodic case).

    Book: Sutton & Barto (2nd ed.), Algorithm 3.2 (Chapter 3).

    Args:
        rewards: See Algorithm 3.2 and the implementation for its role.
        gamma: See Algorithm 3.2 and the implementation for its role.

    Returns:
        compute return as defined in Algorithm 3.2."""
    # carry out the compute return step (book Chapter 3).
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    total = 0.0
    discount = 1.0
    # Iterate over reward: each item in rewards (book algorithm).
    for reward in rewards:
        # total: add value computed from its operands into the running value (book Chapter 3).
        total += discount * reward
        # discount: multiply reference to an existing value into the running value (book Chapter
        discount *= gamma
    return total


def compute_returns_to_go(rewards: list, gamma: float = 0.99) -> list:
    """Compute the reward-to-go G_t for each t (Algorithm 3.2). Book Section 3.3.

    Iterates backwards from the end of the episode:
        G_T = 0;  G_t = R_{t+1} + gamma * G_{t+1}.

    Returns a list with the same length as rewards: out[t] = G_t.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.2 (Chapter 3).

    Args:
        rewards: See Algorithm 3.2 and the implementation for its role.
        gamma: See Algorithm 3.2 and the implementation for its role.

    Returns:
        compute returns to go as defined in Algorithm 3.2."""
    # carry out the compute returns to go step (book Chapter 3).
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    # returns: accumulated returns used as update targets (book Chapter 3).
    returns = [0.0] * len(rewards)
    running = 0.0
    # Iterate over i: one pass per index (book algorithm).
    for i in range(len(rewards) - 1, -1, -1):
        # running: running average accumulator (book Chapter 3).
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
        """init (reference implementation).


        Args:
            n_states: See the relevant book section and function body.
            n_actions: See the relevant book section and function body.
            preferences: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 3.3 (Chapter 3)."""
        # initialize the parameters and state required by the algorithm (book Chapter 3).
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if preferences is None:
            self.H = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        else:
            # preferences: result of np.asarray(...) (book Chapter 3).
            preferences = np.asarray(preferences, dtype=np.float64)
            if preferences.shape != (self.n_states, self.n_actions):
                raise ValueError("preferences must have shape (n_states, n_actions)")
            self.H = preferences.copy()
        self.rng = np.random.default_rng(seed)

    def action_probs(self, state: int) -> np.ndarray:
        """Return pi(.|state) as a probability vector (numerically stable softmax).

        Book: Sutton & Barto (2nd ed.), Algorithm 3.3 (Chapter 3).

        Args:
            state: See Algorithm 3.3 and the implementation for its role.

        Returns:
            action probs as defined in Algorithm 3.3."""
        # carry out the action probs step (book Chapter 3).
        # logits: value computed from its operands (book Chapter 3).
        logits = self.H[state] - np.max(self.H[state])
        # exp logits: result of np.exp(...) (book Chapter 3).
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def greedy_action(self, state: int) -> int:
        """Return argmax_a pi(a|state) (ties broken uniformly at random).

        Book: Sutton & Barto (2nd ed.), Algorithm 3.3 (Chapter 3).

        Args:
            state: See Algorithm 3.3 and the implementation for its role.

        Returns:
            greedy action as defined in Algorithm 3.3."""
        # return the highest-valued action with tie-breaking (book Chapter 3).
        # probs: action probabilities from the current policy (book Chapter 3).
        probs = self.action_probs(state)
        # best: best action or best value found so far (book Chapter 3).
        best = np.flatnonzero(probs == probs.max())
        return int(self.rng.choice(best))

    def sample_action(self, state: int) -> int:
        """Sample A ~ pi(.|state).

        Book: Sutton & Barto (2nd ed.), Algorithm 3.3 (Chapter 3).

        Args:
            state: See Algorithm 3.3 and the implementation for its role.

        Returns:
            sample action as defined in Algorithm 3.3."""
        # carry out the sample action step (book Chapter 3).
        # probs: action probabilities from the current policy (book Chapter 3).
        probs = self.action_probs(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def preferences(self, state: int) -> np.ndarray:
        """Return a copy of the preference row H[state].

        Book: Sutton & Barto (2nd ed.), Algorithm 3.3 (Chapter 3).

        Args:
            state: See Algorithm 3.3 and the implementation for its role.

        Returns:
            preferences as defined in Algorithm 3.3."""
        # carry out the preferences step (book Chapter 3).
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
        # apply the learning update rule to the current estimates (book Chapter 3).
        if step_size < 0.0:
            raise ValueError("step_size must be non-negative")
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        # probs: action probabilities from the current policy (book Chapter 3).
        probs = self.action_probs(state)
        # self.H[state]: add value computed from its operands into the running value (book ...
        self.H[state] += step_size * advantage * (
            np.eye(self.n_actions)[action] - probs
        )


def _policy_matrix(mdp: FiniteMDP, policy_probs: np.ndarray) -> np.ndarray:
    """policy matrix (reference implementation).


    Args:
        mdp: See the relevant book section and function body.
        policy_probs: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
    # internal helper for the policy matrix step (book Chapter 3).
    # policy probs: result of np.asarray(...) (book Chapter 3).
    policy_probs = np.asarray(policy_probs, dtype=np.float64)
    # expected: collection of the listed elements (book Chapter 3).
    expected = (mdp.n_states, mdp.n_actions)
    if policy_probs.shape != expected:
        raise ValueError(f"policy_probs must have shape {expected}")
    # row sums: sum of the values (book Chapter 3).
    row_sums = policy_probs.sum(axis=1)
    # non terminal rows: collection built by comprehension (book Chapter 3).
    non_terminal_rows = [
        i for i, s in enumerate(mdp.states) if s not in mdp.terminal_states
    ]
    # Iterate over i: each item in non_terminal_rows (book algorithm).
    for i in non_terminal_rows:
        if not np.isclose(row_sums[i], 1.0, atol=1e-6):
            raise ValueError(
                f"policy row for state {mdp.states[i]} sums to {row_sums[i]}, expected 1.0"
            )
    return policy_probs


def _non_terminal_values(mdp: FiniteMDP, values: np.ndarray) -> np.ndarray:
    """non terminal values (reference implementation).


    Args:
        mdp: See the relevant book section and function body.
        values: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
    # internal helper for the non terminal values step (book Chapter 3).
    # values: state-value estimates V(s) (book Chapter 3).
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mdp.n_states,):
        raise ValueError(f"values must have shape ({mdp.n_states},)")
    # result: independent copy of the data (book Chapter 3).
    result = values.copy()
    # Iterate over (i, s): one pass per state (book algorithm).
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

    Book: Sutton & Barto (2nd ed.), Algorithm 3.4 (Chapter 3).

    Args:
        mdp: See Algorithm 3.4 and the implementation for its role.
        policy_probs: See Algorithm 3.4 and the implementation for its role.
        values: See Algorithm 3.4 and the implementation for its role.

    Returns:
        bellman backup expectation as defined in Algorithm 3.4."""
    # carry out the bellman backup expectation step (book Chapter 3).
    # policy probs: result of _policy_matrix(...) (book Chapter 3).
    policy_probs = _policy_matrix(mdp, policy_probs)
    # values: state-value estimates V(s) (book Chapter 3).
    values = _non_terminal_values(mdp, values)
    # new values: zero-filled array (book Chapter 3).
    new_values = np.zeros(mdp.n_states, dtype=np.float64)
    # Iterate over (i, s): one pass per state (book algorithm).
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        total = 0.0
        # Iterate over (j, a): one pass per action (book algorithm).
        for j, a in enumerate(mdp.actions):
            if policy_probs[i, j] == 0.0:
                continue
            # total: add value computed from its operands into the running value (book Chapter 3).
            total += policy_probs[i, j] * (
                mdp.expected_reward(s, a) + mdp.gamma * mdp._expected_next_value(s, a, values)
            )
        new_values[i] = total
    return new_values


def bellman_backup_optimality(mdp: FiniteMDP, values: np.ndarray) -> np.ndarray:
    """One optimality Bellman backup (Algorithm 3.4, optimality equation). Book Section 3.5.

        v'(s) = max_a sum_{s',r} p(s',r|s,a) * [r + gamma * v(s')]

    Terminal states back up to 0.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.4 (Chapter 3).

    Args:
        mdp: See Algorithm 3.4 and the implementation for its role.
        values: See Algorithm 3.4 and the implementation for its role.

    Returns:
        bellman backup optimality as defined in Algorithm 3.4."""
    # carry out the bellman backup optimality step (book Chapter 3).
    # values: state-value estimates V(s) (book Chapter 3).
    values = _non_terminal_values(mdp, values)
    # new values: zero-filled array (book Chapter 3).
    new_values = np.zeros(mdp.n_states, dtype=np.float64)
    # Iterate over (i, s): one pass per state (book algorithm).
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        # best: best action or best value found so far (book Chapter 3).
        best = -np.inf
        # Iterate over a: one pass per action (book algorithm).
        for a in mdp.actions:
            # best: best action or best value found so far (book Chapter 3).
            best = max(
                best,
                mdp.expected_reward(s, a) + mdp.gamma * mdp._expected_next_value(s, a, values),
            )
        new_values[i] = best
    return new_values


def solve_bellman_expectation(mdp: FiniteMDP, policy_probs: np.ndarray) -> np.ndarray:
    """Solve the Bellman expectation equations exactly: v_pi = (I - gamma P_pi)^{-1} r_pi.

    Restricted to non-terminal states (terminal values are 0). Book Section 3.5.

    Book: Sutton & Barto (2nd ed.), Section 3.5 (Chapter 3).

    Args:
        mdp: See Section 3.5 and the implementation for its role.
        policy_probs: See Section 3.5 and the implementation for its role.

    Returns:
        solve bellman expectation as defined in Section 3.5."""
    # carry out the solve bellman expectation step (book Chapter 3).
    # policy probs: result of _policy_matrix(...) (book Chapter 3).
    policy_probs = _policy_matrix(mdp, policy_probs)
    # non terminal: collection built by comprehension (book Chapter 3).
    non_terminal = [i for i, s in enumerate(mdp.states) if s not in mdp.terminal_states]
    # n: result of len(...) (book Chapter 3).
    n = len(non_terminal)
    # matrix: result of np.eye(...) (book Chapter 3).
    matrix = np.eye(n, dtype=np.float64)
    # rhs: zero-filled array (book Chapter 3).
    rhs = np.zeros(n, dtype=np.float64)
    # Iterate over (row, i): indexed entries (book algorithm).
    for row, i in enumerate(non_terminal):
        # s: current state index (book Chapter 3).
        s = mdp.states[i]
        rhs[row] = sum(
            policy_probs[i, j] * mdp.expected_reward(s, a)
            for j, a in enumerate(mdp.actions)
        )
        # Iterate over (col, j_state): indexed entries (book algorithm).
        for col, j_state in enumerate(non_terminal):
            # ns: element selected from the indexed collection (book Chapter 3).
            ns = mdp.states[j_state]
            coefficient = 0.0
            # Iterate over (j, a): one pass per action (book algorithm).
            for j, a in enumerate(mdp.actions):
                if policy_probs[i, j] == 0.0:
                    continue
                # coefficient: add value computed from its operands into the running value (book
                coefficient += policy_probs[i, j] * sum(
                    p for p, _, s_next in mdp.transition_probs(s, a) if s_next == ns
                )
            # matrix[row, col]: subtract value computed from its operands into the running value
            matrix[row, col] -= mdp.gamma * coefficient
    # solved: result of np.linalg.solve(...) (book Chapter 3).
    solved = np.linalg.solve(matrix, rhs)
    # values: state-value estimates V(s) (book Chapter 3).
    values = np.zeros(mdp.n_states, dtype=np.float64)
    # Iterate over (row, i): indexed entries (book algorithm).
    for row, i in enumerate(non_terminal):
        values[i] = solved[row]
    return values


def bellman_expectation_residual(
    mdp: FiniteMDP, policy_probs: np.ndarray, values: np.ndarray
) -> float:
    """Max absolute Bellman-expectation residual over non-terminal states.

    A residual of 0 means values satisfy the Bellman expectation equation exactly.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

    Args:
        mdp: See Algorithm 3.1 and the implementation for its role.
        policy_probs: See Algorithm 3.1 and the implementation for its role.
        values: See Algorithm 3.1 and the implementation for its role.

    Returns:
        bellman expectation residual as defined in Algorithm 3.1."""
    # carry out the bellman expectation residual step (book Chapter 3).
    # backed up: result of bellman_backup_expectation(...) (book Chapter 3).
    backed_up = bellman_backup_expectation(mdp, policy_probs, values)
    # normalized: result of _non_terminal_values(...) (book Chapter 3).
    normalized = _non_terminal_values(mdp, values)
    # residuals: collection built by comprehension (book Chapter 3).
    residuals = [
        abs(normalized[i] - backed_up[i])
        for i, s in enumerate(mdp.states)
        if s not in mdp.terminal_states
    ]
    return max(residuals) if residuals else 0.0


def bellman_optimality_residual(mdp: FiniteMDP, values: np.ndarray) -> float:
    """Max absolute Bellman-optimality residual over non-terminal states.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

    Args:
        mdp: See Algorithm 3.1 and the implementation for its role.
        values: See Algorithm 3.1 and the implementation for its role.

    Returns:
        bellman optimality residual as defined in Algorithm 3.1."""
    # carry out the bellman optimality residual step (book Chapter 3).
    # backed up: result of bellman_backup_optimality(...) (book Chapter 3).
    backed_up = bellman_backup_optimality(mdp, values)
    # normalized: result of _non_terminal_values(...) (book Chapter 3).
    normalized = _non_terminal_values(mdp, values)
    # residuals: collection built by comprehension (book Chapter 3).
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
    # carry out the verify bellman step (book Chapter 3).
    if values is None:
        if policy_probs is not None:
            # values: state-value estimates V(s) (book Chapter 3).
            values = solve_bellman_expectation(mdp, policy_probs)
        else:
            # values: state-value estimates V(s) (book Chapter 3).
            values = np.zeros(mdp.n_states, dtype=np.float64)
            # Iterate over _: one pass per index (book algorithm).
            for _ in range(10_000):
                # updated: result of bellman_backup_optimality(...) (book Chapter 3).
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

    Book: Sutton & Barto (2nd ed.), Section 3.4 (Chapter 3).

    Args:
        alpha: See Section 3.4 and the implementation for its role.
        beta: See Section 3.4 and the implementation for its role.
        gamma: See Section 3.4 and the implementation for its role.
        seed: See Section 3.4 and the implementation for its role.

    Returns:
        recycling robot as defined in Section 3.4."""
    # carry out the recycling robot step (book Chapter 3).
    # states: all states in the MDP (book Chapter 3).
    states = ["high", "low"]
    # actions: all actions available in the MDP (book Chapter 3).
    actions = ["search", "wait", "recharge"]
    # transitions: dictionary of key/value pairs (book Chapter 3).
    transitions = {
        ("high", "search"): [(alpha, 1.0, "high"), (1.0 - alpha, 1.0, "low")],
        ("high", "wait"): [(1.0, 0.0, "high")],
        ("high", "recharge"): [(1.0, 0.0, "high")],
        ("low", "search"): [(beta, 1.0, "low"), (1.0 - beta, -3.0, "high")],
        ("low", "wait"): [(1.0, 0.0, "low")],
        ("low", "recharge"): [(1.0, 0.0, "high")],
    }
    return FiniteMDP(states, actions, transitions, gamma=gamma, seed=seed)


def example_3_5_gridworld(gamma: float = 0.9) -> FiniteMDP:
    """Example 3.5: the 5x5 gridworld MDP (Sutton & Barto, 2nd ed., Sec. 3.5).

    5x5 grid; deterministic actions ``up``, ``down``, ``left``, ``right``.
    Moving off the grid leaves the agent in place with reward -1; all other
    moves yield reward 0, except:

        A  (row 0, col 1): any action yields +10 and teleports to A' (row 4, col 1)
        B  (row 0, col 3): any action yields +5  and teleports to B' (row 2, col 3)

    The task is continuing. Under the equiprobable random policy the converged
    values match the book's Figure 3.2 (2018 edition; local draft PDF numbers it
    as part of the Example 3.5 figures), e.g. v_random(A) ~ 8.8 (2017-era drafts
    show ~8.8, consistent with gamma = 0.9). Solving the Bellman optimality
    equations reproduces the v* grid with v*(A) ~ 24.42 (the book: "about 24.4").
    Figure identities cannot be re-verified against the local PDF summary text,
    so this MDP is implemented from the book's dynamics description; all figure
    reproductions are labelled as illustrative demonstrations.

    Book: Sutton & Barto (2nd ed.), Section 3.5 (Chapter 3).

    Args:
        gamma: See Section 3.5 and the implementation for its role.

    Returns:
        example 3 5 gridworld as defined in Section 3.5."""
    # carry out the example 3 5 gridworld step (book Chapter 3).
    A, Ap = (0, 1), (4, 1)
    # (B, Bp): collection of the listed elements (book Chapter 3).
    B, Bp = (0, 3), (2, 3)
    grid = 5
    # states: all states in the MDP (book Chapter 3).
    states = [(r, c) for r in range(grid) for c in range(grid)]
    # actions: all actions available in the MDP (book Chapter 3).
    actions = ["up", "down", "left", "right"]
    # deltas: per-step TD errors or differences (book Chapter 3).
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    # transitions: dictionary of key/value pairs (book Chapter 3).
    transitions = {}
    # Iterate over s: one pass per state (book algorithm).
    for s in states:
        if s == A:
            # Iterate over a: one pass per action (book algorithm).
            for a in actions:
                transitions[(s, a)] = [(1.0, 10.0, Ap)]
            continue
        if s == B:
            # Iterate over a: one pass per action (book algorithm).
            for a in actions:
                transitions[(s, a)] = [(1.0, 5.0, Bp)]
            continue
        # Iterate over a: one pass per action (book algorithm).
        for a in actions:
            # (dr, dc): element selected from the indexed collection (book Chapter 3).
            dr, dc = deltas[a]
            # (nr, nc): collection of the listed elements (book Chapter 3).
            nr, nc = s[0] + dr, s[1] + dc
            if 0 <= nr < grid and 0 <= nc < grid:
                transitions[(s, a)] = [(1.0, 0.0, (nr, nc))]
            else:
                transitions[(s, a)] = [(1.0, -1.0, s)]
    return FiniteMDP(states, actions, transitions, gamma=gamma)


def example_3_5_gridworld_random_policy_values(gamma: float = 0.9) -> np.ndarray:
    """Converged state values v_pi for the equiprobable random policy (pi = 1/4 each action).

    Uses the exact Bellman expectation solve. Book Section 3.5, Example 3.5.

    Book: Sutton & Barto (2nd ed.), Section 3.5 (Chapter 3).

    Args:
        gamma: See Section 3.5 and the implementation for its role.

    Returns:
        example 3 5 gridworld random policy values as defined in Section 3.5."""
    # carry out the example 3 5 gridworld random policy values step (book Chapter 3).
    # mdp: result of example_3_5_gridworld(...) (book Chapter 3).
    mdp = example_3_5_gridworld(gamma=gamma)
    # uniform: array filled with a constant value (book Chapter 3).
    uniform = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
    return solve_bellman_expectation(mdp, uniform)


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 3)
# ---------------------------------------------------------------------------

# Value iteration for the figure panels: stop once the largest value change is
# below tolerance; the iteration cap only guards against a non-converging MDP.
_VI_TOLERANCE = 1e-10
_VI_MAX_ITER = 10_000


def _greedy_policy_from_values(mdp: FiniteMDP, values: np.ndarray) -> np.ndarray:
    """Uniform-over-argmax policy with respect to the given state values.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

    Args:
        mdp: See Algorithm 3.1 and the implementation for its role.
        values: See Algorithm 3.1 and the implementation for its role.

    Returns:
        greedy policy from values as defined in Algorithm 3.1."""
    # policy: policy mapping states to action probabilities (book Chapter 3).
    policy = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    for i, s in enumerate(mdp.states):
        if s in mdp.terminal_states:
            continue
        # One backup scores every action; ties split the probability evenly.
        q_vals = np.array(
            [
                mdp.expected_reward(s, a) + mdp.gamma * mdp._expected_next_value(s, a, values)
                for a in mdp.actions
            ]
        )
        # best: best action or best value found so far (book Chapter 3).
        best = np.flatnonzero(np.isclose(q_vals, q_vals.max()))
        policy[i, best] = 1.0 / len(best)
    return policy


def _optimal_values_and_policy(
    mdp: FiniteMDP, tolerance: float = _VI_TOLERANCE, max_iter: int = _VI_MAX_ITER
):
    """optimal values and policy (reference implementation).


    Args:
        mdp: See the relevant book section and function body.
        tolerance: See the relevant book section and function body.
        max_iter: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3)."""
    # values: state-value estimates V(s) (book Chapter 3).
    values = np.zeros(mdp.n_states, dtype=np.float64)
    for _ in range(max_iter):
        # Reapply the optimality backup so value news propagates across states
        # without needing a new episode of experience.
        updated = bellman_backup_optimality(mdp, values)
        if np.max(np.abs(updated - values)) < tolerance:
            values = updated
            break
        values = updated
    return values, _greedy_policy_from_values(mdp, values)


# Three action bars share one state group; the width also sets their spacing.
_POLICY_BAR_WIDTH = 0.25


def _draw_state_values(ax, states: list, v: list[float]) -> None:
    """Left panel: the converged optimal value of each state.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

    Args:
        ax: See Algorithm 3.1 and the implementation for its role.
        states: See Algorithm 3.1 and the implementation for its role.
        v: See Algorithm 3.1 and the implementation for its role."""
    # internal helper for the draw state values step (book Chapter 3).
    ax.bar(states, v, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("State value v*(s)")
    ax.set_title("Optimal state values (Bellman optimality)")
    # Iterate over (idx, val): indexed entries (book algorithm).
    for idx, val in enumerate(v):
        ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)


def _draw_greedy_policy(
    ax, states: list, actions: list, policy: np.ndarray, mdp: FiniteMDP
) -> None:
    """Right panel: one bar per action, showing its greedy-policy probability.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

    Args:
        ax: See Algorithm 3.1 and the implementation for its role.
        states: See Algorithm 3.1 and the implementation for its role.
        actions: See Algorithm 3.1 and the implementation for its role.
        policy: See Algorithm 3.1 and the implementation for its role.
        mdp: See Algorithm 3.1 and the implementation for its role."""
    # internal helper for the draw greedy policy step (book Chapter 3).
    # x: state feature vector (book Chapter 3).
    x = np.arange(len(states))
    # Iterate over (j, a): one pass per action (book algorithm).
    for j, a in enumerate(actions):
        # probs: action probabilities from the current policy (book Chapter 3).
        probs = [policy[mdp.state_index[s], j] for s in states]
        ax.bar(x + (j - 1) * _POLICY_BAR_WIDTH, probs, _POLICY_BAR_WIDTH, label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Greedy probability")
    ax.set_title("Greedy action(s) w.r.t. v*")
    ax.legend()


def plot_recycling_robot_example(
    save_path: str | None = None,
    show: bool = False,
    alpha: float = 0.7,
    beta: float = 0.4,
    gamma: float = 0.9,
):
    """Recycling robot demonstration (Example 3.3, Sec. 3.4-3.5).

    Sutton & Barto (2nd ed.), Example 3.3 "Recycling Robot": a finite MDP with
    states {high, low} (battery level) and actions {search, wait, recharge}.
    Dynamics/returns follow the implementation's modeling choices (see
    ``recycling_robot``): searching can deplete the battery; rescuing from low
    incurs reward -3. This figure computes the optimal state values v* by
    iterating the Bellman optimality backup (Eq. 3.19/3.20, Sec. 3.6) and shows
    v*(high), v*(low) alongside greedy action preferences, illustrating how
    value functions summarize long-term return under the MDP dynamics.

    Book: Sutton & Barto (2nd ed.), Section 3.4 (Chapter 3).

    Args:
        save_path: See Section 3.4 and the implementation for its role.
        show: See Section 3.4 and the implementation for its role.
        alpha: See Section 3.4 and the implementation for its role.
        beta: See Section 3.4 and the implementation for its role.
        gamma: See Section 3.4 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 3).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mdp: result of recycling_robot(...) (book Chapter 3).
    mdp = recycling_robot(alpha=alpha, beta=beta, gamma=gamma)
    # (values, policy): result of _optimal_values_and_policy(...) (book Chapter 3).
    values, policy = _optimal_values_and_policy(mdp)
    # (fig, (ax1, ax2)): result of plt.subplots(...) (book Chapter 3).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    # states: all states in the MDP (book Chapter 3).
    states = list(mdp.states)
    # v: collection built by comprehension (book Chapter 3).
    v = [values[mdp.state_index[s]] for s in states]
    _draw_state_values(ax1, states, v)
    _draw_greedy_policy(ax2, states, list(mdp.actions), policy, mdp)

    fig.suptitle(
        "S&B (2nd ed.), Example 3.3 Recycling Robot (Sec. 3.4-3.6)\n"
        f"alpha={alpha}, beta={beta}, gamma={gamma}",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_example_3_5_gridworld(save_path: str | None = None, show: bool = False,
                               gamma: float = 0.9):
    """Example 3.5 (5x5 gridworld): v_pi (random policy) and v* (optimal), as in the book.

    Two heatmaps: (a) converged state values under the equiprobable random policy,
    and (b) the optimal state-value function from the Bellman optimality equation
    (value iteration). Special cells A (+10 -> A' teleport) and B (+5 -> B' teleport)
    are marked. Reproduced as an illustrative demonstration; figure identities differ
    across editions (Figs 3.2/3.5 in the 2018 edition; the local in-progress draft PDF
    shows the same dynamics in the Example 3.5 figures and its exercises cite v*(A) ~
    24.4). MDP is implemented from the dynamics description, not from figure pixels.

    Book: Sutton & Barto (2nd ed.), Algorithm 3.1 (Chapter 3).

    Args:
        save_path: See Algorithm 3.1 and the implementation for its role.
        show: See Algorithm 3.1 and the implementation for its role.
        gamma: See Algorithm 3.1 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 3).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mdp: result of example_3_5_gridworld(...) (book Chapter 3).
    mdp = example_3_5_gridworld(gamma=gamma)
    # uniform: array filled with a constant value (book Chapter 3).
    uniform = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
    # v random: result of solve_bellman_expectation(...) (book Chapter 3).
    v_random = solve_bellman_expectation(mdp, _policy_matrix(mdp, uniform))
    # (v_optimal, _): result of _optimal_values_and_policy(...) (book Chapter 3).
    v_optimal, _ = _optimal_values_and_policy(mdp)

    grid = 5
    # vr: result of v_random.reshape(...) (book Chapter 3).
    vr = v_random.reshape(grid, grid)
    # vo: result of v_optimal.reshape(...) (book Chapter 3).
    vo = v_optimal.reshape(grid, grid)

    # (fig, (ax1, ax2)): result of plt.subplots(...) (book Chapter 3).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))
    # Iterate over (ax, data, title): each item in ( (ax1, vr, "(a) v_pi, equiprobable random ...
    for ax, data, title in (
        (ax1, vr, "(a) v_pi, equiprobable random policy"),
        (ax2, vo, "(b) v*, optimal (Bellman optimality)"),
    ):
        # im: result of ax.imshow(...) (book Chapter 3).
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title, fontsize=10)
        # Iterate over r: one pass per index (book algorithm).
        for r in range(grid):
            # Iterate over c: one pass per index (book algorithm).
            for c in range(grid):
                ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Iterate over ((r, c), mark): each item in (((0, 1), "A"), ((0, 3), "B"), ((4, 1), "A'"), ...
    for (r, c), mark in (((0, 1), "A"), ((0, 3), "B"), ((4, 1), "A'"), ((2, 3), "B'")):
        # Iterate over ax: each item in (ax1, ax2) (book algorithm).
        for ax in (ax1, ax2):
            ax.text(c - 0.45, r - 0.45, mark, fontsize=11, ha="center", va="center",
                    color="red", fontweight="bold")

    fig.suptitle("S&B (2nd ed.), Example 3.5: the 5x5 gridworld (illustrative demonstration; "
                 "figure numbering differs across editions)", fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax2

