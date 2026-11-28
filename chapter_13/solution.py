"""Solution implementation for Chapter 13 - Policy Gradient Algorithms.

Reference: Sutton & Barto (2nd Ed), Chapter 13

Implements: Alg 13.2 (policy gradient theorem: numeric check),
Alg 13.3 (REINFORCE: Monte Carlo policy gradient),
Alg 13.4 (REINFORCE with baseline),
Alg 13.5 (one-step actor-critic),
Alg 13.7 (Gaussian policy for continuous actions).

The environment is a simple stochastic bandit (Section 13.1/13.3 style) used
to demonstrate the gradient updates; policies are tabular softmax over
action preferences, following the book's running example.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        logits: See Algorithm 13.2 and the implementation for its role.

    Returns:
        softmax as defined in Algorithm 13.2."""
    # internal helper for the softmax step (book Chapter 13).
    # shifted: value computed from its operands (book Chapter 13).
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    # exp logits: result of np.exp(...) (book Chapter 13).
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _discounted_returns(rewards: list, gamma: float) -> list:
    """G_t = sum_{k=t+1}^{T} gamma^{k-t-1} * R_k, computed backwards in place.

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        rewards: See Algorithm 13.2 and the implementation for its role.
        gamma: See Algorithm 13.2 and the implementation for its role.

    Returns:
        discounted returns as defined in Algorithm 13.2."""
    # internal helper for the discounted returns step (book Chapter 13).
    # returns: accumulated returns used as update targets (book Chapter 13).
    returns = [0.0] * len(rewards)
    running = 0.0
    # Iterate over t: one pass per index (book algorithm).
    for t in range(len(rewards) - 1, -1, -1):
        # running: running average accumulator (book Chapter 13).
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def soft_argmax_row(preferences: np.ndarray) -> np.ndarray:
    """Softmax action probabilities pi(.|s) from preference vector h(s, .).

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        preferences: See Algorithm 13.2 and the implementation for its role.

    Returns:
        soft argmax row as defined in Algorithm 13.2."""
    # carry out the soft argmax row step (book Chapter 13).
    return _softmax(preferences)


class PolicyEnvironment:
    """Stochastic bandit environment for policy gradient (Section 13.1/13.3).

    Each action a pays a deterministic reward r(a); the environment tracks the
    number of times each action was selected. `p` gives the distribution used
    when `select_action` is called without a policy; it is also useful for the
    numeric policy-gradient checks.

    Args:
        n_actions: Number of available actions.
        p: Action-probability vector of length n_actions (rows are validated).
        rewards: Reward vector of length n_actions.
    """

    def __init__(self, n_actions: int, p: np.ndarray, rewards: np.ndarray,
                 seed: int | None = None):
        """init (reference implementation).


        Args:
            n_actions: See the relevant book section and function body.
            p: See the relevant book section and function body.
            rewards: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        self.n_actions = int(n_actions)
        self.p = np.asarray(p, dtype=np.float64)
        self.rewards = np.asarray(rewards, dtype=np.float64)
        if self.p.shape != (self.n_actions,):
            raise ValueError("p must have shape (n_actions,)")
        if self.rewards.shape != (self.n_actions,):
            raise ValueError("rewards must have shape (n_actions,)")
        if not np.isclose(self.p.sum(), 1.0, atol=1e-6):
            raise ValueError("p must sum to 1")
        if (self.p < 0).any():
            raise ValueError("p must be non-negative")
        self.action_count = np.zeros(self.n_actions, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def select_action(self, policy: np.ndarray | None = None, seed: int | None = None) -> int:
        """Select an action.

        Args:
            policy: Probability distribution over actions. If None, uses `self.p`.
            seed: Optional seed for a one-off draw.

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

        Returns:
            select action as defined in Section 13.1."""
        # choose an action from the current estimates (book Chapter 13).
        # rng: seeded random generator so runs are reproducible (book Chapter 13).
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        # probs: action probabilities from the current policy (book Chapter 13).
        probs = np.asarray(policy, dtype=np.float64) if policy is not None else self.p
        if not np.isclose(probs.sum(), 1.0, atol=1e-6):
            raise ValueError("policy must sum to 1")
        # action: action selected for the current step (book Chapter 13).
        action = rng.choice(self.n_actions, p=probs)
        return int(action)

    def step(self, action: int) -> float:
        """Play `action`; return its reward and increment the action count.

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

        Args:
            action: See Section 13.1 and the implementation for its role.

        Returns:
            step as defined in Section 13.1."""
        # advance one step and return the observed transition (book Chapter 13).
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        # self.action_count[action]: add literal constant value into the running value (book ...
        self.action_count[action] += 1
        return float(self.rewards[action])

    def get_action_count(self, action: int) -> float:
        """Return the number of times `action` was played.

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

        Args:
            action: See Section 13.1 and the implementation for its role.

        Returns:
            get action count as defined in Section 13.1."""
        # carry out the get action count step (book Chapter 13).
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        return float(self.action_count[action])


# Central-difference step for the numeric gradient check; the tolerance that
# judges the result is a separate argument and keeps its own default.
_FINITE_DIFF_EPS = 1e-6


def check_policy_gradient_theorem(
    preferences: np.ndarray,
    q_values: np.ndarray,
    tolerance: float = 1e-6,
) -> bool:
    """Numeric check of the policy gradient theorem (Algorithm 13.2). Book Section 13.2.

    For a single-state softmax policy, the theorem gives
        dJ = sum_a q(a) * pi(a) * (onehot(a) - pi).
    Verified by comparing this analytic gradient with a central-difference
    gradient of J(theta) = sum_a softmax(theta)_a * q(a).

    Args:
        preferences: Action-preference vector theta.
        q_values: Action-value vector q(a).
        tolerance: Maximum allowed absolute difference.

    Returns:
        True if the analytic and numeric gradients agree within tolerance.
    """
    # carry out the check policy gradient theorem step (book Chapter 13).
    # theta: result of np.asarray(...) (book Chapter 13).
    theta = np.asarray(preferences, dtype=np.float64)
    # q: action-value estimate Q(s, a) (book Chapter 13).
    q = np.asarray(q_values, dtype=np.float64)
    if theta.shape != q.shape:
        raise ValueError("preferences and q_values must have the same shape")

    # probs: action probabilities from the current policy (book Chapter 13).
    probs = _softmax(theta)
    # analytic: array built from the supplied values (book Chapter 13).
    analytic = np.array(
        [q[i] * probs[i] - probs[i] * float(np.dot(q, probs)) for i in range(len(q))]
    )
    # numeric: result of np.zeros_like(...) (book Chapter 13).
    numeric = np.zeros_like(theta)
    # Iterate over i: one pass per index (book algorithm).
    for i in range(len(theta)):
        # offset: result of np.zeros_like(...) (book Chapter 13).
        offset = np.zeros_like(theta)
        offset[i] = _FINITE_DIFF_EPS
        numeric[i] = (
            float(np.dot(_softmax(theta + offset), q)) - float(np.dot(_softmax(theta - offset), q))
        ) / (2 * _FINITE_DIFF_EPS)
    return bool(np.max(np.abs(analytic - numeric)) <= tolerance)


class ActorCriticAgent:
    """Tabular actor-critic agent (Algorithm 13.5, Section 13.5).

    Holds policy-preference parameters theta (the actor) and state-value
    estimates V (the critic). Sampling from pi(.|s) via softmax.

    Args:
        n_actions: Number of actions.
        n_states: Number of discrete states.
        seed: Optional seed for reproducible action selection.
        theta: Optional initial preference matrix (n_states, n_actions).
        values: Optional initial state-value vector (n_states,).
    """

    def __init__(self, n_actions: int, n_states: int, lr: float = 0.01,
                 gamma: float = 0.99, seed: int | None = None,
                 theta: np.ndarray | None = None, values: np.ndarray | None = None) -> None:
        """init (reference implementation).


        Args:
            n_actions: See the relevant book section and function body.
            n_states: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            seed: See the relevant book section and function body.
            theta: See the relevant book section and function body.
            values: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.5 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        if n_actions <= 0 or n_states <= 0:
            raise ValueError("n_actions and n_states must be positive")
        if lr <= 0.0:
            raise ValueError("learning rate must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.n_actions = int(n_actions)
        self.n_states = int(n_states)
        self.learning_rate = float(lr)
        self.gamma = float(gamma)
        self.theta = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.V = np.zeros(self.n_states, dtype=np.float64)
        if theta is not None:
            # theta: result of np.asarray(...) (book Chapter 13).
            theta = np.asarray(theta, dtype=np.float64)
            if theta.shape != self.theta.shape:
                raise ValueError("theta has wrong shape")
            self.theta = theta.copy()
        if values is not None:
            # values: state-value estimates V(s) (book Chapter 13).
            values = np.asarray(values, dtype=np.float64)
            if values.shape != self.V.shape:
                raise ValueError("V has wrong shape")
            self.V = values.copy()
        self.rng = np.random.default_rng(seed)

    def action_probs(self, state: int) -> np.ndarray:
        """Return pi(.|state) as a probability vector.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.5 (Chapter 13).

        Args:
            state: See Algorithm 13.5 and the implementation for its role.

        Returns:
            action probs as defined in Algorithm 13.5."""
        # carry out the action probs step (book Chapter 13).
        return _softmax(self.theta[state])

    def select_action(self, state: int = 0) -> int:
        """Sample A ~ pi(.|state).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.5 (Chapter 13).

        Args:
            state: See Algorithm 13.5 and the implementation for its role.

        Returns:
            select action as defined in Algorithm 13.5."""
        # choose an action from the current estimates (book Chapter 13).
        # probs: action probabilities from the current policy (book Chapter 13).
        probs = self.action_probs(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def predict(self, state: int = 0) -> int:
        """Return argmax_a pi(a|state) (greedy/deterministic action).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.5 (Chapter 13).

        Args:
            state: See Algorithm 13.5 and the implementation for its role.

        Returns:
            predict as defined in Algorithm 13.5."""
        # compute the model's prediction for the given input (book Chapter 13).
        # probs: action probabilities from the current policy (book Chapter 13).
        probs = self.action_probs(state)
        return int(np.argmax(probs))

    def episode_update(self, states: list, actions: list, rewards: list) -> None:
        """One-step actor-critic updates along a sampled episode (Section 13.5).

        Book: Sutton & Barto (2nd ed.), Section 13.5 (Chapter 13).

        Args:
            states: See Section 13.5 and the implementation for its role.
            actions: See Section 13.5 and the implementation for its role.
            rewards: See Section 13.5 and the implementation for its role."""
        # run one full episode and apply its updates (book Chapter 13).
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions and rewards must have the same length")
        # Iterate over (t, state): one pass per state (book algorithm).
        for t, state in enumerate(states):
            # action: action selected for the current step (book Chapter 13).
            action = int(actions[t])
            # reward: reward observed after taking the action (book Chapter 13).
            reward = rewards[t]
            # terminal: boolean result of a comparison (book Chapter 13).
            terminal = t == len(states) - 1
            # next v: value chosen conditionally (book Chapter 13).
            next_v = 0.0 if terminal else self.V[states[t + 1]]
            # delta: TD error or update difference for this step (book Chapter 13).
            delta = reward + self.gamma * next_v - self.V[state]
            # self.V[state]: add value computed from its operands into the running value (book ...
            self.V[state] += self.learning_rate * delta
            # grad: value computed from its operands (book Chapter 13).
            grad = np.eye(self.n_actions)[action] - self.action_probs(state)
            # self.theta[state]: add value computed from its operands into the running value ...
            self.theta[state] += self.learning_rate * delta * grad


class _PolicyGradientBase:
    """Shared machinery for the REINFORCE family (Sections 13.3-13.5).

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""

    def __init__(self, n_actions: int, n_states: int, lr: float, gamma: float,
                 seed: Optional[int] = None) -> None:
        """init (reference implementation).


        Args:
            n_actions: See the relevant book section and function body.
            n_states: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        if n_actions <= 0 or n_states <= 0:
            raise ValueError("n_actions and n_states must be positive")
        if lr <= 0.0:
            raise ValueError("learning rate must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.n_actions = int(n_actions)
        self.n_states = int(n_states)
        self.learning_rate = float(lr)
        self.gamma = float(gamma)
        self.theta = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def action_probs(self, state: int) -> np.ndarray:
        """Return pi(.|state) as a probability vector.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            state: See Algorithm 13.2 and the implementation for its role.

        Returns:
            action probs as defined in Algorithm 13.2."""
        # carry out the action probs step (book Chapter 13).
        return _softmax(self.theta[state])

    def log_prob(self, state: int, action: int) -> float:
        """Return ln pi(action|state).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            state: See Algorithm 13.2 and the implementation for its role.
            action: See Algorithm 13.2 and the implementation for its role.

        Returns:
            log prob as defined in Algorithm 13.2."""
        # compute the log-probability of an action under the policy (book Chapter 13).
        return float(np.log(self.action_probs(state)[action] + 1e-12))

    def select_action(self, state: int = 0) -> int:
        """Sample A ~ pi(.|state).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            state: See Algorithm 13.2 and the implementation for its role.

        Returns:
            select action as defined in Algorithm 13.2."""
        # choose an action from the current estimates (book Chapter 13).
        # probs: action probabilities from the current policy (book Chapter 13).
        probs = self.action_probs(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def predict(self, state: int = 0) -> int:
        """Return argmax_a pi(a|state).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            state: See Algorithm 13.2 and the implementation for its role.

        Returns:
            predict as defined in Algorithm 13.2."""
        # compute the model's prediction for the given input (book Chapter 13).
        # probs: action probabilities from the current policy (book Chapter 13).
        probs = self.action_probs(state)
        return int(np.argmax(probs))

    def _episode_returns(self, rewards: list) -> list:
        """episode returns (reference implementation).


        Args:
            rewards: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # internal helper for the episode returns step (book Chapter 13).
        return _discounted_returns([float(r) for r in rewards], self.gamma)

    def _policy_gradient_step(self, state: int, action: int, advantage: float) -> None:
        """policy gradient step (reference implementation).


        Args:
            state: See the relevant book section and function body.
            action: See the relevant book section and function body.
            advantage: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # internal helper for the policy gradient step step (book Chapter 13).
        # grad: value computed from its operands (book Chapter 13).
        grad = np.eye(self.n_actions)[action] - self.action_probs(state)
        # self.theta[state]: add value computed from its operands into the running value (book ...
        self.theta[state] += self.learning_rate * advantage * grad

    def episode_update(self, states: list, actions: list, rewards: list) -> None:
        """REINFORCE update for one episode (Sections 13.3/13.4).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            states: See Algorithm 13.2 and the implementation for its role.
            actions: See Algorithm 13.2 and the implementation for its role.
            rewards: See Algorithm 13.2 and the implementation for its role."""
        # run one full episode and apply its updates (book Chapter 13).
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions and rewards must have the same length")
        # returns: accumulated returns used as update targets (book Chapter 13).
        returns = self._episode_returns(rewards)
        # Iterate over (t, state): one pass per state (book algorithm).
        for t, state in enumerate(states):
            # action: action selected for the current step (book Chapter 13).
            action = int(actions[t])
            self._policy_gradient_step(state, action, returns[t] - self._baseline(state))
            self._update_baseline(state, returns[t])

    def fit(self, env: PolicyEnvironment, episodes: int = 100, steps: int = 5,
            seed: int | None = None) -> list:
        """Train for `episodes` episodes of `steps` decisions; return totals.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            env: See Algorithm 13.2 and the implementation for its role.
            episodes: See Algorithm 13.2 and the implementation for its role.
            steps: See Algorithm 13.2 and the implementation for its role.
            seed: See Algorithm 13.2 and the implementation for its role.

        Returns:
            fit as defined in Algorithm 13.2."""
        # carry out the fit step (book Chapter 13).
        return run_policy_gradient_simulation(self, env, episodes=episodes, steps=steps, seed=seed)

    def _baseline(self, state: int) -> float:
        """baseline (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # internal helper for the baseline step (book Chapter 13).
        return 0.0

    def _update_baseline(self, state: int, return_t: float) -> None:
        """update baseline (reference implementation).


        Args:
            state: See the relevant book section and function body.
            return_t: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # internal helper for the update baseline step (book Chapter 13).
        return None


class PolicyGradientAlgorithm(_PolicyGradientBase):
    """REINFORCE: Monte Carlo policy gradient (Algorithm 13.3). Book Section 13.3.

    Args:
        n_actions: Number of actions.
        n_states: Number of discrete states.
        lr: Step size for the policy parameters.
        gamma: Discount factor.
        baseline: Constant subtracted from returns (REINFORCE-with-baseline
            special case; 0.0 means plain REINFORCE).
        seed: Optional seed for reproducible action sampling.
    """

    def __init__(self, n_actions: int, n_states: int, lr: float = 0.01, gamma: float = 0.99,
                 baseline: float = 0.0, seed: Optional[int] = None) -> None:
        """init (reference implementation).


        Args:
            n_actions: See the relevant book section and function body.
            n_states: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            baseline: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.3 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        super().__init__(n_actions, n_states, lr=lr, gamma=gamma, seed=seed)
        self.baseline = float(baseline)

    def _baseline(self, state: int) -> float:
        """baseline (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.3 (Chapter 13)."""
        # internal helper for the baseline step (book Chapter 13).
        return self.baseline


class PolicyGradientWithBaseline(_PolicyGradientBase):
    """REINFORCE with a learned state-value baseline (Algorithm 13.4). Book Section 13.4.

    The baseline b(s) is a per-state value estimate updated with the same step
    size as the policy; the baseline dimension is kept for compatibility with
    feature-based approximators.

    Args:
        n_actions: Number of actions.
        n_states: Number of discrete states.
        lr: Step size for the policy and value parameters.
        gamma: Discount factor.
        baseline_dim: Feature dimension reserved for the baseline (>= 1).
        seed: Optional seed for reproducible action sampling.
    """

    def __init__(self, n_actions: int, n_states: int, lr: float = 0.01, gamma: float = 0.99,
                 baseline_dim: int = 1, seed: Optional[int] = None) -> None:
        """init (reference implementation).


        Args:
            n_actions: See the relevant book section and function body.
            n_states: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            baseline_dim: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.4 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        super().__init__(n_actions, n_states, lr=lr, gamma=gamma, seed=seed)
        if int(baseline_dim) < 1:
            raise ValueError("baseline_dim must be >= 1")
        self.baseline_dim = int(baseline_dim)
        self.baseline_values = np.zeros(self.n_states, dtype=np.float64)

    def _baseline(self, state: int) -> float:
        """baseline (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.4 (Chapter 13)."""
        # internal helper for the baseline step (book Chapter 13).
        return float(self.baseline_values[state])

    def _update_baseline(self, state: int, return_t: float) -> None:
        """update baseline (reference implementation).


        Args:
            state: See the relevant book section and function body.
            return_t: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.4 (Chapter 13)."""
        # internal helper for the update baseline step (book Chapter 13).
        self.baseline_values[state] += (
            self.learning_rate * (return_t - self.baseline_values[state])
        )


class ActorCriticAlgorithm(_PolicyGradientBase):
    """One-step actor-critic (Algorithm 13.5). Book Section 13.5.

    Combines a policy actor (theta) with a value-function critic (V).

    Args:
        n_states: Number of discrete states.
        n_actions: Number of actions.
        lr: Step size for the actor and critic.
        gamma: Discount factor.
        seed: Optional seed for reproducible action sampling.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.01, gamma: float = 0.99,
                 seed: Optional[int] = None) -> None:
        """init (reference implementation).


        Args:
            n_states: See the relevant book section and function body.
            n_actions: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.5 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        super().__init__(n_actions, n_states, lr=lr, gamma=gamma, seed=seed)
        self.V = np.zeros(self.n_states, dtype=np.float64)

    def episode_update(self, states: list, actions: list, rewards: list) -> None:
        """One-step TD actor-critic updates along the episode.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.5 (Chapter 13).

        Args:
            states: See Algorithm 13.5 and the implementation for its role.
            actions: See Algorithm 13.5 and the implementation for its role.
            rewards: See Algorithm 13.5 and the implementation for its role."""
        # run one full episode and apply its updates (book Chapter 13).
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions and rewards must have the same length")
        # Iterate over (t, state): one pass per state (book algorithm).
        for t, state in enumerate(states):
            # action: action selected for the current step (book Chapter 13).
            action = int(actions[t])
            # reward: reward observed after taking the action (book Chapter 13).
            reward = float(rewards[t])
            # terminal: boolean result of a comparison (book Chapter 13).
            terminal = t == len(states) - 1
            # next v: value chosen conditionally (book Chapter 13).
            next_v = 0.0 if terminal else self.V[states[t + 1]]
            # delta: TD error or update difference for this step (book Chapter 13).
            delta = reward + self.gamma * next_v - self.V[state]
            # self.V[state]: add value computed from its operands into the running value (book ...
            self.V[state] += self.learning_rate * delta
            self._policy_gradient_step(state, action, delta)


class ContinuingActorCritic(ActorCriticAlgorithm):
    """Actor-critic for continuing (average-reward) problems (Section 13.6).

    Uses the differential return delta = R - rho + V(S') - V(S) where rho is the
    running average-reward estimate (book Section 13.6, Alg 13.6-style updates).
    No time-limit/termination is required; the stream can run indefinitely.

    Args:
        n_states: Number of discrete states.
        n_actions: Number of actions.
        lr: Step size for actor and critic updates.
        beta: Step size for the average-reward estimate.
        seed: Optional seed for reproducible action sampling.
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.01,
                 beta: float = 0.05, seed: Optional[int] = None) -> None:
        """init (reference implementation).


        Args:
            n_states: See the relevant book section and function body.
            n_actions: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            beta: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Section 13.6 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        super().__init__(n_states=n_states, n_actions=n_actions, lr=lr,
                         gamma=1.0, seed=seed)
        self.beta = float(beta)
        self.r_hat = 0.0

    def continuing_step(self, state: int, action: int, reward: float, next_state: int) -> float:
        """One continuing-mode step: returns the differential TD error.

        Book: Sutton & Barto (2nd ed.), Section 13.6 (Chapter 13).

        Args:
            state: See Section 13.6 and the implementation for its role.
            action: See Section 13.6 and the implementation for its role.
            reward: See Section 13.6 and the implementation for its role.
            next_state: See Section 13.6 and the implementation for its role.

        Returns:
            continuing step as defined in Section 13.6."""
        # carry out the continuing step step (book Chapter 13).
        # delta: TD error or update difference for this step (book Chapter 13).
        delta = reward - self.r_hat + self.V[next_state] - self.V[state]
        # self.r_hat: add value computed from its operands into the running value (book Chapter ...
        self.r_hat += self.beta * delta
        # self.V[state]: add value computed from its operands into the running value (book ...
        self.V[state] += self.learning_rate * delta
        self._policy_gradient_step(state, action, delta)
        return delta


class GaussianPolicy:
    """Gaussian policy for one-dimensional continuous actions (Algorithm 13.7).

    pi(a|s; theta) = N(a; mu(s), sigma^2) with a tabular mean mu and a learnable
    log-standard-deviation per state. Gradient identities (for unit reward advantage A):
        d ln pi / d mu = (a - mu) / sigma^2
        d ln pi / d log_sigma = ((a - mu)^2 / sigma^2 - 1)

    Args:
        n_states: Number of discrete states.
        lr: Step size for both mean and log-std parameters.
        seed: Optional seed for reproducible action sampling.
    """

    def __init__(self, n_states: int, lr: float = 0.01, seed: Optional[int] = None) -> None:
        """init (reference implementation).


        Args:
            n_states: See the relevant book section and function body.
            lr: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.7 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        if n_states <= 0:
            raise ValueError("n_states must be positive")
        if lr <= 0.0:
            raise ValueError("learning rate must be positive")
        self.n_states = int(n_states)
        self.learning_rate = float(lr)
        self.mu = np.zeros(self.n_states, dtype=np.float64)
        self.log_sigma = np.zeros(self.n_states, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def sample(self, state: int, advantage: float = 0.0) -> float:
        """Sample a ~ pi(.|state).

        Book: Sutton & Barto (2nd ed.), Algorithm 13.7 (Chapter 13).

        Args:
            state: See Algorithm 13.7 and the implementation for its role.
            advantage: See Algorithm 13.7 and the implementation for its role.

        Returns:
            sample as defined in Algorithm 13.7."""
        # draw a sample from the learned distribution (book Chapter 13).
        return float(self.rng.normal(self.mu[state], np.exp(self.log_sigma[state])))

    def update(self, state: int, action: float, advantage: float) -> None:
        """REINFORCE-style update with advantage `advantage`.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.7 (Chapter 13).

        Args:
            state: See Algorithm 13.7 and the implementation for its role.
            action: See Algorithm 13.7 and the implementation for its role.
            advantage: See Algorithm 13.7 and the implementation for its role."""
        # apply the learning update rule to the current estimates (book Chapter 13).
        # sigma2: floating-point conversion (book Chapter 13).
        sigma2 = float(np.exp(2 * self.log_sigma[state]))
        # mu grad: value computed from its operands (book Chapter 13).
        mu_grad = (action - self.mu[state]) / sigma2
        # sigma grad: value computed from its operands (book Chapter 13).
        sigma_grad = (action - self.mu[state]) ** 2 / sigma2 - 1.0
        # self.mu[state]: add value computed from its operands into the running value (book ...
        self.mu[state] += self.learning_rate * advantage * mu_grad
        # self.log_sigma[state]: add value computed from its operands into the running value ...
        self.log_sigma[state] += self.learning_rate * advantage * sigma_grad

    def log_prob(self, state: int, action: float) -> float:
        """Return ln pi(action|state) for a Gaussian.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.7 (Chapter 13).

        Args:
            state: See Algorithm 13.7 and the implementation for its role.
            action: See Algorithm 13.7 and the implementation for its role.

        Returns:
            log prob as defined in Algorithm 13.7."""
        # compute the log-probability of an action under the policy (book Chapter 13).
        # sigma: floating-point conversion (book Chapter 13).
        sigma = float(np.exp(self.log_sigma[state]))
        # var: value computed from its operands (book Chapter 13).
        var = sigma * sigma
        return float(-0.5 * np.log(2 * np.pi * var) - (action - self.mu[state]) ** 2 / (2 * var))


def run_policy_gradient_simulation(
    algorithm,
    env: PolicyEnvironment,
    episodes: int = 50,
    steps: int = 5,
    seed: Optional[int] = None,
) -> list:
    """Train `algorithm` on `env` for `episodes` rounds of `steps` decisions.

    Args:
        algorithm: A policy-gradient algorithm exposing episode_update /
            select_action (REINFORCE family or actor-critic).
        env: A PolicyEnvironment bandit.
        episodes: Number of episodes to run.
        steps: Number of decisions per episode.
        seed: Optional seed (recorded; algorithm owns its RNG).

    Returns:
        List of per-episode total rewards (length == episodes).

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
    if not callable(getattr(algorithm, "episode_update", None)):
        raise TypeError("algorithm must expose episode_update(states, actions, rewards)")
    if not isinstance(env, PolicyEnvironment):
        raise TypeError("env must be a PolicyEnvironment")
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")

    # rewards history: list attribute/state used by the algorithm (book Chapter 13).
    rewards_history: list = []
    state = 0  # bandit: a single decision context
    for _ in range(episodes):
        # (states, actions, rewards): collection of the listed elements (book Chapter 13).
        states, actions, rewards = [], [], []
        # Iterate over _: one pass per time step (book algorithm).
        for _ in range(steps):
            # action: action selected for the current step (book Chapter 13).
            action = algorithm.select_action(state)
            # reward: reward observed after taking the action (book Chapter 13).
            reward = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
        algorithm.episode_update(states, actions, rewards)
        rewards_history.append(float(sum(rewards)))
    return rewards_history


def main(seed: Optional[int] = None) -> list:
    """Demo: REINFORCE-with-baseline learns on the bandit environment.

    Args:
        seed: Optional random seed for reproducibility.

    Returns:
        Per-episode total rewards from the simulation.

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
    # carry out the main step (book Chapter 13).
    # p: probability or transition probability (book Chapter 13).
    p = np.array([0.33, 0.33, 0.34])
    # rewards: per-step reward accumulation buffer (book Chapter 13).
    rewards = np.array([1.0, 2.0, 3.0])
    # env: environment instance being controlled (book Chapter 13).
    env = PolicyEnvironment(n_actions=3, p=p, rewards=rewards, seed=seed)
    # algorithm: result of PolicyGradientWithBaseline(...) (book Chapter 13).
    algorithm = PolicyGradientWithBaseline(
        n_actions=3, n_states=10, lr=0.01, gamma=0.99, baseline_dim=1, seed=seed
    )
    # history: result of run_policy_gradient_simulation(...) (book Chapter 13).
    history = run_policy_gradient_simulation(algorithm, env, episodes=50, steps=5, seed=seed)
    print(f"REINFORCE-with-baseline final episode reward: {history[-1]:.2f}")
    print(f"Average over last 10 episodes: {np.mean(history[-10:]):.2f}")
    return history


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 13)
# ---------------------------------------------------------------------------

def plot_policy_gradient_comparison(
    save_path: str | None = None,
    show: bool = False,
    episodes: int = 200,
    runs: int = 30,
    seed: int = 0,
):
    """REINFORCE vs REINFORCE with baseline learning curves (Sec. 13.3-13.4).

    Sutton & Barto (2nd ed.), Ch. 13 (Policy Gradient Methods), Sec. 13.3
    (REINFORCE, Monte Carlo policy gradient) and Sec. 13.4 (REINFORCE with
    baseline): adding a baseline can reduce variance and speed learning. This
    demo trains both algorithms on a simple stochastic bandit (PolicyEnvironment,
    single state) and plots mean episode reward over episodes, averaged over
    runs. Environment is a demonstration task, not a specific book figure.

    Book: Sutton & Barto (2nd ed.), Section 13.3 (Chapter 13).

    Args:
        save_path: See Section 13.3 and the implementation for its role.
        show: See Section 13.3 and the implementation for its role.
        episodes: See Section 13.3 and the implementation for its role.
        runs: See Section 13.3 and the implementation for its role.
        seed: See Section 13.3 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 13).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_actions = 4
    # env rewards: array built from the supplied values (book Chapter 13).
    env_rewards = np.array([1.0, 0.5, 0.2, 0.0])
    # p: probability or transition probability (book Chapter 13).
    p = np.ones(n_actions) / n_actions
    # curves: averaged learning curves by configuration (book Chapter 13).
    curves = {"REINFORCE": np.zeros(episodes), "REINFORCE+baseline": np.zeros(episodes)}

    # Iterate over r: one pass per index (book algorithm).
    for r in range(runs):
        # rs: value computed from its operands (book Chapter 13).
        rs = seed + r
        # env: environment instance being controlled (book Chapter 13).
        env = PolicyEnvironment(n_actions=n_actions, p=p, rewards=env_rewards, seed=rs)
        # alg1: result of PolicyGradientAlgorithm(...) (book Chapter 13).
        alg1 = PolicyGradientAlgorithm(n_actions=n_actions, n_states=1, lr=0.1, gamma=1.0, seed=rs)
        # alg2: result of PolicyGradientWithBaseline(...) (book Chapter 13).
        alg2 = PolicyGradientWithBaseline(n_actions=n_actions, n_states=1, lr=0.1, gamma=1.0, seed=rs)
        # h1: result of run_policy_gradient_simulation(...) (book Chapter 13).
        h1 = run_policy_gradient_simulation(alg1, env, episodes=episodes, steps=1, seed=rs)
        # h2: result of run_policy_gradient_simulation(...) (book Chapter 13).
        h2 = run_policy_gradient_simulation(alg2, env, episodes=episodes, steps=1, seed=rs)
        # curves['REINFORCE']: add array built from the supplied values into the running value ...
        curves["REINFORCE"] += np.array(h1, dtype=np.float64)
        # curves['REINFORCE+baseline']: add array built from the supplied values into the ...
        curves["REINFORCE+baseline"] += np.array(h2, dtype=np.float64)
    # Iterate over k: each item in curves (book algorithm).
    for k in curves:
        # curves[k]: divide reference to an existing value into the running value (book Chapter ...
        curves[k] /= runs

    # (fig, ax): result of plt.subplots(...) (book Chapter 13).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # x: state feature vector (book Chapter 13).
    x = np.arange(1, episodes + 1)
    ax.plot(x, curves["REINFORCE"], label="REINFORCE")
    ax.plot(x, curves["REINFORCE+baseline"], label="REINFORCE with baseline")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode reward (mean over runs)")
    ax.set_title("S&B (2nd ed.), Sec. 13.3-13.4: Policy gradient methods (demo)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Example 13.1: the short corridor (Section 13.1)
# ---------------------------------------------------------------------------

LEFT = 0
RIGHT = 1
N_CORRIDOR_ACTIONS = 2


class ShortCorridorEnv:
    """Short-corridor MDP (Example 13.1, Section 13.1).

    Three nonterminal states (1, 2, 3) and a terminal state (4). Actions are
    LEFT and RIGHT. Every state looks identical to the agent, and the actions
    are switched in the middle state, so the optimal deterministic policy
    (right, left, right) is not reachable by a state-independent stochastic
    policy. Transitions:

        state 1: RIGHT -> 2, LEFT -> 1
        state 2: RIGHT -> 1, LEFT -> 3     (switched actions)
        state 3: RIGHT -> terminal, LEFT -> 2

    Reward is -1 on every step. With state-independent features the agent can
    only choose one P(RIGHT) = p for all states; the best such p is
    p* = 2 - sqrt(2) ~ 0.586 (see ``short_corridor_optimal_p``).

    Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13)."""

    TERMINAL = 4

    def __init__(self, seed: int | None = None) -> None:
        """init (reference implementation).


        Args:
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        self.rng = np.random.default_rng(seed)
        self.state = 1

    def reset(self) -> int:
        """Start a new episode at state 1.


        Returns:
            reset as defined in Section 13.1.

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13)."""
        # reset internal state for a new episode (book Chapter 13).
        self.state = 1
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take ``action`` (LEFT=0 or RIGHT=1); return (next_state, -1.0, done).

        Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

        Args:
            action: See Section 13.1 and the implementation for its role.

        Returns:
            step as defined in Section 13.1."""
        s = self.state
        if s == 1:
            # nxt: value chosen conditionally (book Chapter 13).
            nxt = 2 if action == RIGHT else 1
        elif s == 2:
            # nxt: value chosen conditionally (book Chapter 13).
            nxt = 1 if action == RIGHT else 3
        else:  # s == 3
            nxt = self.TERMINAL if action == RIGHT else 2
        # done: termination flag for the current episode (book Chapter 13).
        done = nxt == self.TERMINAL
        self.state = self.TERMINAL if done else nxt
        return nxt, -1.0, done


def short_corridor_features(action: int) -> np.ndarray:
    """State-independent one-hot features x(s, a): RIGHT -> [1, 0], LEFT -> [0, 1].

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        action: See Algorithm 13.2 and the implementation for its role.

    Returns:
        short corridor features as defined in Algorithm 13.2."""
    # carry out the short corridor features step (book Chapter 13).
    return np.array([1.0, 0.0]) if action == RIGHT else np.array([0.0, 1.0])


def short_corridor_right_prob(theta: np.ndarray) -> float:
    """Softmax policy: pi(RIGHT | s) = sigmoid(theta[0] - theta[1]).

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        theta: See Algorithm 13.2 and the implementation for its role.

    Returns:
        short corridor right prob as defined in Algorithm 13.2."""
    # carry out the short corridor right prob step (book Chapter 13).
    # theta: result of np.asarray(...) (book Chapter 13).
    theta = np.asarray(theta, dtype=np.float64)
    return float(1.0 / (1.0 + np.exp(-(theta[0] - theta[1]))))


def short_corridor_optimal_p() -> tuple[float, float]:
    """Return (p*, v*(p*)) with p* = 2 - sqrt(2) and v* = -6 - 4*sqrt(2).

    Derived in exercises.md (Exercise 13.1) by maximising the closed-form
    start-state value V1(p) = 2(p - 2) / (p(1 - p)).


    Returns:
        short corridor optimal p as defined in Exercise 13.1.

    Book: Sutton & Barto (2nd ed.), Exercise 13.1 (Chapter 13)."""
    # carry out the short corridor optimal p step (book Chapter 13).
    # p: probability or transition probability (book Chapter 13).
    p = 2.0 - np.sqrt(2.0)
    # v: value computed from its operands (book Chapter 13).
    v = -6.0 - 4.0 * np.sqrt(2.0)
    return float(p), float(v)


def short_corridor_value_gradient(p: float) -> float:
    """Analytic derivative dV1/dp = 2(p^2 - 4p + 2) / (p(1-p))^2, 0 < p < 1.

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        p: See Algorithm 13.2 and the implementation for its role.

    Returns:
        short corridor value gradient as defined in Algorithm 13.2."""
    # carry out the short corridor value gradient step (book Chapter 13).
    # p: probability or transition probability (book Chapter 13).
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # num: value computed from its operands (book Chapter 13).
    num = 2.0 * (p * p - 4.0 * p + 2.0)
    # den: value computed from its operands (book Chapter 13).
    den = (p * (1.0 - p)) ** 2
    return float(num / den)


# Interior clip bounds: V1(p) is only defined for 0 < p < 1, and its gradient
# blows up near the endpoints, so both the exact-ascent iterate and the learned
# policy's evaluation stay a small distance inside the open interval.
_ASCENT_P_CLIP = 1e-4
_EVAL_P_CLIP = 1e-3


def run_short_corridor_value_ascent(start_p: float = 0.5, alpha: float = 0.01,
                                    max_iter: int = 10000, tol: float = 1e-8) -> dict:
    """Exact policy-gradient ascent on V1(p) using the closed-form gradient.

    Because V1(p) is known analytically for the short corridor (Section 13.1),
    each update uses dV1/dp directly rather than Monte-Carlo sampling. Starting
    from the neutral policy (p = 0.5, sec 13.1 example) avoids the steep cliff
    near the boundaries; the exact gradient is tiny at the optimum, so the
    iterate converges deterministically. Clips p into (0, 1) each iteration.

    Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

    Args:
        start_p: See Section 13.1 and the implementation for its role.
        alpha: See Section 13.1 and the implementation for its role.
        max_iter: See Section 13.1 and the implementation for its role.
        tol: See Section 13.1 and the implementation for its role.

    Returns:
        run short corridor value ascent as defined in Section 13.1."""
    # run the experiment and collect the figure data (book Chapter 13).
    # p: probability or transition probability (book Chapter 13).
    p = float(start_p)
    # history: dict attribute/state used by the algorithm (book Chapter 13).
    history: dict = {"iter": [], "p": [], "v1": [], "grad": []}
    # Iterate over it: one pass per index (book algorithm).
    for it in range(max_iter):
        # grad: result of short_corridor_value_gradient(...) (book Chapter 13).
        grad = short_corridor_value_gradient(p)
        # p: probability or transition probability (book Chapter 13).
        p = min(max(p + alpha * grad, _ASCENT_P_CLIP), 1.0 - _ASCENT_P_CLIP)
        if abs(grad) < tol:
            break
        history["iter"].append(it)
        history["p"].append(p)
        history["v1"].append(short_corridor_value(p))
        history["grad"].append(grad)
    return history


def short_corridor_value(p: float) -> float:
    """Closed-form start-state value V1(p) = 2(p - 2) / (p(1 - p)), 0 < p < 1.

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        p: See Algorithm 13.2 and the implementation for its role.

    Returns:
        short corridor value as defined in Algorithm 13.2."""
    # carry out the short corridor value step (book Chapter 13).
    # p: probability or transition probability (book Chapter 13).
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return 2.0 * (p - 2.0) / (p * (1.0 - p))


def simulate_short_corridor(p: float, gamma: float = 1.0, episodes: int = 1000,
                            seed: int = 0, max_steps: int = 100000) -> float:
    """Monte-Carlo estimate of V1 under the aggregate policy P(RIGHT) = p.

    ``max_steps`` bounds each episode as a practical safeguard: for 0 < p < 1 the
    MDP terminates with probability 1, but p very near 0 or 1 makes episodes long.

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        p: See Algorithm 13.2 and the implementation for its role.
        gamma: See Algorithm 13.2 and the implementation for its role.
        episodes: See Algorithm 13.2 and the implementation for its role.
        seed: See Algorithm 13.2 and the implementation for its role.
        max_steps: See Algorithm 13.2 and the implementation for its role.

    Returns:
        simulate short corridor as defined in Algorithm 13.2."""
    # carry out the simulate short corridor step (book Chapter 13).
    # rng: seeded random generator so runs are reproducible (book Chapter 13).
    rng = np.random.default_rng(seed)
    total = 0.0
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(episodes):
        # env: environment instance being controlled (book Chapter 13).
        env = ShortCorridorEnv(seed=int(rng.integers(1 << 30)))
        env.reset()
        ret = 0.0
        discount = 1.0
        # Iterate over _: one pass per index (book algorithm).
        for _ in range(max_steps):
            # action: action selected for the current step (book Chapter 13).
            action = RIGHT if rng.random() < p else LEFT
            # (_, reward, done): result of env.step(...) (book Chapter 13).
            _, reward, done = env.step(action)
            # ret: add value computed from its operands into the running value (book Chapter 13).
            ret += discount * reward
            # discount: multiply reference to an existing value into the running value (book ...
            discount *= gamma
            if done:
                break
        # total: add reference to an existing value into the running value (book Chapter 13).
        total += ret
    return total / episodes


class ShortCorridorREINFORCE:
    """REINFORCE on the short corridor (Example 13.1).

    Learns two action-preference parameters theta so that the state-independent
    probability of RIGHT approaches p* = 2 - sqrt(2). The eligibility for this
    parameterisation is x(s,a) - sum_b pi(b|s) x(s,b).

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""

    def __init__(self, lr: float = 0.01, gamma: float = 1.0,
                 seed: int | None = None) -> None:
        """init (reference implementation).


        Args:
            lr: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # initialize the parameters and state required by the algorithm (book Chapter 13).
        self.learning_rate = float(lr)
        self.gamma = float(gamma)
        self.theta = np.zeros(2, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def right_prob(self) -> float:
        """Current P(RIGHT).


        Returns:
            right prob as defined in Algorithm 13.2.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # carry out the right prob step (book Chapter 13).
        return short_corridor_right_prob(self.theta)

    def select_action(self) -> int:
        """Sample an action from the current softmax policy.


        Returns:
            select action as defined in Algorithm 13.2.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13)."""
        # choose an action from the current estimates (book Chapter 13).
        return RIGHT if self.rng.random() < self.right_prob() else LEFT

    def episode_update(self, actions: list, rewards: list) -> None:
        """One REINFORCE update from a finished episode.

        Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

        Args:
            actions: See Algorithm 13.2 and the implementation for its role.
            rewards: See Algorithm 13.2 and the implementation for its role."""
        # run one full episode and apply its updates (book Chapter 13).
        # returns: accumulated returns used as update targets (book Chapter 13).
        returns = _discounted_returns([float(r) for r in rewards], self.gamma)
        # p: probability or transition probability (book Chapter 13).
        p = self.right_prob()
        # delta: TD error or update difference for this step (book Chapter 13).
        delta = np.zeros(2, dtype=np.float64)
        # Iterate over (t, action): one pass per action (book algorithm).
        for t, action in enumerate(actions):
            if action == RIGHT:
                # grad: array built from the supplied values (book Chapter 13).
                grad = np.array([1.0 - p, -(1.0 - p)])
            else:
                # grad: array built from the supplied values (book Chapter 13).
                grad = np.array([-p, p])
            # delta: add value computed from its operands into the running value (book Chapter 13).
            delta += self.learning_rate * returns[t] * grad
        # self.theta: add reference to an existing value into the running value (book Chapter 13).
        self.theta += delta


def run_short_corridor_reinforce(lr: float = 0.01, gamma: float = 1.0,
                                 episodes: int = 1000, seed: int = 0,
                                 eval_every: int = 50,
                                 max_steps: int = 5000) -> tuple[ShortCorridorREINFORCE, dict]:
    """Train REINFORCE on the short corridor; return (agent, learning history).

    ``max_steps`` is a practical safeguard: the MDP terminates with probability
    1 for any 0 < p < 1, but when the learned p drifts near 0 or 1 episodes can
    become arbitrarily long. Episodes that hit the cap are truncated and the
    partial return is used as-is; this never occurs once p stays away from the
    extremes (checked by the tests).

    Book: Sutton & Barto (2nd ed.), Algorithm 13.2 (Chapter 13).

    Args:
        lr: See Algorithm 13.2 and the implementation for its role.
        gamma: See Algorithm 13.2 and the implementation for its role.
        episodes: See Algorithm 13.2 and the implementation for its role.
        seed: See Algorithm 13.2 and the implementation for its role.
        eval_every: See Algorithm 13.2 and the implementation for its role.
        max_steps: See Algorithm 13.2 and the implementation for its role.

    Returns:
        run short corridor reinforce as defined in Algorithm 13.2."""
    # run the experiment and collect the figure data (book Chapter 13).
    # agent: result of ShortCorridorREINFORCE(...) (book Chapter 13).
    agent = ShortCorridorREINFORCE(lr=lr, gamma=gamma, seed=seed)
    # history: dict attribute/state used by the algorithm (book Chapter 13).
    history: dict = {"episode": [], "p": [], "v1": []}
    # Iterate over ep: one pass per episode (book algorithm).
    for ep in range(episodes):
        # env: environment instance being controlled (book Chapter 13).
        env = ShortCorridorEnv()
        env.reset()
        # (actions, rewards): collection of the listed elements (book Chapter 13).
        actions, rewards = [], []
        # Iterate over _: one pass per index (book algorithm).
        for _ in range(max_steps):
            # action: action selected for the current step (book Chapter 13).
            action = agent.select_action()
            # (_, reward, done): result of env.step(...) (book Chapter 13).
            _, reward, done = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        agent.episode_update(actions, rewards)
        if ep % eval_every == 0 or ep == episodes - 1:
            # p: probability or transition probability (book Chapter 13).
            p = agent.right_prob()
            history["episode"].append(ep)
            history["p"].append(p)
            # clipped: minimum of the values (book Chapter 13).
            clipped = min(max(p, _EVAL_P_CLIP), 1.0 - _EVAL_P_CLIP)
            history["v1"].append(short_corridor_value(clipped))
    return agent, history


# ---------------------------------------------------------------------------
# Figures 13.1 and 13.2: the short corridor (Example 13.1)
# ---------------------------------------------------------------------------

def plot_figure_13_1(save_path: str | None = None, show: bool = False,
                     seed: int = 0, n_mc: int = 2000):
    """Value of the short corridor as a function of P(RIGHT) (Example 13.1).

    Sutton & Barto (2nd ed.), Sec. 13.1 / Example 13.1: with state-independent
    features the agent can only choose one p = P(RIGHT); the start-state value
    is maximised at p* = 2 - sqrt(2) ~ 0.586. Illustrative reproduction; the
    exact book figure identity cannot be verified against the local PDF summary.

    Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

    Args:
        save_path: See Section 13.1 and the implementation for its role.
        show: See Section 13.1 and the implementation for its role.
        seed: See Section 13.1 and the implementation for its role.
        n_mc: See Section 13.1 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 13).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ps: array of linearly spaced values (book Chapter 13).
    ps = np.linspace(0.02, 0.98, 200)
    # vals: collection built by comprehension (book Chapter 13).
    vals = [short_corridor_value(p) for p in ps]
    # (p_star, v_star): result of short_corridor_optimal_p(...) (book Chapter 13).
    p_star, v_star = short_corridor_optimal_p()

    # mc ps: array of linearly spaced values (book Chapter 13).
    mc_ps = np.linspace(0.1, 0.9, 9)
    # mc vals: collection built by comprehension (book Chapter 13).
    mc_vals = [simulate_short_corridor(p, episodes=n_mc, seed=seed) for p in mc_ps]

    # (fig, ax): result of plt.subplots(...) (book Chapter 13).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ps, vals, lw=2.0, color="tab:blue", label=r"closed form $V_1(p)$")
    ax.plot(mc_ps, mc_vals, "o", color="tab:orange", label="Monte-Carlo estimate")
    ax.axvline(p_star, color="tab:red", ls="--", lw=1.0,
               label=rf"$p^* = 2-\sqrt{{2}} \approx {p_star:.3f}$")
    ax.plot([p_star], [v_star], "r*", ms=12)
    ax.set_xlabel(r"$p = P(\mathrm{right})$")
    ax.set_ylabel(r"start-state value $V_1(p)$")
    ax.set_title("S&B (2nd ed.), Sec. 13.1 / Example 13.1: short corridor (illustrative)")
    ax.set_ylim(top=-6.0)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_13_2(save_path: str | None = None, show: bool = False,
                     lr: float = 0.02, episodes: int = 1500, seed: int = 0):
    """REINFORCE learning on the short corridor (Example 13.1).

    Sutton & Barto (2nd ed.), Sec. 13.1 / Example 13.1: policy-gradient learning
    of the state-independent right-probability, which converges toward
    p* = 2 - sqrt(2). Illustrative reproduction; the exact book figure identity
    cannot be verified against the local PDF summary.

    Book: Sutton & Barto (2nd ed.), Section 13.1 (Chapter 13).

    Args:
        save_path: See Section 13.1 and the implementation for its role.
        show: See Section 13.1 and the implementation for its role.
        lr: See Section 13.1 and the implementation for its role.
        episodes: See Section 13.1 and the implementation for its role.
        seed: See Section 13.1 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 13).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (_, history): result of run_short_corridor_reinforce(...) (book Chapter 13).
    _, history = run_short_corridor_reinforce(lr=lr, episodes=episodes, seed=seed)
    # (p_star, _): result of short_corridor_optimal_p(...) (book Chapter 13).
    p_star, _ = short_corridor_optimal_p()

    # (fig, (ax1, ax2)): result of plt.subplots(...) (book Chapter 13).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(history["episode"], history["p"], lw=1.5, color="tab:blue")
    ax1.axhline(p_star, color="tab:red", ls="--", lw=1.0,
                label=rf"$p^* \approx {p_star:.3f}$")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(r"$\pi(\mathrm{right})$")
    ax1.set_title("Learned right-probability")
    ax1.legend()

    ax2.plot(history["episode"], history["v1"], lw=1.5, color="tab:green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(r"$V_1(p)$ (closed form)")
    ax2.set_title("Implied start-state value")
    fig.suptitle("S&B (2nd ed.), Sec. 13.1 / Example 13.1: REINFORCE on the "
                 "short corridor (illustrative)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax1
