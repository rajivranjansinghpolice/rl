"""Solution implementation for Chapter 2 - Bandit Algorithms.

Reference: Sutton & Barto (2nd Ed), Chapter 2

Implements: Alg 2.1 (k-armed bandit), Alg 2.2 (action-value est.),
Alg 2.3 (epsilon-greedy), Alg 2.4 (UCB), Alg 2.5 (gradient bandit),
Ex 2.3 (non-stationary), Ex 2.4 (optimistic init), Ex 2.5 (UCB unc.),
Ex 2.6 (contextual bandits).
"""

from __future__ import annotations

import numpy as np


class BanditProblem:
    """k-armed bandit problem (Algorithm 2.1). Book Section 2.2.

    The environment has k actions, each with true expected reward q*(a).
    Reward for action a is drawn from N(q*(a), 1).
    """

    def __init__(self, k: int = 10, seed: int | None = None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star: np.ndarray = self.rng.normal(0.0, 1.0, k)

    def step(self, action: int) -> float:
        return float(self.rng.normal(self.q_star[action], 1.0))

    def best_action(self) -> int:
        return int(np.argmax(self.q_star))


class ActionValueEstimator:
    """Action-value estimation using sample-average (Algorithm 2.2). Book Sec 2.3.

    Maintains estimates Q(a) and counts N(a) for each action.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.Q: np.ndarray = np.zeros(k)
        self.N: np.ndarray = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng()

    def select_action(self) -> int:
        best_actions = np.flatnonzero(self.Q == self.Q.max())
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])


def select_epsilon_greedy(
    Q: np.ndarray, epsilon: float, rng: np.random.Generator | None = None
) -> int:
    """Select action using epsilon-greedy policy (Algorithm 2.3). Book Sec 2.3.

    Args:
        Q: Array of action values.
        epsilon: Exploration probability in [0, 1].
        rng: Random number generator.

    Returns:
        Selected action index.

    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    best_actions = np.flatnonzero(Q == Q.max())
    return int(rng.choice(best_actions))


def select_ucb(Q: np.ndarray, N: np.ndarray, t: int, c: float) -> int:
    """Select action using upper-confidence bound (Algorithm 2.4). Book Sec 2.4.

    Args:
        Q: Array of action value estimates.
        N: Array of action visit counts.
        t: Current time step (>= 1).
        c: Exploration parameter (> 0).

    Returns:
        Selected action index.

    """
    untried = np.flatnonzero(N == 0)
    if len(untried) > 0:
        return int(untried[0])
    ucb = Q + c * np.sqrt(np.log(t) / N)
    best_actions = np.flatnonzero(ucb == ucb.max())
    return int(np.random.default_rng().choice(best_actions))


def select_softmax(preferences: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Select action using softmax policy (Algorithm 2.5). Book Sec 2.5.

    Args:
        preferences: Action preference values H(a).
        rng: Random number generator.

    Returns:
        Selected action index.

    """
    if rng is None:
        rng = np.random.default_rng()
    shifted = preferences - preferences.max()
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()
    return int(rng.choice(len(probs), p=probs))


class GradientBanditAgent:
    """Gradient bandit algorithm (Algorithm 2.5). Book Section 2.5.

    Maintains action preferences H(a) and updates them using the gradient
    of expected reward.
    """

    def __init__(self, k: int, alpha: float = 0.1) -> None:
        self.k = k
        self.alpha = alpha
        self.H: np.ndarray = np.zeros(k)
        self.r_bar: float = 0.0
        self.rng = np.random.default_rng()

    def select_action(self) -> int:
        return select_softmax(self.H, self.rng)

    def update(self, action: int, reward: float) -> None:
        probs = np.zeros(self.k)
        shifted = self.H - self.H.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        delta = reward - self.r_bar
        self.H[action] += self.alpha * delta * (1.0 - probs[action])
        for a in range(self.k):
            if a != action:
                self.H[a] -= self.alpha * delta * probs[a]
        self.r_bar += self.alpha * (reward - self.r_bar)


def simulate_nonstationary(
    k: int = 10,
    steps: int = 1000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    true_reward_change: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate non-stationary bandit with constant step-size (Exercise 2.3).

    Book Exercise 2.3: Non-stationary Bandits.

    Args:
        k: Number of actions.
        steps: Number of steps to simulate.
        epsilon: Epsilon for epsilon-greedy.
        alpha: Constant step-size parameter.
        true_reward_change: Std dev of true reward changes per step.

    Returns:
        Tuple of (action_values, average_rewards).

    """
    rng = np.random.default_rng()
    q_star = rng.normal(0.0, 1.0, k)
    Q = np.zeros(k)
    rewards = np.zeros(steps)

    for t in range(steps):
        q_star += rng.normal(0.0, true_reward_change, k)
        if rng.random() < epsilon:
            action = int(rng.integers(k))
        else:
            action = int(np.argmax(Q))
        reward = float(rng.normal(q_star[action], 1.0))
        rewards[t] = reward
        Q[action] += alpha * (reward - Q[action])

    return Q, rewards


def simulate_optimistic_initial_values(
    k: int = 10,
    steps: int = 1000,
    initial_value: float = 5.0,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Simulate bandit with optimistic initial values (Exercise 2.4).

    Book Exercise 2.4: Optimistic Initial Values.

    Args:
        k: Number of actions.
        steps: Number of steps.
        initial_value: Optimistic initial Q value.
        epsilon: Epsilon (default 0 for pure greedy).

    Returns:
        Tuple of (final_Q, best_action_selection_rate_over_last_100).

    """
    rng = np.random.default_rng()
    bandit = BanditProblem(k, seed=0)
    Q = np.full(k, initial_value)
    N = np.zeros(k, dtype=int)
    best_counts = np.zeros(steps)

    for t in range(steps):
        if rng.random() < epsilon:
            action = int(rng.integers(k))
        else:
            action = int(np.argmax(Q))
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])
        if action == bandit.best_action():
            best_counts[t] = 1.0

    rate = best_counts[-100:].mean()
    return Q, rate


def simulate_ucb_uncertainty(
    k: int = 10,
    steps: int = 1000,
    c: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Simulate UCB with uncertainty parameterization (Exercise 2.5).

    Book Exercise 2.5: UCB with Uncertainty.

    Args:
        k: Number of actions.
        steps: Number of steps.
        c: Exploration parameter.
        seed: Random seed.

    Returns:
        Tuple of (final_Q, best_action_selection_rate).

    """
    np.random.default_rng(seed)
    bandit = BanditProblem(k, seed=seed)
    Q = np.zeros(k)
    N = np.zeros(k, dtype=int)
    best_counts = np.zeros(steps)

    for t in range(1, steps + 1):
        action = select_ucb(Q, N, t, c)
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])
        if action == bandit.best_action():
            best_counts[t - 1] = 1.0

    rate = best_counts[-100:].mean()
    return Q, rate


class ContextualBandit:
    """Contextual bandit algorithm (Exercise 2.6). Book Exercise 2.6.

    Maintains action preferences indexed by (context, action) pairs.
    """

    def __init__(self, num_contexts: int, k: int, alpha: float = 0.1) -> None:
        self.num_contexts = num_contexts
        self.k = k
        self.alpha = alpha
        self.H: np.ndarray = np.zeros((num_contexts, k))
        self.r_bar: np.ndarray = np.zeros(num_contexts)
        self.rng = np.random.default_rng()

    def select_action(self, context: int) -> int:
        return select_softmax(self.H[context], self.rng)

    def update(self, context: int, action: int, reward: float) -> None:
        probs = np.zeros(self.k)
        shifted = self.H[context] - self.H[context].max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        delta = reward - self.r_bar[context]
        self.H[context, action] += self.alpha * delta * (1.0 - probs[action])
        for a in range(self.k):
            if a != action:
                self.H[context, a] -= self.alpha * delta * probs[a]
        self.r_bar[context] += self.alpha * (reward - self.r_bar[context])
