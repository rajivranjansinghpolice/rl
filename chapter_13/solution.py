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
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _discounted_returns(rewards: list, gamma: float) -> list:
    """G_t = sum_{k=t+1}^{T} gamma^{k-t-1} * R_k, computed backwards in place."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def soft_argmax_row(preferences: np.ndarray) -> np.ndarray:
    """Softmax action probabilities pi(.|s) from preference vector h(s, .)."""
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
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        probs = np.asarray(policy, dtype=np.float64) if policy is not None else self.p
        if not np.isclose(probs.sum(), 1.0, atol=1e-6):
            raise ValueError("policy must sum to 1")
        action = rng.choice(self.n_actions, p=probs)
        return int(action)

    def step(self, action: int) -> float:
        """Play `action`; return its reward and increment the action count."""
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        self.action_count[action] += 1
        return float(self.rewards[action])

    def get_action_count(self, action: int) -> float:
        """Return the number of times `action` was played."""
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        return float(self.action_count[action])


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
    theta = np.asarray(preferences, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    if theta.shape != q.shape:
        raise ValueError("preferences and q_values must have the same shape")

    probs = _softmax(theta)
    analytic = np.array(
        [q[i] * probs[i] - probs[i] * float(np.dot(q, probs)) for i in range(len(q))]
    )
    numeric = np.zeros_like(theta)
    eps = 1e-6
    for i in range(len(theta)):
        offset = np.zeros_like(theta)
        offset[i] = eps
        numeric[i] = (
            float(np.dot(_softmax(theta + offset), q)) - float(np.dot(_softmax(theta - offset), q))
        ) / (2 * eps)
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
            theta = np.asarray(theta, dtype=np.float64)
            if theta.shape != self.theta.shape:
                raise ValueError("theta has wrong shape")
            self.theta = theta.copy()
        if values is not None:
            values = np.asarray(values, dtype=np.float64)
            if values.shape != self.V.shape:
                raise ValueError("V has wrong shape")
            self.V = values.copy()
        self.rng = np.random.default_rng(seed)

    def action_probs(self, state: int) -> np.ndarray:
        """Return pi(.|state) as a probability vector."""
        return _softmax(self.theta[state])

    def select_action(self, state: int = 0) -> int:
        """Sample A ~ pi(.|state)."""
        probs = self.action_probs(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def predict(self, state: int = 0) -> int:
        """Return argmax_a pi(a|state) (greedy/deterministic action)."""
        probs = self.action_probs(state)
        return int(np.argmax(probs))

    def episode_update(self, states: list, actions: list, rewards: list) -> None:
        """One-step actor-critic updates along a sampled episode (Section 13.5)."""
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions and rewards must have the same length")
        for t, state in enumerate(states):
            action = int(actions[t])
            reward = rewards[t]
            terminal = t == len(states) - 1
            next_v = 0.0 if terminal else self.V[states[t + 1]]
            delta = reward + self.gamma * next_v - self.V[state]
            self.V[state] += self.learning_rate * delta
            grad = np.eye(self.n_actions)[action] - self.action_probs(state)
            self.theta[state] += self.learning_rate * delta * grad


class _PolicyGradientBase:
    """Shared machinery for the REINFORCE family (Sections 13.3-13.5)."""

    def __init__(self, n_actions: int, n_states: int, lr: float, gamma: float,
                 seed: Optional[int] = None) -> None:
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
        """Return pi(.|state) as a probability vector."""
        return _softmax(self.theta[state])

    def log_prob(self, state: int, action: int) -> float:
        """Return ln pi(action|state)."""
        return float(np.log(self.action_probs(state)[action] + 1e-12))

    def select_action(self, state: int = 0) -> int:
        """Sample A ~ pi(.|state)."""
        probs = self.action_probs(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def predict(self, state: int = 0) -> int:
        """Return argmax_a pi(a|state)."""
        probs = self.action_probs(state)
        return int(np.argmax(probs))

    def _episode_returns(self, rewards: list) -> list:
        return _discounted_returns([float(r) for r in rewards], self.gamma)

    def _policy_gradient_step(self, state: int, action: int, advantage: float) -> None:
        grad = np.eye(self.n_actions)[action] - self.action_probs(state)
        self.theta[state] += self.learning_rate * advantage * grad

    def episode_update(self, states: list, actions: list, rewards: list) -> None:
        """REINFORCE update for one episode (Sections 13.3/13.4)."""
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions and rewards must have the same length")
        returns = self._episode_returns(rewards)
        for t, state in enumerate(states):
            action = int(actions[t])
            self._policy_gradient_step(state, action, returns[t] - self._baseline(state))
            self._update_baseline(state, returns[t])

    def fit(self, env: PolicyEnvironment, episodes: int = 100, steps: int = 5,
            seed: int | None = None) -> list:
        """Train for `episodes` episodes of `steps` decisions; return totals."""
        return run_policy_gradient_simulation(self, env, episodes=episodes, steps=steps, seed=seed)

    def _baseline(self, state: int) -> float:
        return 0.0

    def _update_baseline(self, state: int, return_t: float) -> None:
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
        super().__init__(n_actions, n_states, lr=lr, gamma=gamma, seed=seed)
        self.baseline = float(baseline)

    def _baseline(self, state: int) -> float:
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
        super().__init__(n_actions, n_states, lr=lr, gamma=gamma, seed=seed)
        if int(baseline_dim) < 1:
            raise ValueError("baseline_dim must be >= 1")
        self.baseline_dim = int(baseline_dim)
        self.baseline_values = np.zeros(self.n_states, dtype=np.float64)

    def _baseline(self, state: int) -> float:
        return float(self.baseline_values[state])

    def _update_baseline(self, state: int, return_t: float) -> None:
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
        super().__init__(n_actions, n_states, lr=lr, gamma=gamma, seed=seed)
        self.V = np.zeros(self.n_states, dtype=np.float64)

    def episode_update(self, states: list, actions: list, rewards: list) -> None:
        """One-step TD actor-critic updates along the episode."""
        if not (len(states) == len(actions) == len(rewards)):
            raise ValueError("states, actions and rewards must have the same length")
        for t, state in enumerate(states):
            action = int(actions[t])
            reward = float(rewards[t])
            terminal = t == len(states) - 1
            next_v = 0.0 if terminal else self.V[states[t + 1]]
            delta = reward + self.gamma * next_v - self.V[state]
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
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        super().__init__(n_states=n_states, n_actions=n_actions, lr=lr,
                         gamma=1.0, seed=seed)
        self.beta = float(beta)
        self.r_hat = 0.0

    def continuing_step(self, state: int, action: int, reward: float, next_state: int) -> float:
        """One continuing-mode step: returns the differential TD error."""
        delta = reward - self.r_hat + self.V[next_state] - self.V[state]
        self.r_hat += self.beta * delta
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
        """Sample a ~ pi(.|state)."""
        return float(self.rng.normal(self.mu[state], np.exp(self.log_sigma[state])))

    def update(self, state: int, action: float, advantage: float) -> None:
        """REINFORCE-style update with advantage `advantage`."""
        sigma2 = float(np.exp(2 * self.log_sigma[state]))
        mu_grad = (action - self.mu[state]) / sigma2
        sigma_grad = (action - self.mu[state]) ** 2 / sigma2 - 1.0
        self.mu[state] += self.learning_rate * advantage * mu_grad
        self.log_sigma[state] += self.learning_rate * advantage * sigma_grad

    def log_prob(self, state: int, action: float) -> float:
        """Return ln pi(action|state) for a Gaussian."""
        sigma = float(np.exp(self.log_sigma[state]))
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
    """
    if not callable(getattr(algorithm, "episode_update", None)):
        raise TypeError("algorithm must expose episode_update(states, actions, rewards)")
    if not isinstance(env, PolicyEnvironment):
        raise TypeError("env must be a PolicyEnvironment")
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")

    rewards_history: list = []
    state = 0  # bandit: a single decision context
    for _ in range(episodes):
        states, actions, rewards = [], [], []
        for _ in range(steps):
            action = algorithm.select_action(state)
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
    """
    p = np.array([0.33, 0.33, 0.34])
    rewards = np.array([1.0, 2.0, 3.0])
    env = PolicyEnvironment(n_actions=3, p=p, rewards=rewards, seed=seed)
    algorithm = PolicyGradientWithBaseline(
        n_actions=3, n_states=10, lr=0.01, gamma=0.99, baseline_dim=1, seed=seed
    )
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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_actions = 4
    env_rewards = np.array([1.0, 0.5, 0.2, 0.0])
    p = np.ones(n_actions) / n_actions
    curves = {"REINFORCE": np.zeros(episodes), "REINFORCE+baseline": np.zeros(episodes)}

    for r in range(runs):
        rs = seed + r
        env = PolicyEnvironment(n_actions=n_actions, p=p, rewards=env_rewards, seed=rs)
        alg1 = PolicyGradientAlgorithm(n_actions=n_actions, n_states=1, lr=0.1, gamma=1.0, seed=rs)
        alg2 = PolicyGradientWithBaseline(n_actions=n_actions, n_states=1, lr=0.1, gamma=1.0, seed=rs)
        h1 = run_policy_gradient_simulation(alg1, env, episodes=episodes, steps=1, seed=rs)
        h2 = run_policy_gradient_simulation(alg2, env, episodes=episodes, steps=1, seed=rs)
        curves["REINFORCE"] += np.array(h1, dtype=np.float64)
        curves["REINFORCE+baseline"] += np.array(h2, dtype=np.float64)
    for k in curves:
        curves[k] /= runs

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
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
    """

    TERMINAL = 4

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.state = 1

    def reset(self) -> int:
        """Start a new episode at state 1."""
        self.state = 1
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take ``action`` (LEFT=0 or RIGHT=1); return (next_state, -1.0, done)."""
        s = self.state
        if s == 1:
            nxt = 2 if action == RIGHT else 1
        elif s == 2:
            nxt = 1 if action == RIGHT else 3
        else:  # s == 3
            nxt = self.TERMINAL if action == RIGHT else 2
        done = nxt == self.TERMINAL
        self.state = self.TERMINAL if done else nxt
        return nxt, -1.0, done


def short_corridor_features(action: int) -> np.ndarray:
    """State-independent one-hot features x(s, a): RIGHT -> [1, 0], LEFT -> [0, 1]."""
    return np.array([1.0, 0.0]) if action == RIGHT else np.array([0.0, 1.0])


def short_corridor_right_prob(theta: np.ndarray) -> float:
    """Softmax policy: pi(RIGHT | s) = sigmoid(theta[0] - theta[1])."""
    theta = np.asarray(theta, dtype=np.float64)
    return float(1.0 / (1.0 + np.exp(-(theta[0] - theta[1]))))


def short_corridor_optimal_p() -> tuple[float, float]:
    """Return (p*, v*(p*)) with p* = 2 - sqrt(2) and v* = -6 - 4*sqrt(2).

    Derived in exercises.md (Exercise 13.1) by maximising the closed-form
    start-state value V1(p) = 2(p - 2) / (p(1 - p)).
    """
    p = 2.0 - np.sqrt(2.0)
    v = -6.0 - 4.0 * np.sqrt(2.0)
    return float(p), float(v)


def short_corridor_value_gradient(p: float) -> float:
    """Analytic derivative dV1/dp = 2(p^2 - 4p + 2) / (p(1-p))^2, 0 < p < 1."""
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    num = 2.0 * (p * p - 4.0 * p + 2.0)
    den = (p * (1.0 - p)) ** 2
    return float(num / den)


def run_short_corridor_value_ascent(start_p: float = 0.5, alpha: float = 0.01,
                                    max_iter: int = 10000, tol: float = 1e-8) -> dict:
    """Exact policy-gradient ascent on V1(p) using the closed-form gradient.

    Because V1(p) is known analytically for the short corridor (Section 13.1),
    each update uses dV1/dp directly rather than Monte-Carlo sampling. Starting
    from the neutral policy (p = 0.5, sec 13.1 example) avoids the steep cliff
    near the boundaries; the exact gradient is tiny at the optimum, so the
    iterate converges deterministically. Clips p into (0, 1) each iteration.
    """
    p = float(start_p)
    history: dict = {"iter": [], "p": [], "v1": [], "grad": []}
    for it in range(max_iter):
        grad = short_corridor_value_gradient(p)
        p = min(max(p + alpha * grad, 1e-4), 1.0 - 1e-4)
        if abs(grad) < tol:
            break
        history["iter"].append(it)
        history["p"].append(p)
        history["v1"].append(short_corridor_value(p))
        history["grad"].append(grad)
    return history


def short_corridor_value(p: float) -> float:
    """Closed-form start-state value V1(p) = 2(p - 2) / (p(1 - p)), 0 < p < 1."""
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return 2.0 * (p - 2.0) / (p * (1.0 - p))


def simulate_short_corridor(p: float, gamma: float = 1.0, episodes: int = 1000,
                            seed: int = 0, max_steps: int = 100000) -> float:
    """Monte-Carlo estimate of V1 under the aggregate policy P(RIGHT) = p.

    ``max_steps`` bounds each episode as a practical safeguard: for 0 < p < 1 the
    MDP terminates with probability 1, but p very near 0 or 1 makes episodes long.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(episodes):
        env = ShortCorridorEnv(seed=int(rng.integers(1 << 30)))
        env.reset()
        ret = 0.0
        discount = 1.0
        for _ in range(max_steps):
            action = RIGHT if rng.random() < p else LEFT
            _, reward, done = env.step(action)
            ret += discount * reward
            discount *= gamma
            if done:
                break
        total += ret
    return total / episodes


class ShortCorridorREINFORCE:
    """REINFORCE on the short corridor (Example 13.1).

    Learns two action-preference parameters theta so that the state-independent
    probability of RIGHT approaches p* = 2 - sqrt(2). The eligibility for this
    parameterisation is x(s,a) - sum_b pi(b|s) x(s,b).
    """

    def __init__(self, lr: float = 0.01, gamma: float = 1.0,
                 seed: int | None = None) -> None:
        self.learning_rate = float(lr)
        self.gamma = float(gamma)
        self.theta = np.zeros(2, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def right_prob(self) -> float:
        """Current P(RIGHT)."""
        return short_corridor_right_prob(self.theta)

    def select_action(self) -> int:
        """Sample an action from the current softmax policy."""
        return RIGHT if self.rng.random() < self.right_prob() else LEFT

    def episode_update(self, actions: list, rewards: list) -> None:
        """One REINFORCE update from a finished episode."""
        returns = _discounted_returns([float(r) for r in rewards], self.gamma)
        p = self.right_prob()
        delta = np.zeros(2, dtype=np.float64)
        for t, action in enumerate(actions):
            if action == RIGHT:
                grad = np.array([1.0 - p, -(1.0 - p)])
            else:
                grad = np.array([-p, p])
            delta += self.learning_rate * returns[t] * grad
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
    """
    agent = ShortCorridorREINFORCE(lr=lr, gamma=gamma, seed=seed)
    history: dict = {"episode": [], "p": [], "v1": []}
    for ep in range(episodes):
        env = ShortCorridorEnv()
        env.reset()
        actions, rewards = [], []
        for _ in range(max_steps):
            action = agent.select_action()
            _, reward, done = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        agent.episode_update(actions, rewards)
        if ep % eval_every == 0 or ep == episodes - 1:
            p = agent.right_prob()
            history["episode"].append(ep)
            history["p"].append(p)
            history["v1"].append(short_corridor_value(min(max(p, 1e-3), 1.0 - 1e-3)))
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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = np.linspace(0.02, 0.98, 200)
    vals = [short_corridor_value(p) for p in ps]
    p_star, v_star = short_corridor_optimal_p()

    mc_ps = np.linspace(0.1, 0.9, 9)
    mc_vals = [simulate_short_corridor(p, episodes=n_mc, seed=seed) for p in mc_ps]

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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, history = run_short_corridor_reinforce(lr=lr, episodes=episodes, seed=seed)
    p_star, _ = short_corridor_optimal_p()

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
