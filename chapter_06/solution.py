"""Solution implementation for Chapter 6 - Temporal-Difference Learning.

Reference: Sutton & Barto (2nd Ed), Chapter 6

Implements: Alg 6.1 (Sarsa, on-policy TD control / on-policy prediction),
Alg 6.2 (Q-learning, off-policy TD control),
Alg 6.3 (TD(lambda) prediction),
Exercise 6.1 (random walk, TD(0) vs Monte Carlo comparison).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


from rl_envs.random_walk import RandomWalkEnv, random_walk_true_values


def _uniform_policy(state: str) -> np.ndarray:
    return np.array([0.5, 0.5])


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    n = len(q_values)
    best = np.flatnonzero(q_values == q_values.max())
    probs = np.full(n, epsilon / n, dtype=np.float64)
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def _sample_action(state, policy_fn, rng) -> int:
    probs = np.asarray(policy_fn(state), dtype=np.float64)
    if probs.shape != (len(RandomWalkEnv.ACTIONS),):
        raise ValueError("policy_fn must return a vector of length 2")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("policy probabilities must sum to 1")
    return int(rng.choice(len(probs), p=probs))


def td0_prediction(
    env: RandomWalkEnv,
    policy_fn: Callable | None = None,
    alpha: float = 0.1,
    n_episodes: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Tabular TD(0) prediction (Algorithm 6.1). Book Section 6.1.

    Args:
        env: A RandomWalkEnv (or any env with reset/step).
        policy_fn: Action policy; defaults to the equiprobable policy.
        alpha: Step size.
        n_episodes: Number of episodes to sample.
        gamma: Discount factor.
        seed: Optional seed.

    Returns:
        dict state -> estimated value.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    policy_fn = policy_fn or _uniform_policy
    rng = np.random.default_rng(seed)
    values = {state: 0.0 for state in env.STATES}

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action_index = _sample_action(state, policy_fn, rng)
            reward, next_state, done = env.step(env.ACTIONS[action_index])
            next_v = 0.0 if next_state.endswith("_end") else values[next_state]
            values[state] += alpha * (reward + gamma * next_v - values[state])
            state = next_state if not done else state
    return values


def td_lambda_prediction(
    env: RandomWalkEnv,
    policy_fn: Callable | None = None,
    alpha: float = 0.1,
    lam: float = 0.5,
    n_episodes: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """TD(lambda) prediction with accumulating eligibility traces (Algorithm 6.3).

    Book Section 7 (backward view) / 12.1. lambda = 0 recovers TD(0);
    lambda = 1 recovers MC.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    policy_fn = policy_fn or _uniform_policy
    rng = np.random.default_rng(seed)
    values = {state: 0.0 for state in env.STATES}

    for _ in range(n_episodes):
        traces = {state: 0.0 for state in env.STATES}
        state = env.reset()
        done = False
        while not done:
            action_index = _sample_action(state, policy_fn, rng)
            reward, next_state, done = env.step(env.ACTIONS[action_index])
            next_v = 0.0 if next_state.endswith("_end") else values[next_state]
            delta = reward + gamma * next_v - values[state]
            traces[state] += 1.0
            for s in env.STATES:
                values[s] += alpha * delta * traces[s]
                traces[s] *= gamma * lam
            state = next_state
    return values


def sarsa(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """Sarsa: on-policy TD control (Algorithm 6.1/6.4). Book Section 6.4.

    Returns:
        (policy, Q): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    def behavior_probs(state):
        q = np.array([q_table[(state, a)] for a in env.ACTIONS])
        return _epsilon_greedy_probs(q, epsilon)

    for _ in range(n_episodes):
        state = env.reset()
        a_idx = _sample_action(state, behavior_probs, rng)
        done = False
        while not done:
            action = env.ACTIONS[a_idx]
            reward, next_state, done = env.step(action)
            if done:
                next_q = 0.0
                next_a_idx = a_idx  # unused
            else:
                next_a_idx = _sample_action(next_state, behavior_probs, rng)
                next_q = q_table[(next_state, env.ACTIONS[next_a_idx])]
            q_table[(state, action)] += alpha * (
                reward + gamma * next_q - q_table[(state, action)]
            )
            state, a_idx = next_state if not done else state, next_a_idx

    policy = {
        s: _epsilon_greedy_probs(np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon)
        for s in env.STATES
    }
    return policy, q_table


def q_learning(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """Q-learning: off-policy TD control (Algorithm 6.2). Book Section 6.5.

    Returns:
        (policy, Q): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    def behavior_probs(state):
        q = np.array([q_table[(state, a)] for a in env.ACTIONS])
        return _epsilon_greedy_probs(q, epsilon)

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action_index = _sample_action(state, behavior_probs, rng)
            action = env.ACTIONS[action_index]
            reward, next_state, done = env.step(action)
            best_next = (
                0.0
                if done
                else max(q_table[(next_state, a)] for a in env.ACTIONS)
            )
            q_table[(state, action)] += alpha * (
                reward + gamma * best_next - q_table[(state, action)]
            )
            state = next_state if not done else state

    policy = {
        s: _epsilon_greedy_probs(np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon)
        for s in env.STATES
    }
    return policy, q_table


def mc_prediction_incremental(
    env: RandomWalkEnv,
    policy_fn: Callable | None = None,
    n_episodes: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Incremental (first-visit) Monte Carlo prediction for the random walk.

    Used as the comparison baseline in Exercise 6.1 (the book's sample-average MC).
    """
    policy_fn = policy_fn or _uniform_policy
    rng = np.random.default_rng(seed)
    values = {state: 0.0 for state in env.STATES}
    counts = {state: 0 for state in env.STATES}

    for _ in range(n_episodes):
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action_index = _sample_action(state, policy_fn, rng)
            reward, next_state, done = env.step(env.ACTIONS[action_index])
            trajectory.append((state, reward))
            state = next_state
        returns = [0.0] * len(trajectory)
        running = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            running = trajectory[t][1] + gamma * running
            returns[t] = running
        seen: set = set()
        for t, (state, _) in enumerate(trajectory):
            if state in seen:
                continue
            seen.add(state)
            counts[state] += 1
            values[state] += (returns[t] - values[state]) / counts[state]
    return values


def _rms_error(values: dict, true_values: dict) -> float:
    states = list(true_values)
    errs = [values[s] - true_values[s] for s in states]
    return float(np.sqrt(np.mean(np.square(errs))))


def run_random_walk_comparison(
    repeats: int = 100,
    episodes: int = 100,
    alpha: float = 0.1,
    seed: int | None = 0,
) -> dict:
    """Exercise 6.1 / Example 6.2: TD(0) vs constant-alpha MC on the random walk.

    Book Section 6.2 (Example 6.2). Both methods share the step size alpha and
    start from V(s) = 0.5 for all s. RMS error between the learned and true
    value function, averaged over the five states.

    Returns:
        dict with 'td0' and 'mc' arrays of shape (episodes,): the RMS error
        averaged over `repeats` independent runs.
    """
    true_values = random_walk_true_values()
    rng_seed = seed
    td_errors = np.zeros((repeats, episodes))
    mc_errors = np.zeros((repeats, episodes))

    for repeat in range(repeats):
        repeat_seed = None if rng_seed is None else rng_seed + repeat
        td_values = {state: 0.5 for state in RandomWalkEnv.STATES}
        mc_values = {state: 0.5 for state in RandomWalkEnv.STATES}
        env = RandomWalkEnv(seed=repeat_seed)
        rng = np.random.default_rng(repeat_seed)

        for ep in range(episodes):
            state = env.reset()
            done = False
            trajectory = []
            while not done:
                action_index = int(rng.integers(0, 2))
                reward, next_state, done = env.step(env.ACTIONS[action_index])
                trajectory.append((state, reward))
                old = td_values[state]
                next_v = 0.0 if next_state.endswith("_end") else td_values[next_state]
                td_values[state] = old + alpha * (reward + 1.0 * next_v - old)
                state = next_state
            # Constant-alpha MC: decrement each visited state after the episode.
            returns = [0.0] * len(trajectory)
            running = 0.0
            for t in range(len(trajectory) - 1, -1, -1):
                running = trajectory[t][1] + running
                returns[t] = running
            seen: set = set()
            for t, (state, _) in enumerate(trajectory):
                if state in seen:
                    continue
                seen.add(state)
                mc_values[state] += alpha * (returns[t] - mc_values[state])

            td_errors[repeat, ep] = _rms_error(td_values, true_values)
            mc_errors[repeat, ep] = _rms_error(mc_values, true_values)

    return {"td0": td_errors.mean(axis=0).tolist(), "mc": mc_errors.mean(axis=0).tolist()}


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 6)
# ---------------------------------------------------------------------------

def plot_figure_6_2(
    save_path: str | None = None,
    show: bool = False,
    repeats: int = 100,
    episodes: int = 100,
    alpha: float = 0.1,
    seed: int = 0,
):
    """TD(0) vs constant-alpha MC on the random walk (Example 6.2, Fig. 6.2).

    Sutton & Barto (2nd ed.), Sec. 6.2, Fig. 6.2: RMS error in value estimates
    over episodes for TD(0) and constant-alpha MC on the 5-state random walk
    (Example 6.2, states A-E, start C, right terminal +1). Both methods start
    from V(s)=0.5 and use step size alpha. Curves are averaged over `repeats`
    independent runs; TD(0) typically converges faster.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_random_walk_comparison(repeats=repeats, episodes=episodes, alpha=alpha, seed=seed)
    x = np.arange(1, episodes + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, results["td0"], label=f"TD(0), alpha={alpha}")
    ax.plot(x, results["mc"], label=f"MC, alpha={alpha}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error (averaged over states)")
    ax.set_title("S&B (2nd ed.), Sec. 6.2, Fig. 6.2: Random walk TD vs MC")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
