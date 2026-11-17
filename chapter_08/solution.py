"""Solution implementation for Chapter 8 - Eligibility Traces.

Reference: Sutton & Barto (2nd Ed), Chapter 12 (Eligibility Traces);
the chapter-numbering follows this repository's task list
(algorithm_8.1..8.3).

Implements:
- algorithm_8.1 - TD(lambda) with replacing traces:
    state-value prediction (Eq. 12.17, Section 12.2/12.7) and
    Sarsa(lambda) control with replacing traces (Section 12.7, box on p. 250).
- algorithm_8.2 - TD(lambda) with clearing traces:
    Sarsa(lambda) control with replacing traces plus clearing of the traces
    of the non-selected actions for the visited state (Figure 12.11,
    Section 12.7; van Hasselt et al.'s "clearing" variant).
- algorithm_8.3 - True online TD(lambda) (Section 12.5, Eq. 12.15-12.16,
    box on p. 245), tabular implementation with one-hot features.
- State-value TD(lambda) prediction with accumulating or replacing traces
    (Section 12.2, box on p. 240, Eq. 12.5).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from chapter_06.solution import RandomWalkEnv


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


def td_lambda_prediction(
    env: RandomWalkEnv,
    policy_fn: Callable | None = None,
    alpha: float = 0.1,
    lam: float = 0.5,
    n_episodes: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
    trace: str = "accumulating",
) -> dict:
    """State-value TD(lambda) prediction (Section 12.2, box on p. 240).

    Semi-gradient TD(lambda) with accumulating traces (Eq. 12.5) or replacing
    traces (tabular special case of Eq. 12.17, Section 12.5/12.7).

    Args:
        env: A RandomWalkEnv (or compatible env with STATES/ACTIONS).
        policy_fn: Behavior policy; defaults to the equiprobable policy.
        alpha: Step size.
        lam: Trace decay parameter lambda in [0, 1]. lambda = 0 is TD(0);
            lambda = 1 is TD(1), Monte-Carlo-like.
        n_episodes: Number of episodes to sample.
        gamma: Discount factor.
        seed: Optional seed.
        trace: "accumulating" (z += 1) or "replacing" (z <- 1 for the visited
            state). See Eq. 12.5 and Eq. 12.17 of the book.

    Returns:
        dict state -> estimated value.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if trace not in ("accumulating", "replacing"):
        raise ValueError("trace must be 'accumulating' or 'replacing'")
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
            next_v = 0.0 if next_state not in values else values[next_state]
            delta = reward + gamma * next_v - values[state]
            if trace == "accumulating":
                traces[state] += 1.0
            else:  # replacing: z <- 1 for the visited state (Eq. 12.17)
                traces[state] = 1.0
            for s in env.STATES:
                values[s] += alpha * delta * traces[s]
                traces[s] *= gamma * lam
            state = next_state if not done else state
    return values


def _sarsa_lambda(
    env: RandomWalkEnv,
    n_episodes: int,
    alpha: float,
    lam: float,
    epsilon: float,
    gamma: float,
    seed: int | None,
    trace: str,
    clearing: bool,
) -> tuple[dict, dict]:
    """Shared backward-view Sarsa(lambda) implementation (Section 12.7).

    Follows the box "Sarsa(lambda) with binary features" (p. 250):
    TD error (Eq. 12.22), trace bump, parameter update (Eq. 12.7),
    then trace decay by gamma * lambda. For replacing traces the visited
    pair's trace is set to 1; for clearing traces the traces of the
    non-selected actions for the visited state are also set to 0
    (Figure 12.11, Section 12.7).
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    rng = np.random.default_rng(seed)
    pairs = [(s, a) for s in env.STATES for a in env.ACTIONS]
    q_table = {pair: 0.0 for pair in pairs}

    def behavior_probs(state):
        q = np.array([q_table[(state, a)] for a in env.ACTIONS])
        return _epsilon_greedy_probs(q, epsilon)

    for _ in range(n_episodes):
        traces = {pair: 0.0 for pair in pairs}
        state = env.reset()
        a_idx = _sample_action(state, behavior_probs, rng)
        done = False
        while not done:
            action = env.ACTIONS[a_idx]
            pair = (state, action)
            reward, next_state, done = env.step(action)
            if done:
                next_q = 0.0
                next_a_idx = a_idx  # unused
            else:
                next_a_idx = _sample_action(next_state, behavior_probs, rng)
                next_q = q_table[(next_state, env.ACTIONS[next_a_idx])]
            delta = reward + gamma * next_q - q_table[pair]
            if trace == "accumulating":
                traces[pair] += 1.0
            else:  # replacing: z(S, A) <- 1 (Section 12.7, Eq. 12.23 alt)
                traces[pair] = 1.0
            if clearing:
                # Clearing variant (Figure 12.11): traces of the non-selected
                # actions for the visited state are set to zero.
                for a in env.ACTIONS:
                    if a != action:
                        traces[(state, a)] = 0.0
            for p in pairs:
                q_table[p] += alpha * delta * traces[p]
                traces[p] *= gamma * lam
            state, a_idx = (next_state if not done else state), next_a_idx

    policy = {
        s: _epsilon_greedy_probs(np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon)
        for s in env.STATES
    }
    return policy, q_table


def sarsa_lambda_replacing(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    lam: float = 0.9,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """algorithm_8.1: Sarsa(lambda) with replacing traces. Book Section 12.7.

    Backward-view Sarsa(lambda); replacing traces set the visited pair's
    trace to 1 instead of incrementing (Section 12.7, Eq. 12.23 alternative,
    and Eq. 12.17 for the tabular case). lambda = 0 degenerates to one-step
    Sarsa (Algorithm 6.4); lambda = 1 behaves Monte-Carlo-like.

    Returns:
        (policy, Q): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    return _sarsa_lambda(env, n_episodes, alpha, lam, epsilon, gamma, seed,
                         trace="replacing", clearing=False)


def sarsa_lambda_clearing(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    lam: float = 0.9,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """algorithm_8.2: Sarsa(lambda) with replacing and clearing traces.

    Book Section 12.7, Figure 12.11 ("clearing the traces of other actions").
    As algorithm_8.1, but on each step the traces of the non-selected actions
    for the visited state are set to 0 (in addition to the replacing-trace
    bump of the selected action to 1). lambda = 0 degenerates to one-step
    Sarsa (Algorithm 6.4).

    Returns:
        (policy, Q): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    return _sarsa_lambda(env, n_episodes, alpha, lam, epsilon, gamma, seed,
                         trace="replacing", clearing=True)


def true_online_td_lambda(
    env: RandomWalkEnv,
    policy_fn: Callable | None = None,
    alpha: float = 0.1,
    lam: float = 0.5,
    n_episodes: int = 5000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """algorithm_8.3: True online TD(lambda). Book Section 12.5, Eq. 12.15-12.16.

    Tabular special case of the book's algorithm (box on p. 245) with one-hot
    features x(S) = e_S, so w = table of state values. Update per step:

        delta = R + gamma * V' - V
        z     = gamma * lam * z + (1 - alpha * gamma * lam * z[S]) * e_S
        w     = w + alpha * (delta + V - V_old) * z - alpha * (V - V_old) * e_S

    where V = w . x, V' = w . x' (0 for terminal), and V_old is the value
    estimate used at the previous step (0 at episode start). This is the
    "dutch trace" variant and is proven to match the on-line lambda-return
    algorithm exactly (van Seijen et al., 2016). lambda = 0 reduces exactly
    to TD(0); lambda = 1 is Monte-Carlo-like.

    Returns:
        dict state -> estimated value.
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
        v_old = 0.0
        done = False
        while not done:
            action_index = _sample_action(state, policy_fn, rng)
            reward, next_state, done = env.step(env.ACTIONS[action_index])
            v = values[state]
            v_next = 0.0 if next_state not in values else values[next_state]
            delta = reward + gamma * v_next - v
            # Dutch trace (Eq. 12.16) with one-hot features: z . x = z[state].
            z_dot_x = traces[state]
            for s in env.STATES:
                traces[s] = gamma * lam * traces[s]
            traces[state] = traces[state] + (1.0 - alpha * gamma * lam * z_dot_x)
            # Weight update (Eq. 12.15, tabular/one-hot form of the box).
            for s in env.STATES:
                values[s] += alpha * (delta + v - v_old) * traces[s]
            values[state] -= alpha * (v - v_old)
            v_old = v_next
            state = next_state if not done else state
    return values


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 12 / repo Ch. 8)
# ---------------------------------------------------------------------------

def _rms_error(values: dict, true_values: dict) -> float:
    states = list(true_values)
    errs = [values[s] - true_values[s] for s in states]
    return float(np.sqrt(np.mean(np.square(errs))))


def plot_td_lambda_random_walk(
    save_path: str | None = None,
    show: bool = False,
    lambdas: tuple[float, ...] = (0.0, 0.4, 0.8, 1.0),
    alpha: float = 0.1,
    n_episodes: int = 100,
    runs: int = 50,
    seed: int = 0,
):
    """TD(lambda) prediction on the random walk (eligibility traces demo).

    Sutton & Barto (2nd ed.), Ch. 12 (Eligibility Traces), Sec. 12.2 and
    related random-walk experiments (e.g. Fig. 12.3/12.6 in the book): compares
    backward-view TD(lambda) prediction with accumulating traces for several
    lambda values on the 5-state random walk (Example 6.2). Curves show RMS
    error vs true v* over episodes, averaged over runs. lambda=0 is TD(0);
    larger lambda incorporates longer returns (approaching MC-like behavior as
    lambda->1), illustrating the bias-variance tradeoff discussed in Ch. 12.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chapter_06.solution import RandomWalkEnv, random_walk_true_values

    true_values = random_walk_true_values()
    episodes_axis = np.arange(1, n_episodes + 1)
    curves = {lam: np.zeros(n_episodes, dtype=np.float64) for lam in lambdas}

    for lam in lambdas:
        for r in range(runs):
            rs = None if seed is None else seed + r * 101 + int(lam * 1000)
            env = RandomWalkEnv(seed=rs)
            rng = np.random.default_rng(rs)
            values = {s: 0.5 for s in env.STATES}
            traces = {s: 0.0 for s in env.STATES}
            for ep in range(n_episodes):
                # one episode of accumulating-trace TD(lambda) prediction
                for s in env.STATES:
                    traces[s] = 0.0
                state = env.reset()
                done = False
                while not done:
                    a_idx = int(rng.integers(0, 2))
                    reward, next_state, done = env.step(env.ACTIONS[a_idx])
                    next_v = 0.0 if next_state.endswith("_end") else values[next_state]
                    delta = reward + 1.0 * next_v - values[state]
                    traces[state] += 1.0
                    for s in env.STATES:
                        values[s] += alpha * delta * traces[s]
                        traces[s] *= 1.0 * lam
                    state = next_state
                curves[lam][ep] += _rms_error(values, true_values)
        curves[lam] /= runs

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lam in lambdas:
        ax.plot(episodes_axis, curves[lam], label=f"lambda={lam:g}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error (vs true v*)")
    ax.set_title("S&B (2nd ed.), Ch. 12 (repo Ch. 8): TD(lambda) on random walk")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
