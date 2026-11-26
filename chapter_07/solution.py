"""Solution implementation for Chapter 7 - n-step Bootstrapping.

Reference: Sutton & Barto (2nd Ed), Chapter 7

Implements: the n-step TD prediction box (Sec. 7.1) as `n_step_td_prediction`,
n-step Sarsa (Sec. 7.2) as `n_step_sarsa`,
n-step Tree Backup (Sec. 7.5) as `tree_backup`,
SARSA(lambda) control with accumulating traces (Sec. 12.2-style) as `sarsa_lambda`.

The chapter's 19-state random walk (Example 7.1 / Figure 7.2) is built on the
shared `RandomWalkEnv(n_states=19, left_reward=-1.0, right_reward=1.0)`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rl_envs.random_walk import RandomWalkEnv


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    n = len(q_values)
    best = np.flatnonzero(q_values == q_values.max())
    probs = np.full(n, epsilon / n, dtype=np.float64)
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def _action_index(probs: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    return int(rng.choice(len(probs), p=probs))


def _check_epsilon(epsilon: float) -> None:
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")


def _check_n(n: int) -> None:
    if int(n) < 1:
        raise ValueError("n must be >= 1")


def _action_probs(
    env: RandomWalkEnv, q_table: dict, state, epsilon: float
) -> np.ndarray:
    """Epsilon-greedy probabilities in `state` under the current Q values."""
    return _epsilon_greedy_probs(
        np.array([q_table[(state, a)] for a in env.ACTIONS]), epsilon
    )


def _epsilon_greedy_policy(env: RandomWalkEnv, q_table: dict, epsilon: float) -> dict:
    """Read the final policy off the learned action values."""
    return {s: _action_probs(env, q_table, s, epsilon) for s in env.STATES}


def _n_step_sarsa_rollout(
    env: RandomWalkEnv,
    q_table: dict,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[list, list[int], list[float]]:
    """Collect one episode: states, chosen action indices, and rewards.

    The action at each state is sampled when that state is first visited, so
    the n-step backups below can reuse it without drawing new random numbers.
    """
    states = [env.reset()]
    action_indexes: list[int] = []
    rewards: list[float] = []

    action_indexes.append(_action_index(_action_probs(env, q_table, states[0], epsilon), rng))

    done = False
    while not done:
        action = env.ACTIONS[action_indexes[-1]]
        reward, next_state, done = env.step(action)
        rewards.append(float(reward))
        if done:
            states.append(None)
            break
        states.append(next_state)
        action_indexes.append(_action_index(_action_probs(env, q_table, next_state, epsilon), rng))

    return states, action_indexes, rewards


def _n_step_sarsa_sweep(
    env: RandomWalkEnv,
    q_table: dict,
    states: list,
    action_indexes: list[int],
    rewards: list[float],
    n: int,
    alpha: float,
    gamma: float,
) -> None:
    """Back up every decision time in the episode with its n-step Sarsa target."""
    total_steps = len(rewards)
    for tau in range(total_steps):
        target = 0.0
        for k in range(n):
            if tau + k >= total_steps:
                break
            target += gamma**k * rewards[tau + k]
        if tau + n < total_steps and states[tau + n] is not None:
            next_s = states[tau + n]
            target += gamma**n * q_table[(next_s, env.ACTIONS[action_indexes[tau + n]])]
        tau_state = states[tau]
        tau_action = env.ACTIONS[action_indexes[tau]]
        q_table[(tau_state, tau_action)] += alpha * (target - q_table[(tau_state, tau_action)])


def n_step_sarsa(
    env: RandomWalkEnv,
    n: int = 4,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    n_episodes: int = 5000,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """n-step Sarsa (Sec. 7.2, box titled "State-action n-step Sarsa"). Book Section 7.2.

    Args:
        env: The environment.
        n: Bootstrapping depth (n = 1 is Sarsa(0)).
        alpha: Step size.
        epsilon: epsilon-greedy exploration parameter.
        gamma: Discount factor.
        n_episodes: Number of episodes to run.
        seed: Optional seed.

    Returns:
        (policy, q_table): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    _check_epsilon(epsilon)
    _check_n(n)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    for _ in range(n_episodes):
        # Each episode supplies its own trajectory; the backups follow it.
        states, action_indexes, rewards = _n_step_sarsa_rollout(env, q_table, epsilon, rng)
        _n_step_sarsa_sweep(env, q_table, states, action_indexes, rewards, n, alpha, gamma)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


class _TreeRollout(NamedTuple):
    """The per-step quantities needed by the n-step tree-backup target."""

    states: list
    action_indexes: list[int]
    rewards: list[float]
    deltas: list[float]
    pis: list[float]
    q_stored: list[float]


def _tree_backup_rollout(
    env: RandomWalkEnv,
    q_table: dict,
    epsilon: float,
    gamma: float,
    rng: np.random.Generator,
) -> _TreeRollout:
    """Collect one episode plus the TD errors and behaviour probabilities.

    `pis` and `q_stored` are recorded alongside the action indices so the
    expected (off-policy) target can be reconstructed later without re-sampling.
    """
    states = [env.reset()]
    rewards: list[float] = []
    deltas: list[float] = []
    pis: list[float] = []
    q_stored: list[float] = []

    probs0 = _action_probs(env, q_table, states[0], epsilon)
    action_indexes = [_action_index(probs0, rng)]
    q_stored.append(q_table[(states[0], env.ACTIONS[action_indexes[0]])])

    done = False
    while not done:
        action = env.ACTIONS[action_indexes[-1]]
        reward, next_state, done = env.step(action)
        rewards.append(float(reward))
        if done:
            deltas.append(rewards[-1] - q_stored[-1])
            break

        q_next = np.array([q_table[(next_state, a)] for a in env.ACTIONS])
        probs_next = _epsilon_greedy_probs(q_next, epsilon)
        deltas.append(
            rewards[-1] + gamma * float(np.dot(probs_next, q_next)) - q_stored[-1]
        )
        states.append(next_state)
        a_next = _action_index(probs_next, rng)
        action_indexes.append(a_next)
        pis.append(float(probs_next[a_next]))
        q_stored.append(q_table[(next_state, env.ACTIONS[a_next])])

    return _TreeRollout(states, action_indexes, rewards, deltas, pis, q_stored)


def _tree_backup_sweep(
    env: RandomWalkEnv,
    q_table: dict,
    rollout: _TreeRollout,
    n: int,
    alpha: float,
    gamma: float,
) -> None:
    """Back up every decision time with the n-step expected (tree) target."""
    total_steps = len(rollout.rewards)
    for tau in range(total_steps):
        target = rollout.q_stored[tau]
        z = 1.0
        for k in range(tau, min(tau + n, total_steps)):
            target += z * rollout.deltas[k]
            if k < len(rollout.pis):
                z *= gamma * rollout.pis[k]
        s_tau = rollout.states[tau]
        a_tau = env.ACTIONS[rollout.action_indexes[tau]]
        q_table[(s_tau, a_tau)] += alpha * (target - q_table[(s_tau, a_tau)])


def tree_backup(
    env: RandomWalkEnv,
    n: int = 4,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    n_episodes: int = 5000,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """n-step Tree Backup (Sec. 7.5, box titled "State-action n-step Tree Backup"). Book Section 7.5.

    Off-policy control without importance sampling: the target policy is the
    epsilon-greedy policy w.r.t. q_table (a 50/50 random policy for eps-greedy-0.5,
    or uniform for the very first passes); the behavior policy is the same
    epsilon-greedy here.

    Returns:
        (policy, q_table): policy[state] = probability vector over ACTIONS.
    """
    _check_epsilon(epsilon)
    _check_n(n)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    for _ in range(n_episodes):
        # The expected target needs the behaviour probability of each step, so
        # it is recorded while the trajectory is generated, not afterwards.
        rollout = _tree_backup_rollout(env, q_table, epsilon, gamma, rng)
        _tree_backup_sweep(env, q_table, rollout, n, alpha, gamma)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


def _sarsa_lambda_episode(
    env: RandomWalkEnv,
    q_table: dict,
    alpha: float,
    epsilon: float,
    gamma: float,
    lam: float,
    rng: np.random.Generator,
) -> None:
    """Run one episode: traces carry each TD error back to earlier visits."""
    traces = {pair: 0.0 for pair in q_table}
    state = env.reset()
    a_idx = _action_index(_action_probs(env, q_table, state, epsilon), rng)
    done = False
    while not done:
        action = env.ACTIONS[a_idx]
        reward, next_state, done = env.step(action)
        if done:
            delta = reward - q_table[(state, action)]
            next_a_idx = a_idx
        else:
            next_a_idx = _action_index(_action_probs(env, q_table, next_state, epsilon), rng)
            delta = (
                reward
                + gamma * q_table[(next_state, env.ACTIONS[next_a_idx])]
                - q_table[(state, action)]
            )

        traces[(state, action)] += 1.0
        for pair in q_table:
            q_table[pair] += alpha * delta * traces[pair]
            traces[pair] *= gamma * lam
        state = next_state if not done else state
        a_idx = next_a_idx


def sarsa_lambda(
    env: RandomWalkEnv,
    lam: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    n_episodes: int = 5000,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """SARSA(lambda) with accumulating eligibility traces (Sec. 12.2-style box, adapted for control).

    Book Section 12.2 traces; lambda = 0 recovers one-step Sarsa, lambda = 1
    recovers the Monte Carlo return.

    Returns:
        (policy, q_table): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    _check_epsilon(epsilon)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    for _ in range(n_episodes):
        _sarsa_lambda_episode(env, q_table, alpha, epsilon, gamma, lam, rng)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


# ---------------------------------------------------------------------------
# n-step TD prediction (Sec. 7.1, Example 7.1 / Figure 7.2)
# ---------------------------------------------------------------------------

_STATES_19 = tuple(chr(ord("A") + i) for i in range(19))


def _n_step_td_sweep(
    values: dict,
    states: list,
    rewards: list[float],
    n: int,
    alpha: float,
    gamma: float,
) -> None:
    """One episode of the n-step TD prediction box (Sec. 7.1), in-place."""
    total = len(rewards)
    powers = [gamma**i for i in range(total + 1)]
    for tau in range(total):
        target = 0.0
        for k in range(tau, min(tau + n, total)):
            target += powers[k - tau] * rewards[k]
        if tau + n < total:
            target += powers[n] * values[states[tau + n]]
        values[states[tau]] += alpha * (target - values[states[tau]])


def n_step_td_prediction(
    env: RandomWalkEnv | None = None,
    n: int = 1,
    alpha: float = 0.4,
    n_episodes: int = 10,
    gamma: float = 1.0,
    init_value: float = 0.0,
    seed: int | None = None,
) -> dict:
    """n-step TD prediction for the random walk (box, Sec. 7.1).

    Prediction on the random-walk MRP under the equiprobable policy. The
    book's Example 7.1 uses the 19-state walk (left outcome -1, right +1,
    all values initialized to 0) as the running example; the same 5-state,
    0/+1 walk of Example 6.2 is the default here when `env` is omitted.

    Returns V as a dict over env.STATES.
    """
    _check_n(n)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if env is None:
        env = RandomWalkEnv(seed=seed)
    rng = np.random.default_rng(seed)
    values = {s: float(init_value) for s in env.STATES}
    for _ in range(n_episodes):
        states = [env.reset()]
        rewards = []
        done = False
        while not done:
            action = "left" if rng.random() < 0.5 else "right"
            reward, next_state, done = env.step(action)
            rewards.append(float(reward))
            states.append(next_state)
        _n_step_td_sweep(values, states, rewards, n, alpha, gamma)
    return values


def _sample_walks(repeats: int, n_episodes: int, seed: int):
    """Pre-sample the walk sets used for every (n, alpha) setting.

    The book used the same sets of walks for all parameter settings,
    controlling for walk difficulty. Each walk is a (states, rewards) pair.
    """
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(repeats):
        rep = []
        for _ in range(n_episodes):
            env = RandomWalkEnv(
                n_states=19, left_reward=-1.0, right_reward=1.0,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            states = [env.reset()]
            rewards = []
            done = False
            while not done:
                action = "left" if rng.random() < 0.5 else "right"
                reward, next_state, done = env.step(action)
                rewards.append(float(reward))
                states.append(next_state)
            rep.append((states, rewards))
        walks.append(rep)
    return walks


def run_n_step_prediction_experiment(
    ns: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    repeats: int = 100,
    n_episodes: int = 10,
    gamma: float = 1.0,
    seed: int = 0,
) -> dict:
    """Example 7.1 / Figure 7.2: n-step TD prediction vs alpha, for various n.

    Prediction on the 19-state walk (A..S, left outcome -1, right +1, all
    values initialized to 0) using the equiprobable policy. The same sets of
    walks are used for every (n, alpha) setting, as in the book.

    Performance measure for each setting: the square root of the mean squared
    error against the true values, taken at the end of each of the first
    `n_episodes` episodes, then averaged over episodes and over `repeats`
    repetitions of the whole experiment.

    Returns {'alphas', 'ns', 'curves': {n: [error per alpha]}}.
    """
    if repeats < 1 or n_episodes < 1:
        raise ValueError("repeats and n_episodes must be >= 1")
    for n in ns:
        _check_n(n)
    true_values = RandomWalkEnv(n_states=19, left_reward=-1.0, right_reward=1.0).true_values()
    states_19 = tuple(true_values)
    walks = _sample_walks(repeats, n_episodes, seed)
    curves: dict[int, list[float]] = {}
    for n in ns:
        curve = []
        for alpha in alphas:
            total_error = 0.0
            for rep in walks:
                values = dict.fromkeys(states_19, 0.0)
                for states, rewards in rep:
                    _n_step_td_sweep(values, states, rewards, n, alpha, gamma)
                    err = float(
                        np.sqrt(np.mean([(values[s] - true_values[s]) ** 2 for s in states_19]))
                    )
                    total_error += err
            curve.append(total_error / (repeats * n_episodes))
        curves[n] = curve
    return {"alphas": list(alphas), "ns": list(ns), "curves": curves}


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 7)
# ---------------------------------------------------------------------------

def plot_figure_7_2(
    save_path: str | None = None,
    show: bool = False,
    repeats: int = 100,
    n_episodes: int = 10,
    seed: int = 0,
    ns: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
):
    """Performance of n-step TD methods vs alpha, for various n (Fig. 7.2).

    19-state random walk (Example 7.1, Sec. 7.1). Error = RMS vs the true
    values at the end of each of the first `n_episodes` episodes, averaged
    over episodes and repetitions (book: 100 repetitions); the same walk sets
    are used for every (n, alpha) setting. The x-axis is alpha; each curve is
    one value of n. Intermediate n performs best.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_n_step_prediction_experiment(
        ns=ns, alphas=alphas, repeats=repeats, n_episodes=n_episodes, seed=seed
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n in results["ns"]:
        ax.plot(results["alphas"], results["curves"][n], marker="o", markersize=4, label=str(n))
    ax.set_xlabel("alpha")
    ax.set_ylabel("RMS error (avg over states, first 10 episodes)")
    ax.set_title(
        "S&B (2nd ed.), Sec. 7.1, Example 7.1 / Fig. 7.2:\n"
        "n-step TD prediction on a 19-state random walk"
    )
    ax.legend(title="n")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

