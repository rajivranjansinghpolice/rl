"""Solution implementation for Chapter 7 - n-step Bootstrapping.

Reference: Sutton & Barto (2nd Ed), Chapter 7

Implements: Alg 7.1 (n-step Sarsa),
Alg 7.2 (n-step Tree Backup),
Alg 7.3 (SARSA(lambda) control, accumulating traces).

Uses the RandomWalkEnv from chapter_06 for the random-walk examples.
"""

from __future__ import annotations

import numpy as np

from chapter_06.solution import RandomWalkEnv


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





def n_step_sarsa(
    env: RandomWalkEnv,
    n: int = 4,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    n_episodes: int = 5000,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """n-step Sarsa (Algorithm 7.2). Book Section 7.2.

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
        states = [env.reset()]
        action_indexes: list[int] = []
        rewards: list[float] = []

        probs0 = _epsilon_greedy_probs(
            np.array([q_table[(states[0], a)] for a in env.ACTIONS]), epsilon
        )
        action_indexes.append(_action_index(probs0, rng))

        done = False
        while not done:
            action = env.ACTIONS[action_indexes[-1]]
            reward, next_state, done = env.step(action)
            rewards.append(float(reward))
            if done:
                states.append(None)
                break
            states.append(next_state)
            probs_next = _epsilon_greedy_probs(
                np.array([q_table[(next_state, a)] for a in env.ACTIONS]), epsilon
            )
            action_indexes.append(_action_index(probs_next, rng))

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

    policy = {
        s: _epsilon_greedy_probs(np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon)
        for s in env.STATES
    }
    return policy, q_table


def tree_backup(
    env: RandomWalkEnv,
    n: int = 4,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    n_episodes: int = 5000,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """n-step Tree Backup (Algorithm 7.5). Book Section 7.5.

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
        states = [env.reset()]
        rewards: list[float] = []
        deltas: list[float] = []
        pis: list[float] = []
        q_stored: list[float] = []

        probs0 = _epsilon_greedy_probs(
            np.array([q_table[(states[0], a)] for a in env.ACTIONS]), epsilon
        )
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

        total_steps = len(rewards)
        for tau in range(total_steps):
            target = q_stored[tau]
            z = 1.0
            for k in range(tau, min(tau + n, total_steps)):
                target += z * deltas[k]
                if k < len(pis):
                    z *= gamma * pis[k]
            s_tau = states[tau]
            a_tau = env.ACTIONS[action_indexes[tau]]
            q_table[(s_tau, a_tau)] += alpha * (target - q_table[(s_tau, a_tau)])

    policy = {
        s: _epsilon_greedy_probs(np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon)
        for s in env.STATES
    }
    return policy, q_table


def sarsa_lambda(
    env: RandomWalkEnv,
    lam: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    n_episodes: int = 5000,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """SARSA(lambda) with accumulating eligibility traces (Algorithm 7.3).

    Book Section 12.2 style traces adapted to control; lambda = 0 recovers
    one-step Sarsa, lambda = 1 recovers the Monte Carlo return.

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
        traces = {pair: 0.0 for pair in q_table}
        state = env.reset()
        a_idx = _action_index(
            _epsilon_greedy_probs(
                np.array([q_table[(state, a)] for a in env.ACTIONS]), epsilon
            ),
            rng,
        )
        done = False
        while not done:
            action = env.ACTIONS[a_idx]
            reward, next_state, done = env.step(action)
            if done:
                delta = reward - q_table[(state, action)]
                next_a_idx = a_idx
            else:
                next_a_idx = _action_index(
                    _epsilon_greedy_probs(
                        np.array([q_table[(next_state, a)] for a in env.ACTIONS]), epsilon
                    ),
                    rng,
                )
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

    policy = {
        s: _epsilon_greedy_probs(np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon)
        for s in env.STATES
    }
    return policy, q_table


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 7)
# ---------------------------------------------------------------------------

def _state_values_from_q(env, q_table):
    return {s: max(q_table[(s, a)] for a in env.ACTIONS) for s in env.STATES}


def _rms_vs_true(values: dict) -> float:
    from chapter_06.solution import random_walk_true_values

    true_values = random_walk_true_values()
    errs = [values[s] - true_values[s] for s in true_values]
    return float(np.sqrt(np.mean(np.square(errs))))


def plot_n_step_methods_random_walk(
    save_path: str | None = None,
    show: bool = False,
    n_episodes: int = 200,
    runs: int = 20,
    seed: int = 0,
):
    """Demonstration: n-step/on-policy trace methods on the random walk.

    Sutton & Barto (2nd ed.), Ch. 7 "n-step Bootstrapping" (Sec. 7.2 n-step
    Sarsa, Sec. 7.5 n-step Tree Backup) and Sec. 12.7/related SARSA(lambda)
    control: compares implemented control methods on the 5-state random walk
    (Example 6.2 environment reused here). Plots RMS error of greedy state-value
    estimates max_a Q(s,a) vs true v*(s) over episodes, averaged over runs, for
    n-step Sarsa (n=4), Tree Backup (n=4), and SARSA(lambda) (lambda=0.9).
    This is an implementation demonstration (not a direct reproduction of the
    book's 19-state n-step TD prediction Fig. 7.2, whose prediction experiment
    is not implemented in this module).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chapter_06.solution import RandomWalkEnv

    methods = [
        ("n-step Sarsa (n=4)", lambda env, ep_seed: n_step_sarsa(env, n=4, n_episodes=1, seed=ep_seed)),
        ("Tree Backup (n=4)", lambda env, ep_seed: tree_backup(env, n=4, n_episodes=1, seed=ep_seed)),
        ("SARSA(lambda), l=0.9", lambda env, ep_seed: sarsa_lambda(env, lam=0.9, n_episodes=1, seed=ep_seed)),
    ]
    # Incremental training by running 1 episode repeatedly is not how APIs work;
    # instead run full training up to ep episodes for each checkpoint? cheaper to
    # run once per total episodes and track by re-training is expensive. We'll
    # record error after every episode by training in chunks is not supported.
    # So we simulate learning curves by training for e episodes for a set of e.
    checkpoints = list(range(1, n_episodes + 1, max(1, n_episodes // 40)))
    if checkpoints[-1] != n_episodes:
        checkpoints.append(n_episodes)

    curves = {name: np.zeros(len(checkpoints), dtype=np.float64) for name, _ in methods}
    for r in range(runs):
        base = seed + r * 1009
        for mi, (name, _) in enumerate(methods):
            for ci, ep in enumerate(checkpoints):
                env = RandomWalkEnv(seed=base + mi * 97)
                if name.startswith("n-step Sarsa"):
                    _, q = n_step_sarsa(env, n=4, alpha=0.1, epsilon=0.1, n_episodes=ep, seed=base + mi)
                elif name.startswith("Tree"):
                    _, q = tree_backup(env, n=4, alpha=0.1, epsilon=0.1, n_episodes=ep, seed=base + mi)
                else:
                    _, q = sarsa_lambda(env, lam=0.9, alpha=0.1, epsilon=0.1, n_episodes=ep, seed=base + mi)
                curves[name][ci] += _rms_vs_true(_state_values_from_q(env, q))
    for name in curves:
        curves[name] /= runs

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, _ in methods:
        ax.plot(checkpoints, curves[name], label=name)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error vs true v* (greedy from Q)")
    ax.set_title("S&B (2nd ed.), Ch. 7: n-step/trace methods on random walk (demo)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
