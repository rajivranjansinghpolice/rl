"""Solution implementation for Chapter 12 - Eligibility Traces.

Reference: Sutton & Barto (2nd Ed), Chapter 12 (Eligibility Traces).

Implements:
- algorithm_12.1 - TD(lambda) with replacing traces:
    state-value prediction (Eq. 12.17, Section 12.2/12.7) and
    Sarsa(lambda) control with replacing traces (Section 12.7, box on p. 250).
- algorithm_12.2 - TD(lambda) with clearing traces:
    Sarsa(lambda) control with replacing traces plus clearing of the traces
    of the non-selected actions for the visited state (Figure 12.11,
    Section 12.7; van Hasselt et al.'s "clearing" variant).
- algorithm_12.3 - True online TD(lambda) (Section 12.5, Eq. 12.15-12.16,
    box on p. 245), tabular implementation with one-hot features.
- State-value TD(lambda) prediction with accumulating or replacing traces
    (Section 12.2, box on p. 240, Eq. 12.5).
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


def _q_values(env: RandomWalkEnv, q_table: dict, state) -> np.ndarray:
    """Action values in `state`, in the environment's action order."""
    return np.array([q_table[(state, a)] for a in env.ACTIONS])


def _epsilon_greedy_policy(env: RandomWalkEnv, q_table: dict, epsilon: float) -> dict:
    """Read the final epsilon-greedy policy off the learned action values."""
    return {
        s: _epsilon_greedy_probs(_q_values(env, q_table, s), epsilon)
        for s in env.STATES
    }


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


def _sarsa_lambda_episode(
    env: RandomWalkEnv,
    q_table: dict,
    pairs: list,
    alpha: float,
    epsilon: float,
    gamma: float,
    lam: float,
    trace: str,
    clearing: bool,
    rng: np.random.Generator,
) -> None:
    """Run one episode of backward-view Sarsa(lambda), updating Q in place."""
    traces = {pair: 0.0 for pair in pairs}

    def behavior_probs(state):
        return _epsilon_greedy_probs(_q_values(env, q_table, state), epsilon)

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
        # The TD error is applied to all eligible pairs before the traces
        # decay, so recent visits receive the largest share of the update.
        for p in pairs:
            q_table[p] += alpha * delta * traces[p]
            traces[p] *= gamma * lam
        state, a_idx = (next_state if not done else state), next_a_idx


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

    for _ in range(n_episodes):
        _sarsa_lambda_episode(env, q_table, pairs, alpha, epsilon, gamma, lam,
                              trace, clearing, rng)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


def sarsa_lambda_replacing(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    lam: float = 0.9,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """algorithm_12.1: Sarsa(lambda) with replacing traces. Book Section 12.7.

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
    """algorithm_12.2: Sarsa(lambda) with replacing and clearing traces.

    Book Section 12.7, Figure 12.11 ("clearing the traces of other actions").
    As algorithm_12.1, but on each step the traces of the non-selected actions
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
    """algorithm_12.3: True online TD(lambda). Book Section 12.5, Eq. 12.15-12.16.

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


def _is_greedy(q_values: np.ndarray, action_index: int) -> bool:
    """True iff action_index is in the argmax set of q_values."""
    return bool(q_values[action_index] == q_values.max())


def _watkins_q_episode(
    env: RandomWalkEnv,
    q_table: dict,
    pairs: list,
    alpha: float,
    epsilon: float,
    gamma: float,
    lam: float,
    trace: str,
    rng: np.random.Generator,
) -> None:
    """Run one episode of Watkins's Q(lambda), updating Q in place."""
    traces = {pair: 0.0 for pair in pairs}

    def behavior_probs(state):
        return _epsilon_greedy_probs(_q_values(env, q_table, state), epsilon)

    state = env.reset()
    done = False
    while not done:
        q_before = _q_values(env, q_table, state)
        action_index = _sample_action(state, behavior_probs, rng)
        # Greediness is judged on the pre-update values; ties count as greedy.
        greedy = _is_greedy(q_before, action_index)
        action = env.ACTIONS[action_index]
        pair = (state, action)
        reward, next_state, done = env.step(action)
        best_next = (
            0.0
            if done
            else max(q_table[(next_state, a)] for a in env.ACTIONS)
        )
        delta = reward + gamma * best_next - q_table[pair]
        if trace == "accumulating":
            traces[pair] += 1.0
        else:  # replacing: z(S, A) <- 1 for the visited pair
            traces[pair] = 1.0
        for p in pairs:
            q_table[p] += alpha * delta * traces[p]
        for p in pairs:
            traces[p] *= gamma * lam
        if not greedy:
            # Watkins's cut (Section 12.10, Figure 12.12): after a non-greedy
            # action, all traces are zeroed. Its own TD error has already been
            # applied; only propagation further back is stopped.
            for p in pairs:
                traces[p] = 0.0
        state = next_state if not done else state


def watkins_q_lambda(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    lam: float = 0.9,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    seed: int | None = None,
    trace: str = "replacing",
) -> tuple[dict, dict]:
    """algorithm_10.1: Watkins's Q(lambda), backward view. Book Section 12.10.

    Off-policy control with eligibility traces (Figure 12.12). Like
    Q-learning (Algorithm 6.2), the bootstrap target is
    R + gamma * max_a Q(S', a). Eligibility traces are bumped for the
    visited (S, A) pair — z <- 1 for replacing traces (default, matching
    the tabular practice and keeping z bounded) or z += 1 for accumulating
    traces — and decayed by gamma * lam after every step.

    Trace-cutting behavior (Watkins's rule): after the Q-update at a step
    whose behavior action was non-greedy with respect to Q(s, .) at that
    state (i.e., not in the argmax set; ties count as greedy, matching the
    epsilon-greedy tie-breaking), ALL eligibility traces are set to zero.
    Thus the TD error of a non-greedy step still propagates through any
    pre-existing traces at that same step, but no TD error can propagate
    backward across a non-greedy action: the series of component updates
    ends either with the end of the episode or with the first non-greedy
    action, whichever comes first (Figure 12.12). Greediness is evaluated
    with the Q-values from before the update at the current step. At
    lambda = 0 the trace always starts empty at each step, so the method
    reduces exactly to one-step Q-learning (Algorithm 6.2). Note: with
    accumulating traces this off-policy-with-traces method can be unstable
    at larger alpha; replacing traces are the safer default.

    Watkins's Q-learning off-policy control with eligibility traces is
    semi-gradient and uses the max bootstrap; with an undiscounted (gamma=1)
    task it is known to be unstable and diverges in practice for any
    reasonable alpha once traces are lengthened. The default gamma=0.99
    (rather than 1.0) keeps the backups contracting; Section 12.10 of the
    book itself motivates Tree-Backup(lambda) as a stable successor. Pass
    gamma=1.0 only with small lam/alpha.

    Args:
        env: A RandomWalkEnv (or compatible env with STATES/ACTIONS).
        n_episodes: Number of episodes to sample.
        alpha: Step size.
        lam: Trace decay parameter lambda in [0, 1].
        epsilon: Exploration rate of the epsilon-greedy behavior policy.
        gamma: Discount factor (default 0.99; see stability note above).
        seed: Optional seed.
        trace: "replacing" (z <- 1) or "accumulating" (z += 1).

    Returns:
        (policy, Q): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if trace not in ("replacing", "accumulating"):
        raise ValueError("trace must be 'replacing' or 'accumulating'")
    rng = np.random.default_rng(seed)
    pairs = [(s, a) for s in env.STATES for a in env.ACTIONS]
    q_table = {pair: 0.0 for pair in pairs}

    for _ in range(n_episodes):
        _watkins_q_episode(env, q_table, pairs, alpha, epsilon, gamma, lam, trace, rng)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


def _n_step_return(
    rewards: list, bootstrap_values: list, gamma: float, t: int, u: int
) -> float:
    """G_{t:u}: discounted rewards through u-1, then the bootstrap at u."""
    total = 0.0
    for i in range(t, u):
        total += (gamma ** (i - t)) * rewards[i]
    total += (gamma ** (u - t)) * bootstrap_values[u - 1]
    return total


def _trace_cut_horizon(greedy: list, t: int, n_steps: int) -> int:
    """Steps carried from t before Watkins's cut at the first non-greedy action."""
    first_nongreedy = next((j for j in range(t, n_steps)
                            if not greedy[j]), None)
    if first_nongreedy is None:
        return n_steps - t
    return first_nongreedy - t + 1


def watkins_lambda_return(
    rewards: list,
    bootstrap_values: list,
    greedy: list,
    gamma: float = 1.0,
    lam: float = 0.9,
) -> list:
    """Forward-view Watkins's Q(lambda)-return. Book Section 12.10.

    The theoretical companion to `watkins_q_lambda`: for each time t of one
    episode, computes the trace-cut lambda-return used to update (S_t, A_t)

        G^lambda_t = (1 - lam) * sum_{n=1}^{n_max} lam^(n-1) * G_{t:t+n}
                     + lam^n_max * G_{t:t+n_max},

    where G_{t:u} = sum_{i=t}^{u-1} gamma^(i-t) * rewards[i]
    + gamma^(u-t) * bootstrap_values[u - 1], and the horizon n_max is determined
    by the first non-greedy action: n_max = j - t + 1 with j the first index
    j >= t with greedy[j] False, or n_max = len(rewards) - t if all actions
    are greedy (Section 12.10, Figure 12.12: component updates end either at
    the episode's end or at the first non-greedy action, whichever comes
    first). For lam = 1 the result is the lambda-return to that horizon;
    for lam = 0 it reduces to the one-step Q-learning return G_{t:t+1}.

    Args:
        rewards: rewards R_{t+1}..R_T of one episode (0-based list).
        bootstrap_values: max_a Q(s_k, .) for k = 1..len(rewards);
            bootstrap_values[-1] should be 0.0 for a terminal final state.
        greedy: greedy[k] is True iff the behavior action A_k at step k was
            greedy w.r.t. Q(s_k, .).
        gamma: Discount factor.
        lam: Trace decay parameter lambda in [0, 1].

    Returns:
        list of G^lambda_t for t = 0..len(rewards)-1.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if len(bootstrap_values) != len(rewards):
        raise ValueError("bootstrap_values must have len(rewards) entries")
    if len(greedy) != len(rewards):
        raise ValueError("greedy must have the same length as rewards")
    n_steps = len(rewards)
    returns = []
    for t in range(n_steps):
        # Component updates stop at the first non-greedy action, or at the
        # episode's end if every action along the way was greedy.
        n_max = _trace_cut_horizon(greedy, t, n_steps)
        g_lambda = (1.0 - lam) * sum(
            (lam ** (n - 1))
            * _n_step_return(rewards, bootstrap_values, gamma, t, t + n)
            for n in range(1, n_max + 1)
        ) + (lam ** n_max) * _n_step_return(rewards, bootstrap_values, gamma,
                                            t, t + n_max)
        returns.append(g_lambda)
    return returns


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 12)
# ---------------------------------------------------------------------------

def _rms_error(values: dict, true_values: dict) -> float:
    states = list(true_values)
    errs = [values[s] - true_values[s] for s in states]
    return float(np.sqrt(np.mean(np.square(errs))))


# Random-walk demos: fixed initialisation and well-separated per-run seeds.
_TD_INIT_VALUE = 0.5
_TD_SEED_STRIDE = 101
_WATKINS_SEED_STRIDE = 1009


def _td_lambda_curves(
    true_values: dict,
    lambdas: tuple[float, ...],
    alpha: float,
    n_episodes: int,
    runs: int,
    seed: int | None,
) -> dict:
    """Average the RMS error over runs for each lambda, episode by episode."""
    curves = {lam: np.zeros(n_episodes, dtype=np.float64) for lam in lambdas}
    for lam in lambdas:
        for r in range(runs):
            rs = None if seed is None else seed + r * _TD_SEED_STRIDE + int(lam * 1000)
            env = RandomWalkEnv(seed=rs)
            rng = np.random.default_rng(rs)
            values = {s: _TD_INIT_VALUE for s in env.STATES}
            traces = {s: 0.0 for s in env.STATES}
            for ep in range(n_episodes):
                # One accumulating-trace episode. Traces are cleared first so
                # eligibility cannot leak in from the previous episode.
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
    return curves


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

    from rl_envs.random_walk import random_walk_true_values

    true_values = random_walk_true_values()
    episodes_axis = np.arange(1, n_episodes + 1)
    curves = _td_lambda_curves(true_values, lambdas, alpha, n_episodes, runs, seed)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lam in lambdas:
        ax.plot(episodes_axis, curves[lam], label=f"lambda={lam:g}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error (vs true v*)")
    ax.set_title("S&B (2nd ed.), Ch. 12: TD(lambda) on random walk")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def _watkins_q_curves(
    true_values: dict,
    lambdas: tuple[float, ...],
    alpha: float,
    epsilon: float,
    gamma: float,
    n_episodes: int,
    runs: int,
    seed: int,
    checkpoints: list[int],
) -> dict:
    """Average RMS error at selected episode counts for each lambda."""
    curves = {lam: np.zeros(len(checkpoints), dtype=np.float64) for lam in lambdas}
    for lam in lambdas:
        for r in range(runs):
            rs = seed + r * _WATKINS_SEED_STRIDE + int(lam * 1000)
            for ci, ep in enumerate(checkpoints):
                env = RandomWalkEnv(seed=rs)
                _, q = watkins_q_lambda(
                    env, n_episodes=ep, alpha=alpha, lam=lam, epsilon=epsilon,
                    gamma=gamma, seed=rs,
                )
                values = {s: max(q[(s, a)] for a in env.ACTIONS) for s in env.STATES}
                curves[lam][ci] += _rms_error(values, true_values)
        curves[lam] /= runs
    return curves


def plot_watkins_q_lambda_random_walk(
    save_path: str | None = None,
    show: bool = False,
    lambdas: tuple[float, ...] = (0.0, 0.4, 0.8),
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    n_episodes: int = 150,
    runs: int = 20,
    seed: int = 0,
):
    """Watkins's Q(lambda) on the random walk (Sec. 12.10, Fig. 12.12 context).

    Sutton & Barto (2nd ed.), Sec. 12.10 "Watkins's Q(lambda)" (Fig. 12.12):
    off-policy control with eligibility traces cut at the first non-greedy
    action. This demo trains `watkins_q_lambda` on the 5-state random walk
    (Example 6.2) for several lambda values and plots RMS error of greedy state
    values max_a Q(s,a) vs true v* over episodes (averaged over runs). Uses
    gamma<1 by default for stability (see implementation notes).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_values = random_walk_true_values()
    checkpoints = list(range(1, n_episodes + 1, max(1, n_episodes // 40)))
    if checkpoints[-1] != n_episodes:
        checkpoints.append(n_episodes)
    curves = _watkins_q_curves(true_values, lambdas, alpha, epsilon, gamma,
                               n_episodes, runs, seed, checkpoints)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lam in lambdas:
        ax.plot(checkpoints, curves[lam], label=f"lambda={lam:g}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error (greedy max_a Q vs v*)")
    ax.set_title("S&B (2nd ed.), Sec. 12.10: Watkins's Q(lambda) on random walk")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
