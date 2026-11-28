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
    """epsilon greedy probs (reference implementation).


    Args:
        q_values: See the relevant book section and function body.
        epsilon: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7)."""
    # internal helper for the epsilon greedy probs step (book Chapter 7).
    # q values: action-value estimates for the state (book Chapter 7).
    q_values = np.asarray(q_values, dtype=np.float64)
    # n: result of len(...) (book Chapter 7).
    n = len(q_values)
    # best: best action or best value found so far (book Chapter 7).
    best = np.flatnonzero(q_values == q_values.max())
    # probs: action probabilities from the current policy (book Chapter 7).
    probs = np.full(n, epsilon / n, dtype=np.float64)
    # probs[best]: add value computed from its operands into the running value (book Chapter 7).
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def _action_index(probs: np.ndarray, rng: np.random.Generator) -> int:
    """action index (reference implementation).


    Args:
        probs: See the relevant book section and function body.
        rng: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7)."""
    # internal helper for the action index step (book Chapter 7).
    # probs: action probabilities from the current policy (book Chapter 7).
    probs = np.asarray(probs, dtype=np.float64)
    return int(rng.choice(len(probs), p=probs))


def _check_epsilon(epsilon: float) -> None:
    """check epsilon (reference implementation).


    Args:
        epsilon: See the relevant book section and function body.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7)."""
    # internal helper for the check epsilon step (book Chapter 7).
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")


def _check_n(n: int) -> None:
    """check n (reference implementation).


    Args:
        n: See the relevant book section and function body.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7)."""
    # internal helper for the check n step (book Chapter 7).
    if int(n) < 1:
        raise ValueError("n must be >= 1")


def _action_probs(
    env: RandomWalkEnv, q_table: dict, state, epsilon: float
) -> np.ndarray:
    """Epsilon-greedy probabilities in `state` under the current Q values.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        state: See Section 7.1 and the implementation for its role.
        epsilon: See Section 7.1 and the implementation for its role.

    Returns:
        action probs as defined in Section 7.1."""
    # internal helper for the action probs step (book Chapter 7).
    return _epsilon_greedy_probs(
        np.array([q_table[(state, a)] for a in env.ACTIONS]), epsilon
    )


def _epsilon_greedy_policy(env: RandomWalkEnv, q_table: dict, epsilon: float) -> dict:
    """Read the final policy off the learned action values.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        epsilon: See Section 7.1 and the implementation for its role.

    Returns:
        epsilon greedy policy as defined in Section 7.1."""
    # internal helper for the epsilon greedy policy step (book Chapter 7).
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

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        epsilon: See Section 7.1 and the implementation for its role.
        rng: See Section 7.1 and the implementation for its role.

    Returns:
        n step sarsa rollout as defined in Section 7.1."""
    # internal helper for the n step sarsa rollout step (book Chapter 7).
    # states: all states in the MDP (book Chapter 7).
    states = [env.reset()]
    # action indexes: list[int] attribute/state used by the algorithm (book Chapter 7).
    action_indexes: list[int] = []
    # rewards: per-step reward accumulation buffer (book Chapter 7).
    rewards: list[float] = []

    action_indexes.append(_action_index(_action_probs(env, q_table, states[0], epsilon), rng))

    done = False
    # Repeat while the condition holds: the algorithm's termination loop.
    while not done:
        # action: action selected for the current step (book Chapter 7).
        action = env.ACTIONS[action_indexes[-1]]
        # (reward, next_state, done): result of env.step(...) (book Chapter 7).
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
    """Back up every decision time in the episode with its n-step Sarsa target.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        states: See Section 7.1 and the implementation for its role.
        action_indexes: See Section 7.1 and the implementation for its role.
        rewards: See Section 7.1 and the implementation for its role.
        n: See Section 7.1 and the implementation for its role.
        alpha: See Section 7.1 and the implementation for its role.
        gamma: See Section 7.1 and the implementation for its role."""
    # internal helper for the n step sarsa sweep step (book Chapter 7).
    # total steps: result of len(...) (book Chapter 7).
    total_steps = len(rewards)
    # Iterate over tau: one pass per index (book algorithm).
    for tau in range(total_steps):
        target = 0.0
        # Iterate over k: one pass per index (book algorithm).
        for k in range(n):
            if tau + k >= total_steps:
                break
            # target: add value computed from its operands into the running value (book Chapter 7).
            target += gamma**k * rewards[tau + k]
        if tau + n < total_steps and states[tau + n] is not None:
            # next s: element selected from the indexed collection (book Chapter 7).
            next_s = states[tau + n]
            # target: add value computed from its operands into the running value (book Chapter 7).
            target += gamma**n * q_table[(next_s, env.ACTIONS[action_indexes[tau + n]])]
        # tau state: element selected from the indexed collection (book Chapter 7).
        tau_state = states[tau]
        # tau action: element selected from the indexed collection (book Chapter 7).
        tau_action = env.ACTIONS[action_indexes[tau]]
        # q_table[tau_state, tau_action]: add value computed from its operands into the running ...
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
    # rng: seeded random generator so runs are reproducible (book Chapter 7).
    rng = np.random.default_rng(seed)
    # q table: action-value estimates Q(s, a) (book Chapter 7).
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    for _ in range(n_episodes):
        # Each episode supplies its own trajectory; the backups follow it.
        states, action_indexes, rewards = _n_step_sarsa_rollout(env, q_table, epsilon, rng)
        _n_step_sarsa_sweep(env, q_table, states, action_indexes, rewards, n, alpha, gamma)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


class _TreeRollout(NamedTuple):
    """The per-step quantities needed by the n-step tree-backup target.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7)."""

    # states: all states in the MDP (book Chapter 7).
    states: list
    # action indexes: list[int] attribute/state used by the algorithm (book Chapter 7).
    action_indexes: list[int]
    # rewards: per-step reward accumulation buffer (book Chapter 7).
    rewards: list[float]
    # deltas: per-step TD errors or differences (book Chapter 7).
    deltas: list[float]
    # pis: list[float] attribute/state used by the algorithm (book Chapter 7).
    pis: list[float]
    # q stored: list[float] attribute/state used by the algorithm (book Chapter 7).
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

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        epsilon: See Section 7.1 and the implementation for its role.
        gamma: See Section 7.1 and the implementation for its role.
        rng: See Section 7.1 and the implementation for its role.

    Returns:
        tree backup rollout as defined in Section 7.1."""
    # internal helper for the tree backup rollout step (book Chapter 7).
    # states: all states in the MDP (book Chapter 7).
    states = [env.reset()]
    # rewards: per-step reward accumulation buffer (book Chapter 7).
    rewards: list[float] = []
    # deltas: per-step TD errors or differences (book Chapter 7).
    deltas: list[float] = []
    # pis: list[float] attribute/state used by the algorithm (book Chapter 7).
    pis: list[float] = []
    # q stored: list[float] attribute/state used by the algorithm (book Chapter 7).
    q_stored: list[float] = []

    # probs0: result of _action_probs(...) (book Chapter 7).
    probs0 = _action_probs(env, q_table, states[0], epsilon)
    # action indexes: collection of the listed elements (book Chapter 7).
    action_indexes = [_action_index(probs0, rng)]
    q_stored.append(q_table[(states[0], env.ACTIONS[action_indexes[0]])])

    done = False
    # Repeat while the condition holds: the algorithm's termination loop.
    while not done:
        # action: action selected for the current step (book Chapter 7).
        action = env.ACTIONS[action_indexes[-1]]
        # (reward, next_state, done): result of env.step(...) (book Chapter 7).
        reward, next_state, done = env.step(action)
        rewards.append(float(reward))
        if done:
            deltas.append(rewards[-1] - q_stored[-1])
            break

        # q next: array built from the supplied values (book Chapter 7).
        q_next = np.array([q_table[(next_state, a)] for a in env.ACTIONS])
        # probs next: result of _epsilon_greedy_probs(...) (book Chapter 7).
        probs_next = _epsilon_greedy_probs(q_next, epsilon)
        deltas.append(
            rewards[-1] + gamma * float(np.dot(probs_next, q_next)) - q_stored[-1]
        )
        states.append(next_state)
        # a next: result of _action_index(...) (book Chapter 7).
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
    """Back up every decision time with the n-step expected (tree) target.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        rollout: See Section 7.1 and the implementation for its role.
        n: See Section 7.1 and the implementation for its role.
        alpha: See Section 7.1 and the implementation for its role.
        gamma: See Section 7.1 and the implementation for its role."""
    # internal helper for the tree backup sweep step (book Chapter 7).
    # total steps: result of len(...) (book Chapter 7).
    total_steps = len(rollout.rewards)
    # Iterate over tau: one pass per index (book algorithm).
    for tau in range(total_steps):
        # target: target value for the current update (book Chapter 7).
        target = rollout.q_stored[tau]
        z = 1.0
        # Iterate over k: one pass per index (book algorithm).
        for k in range(tau, min(tau + n, total_steps)):
            # target: add value computed from its operands into the running value (book Chapter 7).
            target += z * rollout.deltas[k]
            if k < len(rollout.pis):
                # z: multiply value computed from its operands into the running value (book ...
                z *= gamma * rollout.pis[k]
        # s tau: element selected from the indexed collection (book Chapter 7).
        s_tau = rollout.states[tau]
        # a tau: element selected from the indexed collection (book Chapter 7).
        a_tau = env.ACTIONS[rollout.action_indexes[tau]]
        # q_table[s_tau, a_tau]: add value computed from its operands into the running value ...
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

    Tree backup replaces importance sampling with an expected target over the
    non-sampled actions, using their behavior probabilities. In this demo the
    target policy is the epsilon-greedy policy w.r.t. q_table (50/50 for
    epsilon=0.5, uniform on the very first passes), and the behavior policy is
    the same epsilon-greedy rule: with target == behavior the backup reduces to
    expected Sarsa, so the defining off-policy case is not exercised here.

    Returns:
        (policy, q_table): policy[state] = probability vector over ACTIONS.

    Book: Sutton & Barto (2nd ed.), Section 7.5 (Chapter 7).

    Args:
        env: See Section 7.5 and the implementation for its role.
        n: See Section 7.5 and the implementation for its role.
        alpha: See Section 7.5 and the implementation for its role.
        epsilon: See Section 7.5 and the implementation for its role.
        gamma: See Section 7.5 and the implementation for its role.
        n_episodes: See Section 7.5 and the implementation for its role.
        seed: See Section 7.5 and the implementation for its role."""
    _check_epsilon(epsilon)
    _check_n(n)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    # rng: seeded random generator so runs are reproducible (book Chapter 7).
    rng = np.random.default_rng(seed)
    # q table: action-value estimates Q(s, a) (book Chapter 7).
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
    """Run one episode: traces carry each TD error back to earlier visits.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        q_table: See Section 7.1 and the implementation for its role.
        alpha: See Section 7.1 and the implementation for its role.
        epsilon: See Section 7.1 and the implementation for its role.
        gamma: See Section 7.1 and the implementation for its role.
        lam: See Section 7.1 and the implementation for its role.
        rng: See Section 7.1 and the implementation for its role."""
    # internal helper for the sarsa lambda episode step (book Chapter 7).
    # traces: collection built by comprehension (book Chapter 7).
    traces = {pair: 0.0 for pair in q_table}
    # state: current state of the environment or MDP (book Chapter 7).
    state = env.reset()
    # a idx: index of the selected action (book Chapter 7).
    a_idx = _action_index(_action_probs(env, q_table, state, epsilon), rng)
    done = False
    # Repeat while the condition holds: the algorithm's termination loop.
    while not done:
        # action: action selected for the current step (book Chapter 7).
        action = env.ACTIONS[a_idx]
        # (reward, next_state, done): result of env.step(...) (book Chapter 7).
        reward, next_state, done = env.step(action)
        if done:
            # delta: TD error or update difference for this step (book Chapter 7).
            delta = reward - q_table[(state, action)]
            next_a_idx = a_idx
        else:
            # next a idx: index of the action in the next state (book Chapter 7).
            next_a_idx = _action_index(_action_probs(env, q_table, next_state, epsilon), rng)
            # delta: TD error or update difference for this step (book Chapter 7).
            delta = (
                reward
                + gamma * q_table[(next_state, env.ACTIONS[next_a_idx])]
                - q_table[(state, action)]
            )

        # traces[state, action]: add literal constant value into the running value (book Chapter
        traces[(state, action)] += 1.0
        # Iterate over pair: each item in q_table (book algorithm).
        for pair in q_table:
            # q_table[pair]: add value computed from its operands into the running value (book ...
            q_table[pair] += alpha * delta * traces[pair]
            # traces[pair]: multiply value computed from its operands into the running value ...
            traces[pair] *= gamma * lam
        # state: current state of the environment or MDP (book Chapter 7).
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

    Book: Sutton & Barto (2nd ed.), Section 12.2 (Chapter 7).

    Args:
        env: See Section 12.2 and the implementation for its role.
        lam: See Section 12.2 and the implementation for its role.
        alpha: See Section 12.2 and the implementation for its role.
        epsilon: See Section 12.2 and the implementation for its role.
        gamma: See Section 12.2 and the implementation for its role.
        n_episodes: See Section 12.2 and the implementation for its role.
        seed: See Section 12.2 and the implementation for its role."""
    # carry out the sarsa lambda step (book Chapter 7).
    _check_epsilon(epsilon)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    # rng: seeded random generator so runs are reproducible (book Chapter 7).
    rng = np.random.default_rng(seed)
    # q table: action-value estimates Q(s, a) (book Chapter 7).
    q_table = {(s, a): 0.0 for s in env.STATES for a in env.ACTIONS}

    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        _sarsa_lambda_episode(env, q_table, alpha, epsilon, gamma, lam, rng)

    return _epsilon_greedy_policy(env, q_table, epsilon), q_table


# ---------------------------------------------------------------------------
# n-step TD prediction (Sec. 7.1, Example 7.1 / Figure 7.2)
# ---------------------------------------------------------------------------

# states 19: result of tuple(...) (book Chapter 7).
_STATES_19 = tuple(chr(ord("A") + i) for i in range(19))


def _n_step_td_sweep(
    values: dict,
    states: list,
    rewards: list[float],
    n: int,
    alpha: float,
    gamma: float,
) -> None:
    """One episode of the n-step TD prediction box (Sec. 7.1), in-place.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        values: See Section 7.1 and the implementation for its role.
        states: See Section 7.1 and the implementation for its role.
        rewards: See Section 7.1 and the implementation for its role.
        n: See Section 7.1 and the implementation for its role.
        alpha: See Section 7.1 and the implementation for its role.
        gamma: See Section 7.1 and the implementation for its role."""
    # internal helper for the n step td sweep step (book Chapter 7).
    # total: running total accumulator (book Chapter 7).
    total = len(rewards)
    # powers: collection built by comprehension (book Chapter 7).
    powers = [gamma**i for i in range(total + 1)]
    # Iterate over tau: one pass per index (book algorithm).
    for tau in range(total):
        target = 0.0
        # Iterate over k: one pass per index (book algorithm).
        for k in range(tau, min(tau + n, total)):
            # target: add value computed from its operands into the running value (book Chapter 7).
            target += powers[k - tau] * rewards[k]
        if tau + n < total:
            # target: add value computed from its operands into the running value (book Chapter 7).
            target += powers[n] * values[states[tau + n]]
        # values[states[tau]]: add value computed from its operands into the running value (book
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

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        env: See Section 7.1 and the implementation for its role.
        n: See Section 7.1 and the implementation for its role.
        alpha: See Section 7.1 and the implementation for its role.
        n_episodes: See Section 7.1 and the implementation for its role.
        gamma: See Section 7.1 and the implementation for its role.
        init_value: See Section 7.1 and the implementation for its role.
        seed: See Section 7.1 and the implementation for its role.

    Returns:
        n step td prediction as defined in Section 7.1."""
    # carry out the n step td prediction step (book Chapter 7).
    _check_n(n)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if env is None:
        # env: environment instance being controlled (book Chapter 7).
        env = RandomWalkEnv(seed=seed)
    # rng: seeded random generator so runs are reproducible (book Chapter 7).
    rng = np.random.default_rng(seed)
    # values: state-value estimates V(s) (book Chapter 7).
    values = {s: float(init_value) for s in env.STATES}
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # states: all states in the MDP (book Chapter 7).
        states = [env.reset()]
        # rewards: per-step reward accumulation buffer (book Chapter 7).
        rewards = []
        done = False
        # Repeat while the condition holds: the algorithm's termination loop.
        while not done:
            # action: action selected for the current step (book Chapter 7).
            action = "left" if rng.random() < 0.5 else "right"
            # (reward, next_state, done): result of env.step(...) (book Chapter 7).
            reward, next_state, done = env.step(action)
            rewards.append(float(reward))
            states.append(next_state)
        _n_step_td_sweep(values, states, rewards, n, alpha, gamma)
    return values


def _sample_walks(repeats: int, n_episodes: int, seed: int):
    """Pre-sample the walk sets used for every (n, alpha) setting.

    The book used the same sets of walks for all parameter settings,
    controlling for walk difficulty. Each walk is a (states, rewards) pair.

    Book: Sutton & Barto (2nd ed.), Section 7.1 (Chapter 7).

    Args:
        repeats: See Section 7.1 and the implementation for its role.
        n_episodes: See Section 7.1 and the implementation for its role.
        seed: See Section 7.1 and the implementation for its role."""
    # internal helper for the sample walks step (book Chapter 7).
    # rng: seeded random generator so runs are reproducible (book Chapter 7).
    rng = np.random.default_rng(seed)
    # walks: collection of the listed elements (book Chapter 7).
    walks = []
    # Iterate over _: one pass per index (book algorithm).
    for _ in range(repeats):
        # rep: collection of the listed elements (book Chapter 7).
        rep = []
        # Iterate over _: one pass per episode (book algorithm).
        for _ in range(n_episodes):
            # env: environment instance being controlled (book Chapter 7).
            env = RandomWalkEnv(
                n_states=19, left_reward=-1.0, right_reward=1.0,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            # states: all states in the MDP (book Chapter 7).
            states = [env.reset()]
            # rewards: per-step reward accumulation buffer (book Chapter 7).
            rewards = []
            done = False
            # Repeat while the condition holds: the algorithm's termination loop.
            while not done:
                # action: action selected for the current step (book Chapter 7).
                action = "left" if rng.random() < 0.5 else "right"
                # (reward, next_state, done): result of env.step(...) (book Chapter 7).
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

    Book: Sutton & Barto (2nd ed.), Figure 7.2 (Chapter 7).

    Args:
        ns: See Figure 7.2 and the implementation for its role.
        alphas: See Figure 7.2 and the implementation for its role.
        repeats: See Figure 7.2 and the implementation for its role.
        n_episodes: See Figure 7.2 and the implementation for its role.
        gamma: See Figure 7.2 and the implementation for its role.
        seed: See Figure 7.2 and the implementation for its role.

    Returns:
        run n step prediction experiment as defined in Figure 7.2."""
    # run the experiment and collect the figure data (book Chapter 7).
    if repeats < 1 or n_episodes < 1:
        raise ValueError("repeats and n_episodes must be >= 1")
    # Iterate over n: each item in ns (book algorithm).
    for n in ns:
        _check_n(n)
    # true values: result of RandomWalkEnv(n_states=19, left_reward=-1.0, ...
    true_values = RandomWalkEnv(n_states=19, left_reward=-1.0, right_reward=1.0).true_values()
    # states 19: result of tuple(...) (book Chapter 7).
    states_19 = tuple(true_values)
    # walks: result of _sample_walks(...) (book Chapter 7).
    walks = _sample_walks(repeats, n_episodes, seed)
    # curves: averaged learning curves by configuration (book Chapter 7).
    curves: dict[int, list[float]] = {}
    # Iterate over n: each item in ns (book algorithm).
    for n in ns:
        # curve: collection of the listed elements (book Chapter 7).
        curve = []
        # Iterate over alpha: each item in alphas (book algorithm).
        for alpha in alphas:
            total_error = 0.0
            # Iterate over rep: each item in walks (book algorithm).
            for rep in walks:
                # values: state-value estimates V(s) (book Chapter 7).
                values = dict.fromkeys(states_19, 0.0)
                # Iterate over (states, rewards): each item in rep (book algorithm).
                for states, rewards in rep:
                    _n_step_td_sweep(values, states, rewards, n, alpha, gamma)
                    # err: floating-point conversion (book Chapter 7).
                    err = float(
                        np.sqrt(np.mean([(values[s] - true_values[s]) ** 2 for s in states_19]))
                    )
                    # total_error: add reference to an existing value into the running value ...
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

    Book: Sutton & Barto (2nd ed.), Figure 7.2 (Chapter 7).

    Args:
        save_path: See Figure 7.2 and the implementation for its role.
        show: See Figure 7.2 and the implementation for its role.
        repeats: See Figure 7.2 and the implementation for its role.
        n_episodes: See Figure 7.2 and the implementation for its role.
        seed: See Figure 7.2 and the implementation for its role.
        ns: See Figure 7.2 and the implementation for its role.
        alphas: See Figure 7.2 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 7).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # results: collected results keyed by configuration (book Chapter 7).
    results = run_n_step_prediction_experiment(
        ns=ns, alphas=alphas, repeats=repeats, n_episodes=n_episodes, seed=seed
    )
    # (fig, ax): result of plt.subplots(...) (book Chapter 7).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Iterate over n: each item in results["ns"] (book algorithm).
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

