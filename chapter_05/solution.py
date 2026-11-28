"""Solution implementation for Chapter 5 - Monte Carlo Methods.

Reference: Sutton & Barto (2nd Ed), Chapter 5

Implements: Alg 5.1 (first-visit / every-visit MC prediction),
Alg 5.3 (on-policy MC control, epsilon-greedy),
Alg 5.4 (off-policy control, ordinary importance sampling),
Alg 5.5 (off-policy control, weighted importance sampling),
Exercise 5.1 (Blackjack (21) environment + MC control).

Policies are probability vectors over ACTIONS; the prediction and control
functions take a policy as a callable ``policy_fn(state) -> probs`` (a
probability vector) and any environment implementing ``reset() -> observation``
and ``step(action) -> (reward, observation, done)`` — see ``BlackjackEnv``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rl_envs.blackjack import ACTIONS, BlackjackEnv


def _validate_policy_fn(policy_fn: Callable) -> None:
    """validate policy fn (reference implementation).


    Args:
        policy_fn: See the relevant book section and function body.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
    # internal helper for the validate policy fn step (book Chapter 5).
    if not callable(policy_fn):
        raise TypeError("policy_fn must be callable")


def _actions_for(env) -> tuple:
    """Action table for an environment (falls back to blackjack's).

    Both ``BlackjackEnv`` (ACTIONS = ('hit', 'stick'), index 0/1) and
    ``RacetrackEnv`` (9 indexed acceleration tuples) expose a class-level
    ``ACTIONS`` tuple whose entries are what ``env.step`` expects.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        env: See Algorithm 5.1 and the implementation for its role.

    Returns:
        actions for as defined in Algorithm 5.1."""
    # internal helper for the actions for step (book Chapter 5).
    return tuple(getattr(env, "ACTIONS", ACTIONS))


def _sample_action(state, policy_fn, rng: np.random.Generator, actions: tuple) -> int:
    """sample action (reference implementation).


    Args:
        state: See the relevant book section and function body.
        policy_fn: See the relevant book section and function body.
        rng: See the relevant book section and function body.
        actions: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
    # internal helper for the sample action step (book Chapter 5).
    # probs: action probabilities from the current policy (book Chapter 5).
    probs = np.asarray(policy_fn(state), dtype=np.float64)
    # n: result of len(...) (book Chapter 5).
    n = len(actions)
    if probs.shape != (n,):
        raise ValueError(f"policy_fn must return a vector of length {n}")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("policy probabilities must sum to 1")
    return int(rng.choice(n, p=probs))


def _run_episode(
    env,
    policy_fn,
    rng: np.random.Generator,
    max_steps: int | None = None,
) -> list[tuple]:
    """Return [(state, action_index, reward), ...] for one episode.

    `max_steps` bounds non-terminating environments; None keeps the book's
    episodic assumption and allows unlimited steps.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        env: See Algorithm 5.1 and the implementation for its role.
        policy_fn: See Algorithm 5.1 and the implementation for its role.
        rng: See Algorithm 5.1 and the implementation for its role.
        max_steps: See Algorithm 5.1 and the implementation for its role.

    Returns:
        run episode as defined in Algorithm 5.1."""
    # run the experiment and collect the figure data (book Chapter 5).
    # actions: all actions available in the MDP (book Chapter 5).
    actions = _actions_for(env)
    # state: current state of the environment or MDP (book Chapter 5).
    state = env.reset()
    # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
    trajectory = []
    steps = 0
    # Repeat while the condition holds: the algorithm's termination loop.
    while True:
        # action index: index of the selected action (book Chapter 5).
        action_index = _sample_action(state, policy_fn, rng, actions)
        # (reward, observation, done): result of env.step(...) (book Chapter 5).
        reward, observation, done = env.step(actions[action_index])
        trajectory.append((state, action_index, reward))
        if done:
            return trajectory
        state = observation
        # steps: add literal constant value into the running value (book Chapter 5).
        steps += 1
        if max_steps is not None and steps >= max_steps:
            return trajectory


def _returns_from_rewards(rewards: list[float], gamma: float) -> list[float]:
    """returns from rewards (reference implementation).


    Args:
        rewards: See the relevant book section and function body.
        gamma: See the relevant book section and function body.

    Returns:
        Value produced by the implementation (see the book section).

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
    # internal helper for the returns from rewards step (book Chapter 5).
    # returns: accumulated returns used as update targets (book Chapter 5).
    returns = [0.0] * len(rewards)
    running = 0.0
    # Iterate over t: one pass per index (book algorithm).
    for t in range(len(rewards) - 1, -1, -1):
        # running: running average accumulator (book Chapter 5).
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _accumulate_first_visit_q(
    trajectory: list[tuple],
    gamma: float,
    n_actions: int,
    returns_by_pair: dict,
    q_table: dict,
) -> None:
    """Add each first-visited (state, action)'s return to its Q average.

    A pair seen twice in one episode contributes only once: first-visit MC
    avoids counting the same reward stream from the same starting point.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        trajectory: See Algorithm 5.1 and the implementation for its role.
        gamma: See Algorithm 5.1 and the implementation for its role.
        n_actions: See Algorithm 5.1 and the implementation for its role.
        returns_by_pair: See Algorithm 5.1 and the implementation for its role.
        q_table: See Algorithm 5.1 and the implementation for its role."""
    # internal helper for the accumulate first visit q step (book Chapter 5).
    # rewards: per-step reward accumulation buffer (book Chapter 5).
    rewards = [r for _, _, r in trajectory]
    # returns: accumulated returns used as update targets (book Chapter 5).
    returns = _returns_from_rewards(rewards, gamma)
    # seen: set attribute/state used by the algorithm (book Chapter 5).
    seen: set = set()
    # Iterate over (t, (state, action, _)): indexed entries (book algorithm).
    for t, (state, action, _) in enumerate(trajectory):
        if (state, action) in seen:
            continue
        seen.add((state, action))
        # pair: collection of the listed elements (book Chapter 5).
        pair = (state, action)
        returns_by_pair.setdefault(pair, []).append(returns[t])
        q_table.setdefault(state, np.zeros(n_actions))
        q_table[state][action] = float(np.mean(returns_by_pair[pair]))


def mc_first_visit_prediction(
    env,
    policy_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict:
    """First-visit MC prediction (Algorithm 5.1). Book Section 5.1.

    Estimates v_pi(s) as the average of returns following FIRST visits to s.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        env: See Algorithm 5.1 and the implementation for its role.
        policy_fn: See Algorithm 5.1 and the implementation for its role.
        n_episodes: See Algorithm 5.1 and the implementation for its role.
        gamma: See Algorithm 5.1 and the implementation for its role.
        seed: See Algorithm 5.1 and the implementation for its role.
        max_steps: See Algorithm 5.1 and the implementation for its role.

    Returns:
        mc first visit prediction as defined in Algorithm 5.1."""
    # carry out the mc first visit prediction step (book Chapter 5).
    _validate_policy_fn(policy_fn)
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)
    # returns by state: dict attribute/state used by the algorithm (book Chapter 5).
    returns_by_state: dict = {}
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
        trajectory = _run_episode(env, policy_fn, rng, max_steps)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in trajectory]
        # returns: accumulated returns used as update targets (book Chapter 5).
        returns = _returns_from_rewards(rewards, gamma)
        # seen: set attribute/state used by the algorithm (book Chapter 5).
        seen: set = set()
        # Iterate over (t, (state, _, _)): indexed entries (book algorithm).
        for t, (state, _, _) in enumerate(trajectory):
            if state in seen:
                continue
            seen.add(state)
            returns_by_state.setdefault(state, []).append(returns[t])
    return {s: float(np.mean(g)) for s, g in returns_by_state.items()}


def mc_every_visit_prediction(
    env,
    policy_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict:
    """Every-visit MC prediction (Algorithm 5.1). Book Section 5.1.

    Estimates v_pi(s) as the average of returns following ALL visits to s.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        env: See Algorithm 5.1 and the implementation for its role.
        policy_fn: See Algorithm 5.1 and the implementation for its role.
        n_episodes: See Algorithm 5.1 and the implementation for its role.
        gamma: See Algorithm 5.1 and the implementation for its role.
        seed: See Algorithm 5.1 and the implementation for its role.
        max_steps: See Algorithm 5.1 and the implementation for its role.

    Returns:
        mc every visit prediction as defined in Algorithm 5.1."""
    # carry out the mc every visit prediction step (book Chapter 5).
    _validate_policy_fn(policy_fn)
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)
    # returns by state: dict attribute/state used by the algorithm (book Chapter 5).
    returns_by_state: dict = {}
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
        trajectory = _run_episode(env, policy_fn, rng, max_steps)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in trajectory]
        # returns: accumulated returns used as update targets (book Chapter 5).
        returns = _returns_from_rewards(rewards, gamma)
        # Iterate over (t, (state, _, _)): indexed entries (book algorithm).
        for t, (state, _, _) in enumerate(trajectory):
            returns_by_state.setdefault(state, []).append(returns[t])
    return {s: float(np.mean(g)) for s, g in returns_by_state.items()}


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    """Epsilon-greedy probabilities (the closest epsilon-soft policy to greedy).

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        q_values: See Algorithm 5.1 and the implementation for its role.
        epsilon: See Algorithm 5.1 and the implementation for its role.

    Returns:
        epsilon greedy probs as defined in Algorithm 5.1."""
    # internal helper for the epsilon greedy probs step (book Chapter 5).
    # q values: action-value estimates for the state (book Chapter 5).
    q_values = np.asarray(q_values, dtype=np.float64)
    # n: result of len(...) (book Chapter 5).
    n = len(q_values)
    # greedy: result of np.flatnonzero(...) (book Chapter 5).
    greedy = np.flatnonzero(q_values == q_values.max())
    # probs: action probabilities from the current policy (book Chapter 5).
    probs = np.full(n, epsilon / n, dtype=np.float64)
    # probs[greedy]: add value computed from its operands into the running value (book Chapter 5).
    probs[greedy] += (1.0 - epsilon) / len(greedy)
    return probs


def mc_control_epsilon_greedy(
    env,
    n_episodes: int = 20_000,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
    max_steps: int | None = None,
) -> tuple[dict, dict, list]:
    """On-policy first-visit MC control for epsilon-soft policies (Algorithm 5.3).

    Book Section 5.4. The behavior and target policy is the epsilon-greedy
    policy with respect to Q; the greedy action gets 1 - epsilon + epsilon/|A|,
    the others share epsilon/|A|.

    Returns:
        (policy, Q, episode_rewards): policy[state] = probability vector over
        ACTIONS; q_table[state] = action-value array of shape (2,).

    Book: Sutton & Barto (2nd ed.), Algorithm 5.3 (Chapter 5).

    Args:
        env: See Algorithm 5.3 and the implementation for its role.
        n_episodes: See Algorithm 5.3 and the implementation for its role.
        epsilon: See Algorithm 5.3 and the implementation for its role.
        gamma: See Algorithm 5.3 and the implementation for its role.
        seed: See Algorithm 5.3 and the implementation for its role.
        max_steps: See Algorithm 5.3 and the implementation for its role."""
    # carry out the mc control epsilon greedy step (book Chapter 5).
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    # actions: all actions available in the MDP (book Chapter 5).
    actions = _actions_for(env)
    # n actions: number of available actions (book Chapter 5).
    n_actions = len(actions)
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)

    # q table: action-value estimates Q(s, a) (book Chapter 5).
    q_table: dict = {}
    # returns by pair: dict attribute/state used by the algorithm (book Chapter 5).
    returns_by_pair: dict = {}

    def behavior_probs(state):
        """behavior probs (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
        # carry out the behavior probs step (book Chapter 5).
        if state not in q_table:
            return np.ones(n_actions) / n_actions
        return _epsilon_greedy_probs(q_table[state], epsilon)

    # episode rewards: list[float] attribute/state used by the algorithm (book Chapter 5).
    episode_rewards: list[float] = []
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
        trajectory = _run_episode(env, behavior_probs, rng, max_steps)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))
        _accumulate_first_visit_q(
            trajectory, gamma, n_actions, returns_by_pair, q_table
        )

    # policy: policy mapping states to action probabilities (book Chapter 5).
    policy = {s: _epsilon_greedy_probs(q, epsilon) for s, q in q_table.items()}
    return policy, q_table, episode_rewards


def _importance_ratios(
    trajectory: list[tuple],
    target_fn: Callable,
    behavior_fn: Callable,
) -> list[float]:
    """Importance sampling ratios rho_{t:T-1} for each timestep.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        trajectory: See Algorithm 5.1 and the implementation for its role.
        target_fn: See Algorithm 5.1 and the implementation for its role.
        behavior_fn: See Algorithm 5.1 and the implementation for its role.

    Returns:
        importance ratios as defined in Algorithm 5.1."""
    # internal helper for the importance ratios step (book Chapter 5).
    # n: result of len(...) (book Chapter 5).
    n = len(trajectory)
    # ratios: value computed from its operands (book Chapter 5).
    ratios = [1.0] * n
    # Iterate over t: one pass per index (book algorithm).
    for t in range(n - 1, -1, -1):
        # (state, action, _): element selected from the indexed collection (book Chapter 5).
        state, action, _ = trajectory[t]
        # pi prob: floating-point conversion (book Chapter 5).
        pi_prob = float(target_fn(state)[action])
        # b prob: floating-point conversion (book Chapter 5).
        b_prob = float(behavior_fn(state)[action])
        # ratio: value chosen conditionally (book Chapter 5).
        ratio = pi_prob / b_prob if b_prob > 0.0 else 0.0
        ratios[t] = ratio * (ratios[t + 1] if t + 1 < n else 1.0)
    return ratios


def mc_off_policy_prediction(
    env,
    target_fn: Callable,
    behavior_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    weighted: bool = True,
    first_visit: bool = True,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict:
    """Off-policy MC prediction via importance sampling (Algorithms 5.4/5.5).

    Book Section 5.5 (ordinary IS, unbiased; Eq. 5.4) and Section 5.5/5.6
    (weighted IS; Eq. 5.5). Behavior policy b must cover the target policy pi.

    Args:
        env: Environment.
        target_fn: Target policy pi(state) -> probability vector.
        behavior_fn: Behavior policy b(state) -> probability vector.
        n_episodes: Number of episodes to sample.
        gamma: Discount factor.
        weighted: If True use weighted IS; otherwise ordinary IS.
        first_visit: If True, only first visits within an episode count.
        seed: Optional seed.

    Returns:
        dict state -> estimated v_pi(s).
    """
    # carry out the mc off policy prediction step (book Chapter 5).
    _validate_policy_fn(target_fn)
    _validate_policy_fn(behavior_fn)
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)

    # scaled: dict attribute/state used by the algorithm (book Chapter 5).
    scaled: dict = {}
    # weights sum: dict attribute/state used by the algorithm (book Chapter 5).
    weights_sum: dict = {}

    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
        trajectory = _run_episode(env, behavior_fn, rng, max_steps)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in trajectory]
        # returns: accumulated returns used as update targets (book Chapter 5).
        returns = _returns_from_rewards(rewards, gamma)
        # ratios: result of _importance_ratios(...) (book Chapter 5).
        ratios = _importance_ratios(trajectory, target_fn, behavior_fn)
        # seen: set attribute/state used by the algorithm (book Chapter 5).
        seen: set = set()
        # Iterate over (t, (state, _, _)): indexed entries (book algorithm).
        for t, (state, _, _) in enumerate(trajectory):
            if first_visit and state in seen:
                continue
            seen.add(state)
            scaled.setdefault(state, []).append((ratios[t], returns[t]))
            weights_sum[state] = weights_sum.get(state, 0.0) + ratios[t]

    # estimates: dict attribute/state used by the algorithm (book Chapter 5).
    estimates: dict = {}
    # Iterate over (state, pairs): each item in scaled.items() (book algorithm).
    for state, pairs in scaled.items():
        if weighted:
            # den: element selected from the indexed collection (book Chapter 5).
            den = weights_sum[state]
            estimates[state] = (
                float(np.sum([w * g for w, g in pairs]) / den) if den > 0.0 else 0.0
            )
        else:
            estimates[state] = float(np.mean([w * g for w, g in pairs]))
    return estimates


# Floor on the weighted-IS visit counter, keeping its denominator finite.
_MIN_WEIGHTED_COUNT = 1e-8


def _backward_off_policy_update(
    trajectory: list[tuple],
    gamma: float,
    weighted: bool,
    n_actions: int,
    q_table: dict,
    counts: dict,
    behavior_probs,
    greedy_action,
) -> None:
    """Sweep one episode backwards, updating Q with importance ratios.

    The return G and the ratio w are built from the tail back to t=0, so the
    update order is fixed by the trajectory rather than by visit order.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        trajectory: See Algorithm 5.1 and the implementation for its role.
        gamma: See Algorithm 5.1 and the implementation for its role.
        weighted: See Algorithm 5.1 and the implementation for its role.
        n_actions: See Algorithm 5.1 and the implementation for its role.
        q_table: See Algorithm 5.1 and the implementation for its role.
        counts: See Algorithm 5.1 and the implementation for its role.
        behavior_probs: See Algorithm 5.1 and the implementation for its role.
        greedy_action: See Algorithm 5.1 and the implementation for its role."""
    g = 0.0
    w = 1.0
    for t in range(len(trajectory) - 1, -1, -1):
        # (state, action, reward): element selected from the indexed collection (book Chapter 5).
        state, action, reward = trajectory[t]
        # g: value computed from its operands (book Chapter 5).
        g = gamma * g + reward
        q_table.setdefault(state, np.zeros(n_actions))
        # pair counts: result of counts.setdefault(...) (book Chapter 5).
        pair_counts = counts.setdefault(state, np.zeros(n_actions))

        if weighted:
            # pair_counts[action]: add reference to an existing value into the running value ...
            pair_counts[action] += w
            pair_counts[action] = max(pair_counts[action], _MIN_WEIGHTED_COUNT)
            # q_table[state][action]: add value computed from its operands into the running ...
            q_table[state][action] += (w / pair_counts[action]) * (g - q_table[state][action])
        else:
            # ordinary IS: simple average of w*g over visits
            pair_counts[action] += 1.0
            # q_table[state][action]: add value computed from its operands into the running ...
            q_table[state][action] += (w * g - q_table[state][action]) / pair_counts[action]

        # target is greedy; stop bootstrapping from non-greedy suffixes
        if action != greedy_action(state):
            break
        # b prob: element selected from the indexed collection (book Chapter 5).
        b_prob = behavior_probs(state)[action]
        # w: value chosen conditionally (book Chapter 5).
        w = w * (1.0 / b_prob) if b_prob > 0.0 else 0.0


def _uniform_greedy_policy(q_table: dict, n_actions: int) -> dict:
    """Greedy policy with probability split evenly among maximizing actions.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        q_table: See Algorithm 5.1 and the implementation for its role.
        n_actions: See Algorithm 5.1 and the implementation for its role.

    Returns:
        uniform greedy policy as defined in Algorithm 5.1."""
    # internal helper for the uniform greedy policy step (book Chapter 5).
    # policy: policy mapping states to action probabilities (book Chapter 5).
    policy = {}
    # Iterate over state: each item in q_table (book algorithm).
    for state in q_table:
        # probs: action probabilities from the current policy (book Chapter 5).
        probs = np.zeros(n_actions)
        # greedy: result of np.flatnonzero(...) (book Chapter 5).
        greedy = np.flatnonzero(q_table[state] == q_table[state].max())
        probs[greedy] = 1.0 / len(greedy)
        policy[state] = probs
    return policy


def mc_control_off_policy(
    env,
    n_episodes: int = 20_000,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    weighted: bool = True,
    seed: int | None = None,
    max_steps: int | None = None,
) -> tuple[dict, dict, list]:
    """Off-policy MC control (Algorithms 5.4/5.5). Book Section 5.7.

    Behavior policy is epsilon-soft (epsilon-greedy w.r.t. Q); the target
    policy is greedy w.r.t. Q. Uses weighted importance sampling (default) or
    ordinary importance sampling (visit-count-normalized scaled returns).

    Returns:
        (policy, Q, episode_rewards): policy[state] = probability vector over ACTIONS.

    Book: Sutton & Barto (2nd ed.), Section 5.7 (Chapter 5).

    Args:
        env: See Section 5.7 and the implementation for its role.
        n_episodes: See Section 5.7 and the implementation for its role.
        epsilon: See Section 5.7 and the implementation for its role.
        gamma: See Section 5.7 and the implementation for its role.
        weighted: See Section 5.7 and the implementation for its role.
        seed: See Section 5.7 and the implementation for its role.
        max_steps: See Section 5.7 and the implementation for its role."""
    # carry out the mc control off policy step (book Chapter 5).
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    # actions: all actions available in the MDP (book Chapter 5).
    actions = _actions_for(env)
    # n actions: number of available actions (book Chapter 5).
    n_actions = len(actions)
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)

    # q table: action-value estimates Q(s, a) (book Chapter 5).
    q_table: dict = {}
    # counts: visit counts N(s, a) or N(a) (book Chapter 5).
    counts: dict = {}

    def behavior_probs(state):
        """behavior probs (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
        # carry out the behavior probs step (book Chapter 5).
        if state not in q_table:
            return np.ones(n_actions) / n_actions
        return _epsilon_greedy_probs(q_table[state], epsilon)

    def greedy_action(state) -> int:
        """greedy action (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
        # return the highest-valued action with tie-breaking (book Chapter 5).
        # q: action-value estimate Q(s, a) (book Chapter 5).
        q = q_table.get(state)
        if q is None:
            return 0
        return int(np.argmax(q))

    # episode rewards: list[float] attribute/state used by the algorithm (book Chapter 5).
    episode_rewards: list[float] = []
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
        trajectory = _run_episode(env, behavior_probs, rng, max_steps)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))
        _backward_off_policy_update(
            trajectory, gamma, weighted, n_actions, q_table, counts,
            behavior_probs, greedy_action,
        )

    return _uniform_greedy_policy(q_table, n_actions), q_table, episode_rewards


def blackjack_stick_on_threshold(stick_threshold: int = 20) -> Callable:
    """Deterministic policy: stick when player_sum >= stick_threshold.

    Book Example 5.1 used "stick on 20 or 21"; returned as a probability vector.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        stick_threshold: See Algorithm 5.1 and the implementation for its role.

    Returns:
        blackjack stick on threshold as defined in Algorithm 5.1."""

    # carry out the blackjack stick on threshold step (book Chapter 5).
    def policy_fn(state) -> np.ndarray:
        """policy fn (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
        # carry out the policy fn step (book Chapter 5).
        # player sum: element selected from the indexed collection (book Chapter 5).
        player_sum = state[0]
        if player_sum >= stick_threshold:
            return np.array([0.0, 1.0])
        return np.array([1.0, 0.0])

    return policy_fn


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 5)
# ---------------------------------------------------------------------------

# Blackjack figure grids span player sums 12-21 and dealer cards 1-10.
_BJ_PLAYER_SUM_MIN = 12
_BJ_PLAYER_SUM_MAX = 21
_BJ_DEALER_MIN = 1
_BJ_DEALER_MAX = 10
_BJ_GRID_SIDE = 10
# bj ace variants: collection of the listed elements (book Chapter 5).
_BJ_ACE_VARIANTS = ((True, "Usable ace"), (False, "No usable ace"))


def _blackjack_value_grid(values: dict, usable_ace: bool):
    """Build a (player_sum 12-21) x (dealer_showing 1-10) grid from MC values.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        values: See Algorithm 5.1 and the implementation for its role.
        usable_ace: See Algorithm 5.1 and the implementation for its role."""
    # internal helper for the blackjack value grid step (book Chapter 5).
    # grid: array filled with a constant value (book Chapter 5).
    grid = np.full((_BJ_GRID_SIDE, _BJ_GRID_SIDE), np.nan, dtype=np.float64)
    # Iterate over player_sum: one pass per index (book algorithm).
    for player_sum in range(_BJ_PLAYER_SUM_MIN, _BJ_PLAYER_SUM_MAX + 1):
        # Iterate over dealer: one pass per index (book algorithm).
        for dealer in range(_BJ_DEALER_MIN, _BJ_DEALER_MAX + 1):
            # state: current state of the environment or MDP (book Chapter 5).
            state = (player_sum, dealer, usable_ace)
            if state in values:
                grid[player_sum - _BJ_PLAYER_SUM_MIN, dealer - _BJ_DEALER_MIN] = values[state]
    return grid


def _blackjack_q_grid(q_table: dict, usable: bool, reducer: Callable[[np.ndarray], float]):
    """Apply `reducer` to each state's action values on the blackjack grid.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        q_table: See Algorithm 5.1 and the implementation for its role.
        usable: See Algorithm 5.1 and the implementation for its role.
        reducer: See Algorithm 5.1 and the implementation for its role."""
    # internal helper for the blackjack q grid step (book Chapter 5).
    # grid: array filled with a constant value (book Chapter 5).
    grid = np.full((_BJ_GRID_SIDE, _BJ_GRID_SIDE), np.nan, dtype=np.float64)
    # Iterate over player_sum: one pass per index (book algorithm).
    for player_sum in range(_BJ_PLAYER_SUM_MIN, _BJ_PLAYER_SUM_MAX + 1):
        # Iterate over dealer: one pass per index (book algorithm).
        for dealer in range(_BJ_DEALER_MIN, _BJ_DEALER_MAX + 1):
            # state: current state of the environment or MDP (book Chapter 5).
            state = (player_sum, dealer, usable)
            if state in q_table:
                grid[player_sum - _BJ_PLAYER_SUM_MIN, dealer - _BJ_DEALER_MIN] = reducer(
                    q_table[state]
                )
    return grid


def plot_blackjack_value_functions(
    save_path: str | None = None,
    show: bool = False,
    n_episodes: int = 500_000,
    seed: int = 0,
):
    """Blackjack state-value functions (Example 5.1, Fig. 5.1).

    Sutton & Barto (2nd ed.), Sec. 5.1, Fig. 5.1: approximate state-value
    functions for the blackjack policy that sticks on 20 or 21, computed by
    Monte Carlo prediction (first-visit MC, Alg. 5.1) from sample episodes.
    States are (player sum, dealer showing, usable ace). Shown for player sums
    12-21 and dealer showing Ace(1)-10, separately for usable and no usable ace.
    Values are expected returns in [-1, 1]. Episode count controls accuracy; book
    used 10,000 and 500,000 episodes.

    Book: Sutton & Barto (2nd ed.), Figure 5.1 (Chapter 5).

    Args:
        save_path: See Figure 5.1 and the implementation for its role.
        show: See Figure 5.1 and the implementation for its role.
        n_episodes: See Figure 5.1 and the implementation for its role.
        seed: See Figure 5.1 and the implementation for its role."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # env: environment instance being controlled (book Chapter 5).
    env = BlackjackEnv(seed=seed)
    # policy: policy mapping states to action probabilities (book Chapter 5).
    policy = blackjack_stick_on_threshold(20)
    # values: state-value estimates V(s) (book Chapter 5).
    values = mc_first_visit_prediction(env, policy, n_episodes=n_episodes, seed=seed)

    # (X, Y): result of np.meshgrid(...) (book Chapter 5).
    X, Y = np.meshgrid(
        np.arange(_BJ_DEALER_MIN, _BJ_DEALER_MAX + 1),
        np.arange(_BJ_PLAYER_SUM_MIN, _BJ_PLAYER_SUM_MAX + 1),
    )

    # fig: result of plt.figure(...) (book Chapter 5).
    fig = plt.figure(figsize=(11, 4.8))
    # Iterate over (idx, (usable, title)): indexed entries (book algorithm).
    for idx, (usable, title) in enumerate(_BJ_ACE_VARIANTS, start=1):
        # ax: result of fig.add_subplot(...) (book Chapter 5).
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        # z: result of _blackjack_value_grid(...) (book Chapter 5).
        Z = _blackjack_value_grid(values, usable)
        # z plot: result of np.where(...) (book Chapter 5).
        Z_plot = np.where(np.isnan(Z), 0.0, Z)
        ax.plot_surface(X, Y, Z_plot, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_zlabel("Value")
        ax.set_zlim(-1, 1)
        ax.set_title(title)
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(12, 22, 2))

    fig.suptitle(
        "S&B (2nd ed.), Sec. 5.1, Fig. 5.1: Blackjack value function\n"
        f"Policy: stick on 20/21, first-visit MC, {n_episodes:,} episodes"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, fig.axes


def plot_figure_5_1(save_path: str | None = None, show: bool = False, n_episodes: int = 500_000, seed: int = 0):
    """Alias for blackjack value function plot (Fig. 5.1).

    Book: Sutton & Barto (2nd ed.), Figure 5.1 (Chapter 5).

    Args:
        save_path: See Figure 5.1 and the implementation for its role.
        show: See Figure 5.1 and the implementation for its role.
        n_episodes: See Figure 5.1 and the implementation for its role.
        seed: See Figure 5.1 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 5).
    return plot_blackjack_value_functions(save_path=save_path, show=show, n_episodes=n_episodes, seed=seed)


# ---------------------------------------------------------------------------
# Monte Carlo ES (Algorithm 5.3) — Figure 5.2
# ---------------------------------------------------------------------------

def _run_exploring_starts_episode(env, actions: tuple, greedy_probs, rng) -> list[tuple]:
    """Generate one episode whose first action is chosen uniformly at random.

    The exploring start is what makes every action in every reachable state
    sampled eventually, even while later actions follow the greedy policy.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        env: See Algorithm 5.1 and the implementation for its role.
        actions: See Algorithm 5.1 and the implementation for its role.
        greedy_probs: See Algorithm 5.1 and the implementation for its role.
        rng: See Algorithm 5.1 and the implementation for its role.

    Returns:
        run exploring starts episode as defined in Algorithm 5.1."""
    # n actions: number of available actions (book Chapter 5).
    n_actions = len(actions)
    # state: current state of the environment or MDP (book Chapter 5).
    state = env.reset()
    a_idx = int(rng.integers(n_actions))  # exploring starts
    trajectory: list[tuple] = []
    # (reward, observation, done): result of env.step(...) (book Chapter 5).
    reward, observation, done = env.step(actions[a_idx])
    trajectory.append((state, a_idx, reward))
    state = observation
    # Repeat while the condition holds: the algorithm's termination loop.
    while not done:
        # a idx: index of the selected action (book Chapter 5).
        a_idx = _sample_action(state, greedy_probs, rng, actions)
        # (reward, observation, done): result of env.step(...) (book Chapter 5).
        reward, observation, done = env.step(actions[a_idx])
        trajectory.append((state, a_idx, reward))
        state = observation
    return trajectory


def mc_exploring_starts(
    env,
    n_episodes: int = 100_000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict, list]:
    """Monte Carlo ES (exploring starts), Algorithm 5.3. Book Section 5.3.

    Every episode starts at a random (state, action) pair; the first action
    is drawn uniformly, which guarantees every action in every reachable
    state has non-zero start probability. Subsequent actions follow the
    current greedy policy (ties split uniformly). Each first-visited
    (state, action) pair in the episode is updated with its return.

    Returns:
        (policy, q_table, episode_rewards): policy[state] = greedy
        probability vector over the action space; q_table[state] =
        action-value array.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.3 (Chapter 5).

    Args:
        env: See Algorithm 5.3 and the implementation for its role.
        n_episodes: See Algorithm 5.3 and the implementation for its role.
        gamma: See Algorithm 5.3 and the implementation for its role.
        seed: See Algorithm 5.3 and the implementation for its role."""
    # carry out the mc exploring starts step (book Chapter 5).
    # actions: all actions available in the MDP (book Chapter 5).
    actions = _actions_for(env)
    # n actions: number of available actions (book Chapter 5).
    n_actions = len(actions)
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)
    # q table: action-value estimates Q(s, a) (book Chapter 5).
    q_table: dict = {}
    # returns by pair: dict attribute/state used by the algorithm (book Chapter 5).
    returns_by_pair: dict = {}

    def greedy_probs(state) -> np.ndarray:
        """greedy probs (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5)."""
        # carry out the greedy probs step (book Chapter 5).
        # q: action-value estimate Q(s, a) (book Chapter 5).
        q = q_table.get(state)
        if q is None:
            return np.full(n_actions, 1.0 / n_actions)
        # greedy: result of np.flatnonzero(...) (book Chapter 5).
        greedy = np.flatnonzero(q == q.max())
        # probs: action probabilities from the current policy (book Chapter 5).
        probs = np.zeros(n_actions, dtype=np.float64)
        probs[greedy] = 1.0 / len(greedy)
        return probs

    # episode rewards: list[float] attribute/state used by the algorithm (book Chapter 5).
    episode_rewards: list[float] = []
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # trajectory: sequence of states, actions and rewards in the episode (book Chapter 5).
        trajectory = _run_exploring_starts_episode(env, actions, greedy_probs, rng)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))
        _accumulate_first_visit_q(
            trajectory, gamma, n_actions, returns_by_pair, q_table
        )

    # policy: policy mapping states to action probabilities (book Chapter 5).
    policy = {s: pd_greedy(q) for s, q in q_table.items()}
    return policy, q_table, episode_rewards


def pd_greedy(q: np.ndarray) -> np.ndarray:
    """Uniform-over-argmax probability vector for action-value array q.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        q: See Algorithm 5.1 and the implementation for its role.

    Returns:
        pd greedy as defined in Algorithm 5.1."""
    # carry out the pd greedy step (book Chapter 5).
    # q: action-value estimate Q(s, a) (book Chapter 5).
    q = np.asarray(q, dtype=np.float64)
    # greedy: result of np.flatnonzero(...) (book Chapter 5).
    greedy = np.flatnonzero(q == q.max())
    # probs: action probabilities from the current policy (book Chapter 5).
    probs = np.zeros(len(q), dtype=np.float64)
    probs[greedy] = 1.0 / len(greedy)
    return probs


class OneStateIsEnv:
    """Example 5.5 one-state MDP (infinite variance of ordinary IS), Sec. 5.5.

    State s0 (reward 0): action ``left`` stays in s0 with probability
    ``p_stay`` (0.9 in the book) and jumps to an absorbing state with
    reward 1 otherwise; action ``right`` jumps directly to the absorbing
    state with reward 0. Target policy pi is "always left" (so v_pi(s0)=1);
    the behavior policy is uniform over the two actions.

    With ordinary importance sampling the per-episode weights are 2^k for a
    trajectory of k "left" decisions, and the estimator has infinite
    variance; weighted IS stays stable.

    Book: Sutton & Barto (2nd ed.), Section 5.5 (Chapter 5)."""

    # actions: collection of the listed elements (book Chapter 5).
    ACTIONS = ("left", "right")

    def __init__(self, p_stay: float = 0.9, seed: int | None = None) -> None:
        """init (reference implementation).


        Args:
            p_stay: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Section 5.5 (Chapter 5)."""
        # initialize the parameters and state required by the algorithm (book Chapter 5).
        if not 0.0 < p_stay < 1.0:
            raise ValueError("p_stay must be in (0, 1)")
        self.p_stay = float(p_stay)
        self.rng = np.random.default_rng(seed)
        self._state = "s0"

    def reset(self) -> str:
        """reset (reference implementation).


        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 5.5 (Chapter 5)."""
        # reset internal state for a new episode (book Chapter 5).
        self._state = "s0"
        return self._state

    def step(self, action: str) -> tuple[float, str, bool]:
        """step (reference implementation).


        Args:
            action: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 5.5 (Chapter 5)."""
        # advance one step and return the observed transition (book Chapter 5).
        if action == "left":
            if self.rng.random() < self.p_stay:
                return 0.0, "s0", False
            self._state = "ABS"
            return 1.0, "ABS", True
        if action == "right":
            self._state = "ABS"
            return 0.0, "ABS", True
        raise ValueError(f"unknown action {action!r}")


def _draw_blackjack_policy_panel(fig, idx: int, usable: bool, title: str, q_table: dict) -> None:
    """Top panel: the greedy action (stick=0, hit=1) at each visible state.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        fig: See Algorithm 5.1 and the implementation for its role.
        idx: See Algorithm 5.1 and the implementation for its role.
        usable: See Algorithm 5.1 and the implementation for its role.
        title: See Algorithm 5.1 and the implementation for its role.
        q_table: See Algorithm 5.1 and the implementation for its role."""
    # ax: result of fig.add_subplot(...) (book Chapter 5).
    ax = fig.add_subplot(2, 2, idx)
    policy_map = _blackjack_q_grid(q_table, usable, lambda q: int(np.argmax(q) == 0))  # 1 = hit
    im = ax.imshow(policy_map, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(_BJ_DEALER_MIN, _BJ_DEALER_MAX + 1))
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(_BJ_PLAYER_SUM_MIN, _BJ_PLAYER_SUM_MAX + 1))
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_title(f"Optimal policy ({title.lower()})")
    # cbar: result of fig.colorbar(...) (book Chapter 5).
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.85)
    cbar.ax.set_yticklabels(["stick", "hit"])


def _draw_blackjack_value_panel(fig, idx: int, usable: bool, title: str, q_table: dict, X, Y) -> None:
    """Bottom panel: v*(s) = max_a q(s, a) as a 3D surface.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        fig: See Algorithm 5.1 and the implementation for its role.
        idx: See Algorithm 5.1 and the implementation for its role.
        usable: See Algorithm 5.1 and the implementation for its role.
        title: See Algorithm 5.1 and the implementation for its role.
        q_table: See Algorithm 5.1 and the implementation for its role.
        X: See Algorithm 5.1 and the implementation for its role.
        Y: See Algorithm 5.1 and the implementation for its role."""
    # internal helper for the draw blackjack value panel step (book Chapter 5).
    # ax: result of fig.add_subplot(...) (book Chapter 5).
    ax = fig.add_subplot(2, 2, idx, projection="3d")
    # z: result of _blackjack_q_grid(...) (book Chapter 5).
    Z = _blackjack_q_grid(q_table, usable, lambda q: float(np.max(q)))
    # z plot: result of np.where(...) (book Chapter 5).
    Z_plot = np.where(np.isnan(Z), 0.0, Z)
    ax.plot_surface(X, Y, Z_plot, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")
    ax.set_zlim(-1, 1)
    ax.set_title(f"v* ({title.lower()})")
    ax.set_xticks(range(_BJ_DEALER_MIN, _BJ_DEALER_MAX + 1))
    ax.set_yticks(range(_BJ_PLAYER_SUM_MIN, _BJ_PLAYER_SUM_MAX + 1, 2))


def plot_blackjack_optimal(
    save_path: str | None = None,
    show: bool = False,
    n_episodes: int = 500_000,
    seed: int = 0,
):
    """Optimal blackjack policy and value function (Example 5.1, Fig. 5.2).

    Sutton & Barto (2nd ed.), Sec. 5.3, Fig. 5.2: the optimal policy and
    state-value function for blackjack, estimated with Monte Carlo ES
    (exploring starts, Algorithm 5.3). Two policy heatmaps (stick=0, hit=1)
    on top and the two v* 3D surfaces below, separately for usable and no
    usable ace. Episode count controls accuracy; the book used 500,000.

    Figure identity note: in the 2nd (2018) edition this is Figure 5.2;
    the in-progress draft and older editions differ, and this is an
    illustrative demonstration, not a pixel-identical redraw of any
    edition's print figure.

    Book: Sutton & Barto (2nd ed.), Figure 5.2 (Chapter 5).

    Args:
        save_path: See Figure 5.2 and the implementation for its role.
        show: See Figure 5.2 and the implementation for its role.
        n_episodes: See Figure 5.2 and the implementation for its role.
        seed: See Figure 5.2 and the implementation for its role."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # env: environment instance being controlled (book Chapter 5).
    env = BlackjackEnv(seed=seed)
    # (_, q_table, episode_rewards): result of mc_exploring_starts(...) (book Chapter 5).
    _, q_table, episode_rewards = mc_exploring_starts(env, n_episodes=n_episodes, seed=seed)

    # (X, Y): result of np.meshgrid(...) (book Chapter 5).
    X, Y = np.meshgrid(
        np.arange(_BJ_DEALER_MIN, _BJ_DEALER_MAX + 1),
        np.arange(_BJ_PLAYER_SUM_MIN, _BJ_PLAYER_SUM_MAX + 1),
    )

    # fig: result of plt.figure(...) (book Chapter 5).
    fig = plt.figure(figsize=(12, 9))
    # Iterate over (idx, (usable, title)): indexed entries (book algorithm).
    for idx, (usable, title) in enumerate(_BJ_ACE_VARIANTS, start=1):
        _draw_blackjack_policy_panel(fig, idx, usable, title, q_table)
    # Iterate over (idx, (usable, title)): indexed entries (book algorithm).
    for idx, (usable, title) in enumerate(_BJ_ACE_VARIANTS, start=3):
        _draw_blackjack_value_panel(fig, idx, usable, title, q_table, X, Y)

    fig.suptitle(
        "S&B (2nd ed.), Sec. 5.3, Fig. 5.2: optimal blackjack policy and value\n"
        f"Monte Carlo ES, {n_episodes:,} episodes (illustrative demonstration)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.35)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, q_table, episode_rewards


def _log_step_list(max_episodes: int) -> list[int]:
    """Log-spaced checkpoint list 1..9, 10..90, ... up to max_episodes.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        max_episodes: See Algorithm 5.1 and the implementation for its role.

    Returns:
        log step list as defined in Algorithm 5.1."""
    # internal helper for the log step list step (book Chapter 5).
    # steps: collection of the listed elements (book Chapter 5).
    steps = []
    base = 1
    # Repeat while the condition holds: the algorithm's termination loop.
    while base < max_episodes:
        # steps: add collection built by comprehension into the running value (book Chapter 5).
        steps += [i * base for i in range(1, 10) if i * base <= max_episodes]
        # base: multiply literal constant value into the running value (book Chapter 5).
        base *= 10
    if not steps or steps[-1] != max_episodes:
        steps.append(max_episodes)
    return steps


# Distinct run seeds keep the repeated curves independent but reproducible.
_IS_RUN_SEED_STRIDE = 1_000_003
# is state of interest: collection of the listed elements (book Chapter 5).
_IS_STATE_OF_INTEREST = (13, 2, True)
_IS_TARGET_STICK_THRESHOLD = 20


def _importance_sampling_errors(
    env,
    behavior,
    target,
    state0: tuple,
    v_ref: float,
    step_list: list[int],
    max_episodes: int,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one run's squared errors at each checkpoint, for both IS rules.

    Ordinary IS accumulates rho*G, weighted IS accumulates rho and rho*G; both
    are updated only on the first visit to `state0` in an episode.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        env: See Algorithm 5.1 and the implementation for its role.
        behavior: See Algorithm 5.1 and the implementation for its role.
        target: See Algorithm 5.1 and the implementation for its role.
        state0: See Algorithm 5.1 and the implementation for its role.
        v_ref: See Algorithm 5.1 and the implementation for its role.
        step_list: See Algorithm 5.1 and the implementation for its role.
        max_episodes: See Algorithm 5.1 and the implementation for its role.
        max_steps: See Algorithm 5.1 and the implementation for its role.

    Returns:
        importance sampling errors as defined in Algorithm 5.1."""
    rng = env.rng
    # (sum_ord, count, sum_w, sum_wG): collection of the listed elements (book Chapter 5).
    sum_ord, count, sum_w, sum_wG = 0.0, 0, 0.0, 0.0
    # err o: array filled with a constant value (book Chapter 5).
    err_o = np.full(len(step_list), np.nan)
    # err w: array filled with a constant value (book Chapter 5).
    err_w = np.full(len(step_list), np.nan)
    ck = 0
    for ep in range(1, max_episodes + 1):
        # traj: result of _run_episode(...) (book Chapter 5).
        traj = _run_episode(env, behavior, rng, max_steps)
        # rewards: per-step reward accumulation buffer (book Chapter 5).
        rewards = [r for _, _, r in traj]
        # returns: accumulated returns used as update targets (book Chapter 5).
        returns = _returns_from_rewards(rewards, 1.0)
        # ratios: result of _importance_ratios(...) (book Chapter 5).
        ratios = _importance_ratios(traj, target, behavior)
        for t, (s, _, _) in enumerate(traj):
            if s != state0:
                continue
            # (g, rho): collection of the listed elements (book Chapter 5).
            g, rho = returns[t], ratios[t]
            # sum_ord: add value computed from its operands into the running value (book Chapter
            sum_ord += rho * g
            # count: add literal constant value into the running value (book Chapter 5).
            count += 1
            # sum_w: add reference to an existing value into the running value (book Chapter 5).
            sum_w += rho
            # sum_wG: add value computed from its operands into the running value (book Chapter 5).
            sum_wG += rho * g
            break  # first-visit only
        if ep == step_list[ck]:
            # ord est: value chosen conditionally (book Chapter 5).
            ord_est = sum_ord / count if count else 0.0
            # wtd est: value chosen conditionally (book Chapter 5).
            wtd_est = sum_wG / sum_w if sum_w else 0.0
            err_o[ck] = (ord_est - v_ref) ** 2
            err_w[ck] = (wtd_est - v_ref) ** 2
            # ck: add literal constant value into the running value (book Chapter 5).
            ck += 1
            if ck >= len(step_list):
                break
    return err_o, err_w


def plot_importance_sampling(
    save_path: str | None = None,
    show: bool = False,
    n_runs: int = 100,
    seed: int = 0,
    max_episodes: int = 10_000,
    ref_episodes: int = 200_000,
):
    """Weighted vs ordinary importance sampling on one blackjack state (Fig. 5.3).

    Sutton & Barto (2nd ed.), Sec. 5.5/5.6, Fig. 5.3: estimate the value of a
    single blackjack state — player sum 13, dealer showing 2, usable ace,
    target stick-on-20 policy — by importance sampling from a uniform-random
    behavior policy, using first-visit returns. Ordinary IS error rises then
    falls; weighted IS error drops steadily. The x-axis is episodes (log
    scale); the y-axis is mean square error vs a reference estimate,
    averaged over ``n_runs`` independent runs. The book reports
    v(s) ~= -0.27726; this reproduction estimates the reference value with a
    200,000-episode first-visit MC prediction instead of hardcoding it.

    Figure identity note: Figure 5.3 in the 2nd (2018) edition; illustrative
    demonstration, not a pixel-identical redraw of any edition's figure.

    Book: Sutton & Barto (2nd ed.), Figure 5.3 (Chapter 5).

    Args:
        save_path: See Figure 5.3 and the implementation for its role.
        show: See Figure 5.3 and the implementation for its role.
        n_runs: See Figure 5.3 and the implementation for its role.
        seed: See Figure 5.3 and the implementation for its role.
        max_episodes: See Figure 5.3 and the implementation for its role.
        ref_episodes: See Figure 5.3 and the implementation for its role."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state0 = _IS_STATE_OF_INTEREST
    # target: target value for the current update (book Chapter 5).
    target = blackjack_stick_on_threshold(_IS_TARGET_STICK_THRESHOLD)
    behavior = lambda s: np.full(2, 0.5)  # noqa: E731

    # env ref: result of BlackjackEnv(...) (book Chapter 5).
    env_ref = BlackjackEnv(seed=seed)
    # v ref: element selected from the indexed collection (book Chapter 5).
    v_ref = mc_first_visit_prediction(
        env_ref, target, n_episodes=ref_episodes, seed=seed
    )[state0]

    # step list: result of _log_step_list(...) (book Chapter 5).
    step_list = _log_step_list(max_episodes)
    # mse ord: zero-filled array (book Chapter 5).
    mse_ord = np.zeros(len(step_list))
    # mse wtd: zero-filled array (book Chapter 5).
    mse_wtd = np.zeros(len(step_list))
    # Iterate over run: one pass per index (book algorithm).
    for run in range(n_runs):
        # env: environment instance being controlled (book Chapter 5).
        env = BlackjackEnv(seed=seed * _IS_RUN_SEED_STRIDE + run)
        # (err_o, err_w): result of _importance_sampling_errors(...) (book Chapter 5).
        err_o, err_w = _importance_sampling_errors(
            env, behavior, target, state0, v_ref, step_list, max_episodes
        )
        # mse_ord: add reference to an existing value into the running value (book Chapter 5).
        mse_ord += err_o
        # mse_wtd: add reference to an existing value into the running value (book Chapter 5).
        mse_wtd += err_w
    # mse_ord: divide reference to an existing value into the running value (book Chapter 5).
    mse_ord /= n_runs
    # mse_wtd: divide reference to an existing value into the running value (book Chapter 5).
    mse_wtd /= n_runs

    # (fig, ax): result of plt.subplots(...) (book Chapter 5).
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(step_list, mse_ord, color="#d62728", label="Ordinary importance sampling")
    ax.plot(step_list, mse_wtd, color="#1f77b4", label="Weighted importance sampling")
    ax.set_xscale("log")
    ax.set_xlabel("Episodes (log scale)")
    ax.set_ylabel(f"Mean square error (average over {n_runs} runs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("S&B (2nd ed.), Sec. 5.5/5.6, Fig. 5.3: importance sampling (illustrative)")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, v_ref


def _one_state_is_samples(
    n_episodes: int, seed: int, p_stay: float = 0.9
) -> np.ndarray:
    """Vectorized episode weights x returns for the Example 5.5 MDP.

    Each simulated episode is one sample of the IS estimator (starting at
    t=0 in s0): returns 1 when a 'left' decision is followed by the
    absorbing transition, weights 2^k for k 'left' decisions.

    Book: Sutton & Barto (2nd ed.), Algorithm 5.1 (Chapter 5).

    Args:
        n_episodes: See Algorithm 5.1 and the implementation for its role.
        seed: See Algorithm 5.1 and the implementation for its role.
        p_stay: See Algorithm 5.1 and the implementation for its role.

    Returns:
        one state is samples as defined in Algorithm 5.1."""
    # internal helper for the one state is samples step (book Chapter 5).
    # rng: seeded random generator so runs are reproducible (book Chapter 5).
    rng = np.random.default_rng(seed)
    # rho: one-filled array (book Chapter 5).
    rho = np.ones(n_episodes)
    # ret: zero-filled array (book Chapter 5).
    ret = np.zeros(n_episodes)
    # alive: one-filled array (book Chapter 5).
    alive = np.ones(n_episodes, dtype=bool)
    # Iterate over _: one pass per index (book algorithm).
    for _ in range(200):
        if not alive.any():
            break
        # act left: boolean result of a comparison (book Chapter 5).
        act_left = rng.random(n_episodes) < 0.5
        # absorb: boolean result of a comparison (book Chapter 5).
        absorb = rng.random(n_episodes) >= p_stay
        # taken left: value computed from its operands (book Chapter 5).
        taken_left = alive & act_left
        # rho[taken_left]: multiply literal constant value into the running value (book Chapter 5).
        rho[taken_left] *= 2.0
        # taken right: value computed from its operands (book Chapter 5).
        taken_right = alive & ~act_left
        rho[taken_right] = 0.0
        # win: value computed from its operands (book Chapter 5).
        win = taken_left & absorb
        ret[win] = 1.0
        # alive: value computed from its operands (book Chapter 5).
        alive = taken_left & ~absorb
    return rho * ret


def plot_ordinary_is_unstable(
    save_path: str | None = None,
    show: bool = False,
    n_runs: int = 10,
    seed: int = 0,
    max_episodes: int = 1_000_000,
):
    """Ordinary importance sampling with unstable estimates (Example 5.5, Fig. 5.4).

    Sutton & Barto (2nd ed.), Sec. 5.5, Example 5.5, Fig. 5.4: the one-state
    MDP whose ordinary-IS estimator has infinite variance. The target policy
    is "always left" (v_pi(s0) = 1); the behavior policy is uniform. Each of
    ``n_runs`` seeds plots its running estimate against episodes (log scale);
    dashed lines mark y = 1 (true value) and y = 2. Episodes beyond the y = 2
    axis are clipped, matching the book's y-range.

    Runtime note: the book's printed figure reaches 10^8 episodes; this
    reproduction defaults to 10^6 per run (vectorized simulation), so the
    rare deep-trajectory spikes are smaller here. Figure identity note:
    Figure 5.4 in the 2nd (2018) edition; illustrative demonstration, not a
    pixel-identical redraw of any edition's figure.

    Book: Sutton & Barto (2nd ed.), Figure 5.4 (Chapter 5).

    Args:
        save_path: See Figure 5.4 and the implementation for its role.
        show: See Figure 5.4 and the implementation for its role.
        n_runs: See Figure 5.4 and the implementation for its role.
        seed: See Figure 5.4 and the implementation for its role.
        max_episodes: See Figure 5.4 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 5).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # step list: result of _log_step_list(...) (book Chapter 5).
    step_list = _log_step_list(max_episodes)
    # run curves: collection of the listed elements (book Chapter 5).
    run_curves = []
    # Iterate over run: one pass per index (book algorithm).
    for run in range(n_runs):
        # samples: result of _one_state_is_samples(...) (book Chapter 5).
        samples = _one_state_is_samples(
            max_episodes, seed * _IS_RUN_SEED_STRIDE + run, p_stay=0.9
        )
        # cumulative: result of np.cumsum(...) (book Chapter 5).
        cumulative = np.cumsum(samples)
        # indices: value computed from its operands (book Chapter 5).
        indices = np.array(step_list, dtype=np.int64) - 1
        # estimates: value computed from its operands (book Chapter 5).
        estimates = cumulative[indices] / np.array(step_list, dtype=np.float64)
        run_curves.append(estimates)

    # (fig, ax): result of plt.subplots(...) (book Chapter 5).
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # Iterate over estimates: each item in run_curves (book algorithm).
    for estimates in run_curves:
        ax.plot(step_list, estimates, color="#1f77b4", alpha=0.65, linewidth=1.1)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="true value (v = 1)")
    ax.axhline(2.0, color="k", linestyle="--", linewidth=1.2, label="y = 2")
    ax.set_ylim(0, 2.2)
    ax.set_xscale("log")
    ax.set_xlabel("Episodes (log scale)")
    ax.set_ylabel("Estimate of v (ordinary IS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("S&B (2nd ed.), Sec. 5.5, Ex. 5.5, Fig. 5.4: ordinary IS (illustrative)")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

