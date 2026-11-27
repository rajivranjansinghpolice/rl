"""Solution implementation for Chapter 6 - Temporal-Difference Learning.

Reference: Sutton & Barto (2nd Ed), Chapter 6

Implements: Alg 6.1 (Sarsa, on-policy TD control / on-policy prediction),
Alg 6.2 (Q-learning, off-policy TD control),
Alg 6.3 (TD(lambda) prediction),
Exercise 6.1 (random walk, TD(0) vs Monte Carlo comparison).

Applications reproduced from Chapter 6 (2nd ed.):
- Example 6.2 / Figure 6.2: random-walk result panel (unnumbered) and the
  Figure 6.2 batch-updating experiment (Example 6.3) - RMS error vs walks
  seen for batch TD(0) and batch constant-alpha MC.
- Example 6.5 / Figure 6.3: Sarsa on the windy gridworld (steps per episode).
- Exercise 6.9: king's-moves variant; Exercise 6.10: stochastic-wind variant
  (no numbered book figure; illustrative curves produced).
- Example 6.6 / Figure 6.4: Cliff Walking - Sarsa vs Q-learning.
- Figure 6.6: interim/asymptotic performance of Sarsa, Q-learning and
  Expected Sarsa on the cliff-walking task as a function of alpha
  (adapted from van Seijen et al., 2009, in the 2nd ed.).
- Example 6.7 / Figure 6.7: maximization bias - Q-learning vs Double
  Q-learning on a toy episodic MDP.
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

def plot_example_6_2_random_walk(
    save_path: str | None = None,
    show: bool = False,
    repeats: int = 100,
    episodes: int = 100,
    alpha: float = 0.1,
    seed: int = 0,
):
    """TD(0) vs constant-alpha MC on the random walk (Example 6.2).

    Sutton & Barto (2nd ed.), Sec. 6.2, Example 6.2: RMS error in value estimates
    over episodes for TD(0) and constant-alpha MC on the 5-state random walk
    (states A-E, start C, right terminal +1). The result panel in the 2nd ed. is
    not a numbered figure; the numbered Figure 6.2 is the batch-updating
    experiment (Example 6.3, see `plot_figure_6_2_batch_updating`). Both methods
    start from V(s)=0.5 and use step size alpha; curves are averaged over
    `repeats` independent runs; TD(0) typically converges faster.
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
    ax.set_title(
        "S&B (2nd ed.), Sec. 6.2, Example 6.2: Random walk TD vs MC\n"
        "(unnumbered result panel; 2nd-ed. Figure 6.2 is batch updating)"
    )
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


# Backward-compatible alias: earlier docs called this plot_figure_6_2.
plot_figure_6_2 = plot_example_6_2_random_walk


# ---------------------------------------------------------------------------
# Chapter 6 application environments (Sutton & Barto, 2nd ed.)
# ---------------------------------------------------------------------------


class CliffWalkingEnv:
    """Cliff Walking (Example 6.6). Book Sec. 6.4/6.5, Figure 6.4.

    4 x 12 grid. Start S=(3,0), goal G=(3,11). The cliff is the bottom row
    between them (row 3, columns 1-10). Actions: up/down/left/right, one cell
    per step, always deterministic. Rewards: -1 per step elsewhere; stepping
    onto a cliff cell returns -100 and sends the agent back to the start
    (the episode does NOT end); reaching the goal ends the episode with -1.

    Undiscounted, episodic. Deterministic.
    """

    ACTIONS = ("up", "down", "left", "right")
    _DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, height: int = 4, width: int = 12) -> None:
        if height < 2 or width < 2:
            raise ValueError("height and width must both be >= 2")
        self.height = height
        self.width = width
        self.start = (height - 1, 0)
        self.goal = (height - 1, width - 1)
        self.cliff_cells = {(height - 1, c) for c in range(1, width - 1)}
        self.state: tuple[int, int] | None = None

    def reset(self) -> tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int | str) -> tuple[float, tuple[int, int], bool]:
        if self.state is None:
            raise RuntimeError("call reset() first")
        if isinstance(action, str):
            action = self.ACTIONS.index(action)
        if action < 0 or action >= len(self.ACTIONS):
            raise ValueError(f"invalid action {action}")
        dr, dc = self._DELTAS[action]
        r, c = self.state
        nr = min(max(r + dr, 0), self.height - 1)
        nc = min(max(c + dc, 0), self.width - 1)
        if (nr, nc) in self.cliff_cells:
            self.state = self.start
            return -100.0, self.state, False
        self.state = (nr, nc)
        done = self.state == self.goal
        return -1.0, self.state, done

    @property
    def n_states(self) -> int:
        # rows x cols nonterminal cells (cliff cells are never occupied states)
        return self.height * self.width - len(self.cliff_cells)


class MaximizationBiasMDP:
    """Maximization-bias toy MDP (Example 6.7). Book Sec. 6.7, Figure 6.7.

    Episodic MDP with two nonterminal states A and B. Episodes always start
    in A. Actions from A: 'left' and 'right'.

    - 'right': terminal state, reward 0 (the safe action).
    - 'left': state B, reward 0. B has
_b_actions possible actions, all of
      which take the agent to the terminal state with a reward drawn from
      a normal distribution with mean -0.1 and variance 1 (N(-0.1, 1)).

    The optimal action from A is 'right' (left's expected return is -0.1),
    but the maximum of many noisy returns through B is optimistically
    biased, so an agent maximizing action-value estimates tends to prefer
    'left'. This is maximization bias.

    gamma = 1 in the book experiment (undiscounted).
    """

    ACTIONS = ("left", "right")
    B_ACTIONS = tuple(range(10))

    def __init__(self, n_b_actions: int = 10, reward_mean: float = -0.1, reward_std: float = 1.0, seed: int | None = None) -> None:
        if n_b_actions < 1:
            raise ValueError("n_b_actions must be >= 1")
        self.B_ACTIONS = tuple(range(n_b_actions))
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.rng = np.random.default_rng(seed)
        self.state: str | None = None

    def actions_for(self, state: str) -> tuple:
        if state == "B":
            return self.B_ACTIONS
        if state == "A":
            return self.ACTIONS
        raise ValueError(f"no actions for terminal/unknown state {state!r}")

    def reset(self) -> str:
        self.state = "A"
        return self.state

    def step(self, action) -> tuple[float, str, bool]:
        if self.state is None:
            raise RuntimeError("call reset() first")
        if self.state == "A":
            action = action if isinstance(action, str) else self.ACTIONS[int(action)]
            if action == "right":
                self.state = "T"
                return 0.0, "T", True
            if action == "left":
                self.state = "B"
                return 0.0, "B", False
            raise ValueError(f"invalid action {action!r}")
        if self.state == "B":
            b_actions = self.B_ACTIONS
            try:
                b_actions.index(action) if isinstance(action, int) else int(action)
            except (ValueError, TypeError):
                raise ValueError(f"invalid action {action}") from None
            reward = float(self.rng.normal(self.reward_mean, self.reward_std))
            self.state = "T"
            return reward, "T", True
        raise RuntimeError(f"episode already terminal in state {self.state!r}")


def _actions_for(env, state) -> tuple:
    actions_for = getattr(env, "actions_for", None)
    if callable(actions_for):
        return tuple(actions_for(state))
    return tuple(env.ACTIONS)


def _q_row(q: dict, state, n_actions: int) -> list[float]:
    return [q.get((state, i), 0.0) for i in range(n_actions)]


def _eps_greedy_sample(q_row, epsilon: float, rng) -> int:
    probs = _epsilon_greedy_probs(np.asarray(q_row, dtype=np.float64), epsilon)
    return int(rng.choice(len(probs), p=probs))


def _validate_td_control_params(alpha: float, epsilon: float, mode: str) -> None:
    """Check the shared TD-control parameters before any environment step."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if mode not in ("sarsa", "q_learning", "expected_sarsa"):
        raise ValueError(f"unknown mode {mode!r}")


def _td_bootstrap_target(
    mode: str, reward: float, gamma: float, next_row, next_a_idx, epsilon: float
) -> float:
    """One-step backup: sampled action, greedy action, or policy expectation."""
    if mode == "q_learning":
        return reward + gamma * max(next_row)
    if mode == "expected_sarsa":
        probs = _epsilon_greedy_probs(np.asarray(next_row, dtype=np.float64), epsilon)
        return reward + gamma * float(np.dot(probs, next_row))
    return reward + gamma * next_row[next_a_idx]  # sarsa: use the sampled action


def _run_td_episode(
    env,
    q: dict,
    rng,
    alpha: float,
    epsilon: float,
    gamma: float,
    mode: str,
    max_steps: int,
) -> tuple[int, float]:
    """Play one (truncated) episode, updating Q in place; return steps, return."""
    state = env.reset()
    n_actions = len(_actions_for(env, state))
    a_idx = _eps_greedy_sample(_q_row(q, state, n_actions), epsilon, rng)
    total = 0.0
    steps = 0
    for t in range(max_steps):
        steps = t + 1
        reward, next_state, done = env.step(a_idx)
        total += reward
        if done:
            target = reward
            next_a_idx = None
        else:
            next_actions = _actions_for(env, next_state)
            next_row = _q_row(q, next_state, len(next_actions))
            # Draw the next action in every mode: it is Sarsa's sample, and
            # using the same epsilon-greedy behavior policy keeps all three
            # modes' RNG streams aligned.
            next_a_idx = _eps_greedy_sample(next_row, epsilon, rng)
            target = _td_bootstrap_target(mode, reward, gamma, next_row, next_a_idx, epsilon)
        current = q.get((state, a_idx), 0.0)
        q[(state, a_idx)] = current + alpha * (target - current)
        if done:
            break
        state = next_state
        a_idx = next_a_idx
    return steps, total


def _td_control(
    env,
    n_episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    seed: int | None,
    mode: str,
    max_steps: int,
) -> tuple[dict, list, list]:
    """Shared online TD-control loop for tabular indexed-action envs.

    mode: 'sarsa' (uses the sampled next action), 'q_learning' (uses the
    greedy next action) or 'expected_sarsa' (uses the expectation under the
    epsilon-greedy policy). Q is keyed (state, action_index) and lazily
    filled with 0.0. Episodes are truncated at max_steps (needed for the
    windy gridworld, where an unlucky step-0 policy can loop forever).

    Returns (q_table, steps_per_episode, returns_per_episode).
    """
    _validate_td_control_params(alpha, epsilon, mode)
    rng = np.random.default_rng(seed)
    q: dict = {}
    steps_per_episode: list[int] = []
    returns_per_episode: list[float] = []

    for _ in range(n_episodes):
        # Each episode starts a fresh trajectory; statistics are recorded in
        # episode order so the three control modes stay directly comparable.
        steps, total = _run_td_episode(
            env, q, rng, alpha, epsilon, gamma, mode, max_steps
        )
        steps_per_episode.append(steps)
        returns_per_episode.append(total)
    return q, steps_per_episode, returns_per_episode


def sarsa_control(env, n_episodes: int = 1000, alpha: float = 0.5, epsilon: float = 0.1, gamma: float = 1.0, seed: int | None = None, max_steps: int = 10_000) -> tuple[dict, list, list]:
    """On-policy Sarsa (Sec. 6.4) on any tabular indexed-action env.

    Q keyed (state, action_index). Returns (q_table, steps_per_episode,
    returns_per_episode).
    """
    return _td_control(env, n_episodes, alpha, epsilon, gamma, seed, "sarsa", max_steps)


def q_learning_control(env, n_episodes: int = 1000, alpha: float = 0.5, epsilon: float = 0.1, gamma: float = 1.0, seed: int | None = None, max_steps: int = 10_000) -> tuple[dict, list, list]:
    """Off-policy Q-learning (Sec. 6.5) on any tabular indexed-action env.

    Q keyed (state, action_index). Returns (q_table, steps_per_episode,
    returns_per_episode).
    """
    return _td_control(env, n_episodes, alpha, epsilon, gamma, seed, "q_learning", max_steps)


def expected_sarsa_control(env, n_episodes: int = 1000, alpha: float = 0.5, epsilon: float = 0.1, gamma: float = 1.0, seed: int | None = None, max_steps: int = 10_000) -> tuple[dict, list, list]:
    """Expected Sarsa (Sec. 6.5/6.6) on any tabular indexed-action env.

    Bootstrap: reward + gamma * sum_a pi(a|s') * Q(s', a), with pi the
    epsilon-greedy policy. Q keyed (state, action_index). Returns
    (q_table, steps_per_episode, returns_per_episode).
    """
    return _td_control(env, n_episodes, alpha, epsilon, gamma, seed, "expected_sarsa", max_steps)


def _random_argmax(row: list[float], rng) -> int:
    """Uniformly pick one maximizer of `row`, as the book's tie-break requires."""
    return int(
        np.flatnonzero(row == np.max(row))[
            rng.integers(0, int(np.sum(row == np.max(row))))
        ]
    )


def _double_q_targets(
    next_state, next_actions, q1: dict, q2: dict, reward: float, gamma: float, rng
) -> tuple[float, float]:
    """Backup targets for Q1 and Q2, each evaluated with the *other* table."""
    nr1 = _q_row(q1, next_state, len(next_actions))
    nr2 = _q_row(q2, next_state, len(next_actions))
    argmax1 = _random_argmax(nr1, rng)
    argmax2 = _random_argmax(nr2, rng)
    target1 = reward + gamma * nr2[argmax1]  # greedy for Q1, evaluated by Q2
    target2 = reward + gamma * nr1[argmax2]  # greedy for Q2, evaluated by Q1
    return target1, target2


def _update_double_q_table(
    q1: dict, q2: dict, state, a_idx: int, target1: float, target2: float, alpha: float, rng
) -> None:
    """Update just one table per step so the two estimates stay independent."""
    if rng.random() < 0.5:
        key = (state, a_idx)
        q1[key] = q1.get(key, 0.0) + alpha * (target1 - q1.get(key, 0.0))
    else:
        key = (state, a_idx)
        q2[key] = q2.get(key, 0.0) + alpha * (target2 - q2.get(key, 0.0))


def double_q_learning(
    env,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> tuple[dict, dict, list]:
    """Double Q-learning (Sec. 6.7) on any tabular indexed-action env.

    Two independent action-value estimates, Q1 and Q2, are learned. Each step
    flips a fair coin to choose which table to update; the action to update
    is the greedy action for that table (ties broken randomly), and its
    target is evaluated with the *other* table, removing maximization bias.
    The behavior policy is epsilon-greedy with respect to Q1 + Q2.

    Returns (q1, q2, actions_at_first_state): the per-episode action index
    taken at the episode's first state (used for the Figure 6.7 experiment),
    q1[(state, i)], q2[(state, i)] lazily filled with 0.0.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    rng = np.random.default_rng(seed)
    q1: dict = {}
    q2: dict = {}
    first_actions: list[int] = []
    for _ in range(n_episodes):
        state = env.reset()
        actions = _actions_for(env, state)
        row1 = _q_row(q1, state, len(actions))
        row2 = _q_row(q2, state, len(actions))
        row_sum = [a + b for a, b in zip(row1, row2)]
        a_idx = _eps_greedy_sample(row_sum, epsilon, rng)
        done = False
        recorded = False
        t = 0
        while not done and t < max_steps:
            t += 1
            if not recorded:
                first_actions.append(a_idx)
                recorded = True
            reward, next_state, done = env.step(a_idx)
            if done:
                target1, target2 = reward, reward
            else:
                next_actions = _actions_for(env, next_state)
                target1, target2 = _double_q_targets(
                    next_state, next_actions, q1, q2, reward, gamma, rng
                )
            _update_double_q_table(q1, q2, state, a_idx, target1, target2, alpha, rng)
            if not done:
                state = next_state
                actions = _actions_for(env, state)
                row1 = _q_row(q1, state, len(actions))
                row2 = _q_row(q2, state, len(actions))
                row_sum = [a + b for a, b in zip(row1, row2)]
                a_idx = _eps_greedy_sample(row_sum, epsilon, rng)
    return q1, q2, first_actions


# Guard against an unlucky run of 'left' actions in the toy MDP.
_MAX_MBIAS_STEPS = 10_000


def _q_learning_left_fractions(
    env, rng, n_episodes: int, alpha: float, epsilon: float, gamma: float, left_idx: int
) -> np.ndarray:
    """One Q-learning run: record whether 'left' was taken at state A."""
    q: dict = {}
    left = np.zeros(n_episodes)
    for ep in range(n_episodes):
        state = env.reset()
        row = _q_row(q, state, len(env.ACTIONS))
        a_idx = _eps_greedy_sample(row, epsilon, rng)
        if state == "A":
            left[ep] = 1.0 if a_idx == left_idx else 0.0
        done = False
        steps = 0
        while not done and steps < _MAX_MBIAS_STEPS:
            steps += 1
            reward, next_state, done = env.step(a_idx)
            if done:
                target = reward
            else:
                next_row = _q_row(q, next_state, len(env.actions_for(next_state)))
                target = reward + gamma * max(next_row)
            key = (state, a_idx)
            q[key] = q.get(key, 0.0) + alpha * (target - q.get(key, 0.0))
            if not done:
                state = next_state
                row = _q_row(q, state, len(env.actions_for(state)))
                a_idx = _eps_greedy_sample(row, epsilon, rng)
    return left


def maximization_bias_experiment(
    n_episodes: int = 300,
    repeats: int = 100,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = 0,
) -> dict:
    """Q-learning vs Double Q-learning on the maximization-bias MDP.

    Figure 6.7 (Example 6.7, Sec. 6.7) protocol: episodic MDP (A -> left/
    right; left -> B with reward 0; B -> terminal with rewards ~ N(-0.1, 1)),
    deterministic safe action 'right' with value 0, gamma = 1. Initial action
    values 0, random tie-breaking, epsilon-greedy behavior with epsilon (book
    - 0.1), alpha (book - 0.1). The book averaged over 10,000 runs;
    repeats defaults to a reduced value - increase it for
    publication-quality curves.

    Returns dict with 'episodes' (1..n_episodes), 'q_learning_left' and
    'double_q_learning_left': fraction of runs taking 'left' at state A per
    episode (percentage), averaged over repeats.
    """
    left_q = np.zeros((repeats, n_episodes))
    left_dq = np.zeros((repeats, n_episodes))
    for r in range(repeats):
        rep_seed = None if seed is None else seed + r
        env = MaximizationBiasMDP(seed=rep_seed)
        rng = np.random.default_rng(rep_seed)
        left_q[r] = _q_learning_left_fractions(
            env, rng, n_episodes, alpha, epsilon, gamma, env.ACTIONS.index("left")
        )
        # --- Double Q-learning ---
        env = MaximizationBiasMDP(seed=rep_seed)
        _, _, first_actions = double_q_learning(
            env, n_episodes=n_episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=rep_seed
        )
        left_idx = env.ACTIONS.index("left")
        left_dq[r] = [1.0 if a == left_idx else 0.0 for a in first_actions]
    episodes = np.arange(1, n_episodes + 1)
    return {
        "episodes": episodes,
        "q_learning_left": 100.0 * left_q.mean(axis=0),
        "double_q_learning_left": 100.0 * left_dq.mean(axis=0),
    }


# ---------------------------------------------------------------------------
# Batch updating (Example 6.3, Figure 6.2)
# ---------------------------------------------------------------------------

def _walk_transitions(rng) -> list[tuple[str, float, str, bool]]:
    """Sample one random-walk episode; return its (state, reward, next, done) list."""
    env = RandomWalkEnv(seed=int(rng.integers(0, 2**31 - 1)))
    transitions = []
    state = env.reset()
    done = False
    while not done:
        action = "left" if rng.integers(0, 2) == 0 else "right"
        reward, next_state, done = env.step(action)
        transitions.append((state, float(reward), next_state, bool(done)))
        state = next_state
    return transitions


def batch_td_fixed_point(batch: list[list[tuple[str, float, str, bool]]], gamma: float = 1.0) -> dict:
    """Exact converged batch TD(0) answer for an MRP batch.

    Under batch updating, TD(0) converges deterministically (for alpha
    sufficiently small) to the V solving V(s) = mean over nonterminal visits
    of (r + gamma * V(s')). This is a 5x5 linear system; solve it directly.
    States never visited in the batch keep V = 0.5 (the initialization used
    in Example 6.2/6.3).
    """
    states = RandomWalkEnv.STATES
    idx = {s: i for i, s in enumerate(states)}
    mat = np.zeros((len(states), len(states)))
    rhs = np.zeros(len(states))
    counts = np.zeros(len(states), dtype=int)
    for trajectory in batch:
        for state, reward, next_state, done in trajectory:
            i = idx[state]
            mat[i, i] += 1.0
            counts[i] += 1
            rhs[i] += reward
            if not done:
                mat[i, idx[next_state]] -= gamma
    values = np.full(len(states), 0.5)
    visited = counts > 0
    if visited.any():
        solved = np.linalg.solve(mat[np.ix_(visited, visited)], rhs[visited])
        values[visited] = solved
    return {state: float(values[i]) for i, state in enumerate(states)}


def batch_mc_values(batch: list[list[tuple[str, float, str, bool]]], gamma: float = 1.0) -> dict:
    """Batch constant-alpha MC answer: sample averages of first-visit returns.

    Under batch updating the constant-alpha MC method converges to
    V(s) = mean of the returns experienced after visiting s (first-visit),
    optimal in the MSE-from-actual-returns sense (Example 6.3/Fig. 6.2).
    States not visited keep V = 0.5.
    """
    values = {s: 0.5 for s in RandomWalkEnv.STATES}
    sums = {s: 0.0 for s in RandomWalkEnv.STATES}
    counts = {s: 0 for s in RandomWalkEnv.STATES}
    for trajectory in batch:
        running = 0.0
        returns = [0.0] * len(trajectory)
        for t in range(len(trajectory) - 1, -1, -1):
            running = trajectory[t][1] + gamma * running
            returns[t] = running
        seen = set()
        for t, (state, _, _, _) in enumerate(trajectory):
            if state in seen:
                continue
            seen.add(state)
            sums[state] += returns[t]
            counts[state] += 1
    for s in RandomWalkEnv.STATES:
        if counts[s] > 0:
            values[s] = sums[s] / counts[s]
    return values


def run_batch_updating(repeats: int = 100, n_walks: int = 100, seed: int = 0) -> dict:
    """Example 6.3 / Figure 6.2: RMS error vs walks seen under batch updating.

    For each repeat: sample n_walks random-walk episodes up front. After walk
    k (k = 1..n_walks), solve the batch problem exactly on the first k walks
    and record the RMS error against the true values. The book averaged over
    100 repetitions of the whole experiment;
    repeats defaults to that.

    Returns dict with 'walks' (1..n_walks), 'td', 'mc': mean RMS errors.
    """
    true_values = random_walk_true_values()
    td = np.zeros((repeats, n_walks))
    mc = np.zeros((repeats, n_walks))
    rng = np.random.default_rng(seed)
    for rep in range(repeats):
        rep_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        walks = [_walk_transitions(rep_rng) for _ in range(n_walks)]
        for k in range(1, n_walks + 1):
            batch = walks[:k]
            td_v = batch_td_fixed_point(batch)
            mc_v = batch_mc_values(batch)
            td[rep, k - 1] = _rms_error(td_v, true_values)
            mc[rep, k - 1] = _rms_error(mc_v, true_values)
    walks = np.arange(1, n_walks + 1)
    return {"walks": walks, "td": td.mean(axis=0), "mc": mc.mean(axis=0)}


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 6)
# ---------------------------------------------------------------------------

def plot_figure_6_2_batch_updating(save_path=None, show=False, repeats=100, n_walks=100, seed=0):
    """Batch updating on the random walk (Example 6.3, Figure 6.2).

    RMS error (averaged over the five states) of batch TD(0) vs batch
    constant-alpha MC, as a function of the number of walks seen so far. The
    batch answer is computed exactly (deterministic, independent of alpha
    for small alpha): TD solves the usual TD fixed-point equations; MC
    returns sample averages of first-visit returns. The book averaged over
    100 repetitions of the experiment.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_batch_updating(repeats=repeats, n_walks=n_walks, seed=seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(results["walks"], results["td"], label="TD")
    ax.plot(results["walks"], results["mc"], label="MC")
    ax.set_xlabel("Walks / Episodes")
    ax.set_ylabel("RMS error, averaged over states")
    ax.set_ylim(bottom=0)
    ax.set_title("S&B (2nd ed.), Sec. 6.3, Example 6.3 / Fig. 6.2:\nBatch updating on the random walk")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_6_3_windy_gridworld(save_path=None, show=False, n_episodes=8000, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, window=100):
    """Sarsa on the windy gridworld (Example 6.5, Figure 6.3).

    Steps per episode vs episode for Sarsa with epsilon-greedy exploration.
    The book used alpha = 0.5, epsilon = 0.1 (u = ..., undiscounted, Q init
    0) and plotted raw steps per episode for a single run. We plot the raw
    curve plus a moving average (window=100), which shows the same trend
    more clearly. After enough episodes the greedy policy is optimal;
    epsilon-greedy keeps averaging somewhat above the optimum.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rl_envs.windy_gridworld import WindyGridWorld

    env = WindyGridWorld()
    _, steps, _ = sarsa_control(env, n_episodes=n_episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed)
    steps = np.asarray(steps, dtype=float)
    x = np.arange(1, n_episodes + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, steps, alpha=0.3, lw=0.8, label="steps per episode")
    if window > 1 and len(steps) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(steps, kernel, mode="valid")
        ax.plot(x[window - 1 :], smooth, label=f"moving avg ({window} episodes)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per episode")
    ax.set_ylim(bottom=0)
    ax.set_title("S&B (2nd ed.), Sec. 6.4, Example 6.5 / Fig. 6.3:\nSarsa on the windy gridworld (epsilon-greedy, alpha=%.1f)" % alpha)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_6_4_cliff_walking(save_path=None, show=False, n_episodes=500, epsilon=0.1, alpha=0.5, gamma=1.0, seed=0):
    """The cliff-walking task (Example 6.6, Figure 6.4).

    Sum of (undiscounted, episodic) rewards per episode over 500 episodes,
    for Sarsa vs Q-learning, epsilon-greedy with epsilon = 0.1. The book
    plots a single run, smoothed by averaging over 10 successive episodes.
    Sarsa (on-policy) prefers the safe route; Q-learning (off-policy)
    prefers the optimal but risky cliff-edge route.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env_s = CliffWalkingEnv()
    env_q = CliffWalkingEnv()
    _, _, ret_sarsa = sarsa_control(env_s, n_episodes=n_episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed)
    _, _, ret_ql = q_learning_control(env_q, n_episodes=n_episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed)
    x = np.arange(1, n_episodes + 1)

    def smooth10(y):
        return np.convolve(y, np.ones(10) / 10, mode="valid")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, ret_ql, alpha=0.35, lw=0.8, color="tab:orange", label="Q-learning (raw)")
    ax.plot(x, ret_sarsa, alpha=0.35, lw=0.8, color="tab:blue", label="Sarsa (raw)")
    ax.plot(x[9:], smooth10(np.asarray(ret_ql, dtype=float)), color="tab:orange", label="Q-learning (avg of 10 episodes)")
    ax.plot(x[9:], smooth10(np.asarray(ret_sarsa, dtype=float)), color="tab:blue", label="Sarsa (avg of 10 episodes)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Sum of rewards during episode")
    ax.set_ylim(-100, 0)  # book-style view: early episodes fall below and are clipped
    ax.set_title("S&B (2nd ed.), Sec. 6.5/6.6, Example 6.6 / Fig. 6.4:\nThe cliff-walking task (epsilon-greedy, epsilon=%.1f)" % epsilon)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


_TD_CONTROL_ALGOS = {
    "sarsa": sarsa_control,
    "q_learning": q_learning_control,
    "expected_sarsa": expected_sarsa_control,
}


def _td_alpha_sweep(
    alphas,
    repeats: int,
    n_episodes: int,
    interim_window: int,
    asymptotic_window: int,
    epsilon: float,
    gamma: float,
    seed: int | None,
) -> tuple[dict, dict, dict, dict]:
    """Collect mean/std returns per algorithm and step size (Figure 6.6)."""
    algos = _TD_CONTROL_ALGOS
    interim = {name: [] for name in algos}
    interim_std = {name: [] for name in algos}
    asympt = {name: [] for name in algos}
    asympt_std = {name: [] for name in algos}
    for alpha in alphas:
        per_algo = {name: [] for name in algos}
        for rep in range(repeats):
            # Reuse a seed across algorithms so each alpha is a like-for-like
            # comparison; only the repeat index changes the environment noise.
            rep_seed = None if seed is None else seed + rep
            for name, fn in algos.items():
                _, _, returns = fn(
                    CliffWalkingEnv(),
                    n_episodes=n_episodes,
                    alpha=alpha,
                    epsilon=epsilon,
                    gamma=gamma,
                    seed=rep_seed,
                )
                returns = np.asarray(returns, dtype=float)
                per_algo[name].append(
                    (returns[:interim_window].mean(), returns[-asymptotic_window:].mean())
                )
        for name in algos:
            values = np.array(per_algo[name])
            interim[name].append(values[:, 0].mean())
            interim_std[name].append(values[:, 0].std())
            asympt[name].append(values[:, 1].mean())
            asympt_std[name].append(values[:, 1].std())
    return interim, interim_std, asympt, asympt_std


def plot_figure_6_6_td_control_performance(
    save_path=None,
    show=False,
    alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    n_episodes=5_000,
    interim_window=100,
    asymptotic_window=1_000,
    repeats=5,
    epsilon=0.1,
    gamma=1.0,
    seed=0,
):
    """Interim and asymptotic performance vs alpha (Figure 6.6).

    Cliff walking; Sarsa, Q-learning, Expected Sarsa. 'Interim' performance
    = average return over the first interim_window episodes; 'asymptotic'
    = average return over the last asymptotic_window episodes.
    All runs use epsilon-greedy behavior with epsilon (book 0.1).

    The 2nd-ed. figure is adapted from van Seijen et al. (2009) and used
    n=100 interim episodes averaged over 50,000 runs, and n=100,000
    asymptotic episodes averaged over 10 runs. We use far fewer runs
    (repeats) and episodes - this reproduces the *shape* of the
    phenomenon, not the published error bars; it is an illustrative
    demonstration, not a pixel-exact reproduction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interim, interim_std, asympt, asympt_std = _td_alpha_sweep(
        alphas,
        repeats,
        n_episodes,
        interim_window,
        asymptotic_window,
        epsilon,
        gamma,
        seed,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    x = list(alphas)
    for name, label in (("sarsa", "Sarsa"), ("q_learning", "Q-learning"), ("expected_sarsa", "Expected Sarsa")):
        ax1.errorbar(x, interim[name], yerr=interim_std[name], marker="o", capsize=3, label=label)
        ax2.errorbar(x, asympt[name], yerr=asympt_std[name], marker="o", capsize=3, label=label)
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Average return per episode")
    ax1.set_title("Interim")
    ax1.legend()
    ax2.set_xlabel("alpha")
    ax2.set_title("Asymptotic")
    ax2.legend()
    fig.suptitle("S&B (2nd ed.), Sec. 6.6, Fig. 6.6 (adapted from van Seijen et al. 2009)\nTD control on the cliff-walking task vs step size (reduced runs; illustrative)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_figure_6_7_double_q_learning(save_path=None, show=False, n_episodes=300, repeats=500, alpha=0.1, epsilon=0.1, gamma=1.0, seed=0, smooth_window=10):
    """Q-learning vs Double Q-learning (Example 6.7, Figure 6.7).

    Percentage of runs taking the 'left' (suboptimal) action from state A
    per episode. The book used 10,000 runs; we use
    repeats (default 100,
    increase for smoother curves). Curves are smoothed over
    smooth_window episodes for display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = maximization_bias_experiment(
        n_episodes=n_episodes, repeats=repeats, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed
    )
    episodes = results["episodes"]

    def smooth(y, w):
        if w <= 1:
            return y
        return np.convolve(y, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(episodes, results["q_learning_left"], alpha=0.3, lw=0.8, color="tab:orange", label="Q-learning (raw)")
    ax.plot(episodes, results["double_q_learning_left"], alpha=0.3, lw=0.8, color="tab:green", label="Double Q-learning (raw)")
    ax.plot(episodes[smooth_window - 1:], smooth(results["q_learning_left"], smooth_window), color="tab:orange", label="Q-learning")
    ax.plot(episodes[smooth_window - 1:], smooth(results["double_q_learning_left"], smooth_window), color="tab:green", label="Double Q-learning")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("% left")
    ax.set_ylim(bottom=0)
    ax.set_title("S&B (2nd ed.), Sec. 6.7, Example 6.7 / Fig. 6.7:\nComparison of Q-learning and Double Q-learning (reduced runs; illustrative)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_6_9_king_moves(save_path=None, show=False, n_episodes=2000, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, window=100):
    """King's-moves windy gridworld (Exercise 6.9) - illustrative.

    No numbered book figure: Exercise 6.9 extends Example 6.5 with 8-direction
    ("king's move") actions. We compare Sarsa steps/episode for the standard
    4-action and 9-action (8 moves + stay, as in the exercise) versions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rl_envs.windy_gridworld import WindyGridWorld

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, kwargs, color in (
        ("4 actions (standard)", {}, "tab:blue"),
        ("9 actions (king's moves + stay)", {"king_moves": True, "allow_stay": True}, "tab:red"),
    ):
        _, steps, _ = sarsa_control(
            WindyGridWorld(**kwargs), n_episodes=n_episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed
        )
        steps = np.asarray(steps, dtype=float)
        x = np.arange(1, n_episodes + 1)
        if window > 1 and len(steps) >= window:
            smooth = np.convolve(steps, np.ones(window) / window, mode="valid")
            ax.plot(x[window - 1:], smooth, color=color, label=f"{label} (moving avg {window})")
        else:
            ax.plot(x, steps, color=color, label=label)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per episode")
    ax.set_ylim(bottom=0)
    ax.set_title("S&B (2nd ed.), Ex. 6.9 (no numbered book figure):\nWindy gridworld with king's-move actions (illustrative)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_6_10_stochastic_wind(save_path=None, show=False, n_episodes=3000, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, window=100):
    """Stochastic-wind gridworld (Exercise 6.10) - illustrative.

    No numbered book figure: Exercise 6.10 re-solves the windy gridworld with
    king's moves when the wind, if any, is stochastic (one cell above / as
    specified / one cell below, each a third of the time). We compare Sarsa
    steps/episode for deterministic king's moves vs stochastic king's moves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rl_envs.windy_gridworld import WindyGridWorld

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, kwargs, color in (
        ("deterministic wind, king's moves", {"king_moves": True, "allow_stay": True}, "tab:blue"),
        ("stochastic wind, king's moves (Ex. 6.10)", {"king_moves": True, "allow_stay": True, "stochastic_wind": True}, "tab:purple"),
    ):
        _, steps, _ = sarsa_control(
            WindyGridWorld(**kwargs), n_episodes=n_episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed
        )
        steps = np.asarray(steps, dtype=float)
        x = np.arange(1, n_episodes + 1)
        if window > 1 and len(steps) >= window:
            smooth = np.convolve(steps, np.ones(window) / window, mode="valid")
            ax.plot(x[window - 1:], smooth, color=color, label=f"{label} (moving avg {window})")
        else:
            ax.plot(x, steps, color=color, label=label)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per episode")
    ax.set_ylim(bottom=0)
    ax.set_title("S&B (2nd ed.), Ex. 6.10 (no numbered book figure):\nWindy gridworld with stochastic wind (illustrative)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
