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

from rl_envs.blackjack import BlackjackEnv, ACTIONS


def _validate_policy_fn(policy_fn: Callable) -> None:
    if not callable(policy_fn):
        raise TypeError("policy_fn must be callable")


def _actions_for(env) -> tuple:
    """Action table for an environment (falls back to blackjack's).

    Both ``BlackjackEnv`` (ACTIONS = ('hit', 'stick'), index 0/1) and
    ``RacetrackEnv`` (9 indexed acceleration tuples) expose a class-level
    ``ACTIONS`` tuple whose entries are what ``env.step`` expects.
    """
    return tuple(getattr(env, "ACTIONS", ACTIONS))


def _sample_action(state, policy_fn, rng: np.random.Generator, actions: tuple) -> int:
    probs = np.asarray(policy_fn(state), dtype=np.float64)
    n = len(actions)
    if probs.shape != (n,):
        raise ValueError(f"policy_fn must return a vector of length {n}")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("policy probabilities must sum to 1")
    return int(rng.choice(n, p=probs))


def _run_episode(env, policy_fn, rng: np.random.Generator) -> list[tuple]:
    """Return [(state, action_index, reward), ...] for one episode."""
    actions = _actions_for(env)
    state = env.reset()
    trajectory = []
    while True:
        action_index = _sample_action(state, policy_fn, rng, actions)
        reward, observation, done = env.step(actions[action_index])
        trajectory.append((state, action_index, reward))
        if done:
            return trajectory
        state = observation


def _returns_from_rewards(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def mc_first_visit_prediction(
    env,
    policy_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """First-visit MC prediction (Algorithm 5.1). Book Section 5.1.

    Estimates v_pi(s) as the average of returns following FIRST visits to s.
    """
    _validate_policy_fn(policy_fn)
    rng = np.random.default_rng(seed)
    returns_by_state: dict = {}
    for _ in range(n_episodes):
        trajectory = _run_episode(env, policy_fn, rng)
        rewards = [r for _, _, r in trajectory]
        returns = _returns_from_rewards(rewards, gamma)
        seen: set = set()
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
) -> dict:
    """Every-visit MC prediction (Algorithm 5.1). Book Section 5.1.

    Estimates v_pi(s) as the average of returns following ALL visits to s.
    """
    _validate_policy_fn(policy_fn)
    rng = np.random.default_rng(seed)
    returns_by_state: dict = {}
    for _ in range(n_episodes):
        trajectory = _run_episode(env, policy_fn, rng)
        rewards = [r for _, _, r in trajectory]
        returns = _returns_from_rewards(rewards, gamma)
        for t, (state, _, _) in enumerate(trajectory):
            returns_by_state.setdefault(state, []).append(returns[t])
    return {s: float(np.mean(g)) for s, g in returns_by_state.items()}


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    """Epsilon-greedy probabilities (the closest epsilon-soft policy to greedy)."""
    q_values = np.asarray(q_values, dtype=np.float64)
    n = len(q_values)
    greedy = np.flatnonzero(q_values == q_values.max())
    probs = np.full(n, epsilon / n, dtype=np.float64)
    probs[greedy] += (1.0 - epsilon) / len(greedy)
    return probs


def mc_control_epsilon_greedy(
    env,
    n_episodes: int = 20_000,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict, list]:
    """On-policy first-visit MC control for epsilon-soft policies (Algorithm 5.3).

    Book Section 5.4. The behavior and target policy is the epsilon-greedy
    policy with respect to Q; the greedy action gets 1 - epsilon + epsilon/|A|,
    the others share epsilon/|A|.

    Returns:
        (policy, Q, episode_rewards): policy[state] = probability vector over
        ACTIONS; q_table[state] = action-value array of shape (2,).
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    actions = _actions_for(env)
    n_actions = len(actions)
    rng = np.random.default_rng(seed)

    q_table: dict = {}
    returns_by_pair: dict = {}

    def behavior_probs(state):
        if state not in q_table:
            return np.ones(n_actions) / n_actions
        return _epsilon_greedy_probs(q_table[state], epsilon)

    episode_rewards: list[float] = []
    for _ in range(n_episodes):
        trajectory = _run_episode(env, behavior_probs, rng)
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))
        returns = _returns_from_rewards(rewards, gamma)
        seen: set = set()
        for t, (state, action, _) in enumerate(trajectory):
            if (state, action) in seen:
                continue
            seen.add((state, action))
            pair = (state, action)
            returns_by_pair.setdefault(pair, []).append(returns[t])
            q_table.setdefault(state, np.zeros(n_actions))
            q_table[state][action] = float(np.mean(returns_by_pair[pair]))

    policy = {s: _epsilon_greedy_probs(q, epsilon) for s, q in q_table.items()}
    return policy, q_table, episode_rewards


def _importance_ratios(
    trajectory: list[tuple],
    target_fn: Callable,
    behavior_fn: Callable,
) -> list[float]:
    """Importance sampling ratios rho_{t:T-1} for each timestep."""
    n = len(trajectory)
    ratios = [1.0] * n
    for t in range(n - 1, -1, -1):
        state, action, _ = trajectory[t]
        pi_prob = float(target_fn(state)[action])
        b_prob = float(behavior_fn(state)[action])
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
    _validate_policy_fn(target_fn)
    _validate_policy_fn(behavior_fn)
    rng = np.random.default_rng(seed)

    scaled: dict = {}
    weights_sum: dict = {}

    for _ in range(n_episodes):
        trajectory = _run_episode(env, behavior_fn, rng)
        rewards = [r for _, _, r in trajectory]
        returns = _returns_from_rewards(rewards, gamma)
        ratios = _importance_ratios(trajectory, target_fn, behavior_fn)
        seen: set = set()
        for t, (state, _, _) in enumerate(trajectory):
            if first_visit and state in seen:
                continue
            seen.add(state)
            scaled.setdefault(state, []).append((ratios[t], returns[t]))
            weights_sum[state] = weights_sum.get(state, 0.0) + ratios[t]

    estimates: dict = {}
    for state, pairs in scaled.items():
        if weighted:
            den = weights_sum[state]
            estimates[state] = (
                float(np.sum([w * g for w, g in pairs]) / den) if den > 0.0 else 0.0
            )
        else:
            estimates[state] = float(np.mean([w * g for w, g in pairs]))
    return estimates


def mc_control_off_policy(
    env,
    n_episodes: int = 20_000,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    weighted: bool = True,
    seed: int | None = None,
) -> tuple[dict, dict, list]:
    """Off-policy MC control (Algorithms 5.4/5.5). Book Section 5.7.

    Behavior policy is epsilon-soft (epsilon-greedy w.r.t. Q); the target
    policy is greedy w.r.t. Q. Uses weighted importance sampling (default) or
    ordinary importance sampling (visit-count-normalized scaled returns).

    Returns:
        (policy, Q, episode_rewards): policy[state] = probability vector over ACTIONS.
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    actions = _actions_for(env)
    n_actions = len(actions)
    rng = np.random.default_rng(seed)

    q_table: dict = {}
    counts: dict = {}

    def behavior_probs(state):
        if state not in q_table:
            return np.ones(n_actions) / n_actions
        return _epsilon_greedy_probs(q_table[state], epsilon)

    def greedy_action(state) -> int:
        q = q_table.get(state)
        if q is None:
            return 0
        return int(np.argmax(q))

    episode_rewards: list[float] = []
    for _ in range(n_episodes):
        trajectory = _run_episode(env, behavior_probs, rng)
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))

        g = 0.0
        w = 1.0
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            g = gamma * g + reward
            q_table.setdefault(state, np.zeros(n_actions))
            pair_counts = counts.setdefault(state, np.zeros(n_actions))

            if weighted:
                pair_counts[action] += w
                pair_counts[action] = max(pair_counts[action], 1e-8)
                q_table[state][action] += (w / pair_counts[action]) * (g - q_table[state][action])
            else:
                # ordinary IS: simple average of w*g over visits
                pair_counts[action] += 1.0
                q_table[state][action] += (w * g - q_table[state][action]) / pair_counts[action]

            # target is greedy; stop bootstrapping from non-greedy suffixes
            if action != greedy_action(state):
                break
            b_prob = behavior_probs(state)[action]
            w = w * (1.0 / b_prob) if b_prob > 0.0 else 0.0

    policy = {}
    for state in q_table:
        probs = np.zeros(n_actions)
        greedy = np.flatnonzero(q_table[state] == q_table[state].max())
        probs[greedy] = 1.0 / len(greedy)
        policy[state] = probs
    return policy, q_table, episode_rewards


def blackjack_stick_on_threshold(stick_threshold: int = 20) -> Callable:
    """Deterministic policy: stick when player_sum >= stick_threshold.

    Book Example 5.1 used "stick on 20 or 21"; returned as a probability vector.
    """

    def policy_fn(state) -> np.ndarray:
        player_sum = state[0]
        if player_sum >= stick_threshold:
            return np.array([0.0, 1.0])
        return np.array([1.0, 0.0])

    return policy_fn


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 5)
# ---------------------------------------------------------------------------

def _blackjack_value_grid(values: dict, usable_ace: bool):
    """Build a (player_sum 12-21) x (dealer_showing 1-10) grid from MC values."""
    grid = np.full((10, 10), np.nan, dtype=np.float64)
    for player_sum in range(12, 22):
        for dealer in range(1, 11):
            state = (player_sum, dealer, usable_ace)
            if state in values:
                grid[player_sum - 12, dealer - 1] = values[state]
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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    env = BlackjackEnv(seed=seed)
    policy = blackjack_stick_on_threshold(20)
    values = mc_first_visit_prediction(env, policy, n_episodes=n_episodes, seed=seed)

    X, Y = np.meshgrid(np.arange(1, 11), np.arange(12, 22))

    fig = plt.figure(figsize=(11, 4.8))
    for idx, (usable, title) in enumerate(((True, "Usable ace"), (False, "No usable ace")), start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        Z = _blackjack_value_grid(values, usable)
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
    """Alias for blackjack value function plot (Fig. 5.1)."""
    return plot_blackjack_value_functions(save_path=save_path, show=show, n_episodes=n_episodes, seed=seed)


# ---------------------------------------------------------------------------
# Monte Carlo ES (Algorithm 5.3) — Figure 5.2
# ---------------------------------------------------------------------------

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
    """
    actions = _actions_for(env)
    n_actions = len(actions)
    rng = np.random.default_rng(seed)
    q_table: dict = {}
    returns_by_pair: dict = {}

    def greedy_probs(state) -> np.ndarray:
        q = q_table.get(state)
        if q is None:
            return np.full(n_actions, 1.0 / n_actions)
        greedy = np.flatnonzero(q == q.max())
        probs = np.zeros(n_actions, dtype=np.float64)
        probs[greedy] = 1.0 / len(greedy)
        return probs

    episode_rewards: list[float] = []
    for _ in range(n_episodes):
        state = env.reset()
        a_idx = int(rng.integers(n_actions))  # exploring starts
        trajectory: list[tuple] = []
        reward, observation, done = env.step(actions[a_idx])
        trajectory.append((state, a_idx, reward))
        state = observation
        while not done:
            a_idx = _sample_action(state, greedy_probs, rng, actions)
            reward, observation, done = env.step(actions[a_idx])
            trajectory.append((state, a_idx, reward))
            state = observation
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))
        returns = _returns_from_rewards(rewards, gamma)
        seen: set = set()
        for t, (s, a, _) in enumerate(trajectory):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            pair = (s, a)
            returns_by_pair.setdefault(pair, []).append(returns[t])
            q_table.setdefault(s, np.zeros(n_actions))
            q_table[s][a] = float(np.mean(returns_by_pair[pair]))

    policy = {s: pd_greedy(q) for s, q in q_table.items()}
    return policy, q_table, episode_rewards


def pd_greedy(q: np.ndarray) -> np.ndarray:
    """Uniform-over-argmax probability vector for action-value array q."""
    q = np.asarray(q, dtype=np.float64)
    greedy = np.flatnonzero(q == q.max())
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
    """

    ACTIONS = ("left", "right")

    def __init__(self, p_stay: float = 0.9, seed: int | None = None) -> None:
        if not 0.0 < p_stay < 1.0:
            raise ValueError("p_stay must be in (0, 1)")
        self.p_stay = float(p_stay)
        self.rng = np.random.default_rng(seed)
        self._state = "s0"

    def reset(self) -> str:
        self._state = "s0"
        return self._state

    def step(self, action: str) -> tuple[float, str, bool]:
        if action == "left":
            if self.rng.random() < self.p_stay:
                return 0.0, "s0", False
            self._state = "ABS"
            return 1.0, "ABS", True
        if action == "right":
            self._state = "ABS"
            return 0.0, "ABS", True
        raise ValueError(f"unknown action {action!r}")


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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    env = BlackjackEnv(seed=seed)
    _, q_table, episode_rewards = mc_exploring_starts(env, n_episodes=n_episodes, seed=seed)

    def q_grid(usable: bool, reducer: Callable[[np.ndarray], float]) -> np.ndarray:
        grid = np.full((10, 10), np.nan, dtype=np.float64)
        for player_sum in range(12, 22):
            for dealer in range(1, 11):
                state = (player_sum, dealer, usable)
                if state in q_table:
                    grid[player_sum - 12, dealer - 1] = reducer(q_table[state])
        return grid

    X, Y = np.meshgrid(np.arange(1, 11), np.arange(12, 22))

    fig = plt.figure(figsize=(12, 9))
    for idx, (usable, title) in enumerate(((True, "Usable ace"), (False, "No usable ace")), start=1):
        ax = fig.add_subplot(2, 2, idx)
        policy_map = q_grid(usable, lambda q: int(np.argmax(q) == 0))  # 1 = hit
        im = ax.imshow(policy_map, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(12, 22))
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title(f"Optimal policy ({title.lower()})")
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.85)
        cbar.ax.set_yticklabels(["stick", "hit"])

    for idx, (usable, title) in enumerate(((True, "Usable ace"), (False, "No usable ace")), start=3):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        Z = q_grid(usable, lambda q: float(np.max(q)))
        Z_plot = np.where(np.isnan(Z), 0.0, Z)
        ax.plot_surface(X, Y, Z_plot, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_zlabel("Value")
        ax.set_zlim(-1, 1)
        ax.set_title(f"v* ({title.lower()})")
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(12, 22, 2))

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
    """Log-spaced checkpoint list 1..9, 10..90, ... up to max_episodes."""
    steps = []
    base = 1
    while base < max_episodes:
        steps += [i * base for i in range(1, 10) if i * base <= max_episodes]
        base *= 10
    if steps[-1] != max_episodes:
        steps.append(max_episodes)
    return steps


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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state0 = (13, 2, True)
    target = blackjack_stick_on_threshold(20)
    behavior = lambda s: np.full(2, 0.5)  # noqa: E731

    env_ref = BlackjackEnv(seed=seed)
    v_ref = mc_first_visit_prediction(
        env_ref, target, n_episodes=ref_episodes, seed=seed
    )[state0]

    step_list = _log_step_list(max_episodes)
    mse_ord = np.zeros(len(step_list))
    mse_wtd = np.zeros(len(step_list))
    for run in range(n_runs):
        env = BlackjackEnv(seed=seed * 1_000_003 + run)
        rng = env.rng
        sum_ord, count, sum_w, sum_wG = 0.0, 0, 0.0, 0.0
        err_o = np.full(len(step_list), np.nan)
        err_w = np.full(len(step_list), np.nan)
        ck = 0
        for ep in range(1, max_episodes + 1):
            traj = _run_episode(env, behavior, rng)
            rewards = [r for _, _, r in traj]
            returns = _returns_from_rewards(rewards, 1.0)
            ratios = _importance_ratios(traj, target, behavior)
            for t, (s, _, _) in enumerate(traj):
                if s != state0:
                    continue
                g, rho = returns[t], ratios[t]
                sum_ord += rho * g
                count += 1
                sum_w += rho
                sum_wG += rho * g
                break  # first-visit only
            if ep == step_list[ck]:
                ord_est = sum_ord / count if count else 0.0
                wtd_est = sum_wG / sum_w if sum_w else 0.0
                err_o[ck] = (ord_est - v_ref) ** 2
                err_w[ck] = (wtd_est - v_ref) ** 2
                ck += 1
                if ck >= len(step_list):
                    break
        mse_ord += err_o
        mse_wtd += err_w
    mse_ord /= n_runs
    mse_wtd /= n_runs

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
    """
    rng = np.random.default_rng(seed)
    rho = np.ones(n_episodes)
    ret = np.zeros(n_episodes)
    alive = np.ones(n_episodes, dtype=bool)
    for _ in range(200):
        if not alive.any():
            break
        act_left = rng.random(n_episodes) < 0.5
        absorb = rng.random(n_episodes) >= p_stay
        taken_left = alive & act_left
        rho[taken_left] *= 2.0
        taken_right = alive & ~act_left
        rho[taken_right] = 0.0
        win = taken_left & absorb
        ret[win] = 1.0
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
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_list = _log_step_list(max_episodes)
    run_curves = []
    for run in range(n_runs):
        samples = _one_state_is_samples(
            max_episodes, seed * 1_000_003 + run, p_stay=0.9
        )
        cumulative = np.cumsum(samples)
        indices = np.array(step_list, dtype=np.int64) - 1
        estimates = cumulative[indices] / np.array(step_list, dtype=np.float64)
        run_curves.append(estimates)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
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

