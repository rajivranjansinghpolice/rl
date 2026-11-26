"""Solution implementation for Chapter 2 - Bandit Algorithms.

Reference: Sutton & Barto (2nd Ed), Chapter 2

Implements: Alg 2.1 (k-armed bandit), Alg 2.2 (action-value est.),
Alg 2.3 (epsilon-greedy), Alg 2.4 (UCB), Alg 2.5 (gradient bandit),
Ex 2.3 (non-stationary), Ex 2.4 (optimistic init), Ex 2.5 (UCB unc.),
Ex 2.6 (contextual bandits), plus the exploration strategies formerly
hosted in the old repository Chapter 11 (UCB action selection, optimism
in the face of uncertainty incl. the descending bonus, and their tabular
Q-learning application on a small MDP).
"""

from __future__ import annotations

import math

import numpy as np

from chapter_03.solution import FiniteMDP

ACTIONS = ("up", "down", "left", "right")


class BanditProblem:
    """k-armed bandit problem (Algorithm 2.1). Book Section 2.2.

    The environment has k actions, each with true expected reward q*(a).
    Reward for action a is drawn from N(q*(a), 1). Tracks the number of times
    each action has been selected (N(a)), as needed by the exploration
    strategies (UCB, optimistic initialization, descending bonus).
    """

    def __init__(self, k: int = 10, seed: int | None = None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star: np.ndarray = self.rng.normal(0.0, 1.0, k)
        self.counts: np.ndarray = np.zeros(k, dtype=int)

    def step(self, action: int) -> float:
        if not 0 <= action < self.k:
            raise ValueError("action out of range")
        self.counts[action] += 1
        return float(self.rng.normal(self.q_star[action], 1.0))

    def best_action(self) -> int:
        return int(np.argmax(self.q_star))


class ActionValueEstimator:
    """Action-value estimation using sample-average (Algorithm 2.2). Book Sec 2.3.

    Maintains estimates Q(a) and counts N(a) for each action.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.Q: np.ndarray = np.zeros(k)
        self.N: np.ndarray = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng()

    def select_action(self) -> int:
        best_actions = np.flatnonzero(self.Q == self.Q.max())
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])


def select_epsilon_greedy(
    Q: np.ndarray, epsilon: float, rng: np.random.Generator | None = None
) -> int:
    """Select action using epsilon-greedy policy (Algorithm 2.3). Book Sec 2.3.

    Args:
        Q: Array of action values.
        epsilon: Exploration probability in [0, 1].
        rng: Random number generator.

    Returns:
        Selected action index.

    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    best_actions = np.flatnonzero(Q == Q.max())
    return int(rng.choice(best_actions))


def ucb_bonus(t: int, count: int, c: float) -> float:
    """UCB1 exploration bonus (book eq. 2.8).

    bonus = c * sqrt(ln(t) / N), where t counts the selections made so far
    and N is the number of times the action was selected. If N == 0 the
    action is untried and its bonus is infinite (book Section 2.7: "If
    N_t(a) = 0, then a is considered to be a maximizing action").

    Args:
        t: Current time step (>= 1).
        count: Visit count N for the action (>= 0).
        c: Exploration parameter (>= 0).

    Returns:
        The UCB bonus for the action (infinite for untried actions).

    """
    if c < 0.0:
        raise ValueError("c must be non-negative")
    if t < 1:
        raise ValueError("t must be >= 1")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return math.inf
    return c * math.sqrt(math.log(t) / count)


def select_ucb(
    Q: np.ndarray, N: np.ndarray, t: int, c: float, rng: np.random.Generator | None = None
) -> int:
    """Select action using upper-confidence bound (Algorithm 2.4). Book Sec 2.4 / Eq. 2.8.

    Selects the action maximizing Q(a) + c * sqrt(ln(t) / N(a)); actions with
    N(a) = 0 are treated as maximizing actions (infinite bonus) and are tried
    first.

    Args:
        Q: Array of action value estimates.
        N: Array of action visit counts.
        t: Current time step (>= 1).
        c: Exploration parameter (> 0).
        rng: Optional random number generator used to break ties uniformly at
            random; when None, ties break toward the lowest action index.

    Returns:
        Selected action index.

    """
    q_values = np.asarray(Q, dtype=np.float64)
    counts = np.asarray(N, dtype=np.int64)
    if q_values.shape != counts.shape:
        raise ValueError("Q and N must have the same shape")
    bonuses = np.array([ucb_bonus(t, int(n), c) for n in counts], dtype=np.float64)
    scores = q_values + bonuses
    best_actions = np.flatnonzero(scores == scores.max())
    if rng is not None:
        return int(rng.choice(best_actions))
    return int(best_actions[0])


def select_softmax(preferences: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Select action using softmax policy (Algorithm 2.5). Book Sec 2.5.

    Args:
        preferences: Action preference values H(a).
        rng: Random number generator.

    Returns:
        Selected action index.

    """
    if rng is None:
        rng = np.random.default_rng()
    shifted = preferences - preferences.max()
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()
    return int(rng.choice(len(probs), p=probs))


class GradientBanditAgent:
    """Gradient bandit algorithm (Algorithm 2.5). Book Section 2.5.

    Maintains action preferences H(a) and updates them using the gradient
    of expected reward.
    """

    def __init__(self, k: int, alpha: float = 0.1) -> None:
        self.k = k
        self.alpha = alpha
        self.H: np.ndarray = np.zeros(k)
        self.r_bar: float = 0.0
        self.rng = np.random.default_rng()

    def select_action(self) -> int:
        return select_softmax(self.H, self.rng)

    def update(self, action: int, reward: float) -> None:
        probs = np.zeros(self.k)
        shifted = self.H - self.H.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        delta = reward - self.r_bar
        self.H[action] += self.alpha * delta * (1.0 - probs[action])
        for a in range(self.k):
            if a != action:
                self.H[a] -= self.alpha * delta * probs[a]
        self.r_bar += self.alpha * (reward - self.r_bar)


def simulate_nonstationary(
    k: int = 10,
    steps: int = 1000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    true_reward_change: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate non-stationary bandit with constant step-size (Exercise 2.3).

    Book Exercise 2.3: Non-stationary Bandits.

    Args:
        k: Number of actions.
        steps: Number of steps to simulate.
        epsilon: Epsilon for epsilon-greedy.
        alpha: Constant step-size parameter.
        true_reward_change: Std dev of true reward changes per step.

    Returns:
        Tuple of (action_values, average_rewards).

    """
    rng = np.random.default_rng()
    q_star = rng.normal(0.0, 1.0, k)
    Q = np.zeros(k)
    rewards = np.zeros(steps)

    for t in range(steps):
        q_star += rng.normal(0.0, true_reward_change, k)
        if rng.random() < epsilon:
            action = int(rng.integers(k))
        else:
            action = int(np.argmax(Q))
        reward = float(rng.normal(q_star[action], 1.0))
        rewards[t] = reward
        Q[action] += alpha * (reward - Q[action])

    return Q, rewards


def simulate_optimistic_initial_values(
    k: int = 10,
    steps: int = 1000,
    initial_value: float = 5.0,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Simulate bandit with optimistic initial values (Exercise 2.4).

    Book Exercise 2.4: Optimistic Initial Values.

    Args:
        k: Number of actions.
        steps: Number of steps.
        initial_value: Optimistic initial Q value.
        epsilon: Epsilon (default 0 for pure greedy).

    Returns:
        Tuple of (final_Q, best_action_selection_rate_over_last_100).

    """
    rng = np.random.default_rng()
    bandit = BanditProblem(k, seed=0)
    Q = np.full(k, initial_value)
    N = np.zeros(k, dtype=int)
    best_counts = np.zeros(steps)

    for t in range(steps):
        if rng.random() < epsilon:
            action = int(rng.integers(k))
        else:
            action = int(np.argmax(Q))
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])
        if action == bandit.best_action():
            best_counts[t] = 1.0

    rate = best_counts[-100:].mean()
    return Q, rate


def simulate_ucb_uncertainty(
    k: int = 10,
    steps: int = 1000,
    c: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Simulate UCB with uncertainty parameterization (Exercise 2.5).

    Book Exercise 2.5: UCB with Uncertainty.

    Args:
        k: Number of actions.
        steps: Number of steps.
        c: Exploration parameter.
        seed: Random seed.

    Returns:
        Tuple of (final_Q, best_action_selection_rate).

    """
    np.random.default_rng(seed)
    bandit = BanditProblem(k, seed=seed)
    Q = np.zeros(k)
    N = np.zeros(k, dtype=int)
    best_counts = np.zeros(steps)

    for t in range(1, steps + 1):
        action = select_ucb(Q, N, t, c)
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])
        if action == bandit.best_action():
            best_counts[t - 1] = 1.0

    rate = best_counts[-100:].mean()
    return Q, rate


class ContextualBandit:
    """Contextual bandit algorithm (Exercise 2.6). Book Exercise 2.6.

    Maintains action preferences indexed by (context, action) pairs.
    """

    def __init__(self, num_contexts: int, k: int, alpha: float = 0.1) -> None:
        self.num_contexts = num_contexts
        self.k = k
        self.alpha = alpha
        self.H: np.ndarray = np.zeros((num_contexts, k))
        self.r_bar: np.ndarray = np.zeros(num_contexts)
        self.rng = np.random.default_rng()

    def select_action(self, context: int) -> int:
        return select_softmax(self.H[context], self.rng)

    def update(self, context: int, action: int, reward: float) -> None:
        probs = np.zeros(self.k)
        shifted = self.H[context] - self.H[context].max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        delta = reward - self.r_bar[context]
        self.H[context, action] += self.alpha * delta * (1.0 - probs[action])
        for a in range(self.k):
            if a != action:
                self.H[context, a] -= self.alpha * delta * probs[a]
        self.r_bar[context] += self.alpha * (reward - self.r_bar[context])


# ---------------------------------------------------------------------------
# Exploration strategies (formerly repo Chapter 11: Exploration)
# Sutton & Barto (2nd ed.), Sections 2.6-2.7, Eq. 2.8, Exercise 11.1 (OFU)
# ---------------------------------------------------------------------------

def descending_bonus(count: int, b0: float = 1.0) -> float:
    """Optimistic bonus that descends with visit counts (Exercise 11.1).

    b(N) = b0 / (1 + N): maximal (b0) for untried actions, strictly
    decreasing, and vanishing as the action is tried. Adding it to greedy
    action selection yields an optimistic greedy-in-the-face-of-uncertainty
    (OFU) rule whose optimism is temporary, like optimistic initialization,
    but driven by visit counts rather than initial values.

    Args:
        count: Visit count N for the action (>= 0).
        b0: Initial optimism bonus (>= 0).

    Returns:
        The descending optimistic bonus b(N).

    """
    if b0 < 0.0:
        raise ValueError("b0 must be non-negative")
    if count < 0:
        raise ValueError("count must be non-negative")
    return b0 / (1.0 + count)


def descending_bonus_select(
    q_values: np.ndarray,
    counts: np.ndarray,
    b0: float = 2.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Greedy action selection with the descending OFU bonus (Exercise 11.1).

    Selects the action maximizing Q(a) + b0 / (1 + N(a)); untried actions get
    the maximal bonus and are tried first.

    Args:
        q_values: Array of action value estimates.
        counts: Array of action visit counts.
        b0: Initial optimism bonus (>= 0).
        rng: Optional random number generator used to break ties uniformly at
            random; when None, ties break toward the lowest action index.

    Returns:
        Selected action index.

    """
    q_values = np.asarray(q_values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if q_values.shape != counts.shape:
        raise ValueError("q_values and counts must have the same shape")
    bonuses = np.array([descending_bonus(int(n), b0) for n in counts], dtype=np.float64)
    scores = q_values + bonuses
    best = np.flatnonzero(scores == scores.max())
    if rng is not None:
        return int(rng.choice(best))
    return int(best[0])


def make_small_gridworld(
    rows: int = 2,
    cols: int = 4,
    start: str = "0,0",
    goal: str | None = None,
    gamma: float = 0.95,
    seed: int | None = None,
) -> FiniteMDP:
    """Deterministic 2x4 gridworld for the exploration experiments.

    Same layout as the Dyna gridworld: +1 reward on entering the terminal
    goal, 0 elsewhere, blocked moves keep the agent in place. Built directly
    on chapter_03's FiniteMDP.

    Args:
        rows: Number of rows in the grid (>= 1).
        cols: Number of columns in the grid (>= 1).
        start: Start state label "r,c".
        goal: Goal state label "r,c" (defaults to the bottom-right cell).
        gamma: Discount factor.
        seed: Optional seed for reproducible sampling.

    Returns:
        A FiniteMDP instance.

    """
    states = [f"{r},{c}" for r in range(rows) for c in range(cols)]
    goal = goal if goal is not None else f"{rows - 1},{cols - 1}"
    if goal not in states:
        raise ValueError(f"goal {goal!r} not in grid")
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def move(state: str, action: str) -> str:
        r, c = (int(x) for x in state.split(","))
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            return f"{nr},{nc}"
        return state

    transitions: dict = {}
    for state in states:
        if state == goal:
            continue
        for action in ACTIONS:
            nxt = move(state, action)
            reward = 1.0 if nxt == goal else 0.0
            transitions[(state, action)] = [(1.0, reward, nxt)]
    return FiniteMDP(states, list(ACTIONS), transitions, gamma=gamma,
                     terminal_states=[goal], seed=seed)


def _q_learning_loop(
    env: FiniteMDP,
    select,
    n_episodes: int,
    alpha: float,
    gamma: float,
    seed: int | None,
    max_steps: int,
    start_state: str | None,
    init_q: float,
) -> tuple[dict, dict, dict]:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    terminal_states = set(env.terminal_states)
    q_table = {(s, a): init_q for s in env.states for a in env.actions}
    counts = {(s, a): 0 for s in env.states for a in env.actions}
    stats = {"episode_steps": [], "episode_rewards": [], "time_steps": 0}
    for _ in range(n_episodes):
        state = env.reset(start_state) if start_state is not None else env.reset()
        done = env.done
        steps = 0
        episode_reward = 0.0
        while not done and steps < max_steps:
            t = stats["time_steps"] + 1
            q_row = np.array([q_table[(state, a)] for a in actions], dtype=np.float64)
            n_row = np.array([counts[(state, a)] for a in actions], dtype=np.int64)
            action_index = select(q_row, n_row, t, rng)
            action = actions[action_index]
            reward, next_state, done = env.step(action)
            counts[(state, action)] += 1
            next_best = (
                0.0
                if next_state in terminal_states
                else max(q_table[(next_state, a)] for a in actions)
            )
            q_table[(state, action)] += alpha * (
                reward + gamma * next_best - q_table[(state, action)]
            )
            stats["time_steps"] += 1
            state = next_state
            steps += 1
            episode_reward += reward
        stats["episode_steps"].append(steps)
        stats["episode_rewards"].append(episode_reward)
    return q_table, counts, stats


def q_learning_ucb(
    env: FiniteMDP,
    n_episodes: int = 100,
    alpha: float = 0.1,
    c: float = 2.0,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
) -> tuple[dict, dict, dict]:
    """Q-learning with UCB1 action selection (Algorithm 11.1, book eq. 2.8).

    Optimistic (UCB) action selection replaces epsilon-greedy exploration:
    at each step, select argmax_a [Q(s, a) + c * sqrt(ln t / N(s, a))] with
    t the total number of steps taken so far. Untried actions are tried
    first. Updates are the usual one-step tabular Q-learning backups.

    Args:
        env: The finite MDP to control.
        n_episodes: Number of episodes to run.
        alpha: Learning rate (> 0).
        c: Exploration parameter (> 0).
        gamma: Discount factor (defaults to the MDP's gamma).
        seed: Optional seed for reproducible sampling.
        max_steps: Maximum steps per episode.
        start_state: Optional start state (defaults to a random reset).

    Returns:
        Tuple of (Q table, visit counts, episode statistics).

    """
    gamma = env.gamma if gamma is None else gamma
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    def select(q_row, n_row, t, rng):
        return select_ucb(q_row, n_row, t, c, rng)

    return _q_learning_loop(env, select, n_episodes, alpha, gamma, seed,
                            max_steps, start_state, 0.0)


def q_learning_optimistic(
    env: FiniteMDP,
    n_episodes: int = 100,
    alpha: float = 0.1,
    initial_q: float = 5.0,
    epsilon: float = 0.0,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
    tie_break: str = "random",
) -> tuple[dict, dict, dict]:
    """Q-learning with optimistic initialization (Exercise 11.1; book Sec 2.6).

    Action values are initialized to `initial_q`, far above the rewards
    available, so the agent is systematically "disappointed" by every action
    it tries and cycles through the alternatives before settling. With
    epsilon = 0 this is the greedy method of book Figure 2.3 (there, +5 on
    the 10-armed testbed). `initial_q = 0` with the same settings gives the
    zero-initialized baseline. tie_break='first' breaks greedy ties by lowest
    action index (deterministic), tie_break='random' draws uniformly.

    Args:
        env: The finite MDP to control.
        n_episodes: Number of episodes to run.
        alpha: Learning rate (> 0).
        initial_q: Optimistic initial action value.
        epsilon: Exploration probability in [0, 1].
        gamma: Discount factor (defaults to the MDP's gamma).
        seed: Optional seed for reproducible sampling.
        max_steps: Maximum steps per episode.
        start_state: Optional start state (defaults to a random reset).
        tie_break: Tie-breaking rule for greedy selection.

    Returns:
        Tuple of (Q table, visit counts, episode statistics).

    """
    gamma = env.gamma if gamma is None else gamma
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if tie_break not in ("first", "random"):
        raise ValueError("tie_break must be 'first' or 'random'")

    def select(q_row, n_row, t, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(q_row)))
        best = np.flatnonzero(q_row == q_row.max())
        if tie_break == "first":
            return int(best[0])
        return int(rng.choice(best))

    return _q_learning_loop(env, select, n_episodes, alpha, gamma, seed,
                            max_steps, start_state, initial_q)


def q_learning_descending_bonus(
    env: FiniteMDP,
    n_episodes: int = 100,
    alpha: float = 0.1,
    b0: float = 2.0,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
) -> tuple[dict, dict, dict]:
    """Q-learning with a descending optimistic bonus (Exercise 11.1).

    Action selection is argmax_a [Q(s, a) + b0 / (1 + N(s, a))]: greedy in the
    face of uncertainty, where the optimism bonus descends with visit count
    N(s, a) and vanishes as actions are tried. Unlike optimistic
    initialization the bonus is state-action specific and self-terminating.

    Args:
        env: The finite MDP to control.
        n_episodes: Number of episodes to run.
        alpha: Learning rate (> 0).
        b0: Initial optimism bonus (>= 0).
        gamma: Discount factor (defaults to the MDP's gamma).
        seed: Optional seed for reproducible sampling.
        max_steps: Maximum steps per episode.
        start_state: Optional start state (defaults to a random reset).

    Returns:
        Tuple of (Q table, visit counts, episode statistics).

    """
    gamma = env.gamma if gamma is None else gamma
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    def select(q_row, n_row, t, rng):
        return descending_bonus_select(q_row, n_row, b0, rng)

    return _q_learning_loop(env, select, n_episodes, alpha, gamma, seed,
                            max_steps, start_state, 0.0)


# ---------------------------------------------------------------------------
# Single-task bandit runners for exploration experiments (Sutton & Barto, 2nd ed.)
# ---------------------------------------------------------------------------


def _bandit_task(
    k: int,
    steps: int,
    select,
    init_q: float,
    seed: int | None,
) -> dict:
    """Run one bandit task with sample-average Q updates (book eq. 2.1).

    Args:
        k: Number of actions.
        steps: Number of selections to make.
        select: Action-selection function (Q, N, t, rng) -> action.
        init_q: Initial action value (optimistic init for exploration runs).
        seed: Optional seed for reproducible sampling.

    Returns:
        Dict with per-step rewards, optimal-action indicators, visit counts,
        the true action values and the final action values.

    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    bandit = BanditProblem(k=k, seed=seed)
    q_values = np.full(k, init_q, dtype=np.float64)
    rewards = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=bool)
    for t in range(1, steps + 1):
        action = select(q_values, bandit.counts, t, rng)
        reward = bandit.step(action)
        # Sample average update of Q (book eq. 2.1 incrementally).
        q_values[action] += (reward - q_values[action]) / bandit.counts[action]
        rewards[t - 1] = reward
        optimal[t - 1] = action == bandit.best_action()
    return {
        "rewards": rewards.tolist(),
        "optimal": optimal.tolist(),
        "counts": bandit.counts.tolist(),
        "q_star": bandit.q_star.tolist(),
        "q_values": q_values.tolist(),
        "optimal_action": bandit.best_action(),
    }


def run_ucb_bandit(
    steps: int = 1000,
    c: float = 2.0,
    init_q: float = 0.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one UCB agent on a k-armed testbed task (book Section 2.7, Figure 2.4).

    Book Figure 2.4 used UCB with c = 2 against epsilon-greedy with
    epsilon = 0.1 on the 10-armed testbed. Sample-average action values are
    kept, following the figure.

    Args:
        steps: Number of selections to make.
        c: Exploration parameter (> 0).
        init_q: Initial action value (0 for the plain bandit runs).
        k: Number of actions in the testbed.
        seed: Optional seed for reproducible sampling.

    Returns:
        Dict with per-step rewards, optimal-action indicators, visit counts,
        the true action values and the final action values.

    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    def select(q_values, counts, t, rng):
        return select_ucb(q_values, counts, t, c, rng)

    return _bandit_task(k, steps, select, init_q, seed)


def run_epsilon_greedy_bandit(
    steps: int = 1000,
    epsilon: float = 0.1,
    init_q: float = 0.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one epsilon-greedy agent on a k-armed testbed task (book Section 2.3).

    With probability epsilon a uniformly random arm is played; otherwise the
    greedy arm is played. Sample-average action values, matching the UCB runs
    for a fair comparison.

    Args:
        steps: Number of selections to make.
        epsilon: Exploration probability in [0, 1].
        init_q: Initial action value (0 for the plain bandit runs).
        k: Number of actions in the testbed.
        seed: Optional seed for reproducible sampling.

    Returns:
        Dict with per-step rewards, optimal-action indicators, visit counts,
        the true action values and the final action values.

    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    def select(q_values, counts, t, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(q_values)))
        best = np.flatnonzero(q_values == q_values.max())
        return int(rng.choice(best))

    return _bandit_task(k, steps, select, init_q, seed)


def run_optimistic_greedy_bandit(
    steps: int = 1000,
    init_q: float = 5.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one greedy agent with optimistic initialization (book Section 2.6).

    Book Section 2.6 / Figure 2.3: all Q(a) initialized to +5 (far above any
    plausible q*(a) drawn from N(0, 1)) induces greediness to explore; each
    chosen arm is "disappointing", so the agent switches to untried arms
    until every arm has been tried several times. Greedy tie-breaking is
    uniform at random.

    Args:
        steps: Number of selections to make.
        init_q: Optimistic initial action value (>= 0).
        k: Number of actions in the testbed.
        seed: Optional seed for reproducible sampling.

    Returns:
        Dict with per-step rewards, optimal-action indicators, visit counts,
        the true action values and the final action values.

    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    def select(q_values, counts, t, rng):
        best = np.flatnonzero(q_values == q_values.max())
        return int(rng.choice(best))

    return _bandit_task(k, steps, select, init_q, seed)


def run_descending_bonus_bandit(
    steps: int = 1000,
    b0: float = 2.0,
    init_q: float = 0.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one descending-bonus OFU agent on a k-armed testbed (Exercise 11.1).

    Action selection is greedy with the descending bonus
    b0 / (1 + N(a)) added to Q(a); the bonus is maximal for untried actions
    and vanishes as they are tried.

    Args:
        steps: Number of selections to make.
        b0: Initial optimism bonus (>= 0).
        init_q: Initial action value (>= 0).
        k: Number of actions in the testbed.
        seed: Optional seed for reproducible sampling.

    Returns:
        Dict with per-step rewards, optimal-action indicators, visit counts,
        the true action values and the final action values.

    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    def select(q_values, counts, t, rng):
        return descending_bonus_select(q_values, counts, b0, rng)

    return _bandit_task(k, steps, select, init_q, seed)


# ---------------------------------------------------------------------------
# Experiment runners and figure reproduction (Sutton & Barto, 2nd ed., Ch. 2)
# ---------------------------------------------------------------------------

def _run_bandit_bandit_episode(
    k: int,
    steps: int,
    seed: int,
    select_fn,
    update_fn,
    q_init: float = 0.0,
    step_size: float | None = None,
    track_optimal: bool = True,
):
    rng = np.random.default_rng(seed)
    bandit = BanditProblem(k=k, seed=seed)
    Q = np.full(k, q_init, dtype=np.float64)
    N = np.zeros(k, dtype=np.int64)
    rewards = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=np.float64) if track_optimal else None
    best = bandit.best_action()
    for t in range(1, steps + 1):
        action = select_fn(Q, N, t, rng)
        reward = bandit.step(action)
        N[action] += 1
        if step_size is None:
            Q[action] += (reward - Q[action]) / N[action]
        else:
            Q[action] += step_size * (reward - Q[action])
        rewards[t - 1] = reward
        if optimal is not None:
            optimal[t - 1] = 1.0 if action == best else 0.0
    return rewards, optimal


# Per-run seed offsets give each bandit task its own q* draw and reward noise
# while keeping the figure averages reproducible (Figs 2.2-2.5).
_EPSILON_SEED_STRIDE = 9_973
_EPSILON_SEED_SCALE = 100_000
_OPTIMISTIC_SEED_STRIDE = 7_919
_Q_INIT_SEED_SCALE = 100
_OPTIMISTIC_EPSILON_SCALE = 1_000
_UCB_SEED_STRIDE = 104_729
_GRADIENT_SEED_STRIDE = 65_537
_GRADIENT_ALPHA_SCALE = 100
_GRADIENT_BASELINE_OFFSET = 10
# Fig. 2.5's testbed draws q*(a) ~ N(4, 1) rather than the usual N(0, 1).
_GRADIENT_Q_STAR_MEAN = 4.0


def _epsilon_greedy_selector(epsilon: float):
    """Build an epsilon-greedy selector; greedy ties break uniformly at random."""

    def select_fn(Q, N, t, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(Q)))
        best_actions = np.flatnonzero(Q == Q.max())
        return int(rng.choice(best_actions))

    return select_fn


def _average_testbed_runs(
    k: int,
    steps: int,
    runs: int,
    run_seed,
    select_fn,
    q_init: float,
    step_size: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average reward/optimal-action curves over independent bandit tasks.

    One seed per run keeps the tasks independent; curves are accumulated in run
    order so the averaged result stays exactly reproducible.
    """
    avg_r = np.zeros(steps, dtype=np.float64)
    avg_o = np.zeros(steps, dtype=np.float64)
    for r in range(runs):
        rewards, optimal = _run_bandit_bandit_episode(
            k,
            steps,
            run_seed(r),
            select_fn,
            lambda *a: None,
            q_init=q_init,
            step_size=step_size,
        )
        # Sum first, divide later: the caller owns the final scaling.
        avg_r += rewards
        avg_o += optimal
    return avg_r, avg_o


def run_epsilon_greedy_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    epsilons: tuple[float, ...] = (0.0, 0.01, 0.1),
    q_init: float = 0.0,
    step_size: float | None = None,
    seed: int = 0,
):
    """Run the 10-armed testbed with epsilon-greedy action selection.

    Sutton & Barto (2nd ed.), Sec. 2.3, Fig. 2.2: average reward and
    % optimal action over steps for epsilon-greedy methods with sample-average
    estimates on the stationary 10-armed testbed (q*(a) ~ N(0,1), reward ~ N(q*,1)).
    Book used 2000 runs x 1000 steps; defaults here use fewer runs for speed and
    are configurable.
    """
    results = {}
    for eps in epsilons:
        avg_r, avg_o = _average_testbed_runs(
            k,
            steps,
            runs,
            lambda r: seed + r * _EPSILON_SEED_STRIDE + int(eps * _EPSILON_SEED_SCALE),
            _epsilon_greedy_selector(eps),
            q_init,
            step_size,
        )
        results[eps] = {
            "average_reward": avg_r / runs,
            "optimal_action": (avg_o / runs) * 100.0,
        }
    return results


def run_optimistic_initial_values_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    configs: tuple[tuple[float, float], ...] = ((5.0, 0.0), (0.0, 0.1)),
    seed: int = 0,
):
    """Compare optimistic initialization with realistic initialization.

    Sutton & Barto (2nd ed.), Sec. 2.6, Fig. 2.3: greedy with optimistic
    Q1=+5, alpha=0.1 vs epsilon-greedy (Q1=0, eps=0.1, alpha=0.1) on the
    10-armed testbed.
    """
    results = {}
    for q_init, eps in configs:
        label = f"Q1={q_init:g}, eps={eps:g}"
        avg_r, avg_o = _average_testbed_runs(
            k,
            steps,
            runs,
            lambda r: (
                seed
                + r * _OPTIMISTIC_SEED_STRIDE
                + int(q_init * _Q_INIT_SEED_SCALE)
                + int(eps * _OPTIMISTIC_EPSILON_SCALE)
            ),
            _epsilon_greedy_selector(eps),
            q_init,
            0.1,
        )
        results[label] = {
            "average_reward": avg_r / runs,
            "optimal_action": (avg_o / runs) * 100.0,
        }
    return results


def _ucb_experiment_selector(name: str, c: float, epsilon: float):
    """Pick the action rule being compared in Fig. 2.4 (UCB or epsilon-greedy)."""
    if name != "ucb":
        return _epsilon_greedy_selector(epsilon)

    def select_fn(Q, N, t, rng):
        # Untried arms must be tried before any bonus arithmetic is meaningful.
        untried = np.flatnonzero(N == 0)
        if len(untried) > 0:
            return int(untried[0])
        ucb = Q + c * np.sqrt(np.log(max(t, 1)) / N)
        best_actions = np.flatnonzero(ucb == ucb.max())
        return int(rng.choice(best_actions))

    return select_fn


def run_ucb_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    c: float = 2.0,
    epsilon: float = 0.1,
    seed: int = 0,
):
    """Compare UCB action selection with epsilon-greedy (Fig. 2.4)."""
    results = {}
    for name in ("ucb", "eps_greedy"):
        avg_r, avg_o = _average_testbed_runs(
            k,
            steps,
            runs,
            lambda r: seed + r * _UCB_SEED_STRIDE + (1 if name == "ucb" else 2),
            _ucb_experiment_selector(name, c, epsilon),
            0.0,
            None,
        )
        label = f"UCB c={c:g}" if name == "ucb" else f"eps-greedy eps={epsilon:g}"
        results[label] = {"average_reward": avg_r / runs, "optimal_action": (avg_o / runs) * 100.0}
    return results


def _gradient_bandit_trial(
    k: int,
    steps: int,
    alpha: float,
    with_baseline: bool,
    trial_seed: int,
    avg_r: np.ndarray,
    avg_o: np.ndarray,
) -> None:
    """Run one gradient-bandit task, adding its curves into the running sums."""
    rng = np.random.default_rng(trial_seed)
    bandit = BanditProblem(k=k, seed=trial_seed)
    bandit.q_star = bandit.rng.normal(_GRADIENT_Q_STAR_MEAN, 1.0, k)
    best = bandit.best_action()
    H = np.zeros(k, dtype=np.float64)
    r_bar = 0.0
    for t in range(steps):
        # Softmax needs the shifted preferences to avoid overflow; the sample
        # rule and the preference gradient then share the same probabilities.
        shifted = H - H.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        action = int(rng.choice(k, p=probs))
        reward = bandit.step(action)
        avg_r[t] += reward
        avg_o[t] += 1.0 if action == best else 0.0
        baseline = r_bar if with_baseline else 0.0
        one_hot = np.zeros(k)
        one_hot[action] = 1.0
        H += alpha * (reward - baseline) * (one_hot - probs)
        r_bar += (reward - r_bar) / (t + 1)


def run_gradient_bandit_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    configs: tuple[tuple[float, bool], ...] = ((0.1, True), (0.4, True), (0.1, False), (0.4, False)),
    seed: int = 0,
):
    """Gradient bandit algorithm comparison (Fig. 2.5)."""
    results = {}
    for alpha, with_baseline in configs:
        label = f"alpha={alpha:g}, baseline={with_baseline}"
        avg_r = np.zeros(steps, dtype=np.float64)
        avg_o = np.zeros(steps, dtype=np.float64)
        for r in range(runs):
            rs = (
                seed
                + r * _GRADIENT_SEED_STRIDE
                + int(alpha * _GRADIENT_ALPHA_SCALE)
                + (_GRADIENT_BASELINE_OFFSET if with_baseline else 0)
            )
            _gradient_bandit_trial(k, steps, alpha, with_baseline, rs, avg_r, avg_o)
        results[label] = {"average_reward": avg_r / runs, "optimal_action": (avg_o / runs) * 100.0}
    return results


def plot_figure_2_1(save_path: str | None = None, show: bool = False, seed: int = 0):
    """Fig 2.1: An example bandit problem (10-armed testbed), Sec. 2.3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bandit = BanditProblem(k=10, seed=seed)
    samples = np.array([[bandit.step(a) for _ in range(400)] for a in range(bandit.k)]).T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot([samples[:, a] for a in range(bandit.k)], showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.scatter(range(1, bandit.k + 1), bandit.q_star, color="black", zorder=3, label="q*(a)")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Action")
    ax.set_ylabel("Reward distribution")
    ax.set_title("Sutton & Barto (2nd ed.), Sec. 2.3, Fig. 2.1: 10-armed testbed example")
    ax.set_xticks(range(1, bandit.k + 1))
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_2_2(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.2: Average performance of epsilon-greedy, Sec. 2.3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_epsilon_greedy_experiment(runs=runs, seed=seed)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    for eps, data in results.items():
        label = f"eps = {eps:g}"
        ax1.plot(data["average_reward"], label=label)
        ax2.plot(data["optimal_action"], label=label)
    ax1.set_ylabel("Average reward")
    ax1.set_title("S&B (2nd ed.), Sec. 2.3, Fig. 2.2: epsilon-greedy (10-armed testbed)")
    ax1.legend()
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_ylim(0, 100)
    ax2.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_figure_2_3(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.3: Optimistic initial values, Sec. 2.6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_optimistic_initial_values_experiment(runs=runs, seed=seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, data in results.items():
        ax.plot(data["optimal_action"], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("% Optimal action")
    ax.set_ylim(0, 100)
    ax.set_title("S&B (2nd ed.), Sec. 2.6, Fig. 2.3: Optimistic initial values")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_2_4(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.4: UCB action selection vs epsilon-greedy, Sec. 2.7.

    Sutton & Barto (2nd ed.), Sec. 2.7, Fig. 2.4 / Eq. 2.10: UCB action
    selection (c=2) compared with epsilon-greedy (epsilon=0.1) on the
    10-armed testbed. Curves show average reward and % optimal action over
    steps, averaged over independent bandit tasks/runs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_ucb_experiment(runs=runs, seed=seed)
    steps = len(next(iter(results.values()))["average_reward"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    x = np.arange(1, steps + 1)
    for label, data in results.items():
        ax1.plot(x, data["average_reward"], label=label)
        ax2.plot(x, data["optimal_action"], label=label)
    ax1.set_ylabel("Average reward")
    ax1.set_title("S&B (2nd ed.), Sec. 2.7, Fig. 2.4: UCB action selection")
    ax1.legend()
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_ylim(0, 100)
    ax2.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_figure_2_5(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.5: Gradient bandit algorithms, Sec. 2.8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_gradient_bandit_experiment(runs=runs, seed=seed)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, data in results.items():
        ax.plot(data["optimal_action"], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("% Optimal action")
    ax.set_ylim(0, 100)
    ax.set_title("S&B (2nd ed.), Sec. 2.8, Fig. 2.5: Gradient bandit algorithms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_all_figures_chapter_2(figures_dir: str | None = None, runs: int = 500, seed: int = 0):
    """Generate and save chapter 2 figures (2.1-2.5) to a directory."""
    import os

    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    outputs = {}
    outputs["fig_2_1"] = os.path.join(figures_dir, "fig_2_1_testbed.png")
    plot_figure_2_1(outputs["fig_2_1"], seed=seed)
    outputs["fig_2_2"] = os.path.join(figures_dir, "fig_2_2_epsilon_greedy.png")
    plot_figure_2_2(outputs["fig_2_2"], runs=runs, seed=seed)
    outputs["fig_2_3"] = os.path.join(figures_dir, "fig_2_3_optimistic_initial_values.png")
    plot_figure_2_3(outputs["fig_2_3"], runs=runs, seed=seed)
    outputs["fig_2_4"] = os.path.join(figures_dir, "fig_2_4_ucb.png")
    plot_figure_2_4(outputs["fig_2_4"], runs=runs, seed=seed)
    outputs["fig_2_5"] = os.path.join(figures_dir, "fig_2_5_gradient_bandit.png")
    plot_figure_2_5(outputs["fig_2_5"], runs=runs, seed=seed)
    return outputs
