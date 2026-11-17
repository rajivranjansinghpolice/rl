"""Solution implementation for Chapter 11 - Exploration.

Reference: Sutton & Barto (2nd Ed); repo Chapter 11 covers exploration
strategies, implemented against the bandit material of book Chapter 2
(Sections 2.3, 2.6, 2.7) and applied to the book's one-step tabular
Q-learning control on a small MDP.

Implements: Alg 11.1 (UCB exploration strategy, book Section 2.7; bandit
testbed of Section 2.3 plus UCB action selection in Q-learning on a small
MDP), Exercise 11.1 (Optimism in the face of uncertainty, book Section 2.6;
optimistic initialization vs zero-init, plus an optimistic bonus that
descends with visit counts).
"""

from __future__ import annotations

import math

import numpy as np

from chapter_03.solution import FiniteMDP

ACTIONS = ("up", "down", "left", "right")


def ucb_bonus(t: int, count: int, c: float) -> float:
    """Upper-confidence-bound exploration bonus (book eq. (2.8)).

    bonus = c * sqrt(ln(t) / N), where t counts the selections made so far
    and N is the number of times the action was selected. If N == 0 the
    action is untried and its bonus is infinite (book Section 2.7: "If
    N_t(a) = 0, then a is considered to be a maximizing action").
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


def ucb_select(
    q_values: np.ndarray,
    counts: np.ndarray,
    t: int,
    c: float,
    rng: np.random.Generator | None = None,
) -> int:
    """UCB1-style action selection (book eq. (2.8), Algorithm 11.1 in the repo).

    Selects argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]. Actions with N(a) = 0
    are always selected before any tried action; ties are broken uniformly at
    random when `rng` is given, otherwise deterministically (lowest index).
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if q_values.shape != counts.shape:
        raise ValueError("q_values and counts must have the same shape")
    bonuses = np.array([ucb_bonus(t, int(n), c) for n in counts], dtype=np.float64)
    scores = q_values + bonuses
    best = np.flatnonzero(scores == scores.max())
    if rng is not None:
        return int(rng.choice(best))
    return int(best[0])


class TenArmedBandit:
    """The 10-armed bandit testbed (book Section 2.3).

    For each task, q*(a) ~ N(0, 1) are drawn once and fixed; each play of
    arm a returns N(q*(a), 1). Rewards use the instance's seeded RNG, so a
    task is reproducible from its seed.
    """

    def __init__(self, k: int = 10, seed: int | None = None) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.standard_normal(self.k)
        self.counts = np.zeros(self.k, dtype=np.int64)

    def optimal_action(self) -> int:
        """Return the optimal arm (argmax of the true action values)."""
        return int(np.argmax(self.q_star))

    def play(self, action: int) -> float:
        """Play `action`; returns N(q*(a), 1) and increments the visit count."""
        if not 0 <= action < self.k:
            raise ValueError("action out of range")
        self.counts[action] += 1
        return float(self.q_star[action] + self.rng.standard_normal())


def _bandit_run(
    steps: int,
    select,
    init_q: float,
    seed: int | None,
    k: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    bandit = TenArmedBandit(k=k, seed=seed)
    q_values = np.full(k, init_q, dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    rewards = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=bool)
    for t in range(1, steps + 1):
        action = select(q_values, counts, t, rng)
        reward = bandit.play(action)
        counts[action] += 1
        # Sample average update of Q (book eq. (2.1) incrementally).
        q_values[action] += (reward - q_values[action]) / counts[action]
        rewards[t - 1] = reward
        optimal[t - 1] = action == bandit.optimal_action()
    return {
        "rewards": rewards.tolist(),
        "optimal": optimal.tolist(),
        "counts": counts.tolist(),
        "q_star": bandit.q_star.tolist(),
        "q_values": q_values.tolist(),
        "optimal_action": bandit.optimal_action(),
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
    kept, following the figure. Returns per-step rewards, optimal-action
    indicators and final arm counts.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    def select(q_values, counts, t, rng):
        return ucb_select(q_values, counts, t, c, rng)

    return _bandit_run(steps, select, init_q, seed, k)


def run_epsilon_greedy_bandit(
    steps: int = 1000,
    epsilon: float = 0.1,
    init_q: float = 0.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one epsilon-greedy agent on a k-armed testbed task (book Section 2.3).

    With probability epsilon a uniformly random arm is played; otherwise the
    greedy arm (ties broken randomly) is played. Sample-average action
    values, matching the UCB runs for a fair comparison.
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

    return _bandit_run(steps, select, init_q, seed, k)


def run_optimistic_greedy_bandit(
    steps: int = 1000,
    init_q: float = 5.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one greedy agent with optimistic initialization (book Section 2.6).

    Book Section 2.6 / Figure 2.3: all Q(a) initialized to +5 (far above any
    plausible q* drawn from N(0, 1)) induces greediness to explore; each
    chosen arm is "disappointing", so the agent switches to untried arms
    until every arm has been tried several times. Greedy tie-breaking is
    uniform at random.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    def select(q_values, counts, t, rng):
        best = np.flatnonzero(q_values == q_values.max())
        return int(rng.choice(best))

    return _bandit_run(steps, select, init_q, seed, k)


def descending_bonus(count: int, b0: float = 1.0) -> float:
    """Optimistic bonus that descends with visit counts (Exercise 11.1).

    b(N) = b0 / (1 + N): maximal (b0) for untried actions, strictly
    decreasing, and vanishing as the action is tried. Adding it to greedy
    action selection yields an optimistic greedy-in-the-face-of-uncertainty
    rule whose optimism is temporary, like optimistic initialization, but
    driven by visit counts rather than initial values.
    """
    if b0 < 0.0:
        raise ValueError("b0 must be non-negative")
    if count < 0:
        raise ValueError("count must be non-negative")
    return b0 / (1.0 + count)


def descending_bonus_select(
    q_values: np.ndarray,
    counts: np.ndarray,
    b0: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Greedy selection with the descending OFU bonus (Exercise 11.1)."""
    q_values = np.asarray(q_values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if q_values.shape != counts.shape:
        raise ValueError("q_values and counts must have the same shape")
    bonuses = np.array([descending_bonus(int(n), b0) for n in counts],
                       dtype=np.float64)
    scores = q_values + bonuses
    best = np.flatnonzero(scores == scores.max())
    if rng is not None:
        return int(rng.choice(best))
    return int(best[0])


def run_descending_bonus_bandit(
    steps: int = 1000,
    b0: float = 2.0,
    init_q: float = 0.0,
    k: int = 10,
    seed: int | None = None,
) -> dict:
    """Run one descending-bonus OFU agent on a k-armed testbed (Exercise 11.1)."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    def select(q_values, counts, t, rng):
        return descending_bonus_select(q_values, counts, b0, rng)

    return _bandit_run(steps, select, init_q, seed, k)


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
    on chapter_03's FiniteMDP; kept local so this chapter stands alone.
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
    """Q-learning with UCB1 action selection (Algorithm 11.1, book eq. (2.8)).

    Optimistic (UCB) action selection replaces epsilon-greedy exploration:
    at each step, select argmax_a [Q(s, a) + c * sqrt(ln t / N(s, a))] with
    t the total number of steps taken so far. Untried actions are tried
    first. Updates are the usual one-step tabular Q-learning backups.
    """
    gamma = env.gamma if gamma is None else gamma
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    def select(q_row, n_row, t, rng):
        return ucb_select(q_row, n_row, t, c, rng)

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

    Action selection is argmax_a [Q(s, a) + b0 / (1 + N(s, a))]: greedy in
    the face of uncertainty, where the optimism bonus descends with visit
    count N(s, a) and vanishes as actions are tried. Unlike optimistic
    initialization the bonus is state-action specific and self-terminating.
    """
    gamma = env.gamma if gamma is None else gamma
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    def select(q_row, n_row, t, rng):
        return descending_bonus_select(q_row, n_row, b0, rng)

    return _q_learning_loop(env, select, n_episodes, alpha, gamma, seed,
                            max_steps, start_state, 0.0)
