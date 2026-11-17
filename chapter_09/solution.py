"""Solution implementation for Chapter 9 - Planning and Learning with Tabular Methods.

Reference: Sutton & Barto (2nd Ed), Chapter 8 (repo Chapter 9).

Implements: Alg 9.1 (Tabular Dyna-Q planner, book Section 8.2),
Exercise 9.1 (Prioritized sweeping, book Section 8.4; Moore & Atkeson 1993),
Exercise 9.2 (Expectation vs sample updates Dyna, book Section 8.5),
Exercise 9.3 (World model with stochastic environments, book Sections 8.1-8.3).
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from chapter_03.solution import FiniteMDP, bellman_backup_optimality

ACTIONS = ("up", "down", "left", "right")


def _max_q(q_table: dict, state, actions: list, terminal_states: set) -> float:
    """Max over actions of Q(state, a); 0.0 for terminal states."""
    if state in terminal_states:
        return 0.0
    return max(q_table[(state, a)] for a in actions)


def _epsilon_greedy_action(
    q_table: dict, state, actions: list, epsilon: float, rng: np.random.Generator
):
    """Epsilon-greedy action selection with uniformly random tie-breaking."""
    q_values = np.array([q_table[(state, a)] for a in actions], dtype=np.float64)
    if rng.random() < epsilon:
        return actions[int(rng.integers(len(actions)))]
    best = np.flatnonzero(q_values == q_values.max())
    return actions[int(rng.choice(best))]


def _q_learning_update(
    q_table: dict,
    state,
    action,
    reward: float,
    next_state,
    alpha: float,
    gamma: float,
    actions: list,
    terminal_states: set,
) -> float:
    """One one-step tabular Q-learning backup (book Algorithm 6.2 / eq. 8.2)."""
    target = reward + gamma * _max_q(q_table, next_state, actions, terminal_states)
    old = q_table[(state, action)]
    q_table[(state, action)] = old + alpha * (target - old)
    return q_table[(state, action)] - old


def make_gridworld(
    rows: int = 2,
    cols: int = 4,
    start: str = "0,0",
    goal: str | None = None,
    slip: float = 0.0,
    gamma: float = 0.95,
    seed: int | None = None,
) -> FiniteMDP:
    """Small gridworld used for the Dyna experiments (adapted from Example 8.1).

    Book Section 8.2 (Example 8.1, Dyna maze). States are 'r,c' strings.
    Actions are up/down/left/right; moving into a wall or off the grid keeps
    the agent in place. Reward is +1 on entering the goal (terminal) and 0
    elsewhere. With `slip > 0`, the intended move happens with probability
    ``1 - slip`` and the two perpendicular moves share the remaining
    probability equally (a stochastic analogue of the book's maze); the
    deterministic world is recovered with ``slip = 0``.
    """
    if not 0.0 <= slip < 1.0:
        raise ValueError("slip must be in [0, 1)")
    states = [f"{r},{c}" for r in range(rows) for c in range(cols)]
    goal = goal if goal is not None else f"{rows - 1},{cols - 1}"
    if goal not in states:
        raise ValueError(f"goal {goal!r} not in grid")
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    perpendiculars = {
        "up": ("left", "right"),
        "down": ("left", "right"),
        "left": ("up", "down"),
        "right": ("up", "down"),
    }

    def move(state: str, action: str) -> str:
        r, c = (int(x) for x in state.split(","))
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            return f"{nr},{nc}"
        return state  # blocked: stay in place

    transitions: dict = {}
    for state in states:
        if state == goal:
            continue
        for action in ACTIONS:
            probs: dict = {}

            def add(p: float, ns: str) -> None:
                reward = 1.0 if ns == goal else 0.0
                probs[(reward, ns)] = probs.get((reward, ns), 0.0) + p

            add(1.0 - slip, move(state, action))
            for perp in perpendiculars[action]:
                add(slip / 2.0, move(state, perp))
            transitions[(state, action)] = [
                (p, r, ns) for (r, ns), p in probs.items() if p > 0.0
            ]
    return FiniteMDP(
        states,
        list(ACTIONS),
        transitions,
        gamma=gamma,
        terminal_states=[goal],
        seed=seed,
    )


class TabularModel:
    """Counting tabular model that stores outcome distributions (book Sec 8.1).

    Book Section 8.1: a sample model produces one outcome; a distribution
    model produces all outcomes with their probabilities. The Dyna-Q model
    in the book is a sample model assuming determinism (book Sec 8.2); this
    class generalizes it to stochastic environments by keeping per-outcome
    counts for every observed (state, action) pair, so the model can be
    queried either as a sample model (`sample`) or as a distribution model
    (`distribution`). It validates P(outcome | s, a) by normalizing counts,
    which avoids the "sample average collapses to a single deterministic
    outcome" failure mode discussed in Exercise 9.3.
    """

    def __init__(self) -> None:
        self._outcome_counts: dict = {}
        self._visited: dict = {}

    def observe(self, state, action, reward: float, next_state) -> None:
        """Record one observed transition (model-learning step, book Sec 8.2 (e))."""
        key = (state, action)
        outcomes = self._outcome_counts.setdefault(key, {})
        outcome = (float(reward), next_state)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        self._visited.setdefault(state, set()).add(action)

    def is_known(self, state, action) -> bool:
        """Return True if (state, action) has been observed at least once."""
        return (state, action) in self._outcome_counts

    def known_pairs(self) -> list:
        """Return all (state, action) pairs observed so far, sorted for determinism."""
        return sorted(self._outcome_counts, key=repr)

    def visited_states(self) -> list:
        """Return all states observed at least once, sorted for determinism."""
        return sorted(self._visited, key=repr)

    def actions_taken(self, state) -> list:
        """Return all actions taken in `state`, sorted for determinism."""
        return sorted(self._visited[state], key=repr)

    def has_pairs(self) -> bool:
        """Return True if at least one (state, action) pair is known."""
        return bool(self._outcome_counts)

    def sample_known_pair(self, rng: np.random.Generator):
        """Sample a random previously-observed pair exactly as Tabular Dyna-Q does.

        Book Section 8.2, step (f): pick a random previously observed state,
        then a random action previously taken in that state.
        """
        visited = self.visited_states()
        state = visited[int(rng.integers(len(visited)))]
        taken = self.actions_taken(state)
        action = taken[int(rng.integers(len(taken)))]
        return state, action

    def distribution(self, state, action) -> list:
        """Return the learned outcome distribution [(prob, reward, next_state)].

        Raises KeyError if the pair has never been observed.
        """
        outcomes = self._outcome_counts[(state, action)]
        total = sum(outcomes.values())
        return [
            (count / total, reward, next_state)
            for (reward, next_state), count in sorted(outcomes.items(), key=repr)
        ]

    def sample(self, state, action, rng: np.random.Generator):
        """Sample one outcome (reward, next_state) ~ learned distribution."""
        dist = self.distribution(state, action)
        probs = np.array([p for p, _, _ in dist], dtype=np.float64)
        idx = int(rng.choice(len(dist), p=probs))
        return dist[idx][1], dist[idx][2]

    def observation_count(self, state, action) -> int:
        """Return the number of times (state, action) was observed."""
        return sum(self._outcome_counts[(state, action)].values())

    def predecessors(self, next_state) -> list:
        """Return all known (state, action) pairs with any outcome into `next_state`."""
        result = []
        for (state, action), outcomes in self._outcome_counts.items():
            if any(ns == next_state for (_, ns) in outcomes):
                result.append((state, action))
        return sorted(result, key=repr)


def expected_q_target(
    model: TabularModel,
    q_table: dict,
    state,
    action,
    gamma: float,
    actions: list,
    terminal_states: set,
) -> float:
    """Expected update target for (state, action) under the learned model.

    Book Section 8.5, eq. (8.1): sum over s', r of p_hat(s', r | s, a) *
    [r + gamma * max_a' Q(s', a')].
    """
    total = 0.0
    for prob, reward, next_state in model.distribution(state, action):
        total += prob * (reward + gamma * _max_q(q_table, next_state, actions, terminal_states))
    return total


def _init_q(env: FiniteMDP, q_init: float = 0.0) -> dict:
    return {(s, a): q_init for s in env.states for a in env.actions}


def _dyna_loop(
    env: FiniteMDP,
    planner: str,
    n_planning: int,
    n_episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    seed: int | None,
    max_steps: int,
    start_state,
):
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    terminal_states = set(env.terminal_states)
    q_table = _init_q(env)
    model = TabularModel()
    stats = {"episode_steps": [], "episode_rewards": [], "planning_updates": 0}
    for _ in range(n_episodes):
        state = env.reset(start_state) if start_state is not None else env.reset()
        done = env.done
        steps = 0
        episode_reward = 0.0
        while not done and steps < max_steps:
            action = _epsilon_greedy_action(q_table, state, actions, epsilon, rng)
            reward, next_state, done = env.step(action)
            # (d) direct RL: one-step tabular Q-learning
            _q_learning_update(q_table, state, action, reward, next_state,
                               alpha, gamma, actions, terminal_states)
            # (e) model-learning
            model.observe(state, action, reward, next_state)
            # (f) planning: n iterations of one-step tabular Q-planning
            for _ in range(n_planning):
                if not model.has_pairs():
                    break
                ps, pa = model.sample_known_pair(rng)
                if planner == "sample":
                    sim_reward, sim_next = model.sample(ps, pa, rng)
                    _q_learning_update(q_table, ps, pa, sim_reward, sim_next,
                                       alpha, gamma, actions, terminal_states)
                else:  # "expected": book eq. (8.1), direct assignment
                    q_table[(ps, pa)] = expected_q_target(
                        model, q_table, ps, pa, gamma, actions, terminal_states
                    )
                stats["planning_updates"] += 1
            state = next_state
            steps += 1
            episode_reward += reward
        stats["episode_steps"].append(steps)
        stats["episode_rewards"].append(episode_reward)
    return q_table, model, stats


def dyna_q(
    env: FiniteMDP,
    n_planning: int = 5,
    n_episodes: int = 50,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
) -> tuple[dict, TabularModel, dict]:
    """Tabular Dyna-Q (Algorithm 9.1 in the repo; book Section 8.2).

    Book pseudocode "Tabular Dyna-Q", steps (a)-(f): act with an
    epsilon-greedy policy, apply a direct Q-learning update, record the
    transition in the model, then repeat `n_planning` times: sample a random
    previously observed state and a random action taken there, query the
    model, and apply a Q-learning update to the simulated experience.

    Args:
        env: The finite MDP (environment) to interact with.
        n_planning: Planning steps n performed per real step (book step (f)).
        n_episodes: Number of episodes to run.
        alpha: Step size for Q-learning updates.
        epsilon: Exploration rate for the epsilon-greedy behavior policy.
        gamma: Discount factor; defaults to env.gamma.
        seed: Seed for reproducible sampling.
        max_steps: Safety cap on steps per episode.
        start_state: Fixed initial state (e.g. the gridworld start).

    Returns:
        (q_table, model, stats): q_table maps (state, action) -> Q(s, a);
        model is the learned TabularModel; stats holds episode lengths,
        episode rewards and the total number of planning updates performed.
    """
    gamma = env.gamma if gamma is None else gamma
    return _dyna_loop(env, "sample", n_planning, n_episodes, alpha, epsilon,
                      gamma, seed, max_steps, start_state)


class PriorityQueue:
    """Max-priority queue with lazy deletion of stale entries (book Sec 8.4).

    Used by prioritized sweeping to hold state-action pairs whose estimated
    values would change nontrivially if updated. Re-inserting a pair that is
    already queued promotes only its highest priority (the previous entry is
    discarded lazily on pop). This matches the book's description: "if there
    is a previous entry of the pair in the queue, then insertion results in
    only the higher priority entry remaining in the queue".
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._counter = itertools.count()
        self._best: dict = {}

    def push(self, pair, priority: float) -> None:
        """Insert `pair` with `priority`; stale lower-priority entries are dropped lazily."""
        if priority <= 0.0:
            return
        self._best[pair] = priority
        heapq.heappush(self._heap, (-priority, next(self._counter), pair))

    def pop(self):
        """Pop the highest-priority (pair, priority), or return None if empty."""
        while self._heap:
            neg_prio, _, pair = heapq.heappop(self._heap)
            if self._best.get(pair) == -neg_prio:
                del self._best[pair]
                return pair, -neg_prio
        return None

    def __len__(self) -> int:
        """Return the number of queued (non-stale) entries."""
        return len(self._best)

    def empty(self) -> bool:
        """Return True if no (non-stale) entries remain."""
        return not self._best


def prioritized_sweeping(
    env: FiniteMDP,
    theta: float = 1e-3,
    n_planning: int = 5,
    n_episodes: int = 50,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
) -> tuple[dict, TabularModel, dict]:
    """Prioritized sweeping (Exercise 9.1; book Section 8.4).

    Book pseudocode "Prioritized sweeping for a deterministic environment",
    steps (a)-(g), extended to stochastic environments with expected updates
    as described in the prose after the box: the model stores outcome counts
    (see TabularModel) and every update, both for queued pairs and for the
    priority computation of predecessors, uses the expected update of eq.
    (8.1); for a deterministic model this reduces exactly to the book's
    sample update.

    Args:
        env: The finite MDP to interact with.
        theta: Threshold theta; pairs whose update would change Q by <= theta
            are not queued.
        n_planning: Planning updates performed per real step (book step (g)).
        n_episodes: Number of episodes to run.
        alpha: Step size for Q-learning updates.
        epsilon: Exploration rate for the epsilon-greedy behavior policy.
        gamma: Discount factor; defaults to env.gamma.
        seed: Seed for reproducible sampling.
        max_steps: Safety cap on steps per episode.
        start_state: Fixed initial state (e.g. the gridworld start).

    Returns:
        (q_table, model, stats) with the same layout as dyna_q plus
        stats['queue_pushes'] and stats['updates'].
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    gamma = env.gamma if gamma is None else gamma
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    terminal_states = set(env.terminal_states)
    q_table = _init_q(env)
    model = TabularModel()
    queue = PriorityQueue()
    stats = {
        "episode_steps": [],
        "episode_rewards": [],
        "queue_pushes": 0,
        "updates": 0,
        "planning_updates": 0,
    }
    for _ in range(n_episodes):
        state = env.reset(start_state) if start_state is not None else env.reset()
        done = env.done
        steps = 0
        episode_reward = 0.0
        while not done and steps < max_steps:
            action = _epsilon_greedy_action(q_table, state, actions, epsilon, rng)
            reward, next_state, done = env.step(action)
            # (d) model(S, A) <- R, S'  (distribution model records the counts)
            model.observe(state, action, reward, next_state)
            # (e)/(f): queue the real transition if its update is nontrivial
            target = reward + gamma * _max_q(q_table, next_state, actions, terminal_states)
            priority = abs(target - q_table[(state, action)])
            if priority > theta:
                queue.push((state, action), priority)
                stats["queue_pushes"] += 1
            # (g): propagate backward in priority order
            budget = n_planning
            while budget > 0 and not queue.empty():
                pair, _ = queue.pop()
                ps, pa = pair
                new_q = expected_q_target(model, q_table, ps, pa, gamma,
                                          actions, terminal_states)
                q_table[pair] += alpha * (new_q - q_table[pair])
                stats["updates"] += 1
                stats["planning_updates"] += 1
                budget -= 1
                for pred_pair in model.predecessors(ps):
                    pred_target = expected_q_target(model, q_table, pred_pair[0],
                                                    pred_pair[1], gamma,
                                                    actions, terminal_states)
                    pred_priority = abs(pred_target - q_table[pred_pair])
                    if pred_priority > theta:
                        queue.push(pred_pair, pred_priority)
                        stats["queue_pushes"] += 1
            state = next_state
            steps += 1
            episode_reward += reward
        stats["episode_steps"].append(steps)
        stats["episode_rewards"].append(episode_reward)
    return q_table, model, stats


def expected_sweep(
    model: TabularModel,
    q_table: dict,
    gamma: float,
    actions: list,
    terminal_states: set,
    sweeps: int = 1,
) -> float:
    """Full distribution sweeps over all known state-action pairs.

    Book Section 8.5 (expected updates, eq. (8.1)); performing the expected
    update at every known pair once per sweep is the small-MDP equivalent of
    the DP sweeps of Chapter 4. Returns the largest absolute change in any
    Q entry during the last sweep.
    """
    max_change = 0.0
    for _ in range(sweeps):
        max_change = 0.0
        for pair in model.known_pairs():
            state, action = pair
            if state in terminal_states:
                continue
            new_q = expected_q_target(model, q_table, state, action, gamma,
                                      actions, terminal_states)
            max_change = max(max_change, abs(new_q - q_table[pair]))
            q_table[pair] = new_q
    return max_change


def expected_dyna_q(
    env: FiniteMDP,
    n_planning: int = 5,
    n_episodes: int = 50,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
) -> tuple[dict, TabularModel, dict]:
    """Dyna-Q with expected model updates (Exercise 9.2, book Section 8.5).

    Identical to dyna_q, except the planning step applies the expected update
    of eq. (8.1) using the learned outcome distribution instead of a sample
    update, and assigns the target directly (no step-size blending), matching
    the exact computation described in the book. Use expected_sweep for full
    distribution sweeps over all known pairs (the deterministic-style update).
    """
    gamma = env.gamma if gamma is None else gamma
    return _dyna_loop(env, "expected", n_planning, n_episodes, alpha, epsilon,
                      gamma, seed, max_steps, start_state)


def optimal_q_values(
    mdp: FiniteMDP, tolerance: float = 1e-10, max_iterations: int = 10000
) -> dict:
    """Compute the optimal action values q* of a finite MDP exactly.

    Value iteration on v* with ch03's bellman_backup_optimality, then
    q*(s, a) = E[R + gamma * v*(s')]. Terminal state-action pairs map to 0.0.
    """
    values = np.zeros(mdp.n_states, dtype=np.float64)
    for _ in range(max_iterations):
        updated = bellman_backup_optimality(mdp, values)
        if np.max(np.abs(updated - values)) < tolerance:
            values = updated
            break
        values = updated
    q_star: dict = {}
    for s in mdp.states:
        for a in mdp.actions:
            if s in mdp.terminal_states:
                q_star[(s, a)] = 0.0
                continue
            total = mdp.expected_reward(s, a)
            for p, _, ns in mdp.transition_probs(s, a):
                total += mdp.gamma * p * values[mdp.state_index[ns]]
            q_star[(s, a)] = float(total)
    return q_star


def compare_sample_vs_expected_dyna(
    env_factory,
    n_planning: int = 10,
    n_episodes: int = 30,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float | None = None,
    seed: int | None = None,
    start_state: str | None = None,
) -> dict:
    """Compare sample-model Dyna vs expected-update Dyna on a stochastic MDP.

    Exercise 9.2 (book Section 8.5). Builds two independent environments via
    `env_factory(seed)`, trains one Dyna-Q agent (sample updates) and one
    expected-update Dyna agent, and reports each agent's max absolute error
    against q* (computed exactly over the true MDP) after training.

    Returns:
        dict with keys 'sample', 'expected' and 'q_star_max'. The 'sample'
        and 'expected' sub-dicts carry 'max_q_error', 'episode_steps' and
        'episode_rewards'.
    """
    q_star = optimal_q_values(env_factory(seed))

    def _run(kind):
        env = env_factory(seed)
        runner = dyna_q if kind == "sample" else expected_dyna_q
        q_table, model, stats = runner(
            env,
            n_planning=n_planning,
            n_episodes=n_episodes,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            seed=seed,
            start_state=start_state,
        )
        known = model.known_pairs()
        error = max(abs(q_table[pair] - q_star[pair]) for pair in known)
        known_error = float(error)
        all_error = max(abs(q_table[pair] - q_star[pair]) for pair in q_star)
        return {
            "max_q_error": float(all_error),
            "known_max_q_error": known_error,
            "episode_steps": stats["episode_steps"],
            "episode_rewards": stats["episode_rewards"],
            "q_table": q_table,
        }

    return {
        "sample": _run("sample"),
        "expected": _run("expected"),
        "q_star_max": float(max(abs(v) for v in q_star.values())),
    }
