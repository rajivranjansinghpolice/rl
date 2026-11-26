"""Solution implementation for Chapter 8 - Planning and Learning with Tabular Methods.

Reference: Sutton & Barto (2nd Ed), Chapter 8

Implements: the Tabular Dyna-Q box (Sec. 8.2, Example 8.1, Dyna maze) as `dyna_q`,
Dyna-Q+ with the exploration bonus (Sec. 8.3, Examples 8.2 and 8.3) as `dyna_q_plus`,
Prioritized sweeping (Sec. 8.4, Moore & Atkeson 1993) as `prioritized_sweeping`,
expected vs sample updates / Dyna-Q with expected updates (Sec. 8.5) as
`expected_dyna_q`, Expectation vs sample Dyna (Sec. 8.5 sample-vs-expected analysis is 
`compare_sample_vs_expected_dyna`), and the TabularModel world model of
Sec. 8.1 (distribution model with per-outcome counts).
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
    """One one-step tabular Q-learning backup (book Sec. 6.2 / eq. 8.2)."""
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
    walls: list[str] | None = None,
) -> FiniteMDP:
    """Small gridworld used for the Dyna experiments (adapted from Example 8.1).

    Book Section 8.2 (Example 8.1, Dyna maze). States are 'r,c' strings.
    Actions are up/down/left/right; moving into a wall or off the grid keeps
    the agent in place. Reward is +1 on entering the goal (terminal) and 0
    elsewhere. With `slip > 0`, the intended move happens with probability
    ``1 - slip`` and the two perpendicular moves share the remaining
    probability equally (a stochastic analogue of the book's maze); the
    deterministic world is recovered with ``slip = 0``.

    With ``walls`` given ('r,c' strings), those cells are removed from the
    state set and act as impassable barriers (the agent stays in place when
    attempting to enter one), matching the Dyna maze of Figure 8.3.
    """
    if not 0.0 <= slip < 1.0:
        raise ValueError("slip must be in [0, 1)")
    walls = list(walls) if walls else []
    states = [f"{r},{c}" for r in range(rows) for c in range(cols) if f"{r},{c}" not in set(walls)]
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

    wall_set = set(walls)

    def move(state: str, action: str) -> str:
        r, c = (int(x) for x in state.split(","))
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and f"{nr},{nc}" not in wall_set:
            return f"{nr},{nc}"
        return state  # blocked by wall or boundary: stay in place

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
    outcome" failure mode discussed in Sec. 8.1/8.3.
    """

    def __init__(self) -> None:
        self._outcome_counts: dict = {}
        self._visited: dict = {}
        # Incremental predecessor index (book Sec. 8.4): scanning every known
        # pair on each pop is the main cost of prioritized sweeping.
        self._predecessors: dict = {}

    def observe(self, state, action, reward: float, next_state) -> None:
        """Record one observed transition (model-learning step, book Sec 8.2 (e))."""
        key = (state, action)
        outcomes = self._outcome_counts.setdefault(key, {})
        outcome = (float(reward), next_state)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        self._visited.setdefault(state, set()).add(action)
        self._predecessors.setdefault(next_state, set()).add(key)

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
        return sorted(self._predecessors.get(next_state, ()), key=repr)


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
    kappa: float | None = None,
    step_callback=None,
):
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if kappa is not None and kappa < 0.0:
        raise ValueError("kappa must be >= 0 when set")
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    terminal_states = set(env.terminal_states)
    q_table = _init_q(env)
    model = TabularModel()
    last_tried: dict = {}
    visited_states: list = []
    visited_set: set = set()
    time_steps = 0
    stats = {"episode_steps": [], "episode_rewards": [], "planning_updates": 0}
    for _loop_episode in range(n_episodes):
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
            if state not in visited_set:
                visited_set.add(state)
                visited_states.append(state)
            last_tried[(state, action)] = time_steps
            time_steps += 1
            # (f) planning: n iterations of one-step tabular Q-planning
            pops = 0
            for _ in range(n_planning):
                if not model.has_pairs():
                    break
                if kappa is None:
                    ps, pa = model.sample_known_pair(rng)
                    untried = False
                else:
                    # Sec. 8.3 / Dyna-Q+: also consider never-tried pairs with the
                    # default model (zero reward, self-loop); stale modelled
                    # pairs get the kappa*sqrt(tau) bonus.
                    # Sampling a uniform visited state and a uniform action is
                    # equivalent to sampling uniformly from all (s, a) pairs,
                    # without materialising the (potentially large) pair list.
                    ps = visited_states[int(rng.integers(len(visited_states)))]
                    pa = actions[int(rng.integers(len(actions)))]
                    untried = not model.is_known(ps, pa)
                if untried:
                    sim_reward, sim_next = 0.0, ps
                elif planner == "sample":
                    sim_reward, sim_next = model.sample(ps, pa, rng)
                else:  # "expected": book eq. (8.1), direct assignment
                    sim_reward, sim_next = None, None
                # Never-tried pairs use the default (zero-reward, self-loop)
                # model with *no* bonus: the bonus in the book is defined for
                # transitions whose model exists but has gone stale. Giving a
                # large sqrt(tau) bonus to unmodelled self-loops makes every
                # frontier action look better than reaching the goal.
                bonus = (
                    kappa * (time_steps - last_tried.get((ps, pa), 0)) ** 0.5
                    if (kappa is not None and not untried)
                    else 0.0
                )
                if planner == "sample" or untried:
                    _q_learning_update(q_table, ps, pa,
                                       (sim_reward if sim_reward is not None else 0.0) + bonus,
                                       sim_next, alpha, gamma, actions, terminal_states)
                else:
                    target = expected_q_target(model, q_table, ps, pa, gamma,
                                               actions, terminal_states)
                    # Sec. 8.3 bonus: add to the modeled reward component
                    # (exact for the deterministic mazes of Examples 8.2/8.3;
                    # stochastic models with a reward bonus are not a book algorithm)
                    target += bonus
                    q_table[(ps, pa)] = target
                stats["planning_updates"] += 1
                pops += 1
            state = next_state
            steps += 1
            episode_reward += reward
            if step_callback is not None:
                step_callback(_loop_episode, steps, state, q_table, pops)
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
    step_callback=None,
) -> tuple[dict, TabularModel, dict]:
    """Tabular Dyna-Q (Tabular Dyna-Q; book Section 8.2).

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
                      gamma, seed, max_steps, start_state, step_callback=step_callback)


def dyna_q_plus(
    env: FiniteMDP,
    n_planning: int = 5,
    n_episodes: int = 50,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float | None = None,
    seed: int | None = None,
    max_steps: int = 1000,
    start_state: str | None = None,
    kappa: float = 1e-4,
    step_callback=None,
) -> tuple[dict, TabularModel, dict]:
    """Dyna-Q+ (Sec. 8.3, exploration bonus; Examples 8.2 and 8.3).

    Identical to tabular Dyna-Q, except that planning updates use a
    "bonus reward" for simulated transitions whose model has gone stale: if
    the modeled reward is r and the pair has not been tried in tau real time
    steps, the simulated transition produces r + kappa * sqrt(tau).
    Never-tried (state, action) pairs in visited states are also considered
    in the planning step, with the default model (zero reward, self-loop),
    as described in the footnote to the book's Sec. 8.3 text; those pairs
    receive no bonus because their model does not yet exist.

    Args:
        env: The finite MDP to interact with.
        n_planning: Planning steps per real step.
        n_episodes: Number of episodes.
        alpha: Step size.
        epsilon: Exploration rate of the epsilon-greedy behavior policy.
        gamma: Discount factor; defaults to env.gamma.
        seed: Seed for reproducible sampling.
        max_steps: Safety cap on steps per episode.
        start_state: Fixed initial state.
        kappa: Bonus coefficient kappa (book Sec. 8.3; small values,
            e.g. 1e-3, are used).

    Returns:
        (q_table, model, stats), same layout as dyna_q.
    """
    if kappa < 0.0:
        raise ValueError("kappa must be >= 0")
    gamma = env.gamma if gamma is None else gamma
    return _dyna_loop(env, "sample", n_planning, n_episodes, alpha, epsilon,
                      gamma, seed, max_steps, start_state, kappa=kappa,
                      step_callback=step_callback)


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
    step_callback=None,
) -> tuple[dict, TabularModel, dict]:
    """Prioritized sweeping (book Section 8.4).

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
        "updates_per_step": [],
    }
    for _loop_episode in range(n_episodes):
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
            pops = 0
            while budget > 0 and not queue.empty():
                pair, _ = queue.pop()
                ps, pa = pair
                new_q = expected_q_target(model, q_table, ps, pa, gamma,
                                          actions, terminal_states)
                q_table[pair] += alpha * (new_q - q_table[pair])
                stats["updates"] += 1
                stats["planning_updates"] += 1
                budget -= 1
                pops += 1
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
            stats["updates_per_step"].append(pops)
            if step_callback is not None:
                step_callback(_loop_episode, steps, state, q_table, pops)
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
    """Dyna-Q with expected model updates (book Section 8.5).

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

    book Section 8.5. Builds two independent environments via
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


# ---------------------------------------------------------------------------
# Sec. 8.3: changing environments (Examples 8.2 blocking, Example 8.3 shortcut)
# ---------------------------------------------------------------------------

def _epsilon_greedy_action_with_bonus(q_table, state, actions, epsilon, rng,
                                      kappa, time_steps, last_tried):
    """Exercise 8.4: bonus used only in action selection, never in updates.

    The executed action maximises Q(s, a) + kappa * sqrt(tau(s, a)); Q itself
    stays an unbiased estimate of the return under the modelled rewards.
    """
    if rng.random() < epsilon:
        return actions[int(rng.integers(len(actions)))]
    bonuses = np.array([
        kappa * np.sqrt(max(0, time_steps - last_tried.get((state, a), 0)))
        for a in actions
    ], dtype=np.float64)
    values = np.array([q_table[(state, a)] for a in actions], dtype=np.float64) + bonuses
    best = np.flatnonzero(values == values.max())
    return actions[int(rng.choice(best))]


class ChangingGridworld:
    """Gridworld whose active barrier layout changes mid-experiment.

    Book Sec. 8.3, Examples 8.2/8.3: one optimal route is available at the
    start; after `change_step` real time steps a different route is opened
    (and, optionally, the original one is blocked). Implemented as a wrapper
    around two gridworld MDPs over the same state set; the internal MDP is
    swapped when the clock reaches `change_step`. A caller's agent keeps its
    Q-table and TabularModel across the swap — the model is stale for the
    swapped cells, which is exactly the situation the book studies.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        start: str,
        goal: str,
        base_walls: list[str],
        alt_walls: list[str],
        change_step: int,
        gamma: float = 0.95,
        seed: int | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start_state = start
        self.goal = goal
        self._gamma = float(gamma)
        self._base_walls = list(base_walls)
        self._alt_walls = list(alt_walls)
        self._change_step = int(change_step)
        self.steps = 0
        self._current_state = start
        self._env = make_gridworld(rows=rows, cols=cols, start=start, goal=goal,
                                   gamma=gamma, seed=seed, walls=self._base_walls)

    @property
    def actions(self):
        return self._env.actions

    @property
    def terminal_states(self):
        return self._env.terminal_states

    @property
    def gamma(self):
        return self._env.gamma

    @property
    def done(self):
        return self._env.done

    def reset(self, start_state: str | None = None) -> str:
        state = self._env.reset(start_state) if start_state is not None else self._env.reset()
        self._current_state = state
        return state

    def step(self, action: str):
        self.steps += 1
        if self.steps == self._change_step + 1:
            self._swap()
        reward, next_state, done = self._env.step(action)
        self._current_state = next_state
        return reward, next_state, done

    def _swap(self) -> None:
        self._env = make_gridworld(rows=self.rows, cols=self.cols, start=self.start_state,
                                   goal=self.goal, gamma=self._gamma, seed=None,
                                   walls=self._alt_walls)
        self._env.reset(self._current_state)


def _grid_state_strings(rows: int, cols: int, walls: list[str]) -> list:
    wall_set = set(walls)
    return [f"{r},{c}" for r in range(rows) for c in range(cols) if f"{r},{c}" not in wall_set]


def changing_maze_layouts():
    """Book layouts for Sec. 8.3 (reconstruction; the draft pages do not print cell coordinates).

    Returns (start, goal, base_walls, blocking_walls, shortcut_walls).
    Grid: 6 rows x 10 cols, barrier along row 2, S bottom center, G top right.
    """
    start, goal = "5,4", "0,9"
    # blocking maze (Example 8.2): gap on the right initially; after the change,
    # the right passage is walled and the left passage (col 0) is open.
    barrier = [f"2,{c}" for c in range(10) if c != 9]
    blocking_alt = [f"2,{c}" for c in range(10) if c != 0]
    # shortcut maze (Example 8.3): gap on the left initially; after the change,
    # a right-side shortcut (col 9) is ALSO opened; the left gap stays open.
    shortcut_base = [f"2,{c}" for c in range(10) if c != 0]
    shortcut_alt = [f"2,{c}" for c in range(10) if c not in (0, 9)]
    return start, goal, barrier, blocking_alt, shortcut_base, shortcut_alt


def changing_maze_experiment(
    layout: str,
    agent: str,
    change_step: int,
    n_episodes: int = 1000,
    n_planning: int = 5,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    kappa: float = 1e-4,
    seed: int = 0,
    max_steps: int = 300,
    steps_cap: int | None = None,
) -> tuple[list[float], list[int]]:
    """Run one Sec. 8.3 changing-maze experiment (Figures 8.5/8.6).

    `layout` is 'blocking' (Fig. 8.5) or 'shortcut' (Fig. 8.6); `agent` is
    'dyna_q', 'dyna_q_plus' or 'selection_bonus' (Ex. 8.4: bonus at action
    selection only, with unchanged updates). Episodes use ChangingGridworld so
    that the barrier swap happens after `change_step` real time steps while the
    agent's Q-table and model persist. Cumulative reward is recorded after every
    time step.

    Returns (cumulative_reward, episode_rewards).
    """
    if layout not in ("blocking", "shortcut"):
        raise ValueError("layout must be 'blocking' or 'shortcut'")
    if agent not in ("dyna_q", "dyna_q_plus", "selection_bonus"):
        raise ValueError(
            "agent must be 'dyna_q', 'dyna_q_plus' or 'selection_bonus'")
    start, goal, barrier, blocking_alt, shortcut_base, shortcut_alt = changing_maze_layouts()
    if layout == "blocking":
        base_walls, alt_walls = barrier, blocking_alt
    else:
        base_walls, alt_walls = shortcut_base, shortcut_alt
    env = ChangingGridworld(6, 10, start, goal, base_walls, alt_walls,
                            change_step=change_step, gamma=gamma, seed=seed)
    rng = np.random.default_rng(seed)
    # Q-table covers the union of both barrier phases (a cell that becomes
    # walkable after the swap must have actions)
    walkable = set(_grid_state_strings(env.rows, env.cols, base_walls)) | \
        set(_grid_state_strings(env.rows, env.cols, alt_walls))
    q_table = {(s, a): 0.0 for s in walkable for a in ACTIONS}
    model = TabularModel()
    actions = list(env.actions)
    terminal_states = set(env.terminal_states)
    last_tried: dict = {}
    visited_states: list = []
    visited_set: set = set()
    time_steps = 0
    cumulative_reward = [0.0]
    episode_rewards = []
    total_steps = 0
    for _loop_episode in range(n_episodes):
        if steps_cap is not None and time_steps >= steps_cap:
            break
        state = env.reset(start)
        done = env.done
        steps = 0
        episode_reward = 0.0
        while not done and steps < max_steps:
            if steps_cap is not None and time_steps >= steps_cap:
                break
            if agent == "selection_bonus":
                action = _epsilon_greedy_action_with_bonus(
                    q_table, state, actions, epsilon, rng, kappa,
                    time_steps, last_tried)
            else:
                action = _epsilon_greedy_action(
                    q_table, state, actions, epsilon, rng)
            reward, next_state, done = env.step(action)
            _q_learning_update(q_table, state, action, reward, next_state,
                               alpha, gamma, actions, terminal_states)
            model.observe(state, action, reward, next_state)
            if state not in visited_set:
                visited_set.add(state)
                visited_states.append(state)
            last_tried[(state, action)] = time_steps
            time_steps += 1
            episode_reward += reward
            total_steps += 1
            cumulative_reward.append(cumulative_reward[-1] + reward)
            # (f) planning: identical to _dyna_loop
            for _ in range(n_planning):
                if not model.has_pairs():
                    break
                if agent in ("dyna_q", "selection_bonus"):
                    ps, pa = model.sample_known_pair(rng)
                    sim_reward, sim_next = model.sample(ps, pa, rng)
                    bonus = 0.0
                else:  # dyna_q_plus
                    ps = visited_states[int(rng.integers(len(visited_states)))]
                    pa = actions[int(rng.integers(len(actions)))]
                    if not model.is_known(ps, pa):
                        sim_reward, sim_next = 0.0, ps
                    else:
                        sim_reward, sim_next = model.sample(ps, pa, rng)
                    tried = model.is_known(ps, pa)
                    bonus = (
                        kappa * (time_steps - last_tried.get((ps, pa), 0)) ** 0.5
                        if tried else 0.0
                    )
                _q_learning_update(q_table, ps, pa, sim_reward + bonus, sim_next,
                                   alpha, gamma, actions, terminal_states)
            state = next_state
            steps += 1
        if steps == 0:
            break
        episode_rewards.append(episode_reward)
    return cumulative_reward, episode_rewards


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 8)
# ---------------------------------------------------------------------------

def plot_figure_8_3_dyna_maze(
    save_path: str | None = None,
    show: bool = False,
    planning_steps: tuple[int, ...] = (0, 5, 50),
    n_episodes: int = 50,
    runs: int = 30,
    seed: int = 0,
):
    """Dyna maze learning curves (Sec. 8.2, Example 8.1 / Fig. 8.3).

    The exact 47-state maze of the book (6 rows x 9 cols; S=(2,0), G=(0,8);
    barriers at (1-3,2), (0-2,7) and (4,5)). Reward is +1 entering the goal,
    gamma = 0.95, alpha = 0.1, epsilon = 0.1 (book settings). Curves show steps
    per episode for episodes 2..n_episodes (the first episode, ~1700 steps,
    is identical for all n because the same initial seed is held constant
    across algorithms, and is not shown). Averaged over `runs` repetitions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    book_walls = ["1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"]
    curves = {n: np.zeros(n_episodes, dtype=np.float64) for n in planning_steps}
    for n in planning_steps:
        for r in range(runs):
            rs = seed + r * 1009
            env = make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                                 gamma=0.95, seed=rs, walls=book_walls)
            _, _, stats = dyna_q(
                env,
                n_planning=n,
                n_episodes=n_episodes,
                alpha=0.1,
                epsilon=0.1,
                gamma=0.95,
                seed=rs,
                max_steps=10000,
                start_state="2,0",
            )
            curves[n] += np.array(stats["episode_steps"], dtype=np.float64)
        curves[n] /= runs

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n in planning_steps:
        ax.plot(np.arange(2, n_episodes + 1), curves[n][1:], label=f"n={n} planning steps")
    ax.set_xlim(2, n_episodes)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per episode")
    ax.set_title("S&B (2nd ed.), Sec. 8.2, Example 8.1 / Fig. 8.3:\nDyna maze learning curves")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax



def _episode2_mid_snapshot(n_planning: int = 50, seed: int = 0):
    """Replay the first two Dyna maze episodes and capture state/Q at the
    midpoint of episode 2 (book, Fig. 8.4)."""
    book_walls = ["1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"]
    env = make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                         gamma=0.95, seed=seed, walls=book_walls)
    log = []

    def cb(ep, steps, state, q_table, pops):
        if ep == 1:  # second episode
            log.append((steps, state, dict(q_table)))

    dyna_q(env, n_planning=n_planning, n_episodes=2, alpha=0.1, epsilon=0.1,
           gamma=0.95, seed=seed, max_steps=10000, start_state="2,0",
           step_callback=cb)
    if not log:
        raise RuntimeError("episode 2 produced no steps")
    idx = len(log) // 2
    return {"steps": log[idx][0], "state": log[idx][1],
            "q_table": log[idx][2], "episode_len": len(log)}


def _draw_policy_panel(ax, q_table, agent_state, title):
    import matplotlib.pyplot as plt

    rows, cols = 6, 9
    book_walls = {"1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"}
    for wall in book_walls:
        r, c = (int(x) for x in wall.split(","))
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor="#4d4d4d", edgecolor="black"))
    ax.text(0, 2, "S", ha="center", va="center", fontsize=13,
            fontweight="bold", color="white")
    ax.text(8, 0, "G", ha="center", va="center", fontsize=13,
            fontweight="bold", color="#0b7a0b")
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    actions = ["up", "down", "left", "right"]
    for r in range(rows):
        for c in range(cols):
            cell = f"{r},{c}"
            if cell in book_walls or cell == "0,8":
                continue
            vals = [q_table.get((cell, a), 0.0) for a in actions]
            best = max(vals)
            if vals.count(best) > 1:
                continue  # no arrow when action values are tied (book rule)
            a = actions[vals.index(best)]
            dr, dc = deltas[a]
            ax.arrow(c, r, dc * 0.32, dr * 0.32, width=0.03,
                     head_width=0.11, head_length=0.08, color="black",
                     length_includes_head=True)
    ar, ac = (int(x) for x in agent_state.split(","))
    ax.add_patch(plt.Rectangle((ac - 0.22, ar - 0.22), 0.44, 0.44,
                               facecolor="black", edgecolor="white", linewidth=1.5))
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(rows - 0.4, -0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)



def plot_figure_8_4_policies(save_path: str | None = None, show: bool = False,
                             seed: int = 0):
    """Policies found by the planning and nonplanning Dyna-Q agents halfway
    through the second episode (Sec. 8.2, Example 8.1, Fig. 8.4).

    Seed held constant across the two agents, as in the book. Arrows indicate
    the greedy action in each state; states whose action values are all equal
    (e.g. everything unvisited, or tied) get no arrow. The black square marks
    the agent's location at the snapshot (midpoint of episode 2).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = {
        "n=0": _episode2_mid_snapshot(n_planning=0, seed=seed),
        "n=50": _episode2_mid_snapshot(n_planning=50, seed=seed),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    _draw_policy_panel(axes[0], snaps["n=0"]["q_table"], snaps["n=0"]["state"],
                       "WITHOUT PLANNING (N=0)")
    _draw_policy_panel(axes[1], snaps["n=50"]["q_table"], snaps["n=50"]["state"],
                       "WITH PLANNING (N=50)")
    fig.suptitle(
        "S&B (2nd ed.), Sec. 8.2, Example 8.1 / Fig. 8.4:\n"
        "Policies halfway through the second episode (illustrative reproduction)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_changing_maze(layout: str, change_step: int, steps_cap: int,
                       save_path: str | None = None, show: bool = False,
                       runs: int = 20, n_episodes: int = 3000,
                       seed: int = 0,
                       agents: tuple[str, ...] = ("dyna_q", "dyna_q_plus")):
    """Changing-maze experiment (Sec. 8.3, Example 8.2 blocking / Fig. 8.5,
    Example 8.3 shortcut / Fig. 8.6).

    Both the regular Dyna-Q agent and the Dyna-Q+ agent (exploration bonus
    kappa = 1e-4) act on the same maze; after `change_step` real time steps
    the barrier layout changes while the agents' Q-tables and models persist
    (the model is stale). Curves are cumulative reward averaged over `runs`
    repetitions (the same change thresholds, steps caps and layout as the
    book: 1000/3000 for blocking, 3000/6000 for shortcut). Book y-axis upper
    limits (~150 blocking, ~400 shortcut) are reproduced only approximately.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {}
    for agent in agents:
        acc = np.zeros(steps_cap + 1, dtype=np.float64)
        for r in range(runs):
            rs = seed + r * 1009
            cum, _ = changing_maze_experiment(
                layout=layout, agent=agent, change_step=change_step,
                n_episodes=n_episodes, n_planning=5, alpha=0.1, epsilon=0.1,
                gamma=0.95, kappa=1e-4, seed=rs, max_steps=400,
                steps_cap=steps_cap)
            acc[:len(cum)] += np.array(cum, dtype=np.float64)
        curves[agent] = acc / runs

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(steps_cap + 1)
    styles = {"dyna_q": ("Dyna-Q", "-"), "dyna_q_plus": ("Dyna-Q+", "--"),
              "selection_bonus": ("Selection-only bonus (Ex. 8.4)", ":")}
    for agent in agents:
        label, style = styles[agent]
        ax.plot(x, curves[agent], label=label, linestyle=style)
    ax.axvline(change_step, color="red", linestyle=":", linewidth=1.2,
               label=f"layout changes at t={change_step}")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Cumulative reward")
    if layout == "blocking":
        ex, figno = "Example 8.2", "Fig. 8.5"
    else:
        ex, figno = "Example 8.3", "Fig. 8.6"
    ax.set_title(f"S&B (2nd ed.), Sec. 8.3, {ex} / {figno}: {layout} maze")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_8_8_update_efficiency(save_path: str | None = None, show: bool = False):
    """Expected vs sample update efficiency (Sec. 8.5, Fig. 8.8).

    Sample updates: with b equally likely successor outcomes, all with the
    estimate error initially 1 and the next-state values correct, the RMS
    error after t sample updates (sample averages, alpha = 1/t) is
    sqrt((b - 1) / (b * t)). Expected updates: one expected update reduces
    the error to 0, at a cost of b computations. x-axis: number of
    computations (log scale), y-axis: RMS error. Book x-axes show 0, 1, b,
    2b per curve; the log-scale shared axis here conveys the same comparison.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.logspace(0, 5, 400)
    for b in (2, 10, 100, 1000, 10000):
        ys = np.sqrt((b - 1.0) / (b * xs))
        ys = np.where(xs <= 2.0 * b, ys, np.nan)
        ax.plot(xs, ys, label=f"sample updates, b={b}")
        ax.plot([b, b], [0.0, 1.0], linestyle=":", linewidth=1, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("Number of computations")
    ax.set_ylabel("RMS error in value estimate")
    ax.set_ylim(0, 1.05)
    ax.set_title("S&B (2nd ed.), Sec. 8.5 / Fig. 8.8: efficiency of expected\n"
                 "and sample updates (dotted lines: cost b of one expected update)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_8_10_heuristic_search(save_path: str | None = None, show: bool = False):
    """Heuristic search as a sequence of one-step updates (Sec. 8.8, Fig. 8.10).

    Selective depth-first search update ordering for a 10-node tree: boxes
    are updated in post-order (a box is updated after its descendants),
    numbered 1 (first leaf to update) through 10 (the root). Only states on
    the search path receive updates; the ordering is what makes the
    backups informative so early. Simplified schematic redrawn for this repo;
    it is not a facsimile of the printed figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = [
        (0.0, 6.0, "10"),   # root
        (-2.0, 4.0, "7"),   # left child
        (2.0, 4.0, "9"),    # right child
        (-3.0, 2.0, "5"),   # left grandchild
        (-1.0, 2.0, "6"),   # right grandchild
        (2.0, 2.0, "8"),    # right leaf-child
        (-3.75, 0.0, "1"),  # leaves under left grandchild
        (-3.0, 0.0, "2"),
        (-2.25, 0.0, "3"),
        (-1.0, 0.0, "4"),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8), (4, 9)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                color="0.35", linewidth=1.4, zorder=1)
    for x, y, label in nodes:
        ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                   facecolor="#e8e8e8", edgecolor="black",
                                   linewidth=1.3, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                zorder=3)
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-1.1, 7.1)
    ax.axis("off")
    ax.set_title("S&B (2nd ed.), Sec. 8.8 / Fig. 8.10: heuristic search as a\n"
                 "sequence of one-step updates (update order = box number)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def _backup_panel(ax, row: str, kind: str, label: str):
    """Draw one backup diagram: simplified schematic of the 2nd-ed. Fig. 8.7."""
    import matplotlib.pyplot as plt

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    is_action_root = row.startswith("q")  # q-row: square (action) at left
    circle = lambda x, y, r=0.52, fc="white": ax.add_patch(
        plt.Circle((x, y), r, facecolor=fc, edgecolor="black", linewidth=1.4, zorder=3))
    square = lambda x, y, r=0.52, fc="black": ax.add_patch(
        plt.Rectangle((x - r, y - r), 2 * r, 2 * r, facecolor=fc,
                      edgecolor="black", linewidth=1.4, zorder=3))

    root = (1.1, 5.0)
    if is_action_root:
        square(*root, fc="white")  # square = action node
    else:
        circle(*root)  # circle = state node

    n_children = {"expected": 2, "sample": 1}[kind]
    if is_action_root:
        child_rows = [7.0]  # action root: single next state
    elif row[1] == "*":
        child_rows = [7.0, 5.0, 3.0]  # v*: all actions
    else:
        child_rows = [7.0, 3.0]  # v-pi: policy-weighted actions

    child_ax = 4.2
    node_axes = [root]
    for child_y in child_rows:
        edge_kw = dict(color="0.25", linewidth=1.3, zorder=1)
        ax.plot([root[0] + 0.55, child_ax - 0.55], [root[1], child_y], **edge_kw)
        if is_action_root:
            circle(child_ax, child_y)          # next state
            node_axes.append((child_ax, child_y))
        else:
            square(child_ax, child_y, fc="white")   # action
            node_axes.append((child_ax, child_y))
            leaf_rows = [child_y + 1.1, child_y - 1.1] if kind == "expected" else [child_y]
            for leaf_y in leaf_rows:
                ax.plot([child_ax + 0.55, 7.4 - 0.55], [child_y, leaf_y], **edge_kw)
                circle(7.4, leaf_y)

    ax.set_title(label, fontsize=9)


def plot_figure_8_7_backup_diagrams(save_path: str | None = None, show: bool = False):
    """Backup diagrams (end of Sec. 8.4, Fig. 8.7).

    Four update targets x two update types. Expected updates (left column)
    branch over all successors; sample updates (right column) follow one
    sampled successor. The eighth case (sample v*) does not correspond to any
    useful update in the book's classification and is marked as such.

    Redrawn simplified schematic; not a facsimile of the printed figure.
    Rows: v_pi / v* (state values), q_pi / q* (action values).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    rows = [
        ("v_pi", "vπ(s)", "policy evaluation", "TD(0)"),
        ("v_*", "v*(s)", "value iteration", "(no useful update)"),
        ("q_pi", "qπ(s, a)", "q-policy evaluation", "Sarsa"),
        ("q_*", "q*(s, a)", "q-value iteration", "Q-learning"),
    ]
    for i, (row, title, exp_label, samp_label) in enumerate(rows):
        _backup_panel(axes[i, 0], row, "expected", f"{title} expected: {exp_label}")
        _backup_panel(axes[i, 1], row, "sample", f"{title} sample: {samp_label}")
    axes[0, 0].set_ylabel("state values / action values")
    for j, col in enumerate(("Expected update (tabular DP)", "Sample update (one-step TD)")):
        axes[0, j].text(5.0, 10.4, col, ha="center", va="bottom", fontsize=11,
                        fontweight="bold")
    fig.suptitle("S&B (2nd ed.), Fig. 8.7: backup diagrams for the eight one-step updates\n"
                 "(simplified schematic)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Sec. 8.6 / Fig. 8.9: trajectory sampling vs uniform updates (reduced scale)
# ---------------------------------------------------------------------------

def _make_random_task(n_states: int, b: int, rng: np.random.Generator):
    """Random episodic task (Fig. 8.9 reconstruction).

    2 actions per state; from each (s, a), b distinct next states all equally
    likely; a 0.1 probability of transition to the terminal state on every
    transition; reward for taking action a in state s ~ U[0, 1] (the book does
    not print the reward model; any non-trivial reward works because the
    termination probability is action-independent, so expected episode
    length is policy-independent). Undiscounted.
    """
    succ = np.empty((n_states, 2, b), dtype=np.int64)
    for s in range(n_states):
        for a in range(2):
            succ[s, a] = rng.choice(n_states, size=b, replace=False)
    rewards = rng.random((n_states, 2))
    return succ, rewards


def _greedy_value_start(succ: np.ndarray, rewards: np.ndarray, q: np.ndarray) -> float:
    """Value of the start state under the greedy policy, from the learned Q.

    Returns max_a Q(0, a). (The book's y-axis starts at 0 and rises with
    updates; that is the learned action value, not the exact v of the
    induced greedy policy, which is constant-ish under U[0,1] rewards.)
    """
    return float(q[0].max())


def _expected_update(succ: np.ndarray, rewards: np.ndarray, q: np.ndarray,
                     s: int, a: int) -> float:
    nxt = succ[s, a]
    return rewards[s, a] + 0.9 * q[nxt].max(axis=1).mean()


def trajectory_sampling_experiment(
    n_states: int = 1000,
    b: int = 1,
    tasks: int = 20,
    full_backups_max: float = 60.0,
    n_checkpoints: int = 12,
    seed: int = 0,
    epsilon: float = 0.1,
):
    """Trajectory-sampling vs uniform-update experiment (Sec. 8.6, Fig. 8.9).

    Reduced-scale reproduction: the book averages over 200 tasks and runs to
    20,000/200,000 full backups; here `tasks` defaults to 30 and the x-axis
    spans `full_backups_max` full backups (1 full backup = |S| x |A| single
    expected updates). Tasks are undiscounted episodic; updates are expected
    tabular updates (book eq. 8.1); the on-policy agent follows episodes from
    state 0 under an epsilon-greedy (epsilon=0.1) policy.

    Returns (updates_grid, uniform_v, onpolicy_v) with arrays of shape
    (n_checkpoints,) in full-backup units times `tasks` repetitions averaged,
    plus the raw per-task values for inspection.
    """
    pair_updates_max = int(full_backups_max * n_states * 2)  # noqa: F841
    checkpoints = np.unique(np.round(
        np.linspace(0.05, full_backups_max, n_checkpoints) * n_states * 2).astype(np.int64))
    uniform_vals = []
    onpolicy_vals = []
    for t in range(tasks):
        rng = np.random.default_rng(seed + t)
        succ, rewards = _make_random_task(n_states, b, rng)
        for dist in ("uniform", "onpolicy"):
            q = np.zeros((n_states, 2), dtype=np.float64)
            updates = 0
            vals = []
            ck_idx = 0
            if dist == "uniform":
                # cycle through all pairs in place, wrap-around
                s, a = 0, 0
                while ck_idx < len(checkpoints):
                    q[s, a] = _expected_update(succ, rewards, q, s, a)
                    updates += 1
                    if updates >= checkpoints[ck_idx]:
                        vals.append(_greedy_value_start(succ, rewards, q))
                        ck_idx += 1
                    a += 1
                    if a == 2:
                        a = 0
                        s += 1
                        if s == n_states:
                            s = 0
            else:
                rng2 = np.random.default_rng(seed + 100003 + t)
                while ck_idx < len(checkpoints):
                    s = 0
                    # one episode from state 0
                    while True:
                        vals_q = q[s]
                        if rng2.random() < epsilon:
                            a = int(rng2.integers(0, 2))
                        else:
                            best = vals_q.max()
                            if (vals_q == best).sum() == 2:
                                a = int(rng2.integers(0, 2))
                            else:
                                a = int(np.argmax(vals_q))
                        q[s, a] = _expected_update(succ, rewards, q, s, a)
                        updates += 1
                        if updates >= checkpoints[ck_idx]:
                            vals.append(_greedy_value_start(succ, rewards, q))
                            ck_idx += 1
                            if ck_idx == len(checkpoints):
                                break
                        if rng2.random() < 0.1:
                            break
                        s = int(succ[s, a, rng2.integers(0, b)])
            (uniform_vals if dist == "uniform" else onpolicy_vals).append(vals)
    uniform_arr = np.array(uniform_vals)
    onpolicy_arr = np.array(onpolicy_vals)
    grid = checkpoints / (n_states * 2)
    return grid, uniform_arr.mean(axis=0), onpolicy_arr.mean(axis=0)


def plot_figure_8_9_trajectory_sampling(
    save_path: str | None = None, show: bool = False,
    tasks: int = 30,
    seed: int = 0,
):
    """On-policy vs uniform backups on random tasks (Sec. 8.6, Fig. 8.9).

    Upper panel: 1000-state tasks with branching factor b in {1, 3, 10}.
    Lower panel: 4000-state tasks with b = 1 (the book uses 10,000 states;
    reduced for tractability — illustrative reproduction). Averaged over
    `tasks` random tasks (book: 200). x-axis in full backups (one full backup
    = |S| x |A| single expected updates).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    styles = {1: "-", 3: "--", 10: ":"}
    for i, b in enumerate((1, 3, 10)):
        grid, uv, ov = trajectory_sampling_experiment(
            n_states=1000, b=b, tasks=tasks, full_backups_max=60.0,
            n_checkpoints=14, seed=seed)
        axes[0].plot(grid, uv, styles[b], color="0.25", alpha=0.45,
                     label=f"uniform b={b}")
        axes[0].plot(grid, ov, styles[b], color="black",
                     label=f"on-policy b={b}")
    axes[0].set_title("1000 states (b = 1, 3, 10)")
    axes[0].set_xlabel("Computation time, in full backups")
    axes[0].set_ylabel("Value of start state under greedy policy")
    axes[0].legend(fontsize=8)

    grid, uv, ov = trajectory_sampling_experiment(
        n_states=4000, b=1, tasks=max(5, tasks // 2), full_backups_max=25.0,
        n_checkpoints=12, seed=seed)
    axes[1].plot(grid, uv, "--", color="0.25", alpha=0.45, label="uniform b=1")
    axes[1].plot(grid, ov, "-", color="black", label="on-policy b=1")
    axes[1].set_title("4000 states (b = 1; book: 10,000)")
    axes[1].set_xlabel("Computation time, in full backups")
    axes[1].set_ylabel("Value of start state under greedy policy")
    axes[1].legend(fontsize=8)
    fig.suptitle("S&B (2nd ed.), Sec. 8.6 / Fig. 8.9: trajectory sampling vs uniform "
                 "backups (illustrative reproduction, reduced scale)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Sec. 8.4, Example 8.4: prioritized sweeping on scaled mazes (illustrative)
# ---------------------------------------------------------------------------

def _scaled_dyna_maze(k: int):
    """Scaled Dyna maze: same structure as the 6x9 book maze at resolution k.

    Walls are scaled to k x k blocks; start at (2k, 0), goal at (0, 8k).
    """
    base_walls = [(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (2, 7), (4, 5)]
    walls = []
    for r, c in base_walls:
        for dr in range(k):
            for dc in range(k):
                walls.append(f"{r * k + dr},{c * k + dc}")
    rows, cols = 6 * k, 9 * k
    return rows, cols, f"{2 * k},0", f"0,{8 * k}", walls


def _bfs_path_len(rows: int, cols: int, walls: list[str], start: str, goal: str):
    from collections import deque

    wall_set = set(walls)
    dist = {start: 0}
    q = deque([start])
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while q:
        s = q.popleft()
        if s == goal:
            return dist[s]
        r, c = (int(x) for x in s.split(","))
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            ns = f"{nr},{nc}"
            if (0 <= nr < rows and 0 <= nc < cols and ns not in wall_set
                    and ns not in dist):
                dist[ns] = dist[s] + 1
                q.append(ns)
    return None


def _greedy_path_len(q_table, rows, cols, walls, start, goal, optimal_len):
    """Length of the fixed-tie-break greedy path from start.

    Returns the path length in environment steps (bumps count as steps, as
    they do for the agent) when the goal is reached within
    optimal_len + slack steps, else None. Cycles and endless bumping are
    rejected by the step budget. Action order breaks argmax ties.
    """
    wall_set = set(walls)
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    actions = ["up", "down", "left", "right"]
    slack = 20
    s = start
    steps = 0
    budget = 2 * (optimal_len + slack)
    for _ in range(budget):
        if s == goal:
            return steps
        if steps > optimal_len + slack:
            return None
        vals = [q_table.get((s, a), 0.0) for a in actions]
        best = max(vals)
        a = actions[vals.index(best)]  # first of the tied actions wins
        r, c = (int(x) for x in s.split(","))
        dr, dc = deltas[a]
        nr, nc = r + dr, c + dc
        ns = f"{nr},{nc}"
        if 0 <= nr < rows and 0 <= nc < cols and ns not in wall_set:
            s = ns
        steps += 1
    return None


def _vi_qstar(rows, cols, walls, goal, gamma=0.95):
    """Exact Q* of the deterministic Dyna-maze MDP (terminal +1 at the goal).

    Value iteration on all walkable cells; used only to measure convergence
    in the Example 8.4 scaling experiment.
    """
    wall_set = set(walls)
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    cells = [f"{r},{c}" for r in range(rows) for c in range(cols) if f"{r},{c}" not in wall_set]
    q = {(s, a): 0.0 for s in cells for a in deltas}
    v = {s: 0.0 for s in cells}
    goal_reward = 1.0
    for _ in range(10000):
        new_q = {}
        new_v = {}
        delta = 0.0
        for s in cells:
            r, c = (int(x) for x in s.split(","))
            if s == goal:
                new_v[s] = 0.0
                for a in deltas:
                    new_q[(s, a)] = 0.0
                continue
            best = -np.inf
            for a, (dr, dc) in deltas.items():
                ns = f"{r + dr},{c + dc}"
                if ns not in wall_set and 0 <= r + dr < rows and 0 <= c + dc < cols:
                    rew = goal_reward if ns == goal else 0.0
                    nxt = rew + gamma * v[ns] if ns != goal else rew
                    val = nxt
                else:
                    val = 0.0 + gamma * v[s]  # bump: stay, reward 0
                new_q[(s, a)] = val
                if val > best:
                    best = val
            new_v[s] = best
            delta = max(delta, abs(new_v[s] - v[s]))
        q, v = new_q, new_v
        if delta < 1e-12:
            break
    return q


class _Solved(Exception):
    """Control-flow exception: the agent's greedy policy is optimal; stop the run."""


class _Budget(Exception):
    """Control-flow exception: the per-run computation budget was exhausted."""


def _scaling_run_one(k, agent, repeat, seed_base, max_steps, qstar_tol=0.01,
                     budget_updates: int = 80_000):
    rows, cols, start, goal, walls = _scaled_dyna_maze(k)
    n_states = rows * cols - len(walls)
    optimal_len = _bfs_path_len(rows, cols, walls, start, goal)
    rs = seed_base + k * 1000 + repeat * 137
    counter = {"updates": 0}

    def cb(ep, steps, state, q_table, pops):
        counter["updates"] += 1 + pops
        if counter["updates"] >= budget_updates:
            raise _Budget()
        if ep < 1:
            return  # burn the first episode; nothing has been learned yet
        # Book convention: the solution is found when the greedy policy from S
        # first reaches G along a shortest path. Checking Q* over *all* pairs
        # would be far stricter than the book's criterion (many off-path pairs
        # need not be accurate for the start-state policy to be optimal).
        path_len = _greedy_path_len(q_table, rows, cols, walls, start, goal,
                                    optimal_len)
        if path_len == optimal_len:
            raise _Solved()

    env = make_gridworld(rows=rows, cols=cols, start=start, goal=goal,
                         gamma=0.95, seed=rs, walls=walls)
    try:
        if agent == "dyna_q":
            dyna_q(env, n_planning=5, n_episodes=60, alpha=0.1, epsilon=0.1,
                   gamma=0.95, seed=rs, max_steps=max_steps,
                   start_state=start, step_callback=cb)
        else:
            # Exact (alpha=1) backups are the natural choice for the book's
            # deterministic maze; a small alpha only adds needless delay.
            prioritized_sweeping(env, theta=1e-3, n_planning=5, n_episodes=60,
                                 alpha=1.0, epsilon=0.1, gamma=0.95, seed=rs,
                                 max_steps=max_steps, start_state=start,
                                 step_callback=cb)
        solved_at = None
    except _Solved:
        solved_at = counter["updates"]
    except _Budget:
        # No optimal start-state policy within the computation cap: report
        # None rather than inventing a convergence time.
        solved_at = None
    return n_states, optimal_len, solved_at


def scaling_maze_experiment(
    scale_levels: tuple[int, ...] = (1, 2, 3),
    repeats: int = 3,
    seed: int = 0,
    max_steps: int = 1200,
    budget_updates: int = 80_000,
):
    """Updates until the optimal solution is found, on scaled Dyna mazes.

    Example 8.4 (Sec. 8.4): the same maze structure at increasing grid
    resolutions, for Dyna-Q and prioritized sweeping, both making at most
    n = 5 updates per environmental interaction. "Optimal solution found"
    means the tie-free greedy path from S to G is the BFS shortest path.
    Illustrative reduced-scale reproduction of the book's figure (adapted
    from Peng & Williams 1993); the book spans ~10^2 to ~10^7 states and
    ~10^2 to ~10^6 updates.

    Runs are capped at `budget_updates` updates: a run that has not found an
    optimal start-state policy by then is reported as ``None`` (not silently
    treated as solved).

    Returns dict: {(agent, scale): (n_states, [updates, ...])} with None for
    unsolved runs.
    """
    results = {}
    for k in scale_levels:
        for agent in ("dyna_q", "prioritized_sweeping"):
            rows, cols, start, goal, walls = _scaled_dyna_maze(k)
            n_states = rows * cols - len(walls)
            runs = []
            for r in range(repeats):
                _, _, solved_at = _scaling_run_one(
                    k, agent, r, seed, max_steps, budget_updates=budget_updates)
                runs.append(solved_at)
            results[(agent, k)] = (n_states, runs)
    return results


def plot_example_8_4_scaling(
    save_path: str | None = None, show: bool = False,
    scale_levels: tuple[int, ...] = (1, 2, 3),
    repeats: int = 3,
    seed: int = 0,
    max_steps: int = 1200,
    budget_updates: int = 80_000,
):
    """Example 8.4 figure: updates until optimal solution vs maze size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = scaling_maze_experiment(scale_levels=scale_levels,
                                      repeats=repeats, seed=seed,
                                      max_steps=max_steps,
                                      budget_updates=budget_updates)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for agent, marker in (("dyna_q", "o"), ("prioritized_sweeping", "s")):
        xs, means, stds = [], [], []
        for k in scale_levels:
            n_states, runs = results[(agent, k)]
            solved = [u for u in runs if u is not None]
            if not solved:
                continue
            xs.append(n_states)
            means.append(np.mean(solved))
            stds.append(np.std(solved) if len(solved) > 1 else 0.0)
        ax.errorbar(xs, means, yerr=stds, marker=marker, capsize=5,
                    label={"dyna_q": "Dyna-Q (n=5)",
                           "prioritized_sweeping": "Prioritized sweeping (n=5)"}[agent])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gridworld size (#states)")
    ax.set_ylabel("Updates until optimal solution")
    ax.set_title("S&B (2nd ed.), Sec. 8.4, Example 8.4: prioritized sweeping on\n"
                 "mazes (Peng & Williams 1993; reduced scale; unsolved runs omitted)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, results
