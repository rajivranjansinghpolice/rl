"""Solution implementation for Chapter 17 - Frontiers.

Reference: Sutton & Barto (2nd Ed), Chapter 17 (Frontiers).

Chapter 17 discusses open problems; its two sections with implementable
content are 17.2 (Temporal Abstraction via Options) and 17.4 (Designing
Reward Signals, incl. intrinsic motivation / curiosity). This chapter
implements demos of both:

- Alg 17.1 — Options / hierarchical RL in a 4-room GridWorld (Section 17.2)
- Alg 17.2 — Intrinsic motivation via a count-based novelty bonus
  R+ = R + beta / sqrt(N(s) + 1) (Section 17.4)

Both run on a shared tiny tabular 7x7 four-room GridWorld. The option
implementations use BFS-greedy subgoal policies for the low-level skills,
as is standard in options demonstrations; option *values* are learned with
SMDP Q-learning (Section 17.2: options extend the action space).
"""

from __future__ import annotations

import math

import numpy as np

# The shared four-room GridWorld and option utilities live in the
# ``rl_envs`` package (``rl_envs.four_rooms``); they are re-exported here
# so existing imports (``from chapter_17.solution import GridWorld, ...``)
# keep working.
from rl_envs.four_rooms import (
    _START,
    DEFAULT_GRID,
    GridWorld,
    Option,
    OptionResult,
    bfs_action_toward,
    bfs_distances,
    default_options,
    is_wall,
    make_subgoal_option,
    run_option,
)

__all__ = [
    # Re-exported from rl_envs.four_rooms (shared with the book's examples)
    "DEFAULT_GRID",
    "GridWorld",
    "Option",
    "OptionResult",
    "bfs_action_toward",
    "bfs_distances",
    "default_options",
    "is_wall",
    "make_subgoal_option",
    "run_option",
    # Chapter 17 implementations
    "greedy_smdp_action",
    "novelty_bonus",
    "plot_options_and_novelty_demos",
    "primitive_q_learning",
    "q_learning_novelty",
    "rollout_primitive",
    "rollout_smdp",
    "smdp_q_learning",
]


def _epsilon_greedy_probs(
    q_table: dict, state: int, n_actions: int, epsilon: float,
) -> np.ndarray:
    """Return epsilon-greedy action probabilities, splitting greedy ties evenly.

    Primitive and option indices share the same tie rule, so extending the
    action space does not change how exploratory or greedy choices are made.
    """
    q_values = np.array([q_table[(state, a)] for a in range(n_actions)])
    best = np.flatnonzero(q_values == q_values.max())
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def _smdp_transition(
    env: GridWorld,
    option_list: list[Option],
    action: int,
    gamma: float,
    max_option_steps: int,
) -> tuple[int, float, int, bool]:
    """Unroll one primitive action or option into (state, return, tau, done).

    An option's tau is the number of primitive steps it took; it drives both
    the gamma**tau backup and the outer episode's safety budget.
    """
    if action < len(GridWorld.ACTIONS):
        next_state, reward, done = env.step(action)
        return next_state, reward, 1, done
    option = option_list[action - len(GridWorld.ACTIONS)]
    result = run_option(env, option, gamma=gamma, max_steps=max_option_steps)
    return env.state, result.discounted_return, result.steps, result.done


def smdp_q_learning(
    n_episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int | None = None,
    options: list[Option] | None = None,
    max_option_steps: int = 50,
    max_primitive_steps: int = 200,
) -> tuple[dict, list[Option]]:
    """SMDP Q-learning over primitives and options (Section 17.2).

    Options extend the action space: action indices [0, 4) are primitive
    moves; indices [4, 4 + len(options)) are options, executed to
    termination with duration tau. The update uses the generalized
    (semi-MDP) Bellman target:

        Q(s, o) <- Q(s, o) + alpha * [R + gamma**tau * max_a Q(s', a) - Q(s, o)]

    where R is the discounted return over the option and tau its duration.

    Returns:
        (q_table, options): dict[(state, action_index)] -> value, and the
        option list used (indices 4.. in the action space).
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    env = GridWorld()
    option_list = list(options) if options is not None else default_options()
    n_actions = len(GridWorld.ACTIONS) + len(option_list)
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in range(env.n_states) for a in range(n_actions)}

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        primitive_steps = 0
        while not done and primitive_steps < max_primitive_steps:
            probs = _epsilon_greedy_probs(q_table, state, n_actions, epsilon)
            action = int(rng.choice(n_actions, p=probs))
            next_state, cumulative, tau, done = _smdp_transition(
                env, option_list, action, gamma, max_option_steps
            )
            primitive_steps += tau
            next_q = (
                0.0
                if done
                else max(q_table[(next_state, a)] for a in range(n_actions))
            )
            target = cumulative + gamma**tau * next_q
            q_table[(state, action)] += alpha * (target - q_table[(state, action)])
            state = next_state

    return q_table, option_list


def greedy_smdp_action(q_table: dict, state: int, n_actions: int) -> int:
    """Greedy action (primitive or option index) for `state`."""
    return int(np.argmax([q_table[(state, a)] for a in range(n_actions)]))


def rollout_smdp(
    q_table: dict,
    n_actions: int,
    n_options: int,
    start: tuple[int, int] = _START,
    max_decisions: int = 100,
    max_option_steps: int = 50,
    env: GridWorld | None = None,
    options: list[Option] | None = None,
) -> dict:
    """Greedy rollout under an SMDP policy.

    Args:
        q_table: Trained SMDP Q-table over (state, action_index).
        n_actions: Total action count (primitives + options).
        n_options: Number of options (indices 4..4+n_options-1).
        start: Start cell for the rollout.
        max_decisions: Decision budget (macro-steps).
        max_option_steps: Primitive step cap per option unroll.
        env: Optional environment instance (defaults to a fresh GridWorld).
        options: Option list to unroll (defaults to default_options());
            pass the list the agent was trained with for custom options.

    Returns:
        dict with 'decisions' (macro-decisions taken), 'primitive_steps',
        'option_decisions', 'final_cell', 'success'.
    """
    world = env if env is not None else GridWorld()
    world.start = start
    world.reset()
    decisions = 0
    option_decisions = 0
    primitive_steps = 0
    option_list = list(options) if options is not None else default_options()[:n_options]
    done = False
    while not done and decisions < max_decisions:
        state = world.state
        action = greedy_smdp_action(q_table, state, n_actions)
        decisions += 1
        if action < len(GridWorld.ACTIONS):
            _, _, done = world.step(action)
            primitive_steps += 1
        else:
            option = option_list[action - len(GridWorld.ACTIONS)]
            result = run_option(world, option, gamma=0.99, max_steps=max_option_steps)
            primitive_steps += result.steps
            option_decisions += 1
            done = result.done
    return {
        "decisions": decisions,
        "primitive_steps": primitive_steps,
        "option_decisions": option_decisions,
        "final_cell": world.cell(world.state),
        "success": bool(done),
    }


# Safety bound on an episode, not a book parameter: this GridWorld terminates
# on its own, but a pathological policy must not be able to loop forever.
_PRIMITIVE_EPISODE_STEP_CAP = 200


def primitive_q_learning(
    n_episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict:
    """Tabular Q-learning over the 4 primitive actions (Chapter 6 flavor).

    Baseline for the options comparison: the agent acts only at primitive
    one-step granularity. The loop exists to bound a runaway episode; it is a
    safety cap, not part of the algorithm.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    env = GridWorld()
    n_actions = len(GridWorld.ACTIONS)
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in range(env.n_states) for a in range(n_actions)}

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < _PRIMITIVE_EPISODE_STEP_CAP:
            probs = _epsilon_greedy_probs(q_table, state, n_actions, epsilon)
            action = int(rng.choice(n_actions, p=probs))
            next_state, reward, done = env.step(action)
            next_q = (
                0.0
                if done
                else max(q_table[(next_state, a)] for a in range(n_actions))
            )
            q_table[(state, action)] += alpha * (
                reward + gamma * next_q - q_table[(state, action)]
            )
            state = next_state
            steps += 1

    return q_table


def rollout_primitive(
    q_table: dict,
    start: tuple[int, int] = _START,
    max_steps: int = 500,
) -> dict:
    """Greedy rollout of a primitive-action Q table."""
    env = GridWorld()
    env.start = start
    env.reset()
    steps = 0
    done = False
    while not done and steps < max_steps:
        state = env.state
        action = greedy_smdp_action(q_table, state, len(GridWorld.ACTIONS))
        _, _, done = env.step(action)
        steps += 1
    return {"primitive_steps": steps, "decisions": steps, "success": bool(done)}


# ============================================================================
# Algorithm 17.2: Intrinsic motivation via count-based novelty bonus
# Book Section 17.4 (Designing Reward Signals); R+ = R + beta / sqrt(N + 1)
# ============================================================================


def novelty_bonus(visits: int, beta: float) -> float:
    """Count-based novelty bonus: beta / sqrt(visits + 1).

    Args:
        visits: Number of times the state has been visited (0 for a novel
            state), i.e. N(s) in R+ = R + beta / sqrt(N(s) + 1).
        beta: Bonus scale.
    """
    if visits < 0:
        raise ValueError("visits must be non-negative")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    return float(beta / math.sqrt(visits + 1))


def _build_gridworld(
    grid: list[str] | None,
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
) -> GridWorld:
    """Build the experiment world, overriding only the fields that were supplied.

    The kwargs are inserted in a fixed order so every subset of overrides gives
    the same environment the original inline construction produced.
    """
    env_kwargs: dict = {}
    if grid is not None:
        env_kwargs["grid"] = grid
    if start is not None:
        env_kwargs["start"] = start
    if goal is not None:
        env_kwargs["goal"] = goal
    return GridWorld(**env_kwargs)


def _novelty_step_bonus(
    counts: np.ndarray, state: int, next_state: int, beta: float,
) -> float:
    """Return the intrinsic bonus for a transition, zero for a no-op.

    A bump against a wall reveals nothing new, and rewarding it would make
    wall-hugging an endless reward source. The count is read before it is
    incremented, so a first visit gets the full beta / sqrt(1) bonus.
    """
    if next_state == state:
        return 0.0
    return novelty_bonus(int(counts[next_state]), beta)


def q_learning_novelty(
    n_episodes: int = 20,
    beta: float = 0.5,
    alpha: float = 1.0,
    gamma: float = 0.0,
    epsilon: float = 0.2,
    seed: int | None = None,
    grid: list[str] | None = None,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    max_steps: int = 200,
    q_init: float = 0.5,
) -> dict:
    """Q-learning with a count-based novelty reward bonus (Section 17.4).

    The bonus for landing in a state is added to the extrinsic reward:
    R+ = R + beta / sqrt(N(s') + 1). Counts are accumulated across the
    whole run (all episodes). Implementation notes, documented honestly:

    - No-op (bump) transitions harvest no intrinsic bonus: they carry no
      novel information, and a non-zero bonus would make wall-hugging an
      infinite reward source.
    - Default gamma is 0.0 (bandit-style): with discounting near 1, the
      bootstrapped bonus re-inflates values around a revisitation cycle,
      trapping the tabular agent. With gamma = 0 the bonus is a clean
      one-step frontier signal.
    - Default q_init = 0.5 (optimistic initialization at the never-visited
      bonus level, beta): the standard partner of count-based exploration.
      Without it, untried actions lose to already-visited inflated Q
      values and the agent cannot reach the frontier.

    Returns:
        dict with 'q_table', 'counts' (np.ndarray over state ids),
        'unique_states' (number of distinct states visited), 'steps',
        'total_intrinsic_reward'.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    env = _build_gridworld(grid, start, goal)
    rng = np.random.default_rng(seed)
    n_actions = len(GridWorld.ACTIONS)
    q_table = {(s, a): q_init for s in range(env.n_states) for a in range(n_actions)}
    counts = np.zeros(env.n_states, dtype=np.int64)
    total_intrinsic = 0.0
    total_steps = 0

    for _ in range(n_episodes):
        state = env.reset()
        counts[state] += 1
        done = False
        steps = 0
        while not done and steps < max_steps:
            probs = _epsilon_greedy_probs(q_table, state, n_actions, epsilon)
            action = int(rng.choice(n_actions, p=probs))
            next_state, reward, done = env.step(action)
            # No-op bumps still update their count, but earn no bonus.
            bonus = _novelty_step_bonus(counts, state, next_state, beta)
            total_intrinsic += bonus
            shaped_reward = reward + bonus
            next_q = (
                0.0
                if done
                else max(q_table[(next_state, a)] for a in range(n_actions))
            )
            q_table[(state, action)] += alpha * (
                shaped_reward + gamma * next_q - q_table[(state, action)]
            )
            counts[next_state] += 1
            state = next_state
            steps += 1
            total_steps += 1

    return {
        "q_table": q_table,
        "counts": counts,
        "unique_states": int(np.count_nonzero(counts)),
        "steps": total_steps,
        "total_intrinsic_reward": total_intrinsic,
    }


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 17)
# ---------------------------------------------------------------------------

# Fixed demo settings; the algorithms' own defaults are left untouched.
_PLOT_TRAIN_EPISODES = 200
_PLOT_ROLLOUTS = 20
_PLOT_ROLLOUT_MAX_STEPS = 200
_PLOT_ROLLOUT_MAX_DECISIONS = 100
_PLOT_OPTION_MAX_STEPS = 50
_PLOT_NOVELTY_EPISODES = 12
_PLOT_UNKNOWN_Q = -1e9


def _greedy_primitive_rollout_steps(q: dict, env: GridWorld, max_steps: int) -> int:
    """Count primitive steps to goal under a greedy policy, or cap the attempt.

    A cap is needed because a failed policy would otherwise roll out forever.
    """
    s = env.reset()
    for steps in range(1, max_steps + 1):
        qs = [q.get((s, a), 0.0) for a in range(len(GridWorld.ACTIONS))]
        a = int(np.argmax(qs))
        s, _, done = env.step(a)
        if done:
            return steps
    return max_steps


def _greedy_smdp_rollout_steps(
    q: dict,
    env: GridWorld,
    options: list[Option],
    n_actions: int,
    max_steps: int,
) -> int:
    """Count primitive steps to goal under greedy SMDP (option) play.

    Options are unrolled with their own subgoal policy until termination; the
    decision budget is separate from the primitive-step budget.
    """
    n_prim_actions = len(GridWorld.ACTIONS)
    s = env.reset()
    total_steps = 0
    decisions = 0
    while total_steps < max_steps and decisions < _PLOT_ROLLOUT_MAX_DECISIONS:
        vals = [q.get((s, a), _PLOT_UNKNOWN_Q) for a in range(n_actions)]
        a = int(np.argmax(vals))
        if a < n_prim_actions:
            s, _, done = env.step(a)
            total_steps += 1
        else:
            opt = options[a - n_prim_actions]
            for _ in range(_PLOT_OPTION_MAX_STEPS):
                act = opt.policy(env)
                s, _, done = env.step(act)
                total_steps += 1
                if opt.terminates(env) or done or total_steps >= max_steps:
                    break
        decisions += 1
        if env.done:
            return total_steps
    return max_steps


def plot_options_and_novelty_demos(
    save_path: str | None = None,
    show: bool = False,
    seed: int = 0,
):
    """Temporal abstraction (options) and intrinsic motivation demos (Ch. 17).

    Sutton & Barto (2nd ed.), Ch. 17 "Frontiers":
      - Sec. 17.2 (Temporal abstraction via options): SMDP Q-learning with
        hand-crafted options vs primitive-only Q-learning in the four-room
        GridWorld. Left panel compares mean primitive steps to reach the goal
        under greedy rollouts of learned policies.
      - Sec. 17.4 (Designing reward signals / intrinsic motivation): count-based
        novelty bonus encourages broader exploration. Right panel compares the
        number of distinct states visited with novelty bonus vs without.
    Small demonstrations of implemented ideas, not direct book-figure copies.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_smdp, options = smdp_q_learning(
        n_episodes=_PLOT_TRAIN_EPISODES, alpha=0.1, gamma=0.99, epsilon=0.1, seed=seed,
    )
    q_prim = primitive_q_learning(
        n_episodes=_PLOT_TRAIN_EPISODES, alpha=0.1, gamma=0.99, epsilon=0.1, seed=seed,
    )
    n_actions = len(GridWorld.ACTIONS) + len(options)

    # Each rollout starts from a fresh world; average over repeated greedy runs.
    prim_steps = float(np.mean([
        _greedy_primitive_rollout_steps(q_prim, GridWorld(), _PLOT_ROLLOUT_MAX_STEPS)
        for _ in range(_PLOT_ROLLOUTS)
    ]))
    smdp_steps = float(np.mean([
        _greedy_smdp_rollout_steps(
            q_smdp, GridWorld(), options, n_actions, _PLOT_ROLLOUT_MAX_STEPS
        )
        for _ in range(_PLOT_ROLLOUTS)
    ]))

    res_nov = q_learning_novelty(n_episodes=_PLOT_NOVELTY_EPISODES, beta=0.5, seed=seed)
    res_plain = q_learning_novelty(n_episodes=_PLOT_NOVELTY_EPISODES, beta=0.0, seed=seed)
    nov_count = int(res_nov.get("unique_states", 0))
    plain_count = int(res_plain.get("unique_states", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.bar(["Primitive\nQ-learning", "SMDP with\noptions"], [prim_steps, smdp_steps],
            color=["#f58518", "#4c78a8"])
    ax1.set_ylabel("Mean primitive steps to goal")
    ax1.set_title("Temporal abstraction (Sec. 17.2)")

    ax2.bar(["No bonus\n(beta=0)", "Novelty bonus\n(beta>0)"], [plain_count, nov_count],
            color=["#f58518", "#4c78a8"])
    ax2.set_ylabel("Distinct states visited")
    ax2.set_title("Intrinsic motivation (Sec. 17.4)")

    fig.suptitle("S&B (2nd ed.), Ch. 17: Frontiers demos")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)
