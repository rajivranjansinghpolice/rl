"""Example 8.6 (Sutton & Barto, 2nd ed., Sec. 8.7): RTDP on the Racetrack.

RTDP is an asynchronous, on-policy-trajectory-sampling version of value
iteration: it performs expected value-iteration backups only on states visited
in simulated trajectories from the start states. The book reports that on a
small racetrack RTDP used about half as many updates as sweep-based value
iteration to obtain an equally good policy, because it concentrated updates
on the states relevant to the start-goal objective.

The model here is built from the shared :mod:`rl_envs.racetrack` environment
with ``noise_prob=0``: transitions are deterministic except for the random
reset to a start cell after hitting a boundary, modelled explicitly as a
uniform distribution over start states. This small track is an illustrative
reconstruction, not the exact 9,115-state track printed in the book.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rl_envs.racetrack import ACTIONS, RacetrackEnv

TERMINAL = None  # value zero; no outgoing transitions
Start = tuple[int, int, int, int]


def _transition(
    env: RacetrackEnv, state: Start, action_index: int
) -> list[tuple[float, float, Start | None]]:
    """Deterministic/noise-free transition of RacetrackEnv.step as a model.

    Boundary hits return to a uniformly random start state and the episode
    continues (book Exercise 5.7 / Example 8.6).
    """
    row, col, vx, vy = state
    dvx, dvy = ACTIONS[action_index]
    new_vx = int(np.clip(vx + dvx, 0, env.MAX_VELOCITY))
    new_vy = int(np.clip(vy + dvy, 0, env.MAX_VELOCITY))
    if new_vx == 0 and new_vy == 0 and (row, col) not in env.start_cells:
        new_vx, new_vy = vx, vy
    path = env.projected_path(row, col, new_vx, new_vy)
    if any(env._is_finish(r, c) for r, c in path):
        return [(1.0, -1.0, TERMINAL)]
    if any(not env._is_legal_position(r, c) for r, c in path):
        prob = 1.0 / len(env.start_cells)
        return [(prob, -1.0, (r, c, 0, 0)) for r, c in env.start_cells]
    return [(1.0, -1.0, (row - new_vy, col + new_vx, new_vx, new_vy))]


def build_model(env: RacetrackEnv) -> tuple[dict, list[Start]]:
    """Build the finite reachable-state model by BFS from all start states."""
    start_states = [(r, c, 0, 0) for r, c in env.start_cells]
    model: dict = {}
    seen = set(start_states)
    queue = deque(start_states)
    while queue:
        state = queue.popleft()
        for a in range(len(ACTIONS)):
            outcomes = _transition(env, state, a)
            model[(state, a)] = outcomes
            for _p, _r, nxt in outcomes:
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return model, start_states


def _q_value(v: dict, model: dict, state: Start, a: int, gamma: float) -> float:
    return sum(
        prob * (reward + (0.0 if nxt is TERMINAL else gamma * v[nxt]))
        for prob, reward, nxt in model[(state, a)]
    )


def greedy_action(v: dict, model: dict, state: Start, gamma: float, rng) -> int:
    values = [_q_value(v, model, state, a, gamma) for a in range(len(ACTIONS))]
    best = max(values)
    ties = [a for a, q in enumerate(values) if q == best]
    return int(ties[int(rng.integers(len(ties)))])


def evaluate_policy(v: dict, model: dict, start_states: list[Start], gamma: float,
                    rng, episodes: int = 20, max_steps: int = 500) -> float:
    steps = []
    for _ in range(episodes):
        state = start_states[int(rng.integers(len(start_states)))]
        for t in range(1, max_steps + 1):
            a = greedy_action(v, model, state, gamma, rng)
            outcomes = model[(state, a)]
            probs = [p for p, _r, _s in outcomes]
            idx = int(rng.choice(len(outcomes), p=probs))
            state = outcomes[idx][2]
            if state is TERMINAL:
                steps.append(t)
                break
        else:
            steps.append(max_steps)
    return float(np.mean(steps))


def rtdp(
    model: dict,
    start_states: list[Start],
    n_episodes: int = 4000,
    gamma: float = 1.0,
    seed: int = 0,
    eval_interval: int = 100,
    patience: int = 3,
    tol: float = 0.3,
) -> dict:
    """RTDP: expected backups only on states visited in greedy trajectories.

    Training stops when the greedy policy's evaluated episode length has
    stabilised for ``patience`` consecutive checkpoints (the book uses a
    similar visual convergence criterion).
    """
    rng = np.random.default_rng(seed)
    v = {s: 0.0 for s, _a in model}
    update_counts: dict = {s: 0 for s in v}
    history: list[tuple[int, int, float]] = []
    total_updates = 0
    for episode in range(1, n_episodes + 1):
        state = start_states[int(rng.integers(len(start_states)))]
        for _ in range(500):
            a = greedy_action(v, model, state, gamma, rng)
            # Expected (value-iteration) backup of the current state only.
            v[state] = _q_value(v, model, state, a, gamma)
            update_counts[state] += 1
            total_updates += 1
            outcomes = model[(state, a)]
            probs = [p for p, _r, _s in outcomes]
            idx = int(rng.choice(len(outcomes), p=probs))
            state = outcomes[idx][2]
            if state is TERMINAL:
                break
        if episode % eval_interval == 0:
            steps = evaluate_policy(v, model, start_states, gamma, rng)
            history.append((episode, total_updates, steps))
            if len(history) >= patience + 1:
                recent = [h[2] for h in history[-patience:]]
                if max(recent) - min(recent) <= tol:
                    break
    counts = np.array(list(update_counts.values()))
    return {
        "updates": total_updates,
        "episodes": n_episodes,
        "policy_steps": evaluate_policy(v, model, start_states, gamma, rng),
        "history": history,
        "pct_zero_updates": float(100 * np.mean(counts == 0)),
        "pct_le_10": float(100 * np.mean(counts <= 10)),
        "pct_le_100": float(100 * np.mean(counts <= 100)),
        "v": v,
    }


def value_iteration(
    model: dict,
    start_states: list[Start],
    gamma: float = 1.0,
    theta: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Conventional Gauss-Seidel value iteration (in-place sweeps)."""
    rng = np.random.default_rng(seed)
    v = {s: 0.0 for s, _a in model}
    updates = 0
    sweeps = 0
    history = []
    while sweeps < 2000:
        delta = 0.0
        # Arbitrary but fixed state order, as in the book's comparison.
        for state in sorted(v):
            best = max(_q_value(v, model, state, a, gamma)
                       for a in range(len(ACTIONS)))
            delta = max(delta, abs(best - v[state]))
            v[state] = best
            updates += 1
        sweeps += 1
        if sweeps % 5 == 0:
            history.append((sweeps, updates,
                            evaluate_policy(v, model, start_states, gamma, rng)))
        if delta < theta:
            break
    return {
        "updates": updates,
        "sweeps": sweeps,
        "policy_steps": evaluate_policy(v, model, start_states, gamma, rng),
        "history": history,
        "v": v,
    }


def rtdp_experiment(seed: int = 0) -> dict:
    """Compare RTDP and exhaustive value iteration on the small racetrack."""
    # A longer illustrative track (still in the spirit of book Fig. 5.5)
    # gives enough states for the focus of RTDP to be visible.
    track = [
        "###..........F",
        "##...........F",
        "##...........F",
        "##........####",
        "##........####",
        "##........####",
        "##........####",
        "##........####",
        "##........####",
        "#SSSSSSSSSS###",
    ]
    env = RacetrackEnv(track=track, noise_prob=0.0, seed=seed)
    model, starts = build_model(env)
    dp = value_iteration(model, starts, seed=seed)
    rtdp_result = rtdp(model, starts, seed=seed)
    return {
        "n_states": len(dp["v"]),
        "n_start_states": len(starts),
        "dp": dp,
        "rtdp": rtdp_result,
    }


def plot_example_8_6_rtdp(save_path: str | None = None, show: bool = False) -> dict:
    """Summary figure for Example 8.6 (the book prints a table, not a figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = rtdp_experiment()
    dp, rt = result["dp"], result["rtdp"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(["Value iteration", "RTDP"], [dp["updates"], rt["updates"]],
                color=["0.6", "tab:blue"])
    axes[0].set_ylabel("Updates to convergence")
    axes[0].set_title(f"Updates (track: {result['n_states']:,} reachable states)")
    for i, value in enumerate((dp["updates"], rt["updates"])):
        axes[0].text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    axes[1].plot([h[0] for h in dp["history"]], [h[2] for h in dp["history"]],
                 "o-", label="Value iteration (x = sweeps)")
    axes[1].plot([h[0] for h in rt["history"]], [h[2] for h in rt["history"]],
                 "s-", label="RTDP (x = 250 episodes)")
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("Steps to finish under greedy policy")
    axes[1].legend(fontsize=8)
    axes[1].set_title("On-policy focus reaches a good policy with fewer updates")
    fig.suptitle("S&B (2nd ed.), Sec. 8.7, Example 8.6: RTDP on the racetrack\n"
                 "illustrative small-track reproduction (book: 9,115 states)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return result
