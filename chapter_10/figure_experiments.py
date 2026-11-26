"""Illustrative reproductions of Chapter 10 book figures (S&B 2nd ed.).

Verified captions from the local PDF:

    Fig. 10.1  Mountain Car task and cost-to-go (-max_a q) at several episodes
    Fig. 10.2  semi-gradient Sarsa Mountain Car learning curves vs alpha
    Fig. 10.3  n-step semi-gradient Sarsa: n = 1 vs n = 8
    Fig. 10.4  alpha x n grid of early Mountain Car performance
    Fig. 10.5  access-control queuing: learned policy and value function

Experiments are reduced-scale (fewer episodes/steps than the book). Reuse:
`rl_envs.MountainCarEnv`, `chapter_10.solution.mountain_car_features`,
`run_mountain_car`, `run_access_control`, `AccessControlQueue`.
"""

from __future__ import annotations

import numpy as np

from chapter_10.solution import (
    AccessControlQueue,
    mountain_car_features,
    run_access_control,
    run_mountain_car,
)
from rl_envs.mountain_car import MountainCarEnv

TILE_SCALE = 8.0  # the book scales alpha by the number of tilings (8)


def run_nstep_mountain_car(
    n: int = 1,
    alpha: float = 0.5,
    episodes: int = 300,
    seed: int = 0,
    max_steps: int = 200,
    epsilon: float = 0.1,
    snapshots: tuple[int, ...] = (),
) -> tuple[np.ndarray, list[int], dict[int, np.ndarray]]:
    """n-step semi-gradient Sarsa with tile-coded linear action values."""
    rng = np.random.default_rng(seed)
    env = MountainCarEnv(seed=seed)
    dim = 8 * 8 * 8
    w = np.zeros((3, dim), dtype=np.float64)
    lengths: list[int] = []
    snaps: dict[int, np.ndarray] = {}

    def act(x: np.ndarray) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(3))
        return int(np.argmax(w @ x))

    for ep in range(1, episodes + 1):
        state = env.reset()
        x = mountain_car_features(*state)
        a = act(x)
        xs, actions, rewards = [x], [a], []
        for _ in range(max_steps):
            state_next, reward, done = env.step(a)
            rewards.append(reward)
            x_next = None if done else mountain_car_features(*state_next)
            if done:
                a_next = None
            else:
                a_next = act(x_next)
                xs.append(x_next)
                actions.append(a_next)
            T = len(rewards)
            # Only update states whose full n-step return is already observed
            # (tau + n <= T - 1); the final n states are updated on later
            # steps. Updating the most recent step without its bootstrap would
            # push every non-terminal value toward the -1 step reward.
            for tau in range(max(0, T - 1 - n), T - n):
                end = tau + n
                g = 0.0  # gamma = 1
                for k in range(end - 1, tau - 1, -1):
                    g = rewards[k] + g
                g += float(w[actions[end]] @ xs[end])
                x_t, a_t = xs[tau], actions[tau]
                w[a_t] += (alpha / TILE_SCALE) * (g - float(w[a_t] @ x_t)) * x_t
            if ep in snapshots:
                snaps[ep] = w.copy()
            if done:
                break
            x, a = x_next, a_next
        lengths.append(len(rewards))
    return w, lengths, snaps


def cost_to_go_grid(weights: np.ndarray, n_pos: int = 30,
                    n_vel: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-max_a q(s, a) over a position x velocity grid."""
    positions = np.linspace(-1.2, 0.5, n_pos)
    velocities = np.linspace(-0.07, 0.07, n_vel)
    grid = np.zeros((n_pos, n_vel), dtype=np.float64)
    for i, p in enumerate(positions):
        for j, v in enumerate(velocities):
            x = mountain_car_features(float(p), float(v))
            grid[i, j] = -float(np.max(weights @ x))
    return positions, velocities, grid


def figure_10_1_data(episodes: int = 300, seed: int = 0) -> dict:
    """Cost-to-go snapshots for Fig. 10.1 (reduced episode numbers)."""
    snaps_at = (1, 12, 104, 300)
    _w, _lengths, snaps = run_nstep_mountain_car(
        n=1, alpha=0.5, episodes=episodes, seed=seed, snapshots=snaps_at)
    return {"snapshots": {ep: cost_to_go_grid(w) for ep, w in snaps.items()}}


def figure_10_2_data(episodes: int = 300, seed: int = 0) -> dict:
    """Learning curves for several step sizes (Fig. 10.2)."""
    out = {}
    for alpha in (0.1, 0.2, 0.5):
        _agent, lengths = run_mountain_car(episodes=episodes, seed=seed,
                                           alpha=alpha)
        out[alpha] = lengths
    return {"curves": out, "episodes": episodes}


def figure_10_3_data(episodes: int = 300, seed: int = 0) -> dict:
    """n = 1 vs n = 8 (Fig. 10.3)."""
    out = {}
    for n, alpha in ((1, 0.5), (8, 0.3)):
        _, lengths, _ = run_nstep_mountain_car(n=n, alpha=alpha,
                                                episodes=episodes, seed=seed)
        out[n] = lengths
    return {"curves": out, "episodes": episodes}


def figure_10_4_data(episodes: int = 100, seed: int = 0) -> dict:
    """Early performance over an alpha x n grid (Fig. 10.4, reduced)."""
    alphas = (0.1, 0.3, 0.5, 0.8, 1.2)
    ns = (1, 2, 4, 8)
    grid = np.zeros((len(ns), len(alphas)), dtype=np.float64)
    for i, n in enumerate(ns):
        for j, alpha in enumerate(alphas):
            _, lengths, _ = run_nstep_mountain_car(
                n=n, alpha=alpha, episodes=episodes, seed=seed,
                max_steps=200, snapshots=())
            grid[i, j] = float(np.mean(lengths[:50]))
    return {"alphas": list(alphas), "ns": list(ns), "grid": grid}


def figure_10_5_data(n_steps: int = 20_000, seed: int = 0) -> dict:
    """Access-control policy and value heatmaps (Fig. 10.5, reduced steps)."""
    agent, _moving = run_access_control(n_steps=n_steps, seed=seed)
    env = AccessControlQueue(seed=seed)
    n = env.capacity + 1
    value = np.zeros((n, n), dtype=np.float64)
    policy = np.zeros((n, n), dtype=np.int64)
    for h in range(n):
        for l in range(n):
            x = env.state_features((h, l))
            q = agent.q_values(x)
            value[h, l] = float(np.max(q))
            policy[h, l] = int(np.argmax(q))
    return {"value": value, "policy": policy, "r_bar": agent.r_bar}
