"""Illustrative reproductions of Chapter 9 book figures (S&B 2nd ed.).

Verified captions from the local PDF:

    Fig. 9.1  state aggregation on the 1000-state random walk, gradient MC
    Fig. 9.2  state aggregation on the 1000-state random walk, bootstrapping
              (semi-gradient TD, asymptotic values + n-step performance)
    Fig. 9.5  Fourier basis vs polynomials on the 1000-state random walk
              (gradient Monte Carlo learning curves)
    Fig. 9.10 single tiling vs multiple tilings on the 1000-state random walk

All experiments here are reduced-scale (100 states instead of 1000, fewer
episodes/repetitions) but use the same task structure and feature families.
The random walk is implemented with integer states for speed; it is the same
MDP as `rl_envs.RandomWalkEnv` (equiprobable left/right, terminal rewards 0/1,
gamma = 1) with `n_states` states.
"""

from __future__ import annotations

import numpy as np

from chapter_09.solution import fourier_features


def sample_episodes(
    n_states: int, n_episodes: int, seed: int
) -> list[tuple[list[int], float]]:
    """Sample walks as (states, terminal_reward) pairs (reward 0 or 1)."""
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        state = n_states // 2
        path = [state]
        while True:
            state += 1 if rng.random() < 0.5 else -1
            if state == 0 or state == n_states + 1:
                episodes.append((path, 1.0 if state > n_states else 0.0))
                break  # terminal (reward 0 or 1 respectively)
            path.append(state)
    return episodes


def true_values(n_states: int) -> np.ndarray:
    """v(s) = s / (n_states + 1), s = 1..n_states (linear, gamma = 1)."""
    return np.arange(1, n_states + 1, dtype=np.float64) / (n_states + 1)


def state_distribution(episodes: list[list[int]], n_states: int) -> np.ndarray:
    """Empirical state- visitation distribution mu(s), normalised."""
    counts = np.zeros(n_states + 2, dtype=np.float64)
    for path, _reward in episodes:
        for s in path:
            counts[s] += 1.0
    total = counts.sum()
    return counts[1:n_states + 1] / total if total else counts[1:n_states + 1]


def aggregation_features(states: list[int], n_states: int, n_groups: int) -> np.ndarray:
    """One-hot state-aggregation features (book Fig. 9.1/9.2)."""
    groups = np.minimum(
        (np.asarray(states).astype(int) * n_groups) // n_states, n_groups - 1)
    feat = np.zeros((len(states), n_groups), dtype=np.float64)
    feat[np.arange(len(states)), groups] = 1.0
    return feat


def polynomial_features(x: np.ndarray, order: int) -> np.ndarray:
    """Polynomial basis [1, x, x^2, ..., x^order] for x in (0, 1)."""
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x ** k for k in range(order + 1)], axis=-1)


def tiling_features_1d(x: np.ndarray, n_tilings: int,
                       tiles_per_tiling: int) -> np.ndarray:
    """One-dimensional tile coding with uniformly offset tilings (Fig. 9.10)."""
    x = np.asarray(x, dtype=np.float64)
    feat = np.zeros((*x.shape, n_tilings, tiles_per_tiling), dtype=np.float64)
    width = 1.0 / tiles_per_tiling
    for t in range(n_tilings):
        shifted = np.clip(x + t * width / n_tilings, 0.0, 1.0 - 1e-12)
        idx = np.floor(shifted / width).astype(int)
        feat[np.arange(len(x)), t, idx] = 1.0
    return feat.reshape(*x.shape, n_tilings * tiles_per_tiling)


def gradient_mc_curve(
    episodes: list[list[int]],
    n_states: int,
    features,
    alpha: float,
    eval_every: int = 1,
) -> np.ndarray:
    """RMSVE after each episode for constant-alpha Monte Carlo prediction."""
    v_true = true_values(n_states)
    x_grid = (np.arange(1, n_states + 1) / (n_states + 1))
    eval_feat = features(x_grid)
    w = np.zeros(eval_feat.shape[1], dtype=np.float64)
    curve = []
    for ep, (path, reward) in enumerate(episodes):
        # Returns: G_t = reward * gamma^(T-1-t) with gamma = 1 -> all G_t = reward
        # for this undiscounted episodic task (only the terminal step has a reward).
        states = np.asarray(path[:], dtype=np.float64) / (n_states + 1)
        feat = features(states)
        w += alpha * (feat.T @ (reward - feat @ w))
        if (ep + 1) % eval_every == 0:
            pred = eval_feat @ w
            curve.append(float(np.sqrt(np.mean((pred - v_true) ** 2))))
    return np.asarray(curve)


def n_step_td_curve(
    episodes: list[list[int]],
    n_states: int,
    features,
    alpha: float,
    n: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """RMSVE after each episode for n-step semi-gradient TD prediction."""
    v_true = true_values(n_states)
    x_grid = (np.arange(1, n_states + 1) / (n_states + 1))
    eval_feat = features(x_grid)
    w = np.zeros(eval_feat.shape[1], dtype=np.float64)
    curve = []
    for path, reward in episodes:
        states = np.asarray(path) / (n_states + 1)
        feat = features(states)
        T = len(path)
        g = np.zeros(T, dtype=np.float64)
        for tau in range(T):
            end = min(tau + n, T - 1)
            g[tau] = (reward if end == T - 1 else
                      (gamma ** (end - tau)) * float(feat[end] @ w))
        w += alpha * np.sum((g - feat @ w)[:, None] * feat, axis=0)
        pred = eval_feat @ w
        curve.append(float(np.sqrt(np.mean((pred - v_true) ** 2))))
    return np.asarray(curve)


def figure_9_1_data(n_states: int = 100, episodes: int = 30,
                    seed: int = 0, n_groups: int = 10, alpha: float = 1e-4) -> dict:
    """Gradient MC with state aggregation (Fig. 9.1, reduced scale)."""
    walks = sample_episodes(n_states, episodes, seed)
    features = lambda x: aggregation_features(np.asarray(x), n_states, n_groups)
    w = np.zeros(n_groups, dtype=np.float64)
    for path, reward in walks:
        feat = features(np.asarray(path, dtype=np.float64))
        w += alpha * np.sum((reward - feat @ w)[:, None] * feat, axis=0)
    # Map group weights back to state values.
    groups = np.minimum((np.arange(1, n_states + 1) * n_groups) // n_states,
                        n_groups - 1)
    approx = w[groups]
    return {
        "states": np.arange(1, n_states + 1),
        "true": true_values(n_states),
        "approx": approx,
        "mu": state_distribution(walks, n_states),
        "n_states": n_states,
    }


def figure_9_2_data(n_states: int = 100, episodes: int = 12,
                    repeats: int = 3, seed: int = 0,
                    ns: tuple[int, ...] = (1, 2, 4, 8, 16),
                    alphas: tuple[float, ...] = (1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3)) -> dict:
    """State-aggregation TD: asymptotic values + best alpha per n (Fig. 9.2)."""
    features = lambda x: aggregation_features(np.asarray(x), n_states, 10)
    walks = [sample_episodes(n_states, episodes, seed + r * 977)
             for r in range(repeats)]
    curves = {}
    for n in ns:
        vals = []
        for alpha in alphas:
            errs = []
            for rep in walks:
                c = n_step_td_curve(rep, n_states, features, alpha, n)
                errs.append(np.mean(c[-max(1, len(c) // 2):]))
            vals.append(float(np.mean(errs)))
        curves[n] = np.asarray(vals)
    # Asymptotic values for the best (n, alpha) pair.
    best_n = min(curves, key=lambda k: curves[k].min())
    best_alpha = alphas[int(curves[best_n].argmin())]
    w = np.zeros(10, dtype=np.float64)
    for path, _reward in walks[0]:
        states = np.asarray(path) / (n_states + 1)
        feat = features(states)
        reward = 1.0 if path[-1] >= n_states else 0.0
        T = len(path)
        g = np.zeros(T)
        for tau in range(T):
            end = min(tau + best_n, T - 1)
            g[tau] = (reward if end == T - 1 else
                      float(feat[end] @ w))
        w += best_alpha * np.sum((g - feat @ w)[:, None] * feat, axis=0)
    groups = np.minimum((np.arange(1, n_states + 1) * 10) // n_states, 9)
    return {
        "alphas": list(alphas), "ns": list(ns), "curves": curves,
        "states": np.arange(1, n_states + 1), "true": true_values(n_states),
        "approx": w[groups], "best_n": best_n, "best_alpha": best_alpha,
        "n_states": n_states,
    }


def figure_9_5_data(n_states: int = 100, episodes: int = 15,
                    seed: int = 0) -> dict:
    """Fourier vs polynomial bases (Fig. 9.5, reduced scale)."""
    walks = sample_episodes(n_states, episodes, seed)
    out = {}
    for family in ("fourier", "polynomial"):
        for order in (1, 2, 3, 5):
            if family == "fourier":
                features = lambda x: np.array([fourier_features(float(v), order)
                                               for v in np.asarray(x).ravel()])
            else:
                features = lambda x: polynomial_features(x, order)
            out[(family, order)] = gradient_mc_curve(walks, n_states, features,
                                                     alpha=1e-4)
    return {"curves": out, "episodes": episodes, "n_states": n_states}


def figure_9_10_data(n_states: int = 100, episodes: int = 15,
                     seed: int = 0) -> dict:
    """Single tiling vs multiple tilings (Fig. 9.10, reduced scale)."""
    walks = sample_episodes(n_states, episodes, seed)
    out = {}
    for n_tilings in (1, 2, 4, 8):
        features = lambda x: tiling_features_1d(x, n_tilings, 8)
        out[n_tilings] = gradient_mc_curve(walks, n_states, features,
                                           alpha=2e-5 / n_tilings)
    return {"curves": out, "episodes": episodes, "n_states": n_states}
