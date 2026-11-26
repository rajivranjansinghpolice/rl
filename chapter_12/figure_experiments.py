"""Illustrative reproductions of Figures 12.3, 12.6, 12.8, 12.10, 12.11, 12.14.

Sutton & Barto, 2nd ed., Chapter 12 (Eligibility Traces). All figures are
reduced-scale demonstrations: the book averages over 100 repetitions and uses
full-scale tasks, whereas here we use fewer repetitions and share the same
pre-sampled random walks across all parameter settings (as the book does).

The three random-walk figures use the 19-state walk of Example 7.1 with the
same sampled walks for every (lambda, alpha) pair:

    fig 12.3  offline lambda-return algorithm
    fig 12.6  TD(lambda) vs offline lambda-return
    fig 12.8  online lambda-return (true online TD(lambda)) vs offline

The Mountain Car panels (figs 12.10/12.11) use the shared environment and the
tile-coded linear Sarsa agent of chapter_10, extended with eligibility traces.
Fig. 12.14 is a four-panel qualitative summary of lambda's effect; the book's
four tasks are not all available here, so the extra panels are illustrative.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from chapter_07.solution import _sample_walks
from chapter_10.solution import mountain_car_features
from rl_envs.mountain_car import MountainCarEnv
from rl_envs.random_walk import RandomWalkEnv

LAMBDAS = (0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0)
ALPHAS = tuple(np.arange(0.0, 1.01, 0.1))


def _walk_tasks(n_states: int = 19) -> tuple[list, dict, tuple]:
    env = RandomWalkEnv(n_states=n_states, left_reward=-1.0, right_reward=1.0)
    true_values = env.true_values()
    return list(true_values), true_values, tuple(true_values)


def _rms(values: dict, true_values: dict, states: tuple) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        diffs = np.array([values[s] - true_values[s] for s in states],
                         dtype=np.float64)
        if not np.all(np.isfinite(diffs)):
            return float("inf")  # TD(lambda) can diverge at large alpha
        return float(np.sqrt(np.mean(diffs ** 2)))


def _offline_lambda_sweep(values: dict, states: tuple, state_seq: list,
                          rewards: list[float], lam: float, alpha: float,
                          gamma: float, true_values: dict) -> float:
    old = dict(values)
    g_next = 0.0
    targets: list[float] = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        s = state_seq[t]
        boot = old.get(state_seq[t + 1], 0.0)
        g = rewards[t] + gamma * ((1.0 - lam) * boot + lam * g_next)
        targets[t] = g  # one target per visit (not per state)
        g_next = g
    for t in range(len(rewards)):
        s = state_seq[t]
        values[s] += alpha * (targets[t] - old[s])
    return _rms(values, true_values, states)


@np.errstate(over="ignore", invalid="ignore")
def _td_lambda_sweep(values: dict, traces: dict, states: tuple,
                     state_seq: list, rewards: list[float], lam: float,
                     alpha: float, gamma: float, true_values: dict) -> float:
    """Online TD(lambda), accumulating traces (book box, Sec. 12.2)."""
    for t in range(len(rewards)):
        s = state_seq[t]
        nxt = state_seq[t + 1]
        boot = values.get(nxt, 0.0)
        delta = rewards[t] + gamma * boot - values[s]
        traces[s] += 1.0
        for st in states:
            values[st] += alpha * delta * traces[st]
            traces[st] *= gamma * lam
    return _rms(values, true_values, states)


@np.errstate(over="ignore", invalid="ignore")
def _true_online_sweep(values: dict, traces: dict, states: tuple,
                       state_seq: list, rewards: list[float], lam: float,
                       alpha: float, gamma: float, true_values: dict) -> float:
    """Tabular true online TD(lambda) (book Sec. 12.5, Eq. 12.15-12.16)."""
    for t in range(len(rewards)):
        s = state_seq[t]
        nxt = state_seq[t + 1]
        v_old = values[s]
        v_next = values.get(nxt, 0.0)
        delta = rewards[t] + gamma * v_next - v_old
        old_z_s = traces[s]
        for st in states:
            traces[st] *= gamma * lam
        traces[s] += 1.0 - alpha * gamma * lam * old_z_s
        for st in states:
            values[st] += alpha * (delta + v_old - values[st]) * traces[st]
        values[s] -= alpha * (v_old - values[s])
    return _rms(values, true_values, states)


def random_walk_lambda_curves(repeats: int = 10, n_episodes: int = 10,
                              seed: int = 0) -> dict:
    """RMS-error curves over alpha for offline, TD(lambda) and true online."""
    states, true_values, state_tuple = _walk_tasks()
    walks = _sample_walks(repeats, n_episodes, seed)
    n_alphas = len(ALPHAS)
    results = {name: {} for name in ("offline", "td", "true_online")}
    for lam in LAMBDAS:
        for name in results:
            results[name][lam] = np.zeros(n_alphas, dtype=np.float64)
    for lam in LAMBDAS:
        for ai, alpha in enumerate(ALPHAS):
            for name in results:
                total = 0.0
                for rep in walks:
                    values = dict.fromkeys(states, 0.0)
                    traces = dict.fromkeys(states, 0.0)
                    for state_seq, rewards in rep:
                        if name == "offline":
                            err = _offline_lambda_sweep(
                                values, state_tuple, state_seq, rewards,
                                lam, alpha, 1.0, true_values)
                        elif name == "td":
                            err = _td_lambda_sweep(
                                values, traces, state_tuple, state_seq, rewards,
                                lam, alpha, 1.0, true_values)
                        else:
                            err = _true_online_sweep(
                                values, traces, state_tuple, state_seq, rewards,
                                lam, alpha, 1.0, true_values)
                        total += err
                results[name][lam][ai] = total / (repeats * n_episodes)
    return {"alphas": list(ALPHAS), "lambdas": list(LAMBDAS), "curves": results}


def _sarsa_lambda_mountain_car(lam: float, trace_kind: str, episodes: int = 200,
                               seed: int = 0, alpha: float = 0.5,
                               max_steps: int = 200) -> list[float]:
    """Linear Sarsa(lambda) with accumulating or replacing traces."""
    rng = np.random.default_rng(seed)
    env = MountainCarEnv(seed=seed)
    dim = 8 * 8 * 8
    w = np.zeros((3, dim), dtype=np.float64)
    z = np.zeros((3, dim), dtype=np.float64)
    lengths = []
    for _ in range(episodes):
        state = env.reset()
        x = mountain_car_features(*state)
        q = w @ x
        action = int(np.argmax(q)) if rng.random() >= 0.1 else int(rng.integers(3))
        steps = 0
        while True:
            state_next, reward, done = env.step(action)
            steps += 1
            x_next = None if done else mountain_car_features(*state_next)
            if done:
                next_action = None
            else:
                q_next = w @ x_next
                next_action = int(np.argmax(q_next)) if rng.random() >= 0.1 \
                    else int(rng.integers(3))
            target = 0.0 if done else float((w @ x_next)[next_action])
            delta = reward + target - float((w @ x)[action])
            for a in range(3):
                z[a] *= 0.99 * lam
            if trace_kind == "replacing":
                z[action] = x.copy()
            else:
                z[action] += x
            w += (alpha / 8.0) * delta * z
            if done or steps >= max_steps:
                break
            state, x, action = state_next, x_next, next_action
        lengths.append(float(steps))
    return lengths


def mountain_car_lambda_curves(episodes: int = 200, seed: int = 0) -> dict:
    """Steps per episode for Sarsa(lambda) variants on Mountain Car."""
    variants = {
        "lambda=0": (0.0, "accumulating"),
        "lambda=0.8": (0.8, "accumulating"),
        "lambda=0.92 replacing": (0.92, "replacing"),
        "lambda=0.96 replacing": (0.96, "replacing"),
    }
    return {
        name: _sarsa_lambda_mountain_car(lam, kind, episodes=episodes, seed=seed)
        for name, (lam, kind) in variants.items()
    }
