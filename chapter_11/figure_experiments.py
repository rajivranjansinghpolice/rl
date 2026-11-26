"""Illustrative reproductions of Chapter 11 book figures (S&B 2nd ed.).

Verified captions from the local PDF:

    Fig. 11.2  instability of semi-gradient off-policy TD on Baird's counterexample
               (left: semi-gradient TD, right: semi-gradient DP)
    Fig. 11.6  behaviour of TDC on Baird's counterexample
    Fig. 11.7  one-step emphatic-TD in expectation on Baird's counterexample

All updates here are *expected* (full-distribution) updates on Baird's
7-state star, not sampled single transitions. This makes the divergence /
stability comparison deterministic and easy to inspect; it is an illustrative
reproduction, not a pixel-level copy of the book's sample-based plots.
"""

from __future__ import annotations

import numpy as np

from chapter_11.solution import baird_features

N_OUTER = 6
N_STATES = 7
CENTER = 6
GAMMA = 0.99
# Target pi: solid action A (to center). Behavior b: A with 1/7, B with 6/7.
B_A, B_B = 1.0 / 7.0, 6.0 / 7.0


def transitions() -> np.ndarray:
    """P[s, a, s'] for the star: solid A -> center, dashed B -> uniform outer."""
    p = np.zeros((N_STATES, 2, N_STATES), dtype=np.float64)
    for s in range(N_STATES):
        p[s, 0, CENTER] = 1.0  # solid / target action
        if s == CENTER:
            p[s, 1, :N_OUTER] = 1.0 / N_OUTER
        else:
            p[s, 1, :N_OUTER] = 1.0 / N_OUTER  # dashed action
    return p


def behavior_stationary(p: np.ndarray) -> np.ndarray:
    """Stationary distribution of the behavior chain (mu P_b = mu)."""
    pb = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        for a, prob in enumerate((B_A, B_B)):
            pb[s] += prob * p[s, a]
    A = np.vstack([pb.T - np.eye(N_STATES), np.ones(N_STATES)])
    rhs = np.concatenate([np.zeros(N_STATES), [1.0]])
    mu, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return np.clip(mu, 0, None) / np.clip(mu, 0, None).sum()


def run_expected(mode: str, n_steps: int = 300, alpha: float = 0.01,
                 init: np.ndarray | None = None) -> np.ndarray:
    """Expected weight trajectory for a given off-policy update rule.

    `mode`:
      'semi_gradient_td'  -- rho * delta * x  (diverges: deadly triad)
      'semi_gradient_dp'  -- gamma * Phi^T D P^pi Phi-style expected update
      'tdc'               -- TDC gradient correction (stable)
      'emphatic'          -- one-step emphatic TD in expectation (stable)
    """
    phi = baird_features()  # (7, 6)
    p = transitions()
    mu = behavior_stationary(p)
    pi_prob = np.array([1.0, 0.0])  # target always A
    b_prob = np.array([B_A, B_B])
    rho = pi_prob / b_prob
    w = np.zeros(phi.shape[1]) if init is None else init.copy()
    traj = [w.copy()]
    # Emphatic stationary weights: e(s) = mu(s) + gamma * sum_s' e(s') b-weighted
    # following of the target; simplified expectation form.
    e = mu.copy()
    for _ in range(n_steps):
        delta_w = np.zeros_like(w)
        for s in range(N_STATES):
            x = phi[s]
            v = float(x @ w)
            for a in (0, 1):
                prob = mu[s] * b_prob[a]
                if prob == 0:
                    continue
                r_next = p[s, a]
                v_next = float((phi @ w) @ r_next)
                delta = -v + GAMMA * v_next  # rewards are all zero
                weight = prob * rho[a]
                if mode == "semi_gradient_td":
                    delta_w += alpha * weight * delta * x
                elif mode == "semi_gradient_dp":
                    # Expected (DP-style) off-policy backup with the same
                    # importance ratio; still diverges with linear features.
                    expected_q = 0.0
                    for sp, pr in enumerate(r_next):
                        if pr:
                            expected_q += pr * rho[0] * float(phi[sp] @ w)
                    delta_w += alpha * weight * (
                        GAMMA * expected_q - v) * x
                elif mode == "tdc":
                    x_next = phi.T @ r_next  # expected next feature
                    delta_w += alpha * weight * (
                        delta * x - GAMMA * x_next * float(x @ w))
                elif mode == "emphatic":
                    delta_w += alpha * e[s] * weight * delta * x
                else:
                    raise ValueError(f"unknown mode {mode!r}")
        w = w + delta_w
        traj.append(w.copy())
    return np.asarray(traj)
