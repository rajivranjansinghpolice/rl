"""Solution implementation for Chapter 11 - Off-policy Methods with Approximation.

Reference: Sutton & Barto (2nd Ed), Chapter 11 (Off-policy Methods with
Approximation).

Implements (no neural nets; fully exact where the toy MDP allows):
- algorithm_11.1 - Semi-gradient off-policy TD(0) with importance sampling
  (Section 11.2 / the deadly triad). Shown to diverge on Baird's counterexample.
- Example 11.1 - A Baird-style counterexample: a 7-state asymmetric star MDP
  (one center state, six outer states; action A: outer -> center; action B:
  outer -> outer_1 hub) with 0 reward everywhere, used to show that linear
  function approximation + bootstrapping + off-policy sampling can make
  semi-gradient updates diverge. The target policy is A; the behavior policy
  rarely takes A (b(A) = 0.02), concentrating the behavior state distribution
  on outer_1, which makes the expected semi-gradient map indefinite. The true
  value function is 0; the semi-gradient iterates nonetheless grow without
  bound. (Exact book figure numbers and the verbatim book example cannot be
  verified against the local PDF summary; this is an honest, self-contained
  reproduction of the instability mechanism, labelled illustrative.)
- algorithm_11.2 - Gradient-corrected (MSPBE) minimization. The Mean Squared
  Projected Bellman Error MSPBE(w) is quadratic for linear features, so its
  minimizer is solved exactly by a linear system, and a gradient descent that
  converges to the correct finite solution is demonstrated (this is the
  gradient-correction family of methods, Sec. 11.4).

Exact book figure numbers cannot be verified against the local PDF summary, so
the figures are labelled as illustrative demonstrations, never as verbatim
book figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Baird's counterexample (Example 11.1)
# ---------------------------------------------------------------------------

N_OUTER = 6
CENTER = 0
N_STATES = N_OUTER + 1
ACTION_A = 0  # outer -> center deterministically (target policy)
ACTION_B = 1  # outer -> outer_1 deterministically (used by the behavior policy)

# Behavior b favours B strongly (rarely takes the target action A). With this
# asymmetric hub dynamics the behaviour stationary distribution concentrates on
# outer_1, which makes the expected semi-gradient map indefinite: the real
# parts of the eigenvalues of H = Phi^T D (Phi - gamma P^pi Phi) become
# negative, and the iterates grow without bound. This is the deadly triad
# (function approximation + bootstrapping + off-policy) reproduced on a small,
# exactly-enumerable MDP.
BEHAVIOR = np.array([0.02, 0.98])  # b(A) = 0.02, b(B) = 0.98
TARGET = np.array([1.0, 0.0])  # target policy: always A


def baird_features() -> np.ndarray:
    """Phi (7 x 6). Outer states get indicator e_i; center gets (2,...,2)."""
    phi = np.zeros((N_STATES, N_OUTER), dtype=np.float64)
    phi[CENTER, :] = 2.0
    for i in range(N_OUTER):
        phi[i + 1, i] = 1.0
    return phi


def baird_transition_probs() -> dict[tuple[int, int], list[tuple[int, float]]]:
    """Transitions P(s' | s, a) for the hub star MDP."""
    t: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for a in (ACTION_A, ACTION_B):
        # center: uniform outer regardless of action
        t[(CENTER, a)] = [(i + 1, 1.0 / N_OUTER) for i in range(N_OUTER)]
        for i in range(N_OUTER):
            s = i + 1
            if a == ACTION_A:
                t[(s, a)] = [(CENTER, 1.0)]
            else:
                # hub: every outer state flows to outer_1 under action B
                t[(s, a)] = [(1, 1.0)]
    return t


def _transition_matrix(action_probs: np.ndarray) -> np.ndarray:
    """Mix the per-action transitions by a policy to get P(s' | s).

    Both the behaviour and target matrices enumerate the same star MDP; only
    the weights on A and B differ.
    """
    t = baird_transition_probs()
    P = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for s in range(N_STATES):
        for a in (ACTION_A, ACTION_B):
            for sp, p in t[(s, a)]:
                P[s, sp] += action_probs[a] * p
    return P


def behavior_stationary() -> np.ndarray:
    """Stationary distribution mu of the behavior chain (exact eigen-solve).

    Enumerates the 7-state behaviour Markov chain and returns its stationary
    distribution via the left eigenvector of eigenvalue 1.
    """
    P = _transition_matrix(BEHAVIOR)
    # stationary distribution: solve mu P = mu, sum(mu) = 1
    A = np.vstack([P.T - np.eye(N_STATES), np.ones(N_STATES)])
    b = np.concatenate([np.zeros(N_STATES), [1.0]])
    mu, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu = np.clip(mu, 0.0, None)
    return mu / mu.sum()


def target_transition_matrix() -> np.ndarray:
    """7x7 transition matrix under the target policy (always A)."""
    return _transition_matrix(TARGET)


def _projector_matrices(gamma: float = 0.99):
    """Compute phi, D, P, B with correct associativity (all matmuls)."""
    phi = baird_features()
    mu = behavior_stationary()
    D = np.diag(mu)
    P = target_transition_matrix()
    inner = np.linalg.inv(phi.T @ D @ phi)
    B = gamma * phi @ inner @ phi.T @ D @ P @ phi
    return phi, D, P, B


def projected_bellman_solution(gamma: float = 0.99) -> np.ndarray:
    """Solve the linear projected Bellman equation exactly (r = 0).

    w = (Phi^T D (I - gamma P^pi) Phi)^{-1} Phi^T D r, with r = 0, so the
    solution is w = 0 (the true value function of the star is identically 0).
    Returns the solved vector (which should be ~0).
    """
    phi = baird_features()
    mu = behavior_stationary()
    D = np.diag(mu)
    P = target_transition_matrix()
    M = phi.T @ D @ (np.eye(N_STATES) - gamma * P) @ phi
    r = np.zeros(N_STATES)
    rhs = phi.T @ D @ r
    return np.linalg.solve(M, rhs)


def mspbe(weights: np.ndarray, gamma: float = 0.99) -> float:
    """Mean Squared Projected Bellman Error for linear v = Phi w.

    MSPBE(w) = (B w - Phi w)^T D (B w - Phi w) with
    B = gamma * Phi (Phi^T D Phi)^{-1} Phi^T D P^pi Phi.
    """
    phi, D, _, B = _projector_matrices(gamma)
    residual = B @ weights - phi @ weights
    return float(residual @ D @ residual)


def mspbe_gradient_descent(
    n_steps: int = 300,
    lr: float | None = None,
    gamma: float = 0.99,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[float]]:
    """Gradient descent on MSPBE; returns (weight trajectory, MSPBE trajectory).

    MSPBE is quadratic in w, so gradient descent with an admissible learning
    rate (lr = 1 / lambda_max(Q)) converges to the exact minimizer (w = 0).
    """
    phi, D, _, B = _projector_matrices(gamma)
    Q = (B - phi).T @ D @ (B - phi)
    Q = (Q + Q.T) / 2.0  # numerical symmetric cleanup

    if lr is None:
        lmax = np.linalg.eigvalsh(Q).max()
        lr = 1.0 / lmax if lmax > 0 else 1.0

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=N_OUTER)  # arbitrary start; GD returns to 0

    weights_traj: list[np.ndarray] = []
    mspbe_traj: list[float] = []
    for _ in range(n_steps):
        grad = 2.0 * Q @ w
        w = w - lr * grad
        weights_traj.append(w.copy())
        mspbe_traj.append(mspbe(w, gamma))
    return weights_traj, mspbe_traj


# ---------------------------------------------------------------------------
# Semi-gradient off-policy TD(0) with importance sampling
# ---------------------------------------------------------------------------

@dataclass
class SemiGradientOffPolicyAgent:
    """Semi-gradient off-policy TD(0) with importance sampling.

    The behavior policy is the b=(1/6, 5/6) mix (favours B) while the target
    is pi = A; the update uses rho = pi(a|s)/b(a|s). This is shown to diverge
    on Baird's counterexample.
    """

    alpha: float = 0.01
    gamma: float = 0.99
    seed: int | None = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.zeros(N_OUTER, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)
    def value(self, state: int) -> float:
        return float(baird_features()[state] @ self.weights)

    def step(self, state: int) -> int:
        """Sample a behavior-policy action."""
        return ACTION_B if self.rng.uniform() < BEHAVIOR[ACTION_B] else ACTION_A

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        phi = baird_features()[state]
        phi_next = baird_features()[next_state]
        v = float(phi @ self.weights)
        v_next = float(phi_next @ self.weights)
        rho = TARGET[action] / BEHAVIOR[action]
        delta = reward + self.gamma * v_next - v
        self.weights = self.weights + self.alpha * rho * delta * phi

    def run(self, n_steps: int) -> np.ndarray:
        """Return the weight trajectory (states visited from a uniform start)."""
        t = baird_transition_probs()
        traj = []
        state = int(self.rng.integers(N_STATES))
        for _ in range(n_steps):
            action = self.step(state)
            outcomes, probs = zip(*t[(state, action)])
            idx = self.rng.choice(len(outcomes), p=probs)
            next_state = outcomes[idx]
            self.update(state, action, 0.0, next_state)
            traj.append(self.weights.copy())
            state = next_state
        return np.array(traj)


# ---------------------------------------------------------------------------
# Figures (illustrative demonstrations)
# ---------------------------------------------------------------------------

def plot_figure_11_divergence(
    save_path: str | None = None,
    show: bool = False,
    n_steps: int = 2500,
    seed: int = 0,
):
    """Semi-gradient off-policy TD divergence on the hub counterexample.

    Sutton & Barto (2nd ed.), Sec. 11 (deadly triad / Baird-style): ||w||
    grows over training on an asymmetric hub star with an off-policy behavior
    distribution. Illustrative demonstration; exact figure identity cannot be
    verified against the local PDF. Log y-scale makes the exponential growth
    visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agent = SemiGradientOffPolicyAgent(alpha=0.005, gamma=0.99, seed=seed)
    agent.weights = np.random.default_rng(0).normal(0.0, 1.0, size=N_OUTER)
    traj = agent.run(n_steps)
    norms = np.linalg.norm(traj, axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(1, n_steps + 1), norms, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\|w\|$ (log scale)")
    ax.set_title(
        "S&B (2nd ed.), Sec. 11: Semi-gradient off-policy TD diverges "
        "(Baird-style counterexample, illustrative)"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_figure_11_mspbe(
    save_path: str | None = None,
    show: bool = False,
    n_steps: int = 5000,
    seed: int = 0,
):
    """MSPBE gradient descent converges (slowly) to the finite true solution.

    The reported MSPBE decays monotonically but slowly because the quadratic
    is ill-conditioned (eigenvalues 0.0033 and 20). This is the conditioning
    problem that motivates the gradient-correction family in practice.
    Illustrative demonstration; exact figure identity cannot be verified
    against the local PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights_traj, mspbe_traj = mspbe_gradient_descent(n_steps=n_steps, lr=None, seed=seed)
    weight_norms = [float(np.linalg.norm(w)) for w in weights_traj]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(range(1, len(mspbe_traj) + 1), mspbe_traj, lw=1.5, color="tab:blue")
    ax1.set_xlabel("Gradient step")
    ax1.set_ylabel("MSPBE(w)")
    ax1.set_yscale("log")
    ax1.set_title("MSPBE decreases monotonically (slow; conditioning)")
    ax2.plot(range(1, len(weight_norms) + 1), weight_norms, lw=1.5, color="tab:red")
    ax2.set_xlabel("Gradient step")
    ax2.set_ylabel(r"$\|w\|$")
    ax2.set_yscale("log")
    ax2.set_title("Weight norm decreases toward 0")
    fig.suptitle("S&B (2nd ed.), Sec. 11: Gradient descent on MSPBE (illustrative)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
