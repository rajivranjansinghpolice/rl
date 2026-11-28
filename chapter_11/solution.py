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
# n states: value computed from its operands (book Chapter 11).
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
    """Phi (7 x 6). Outer states get indicator e_i; center gets (2,...,2).


    Returns:
        baird features as defined in Section 11.2.

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
    # carry out the baird features step (book Chapter 11).
    # phi: zero-filled array (book Chapter 11).
    phi = np.zeros((N_STATES, N_OUTER), dtype=np.float64)
    phi[CENTER, :] = 2.0
    # Iterate over i: one pass per index (book algorithm).
    for i in range(N_OUTER):
        phi[i + 1, i] = 1.0
    return phi


def baird_transition_probs() -> dict[tuple[int, int], list[tuple[int, float]]]:
    """Transitions P(s' | s, a) for the hub star MDP.


    Returns:
        baird transition probs as defined in Section 11.2.

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
    # t: dict[tuple[int, int], list[tuple[int, float]]] attribute/state used by the algorithm ...
    t: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for a in (ACTION_A, ACTION_B):
        # center: uniform outer regardless of action
        t[(CENTER, a)] = [(i + 1, 1.0 / N_OUTER) for i in range(N_OUTER)]
        for i in range(N_OUTER):
            # s: current state index (book Chapter 11).
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

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        action_probs: See Section 11.2 and the implementation for its role.

    Returns:
        transition matrix as defined in Section 11.2."""
    # internal helper for the transition matrix step (book Chapter 11).
    # t: result of baird_transition_probs(...) (book Chapter 11).
    t = baird_transition_probs()
    # p: zero-filled array (book Chapter 11).
    P = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    # Iterate over s: one pass per index (book algorithm).
    for s in range(N_STATES):
        # Iterate over a: each item in (ACTION_A, ACTION_B) (book algorithm).
        for a in (ACTION_A, ACTION_B):
            # Iterate over (sp, p): each item in t[(s, a)] (book algorithm).
            for sp, p in t[(s, a)]:
                # P[s, sp]: add value computed from its operands into the running value (book ...
                P[s, sp] += action_probs[a] * p
    return P


def behavior_stationary() -> np.ndarray:
    """Stationary distribution mu of the behavior chain (exact eigen-solve).

    Enumerates the 7-state behaviour Markov chain and returns its stationary
    distribution via the left eigenvector of eigenvalue 1.


    Returns:
        behavior stationary as defined in Section 11.2.

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
    # p: result of _transition_matrix(...) (book Chapter 11).
    P = _transition_matrix(BEHAVIOR)
    # stationary distribution: solve mu P = mu, sum(mu) = 1
    A = np.vstack([P.T - np.eye(N_STATES), np.ones(N_STATES)])
    # b: result of np.concatenate(...) (book Chapter 11).
    b = np.concatenate([np.zeros(N_STATES), [1.0]])
    # (mu, *_): result of np.linalg.lstsq(...) (book Chapter 11).
    mu, *_ = np.linalg.lstsq(A, b, rcond=None)
    # mu: result of np.clip(...) (book Chapter 11).
    mu = np.clip(mu, 0.0, None)
    return mu / mu.sum()


def target_transition_matrix() -> np.ndarray:
    """7x7 transition matrix under the target policy (always A).


    Returns:
        target transition matrix as defined in Section 11.2.

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
    # carry out the target transition matrix step (book Chapter 11).
    return _transition_matrix(TARGET)


def _projector_matrices(gamma: float = 0.99):
    """Compute phi, D, P, B with correct associativity (all matmuls).

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        gamma: See Section 11.2 and the implementation for its role."""
    # internal helper for the projector matrices step (book Chapter 11).
    # phi: result of baird_features(...) (book Chapter 11).
    phi = baird_features()
    # mu: result of behavior_stationary(...) (book Chapter 11).
    mu = behavior_stationary()
    # d: result of np.diag(...) (book Chapter 11).
    D = np.diag(mu)
    # p: result of target_transition_matrix(...) (book Chapter 11).
    P = target_transition_matrix()
    # inner: result of np.linalg.inv(...) (book Chapter 11).
    inner = np.linalg.inv(phi.T @ D @ phi)
    # b: value computed from its operands (book Chapter 11).
    B = gamma * phi @ inner @ phi.T @ D @ P @ phi
    return phi, D, P, B


def projected_bellman_solution(gamma: float = 0.99) -> np.ndarray:
    """Solve the linear projected Bellman equation exactly (r = 0).

    w = (Phi^T D (I - gamma P^pi) Phi)^{-1} Phi^T D r, with r = 0, so the
    solution is w = 0 (the true value function of the star is identically 0).
    Returns the solved vector (which should be ~0).

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        gamma: See Section 11.2 and the implementation for its role.

    Returns:
        projected bellman solution as defined in Section 11.2."""
    # carry out the projected bellman solution step (book Chapter 11).
    # phi: result of baird_features(...) (book Chapter 11).
    phi = baird_features()
    # mu: result of behavior_stationary(...) (book Chapter 11).
    mu = behavior_stationary()
    # d: result of np.diag(...) (book Chapter 11).
    D = np.diag(mu)
    # p: result of target_transition_matrix(...) (book Chapter 11).
    P = target_transition_matrix()
    # m: value computed from its operands (book Chapter 11).
    M = phi.T @ D @ (np.eye(N_STATES) - gamma * P) @ phi
    # r: zero-filled array (book Chapter 11).
    r = np.zeros(N_STATES)
    # rhs: value computed from its operands (book Chapter 11).
    rhs = phi.T @ D @ r
    return np.linalg.solve(M, rhs)


def mspbe(weights: np.ndarray, gamma: float = 0.99) -> float:
    """Mean Squared Projected Bellman Error for linear v = Phi w.

    MSPBE(w) = (B w - Phi w)^T D (B w - Phi w) with
    B = gamma * Phi (Phi^T D Phi)^{-1} Phi^T D P^pi Phi.

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        weights: See Section 11.2 and the implementation for its role.
        gamma: See Section 11.2 and the implementation for its role.

    Returns:
        mspbe as defined in Section 11.2."""
    # carry out the mspbe step (book Chapter 11).
    phi, D, _, B = _projector_matrices(gamma)
    # residual: value computed from its operands (book Chapter 11).
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

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        n_steps: See Section 11.2 and the implementation for its role.
        lr: See Section 11.2 and the implementation for its role.
        gamma: See Section 11.2 and the implementation for its role.
        seed: See Section 11.2 and the implementation for its role.

    Returns:
        mspbe gradient descent as defined in Section 11.2."""
    # (phi, D, _, B): result of _projector_matrices(...) (book Chapter 11).
    phi, D, _, B = _projector_matrices(gamma)
    # q: value computed from its operands (book Chapter 11).
    Q = (B - phi).T @ D @ (B - phi)
    Q = (Q + Q.T) / 2.0  # numerical symmetric cleanup

    if lr is None:
        # lmax: maximum of the values (book Chapter 11).
        lmax = np.linalg.eigvalsh(Q).max()
        # lr: value chosen conditionally (book Chapter 11).
        lr = 1.0 / lmax if lmax > 0 else 1.0

    # rng: seeded random generator so runs are reproducible (book Chapter 11).
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=N_OUTER)  # arbitrary start; GD returns to 0

    # weights traj: list[np.ndarray] attribute/state used by the algorithm (book Chapter 11).
    weights_traj: list[np.ndarray] = []
    # mspbe traj: list[float] attribute/state used by the algorithm (book Chapter 11).
    mspbe_traj: list[float] = []
    # Iterate over _: one pass per time step (book algorithm).
    for _ in range(n_steps):
        # grad: value computed from its operands (book Chapter 11).
        grad = 2.0 * Q @ w
        # w: value computed from its operands (book Chapter 11).
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

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""

    # alpha: float attribute/state used by the algorithm (book Chapter 11).
    alpha: float = 0.01
    # gamma: discount factor for future rewards (book Chapter 11).
    gamma: float = 0.99
    # seed: int | None attribute/state used by the algorithm (book Chapter 11).
    seed: int | None = None
    # weights: np.ndarray | None attribute/state used by the algorithm (book Chapter 11).
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        """post init (reference implementation).

        Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
        # derive cached state after dataclass initialization (book Chapter 11).
        if self.weights is None:
            self.weights = np.zeros(N_OUTER, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)
    def value(self, state: int) -> float:
        """value (reference implementation).


        Args:
            state: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
        # carry out the value step (book Chapter 11).
        return float(baird_features()[state] @ self.weights)

    def step(self, state: int) -> int:
        """Sample a behavior-policy action.

        Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

        Args:
            state: See Section 11.2 and the implementation for its role.

        Returns:
            step as defined in Section 11.2."""
        # advance one step and return the observed transition (book Chapter 11).
        return ACTION_B if self.rng.uniform() < BEHAVIOR[ACTION_B] else ACTION_A

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """update (reference implementation).


        Args:
            state: See the relevant book section and function body.
            action: See the relevant book section and function body.
            reward: See the relevant book section and function body.
            next_state: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11)."""
        # apply the learning update rule to the current estimates (book Chapter 11).
        # phi: element selected from the indexed collection (book Chapter 11).
        phi = baird_features()[state]
        # phi next: element selected from the indexed collection (book Chapter 11).
        phi_next = baird_features()[next_state]
        # v: floating-point conversion (book Chapter 11).
        v = float(phi @ self.weights)
        # v next: floating-point conversion (book Chapter 11).
        v_next = float(phi_next @ self.weights)
        # rho: value computed from its operands (book Chapter 11).
        rho = TARGET[action] / BEHAVIOR[action]
        # delta: TD error or update difference for this step (book Chapter 11).
        delta = reward + self.gamma * v_next - v
        self.weights = self.weights + self.alpha * rho * delta * phi

    def run(self, n_steps: int) -> np.ndarray:
        """Return the weight trajectory (states visited from a uniform start).

        Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

        Args:
            n_steps: See Section 11.2 and the implementation for its role.

        Returns:
            run as defined in Section 11.2."""
        # carry out the run step (book Chapter 11).
        # t: result of baird_transition_probs(...) (book Chapter 11).
        t = baird_transition_probs()
        # traj: collection of the listed elements (book Chapter 11).
        traj = []
        # state: current state of the environment or MDP (book Chapter 11).
        state = int(self.rng.integers(N_STATES))
        # Iterate over _: one pass per time step (book algorithm).
        for _ in range(n_steps):
            # action: action selected for the current step (book Chapter 11).
            action = self.step(state)
            # (outcomes, probs): result of zip(...) (book Chapter 11).
            outcomes, probs = zip(*t[(state, action)])
            # idx: random choice from the given options (book Chapter 11).
            idx = self.rng.choice(len(outcomes), p=probs)
            # next state: element selected from the indexed collection (book Chapter 11).
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

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        save_path: See Section 11.2 and the implementation for its role.
        show: See Section 11.2 and the implementation for its role.
        n_steps: See Section 11.2 and the implementation for its role.
        seed: See Section 11.2 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 11).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # agent: result of SemiGradientOffPolicyAgent(...) (book Chapter 11).
    agent = SemiGradientOffPolicyAgent(alpha=0.005, gamma=0.99, seed=seed)
    agent.weights = np.random.default_rng(0).normal(0.0, 1.0, size=N_OUTER)
    # traj: result of agent.run(...) (book Chapter 11).
    traj = agent.run(n_steps)
    # norms: result of np.linalg.norm(...) (book Chapter 11).
    norms = np.linalg.norm(traj, axis=1)

    # (fig, ax): result of plt.subplots(...) (book Chapter 11).
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

    Book: Sutton & Barto (2nd ed.), Section 11.2 (Chapter 11).

    Args:
        save_path: See Section 11.2 and the implementation for its role.
        show: See Section 11.2 and the implementation for its role.
        n_steps: See Section 11.2 and the implementation for its role.
        seed: See Section 11.2 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 11).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (weights_traj, mspbe_traj): result of mspbe_gradient_descent(...) (book Chapter 11).
    weights_traj, mspbe_traj = mspbe_gradient_descent(n_steps=n_steps, lr=None, seed=seed)
    # weight norms: collection built by comprehension (book Chapter 11).
    weight_norms = [float(np.linalg.norm(w)) for w in weights_traj]

    # (fig, (ax1, ax2)): result of plt.subplots(...) (book Chapter 11).
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
