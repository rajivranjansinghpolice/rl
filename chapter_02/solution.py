"""Solution implementation for Chapter 2 - Bandit Algorithms.

Reference: Sutton & Barto (2nd Ed), Chapter 2

Implements: Alg 2.1 (k-armed bandit), Alg 2.2 (action-value est.),
Alg 2.3 (epsilon-greedy), Alg 2.4 (UCB), Alg 2.5 (gradient bandit),
Ex 2.3 (non-stationary), Ex 2.4 (optimistic init), Ex 2.5 (UCB unc.),
Ex 2.6 (contextual bandits).
"""

from __future__ import annotations

import numpy as np


class BanditProblem:
    """k-armed bandit problem (Algorithm 2.1). Book Section 2.2.

    The environment has k actions, each with true expected reward q*(a).
    Reward for action a is drawn from N(q*(a), 1).
    """

    def __init__(self, k: int = 10, seed: int | None = None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star: np.ndarray = self.rng.normal(0.0, 1.0, k)

    def step(self, action: int) -> float:
        return float(self.rng.normal(self.q_star[action], 1.0))

    def best_action(self) -> int:
        return int(np.argmax(self.q_star))


class ActionValueEstimator:
    """Action-value estimation using sample-average (Algorithm 2.2). Book Sec 2.3.

    Maintains estimates Q(a) and counts N(a) for each action.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.Q: np.ndarray = np.zeros(k)
        self.N: np.ndarray = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng()

    def select_action(self) -> int:
        best_actions = np.flatnonzero(self.Q == self.Q.max())
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])


def select_epsilon_greedy(
    Q: np.ndarray, epsilon: float, rng: np.random.Generator | None = None
) -> int:
    """Select action using epsilon-greedy policy (Algorithm 2.3). Book Sec 2.3.

    Args:
        Q: Array of action values.
        epsilon: Exploration probability in [0, 1].
        rng: Random number generator.

    Returns:
        Selected action index.

    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    best_actions = np.flatnonzero(Q == Q.max())
    return int(rng.choice(best_actions))


def select_ucb(Q: np.ndarray, N: np.ndarray, t: int, c: float) -> int:
    """Select action using upper-confidence bound (Algorithm 2.4). Book Sec 2.4.

    Args:
        Q: Array of action value estimates.
        N: Array of action visit counts.
        t: Current time step (>= 1).
        c: Exploration parameter (> 0).

    Returns:
        Selected action index.

    """
    untried = np.flatnonzero(N == 0)
    if len(untried) > 0:
        return int(untried[0])
    ucb = Q + c * np.sqrt(np.log(t) / N)
    best_actions = np.flatnonzero(ucb == ucb.max())
    return int(np.random.default_rng().choice(best_actions))


def select_softmax(preferences: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Select action using softmax policy (Algorithm 2.5). Book Sec 2.5.

    Args:
        preferences: Action preference values H(a).
        rng: Random number generator.

    Returns:
        Selected action index.

    """
    if rng is None:
        rng = np.random.default_rng()
    shifted = preferences - preferences.max()
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()
    return int(rng.choice(len(probs), p=probs))


class GradientBanditAgent:
    """Gradient bandit algorithm (Algorithm 2.5). Book Section 2.5.

    Maintains action preferences H(a) and updates them using the gradient
    of expected reward.
    """

    def __init__(self, k: int, alpha: float = 0.1) -> None:
        self.k = k
        self.alpha = alpha
        self.H: np.ndarray = np.zeros(k)
        self.r_bar: float = 0.0
        self.rng = np.random.default_rng()

    def select_action(self) -> int:
        return select_softmax(self.H, self.rng)

    def update(self, action: int, reward: float) -> None:
        probs = np.zeros(self.k)
        shifted = self.H - self.H.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        delta = reward - self.r_bar
        self.H[action] += self.alpha * delta * (1.0 - probs[action])
        for a in range(self.k):
            if a != action:
                self.H[a] -= self.alpha * delta * probs[a]
        self.r_bar += self.alpha * (reward - self.r_bar)


def simulate_nonstationary(
    k: int = 10,
    steps: int = 1000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    true_reward_change: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate non-stationary bandit with constant step-size (Exercise 2.3).

    Book Exercise 2.3: Non-stationary Bandits.

    Args:
        k: Number of actions.
        steps: Number of steps to simulate.
        epsilon: Epsilon for epsilon-greedy.
        alpha: Constant step-size parameter.
        true_reward_change: Std dev of true reward changes per step.

    Returns:
        Tuple of (action_values, average_rewards).

    """
    rng = np.random.default_rng()
    q_star = rng.normal(0.0, 1.0, k)
    Q = np.zeros(k)
    rewards = np.zeros(steps)

    for t in range(steps):
        q_star += rng.normal(0.0, true_reward_change, k)
        if rng.random() < epsilon:
            action = int(rng.integers(k))
        else:
            action = int(np.argmax(Q))
        reward = float(rng.normal(q_star[action], 1.0))
        rewards[t] = reward
        Q[action] += alpha * (reward - Q[action])

    return Q, rewards


def simulate_optimistic_initial_values(
    k: int = 10,
    steps: int = 1000,
    initial_value: float = 5.0,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Simulate bandit with optimistic initial values (Exercise 2.4).

    Book Exercise 2.4: Optimistic Initial Values.

    Args:
        k: Number of actions.
        steps: Number of steps.
        initial_value: Optimistic initial Q value.
        epsilon: Epsilon (default 0 for pure greedy).

    Returns:
        Tuple of (final_Q, best_action_selection_rate_over_last_100).

    """
    rng = np.random.default_rng()
    bandit = BanditProblem(k, seed=0)
    Q = np.full(k, initial_value)
    N = np.zeros(k, dtype=int)
    best_counts = np.zeros(steps)

    for t in range(steps):
        if rng.random() < epsilon:
            action = int(rng.integers(k))
        else:
            action = int(np.argmax(Q))
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])
        if action == bandit.best_action():
            best_counts[t] = 1.0

    rate = best_counts[-100:].mean()
    return Q, rate


def simulate_ucb_uncertainty(
    k: int = 10,
    steps: int = 1000,
    c: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Simulate UCB with uncertainty parameterization (Exercise 2.5).

    Book Exercise 2.5: UCB with Uncertainty.

    Args:
        k: Number of actions.
        steps: Number of steps.
        c: Exploration parameter.
        seed: Random seed.

    Returns:
        Tuple of (final_Q, best_action_selection_rate).

    """
    np.random.default_rng(seed)
    bandit = BanditProblem(k, seed=seed)
    Q = np.zeros(k)
    N = np.zeros(k, dtype=int)
    best_counts = np.zeros(steps)

    for t in range(1, steps + 1):
        action = select_ucb(Q, N, t, c)
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])
        if action == bandit.best_action():
            best_counts[t - 1] = 1.0

    rate = best_counts[-100:].mean()
    return Q, rate


class ContextualBandit:
    """Contextual bandit algorithm (Exercise 2.6). Book Exercise 2.6.

    Maintains action preferences indexed by (context, action) pairs.
    """

    def __init__(self, num_contexts: int, k: int, alpha: float = 0.1) -> None:
        self.num_contexts = num_contexts
        self.k = k
        self.alpha = alpha
        self.H: np.ndarray = np.zeros((num_contexts, k))
        self.r_bar: np.ndarray = np.zeros(num_contexts)
        self.rng = np.random.default_rng()

    def select_action(self, context: int) -> int:
        return select_softmax(self.H[context], self.rng)

    def update(self, context: int, action: int, reward: float) -> None:
        probs = np.zeros(self.k)
        shifted = self.H[context] - self.H[context].max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        delta = reward - self.r_bar[context]
        self.H[context, action] += self.alpha * delta * (1.0 - probs[action])
        for a in range(self.k):
            if a != action:
                self.H[context, a] -= self.alpha * delta * probs[a]
        self.r_bar[context] += self.alpha * (reward - self.r_bar[context])


# ---------------------------------------------------------------------------
# Experiment runners and figure reproduction (Sutton & Barto, 2nd ed., Ch. 2)
# ---------------------------------------------------------------------------

def _run_bandit_bandit_episode(
    k: int,
    steps: int,
    seed: int,
    select_fn,
    update_fn,
    q_init: float = 0.0,
    step_size: float | None = None,
    track_optimal: bool = True,
):
    rng = np.random.default_rng(seed)
    bandit = BanditProblem(k=k, seed=seed)
    Q = np.full(k, q_init, dtype=np.float64)
    N = np.zeros(k, dtype=np.int64)
    rewards = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=np.float64) if track_optimal else None
    best = bandit.best_action()
    for t in range(1, steps + 1):
        action = select_fn(Q, N, t, rng)
        reward = bandit.step(action)
        N[action] += 1
        if step_size is None:
            Q[action] += (reward - Q[action]) / N[action]
        else:
            Q[action] += step_size * (reward - Q[action])
        rewards[t - 1] = reward
        if optimal is not None:
            optimal[t - 1] = 1.0 if action == best else 0.0
    return rewards, optimal


def run_epsilon_greedy_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    epsilons: tuple[float, ...] = (0.0, 0.01, 0.1),
    q_init: float = 0.0,
    step_size: float | None = None,
    seed: int = 0,
):
    """Run the 10-armed testbed with epsilon-greedy action selection.

    Sutton & Barto (2nd ed.), Sec. 2.3, Fig. 2.2: average reward and
    % optimal action over steps for epsilon-greedy methods with sample-average
    estimates on the stationary 10-armed testbed (q*(a) ~ N(0,1), reward ~ N(q*,1)).
    Book used 2000 runs x 1000 steps; defaults here use fewer runs for speed and
    are configurable.
    """
    results = {}
    for eps in epsilons:
        avg_r = np.zeros(steps, dtype=np.float64)
        avg_o = np.zeros(steps, dtype=np.float64)
        for r in range(runs):
            rs = seed + r * 9973 + int(eps * 100000)

            def select_fn(Q, N, t, rng, e=eps):
                if rng.random() < e:
                    return int(rng.integers(len(Q)))
                best_actions = np.flatnonzero(Q == Q.max())
                return int(rng.choice(best_actions))

            rewards, optimal = _run_bandit_bandit_episode(
                k, steps, rs, select_fn, lambda *a: None, q_init=q_init, step_size=step_size
            )
            avg_r += rewards
            avg_o += optimal
        results[eps] = {
            "average_reward": avg_r / runs,
            "optimal_action": (avg_o / runs) * 100.0,
        }
    return results


def run_optimistic_initial_values_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    configs: tuple[tuple[float, float], ...] = ((5.0, 0.0), (0.0, 0.1)),
    seed: int = 0,
):
    """Compare optimistic initialization with realistic initialization.

    Sutton & Barto (2nd ed.), Sec. 2.6, Fig. 2.3: greedy with optimistic
    Q1=+5, alpha=0.1 vs epsilon-greedy (Q1=0, eps=0.1, alpha=0.1) on the
    10-armed testbed.
    """
    results = {}
    for q_init, eps in configs:
        label = f"Q1={q_init:g}, eps={eps:g}"
        avg_r = np.zeros(steps, dtype=np.float64)
        avg_o = np.zeros(steps, dtype=np.float64)
        for r in range(runs):
            rs = seed + r * 7919 + int(q_init * 100) + int(eps * 1000)

            def select_fn(Q, N, t, rng, e=eps):
                if rng.random() < e:
                    return int(rng.integers(len(Q)))
                best_actions = np.flatnonzero(Q == Q.max())
                return int(rng.choice(best_actions))

            rewards, optimal = _run_bandit_bandit_episode(
                k, steps, rs, select_fn, lambda *a: None, q_init=q_init, step_size=0.1
            )
            avg_r += rewards
            avg_o += optimal
        results[label] = {
            "average_reward": avg_r / runs,
            "optimal_action": (avg_o / runs) * 100.0,
        }
    return results


def run_ucb_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    c: float = 2.0,
    epsilon: float = 0.1,
    seed: int = 0,
):
    """Compare UCB action selection with epsilon-greedy (Fig. 2.4)."""
    results = {}
    for name in ("ucb", "eps_greedy"):
        avg_r = np.zeros(steps, dtype=np.float64)
        avg_o = np.zeros(steps, dtype=np.float64)
        for r in range(runs):
            rs = seed + r * 104729 + (1 if name == "ucb" else 2)
            if name == "ucb":
                def select_fn(Q, N, t, rng):
                    untried = np.flatnonzero(N == 0)
                    if len(untried) > 0:
                        return int(untried[0])
                    ucb = Q + c * np.sqrt(np.log(max(t, 1)) / N)
                    best_actions = np.flatnonzero(ucb == ucb.max())
                    return int(rng.choice(best_actions))
            else:
                def select_fn(Q, N, t, rng, e=epsilon):
                    if rng.random() < e:
                        return int(rng.integers(len(Q)))
                    best_actions = np.flatnonzero(Q == Q.max())
                    return int(rng.choice(best_actions))
            rewards, optimal = _run_bandit_bandit_episode(
                k, steps, rs, select_fn, lambda *a: None, q_init=0.0, step_size=None
            )
            avg_r += rewards
            avg_o += optimal
        label = f"UCB c={c:g}" if name == "ucb" else f"eps-greedy eps={epsilon:g}"
        results[label] = {"average_reward": avg_r / runs, "optimal_action": (avg_o / runs) * 100.0}
    return results


def run_gradient_bandit_experiment(
    k: int = 10,
    steps: int = 1000,
    runs: int = 200,
    configs: tuple[tuple[float, bool], ...] = ((0.1, True), (0.4, True), (0.1, False), (0.4, False)),
    seed: int = 0,
):
    """Gradient bandit algorithm comparison (Fig. 2.5)."""
    results = {}
    for alpha, with_baseline in configs:
        label = f"alpha={alpha:g}, baseline={with_baseline}"
        avg_r = np.zeros(steps, dtype=np.float64)
        avg_o = np.zeros(steps, dtype=np.float64)
        for r in range(runs):
            rs = seed + r * 65537 + int(alpha * 100) + (10 if with_baseline else 0)
            rng = np.random.default_rng(rs)
            bandit = BanditProblem(k=k, seed=rs)
            bandit.q_star = bandit.rng.normal(4.0, 1.0, k)
            best = bandit.best_action()
            H = np.zeros(k, dtype=np.float64)
            r_bar = 0.0
            for t in range(steps):
                shifted = H - H.max()
                exp_vals = np.exp(shifted)
                probs = exp_vals / exp_vals.sum()
                action = int(rng.choice(k, p=probs))
                reward = bandit.step(action)
                avg_r[t] += reward
                avg_o[t] += 1.0 if action == best else 0.0
                baseline = r_bar if with_baseline else 0.0
                one_hot = np.zeros(k)
                one_hot[action] = 1.0
                H += alpha * (reward - baseline) * (one_hot - probs)
                r_bar += (reward - r_bar) / (t + 1)
        results[label] = {"average_reward": avg_r / runs, "optimal_action": (avg_o / runs) * 100.0}
    return results


def plot_figure_2_1(save_path: str | None = None, show: bool = False, seed: int = 0):
    """Fig 2.1: An example bandit problem (10-armed testbed), Sec. 2.3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bandit = BanditProblem(k=10, seed=seed)
    samples = np.array([[bandit.step(a) for _ in range(400)] for a in range(bandit.k)]).T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot([samples[:, a] for a in range(bandit.k)], showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.scatter(range(1, bandit.k + 1), bandit.q_star, color="black", zorder=3, label="q*(a)")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Action")
    ax.set_ylabel("Reward distribution")
    ax.set_title("Sutton & Barto (2nd ed.), Sec. 2.3, Fig. 2.1: 10-armed testbed example")
    ax.set_xticks(range(1, bandit.k + 1))
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_2_2(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.2: Average performance of epsilon-greedy, Sec. 2.3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_epsilon_greedy_experiment(runs=runs, seed=seed)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    for eps, data in results.items():
        label = f"eps = {eps:g}"
        ax1.plot(data["average_reward"], label=label)
        ax2.plot(data["optimal_action"], label=label)
    ax1.set_ylabel("Average reward")
    ax1.set_title("S&B (2nd ed.), Sec. 2.3, Fig. 2.2: epsilon-greedy (10-armed testbed)")
    ax1.legend()
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_ylim(0, 100)
    ax2.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_figure_2_3(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.3: Optimistic initial values, Sec. 2.6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_optimistic_initial_values_experiment(runs=runs, seed=seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, data in results.items():
        ax.plot(data["optimal_action"], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("% Optimal action")
    ax.set_ylim(0, 100)
    ax.set_title("S&B (2nd ed.), Sec. 2.6, Fig. 2.3: Optimistic initial values")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_2_4(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.4: UCB action selection, Sec. 2.7."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_ucb_experiment(runs=runs, seed=seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, data in results.items():
        ax.plot(data["average_reward"], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average reward")
    ax.set_title("S&B (2nd ed.), Sec. 2.7, Fig. 2.4: UCB action selection")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_figure_2_5(save_path: str | None = None, show: bool = False, runs: int = 500, seed: int = 0):
    """Fig 2.5: Gradient bandit algorithms, Sec. 2.8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = run_gradient_bandit_experiment(runs=runs, seed=seed)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, data in results.items():
        ax.plot(data["optimal_action"], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("% Optimal action")
    ax.set_ylim(0, 100)
    ax.set_title("S&B (2nd ed.), Sec. 2.8, Fig. 2.5: Gradient bandit algorithms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_all_figures_chapter_2(figures_dir: str | None = None, runs: int = 500, seed: int = 0):
    """Generate and save chapter 2 figures (2.1-2.5) to a directory."""
    import os

    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    outputs = {}
    outputs["fig_2_1"] = os.path.join(figures_dir, "fig_2_1_testbed.png")
    plot_figure_2_1(outputs["fig_2_1"], seed=seed)
    outputs["fig_2_2"] = os.path.join(figures_dir, "fig_2_2_epsilon_greedy.png")
    plot_figure_2_2(outputs["fig_2_2"], runs=runs, seed=seed)
    outputs["fig_2_3"] = os.path.join(figures_dir, "fig_2_3_optimistic_initial_values.png")
    plot_figure_2_3(outputs["fig_2_3"], runs=runs, seed=seed)
    outputs["fig_2_4"] = os.path.join(figures_dir, "fig_2_4_ucb.png")
    plot_figure_2_4(outputs["fig_2_4"], runs=runs, seed=seed)
    outputs["fig_2_5"] = os.path.join(figures_dir, "fig_2_5_gradient_bandit.png")
    plot_figure_2_5(outputs["fig_2_5"], runs=runs, seed=seed)
    return outputs
