"""Solution implementation for Chapter 10 - On-policy Control with Approximation.

Reference: Sutton & Barto (2nd Ed), Chapter 10 (On-policy Control with
Approximation).

Implements control (finding an approximate optimal policy) rather than
prediction. The core example is the Mountain Car (book Sec. 10.1,
Example 10.1) with linear semi-gradient Sarsa and tile coding.

Implements:
- algorithm_10.1 - Episodic semi-gradient Sarsa with linear function
  approximation and tile coding (Section 10.1). Per action a, Q(s, a) =
  w_a^T x(s). epsilon-greedy action selection; terminal states have bootstrap
  value 0.
- algorithm_10.2 - Average-reward differential semi-gradient n-step Sarsa
  (continuing setting, Section 10.2/Sec. 10.3), used for the Access-Control
  queuing example.
- Example 10.2 - Access-Control Queuing: a continuing two-priority admission
  control problem; differential semi-gradient Sarsa learns to admit customers
  optimally (Section 10.2/10.3).

Figure identities are honest demonstrations: the Mountain Car learning curve
and the access-control average-reward curve are illustrative reproductions of
the book concepts, not verbatim copies, and the local PDF summary does not
allow exact book figure numbers to be verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chapter_09.solution import TileCoder
from rl_envs.mountain_car import MountainCarEnv

# ---------------------------------------------------------------------------
# Mountain Car features (Section 10.1, Example 10.1)
# ---------------------------------------------------------------------------


def mountain_car_features(position: float, velocity: float) -> np.ndarray:
    """Tile-coded features for (position, velocity)."""
    coder = TileCoder((-1.2, 0.5), (-0.07, 0.07), tiles_per_tiling=8, n_tilings=8)
    return coder.features(position, velocity)


# ---------------------------------------------------------------------------
# Algorithm 10.1: Episodic semi-gradient Sarsa (linear, tile coding)
# ---------------------------------------------------------------------------

@dataclass
class EpisodicSemiGradientSarsaAgent:
    """Linear semi-gradient Sarsa with per-action weights.

    Q(s, a) = w_a^T x(s). Action selection is epsilon-greedy over the three
    Q values. The step size is alpha scaled as alpha / n_tilings (each step
    activates 8 tiling features).
    """

    n_features: int
    n_actions: int = 3
    alpha: float = 0.1
    gamma: float = 1.0
    epsilon: float = 0.0
    seed: int | None = None
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def q_values(self, x: np.ndarray) -> np.ndarray:
        return self.weights @ x

    def act(self, x: np.ndarray) -> int:
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(x)))

    def update_sarsa(self, x, action, reward, next_x, next_action) -> None:
        """Exact on-policy semi-gradient Sarsa update using Q(s', A').

        Uses the action actually taken in the next state under the current
        policy (on-policy), with a zero bootstrap when next_x is None
        (terminal).
        """
        q_next = 0.0 if next_x is None else float(self.q_values(next_x)[next_action])
        target = reward + self.gamma * q_next
        delta = target - float(self.q_values(x)[action])
        self.weights[action] += (self.alpha / 8.0) * delta * x


def run_mountain_car(
    episodes: int = 200,
    seed: int = 0,
    max_steps: int = 200,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
) -> tuple[EpisodicSemiGradientSarsaAgent, list[int]]:
    """Train the episodic semi-gradient Sarsa agent on Mountain Car.

    Returns (agent, episode_lengths). The episode length is -1 per step so a
    shorter episode means the agent escaped the valley faster; over training
    this should decrease from the 200-step cap. Exploration is epsilon-greedy
    (a greedy policy from zero weights never escapes the valley).
    """
    env = MountainCarEnv(seed=seed)
    dim = 8 * 8 * 8
    agent = EpisodicSemiGradientSarsaAgent(dim, n_actions=3, alpha=alpha,
                                           gamma=gamma, epsilon=epsilon, seed=seed)
    episode_lengths: list[int] = []
    for _ in range(episodes):
        state = env.reset()
        x = mountain_car_features(*state)
        action = agent.act(x)
        length = 0
        while True:
            position, velocity = state
            state_next, reward, done = env.step(action)
            length += 1
            x_next = None if done else mountain_car_features(*state_next)
            next_action = agent.act(x_next) if not done else None
            agent.update_sarsa(x, action, reward, x_next, next_action)
            state, x, action = state_next, x_next if x_next is not None else x, next_action
            if done or length >= max_steps:
                break
        episode_lengths.append(length)
    return agent, episode_lengths


# ---------------------------------------------------------------------------
# Algorithm 10.2 / average reward: differential semi-gradient Sarsa
# ---------------------------------------------------------------------------

@dataclass
class DifferentialSarsaAgent:
    """Average-reward differential semi-gradient Sarsa.

    Used for continuing tasks (no discounting). Maintains r_bar, the average
    reward per step, and updates weights toward the differential target
    R - r_bar + Q(s', a'). Feature weights are per-action like the episodic
    agent.
    """

    n_features: int
    n_actions: int = 2
    alpha: float = 0.05
    alpha_bar: float = 0.01
    epsilon: float = 0.1
    seed: int | None = None
    weights: np.ndarray = field(init=False)
    r_bar: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.weights = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def q_values(self, x: np.ndarray) -> np.ndarray:
        return self.weights @ x

    def act(self, x: np.ndarray) -> int:
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(x)))

    def update(self, x, action, reward, next_x, next_action) -> None:
        q_next = float(self.q_values(next_x)[next_action])
        delta = reward - self.r_bar + q_next - float(self.q_values(x)[action])
        self.weights[action] += self.alpha * delta * x
        self.r_bar += self.alpha_bar * delta


# ---------------------------------------------------------------------------
# Example 10.2: Access-Control Queuing (two-priority admission control)
# ---------------------------------------------------------------------------

@dataclass
class AccessControlQueue:
    """Two-priority access-control queuing MDP (continuing task).

    State: (h, l) = number of high- and low-priority customers waiting, each
    capped at `capacity`. Arrivals are Bernoulli: high with prob `p_high`,
    low with prob `p_low` (arrivals when the queue is full are rejected, not
    penalised). Actions: 0 = do not admit anyone, 1 = admit a high-priority
    customer, 2 = admit a low-priority customer. Reward: +8 for serving a
    high-priority customer, +1 for a low-priority one (the asymmetry is what
    makes admission worth learning). This is an illustrative implementation of
    the book's access-control example; exact book probs/rewards are not
    verifiable from the local PDF summary.
    """

    capacity: int = 10
    p_high: float = 0.5
    p_low: float = 0.6
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.h: int = 0
        self.l: int = 0

    def reset(self) -> tuple[int, int]:
        self.h, self.l = 0, 0
        return (self.h, self.l)

    def state_features(self, state: tuple[int, int]) -> np.ndarray:
        h, l = state
        scale = float(self.capacity)
        return np.array([
            h / scale,
            l / scale,
            (h / scale) * (l / scale),
            (h / scale) ** 2,
            (l / scale) ** 2,
            1.0,
        ], dtype=np.float64)

    def step(self, action: int) -> tuple[tuple[int, int], float]:
        """Take one time step. Returns ((h, l), reward)."""
        assert action in (0, 1, 2), "action must be 0 (none), 1 (high), 2 (low)"

        reward = 0.0
        # Serve one customer if admitted and available.
        if action == 1 and self.h > 0:
            self.h -= 1
            reward += 8.0
        elif action == 2 and self.l > 0:
            self.l -= 1
            reward += 1.0

        # Arrivals (capped at capacity).
        if self.rng.uniform() < self.p_high and self.h < self.capacity:
            self.h += 1
        if self.rng.uniform() < self.p_low and self.l < self.capacity:
            self.l += 1

        return (self.h, self.l), reward


def run_access_control(
    n_steps: int = 5000,
    seed: int = 0,
    plot_window: int = 200,
) -> tuple[DifferentialSarsaAgent, list[float]]:
    """Train differential Sarsa on the access-control queue.

    Returns (agent, moving_average_rewards).
    """
    env = AccessControlQueue(seed=seed)
    dim = 6
    agent = DifferentialSarsaAgent(dim, n_actions=3, alpha=0.02, alpha_bar=0.005,
                                   epsilon=0.1, seed=seed)
    state = env.reset()
    x = env.state_features(state)
    action = agent.act(x)

    rewards: list[float] = []
    moving: list[float] = []
    for _ in range(n_steps):
        next_state, reward = env.step(action)
        next_x = env.state_features(next_state)
        next_action = agent.act(next_x)
        agent.update(x, action, reward, next_x, next_action)
        rewards.append(reward)
        if len(rewards) >= plot_window:
            moving.append(float(np.mean(rewards[-plot_window:])))
        state, x, action = next_state, next_x, next_action
    return agent, moving


# ---------------------------------------------------------------------------
# Figures (illustrative demonstrations)
# ---------------------------------------------------------------------------

def plot_figure_10_mountain_car(
    save_path: str | None = None,
    show: bool = False,
    episodes: int = 1500,
    seed: int = 0,
):
    """Mountain Car learning curve (illustrative demonstration).

    Sutton & Barto (2nd ed.), Sec. 10.1 / Example 10.1 (Mountain Car):
    episodic semi-gradient Sarsa with tile coding. Plots the episode length
    per episode (raw and 20-episode moving average); over training the car
    escapes the valley sooner, so the curve trends downward from the 200-step
    cap. Illustrative demonstration; the exact book figure identity cannot be
    verified against the local PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, lengths = run_mountain_car(episodes=episodes, seed=seed, max_steps=200)
    window = 20
    avg = [
        float(np.mean(lengths[max(0, i - window + 1):i + 1]))
        for i in range(len(lengths))
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(1, len(lengths) + 1), lengths, lw=0.6, alpha=0.35, label="episode length")
    ax.plot(range(1, len(lengths) + 1), avg, lw=2.0, color="tab:red",
            label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode steps")
    ax.set_ylim(bottom=0)
    ax.set_title(
        "S&B (2nd ed.), Sec. 10.1: Semi-gradient Sarsa on Mountain Car "
        "(illustrative)"
    )
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_figure_10_access_control(
    save_path: str | None = None,
    show: bool = False,
    n_steps: int = 20000,
    seed: int = 0,
):
    """Access-control moving average reward (illustrative demonstration).

    Sutton & Barto (2nd ed.), Sec. 10.2/10.3 / Example 10.2 (Access-Control
    Queuing): differential semi-gradient Sarsa on a continuing two-priority
    admission problem. Plots the moving-average reward per step.
    Illustrative demonstration; the exact book figure identity cannot be
    verified against the local PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, moving = run_access_control(n_steps=n_steps, seed=seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(1, len(moving) + 1), moving, lw=1.5)
    ax.set_xlabel("Step (moving average over the previous 200 steps)")
    ax.set_ylabel("Average reward per step")
    ax.set_title(
        "S&B (2nd ed.), Sec. 10.2/10.3: differential Sarsa on Access-Control "
        "(illustrative)"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(seed: int | None = None) -> None:
    """Train Mountain Car and the access-control queue and print summaries."""
    _, lengths = run_mountain_car(episodes=100, seed=seed if seed is not None else 0,
                                  max_steps=200)
    print(f"mountain car: first={lengths[0]}, last={lengths[-1]}, mean={np.mean(lengths):.1f}")
    agent, moving = run_access_control(n_steps=2000, seed=seed if seed is not None else 0)
    print(f"access control: final r_bar={agent.r_bar:.4f}")
