"""
Exercise 1.1 — Gridworld Environment and Agent Simulation

Reference: Sutton & Barto, Section 1.3 "An Example Agent" — The Gridworld Example
           (Reinforcement Learning: An Introduction, 2nd Edition, Figure 1.2)

This module implements:
1. A GridWorld environment with the MDP structure from Figure 1.2
2. A simple agent that interacts with the environment
3. Visualization of state values and policies

Environment Structure (4x4 grid):
    (0,0) [A]  (0,1)     (0,2)     (0,3) [B]
    (1,0)      (1,1)     (1,2)     (1,3)
    (2,0)      (2,1)     (2,2)     (2,3)
    (3,0) [A+] (3,1)     (3,2)     (3,3) [D]

Special States:
    - A  (0,0): Actions "up" or "left" -> A+ with +10 reward; others -> B with +5
    - B  (0,3): Any action -> C (0,2) with +5 reward
    - A+ (3,0): Terminal state (absorbing), reward +10
    - D  (3,3): Terminal state (absorbing), reward -10

All other states: deterministic transitions in the chosen direction, reward -1.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class GridWorld:
    """
    Gridworld environment from Sutton & Barto, Section 1.3, Figure 1.2.

    A 4x4 grid world where the agent navigates from starting positions toward
    terminal states. The environment defines the MDP with states, actions,
    transition dynamics, and reward structure.

    Parameters:
        grid_width: Width of the grid (default 4).
        grid_height: Height of the grid (default 4).
        discount: Discount factor gamma for future rewards (default 1.0).

    Book Reference:
        Sutton & Barto (2nd Ed.), Section 1.3 "An Example Agent"
    """

    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTION_NAMES = ["up", "down", "left", "right"]

    # Action deltas: (row_change, col_change)
    ACTION_DELTAS: list[tuple[int, int]] = [
        (-1, 0),  # UP
        (1, 0),   # DOWN
        (0, -1),  # LEFT
        (0, 1),   # RIGHT
    ]

    def __init__(self, grid_width: int = 4, grid_height: int = 4, discount: float = 1.0) -> None:
        """
        Initialize the GridWorld environment.

        Sets up the grid dimensions, special states, and terminal state tracking.

        Parameters:
            grid_width: Number of columns in the grid.
            grid_height: Number of rows in the grid.
            discount: Discount factor gamma (0 <= gamma <= 1).
        """
        if not (0.0 <= discount <= 1.0):
            raise ValueError(f"Discount factor must be between 0 and 1, got {discount}")
        if grid_width < 2 or grid_height < 2:
            raise ValueError("Grid must be at least 2x2")

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.discount = discount

        # Define special states (following Figure 1.2 layout)
        self.state_A: tuple[int, int] = (0, 0)
        self.state_B: tuple[int, int] = (0, 3)
        self.state_A_plus: tuple[int, int] = (3, 0)
        self.state_D: tuple[int, int] = (3, 3)

        # Map state to its "special" identifier
        self._state_map: dict[tuple[int, int], str] = {
            self.state_A: "A",
            self.state_B: "B",
            self.state_A_plus: "A+",
            self.state_D: "D",
        }

        # Terminal states
        self._terminal_states: set[tuple[int, int]] = {self.state_A_plus, self.state_D}

        # State C (reached from B via UP or RIGHT)
        self.state_C: tuple[int, int] = (0, 2)

        # Action space
        self.n_actions = 4

        # Reset state
        self._current_state: tuple[int, int] | None = None

    @property
    def state(self) -> tuple[int, int] | None:
        """Return the current state of the environment."""
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        """Return True if the current state is terminal."""
        return self._current_state in self._terminal_states if self._current_state else True

    def _is_valid_state(self, state: tuple[int, int]) -> bool:
        """Check if a state is within the grid bounds."""
        row, col = state
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width

    def reset(self) -> tuple[int, int]:
        """Reset the environment to the initial state."""
        self._current_state = (0, 0)
        return self._current_state

    def step(self, action: int) -> tuple[tuple[int, int] | None, float, bool, dict[str, Any]]:
        """Execute an action in the Gridworld environment."""
        if self._current_state is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Action must be between 0 and {self.n_actions - 1}, got {action}")

        info: dict[str, Any] = {
            "state": self._current_state,
            "action": action,
            "action_name": self.ACTION_NAMES[action],
        }
        current_state = self._current_state

        # Handle special state A
        if current_state == self.state_A:
            if action in (self.UP, self.LEFT):
                next_state, reward = self.state_A_plus, 10.0
            else:
                next_state, reward = self.state_B, 5.0
            done = next_state in self._terminal_states
            info.update(special_state="A", transition_target=next_state)
            self._current_state = next_state
            return next_state, reward, done, info

        # Handle special state B: UP/RIGHT -> C, DOWN/LEFT -> D
        if current_state == self.state_B:
            if action in (self.UP, self.RIGHT):
                next_state, reward = self.state_C, 5.0
            else:
                next_state, reward = self.state_D, 5.0
            done = next_state in self._terminal_states
            info.update(special_state="B", transition_target=next_state)
            self._current_state = next_state
            return next_state, reward, done, info

        # Terminal states
        if current_state in self._terminal_states:
            return None, 0.0, True, {**info, "special_state": "terminal"}

        # Normal states: deterministic transition
        row, col = current_state
        dr, dc = self.ACTION_DELTAS[action]
        next_row, next_col = row + dr, col + dc
        if self._is_valid_state((next_row, next_col)):
            next_state = (next_row, next_col)
        else:
            next_state = current_state

        reward = -1.0
        done = next_state in self._terminal_states
        info["transition_target"] = next_state
        self._current_state = next_state
        return next_state, reward, done, info

    def get_reward(self, state: tuple[int, int], action: int) -> float:
        """Get the immediate reward for (state, action)."""
        if state == self.state_A:
            return 10.0 if action in (self.UP, self.LEFT) else 5.0
        if state == self.state_B:
            return 5.0
        if state in self._terminal_states:
            return 0.0
        return -1.0

    def get_next_state(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        """Get the next state for a given state and action."""
        if state in self._terminal_states:
            return state
        # Special state A: UP or LEFT -> A+, other actions -> B
        if state == self.state_A:
            if action in (self.UP, self.LEFT):
                return self.state_A_plus
            return self.state_B
        # Special state B: UP/RIGHT -> C, DOWN/LEFT -> D
        if state == self.state_B:
            if action in (self.UP, self.RIGHT):
                return self.state_C
            return self.state_D
        row, col = state
        dr, dc = self.ACTION_DELTAS[action]
        next_row, next_col = row + dr, col + dc
        if self._is_valid_state((next_row, next_col)):
            return (next_row, next_col)
        return state

    def render(self, values: dict[tuple[int, int], float] | None = None,
               policy: dict[tuple[int, int], int] | None = None) -> str:
        """Render the gridworld as a string."""
        lines: list[str] = []
        action_symbols = ["↑", "↓", "←", "→"]
        for row_idx in range(self.grid_height):
            row_str = "  "
            for col_idx in range(self.grid_width):
                state = (row_idx, col_idx)
                state_id = self._state_map.get(state, f"({row_idx},{col_idx})")
                if state in self._state_map:
                    cell = f"[{state_id}] "
                else:
                    val_str = f"{values[state]:6.2f}" if values and state in values else ""
                    act_str = action_symbols[policy[state]] if policy and state in policy else ""
                    cell = f"{val_str} {act_str}"
                row_str += cell
            lines.append(row_str)
        return "\n".join(lines)


class RandomAgent:
    """A random agent that selects actions uniformly at random. Book Ref: S&B Sec 1.3."""

    def __init__(self, n_actions: int = 4) -> None:
        self.n_actions = n_actions

    def select_action(self, state: tuple[int, int] | None, epsilon: float = 0.0) -> int:
        """Select an action uniformly at random."""
        return int(np.random.randint(0, self.n_actions))


class GreedyAgent:
    """A greedy agent that selects the highest-value action. Book Ref: S&B Sec 1.3."""

    def __init__(self, n_actions: int = 4) -> None:
        self.n_actions = n_actions
        self._values: dict[tuple[int, int], list[float]] = {}

    def update_value(self, state: tuple[int, int], action: int, value: float) -> None:
        """Update the action-value estimate for a state-action pair."""
        if state not in self._values:
            self._values[state] = [0.0] * self.n_actions
        self._values[state][action] = value

    def select_action(self, state: tuple[int, int] | None, epsilon: float = 0.0) -> int:
        """Greedy action selection with epsilon-greedy exploration."""
        if state is None or state not in self._values:
            return int(np.random.randint(0, self.n_actions))
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        values = self._values[state]
        max_val = max(values)
        best_actions = [i for i, v in enumerate(values) if v == max_val]
        return int(np.random.choice(best_actions))


def run_episode(
    env: GridWorld,
    agent: RandomAgent | GreedyAgent,
    max_steps: int = 1000,
    epsilon: float = 0.0,
    render_every: int = 0,
) -> list[tuple[tuple[int, int], int, float, tuple[int, int] | None]]:
    """Run a single episode. Returns list of (state, action, reward, next_state)."""
    trajectory: list[tuple[tuple[int, int], int, float, tuple[int, int] | None]] = []
    state = env.reset()
    for _ in range(max_steps):
        if env.is_terminal:
            break
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, next_state))
        if done:
            break
        state = next_state
    return trajectory


def run_multiple_episodes(
    env: GridWorld,
    agent: RandomAgent | GreedyAgent,
    n_episodes: int = 100,
    epsilon: float = 0.0,
) -> dict[str, list[float]]:
    """Run multiple episodes and collect total_reward and steps stats."""
    stats: dict[str, list[float]] = {"total_reward": [], "steps": []}
    for _ in range(n_episodes):
        traj = run_episode(env, agent, max_steps=1000, epsilon=epsilon)
        stats["total_reward"].append(sum(r for _, _, r, _ in traj))
        stats["steps"].append(len(traj))
    return stats


def plot_gridworld_figure(save_path: str | None = None, show: bool = False):
    """Create a demonstration chart of this chapter's GridWorld example.

    Sutton & Barto (2nd ed.), Section 1.3 "An Example Agent", Figure 1.2
    (gridworld example): the book introduces the RL agent-environment interface
    with a gridworld MDP (states as grid cells, actions as moves, rewards on
    transitions, and terminal/special outcomes).

    This repository implements an adapted 4x4 gridworld demonstration (see
    ``GridWorld`` in this module and ``chapter_01/README.md``): special states
    A (0,0), B (0,3), C (0,2), terminals A+ (3,0, reward +10 on entry from A)
    and D (3,3), with deterministic moves and step reward -1 in ordinary cells.

    The figure visualizes the implemented environment layout and the special
    transition/reward structure so students can connect code to the book's
    gridworld concepts (states, actions, rewards, terminal states).

    Args:
        save_path: Optional path to save the figure (e.g. PNG). Parent
            directories should exist.
        show: If True, display the figure window (default False for headless use).

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    env = GridWorld(grid_width=4, grid_height=4, discount=1.0)
    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    # Draw grid cells (row 0 at top to match render/layout in this module).
    for r in range(env.grid_height):
        for c in range(env.grid_width):
            state = (r, c)
            if state == env.state_A_plus:
                face = "#d9f2d9"
            elif state == env.state_D:
                face = "#f9d6d5"
            elif state == env.state_A:
                face = "#dcecff"
            elif state == env.state_B:
                face = "#fff2cc"
            elif state == env.state_C:
                face = "#f3e8ff"
            else:
                face = "#ffffff"
            rect = Rectangle((c, r), 1, 1, facecolor=face, edgecolor="#444444", linewidth=1.2)
            ax.add_patch(rect)
            label = f"({r},{c})"
            if state == env.state_A:
                label = "A (0,0)"
            elif state == env.state_B:
                label = "B (0,3)"
            elif state == env.state_A_plus:
                label = "A+ terminal\nr=+10"
            elif state == env.state_D:
                label = "D terminal"
            elif state == env.state_C:
                label = "C (0,2)"
            ax.text(c + 0.5, r + 0.5, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, text, color="#333333"):
        patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12,
                                linewidth=1.4, color=color)
        ax.add_patch(patch)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 0.08, text, fontsize=7.5,
                ha="center", va="top", color=color,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})

    # Annotate special transitions (book concepts: actions -> next state, reward).
    # From A: UP/LEFT -> A+ (+10), DOWN/RIGHT -> B (+5)
    arrow(0.35, 0.65, 0.35, 2.6, "A: up/left\nr=+10", color="#2e7d32")
    arrow(0.75, 0.35, 2.6, 0.35, "A: down/right\nr=+5", color="#1565c0")
    # From B: UP/RIGHT -> C (+5), DOWN/LEFT -> D (+5)
    arrow(3.35, 0.65, 2.65, 0.65, "B: up/right\nr=+5", color="#6a1b9a")
    arrow(3.55, 0.75, 3.55, 2.6, "B: down/left\nr=+5", color="#6a1b9a")

    ax.set_xlim(0, env.grid_width)
    ax.set_ylim(env.grid_height, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.grid_width + 1))
    ax.set_yticks(range(env.grid_height + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_title(
        "GridWorld demonstration (adapted from Sutton & Barto, 2nd ed.,\n"
        "Sec. 1.3, Fig. 1.2): states, actions, rewards, terminals"
    )
    ax.text(0.0, 4.35,
            "Ordinary moves: deterministic step, r = -1 (unless off-grid: stay in place).",
            fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        matplotlib.use("TkAgg", force=True)
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def main() -> None:
    """Run a demo of the Gridworld environment."""
    print("=" * 60)
    print("Gridworld Environment -- Sutton & Barto, Section 1.3")
    print("=" * 60)

    env = GridWorld(grid_width=4, grid_height=4, discount=1.0)
    agent = RandomAgent(n_actions=4)

    print("\n--- Initial Grid ---")
    print(env.render())

    print("\n--- Running 5 episodes with random agent ---")
    stats = run_multiple_episodes(env, agent, n_episodes=5, epsilon=0.0)
    print(f"\nAverage total reward: {np.mean(stats['total_reward']):.2f}")
    print(f"Average steps per episode: {np.mean(stats['steps']):.1f}")

    print("\n--- Running 5 episodes with greedy agent ---")
    greedy_agent = GreedyAgent(n_actions=4)
    stats2 = run_multiple_episodes(env, greedy_agent, n_episodes=5, epsilon=0.0)
    print(f"\nAverage total reward: {np.mean(stats2['total_reward']):.2f}")
    print(f"Average steps per episode: {np.mean(stats2['steps']):.1f}")

    print("\n--- Special State Transitions ---")
    env.reset()
    print(f"State A: {env.state_A}")
    for name, idx in [("UP", GridWorld.UP), ("LEFT", GridWorld.LEFT),
                      ("DOWN", GridWorld.DOWN), ("RIGHT", GridWorld.RIGHT)]:
        print(f"  {name:6s} -> {env.get_next_state(env.state_A, idx)} reward={env.get_reward(env.state_A, idx)}")
    print(f"\nState B: {env.state_B}")
    for name, idx in [("UP", GridWorld.UP), ("LEFT", GridWorld.LEFT),
                      ("DOWN", GridWorld.DOWN), ("RIGHT", GridWorld.RIGHT)]:
        print(f"  {name:6s} -> {env.get_next_state(env.state_B, idx)} reward={env.get_reward(env.state_B, idx)}")


if __name__ == "__main__":
    main()