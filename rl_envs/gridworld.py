"""GridWorld environments (Sutton & Barto, 2nd ed.).

Two gridworlds live here:

- ``GridWorld``: the adapted 4x4 gridworld demonstration from Section 1.3,
  Figure 1.2 (states as grid cells, actions as moves, rewards on
  transitions, special/terminal outcomes). Ported from
  ``chapter_01/exercise_1_1.py``.
- ``Example41GridWorld``: the classic 4x4 episodic gridworld of Example 3.5
  ("Gridworld") / Example 4.1 (policy evaluation). All transitions give
  reward -1; the two corner cells (0, 0) and (3, 3) are terminal; the
  equiprobable random policy's converged values are shown in Figure 4.1.
  The optimal policy proceeds to a terminal state in the minimum number of
  steps.

Also provides the small random/greedy agents and episode runners used by
Chapter 1's demo, plus ``plot_gridworld_figure`` lives in the chapter
module (it is chapter-specific matplotlib code).
"""

from __future__ import annotations

from typing import Any

import numpy as np


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


class Example41GridWorld:
    """The classic 4x4 episodic gridworld of Example 3.5 / Example 4.1.

    All transitions give reward -1. The terminal cells are the two
    main-diagonal corners (0, 0) and (3, 3) (shown shaded in the book).
    Actions that would move off the grid leave the agent in place. The
    equiprobable random policy's converged values are (Figure 4.1, random
    policy, k -> infinity)::

         0.0  -14.0  -20.0  -22.0
       -14.0  -18.0  -20.0  -20.0
       -20.0  -20.0  -18.0  -14.0
       -22.0  -20.0  -14.0    0.0

    The optimal policy proceeds to a terminal state in the minimum number of
    steps; policy/value iteration find the optimal policy after a single
    sweep. ``gamma`` is 1.0 by default (undiscounted, episodic).
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = ["up", "down", "left", "right"]
    ACTION_DELTAS = GridWorld.ACTION_DELTAS

    TERMINAL_CELLS = ((0, 0), (3, 3))

    def __init__(self, gamma: float = 1.0) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be between 0 and 1, got {gamma}")
        self.gamma = gamma
        self.n_actions = 4
        self.n_states = 16
        self._current_state: tuple[int, int] | None = None

    @staticmethod
    def to_index(cell: tuple[int, int]) -> int:
        """Row-major state index of a cell."""
        row, col = cell
        return row * 4 + col

    @staticmethod
    def to_cell(index: int) -> tuple[int, int]:
        """Cell of a row-major state index."""
        if not 0 <= index < 16:
            raise ValueError(f"index must be in [0, 16), got {index}")
        return divmod(index, 4)

    @property
    def state(self) -> tuple[int, int] | None:
        return self._current_state

    def is_terminal_cell(self, cell: tuple[int, int]) -> bool:
        return cell in self.TERMINAL_CELLS

    def is_terminal(self) -> bool:
        return self._current_state in self.TERMINAL_CELLS if self._current_state else True

    def reset(self) -> tuple[int, int]:
        """Reset to a nonterminal start cell.

        Default start is the cell next to the top-left terminal ((0, 1));
        the book's examples use arbitrary starts since the task is episodic
        and undiscounted.
        """
        self._current_state = (0, 1)
        return self._current_state

    def set_state(self, cell: tuple[int, int]) -> tuple[int, int]:
        """Place the agent in ``cell`` (used by dynamic-programming sweeps)."""
        row, col = cell
        if not (0 <= row < 4 and 0 <= col < 4):
            raise ValueError(f"cell out of grid: {cell}")
        self._current_state = (row, col)
        return self._current_state

    def get_next_state(self, cell: tuple[int, int], action: int) -> tuple[int, int]:
        """Deterministic transition (off-grid moves stay in place)."""
        if action not in (self.UP, self.DOWN, self.LEFT, self.RIGHT):
            raise ValueError(f"action must be in [0, 3], got {action}")
        if self.is_terminal_cell(cell):
            return cell
        row, col = cell
        dr, dc = self.ACTION_DELTAS[action]
        nxt = (row + dr, col + dc)
        if not (0 <= nxt[0] < 4 and 0 <= nxt[1] < 4):
            return cell
        return nxt

    def get_reward(self, cell: tuple[int, int], action: int) -> float:
        """-1 on every transition out of a nonterminal cell, else 0."""
        if self.is_terminal_cell(cell):
            return 0.0
        return -1.0

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """Execute an action; return (next_cell, reward, done)."""
        if self._current_state is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")
        current = self._current_state
        next_state = self.get_next_state(current, action)
        reward = self.get_reward(current, action)
        self._current_state = next_state
        return next_state, reward, self.is_terminal_cell(next_state)
