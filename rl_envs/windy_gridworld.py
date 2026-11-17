"""Windy Gridworld (Sutton & Barto, 2nd ed., Example 6.5, Sec. 6.4)."""

from __future__ import annotations

import numpy as np


class WindyGridWorld:
    """Windy Gridworld environment (Example 6.5). Book Section 6.4.

    Standard 7 (rows) x 10 (columns) grid. Start S=(3,0), goal G=(3,7).
    Each column has an upward wind strength: [0,0,0,1,1,1,2,2,1,0].
    Actions move one step (up, down, left, right) then wind pushes up by the
    column's strength. Agent stays in bounds. Reward -1 per step until goal.

    Args:
        height: Number of rows.
        width: Number of columns.
        start: (row, col) start state.
        goal: (row, col) goal state.
        wind: Sequence of upward wind strengths per column.
        king_moves: If True, allow 8 diagonal/king moves (Example 6.6 variant).
        allow_stay: If True and king_moves, include a no-op action.
        stochastic_wind: If True, wind varies by {-1,0,+1} with equal prob.
        seed: RNG seed for stochastic wind.
    """

    ACTIONS_4 = ("up", "down", "left", "right")
    DELTAS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    ACTIONS_8 = ("up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right")
    DELTAS_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    ACTIONS_9 = ACTIONS_8 + ("stay",)
    DELTAS_9 = DELTAS_8 + ((0, 0),)

    def __init__(
        self,
        height: int = 7,
        width: int = 10,
        start: tuple[int, int] = (3, 0),
        goal: tuple[int, int] = (3, 7),
        wind: tuple[int, ...] | list[int] | None = None,
        king_moves: bool = False,
        allow_stay: bool = False,
        stochastic_wind: bool = False,
        seed: int | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.wind = tuple(wind) if wind is not None else (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
        if len(self.wind) != width:
            raise ValueError("wind length must equal width")
        self.king_moves = king_moves
        self.allow_stay = allow_stay
        self.stochastic_wind = stochastic_wind
        self.rng = np.random.default_rng(seed)
        if king_moves and allow_stay:
            self.ACTIONS = self.ACTIONS_9
            self._deltas = self.DELTAS_9
        elif king_moves:
            self.ACTIONS = self.ACTIONS_8
            self._deltas = self.DELTAS_8
        else:
            self.ACTIONS = self.ACTIONS_4
            self._deltas = self.DELTAS_4
        self.n_actions = len(self.ACTIONS)
        self.state: tuple[int, int] | None = None

    def reset(self, state: tuple[int, int] | None = None) -> tuple[int, int]:
        self.state = self.start if state is None else state
        return self.state

    def _clip(self, r: int, c: int) -> tuple[int, int]:
        return (max(0, min(self.height - 1, r)), max(0, min(self.width - 1, c)))

    def step(self, action: int | str) -> tuple[tuple[int, int], float, bool]:
        if self.state is None:
            raise RuntimeError("call reset() first")
        if isinstance(action, str):
            action = self.ACTIONS.index(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action {action}")
        r, c = self.state
        dr, dc = self._deltas[action]
        nr, nc = r + dr, c + dc
        w = self.wind[c] if 0 <= c < len(self.wind) else 0
        if self.stochastic_wind and w > 0:
            w = max(0, w + int(self.rng.integers(-1, 2)))
        nr -= w  # wind blows upward (decreasing row index)
        nr, nc = self._clip(nr, nc)
        self.state = (nr, nc)
        done = self.state == self.goal
        reward = 0.0 if done else -1.0
        return self.state, reward, done

    def n_states(self) -> int:
        return self.height * self.width
