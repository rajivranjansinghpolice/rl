"""Four-room GridWorld and options utilities (Sutton & Barto, 2nd ed., Ch. 17)."""

from __future__ import annotations

import math
from collections import deque
from typing import Callable

DEFAULT_GRID = [
    ".......",
    ".......",
    ".......",
    "##.#.##",
    "...#...",
    "...#...",
    ".......",
]

_START = (0, 0)
_GOAL = (6, 6)


def is_wall(grid: list[str], cell: tuple[int, int]) -> bool:
    """True if `cell` (row, col) is a wall/out-of-bounds cell."""
    r, c = cell
    return r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == "#"


def bfs_distances(grid: list[str], target: tuple[int, int]) -> dict:
    """Shortest-path distances from every free cell to `target`."""
    distances: dict[tuple[int, int], int] = {}
    queue = deque([target])
    distances[target] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if not is_wall(grid, nxt) and nxt not in distances:
                distances[nxt] = distances[(r, c)] + 1
                queue.append(nxt)
    return distances


def bfs_action_toward(
    grid: list[str],
    cell: tuple[int, int],
    target: tuple[int, int],
) -> int:
    """Action index (GridWorld.ACTIONS) moving along a shortest path."""
    distances = bfs_distances(grid, target)
    best_action = 0
    best_distance = distances.get(cell, 10**9)
    for idx, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        nxt = (cell[0] + dr, cell[1] + dc)
        if is_wall(grid, nxt):
            continue
        distance = distances.get(nxt, 10**9)
        if distance < best_distance:
            best_distance = distance
            best_action = idx
    return best_action


class GridWorld:
    """Small tabular four-room GridWorld (7x7). Book Sec. 17.2 demo."""

    ACTIONS = ("up", "down", "left", "right")
    _DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        grid: list[str] | None = None,
        start: tuple[int, int] = _START,
        goal: tuple[int, int] = _GOAL,
        step_cost: float = -0.01,
        wall_penalty: float = -0.25,
    ) -> None:
        self.grid = grid if grid is not None else list(DEFAULT_GRID)
        self.start = start
        self.goal = goal
        self.step_cost = float(step_cost)
        self.wall_penalty = float(wall_penalty)
        self._cells = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, ch in enumerate(row)
            if ch != "#"
        ]
        self._index = {cell: i for i, cell in enumerate(self._cells)}
        self.n_states = len(self._cells)
        self.state: int | None = None
        self.done = False

    def reset(self) -> int:
        self.state = self._index[self.start]
        self.done = False
        return self.state

    def cell(self, state: int) -> tuple[int, int]:
        return self._cells[state]

    def state_id(self, cell: tuple[int, int]) -> int:
        return self._index[cell]

    def step(self, action: int) -> tuple[int, float, bool]:
        if self.state is None:
            raise RuntimeError("call reset() first")
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError("action must be in [0, 3]")
        r, c = self._cells[self.state]
        dr, dc = self._DELTAS[action]
        nxt = (r + dr, c + dc)
        if is_wall(self.grid, nxt):
            self.state = self._index[(r, c)]
            return self.state, self.wall_penalty, self.done
        self.state = self._index[nxt]
        if nxt == self.goal:
            self.done = True
            return self.state, 1.0, True
        self.done = False
        return self.state, self.step_cost, False


class Option:
    """Temporally abstract action (Section 17.2)."""

    def __init__(
        self,
        name: str,
        policy_fn: Callable[[GridWorld], int],
        termination_fn: Callable[[GridWorld], bool],
    ) -> None:
        self.name = name
        self.policy_fn = policy_fn
        self.termination_fn = termination_fn

    def policy(self, env: GridWorld) -> int:
        return int(self.policy_fn(env))

    def terminates(self, env: GridWorld) -> bool:
        return bool(self.termination_fn(env))


def make_subgoal_option(
    grid: list[str] | None,
    subgoal: tuple[int, int],
    name: str,
) -> Option:
    base_grid = list(grid) if grid is not None else list(DEFAULT_GRID)

    def current_cell(env: GridWorld) -> tuple[int, int]:
        if env.state is None:
            raise RuntimeError("call reset() first")
        return env.cell(env.state)

    def policy(env: GridWorld) -> int:
        return bfs_action_toward(base_grid, current_cell(env), subgoal)

    def termination(env: GridWorld) -> bool:
        return current_cell(env) == subgoal

    return Option(name, policy, termination)


def default_options() -> list[Option]:
    return [
        make_subgoal_option(None, (3, 2), "option_gap_left"),
        make_subgoal_option(None, (3, 4), "option_gap_right"),
        make_subgoal_option(None, _GOAL, "option_goal"),
    ]


class OptionResult:
    def __init__(self, discounted_return: float, steps: int, done: bool) -> None:
        self.discounted_return = float(discounted_return)
        self.steps = int(steps)
        self.done = bool(done)


def run_option(
    env: GridWorld,
    option: Option,
    gamma: float = 0.99,
    max_steps: int = 50,
) -> OptionResult:
    if env.state is None:
        raise RuntimeError("call reset() first")
    discounted_return = 0.0
    discount = 1.0
    steps = 0
    while steps < max_steps:
        action = option.policy(env)
        _, reward, done = env.step(action)
        discounted_return += discount * reward
        discount *= gamma
        steps += 1
        if done or option.terminates(env):
            return OptionResult(discounted_return, steps, done)
    return OptionResult(discounted_return, steps, env.done)
