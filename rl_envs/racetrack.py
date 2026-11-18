"""Racetrack environment (Sutton & Barto, 2nd ed., Exercise 5.8 / 5.12).

The book's racetrack (Sec. 5.5/5.7, Figure 5.5 shows two right-turn tracks):

- The car is on a track made of grid cells, off-track cells are boundaries.
- The state is the car's position and discrete velocity. Both velocity
  components are nonnegative and strictly less than 5; they may both be zero
  only on the starting line.
- There are 9 actions: each velocity component can be increased by 1,
  kept unchanged, or decreased by 1 in each step.
- Each episode begins at a random starting-line cell with velocity (0, 0)
  and ends when the car crosses the finish line.
- The reward is -1 for every step. If the car hits the track boundary, its
  position is reset to a random starting-line cell, its velocity is zeroed,
  and the episode *continues*.
- With probability ``noise_prob`` (0.1 in the book) the velocity increments
  are forced to zero on that step, so the car cannot accelerate that step.
- Before a move is applied, the straight projected path is checked: if the
  projected path crosses the finish line, the episode ends; if it crosses
  anywhere else (off-track), the car is considered to have hit the boundary
  and is sent back to a random starting-line cell.

Tracks are defined by ASCII maps: ``#`` = boundary (off-track),
``.`` = on-track, ``S`` = starting-line cell, ``F`` = finish-line cell. The
default track is a small right-turn track in the spirit of the book's
Figure 5.5; it is an illustrative demonstration, not a pixel-identical copy
of the book's printed track (the exact book grid is edition-specific and
not reproduced here).

Velocity convention: ``vx`` is the horizontal speed (each step moves the
car ``vx`` cells to the right), ``vy`` is the vertical speed (each step
moves the car ``vy`` cells *upwards*, i.e. toward row 0).
"""

from __future__ import annotations

import numpy as np

# 9 velocity-change actions: (dvx, dvy) with d* in {-1, 0, +1}.
# Index = dvy_index * 3 + dvx_index (dy first), i.e.
# 0=(-1,-1) 1=(0,-1) 2=(+1,-1) 3=(-1,0) 4=(0,0) 5=(+1,0) 6=(-1,+1) 7=(0,+1) 8=(+1,+1)
ACTIONS = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
N_ACTIONS = len(ACTIONS)
MAX_VELOCITY = 4  # both components are in {0..4}; (0, 0) only on the start line

_DEFAULT_TRACK = [
    "###......F",
    "##.......F",
    "##.......F",
    "#......###",
    "#......###",
    "#......###",
    "#SSSSS####",
]


class RacetrackEnv:
    """Small right-turn racetrack (Exercise 5.8 / 5.12), book Section 5.5/5.7.

    Observation: ``(row, col, vx, vy)`` with ``row``/``col`` the grid cell
    and ``vx``, ``vy`` in ``0..4``. ``step(action)`` accepts either an
    index into :data:`ACTIONS` (0-8) or an ``(dvx, dvy)`` tuple.

    Args:
        track: optional list of ASCII rows (see module docstring). Must
            contain at least one ``S`` and one ``F`` cell; all rows equal
            length.
        noise_prob: probability that the velocity increments are forced to
            zero on a step (0.1 in the book).
        seed: seed for the environment's RNG (start-cell sampling, boundary
            teleport sampling, and the noise coin).
    """

    ACTIONS = ACTIONS
    MAX_VELOCITY = MAX_VELOCITY

    def __init__(
        self,
        track: list[str] | None = None,
        noise_prob: float = 0.1,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        if not 0.0 <= noise_prob <= 1.0:
            raise ValueError("noise_prob must be in [0, 1]")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1 or None")
        track = track if track is not None else list(_DEFAULT_TRACK)
        if not track or any(len(row) != len(track[0]) for row in track):
            raise ValueError("track rows must all have equal, non-zero length")
        grid = np.array([list(row) for row in track])
        start_cells = [
            (int(r), int(c)) for r, c in zip(*np.where(grid == "S"))
        ]
        finish_cells = [
            (int(r), int(c)) for r, c in zip(*np.where(grid == "F"))
        ]
        if not start_cells:
            raise ValueError("track must contain at least one 'S' (start) cell")
        if not finish_cells:
            raise ValueError("track must contain at least one 'F' (finish) cell")
        on_track = grid != "#"
        self.track = track
        self.grid = grid
        self.on_track = on_track
        self.start_cells = start_cells
        self.finish_cells = finish_cells
        self.noise_prob = float(noise_prob)
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self._steps = 0
        self.n_rows, self.n_cols = grid.shape
        self.n_actions = N_ACTIONS
        self._state = (self.start_cells[0][0], self.start_cells[0][1], 0, 0)
        self.reset()

    # -- helpers ---------------------------------------------------------
    @property
    def state(self) -> tuple[int, int, int, int]:
        return self._state

    def n_states(self) -> int:
        on_track_count = int(self.on_track.sum())
        return on_track_count * (self.MAX_VELOCITY + 1) ** 2

    def projected_path(
        self, row: int, col: int, vx: int, vy: int
    ) -> list[tuple[int, int]]:
        """Cells visited when moving (vx right, vy up) from (row, col).

        Rounds the straight-line segment to grid cells, excluding the
        origin cell itself.
        """
        r_end, c_end = row - vy, col + vx
        dr, dc = r_end - row, c_end - col
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return []
        cells: list[tuple[int, int]] = []
        for t in range(1, steps + 1):
            r = int(round(row + dr * t / steps))
            c = int(round(col + dc * t / steps))
            if not cells or cells[-1] != (r, c):
                cells.append((r, c))
        return cells

    def _is_legal_position(self, r: int, c: int) -> bool:
        return (
            0 <= r < self.n_rows and 0 <= c < self.n_cols and bool(self.on_track[r, c])
        )

    def _is_finish(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols and self.grid[r, c] == "F"

    def _random_start(self) -> tuple[int, int, int, int]:
        idx = int(self.rng.integers(len(self.start_cells)))
        r, c = self.start_cells[idx]
        return (r, c, 0, 0)

    # -- env API ---------------------------------------------------------
    def reset(self) -> tuple[int, int, int, int]:
        """Reset to a random starting-line cell with velocity (0, 0)."""
        self._state = self._random_start()
        self._steps = 0
        return self._state

    def step(
        self, action: int | tuple[int, int]
    ) -> tuple[float, tuple[int, int, int, int], bool]:
        """Apply ``action`` and return ``(reward, observation, done)``."""
        row, col, vx, vy = self._state
        self._steps += 1
        if isinstance(action, tuple):
            if action not in ACTIONS:
                raise ValueError(f"action tuple {action} not in ACTIONS")
            dvx, dvy = action
        else:
            if not 0 <= int(action) < N_ACTIONS:
                raise ValueError(f"action index {action} out of range 0-8")
            dvx, dvy = ACTIONS[int(action)]
        # Book training condition: with probability noise_prob the velocity
        # increments are both zero (no acceleration this step).
        if self.rng.random() < self.noise_prob:
            dvx, dvy = 0, 0
        new_vx = int(np.clip(vx + dvx, 0, self.MAX_VELOCITY))
        new_vy = int(np.clip(vy + dvy, 0, self.MAX_VELOCITY))
        if new_vx == 0 and new_vy == 0:
            # Velocity may not be (0, 0) off the starting line.
            if (row, col) not in self.start_cells:
                new_vx, new_vy = vx, vy
        path = self.projected_path(row, col, new_vx, new_vy)
        cap_hit = self.max_steps is not None and self._steps >= self.max_steps
        # Finish is checked before the boundary, per the book's spec.
        if any(self._is_finish(r, c) for r, c in path):
            self._state = (row - new_vy, col + new_vx, new_vx, new_vy)
            return -1.0, self._state, True
        if any(not self._is_legal_position(r, c) for r, c in path):
            # Boundary hit: back to a random starting-line cell, v = (0, 0),
            # the episode continues.
            self._state = self._random_start()
            return -1.0, self._state, cap_hit
        self._state = (row - new_vy, col + new_vx, new_vx, new_vy)
        return -1.0, self._state, cap_hit

    def render(self) -> str:
        """ASCII rendering with the car marked as ``*``."""
        row, col, vx, vy = self._state
        lines = []
        for r, track_row in enumerate(self.track):
            line = list(track_row)
            if r == row:
                line[col] = "*"
            lines.append("".join(line))
        return "\n".join(lines)

    def copy(self) -> "RacetrackEnv":
        new = RacetrackEnv.__new__(RacetrackEnv)
        new.__dict__.update(self.__dict__)
        new.rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        return new
