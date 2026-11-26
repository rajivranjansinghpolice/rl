"""Example 8.5 (Sutton & Barto, 2nd ed., Sec. 8.4): Rod Maneuvering.

The book describes the task as follows: move a rod around awkwardly placed
obstacles in a limited rectangular workspace, in as few steps as possible.
The rod can be translated along its long axis or perpendicular to it, and
rotated in either direction about its center. Moves are about 1/20 of the
workspace, rotations are 10 degrees, translations are quantized to a
20 x 20 grid, and there are 14,400 possible states (20 x 20 positions x 36
orientations). The book reports that prioritized sweeping found the shortest
solution; the figure is reprinted from Moore and Atkeson (1993) and does not
print the obstacle coordinates, so the layout here is an **illustrative
reconstruction**, not a pixel-level copy.

Everything in this module is deterministic once the layout is fixed. Instead
of pretending to rediscover a model from thousands of random episodes, we
build the exact one-step model of the (finite) rod MDP and run prioritized
sweeping with a known model. This isolates the book's actual point: priority
ordering spends its updates on states relevant to the start-goal path, unlike
exhaustive value-iteration sweeps.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

from chapter_08.solution import PriorityQueue

ACTIONS = ("forward", "side", "rotate_cw", "rotate_ccw")
N_ORIENTATIONS = 36  # 10-degree increments
GRID_SIZE = 20
ROD_HALF_LENGTH = 1.5  # the rod occupies roughly three grid cells

# Illustrative obstacles. The book figure gives no cell coordinates.
DEFAULT_OBSTACLES: tuple[tuple[int, int], ...] = tuple(
    [(x, y) for x in (7, 8) for y in range(0, 13)]
    + [(x, y) for x in range(10, 18) for y in (6, 7)]
    + [(x, y) for x in (4, 5) for y in range(13, 18)]
)


class RodWorld:
    """Rod-manipulation MDP of Example 8.5 (illustrative reconstruction).

    A state is ``(x, y, k)``: the rod center on the 20 x 20 cell grid and an
    orientation index ``k`` in ``0..35`` (``k * 10`` degrees). Actions are
    four: translate one cell along the rod, one cell perpendicular to it, or
    rotate 10 degrees clockwise/counter-clockwise. A move that would collide
    with an obstacle or leave the workspace leaves the rod where it is.
    Entering the goal state gives +1 and terminates the episode.

    Args:
        obstacles: iterable of ``(x, y)`` blocked cells.
        start: initial ``(x, y, k)`` configuration.
        goal: terminal ``(x, y, k)`` configuration.
    """

    actions = ACTIONS
    gamma = 0.95

    def __init__(
        self,
        obstacles: Iterable[tuple[int, int]] | None = None,
        start: tuple[int, int, int] = (2, 2, 0),
        goal: tuple[int, int, int] = (17, 17, 0),
    ) -> None:
        self.obstacles = frozenset(
            DEFAULT_OBSTACLES if obstacles is None else obstacles
        )
        self.start_state = tuple(start)
        self.goal = tuple(goal)
        self._state = self.start_state
        self.done = False
        self._cells: dict[tuple[int, int, int], frozenset[tuple[int, int]]] = {}
        self._states: list[tuple[int, int, int]] = []
        for k in range(N_ORIENTATIONS):
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    cells = self._occupied(x, y, k)
                    if not cells:
                        continue
                    self._cells[(x, y, k)] = cells
                    self._states.append((x, y, k))
        self._valid = frozenset(self._states)
        if self.start_state not in self._valid or self.goal not in self._valid:
            raise ValueError("start and goal configurations must be collision-free")

    @property
    def states(self) -> list[tuple[int, int, int]]:
        return list(self._states)

    @property
    def terminal_states(self) -> frozenset[tuple[int, int, int]]:
        return frozenset({self.goal})

    def reset(self, start_state: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
        state = self.start_state if start_state is None else tuple(start_state)
        if state not in self._valid:
            raise ValueError(f"invalid rod configuration: {state}")
        self._state = state
        self.done = state == self.goal
        return state

    def _occupied(self, x: int, y: int, k: int) -> frozenset[tuple[int, int]]:
        """Cells overlapped by the rod, or an empty set if it is invalid."""
        theta = np.deg2rad(10 * k)
        ux, uy = float(np.cos(theta)), float(np.sin(theta))
        cells: set[tuple[int, int]] = set()
        for offset in np.linspace(-ROD_HALF_LENGTH, ROD_HALF_LENGTH, 9):
            cx = int(round(x + offset * ux))
            cy = int(round(y + offset * uy))
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                return frozenset()
            if (cx, cy) in self.obstacles:
                return frozenset()
            cells.add((cx, cy))
        return frozenset(cells)

    def _next(self, state: tuple[int, int, int], action: str) -> tuple[int, int, int]:
        x, y, k = state
        if action == "rotate_cw":
            candidate = (x, y, (k + 1) % N_ORIENTATIONS)
        elif action == "rotate_ccw":
            candidate = (x, y, (k - 1) % N_ORIENTATIONS)
        else:
            theta = np.deg2rad(10 * k)
            ux, uy = float(np.cos(theta)), float(np.sin(theta))
            if action == "forward":
                dx, dy = int(round(ux)), int(round(uy))
            elif action == "side":
                dx, dy = int(round(-uy)), int(round(ux))
            else:
                raise ValueError(f"unknown rod action: {action}")
            candidate = (x + dx, y + dy, k)
        return candidate if candidate in self._valid else state  # collided: stay

    def step(self, action: str) -> tuple[float, tuple[int, int, int], bool]:
        if self.done:
            raise RuntimeError("episode is already over; call reset() first")
        next_state = self._next(self._state, action)
        self._state = next_state
        self.done = next_state == self.goal
        return (1.0 if self.done else 0.0), next_state, self.done

    def build_model(self) -> dict[tuple, list[tuple[float, float, tuple]]]:
        """Exact deterministic model; (state, action) -> [(prob, reward, next)]."""
        model = {}
        for state in self._states:
            if state == self.goal:
                continue
            for action in ACTIONS:
                nxt = self._next(state, action)
                reward = 1.0 if nxt == self.goal else 0.0
                model[(state, action)] = [(1.0, reward, nxt)]
        return model

    def occupied_cells(self, state: tuple[int, int, int]) -> frozenset[tuple[int, int]]:
        return self._cells[state]


# ---------------------------------------------------------------------------
# Planners over the exact model
# ---------------------------------------------------------------------------


def _expected_target(q: dict, model: dict, state, action: str, gamma: float) -> float:
    """One-step expected Q target; terminal states have no outgoing pairs."""
    total = 0.0
    for prob, reward, nxt in model[(state, action)]:
        if (nxt, ACTIONS[0]) in q:
            total += prob * (reward + gamma * max(q[(nxt, a)] for a in ACTIONS))
        else:  # goal state: no bootstrap
            total += prob * reward
    return total


def _successors(model: dict) -> dict:
    preds: dict = {}
    for (state, _action), outcomes in model.items():
        for _p, _r, nxt in outcomes:
            preds.setdefault(nxt, set()).add(state)
    return preds


def shortest_path(model: dict, start, goal) -> list | None:
    """BFS shortest action sequence on the deterministic rod MDP."""
    parents: dict = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == goal:
            path = []
            while state is not None:
                path.append(state)
                state = parents[state]
            return list(reversed(path))
        for action in ACTIONS:
            if (state, action) not in model:
                continue
            _p, _r, nxt = model[(state, action)][0]
            if nxt not in parents:
                parents[nxt] = state
                queue.append(nxt)
    return None


def _greedy_path(q: dict, model: dict, start, goal, budget: int = 200) -> list | None:
    """Follow argmax_Q from start, breaking ties by the ACTIONS order."""
    state = start
    path = [state]
    for _ in range(budget):
        if state == goal:
            return path
        action = max(ACTIONS, key=lambda a: q[(state, a)])
        _p, _r, state = model[(state, action)][0]
        path.append(state)
    return None


def prioritized_sweeping_known_model(
    model: dict,
    start,
    goal,
    gamma: float = 0.95,
    theta: float = 1e-4,
    max_updates: int = 200_000,
) -> tuple[dict, int]:
    """Prioritized sweeping with a known deterministic model (Example 8.5).

    Updates are in-place expected (value-iteration) backups, alpha = 1. The
    initial queue contains all actions at ``start``; predecessors are pushed
    whenever an update changes Q by more than ``theta``. Returns the learned
    Q table and the number of updates made before the greedy path from
    ``start`` first matches the BFS shortest path.
    """
    q = {pair: 0.0 for pair in model}
    preds = _successors(model)
    optimal = shortest_path(model, start, goal)
    if optimal is None:
        raise ValueError("goal is unreachable from start in this layout")
    optimal_len = len(optimal) - 1
    queue = PriorityQueue()

    def push_pair(state, action: str) -> None:
        target = _expected_target(q, model, state, action, gamma)
        priority = abs(target - q[(state, action)])
        if priority > theta:
            queue.push((state, action), priority)

    # With all values initialised to zero, only pairs with an immediate
    # reward have a nonzero priority. Seed those (the goal-adjacent pairs)
    # and let predecessor propagation carry the change back toward start.
    for pair in q:
        push_pair(*pair)
    updates = 0
    while not queue.empty() and updates < max_updates:
        (state, action), _priority = queue.pop()
        target = _expected_target(q, model, state, action, gamma)
        q[(state, action)] = target
        updates += 1
        if updates % 25 == 0:
            greedy = _greedy_path(q, model, start, goal, budget=optimal_len + 20)
            if greedy is not None and len(greedy) - 1 == optimal_len:
                break
        for pred in preds.get(state, ()):
            for pred_action in ACTIONS:
                if (pred, pred_action) in q:
                    push_pair(pred, pred_action)
    return q, updates


def value_iteration_known_model(
    model: dict,
    start,
    goal,
    gamma: float = 0.95,
    theta: float = 1e-6,
    max_updates: int = 2_000_000,
) -> tuple[dict, int]:
    """Gauss-Seidel value iteration over *all* pairs, for the comparison."""
    q = {pair: 0.0 for pair in model}
    optimal = shortest_path(model, start, goal)
    optimal_len = len(optimal) - 1
    updates = 0
    for _ in range(max_updates // max(1, len(model)) + 1):
        delta = 0.0
        for (state, action), outcomes in model.items():
            target = _expected_target(q, model, state, action, gamma)
            delta = max(delta, abs(target - q[(state, action)]))
            q[(state, action)] = target
            updates += 1
        greedy = _greedy_path(q, model, start, goal, budget=optimal_len + 20)
        if greedy is not None and len(greedy) - 1 == optimal_len:
            break
        if delta < theta:
            break
    return q, updates


def rod_experiment() -> dict:
    """Run both planners on Example 8.5 and report update counts."""
    world = RodWorld()
    model = world.build_model()
    path = shortest_path(model, world.start_state, world.goal)
    if path is None:
        raise RuntimeError("no solution in the illustrative rod layout")
    _, ps_updates = prioritized_sweeping_known_model(
        model, world.start_state, world.goal
    )
    _, vi_updates = value_iteration_known_model(
        model, world.start_state, world.goal
    )
    return {
        "n_states": len(world.states),
        "n_model_pairs": len(model),
        "shortest_path_steps": len(path) - 1,
        "path": path,
        "prioritized_sweeping_updates": ps_updates,
        "value_iteration_updates": vi_updates,
    }


def plot_example_8_5_rod(save_path: str | None = None, show: bool = False):
    """Draw the rod workspace with the prioritized-sweeping shortest path.

    The book's figure is reprinted from Moore and Atkeson (1993) and prints
    no obstacle coordinates; this is therefore an illustrative reconstruction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    world = RodWorld()
    result = rod_experiment()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(color="0.92", linewidth=0.5)
    blocked = np.array([[x, y] for x, y in world.obstacles], dtype=float).reshape(-1, 2)
    if len(blocked):
        ax.scatter(blocked[:, 0], blocked[:, 1], s=150, marker="s",
                   color="0.35", label="obstacles")

    path = result["path"]
    xs = [s[0] for s in path]
    ys = [s[1] for s in path]
    ax.plot(xs, ys, "-o", color="tab:blue", linewidth=1.5, markersize=3,
            label=f"shortest path ({result['shortest_path_steps']} steps)")
    for label, state, color in (("start", world.start_state, "tab:green"),
                                ("goal", world.goal, "tab:red")):
        cells = list(world.occupied_cells(state))
        ax.scatter([c[0] for c in cells], [c[1] for c in cells], s=45,
                   color=color, label=label, zorder=3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("workspace cell (x)")
    ax.set_ylabel("workspace cell (y)")
    ax.set_title(
        "S&B (2nd ed.), Sec. 8.4, Example 8.5: rod maneuvering\n"
        "illustrative reconstruction; prioritized-sweeping shortest path\n"
        f"P.S. updates = {result['prioritized_sweeping_updates']:,}; "
        f"full-sweep updates = {result['value_iteration_updates']:,}",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, result
