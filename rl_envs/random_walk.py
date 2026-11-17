"""Random walk environment (Sutton & Barto, 2nd ed., Example 6.2, Sec. 6.2)."""

from __future__ import annotations

import numpy as np


class RandomWalkEnv:
    """Random walk environment (Example 6.2). Book Section 6.2.

    States: 'A'..'E'. Episodes start in state 'C'. The equiprobable random
    policy moves left or right at random. Terminal states ('left_end',
    'right_end') give reward +1 (right) or 0 (left) on entry; gamma = 1.
    """

    STATES = ("A", "B", "C", "D", "E")
    ACTIONS = ("left", "right")

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.state: str | None = None

    def reset(self) -> str:
        self.state = "C"
        return self.state

    def step(self, action: str) -> tuple:
        if self.state is None:
            raise RuntimeError("call reset() first")
        if action not in self.ACTIONS:
            raise ValueError("action must be 'left' or 'right'")
        index = self.STATES.index(self.state)
        if action == "left":
            index -= 1
        else:
            index += 1
        if index < 0:
            self.state = "left_end"
            return 0.0, self.state, True
        if index >= len(self.STATES):
            self.state = "right_end"
            return 1.0, self.state, True
        self.state = self.STATES[index]
        return 0.0, self.state, False


def random_walk_true_values() -> dict:
    """Ground-truth state values under the equiprobable random policy.

    v(s) = (position + 1) / 6 for positions A..E (0-based), terminals 0/1.
    """
    return {state: (i + 1) / 6.0 for i, state in enumerate(RandomWalkEnv.STATES)}
