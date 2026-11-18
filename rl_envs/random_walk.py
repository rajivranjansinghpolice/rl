"""Random walk environment (Sutton & Barto, 2nd ed., Example 6.2, Sec. 6.2)."""

from __future__ import annotations

import numpy as np


class RandomWalkEnv:
    """Random walk environment (Example 6.2). Book Section 6.2.

    States: 'A'..'E' by default (parameterize with `n_states`/`states`).
    Episodes start in the middle state. The equiprobable random
    policy moves left or right at random. Terminal states ('left_end',
    'right_end') give `left_reward` (default 0) or `right_reward` (default +1)
    on entry; gamma = 1. The 2nd-ed. Chapter 7 examples use a 19-state walk
    (A..S) with left_reward=-1.0, right_reward=+1.0 (Example 7.1, Figure 7.2).
    """

    STATES = ("A", "B", "C", "D", "E")
    ACTIONS = ("left", "right")

    def __init__(
        self,
        seed: int | None = None,
        n_states: int = 5,
        states: tuple[str, ...] | list[str] | None = None,
        left_reward: float = 0.0,
        right_reward: float = 1.0,
    ) -> None:
        if states is not None:
            self.STATES = tuple(states)
        elif n_states != 5:
            self.STATES = tuple(chr(ord("A") + i) for i in range(n_states))
        # otherwise the class default ("A".."E") is kept
        if len(self.STATES) < 2:
            raise ValueError("need at least 2 states")
        self.n_states = len(self.STATES)
        self.left_reward = float(left_reward)
        self.right_reward = float(right_reward)
        self.start_state = self.STATES[self.n_states // 2]
        self.rng = np.random.default_rng(seed)
        self.state: str | None = None

    def reset(self) -> str:
        self.state = self.start_state
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
            return self.left_reward, self.state, True
        if index >= len(self.STATES):
            self.state = "right_end"
            return self.right_reward, self.state, True
        self.state = self.STATES[index]
        return 0.0, self.state, False

    def true_values(self) -> dict:
        """Ground-truth state values under the equiprobable random policy.

        Linear interpolation between the terminal rewards: v*(i) =
        left_reward + (i + 1) * (right_reward - left_reward) / (N + 1).
        For the default 5-state walk this gives (i + 1) / 6.
        """
        span = self.right_reward - self.left_reward
        return {
            state: self.left_reward + (i + 1) * span / (self.n_states + 1)
            for i, state in enumerate(self.STATES)
        }


def random_walk_true_values() -> dict:
    """Ground-truth state values under the equiprobable random policy.

    v(s) = (position + 1) / 6 for positions A..E (0-based), terminals 0/1.
    """
    return RandomWalkEnv().true_values()
