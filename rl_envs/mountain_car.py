"""Mountain Car environment (Sutton & Barto, 2nd ed., Example 10.1)."""

from __future__ import annotations

import math

import numpy as np


class MountainCarEnv:
    """Mountain Car task (Example 10.1). Book Section 10.1.

    State: (position, velocity).
      position in [-1.2, 0.5], velocity in [-0.07, 0.07].
    Start: position uniform in [-0.6, -0.4], velocity 0.
    Goal: position >= 0.5 (terminal). Reward -1 per step.
    Actions: 0=reverse (-1), 1=coast (0), 2=forward (+1).

    Dynamics (Eq. in Sec. 10.1):
      v_{t+1} = clip(v_t + 0.001*A_t - 0.0025*cos(3*x_t), -0.07, 0.07)
      x_{t+1} = clip(x_t + v_{t+1}, -1.2, 0.5)
    If x_{t+1} == -1.2 then v_{t+1} = 0.
    """

    ACTIONS = (0, 1, 2)  # reverse, coast, forward
    POSITION_MIN = -1.2
    POSITION_MAX = 0.5
    VELOCITY_MIN = -0.07
    VELOCITY_MAX = 0.07

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.position: float = 0.0
        self.velocity: float = 0.0

    def reset(self, position: float | None = None, velocity: float = 0.0):
        if position is None:
            self.position = float(self.rng.uniform(-0.6, -0.4))
        else:
            self.position = float(position)
        self.velocity = float(velocity)
        return (self.position, self.velocity)

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")
        thrust = action - 1  # -1, 0, +1
        v = self.velocity + 0.001 * thrust - 0.0025 * math.cos(3.0 * self.position)
        v = float(np.clip(v, self.VELOCITY_MIN, self.VELOCITY_MAX))
        x = self.position + v
        x = float(np.clip(x, self.POSITION_MIN, self.POSITION_MAX))
        if x <= self.POSITION_MIN:
            v = 0.0
        self.position, self.velocity = x, v
        done = self.position >= self.POSITION_MAX
        reward = -1.0
        return (self.position, self.velocity), reward, done
