import numpy as np

from rl_envs.random_walk import RandomWalkEnv, random_walk_true_values
from rl_envs.blackjack import BlackjackEnv
from rl_envs.windy_gridworld import WindyGridWorld
from rl_envs.mountain_car import MountainCarEnv


def test_random_walk_basic():
    env = RandomWalkEnv(seed=0)
    s = env.reset()
    assert s == "C"
    tv = random_walk_true_values()
    assert tv == {"A": 1/6, "B": 2/6, "C": 3/6, "D": 4/6, "E": 5/6}


def test_blackjack_reset_step():
    env = BlackjackEnv(seed=0)
    obs = env.reset()
    assert len(obs) == 3
    # take a step if not terminal natural
    r, obs2, done = env.step("hit")
    assert isinstance(r, float)


def test_windy_gridworld_standard():
    env = WindyGridWorld()
    s = env.reset()
    assert s == (3, 0)
    s2, r, done = env.step("right")
    assert r == -1.0 and not done
    assert s2[1] == 1


def test_mountain_car():
    env = MountainCarEnv(seed=0)
    s = env.reset()
    assert env.POSITION_MIN <= s[0] <= env.POSITION_MAX
    s2, r, done = env.step(2)
    assert r == -1.0
    assert isinstance(done, bool)
