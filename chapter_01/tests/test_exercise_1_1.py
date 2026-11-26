"""
Unit tests for Exercise 1.1 - Gridworld environment and agent simulation.

Book Reference: Sutton & Barto (2nd Ed.), Section 1.3, Figure 1.2
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exercise_1_1 import GreedyAgent, GridWorld, RandomAgent, run_episode, run_multiple_episodes


class TestGridWorldInit:
    """Tests for GridWorld initialization."""

    def test_default_initialization(self) -> None:
        env = GridWorld()
        assert env.grid_width == 4
        assert env.grid_height == 4
        assert env.discount == 1.0
        assert env.n_actions == 4
        assert env.state_A == (0, 0)
        assert env.state_B == (0, 3)
        assert env.state_A_plus == (3, 0)
        assert env.state_D == (3, 3)

    def test_custom_grid_size(self) -> None:
        env = GridWorld(grid_width=5, grid_height=5)
        assert env.grid_width == 5
        assert env.grid_height == 5

    def test_invalid_discount_raises(self) -> None:
        with pytest.raises(ValueError):
            GridWorld(discount=-0.1)
        with pytest.raises(ValueError):
            GridWorld(discount=1.1)

    def test_invalid_grid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            GridWorld(grid_width=1, grid_height=4)
        with pytest.raises(ValueError):
            GridWorld(grid_width=4, grid_height=1)

    def test_reset_returns_initial_state(self) -> None:
        env = GridWorld()
        state = env.reset()
        assert state == (0, 0)

    def test_reset_sets_state(self) -> None:
        env = GridWorld()
        env.reset()
        env.step(GridWorld.RIGHT)
        assert env.state == (0, 3)  # A->B via RIGHT
        env.reset()
        assert env.state == (0, 0)


class TestGridWorldStep:
    """Tests for GridWorld step transitions."""

    def test_normal_transition_right(self) -> None:
        env = GridWorld()
        env.reset()
        # From A(0,0), RIGHT goes to B(0,3)
        next_state, reward, done, info = env.step(GridWorld.RIGHT)
        assert next_state == (0, 3)
        assert reward == 5.0

    def test_normal_transition_up(self) -> None:
        env = GridWorld()
        env.reset()
        # From A(0,0), UP goes to A+(3,0)
        next_state, reward, done, info = env.step(GridWorld.UP)
        assert next_state == (3, 0)
        assert reward == 10.0
        assert done is True

    def test_normal_transition_down(self) -> None:
        env = GridWorld()
        env.reset()
        # From A(0,0), DOWN goes to B(0,3) with reward 5.0
        next_state, reward, done, info = env.step(GridWorld.DOWN)
        assert next_state == (0, 3)
        assert reward == 5.0

    def test_terminal_state_A_plus_from_up(self) -> None:
        env = GridWorld()
        env.reset()
        next_state, reward, done, info = env.step(GridWorld.UP)
        assert next_state == (3, 0)
        assert reward == 10.0
        assert done is True

    def test_terminal_state_A_plus_from_left(self) -> None:
        env = GridWorld()
        env.reset()
        next_state, reward, done, info = env.step(GridWorld.LEFT)
        assert next_state == (3, 0)
        assert reward == 10.0
        assert done is True

    def test_state_B_transition(self) -> None:
        env = GridWorld()
        env.reset()
        # Reach B via RIGHT from A
        next_state, reward, done, info = env.step(GridWorld.RIGHT)
        assert next_state == (0, 3)
        assert reward == 5.0
        # From B: UP/RIGHT -> C(0,2), DOWN/LEFT -> D(3,3)
        next_s, r, d, _ = env.step(GridWorld.UP)
        assert next_s == (0, 2)
        assert r == 5.0
        env.reset()
        env.step(GridWorld.RIGHT)  # A -> B
        next_s, r, d, _ = env.step(GridWorld.DOWN)
        assert next_s == (3, 3)  # D
        assert r == 5.0
        assert d is True

    def test_terminal_state_D(self) -> None:
        env = GridWorld()
        env.reset()
        # Path to D: A(0,0) -> DOWN/RIGHT -> B(0,3) -> DOWN/LEFT -> D(3,3)
        env.step(GridWorld.RIGHT)  # A -> B
        next_state, reward, done, _ = env.step(GridWorld.DOWN)  # B -> D
        assert next_state == (3, 3)
        assert done is True

    def test_invalid_action_raises(self) -> None:
        env = GridWorld()
        env.reset()
        with pytest.raises(ValueError):
            env.step(5)
        with pytest.raises(ValueError):
            env.step(-1)

    def test_step_without_reset_raises(self) -> None:
        env = GridWorld()
        with pytest.raises(RuntimeError):
            env.step(GridWorld.RIGHT)

    def test_terminal_returns_none_next_state(self) -> None:
        env = GridWorld()
        env.reset()
        env.step(GridWorld.UP)
        assert env.is_terminal is True
        next_state, reward, done, _ = env.step(GridWorld.RIGHT)
        assert next_state is None
        assert done is True


class TestGridWorldGetReward:
    """Tests for GridWorld.get_reward method."""

    def test_reward_A_up(self) -> None:
        env = GridWorld()
        assert env.get_reward((0, 0), GridWorld.UP) == 10.0
        assert env.get_reward((0, 0), GridWorld.LEFT) == 10.0
        assert env.get_reward((0, 0), GridWorld.DOWN) == 5.0
        assert env.get_reward((0, 0), GridWorld.RIGHT) == 5.0

    def test_reward_B(self) -> None:
        env = GridWorld()
        for action in range(4):
            assert env.get_reward((0, 3), action) == 5.0

    def test_reward_terminal_states(self) -> None:
        env = GridWorld()
        assert env.get_reward((3, 0), 0) == 0.0
        assert env.get_reward((3, 3), 0) == 0.0

    def test_reward_normal_states(self) -> None:
        env = GridWorld()
        assert env.get_reward((0, 1), GridWorld.RIGHT) == -1.0
        assert env.get_reward((1, 1), GridWorld.DOWN) == -1.0


class TestGridWorldIsTerminal:
    def test_non_terminal_state(self) -> None:
        env = GridWorld()
        env.reset()
        assert env.is_terminal is False
        env.step(GridWorld.RIGHT)
        assert env.is_terminal is False

    def test_terminal_A_plus(self) -> None:
        env = GridWorld()
        env.reset()
        env.step(GridWorld.UP)
        assert env.is_terminal is True

    def test_terminal_D(self) -> None:
        env = GridWorld()
        env.reset()
        # Path to D: A->DOWN->B->DOWN->D
        env.step(GridWorld.RIGHT)  # A -> B
        env.step(GridWorld.DOWN)   # B -> D
        assert env.is_terminal is True


class TestRandomAgent:
    def test_select_action_valid_index(self) -> None:
        agent = RandomAgent(n_actions=4)
        for _ in range(100):
            action = agent.select_action((0, 0))
            assert 0 <= action < 4

    def test_select_action_distribution(self) -> None:
        np.random.seed(42)
        agent = RandomAgent(n_actions=4)
        actions = [agent.select_action((0, 0)) for _ in range(1000)]
        counts = np.bincount(actions, minlength=4)
        for count in counts:
            assert 100 <= count <= 400


class TestGreedyAgent:
    def test_greedy_selects_best(self) -> None:
        agent = GreedyAgent(n_actions=4)
        agent.update_value((0, 0), 0, 1.0)
        agent.update_value((0, 0), 1, 2.0)
        agent.update_value((0, 0), 2, 0.5)
        agent.update_value((0, 0), 3, 1.5)
        for _ in range(10):
            assert agent.select_action((0, 0), epsilon=0.0) == 1

    def test_greedy_epsilon_exploration(self) -> None:
        np.random.seed(42)
        agent = GreedyAgent(n_actions=4)
        agent.update_value((0, 0), 0, 1.0)
        agent.update_value((0, 0), 1, 2.0)
        agent.update_value((0, 0), 2, 0.0)
        agent.update_value((0, 0), 3, 0.0)
        actions = [agent.select_action((0, 0), epsilon=0.5) for _ in range(1000)]
        assert 0 in actions or 2 in actions or 3 in actions

    def test_greedy_unknown_state(self) -> None:
        agent = GreedyAgent(n_actions=4)
        action = agent.select_action((0, 0), epsilon=0.0)
        assert 0 <= action < 4

    def test_greedy_none_state(self) -> None:
        agent = GreedyAgent(n_actions=4)
        action = agent.select_action(None, epsilon=0.0)
        assert 0 <= action < 4


class TestRunEpisode:
    def test_episode_ends_at_terminal(self) -> None:
        env = GridWorld()
        agent = RandomAgent(n_actions=4)
        np.random.seed(42)
        trajectory = run_episode(env, agent, max_steps=100)
        assert len(trajectory) > 0
        # Episode should end at either A+ or D, or at max_steps
        # With 100 steps, check that the episode completed (reached terminal)
        assert env.is_terminal or len(trajectory) < 100

    def test_episode_returns_trajectory(self) -> None:
        env = GridWorld()
        agent = RandomAgent(n_actions=4)
        np.random.seed(42)
        trajectory = run_episode(env, agent, max_steps=1000)
        assert isinstance(trajectory, list)
        for state, action, reward, next_state in trajectory:
            assert isinstance(state, tuple)
            assert len(state) == 2
            assert 0 <= action < 4
            assert isinstance(reward, float)

    def test_episode_with_greedy_agent(self) -> None:
        env = GridWorld()
        agent = GreedyAgent(n_actions=4)
        trajectory = run_episode(env, agent, max_steps=100)
        assert isinstance(trajectory, list)


class TestRunMultipleEpisodes:
    def test_runs_correct_number(self) -> None:
        env = GridWorld()
        agent = RandomAgent(n_actions=4)
        np.random.seed(42)
        stats = run_multiple_episodes(env, agent, n_episodes=10)
        assert len(stats["total_reward"]) == 10
        assert len(stats["steps"]) == 10

    def test_reward_values_finite(self) -> None:
        env = GridWorld()
        agent = RandomAgent(n_actions=4)
        np.random.seed(42)
        stats = run_multiple_episodes(env, agent, n_episodes=50)
        assert np.isfinite(np.mean(stats["total_reward"]))
        assert np.mean(stats["steps"]) > 0

    def test_stats_has_expected_keys(self) -> None:
        env = GridWorld()
        agent = RandomAgent(n_actions=4)
        np.random.seed(42)
        stats = run_multiple_episodes(env, agent, n_episodes=5)
        assert "total_reward" in stats
        assert "steps" in stats
