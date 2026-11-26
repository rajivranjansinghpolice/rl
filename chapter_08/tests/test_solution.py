"""
Tests for Chapter 8: Planning and Learning with Tabular Methods.

Reference: Sutton & Barto (2nd Ed), Chapter 8.

Tests for: Alg 8.1 (Dyna-Q planner, book Sec 8.2), Exercise 8.1
(prioritized sweeping, book Sec 8.4), Exercise 8.2 (expected vs sample
updates Dyna, book Sec 8.5), Exercise 8.3 (world model for stochastic
environments, book Sec 8.1-8.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_08.solution import (
    PriorityQueue,
    TabularModel,
    compare_sample_vs_expected_dyna,
    dyna_q,
    dyna_q_plus,
    expected_dyna_q,
    expected_sweep,
    make_gridworld,
    optimal_q_values,
    prioritized_sweeping,
)


def greedy_episode_steps(q_table, seed: int = 0, max_steps: int = 1000) -> int:
    """Simulate a greedy episode on the deterministic gridworld."""
    env = make_gridworld(slip=0.0, seed=seed)
    state = env.reset("0,0")
    rng = np.random.default_rng(seed)
    done = env.done
    steps = 0
    actions = list(env.actions)
    while not done and steps < max_steps:
        q_values = np.array([q_table[(state, a)] for a in actions], dtype=np.float64)
        best = np.flatnonzero(q_values == q_values.max())
        action = actions[int(rng.choice(best))]
        _, state, done = env.step(action)
        steps += 1
    return steps


class TestMakeGridworld:
    """Gridworld construction and slip dynamics. Book Sec 8.2 (Example 8.1)."""

    def test_dimensions(self) -> None:
        env = make_gridworld()
        assert env.n_states == 8
        assert env.n_actions == 4
        assert env.terminal_states == {"1,3"}

    def test_invalid_goal_raises(self) -> None:
        with pytest.raises(ValueError):
            make_gridworld(goal="9,9")

    def test_slip_must_be_valid(self) -> None:
        with pytest.raises(ValueError):
            make_gridworld(slip=1.0)

    def test_blocked_move_stays_in_place(self) -> None:
        env = make_gridworld()
        outcomes = env.transition_probs("0,0", "up")
        assert outcomes == [(1.0, 0.0, "0,0")]

    def test_slip_produces_multiple_outcomes(self) -> None:
        env = make_gridworld(slip=0.2)
        outcomes = env.transition_probs("0,1", "right")
        probs = [p for p, _, _ in outcomes]
        assert probs == pytest.approx([1 - 0.2, 0.1, 0.1])
        assert sum(probs) == pytest.approx(1.0)
        assert len(outcomes) == 3

    def test_deterministic_single_outcome(self) -> None:
        env = make_gridworld(slip=0.0)
        for s in env.states:
            if s in env.terminal_states:
                continue
            for a in env.actions:
                assert len(env.transition_probs(s, a)) == 1

    def test_goal_reward_is_terminal(self) -> None:
        env = make_gridworld()
        outcomes = env.transition_probs("0,3", "down")
        assert outcomes == [(1.0, 1.0, "1,3")]


class TestTabularModel:
    """Distribution (counting) model. Book Sec 8.1; Exercise 9.3."""

    def test_unknown_pair_distribution_raises(self) -> None:
        model = TabularModel()
        with pytest.raises(KeyError):
            model.distribution("s", "a")

    def test_distribution_sums_to_one(self) -> None:
        model = TabularModel()
        env = make_gridworld(slip=0.2, seed=0)
        for _ in range(150):
            env.reset("0,1")
            reward, ns, _ = env.step("right")
            model.observe("0,1", "right", reward, ns)
        dist = model.distribution("0,1", "right")
        assert sum(p for p, _, _ in dist) == pytest.approx(1.0)

    def test_distribution_reflects_counts(self) -> None:
        model = TabularModel()
        for _ in range(60):
            model.observe("s", "a", 1.0, "s1")
        for _ in range(40):
            model.observe("s", "a", -1.0, "s2")
        dist = model.distribution("s", "a")
        probs = {ns: p for p, _, ns in dist}
        assert probs["s1"] == pytest.approx(0.6)
        assert probs["s2"] == pytest.approx(0.4)

    def test_stochastic_model_does_not_collapse_to_mean(self) -> None:
        # Exercise 9.3: the model must keep the outcome distribution,
        # not average rewards into a single fictitious outcome.
        model = TabularModel()
        for _ in range(50):
            model.observe("s", "a", 1.0, "s1")
            model.observe("s", "a", 0.0, "s2")
        dist = model.distribution("s", "a")
        assert len(dist) == 2
        assert {r for _, r, _ in dist} == {0.0, 1.0}

    def test_sample_returns_observed_outcomes(self) -> None:
        model = TabularModel()
        model.observe("s", "a", 1.0, "s1")
        model.observe("s", "a", 0.0, "s2")
        rng = np.random.default_rng(0)
        outcomes = {model.sample("s", "a", rng) for _ in range(50)}
        assert outcomes <= {(1.0, "s1"), (0.0, "s2")}

    def test_predecessors(self) -> None:
        model = TabularModel()
        model.observe("a", "x", 0.0, "b")
        model.observe("c", "y", 0.0, "b")
        model.observe("d", "z", 0.0, "e")
        preds = model.predecessors("b")
        assert set(preds) == {("a", "x"), ("c", "y")}

    def test_sample_known_pair_comes_from_observed(self) -> None:
        model = TabularModel()
        model.observe("s", "a", 0.0, "s2")
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert model.sample_known_pair(rng) == ("s", "a")


class TestDynaQ:
    """Tabular Dyna-Q. Book Sec 8.2 (repo Alg 9.1)."""

    def test_stats_structure(self) -> None:
        env = make_gridworld(slip=0.0, seed=0)
        _, _, stats = dyna_q(env, n_planning=5, n_episodes=6, seed=0, start_state="0,0")
        assert len(stats["episode_steps"]) == 6
        assert len(stats["episode_rewards"]) == 6
        assert stats["planning_updates"] == 5 * sum(stats["episode_steps"])

    def test_learns_short_path(self) -> None:
        q_table, _, stats = dyna_q(
            make_gridworld(slip=0.0, seed=0),
            n_planning=20,
            n_episodes=15,
            seed=0,
            start_state="0,0",
        )
        assert np.mean(stats["episode_steps"][-3:]) <= 5.0
        assert greedy_episode_steps(q_table, seed=0) <= 6

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_planning_accelerates_learning(self, seed: int) -> None:
        _, _, no_plan = dyna_q(
            make_gridworld(slip=0.0, seed=seed),
            n_planning=0, n_episodes=10, seed=seed, start_state="0,0",
        )
        _, _, with_plan = dyna_q(
            make_gridworld(slip=0.0, seed=seed),
            n_planning=50, n_episodes=10, seed=seed, start_state="0,0",
        )
        assert np.mean(with_plan["episode_steps"][1:]) < np.mean(no_plan["episode_steps"][1:])

    def test_alpha_and_epsilon_validated(self) -> None:
        env = make_gridworld(seed=0)
        with pytest.raises(ValueError):
            dyna_q(env, alpha=0.0)
        with pytest.raises(ValueError):
            dyna_q(env, epsilon=0.0)

    def test_reproducible_with_seed(self) -> None:
        _, _, s1 = dyna_q(make_gridworld(seed=0), n_episodes=5, seed=42, start_state="0,0")
        _, _, s2 = dyna_q(make_gridworld(seed=0), n_episodes=5, seed=42, start_state="0,0")
        assert s1["episode_steps"] == s2["episode_steps"]


class TestPriorityQueue:
    """Priority queue with lazy deletion. Book Sec 8.4; Exercise 9.1."""

    def test_pops_highest_priority_first(self) -> None:
        queue = PriorityQueue()
        queue.push(("a", "x"), 0.5)
        queue.push(("b", "y"), 0.9)
        queue.push(("c", "z"), 0.7)
        assert queue.pop() == (("b", "y"), pytest.approx(0.9))
        assert queue.pop() == (("c", "z"), pytest.approx(0.7))
        assert queue.pop() == (("a", "x"), pytest.approx(0.5))

    def test_stale_lower_priority_entry_is_skipped(self) -> None:
        queue = PriorityQueue()
        queue.push(("a", "x"), 0.5)
        queue.push(("a", "x"), 1.5)  # supersedes the previous entry
        queue.push(("b", "y"), 0.9)
        assert queue.pop() == (("a", "x"), pytest.approx(1.5))
        assert queue.pop() == (("b", "y"), pytest.approx(0.9))
        assert queue.pop() is None  # the stale 0.5 entry was discarded

    def test_len_counts_non_stale_entries(self) -> None:
        queue = PriorityQueue()
        queue.push(("a", "x"), 0.5)
        queue.push(("a", "x"), 1.5)
        queue.push(("b", "y"), 0.9)
        assert len(queue) == 2

    def test_non_positive_priority_not_queued(self) -> None:
        queue = PriorityQueue()
        queue.push(("a", "x"), 0.0)
        queue.push(("a", "x"), -1.0)
        assert queue.empty()
        assert queue.pop() is None


class TestPrioritizedSweeping:
    """Prioritized sweeping. Book Sec 8.4; Exercise 9.1."""

    def test_theta_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            prioritized_sweeping(make_gridworld(seed=0), theta=0.0)

    def test_high_threshold_suppresses_planning_updates(self) -> None:
        _, _, stats = prioritized_sweeping(
            make_gridworld(slip=0.0, seed=0),
            theta=100.0, n_planning=10, n_episodes=3, seed=0, start_state="0,0",
        )
        assert stats["queue_pushes"] == 0
        assert stats["updates"] == 0

    def test_small_threshold_runs_updates(self) -> None:
        _, _, stats = prioritized_sweeping(
            make_gridworld(slip=0.0, seed=0),
            theta=1e-4, n_planning=10, n_episodes=5, seed=0, start_state="0,0",
        )
        assert stats["updates"] > 0
        assert stats["queue_pushes"] > 0

    def test_propagates_values_backward_to_start(self) -> None:
        q_table, _, stats = prioritized_sweeping(
            make_gridworld(slip=0.0, seed=0),
            theta=1e-4, n_planning=25, n_episodes=4, seed=0, start_state="0,0",
        )
        # after a few episodes, optimal-path actions from the start state are
        # strictly better than the others, and the greedy path reaches the goal
        best = max(q_table[("0,0", a)] for a in ("up", "down", "left", "right"))
        assert best > 0.0
        assert q_table[("0,0", "up")] < best
        assert greedy_episode_steps(q_table, seed=0) <= 6
        assert stats["updates"] > 10

    def test_learning_is_reproducible(self) -> None:
        _, _, s1 = prioritized_sweeping(
            make_gridworld(seed=0), n_episodes=4, seed=7, start_state="0,0"
        )
        _, _, s2 = prioritized_sweeping(
            make_gridworld(seed=0), n_episodes=4, seed=7, start_state="0,0"
        )
        assert s1["episode_steps"] == s2["episode_steps"]
        assert s1["updates"] == s2["updates"]


class TestExpectedVsSampleDyna:
    """Expected vs sample updates. Book Sec 8.5; Exercise 9.2."""

    def test_expected_sweep_recovers_q_star(self) -> None:
        env = make_gridworld(slip=0.2, seed=0)
        model = TabularModel()
        # build a near-perfect model from exact outcome counts
        for s in env.states:
            if s in env.terminal_states:
                continue
            for a in env.actions:
                for p, r, ns in env.transition_probs(s, a):
                    for _ in range(round(p * 2000)):
                        model.observe(s, a, r, ns)
        q_table = {(s, a): 0.0 for s in env.states for a in env.actions}
        terminal = set(env.terminal_states)
        expected_sweep(model, q_table, env.gamma, list(env.actions), terminal,
                       sweeps=300)
        q_star = optimal_q_values(env)
        error = max(
            abs(q_table[p] - q_star[p]) for p in q_table if p[0] not in terminal
        )
        assert error < 1e-6

    def test_expected_sweep_reduces_error(self) -> None:
        env = make_gridworld(slip=0.1, seed=0)
        model = TabularModel()
        for s in env.states:
            if s in env.terminal_states:
                continue
            for a in env.actions:
                for p, r, ns in env.transition_probs(s, a):
                    for _ in range(round(p * 500)):
                        model.observe(s, a, r, ns)
        q_star = optimal_q_values(env)
        q_table = {(s, a): 0.0 for s in env.states for a in env.actions}
        before = max(
            abs(q_table[p] - q_star[p]) for p in q_table if p[0] not in env.terminal_states
        )
        expected_sweep(model, q_table, env.gamma, list(env.actions),
                       set(env.terminal_states), sweeps=20)
        after = max(
            abs(q_table[p] - q_star[p]) for p in q_table if p[0] not in env.terminal_states
        )
        assert after < before * 0.01

    def test_compare_sample_vs_expected(self) -> None:
        env_factory = lambda seed: make_gridworld(slip=0.1, seed=seed)  # noqa: E731
        results = [
            compare_sample_vs_expected_dyna(
                env_factory, n_planning=20, n_episodes=30, seed=seed, start_state="0,0"
            )
            for seed in range(4)
        ]
        sample_err = np.mean([r["sample"]["known_max_q_error"] for r in results])
        expected_err = np.mean([r["expected"]["known_max_q_error"] for r in results])
        # both learn the stochastic MDP; expected updates are at least as good
        # on average in this small-branching setting
        assert sample_err < 0.5
        assert expected_err < 0.5
        assert expected_err <= sample_err + 0.05

    def test_compare_returns_expected_structure(self) -> None:
        env_factory = lambda seed: make_gridworld(slip=0.1, seed=seed)  # noqa: E731
        result = compare_sample_vs_expected_dyna(
            env_factory, n_planning=5, n_episodes=5, seed=0, start_state="0,0"
        )
        assert set(result) == {"sample", "expected", "q_star_max"}
        for kind in ("sample", "expected"):
            assert set(result[kind]) == {
                "max_q_error",
                "known_max_q_error",
                "episode_steps",
                "episode_rewards",
                "q_table",
            }

    def test_expected_dyna_learns_short_path(self) -> None:
        _, _, stats = expected_dyna_q(
            make_gridworld(slip=0.1, seed=0),
            n_planning=20, n_episodes=40, seed=0, start_state="0,0",
        )
        assert np.mean(stats["episode_steps"][-5:]) <= 10.0


class TestStochasticWorldModel:
    """World model for stochastic environments. Book Sec 8.1-8.3; Exercise 9.3."""

    def test_stochastic_dyna_learns_comparably_to_deterministic(self) -> None:
        _, _, det_stats = dyna_q(
            make_gridworld(slip=0.0, seed=0),
            n_planning=10, n_episodes=15, seed=0, start_state="0,0",
        )
        _, _, stoch_stats = dyna_q(
            make_gridworld(slip=0.1, seed=0),
            n_planning=10, n_episodes=15, seed=0, start_state="0,0",
        )
        det_final = np.mean(det_stats["episode_steps"][-5:])
        stoch_final = np.mean(stoch_stats["episode_steps"][-5:])
        # slips lengthen optimal paths slightly, but learning is comparably effective
        assert stoch_final <= det_final + 4.0
        assert stoch_final <= 9.0
        assert stoch_stats["episode_steps"][-1] < stoch_stats["episode_steps"][0]

    def test_model_estimates_outcome_distribution(self) -> None:
        env = make_gridworld(slip=0.2, seed=0)
        model = TabularModel()
        # exact outcome counts (deterministic model coverage):
        # 'right' from (0,1): 0.8 -> (0,2); 0.1 -> up blocked -> (0,1);
        # 0.1 -> down -> (1,1)
        for p, r, ns in env.transition_probs("0,1", "right"):
            for _ in range(round(p * 2000)):
                model.observe("0,1", "right", r, ns)
        dist = {ns: (r, p) for p, r, ns in model.distribution("0,1", "right")}
        assert len(dist) == 3
        assert dist["0,2"][1] == pytest.approx(0.8)
        assert dist["0,1"][1] == pytest.approx(0.1)
        assert dist["1,1"][1] == pytest.approx(0.1)
        # only the intended move into the terminal goal carries reward 1.0
        goal_outcomes = env.transition_probs("0,3", "down")
        good = [o for o in goal_outcomes if o[1] == pytest.approx(1.0)]
        assert len(good) == 1
        assert good[0][2] == "1,3"
        assert good[0][0] == pytest.approx(0.8)

    def test_deterministic_world_single_outcome_model(self) -> None:
        _, model, _ = dyna_q(
            make_gridworld(slip=0.0, seed=0), n_planning=2, n_episodes=5,
            seed=0, start_state="0,0",
        )
        for pair in model.known_pairs():
            assert len(model.distribution(*pair)) == 1

    def test_stochastic_world_multi_outcome_model(self) -> None:
        _, model, _ = dyna_q(
            make_gridworld(slip=0.2, seed=0), n_planning=2, n_episodes=10,
            seed=0, start_state="0,0",
        )
        multi = [p for p in model.known_pairs() if len(model.distribution(*p)) > 1]
        assert multi, "expected at least one stochastic pair to be learned"


# ---------------------------------------------------------------------------
# Example 8.1-8.6 / Sections 8.3, 8.6, 8.7 (added for the book-aligned revision)
# ---------------------------------------------------------------------------


class TestDynaQPlus:
    """Sec. 8.3, Dyna-Q+ exploration bonus (Eq. 8.12)."""

    def test_rejects_negative_kappa(self):
        env = make_gridworld(seed=0)
        with pytest.raises(ValueError):
            dyna_q_plus(env, kappa=-1.0)

    def test_kappa_zero_still_plans(self):
        # With kappa = 0 the bonus vanishes, but Dyna-Q+ still samples from
        # never-tried pairs (book footnote), so its trajectory differs from
        # plain Dyna-Q. What must hold is that planning still happens.
        env = make_gridworld(seed=0)
        _, _, stats_plus = dyna_q_plus(
            env, n_planning=2, n_episodes=3, kappa=0.0, seed=7)
        assert stats_plus["planning_updates"] > 0

    def test_learns_maze_with_positive_kappa(self):
        walls = ["2," + str(c) for c in range(10) if c != 9]
        env = make_gridworld(6, 10, "5,4", "0,9", walls=walls,
                             gamma=0.95, seed=0)
        q, _, stats = dyna_q_plus(
            env, n_planning=5, n_episodes=40, seed=0, kappa=1e-4,
            start_state="5,4", max_steps=400)
        assert stats["planning_updates"] > 0
        assert max(q.values()) > 0.0


class TestChangingMaze:
    """Sec. 8.3, Examples 8.2/8.3 (blocking and shortcut mazes)."""

    def test_layouts_are_connected(self):
        from chapter_08.solution import (
            _bfs_path_len,
            changing_maze_layouts,
        )

        start, goal, base, blocking_alt, shortcut_base, shortcut_alt = (
            changing_maze_layouts())
        for walls in (base, blocking_alt, shortcut_base, shortcut_alt):
            assert _bfs_path_len(6, 10, walls, start, goal) is not None

    def test_gridworld_swaps_layout(self):
        from chapter_08.solution import ChangingGridworld

        env = ChangingGridworld(5, 5, "4,0", "0,4", ["2,2"], [], change_step=2)
        env.reset()
        env.step("right")
        assert env.steps == 1
        env.step("right")
        assert env.steps == 2
        env.step("right")  # triggers the swap on this step
        assert env.steps == 3
        assert "2,2" in env._env.states  # previously blocked cell is now walkable

    def test_short_experiment(self):
        from chapter_08.solution import changing_maze_experiment

        cum, episodes = changing_maze_experiment(
            "blocking", "dyna_q", change_step=5, n_episodes=2,
            steps_cap=40, max_steps=20, seed=0)
        assert len(cum) <= 41
        assert cum[0] == 0.0
        assert len(episodes) >= 1


class TestTrajectorySampling:
    """Sec. 8.6: reduced-scale Fig. 8.9 experiment."""

    def test_small_experiment(self):
        from chapter_08.solution import trajectory_sampling_experiment

        grid, uniform, onpolicy = trajectory_sampling_experiment(
            n_states=20, b=1, tasks=2, full_backups_max=2.0,
            n_checkpoints=4, seed=0)
        assert grid.shape == uniform.shape == onpolicy.shape == (4,)
        assert np.all(np.diff(grid) > 0)
        assert np.all(np.isfinite(uniform)) and np.all(np.isfinite(onpolicy))


class TestScalingCriterion:
    """Example 8.4: 'optimal solution' means a shortest greedy path."""

    def test_greedy_path_helper(self):
        from chapter_08.solution import (
            _bfs_path_len,
            _greedy_path_len,
            _vi_qstar,
            _scaled_dyna_maze,
        )

        rows, cols, start, goal, walls = _scaled_dyna_maze(1)
        q_star = _vi_qstar(rows, cols, walls, goal, gamma=0.95)
        optimal = _bfs_path_len(rows, cols, walls, start, goal)
        assert _greedy_path_len(q_star, rows, cols, walls, start, goal,
                                optimal) == optimal

    def test_dyna_q_solves_small_maze(self):
        from chapter_08.solution import _scaling_run_one

        n_states, optimal, solved_at = _scaling_run_one(
            1, "dyna_q", 0, 0, max_steps=1200)
        assert n_states == 47 and optimal == 14
        assert solved_at is not None and solved_at > 0


class TestRodExample:
    """Sec. 8.4, Example 8.5: illustrative rod reconstruction."""

    def test_state_space_and_actions(self):
        from chapter_08.rod import ACTIONS, RodWorld

        world = RodWorld()
        assert tuple(world.actions) == ACTIONS
        assert len(ACTIONS) == 4
        assert len(world.states) == len(set(world.states)) < 20 * 20 * 36
        assert world.goal in world.terminal_states

    def test_collision_keeps_state(self):
        from chapter_08.rod import RodWorld

        world = RodWorld()
        state = world.reset((1, 19, 0))
        next_state = world._next(state, "side")  # leaves the workspace
        assert next_state == state

    def test_prioritized_sweeping_beats_full_sweeps(self):
        from chapter_08.rod import (
            RodWorld,
            prioritized_sweeping_known_model,
            shortest_path,
            value_iteration_known_model,
        )

        world = RodWorld()
        model = world.build_model()
        assert shortest_path(model, world.start_state, world.goal) is not None
        _, ps_updates = prioritized_sweeping_known_model(
            model, world.start_state, world.goal)
        _, vi_updates = value_iteration_known_model(
            model, world.start_state, world.goal)
        assert 0 < ps_updates < vi_updates


class TestRTDPExample:
    """Sec. 8.7, Example 8.6: RTDP on the racetrack."""

    def test_rtdp_focuses_updates(self):
        from chapter_08.rtdp import build_model, rtdp, value_iteration
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(noise_prob=0.0, seed=0)
        model, starts = build_model(env)
        dp = value_iteration(model, starts)
        rt = rtdp(model, starts, n_episodes=400, eval_interval=100)
        assert dp["sweeps"] > 0 and dp["updates"] > 0
        assert rt["updates"] > 0
        assert 0 <= rt["pct_zero_updates"] <= 100
        assert rt["policy_steps"] < 500
