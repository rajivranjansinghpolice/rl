import math

import numpy as np
import pytest

from rl_envs.random_walk import RandomWalkEnv, random_walk_true_values
from rl_envs.blackjack import BlackjackEnv
from rl_envs.windy_gridworld import WindyGridWorld
from rl_envs.mountain_car import MountainCarEnv
from rl_envs.gridworld import (
    Example41GridWorld,
    GridWorld,
    RandomAgent,
    run_episode,
    run_multiple_episodes,
)
from rl_envs.tictactoe import TicTacToe, MCTSPlayer, GreedyPlayer, play_game, random_rollout, uct_score
from rl_envs.pip_race import (
    MAX_TURNS,
    N_FEATURES,
    PIP_RACE_FEATURES,
    PIECES_PER_PLAYER,
    TRACK_LENGTH,
    WHITE,
    PipRaceEnv,
    TDGammonAgent,
    pip_race_features,
    semi_gradient_td_lambda_update,
    simulate_game,
    evaluate_match,
)
from rl_envs.four_rooms import GridWorld as FourRoomsGridWorld, make_subgoal_option, run_option, default_options


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


def test_windy_gridworld_king_moves():
    env = WindyGridWorld(king_moves=True, allow_stay=True)
    s = env.reset()
    assert s == (3, 0)
    # 8 king moves + no-op
    assert env.n_actions == 9
    # move diagonally up-right: lands on (2, 1), wind applies afterwards
    s2, r, done = env.step("up-right")
    assert s2[1] == 1
    assert r == -1.0


def test_mountain_car():
    env = MountainCarEnv(seed=0)
    s = env.reset()
    assert env.POSITION_MIN <= s[0] <= env.POSITION_MAX
    s2, r, done = env.step(2)
    assert r == -1.0
    assert isinstance(done, bool)


# ---------------------------------------------------------------------------
# GridWorld (Section 1.3, Figure 1.2) and its agents
# ---------------------------------------------------------------------------

def test_gridworld_special_state_A():
    env = GridWorld()
    env.reset()
    # UP/LEFT from A -> A+ (+10, terminal)
    ns, r, done, _ = env.step(GridWorld.UP)
    assert ns == env.state_A_plus and r == 10.0 and done


def test_gridworld_special_state_A_other_actions():
    env = GridWorld()
    env.reset()
    ns, r, done, _ = env.step(GridWorld.RIGHT)
    assert ns == env.state_B and r == 5.0 and not done


def test_gridworld_special_state_B():
    env = GridWorld()
    env.reset()
    # Move A -> B via RIGHT, then B's UP goes to C with +5
    env.step(GridWorld.RIGHT)
    ns, r, done, _ = env.step(GridWorld.UP)
    assert ns == env.state_C and r == 5.0 and not done
    # A fresh trip A -> B via RIGHT, then B DOWN -> D (+5, terminal)
    env.reset()
    env.step(GridWorld.RIGHT)
    ns, r, done, _ = env.step(GridWorld.DOWN)
    assert ns == env.state_D and r == 5.0 and done


def test_gridworld_off_grid_stays_and_rewards():
    env = GridWorld()
    env.reset()
    # get_reward/get_next_state for a normal cell (both endpoints in-grid and off-grid)
    assert env.get_reward((1, 1), GridWorld.UP) == -1.0
    assert env.get_next_state((1, 1), GridWorld.UP) == (0, 1)
    assert env.get_next_state((0, 1), GridWorld.UP) == (0, 1)  # off-grid: stay


def test_gridworld_run_episode_and_multiple_episodes():
    np.random.seed(0)
    env = GridWorld()
    traj = run_episode(env, RandomAgent(), max_steps=500)
    assert len(traj) > 0
    stats = run_multiple_episodes(env, RandomAgent(), n_episodes=3)
    assert len(stats["total_reward"]) == 3
    assert len(stats["steps"]) == 3


# ---------------------------------------------------------------------------
# Example 4.1 gridworld (Examples 3.5/4.1, Figure 4.1)
# ---------------------------------------------------------------------------

# Converged values for the random policy, k -> infinity (Figure 4.1).
BOOK_V = np.array([
    [0.0, -14.0, -20.0, -22.0],
    [-14.0, -18.0, -20.0, -20.0],
    [-20.0, -20.0, -18.0, -14.0],
    [-22.0, -20.0, -14.0, 0.0],
])


def test_example_4_1_invalid_gamma_raises():
    with pytest.raises(ValueError):
        Example41GridWorld(gamma=-0.1)
    with pytest.raises(ValueError):
        Example41GridWorld(gamma=1.1)


def test_example_4_1_terminal_cells_and_step():
    env = Example41GridWorld()
    assert env.is_terminal_cell((0, 0))
    assert env.is_terminal_cell((3, 3))
    assert not env.is_terminal_cell((1, 1))
    env.set_state((0, 1))
    ns, r, done = env.step(Example41GridWorld.LEFT)  # into (0, 0), terminal
    assert ns == (0, 0) and r == -1.0 and done


def test_example_4_1_book_values_satisfy_bellman_equation():
    env = Example41GridWorld()
    for cell in [(r, c) for r in range(4) for c in range(4)]:
        if env.is_terminal_cell(cell):
            assert BOOK_V[cell] == 0.0
            continue
        expected = -1.0 + 0.25 * sum(
            BOOK_V[env.get_next_state(cell, a)] for a in range(4)
        )
        assert BOOK_V[cell] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Tic Tac Toe and MCTS (Section 8.11)
# ---------------------------------------------------------------------------

def test_tictactoe_rules():
    state = TicTacToe.initial_state()
    assert TicTacToe.legal_moves(state) == list(range(9))
    assert TicTacToe.current_player(state) == TicTacToe.X
    state = TicTacToe.apply_move(state, 0)  # X plays cell 0
    assert TicTacToe.current_player(state) == TicTacToe.NOUGHT
    with pytest.raises(ValueError):
        TicTacToe.apply_move(state, 0)  # occupied
    with pytest.raises(ValueError):
        TicTacToe.apply_move(state, 9)  # out of range


def test_tictactoe_winner_and_outcome():
    state = ("X", "X", "X", " ", "O", " ", " ", " ", "O")
    assert TicTacToe.winner(state) == "X"
    assert TicTacToe.is_terminal(state)
    assert TicTacToe.outcome(state, "X") == 1.0
    assert TicTacToe.outcome(state, "O") == -1.0
    with pytest.raises(ValueError):
        TicTacToe.outcome(TicTacToe.initial_state(), "X")  # non-terminal


def test_uct_score():
    assert uct_score(0.5, 0, 10, math.sqrt(2.0)) == math.inf
    expected = 0.5 + math.sqrt(2.0) * math.sqrt(math.log(10) / 4)
    assert uct_score(0.5, 4, 10, math.sqrt(2.0)) == pytest.approx(expected)


def test_random_rollout_returns_valid_outcome():
    rng = np.random.default_rng(0)
    value = random_rollout(TicTacToe.initial_state(), rng)
    assert value in (-1.0, 0.0, 1.0)


def test_mcts_takes_immediate_win():
    # X to move, X has two in a row on the top row: MCTS should play cell 2.
    state = ("X", "X", " ", "O", "O", " ", " ", " ", " ")
    assert TicTacToe.current_player(state) == "X"
    player = MCTSPlayer(n_simulations=50, seed=0)
    assert player.search(state) == 2


def test_play_game_returns_valid_result():
    result = play_game(GreedyPlayer(seed=0), GreedyPlayer(seed=1))
    assert result in ("X", "O", None)


# ---------------------------------------------------------------------------
# PipRace (Section 16.1, TD-Gammon-like)
# ---------------------------------------------------------------------------

def test_pip_race_initial_state_and_pip_count():
    state = PipRaceEnv.initial_state()
    white, black, turn, white_borne, black_borne = state
    assert turn == WHITE
    assert len(white) == PIECES_PER_PLAYER == len(black)
    assert PipRaceEnv.pip_count(white, WHITE) == TRACK_LENGTH * PIECES_PER_PLAYER
    assert PipRaceEnv.pip_count(black, "B") == TRACK_LENGTH * PIECES_PER_PLAYER
    assert white_borne == 0 and black_borne == 0


def test_pip_race_features_shape_and_bias():
    state = PipRaceEnv.initial_state()
    features = pip_race_features(state)
    assert features.shape == (N_FEATURES,)
    assert len(PIP_RACE_FEATURES) == N_FEATURES
    assert features[0] == 1.0  # bias


def test_pip_race_legal_dice_plays_flip_turn():
    env = PipRaceEnv(seed=0)
    state = env.reset()
    plays = PipRaceEnv.legal_dice_plays(state, WHITE, (6, 6))
    assert len(plays) > 0
    assert all(play[2] != WHITE for play in plays)  # turn flipped


def test_pip_race_blocked_point_illegal():
    # Two black pieces on point 3: a white piece may not land there.
    state = ((0,), (3, 3), WHITE, 0, 0)
    assert PipRaceEnv.legal_submoves(state, WHITE, 3) == []
    # Overshoot bear-off (die larger than distance to end) is allowed.
    state = ((TRACK_LENGTH - 1,), (3, 3), WHITE, 0, 0)
    plays = PipRaceEnv.legal_submoves(state, WHITE, 6)
    assert plays and plays[0][3] == 1  # one white piece borne off


def test_pip_race_play_turn_rejects_illegal_play():
    env = PipRaceEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.play_turn((6, 6), PipRaceEnv.initial_state())  # not a legal outcome


def test_pip_race_semi_gradient_td_lambda_update():
    weights = np.zeros(N_FEATURES)
    eligibility = np.zeros(N_FEATURES)
    features = np.ones(N_FEATURES)
    # zero delta -> weights unchanged
    w2, e2 = semi_gradient_td_lambda_update(weights.copy(), eligibility.copy(), features, 0.0, 0.1, 1.0, 0.7)
    assert np.allclose(w2, 0.0)
    assert np.allclose(e2, features)
    with pytest.raises(ValueError):
        semi_gradient_td_lambda_update(weights.copy(), eligibility.copy(), features, 1.0, 0.0, 1.0, 0.7)
    with pytest.raises(ValueError):
        semi_gradient_td_lambda_update(weights.copy(), eligibility.copy(), features, 1.0, 0.1, 1.0, 1.5)


def test_pip_race_agent_chooses_legal_play_and_validation():
    env = PipRaceEnv(seed=0)
    state = env.reset()
    dice = env.roll_dice()
    agent = TDGammonAgent()
    play = agent.choose_play(state, dice)
    assert play in PipRaceEnv.legal_dice_plays(state, WHITE, dice)
    with pytest.raises(ValueError):
        evaluate_match(agent, n_games=0)


def test_pip_race_simulate_game_seeded_terminates():
    outcome = simulate_game(TDGammonAgent(), TDGammonAgent(), seed=0)
    assert outcome in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Four-room GridWorld and options (Section 17.2)
# ---------------------------------------------------------------------------

def test_four_rooms_reset_step_goal():
    env = FourRoomsGridWorld()
    state = env.reset()
    assert env.cell(state) == (0, 0)
    state, reward, done = env.step(1)  # down
    assert env.cell(state) == (1, 0) and not done
    # teleport next to the goal and step onto it
    env.state = env.state_id((6, 5))
    state, reward, done = env.step(3)  # right into the goal
    assert env.cell(state) == (6, 6) and done and reward == pytest.approx(1.0)


def test_four_rooms_wall_bump_stays_penalized():
    env = FourRoomsGridWorld()
    env.reset()
    state, reward, done = env.step(0)  # up into the outer wall
    assert env.cell(state) == (0, 0)
    assert reward == pytest.approx(env.wall_penalty)


def test_four_rooms_option_reaches_subgoal():
    env = FourRoomsGridWorld()
    env.reset()
    option = make_subgoal_option(None, (0, 3), "option_top_room")
    result = run_option(env, option, gamma=0.99, max_steps=20)
    assert option.terminates(env)
    assert result.steps > 0


def test_four_rooms_default_options_reach_goal():
    env = FourRoomsGridWorld()
    env.reset()
    goal_option = default_options()[-1]
    env.state = env.state_id((6, 5))  # adjacent to the goal
    result = run_option(env, goal_option, gamma=1.0, max_steps=50)
    assert result.steps > 0


class TestRacetrack:
    def test_reset_returns_start_cell_with_zero_velocity(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=0)
        row, col, vx, vy = env.reset()
        assert (row, col) in env.start_cells
        assert (vx, vy) == (0, 0)
        assert all(isinstance(v, int) for v in (row, col, vx, vy))

    def test_default_track_geometry(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=0)
        assert env.finish_cells == [(0, 9), (1, 9), (2, 9)]
        assert len(env.start_cells) == 5
        assert env.n_actions == 9
        # 46 on-track cells x 25 velocities.
        assert env.n_states() == 46 * 25
        assert len(env.ACTIONS) == 9

    def test_step_reward_and_continuation(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=42, noise_prob=0.0)
        row, col, vx, vy = env.reset()
        reward, obs, done = env.step(7)  # (0, +1)
        assert reward == -1.0
        assert done is False
        assert obs[3] == 1  # vy incremented

    def test_crossing_finish_line_ends_episode(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=0, noise_prob=0.0)
        env._state = (1, 8, 1, 1)
        reward, obs, done = env.step(4)  # (0, 0): keep v = (1, 1)
        # Projected path (0, 9) is the finish line -> episode ends.
        assert reward == -1.0
        assert done is True
        assert env.grid[0, 9] == "F"

    def test_crossing_boundary_teleports_to_random_start(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=7, noise_prob=0.0)
        env._state = (3, 1, 0, 4)  # column 1 goes straight up into the wall
        reward, obs, done = env.step(4)  # (0, 0): keep v = (0, 4)
        assert reward == -1.0
        assert done is False
        row, col, vx, vy = obs
        assert (row, col) in env.start_cells
        assert (vx, vy) == (0, 0)

    def test_noise_forces_zero_acceleration(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=0, noise_prob=1.0)
        start = env.reset()
        for _ in range(5):
            env.step(8)  # (+1, +1) but noise forces (0, 0)
        assert env.state == start

    def test_action_tuple_equivalent_to_index(self):
        from rl_envs.racetrack import RacetrackEnv, ACTIONS

        env1 = RacetrackEnv(seed=0, noise_prob=0.0)
        env2 = RacetrackEnv(seed=0, noise_prob=0.0)
        s = env1.reset()
        env2._state = s
        env2.reset = lambda: None  # keep state; use directly below
        env2._state = s
        r1, o1, d1 = env1.step(0)
        r2, o2, d2 = env2.step(ACTIONS[0])
        assert (r1, o1, d1) == (r2, o2, d2)

    def test_render_marks_car(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=0)
        text = env.render()
        assert "*" in text
        assert text.count("\n") == len(env.track) - 1

    def test_invalid_inputs_raise(self):
        from rl_envs.racetrack import RacetrackEnv

        with pytest.raises(ValueError):
            RacetrackEnv(noise_prob=1.5)
        with pytest.raises(ValueError):
            RacetrackEnv(track=["####", "abc"])  # ragged
        with pytest.raises(ValueError):
            RacetrackEnv(track=["####", "...."])  # no S / no F
        env = RacetrackEnv(seed=0)
        with pytest.raises(ValueError):
            env.step(9)
        with pytest.raises(ValueError):
            env.step((2, 2))

    def test_max_steps_cap_ends_episode(self):
        from rl_envs.racetrack import RacetrackEnv

        env = RacetrackEnv(seed=0, noise_prob=1.0, max_steps=5)  # noise=1: can never move
        start = env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done = env.step(8)
            steps += 1
        assert steps == 5
        assert env.state == start

    def test_max_steps_cap_validated(self):
        from rl_envs.racetrack import RacetrackEnv

        with pytest.raises(ValueError):
            RacetrackEnv(max_steps=0)
