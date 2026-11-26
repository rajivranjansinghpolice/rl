"""
Tests for Chapter 16: Applications and Case Studies.

Reference: Sutton & Barto (2nd Ed), Chapter 16 (Applications and Case Studies).

The book's chapter 16 surveys real-world RL applications and gives no boxed
algorithms. This suite tests two small classic control applications in the
spirit of the chapter:

- Alg 16.1 — Inventory management controlled with Sarsa (order-up-to MDP)
- Alg 16.2 — Single-server queue MDP with an expedited-service decision
  (solved with Q-learning; truncation via bellman_target)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chapter_16.solution import (
    InventoryEnv,
    QueueEnv,
    bellman_target,
    evaluate_inventory_policy,
    evaluate_queue_policy,
    greedy_inventory_action,
    greedy_queue_action,
    q_learning_queue,
    sarsa_inventory,
)
from rl_envs.pip_race import (
    BLACK,
    MAX_TURNS,
    N_FEATURES,
    PIP_RACE_FEATURES,
    WHITE,
    PipRaceEnv,
    RandomPipRaceAgent,
    TDGammonAgent,
    evaluate_match,
    linear_value,
    pip_race_features,
    semi_gradient_td_lambda_update,
    simulate_game,
    train_td_gammon,
)
from rl_envs.tictactoe import (
    GreedyPlayer,
    MCTSPlayer,
    RandomPlayer,
    TicTacToe,
    play_game,
    random_rollout,
    uct_score,
)


class TestInventoryEnv:
    """Tests for the inventory MDP environment (algorithm 16.1)."""

    def test_episode_runs_and_costs_finite(self) -> None:
        env = InventoryEnv(seed=0)
        state = env.reset(start_inventory=5)
        assert state == 5
        total_cost = 0.0
        steps = 0
        done = False
        while not done:
            state, reward, done = env.step(6)
            assert 0 <= state <= env.s_max
            total_cost += -reward
            steps += 1
        assert steps == env.periods
        assert np.isfinite(total_cost)
        assert total_cost >= 0.0

    def test_state_stays_within_bounds_under_stochastic_demand(self) -> None:
        env = InventoryEnv(demand_rate=8.0, seed=1)
        env.reset(start_inventory=0)
        for _ in range(50):
            _, _, done = env.step(10)
            assert 0 <= env.state <= env.s_max
            if done:
                env.reset(start_inventory=0)

    def test_cannot_order_up_below_or_above_bounds(self) -> None:
        env = InventoryEnv(seed=0)
        env.reset(start_inventory=5)
        with pytest.raises(ValueError):
            env.step(env.s_max + 1)
        with pytest.raises(ValueError):
            env.step(-1)

    def test_step_before_reset_raises(self) -> None:
        with pytest.raises(RuntimeError):
            InventoryEnv(seed=0).step(0)

    def test_order_raises_position_and_covers_demand(self) -> None:
        # Zero demand rate would be nice; use demand_rate tiny so demand = 0 w.h.p.
        env = InventoryEnv(demand_rate=1e-9, seed=0)
        env.reset(start_inventory=2)
        state, reward, _ = env.step(7)
        # Ordered 5 units at c_order=1 each; nothing sold; leftover 7.
        assert state == 7
        assert reward == pytest.approx(-(5.0 + 7.0 * env.h))


class TestSarsaInventory:
    """Tests for Sarsa-based inventory control (algorithm 16.1)."""

    def test_q_table_structure_and_finite_values(self) -> None:
        q_table, _ = sarsa_inventory(n_episodes=20, seed=0)
        assert len(q_table) == 11 * 11
        assert all(np.isfinite(v) for v in q_table.values())

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            sarsa_inventory(alpha=0.0)
        with pytest.raises(ValueError):
            sarsa_inventory(epsilon=0.0)
        with pytest.raises(ValueError):
            sarsa_inventory(gamma=1.5)

    def test_learned_policy_beats_zero_order_policy(self) -> None:
        q_table, _ = sarsa_inventory(n_episodes=600, seed=0)
        greedy = evaluate_inventory_policy(
            lambda s: greedy_inventory_action(q_table, s, 10), episodes=200, seed=99,
        )
        zero_order = evaluate_inventory_policy(lambda s: 0, episodes=200, seed=99)
        assert greedy < zero_order

    def test_learned_policy_beats_random_policy(self) -> None:
        q_table, _ = sarsa_inventory(n_episodes=600, seed=0)
        greedy = evaluate_inventory_policy(
            lambda s: greedy_inventory_action(q_table, s, 10), episodes=200, seed=99,
        )
        rng = np.random.default_rng(7)
        random_cost = evaluate_inventory_policy(
            lambda s: int(rng.integers(0, 11)), episodes=200, seed=99,
        )
        assert greedy < random_cost

    def test_learned_policy_orders_when_empty(self) -> None:
        q_table, _ = sarsa_inventory(n_episodes=600, seed=0)
        assert greedy_inventory_action(q_table, 0, 10) >= 2


class TestQueueEnv:
    """Tests for the single-server queue MDP (algorithm 16.2)."""

    def test_episode_runs_and_costs_finite(self) -> None:
        env = QueueEnv(seed=0)
        assert env.reset() == 0
        total_cost = 0.0
        steps = 0
        done = False
        while not done:
            state, reward, done = env.step(0)
            assert 0 <= state <= env.q_max
            total_cost += -reward
            steps += 1
        assert steps == env.periods
        assert np.isfinite(total_cost)

    def test_queue_never_exceeds_capacity(self) -> None:
        env = QueueEnv(arrival_rate=5.0, seed=2)
        env.reset()
        for _ in range(30):
            state, _, done = env.step(0)
            assert state <= env.q_max
            if done:
                env.reset()

    def test_empty_queue_service_costs_nothing(self) -> None:
        env = QueueEnv(arrival_rate=1e-9, seed=0)
        env.reset()
        _, reward, _ = env.step(1)  # expedite an empty system
        assert reward == 0.0

    def test_invalid_action_and_step_before_reset_raise(self) -> None:
        env = QueueEnv(seed=0)
        with pytest.raises(RuntimeError):
            env.step(0)
        env.reset()
        with pytest.raises(ValueError):
            env.step(2)


class TestQLearningQueue:
    """Tests for Q-learning on the queue MDP (algorithm 16.2)."""

    def test_bellman_target_true_terminal(self) -> None:
        next_q = np.array([10.0, 20.0])
        assert bellman_target(1.0, next_q, 0.95, terminal=True) == pytest.approx(1.0)

    def test_bellman_target_bootstraps_on_truncation(self) -> None:
        next_q = np.array([-4.0, -9.0])
        target = bellman_target(-1.0, next_q, 0.5, terminal=False)
        assert target == pytest.approx(-1.0 + 0.5 * -4.0)
        # With negative next values, bootstrapping lowers the target;
        # with zeros, it matches the immediate reward.
        assert target < bellman_target(-1.0, np.zeros(2), 0.5, terminal=False)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            q_learning_queue(alpha=0.0)
        with pytest.raises(ValueError):
            q_learning_queue(epsilon=0.0)
        with pytest.raises(ValueError):
            q_learning_queue(gamma=1.5)
        with pytest.raises(ValueError):
            bellman_target(0.0, np.zeros(2), 1.5, terminal=False)

    def test_learned_policy_beats_random_service(self) -> None:
        q_table, _ = q_learning_queue(n_episodes=1500, seed=0)
        assert all(np.isfinite(v) for v in q_table.values())
        greedy = evaluate_queue_policy(
            lambda s: greedy_queue_action(q_table, s), episodes=300, seed=123,
        )
        rng = np.random.default_rng(7)
        random_cost = evaluate_queue_policy(
            lambda s: int(rng.integers(0, 2)), episodes=300, seed=123,
        )
        assert greedy < random_cost

    def test_learned_policy_beats_never_expediting(self) -> None:
        q_table, _ = q_learning_queue(n_episodes=1500, seed=0)
        greedy = evaluate_queue_policy(
            lambda s: greedy_queue_action(q_table, s), episodes=300, seed=123,
        )
        normal_only = evaluate_queue_policy(lambda s: 0, episodes=300, seed=123)
        assert greedy < normal_only


# ---------------------------------------------------------------------------
# Case-study tests (Sec 16.1 TD-Gammon / Sec 16.6 MCTS)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tic Tac Toe rules (support for algorithm_16.1)
# ---------------------------------------------------------------------------

class TestTicTacToeRules:
    """Board mechanics and winner detection. Book Sec 8.11 setup."""

    def test_initial_state_and_legal_moves(self) -> None:
        state = TicTacToe.initial_state()
        assert len(state) == 9
        assert TicTacToe.legal_moves(state) == list(range(9))
        assert TicTacToe.current_player(state) == TicTacToe.X

    def test_current_player_alternates(self) -> None:
        state = TicTacToe.apply_move(TicTacToe.initial_state(), 0)
        assert TicTacToe.current_player(state) == TicTacToe.NOUGHT
        state = TicTacToe.apply_move(state, 4)
        assert TicTacToe.current_player(state) == TicTacToe.X

    def test_apply_move_rejects_occupied_cell(self) -> None:
        state = TicTacToe.apply_move(TicTacToe.initial_state(), 3)
        with pytest.raises(ValueError):
            TicTacToe.apply_move(state, 3)

    def test_apply_move_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            TicTacToe.apply_move(TicTacToe.initial_state(), 9)

    def test_winner_detection_rows_columns_diagonals(self) -> None:
        for a, b, c in TicTacToe.WIN_LINES:
            cells = [TicTacToe.EMPTY] * 9
            cells[a] = cells[b] = cells[c] = TicTacToe.X
            assert TicTacToe.winner(tuple(cells)) == TicTacToe.X
            cells[a] = cells[b] = cells[c] = TicTacToe.NOUGHT
            assert TicTacToe.winner(tuple(cells)) == TicTacToe.NOUGHT

    def test_winner_detection_draw_and_in_progress(self) -> None:
        # X O X / X O O / O X X - full board, no line -> draw.
        board = ("X", "O", "X", "X", "O", "O", "O", "X", "X")
        assert TicTacToe.is_terminal(board)
        assert TicTacToe.winner(board) is None
        assert TicTacToe.outcome(board, TicTacToe.X) == 0.0
        assert not TicTacToe.is_terminal(TicTacToe.initial_state())

    def test_outcome_rejects_non_terminal(self) -> None:
        with pytest.raises(ValueError):
            TicTacToe.outcome(TicTacToe.initial_state(), TicTacToe.X)

    def test_canonical_key_is_hashable_and_stable(self) -> None:
        state = TicTacToe.initial_state()
        key = TicTacToe.canonical_key(state)
        assert key == state
        hash(key)


# ---------------------------------------------------------------------------
# UCT / UCB1 selection (algorithm_16.1)
# ---------------------------------------------------------------------------

class TestUCTScore:
    """UCB1: score = Q + c * sqrt(ln(N_parent) / N_child). Book Sec 2.7 + 8.11."""

    def test_unvisited_child_has_infinite_score(self) -> None:
        assert uct_score(0.0, 0, 100, 1.4) == math.inf

    def test_known_answer_log_base_e(self) -> None:
        # ln(e) = 1, so the exploration term equals c exactly at N = e.
        assert uct_score(0.0, 1, math.e, 2.0) == pytest.approx(2.0)

    def test_exploitation_dominates_with_large_equal_visits(self) -> None:
        # As visit counts grow with a fixed ratio, the value term dominates:
        # the higher-Q child wins regardless of the (shared) exploration bonus.
        high = uct_score(0.9, 10_000, 10_000, 1.4)
        low = uct_score(0.1, 10_000, 10_000, 1.4)
        assert high > low

    def test_exploration_term_decreases_with_child_visits(self) -> None:
        rare = uct_score(0.5, 5, 10_000, 1.4)
        common = uct_score(0.5, 5_000, 10_000, 1.4)
        assert rare > common

    def test_exploration_term_grows_slowly_with_parent_visits(self) -> None:
        small = uct_score(0.5, 10, 10, 1.4)
        large = uct_score(0.5, 10, 10_000, 1.4)
        assert large > small  # log growth, not linear


class TestMCTS:
    """MCTS algorithm tests. Book Sec 8.11 (Selection, Expansion, Simulation,
    Backup); AlphaGo is Sec 16.6."""

    def test_search_returns_legal_move(self) -> None:
        state = TicTacToe.apply_move(TicTacToe.initial_state(), 0)
        player = MCTSPlayer(n_simulations=50, seed=0)
        assert player.search(state) in TicTacToe.legal_moves(state)

    def test_search_rejects_terminal_state(self) -> None:
        board = ("X", "X", "X", "O", "O", " ", " ", " ", " ")
        with pytest.raises(ValueError):
            MCTSPlayer(n_simulations=10, seed=0).search(board)

    def test_rollout_stops_at_terminal(self) -> None:
        rng = np.random.default_rng(0)
        win = ("X", "X", "X", "O", "O", " ", " ", " ", " ")
        # Player to move is O; rollout must not play past the terminal state.
        assert random_rollout(win, rng) == TicTacToe.outcome(win, TicTacToe.NOUGHT) == -1.0

    def test_rollout_returns_valid_outcome(self) -> None:
        rng = np.random.default_rng(0)
        assert random_rollout(TicTacToe.initial_state(), rng) in (-1.0, 0.0, 1.0)

    def test_mcts_takes_immediate_win(self) -> None:
        # X has 0 and 1; winning move is 2. O sits far away.
        state = ("X", "X", " ", "O", " ", " ", "O", " ", " ")
        assert TicTacToe.current_player(state) == TicTacToe.X
        move = MCTSPlayer(n_simulations=200, seed=0).search(state)
        assert move == 2

    def test_mcts_blocks_opponent_win(self) -> None:
        # O threatens 3-4-5; X must play 5 (X has no immediate win).
        state = ("X", " ", " ", "O", "O", " ", " ", " ", "X")
        assert TicTacToe.current_player(state) == TicTacToe.X
        move = MCTSPlayer(n_simulations=300, seed=0).search(state)
        assert move == 5

    def test_mcts_beats_random_player(self) -> None:
        wins = draws = losses = 0
        for i in range(50):
            for x_first in (True, False):
                x, o = (
                    (MCTSPlayer(n_simulations=100, seed=i), RandomPlayer(seed=i))
                    if x_first
                    else (RandomPlayer(seed=i), MCTSPlayer(n_simulations=100, seed=i))
                )
                result = play_game(x, o)
                if result == ("X" if x_first else "O"):
                    wins += 1
                elif result is None:
                    draws += 1
                else:
                    losses += 1
        assert (wins + 0.5 * draws) / 100 >= 0.8
        assert losses <= 5  # MCTS should almost never lose to uniform random

    def test_mcts_never_loses_against_greedy(self) -> None:
        # MCTS with a decent budget must at least draw every game vs a
        # win-or-block heuristic player.
        for i in range(6):
            result = play_game(
                GreedyPlayer(seed=i), MCTSPlayer(n_simulations=800, seed=i)
            )
            assert result != "X"
            result = play_game(
                MCTSPlayer(n_simulations=800, seed=i), GreedyPlayer(seed=i)
            )
            assert result != "O"


# ---------------------------------------------------------------------------
# PipRace environment rules (exercise_16.1)
# ---------------------------------------------------------------------------

class TestPipRaceRules:
    """Game mechanics of the backgammon approximation. Book Sec 16.1."""

    def _state(
        self,
        white: list[int],
        black: list[int],
        turn: str = WHITE,
        wb: int = 0,
        bb: int = 0,
    ) -> tuple:
        white = tuple(sorted(white))
        black = tuple(sorted(black))
        return (white, black, turn, wb, bb)

    def test_initial_state(self) -> None:
        state = PipRaceEnv.initial_state()
        white, black, turn, white_borne, black_borne = state
        assert white == (0,) * 5
        assert black == (23,) * 5
        assert turn == WHITE
        assert (white_borne, black_borne) == (0, 0)
        assert not PipRaceEnv.is_over(state, 0)

    def test_pip_count_fixture(self) -> None:
        state = self._state([2, 8], [15], WHITE, 1, 2)
        assert PipRaceEnv.pip_count(state[0], WHITE) == (24 - 2) + (24 - 8) == 38
        assert PipRaceEnv.pip_count(state[1], BLACK) == 15 + 1 == 16

    def test_winner_detection_terminal(self) -> None:
        state = self._state([], [5, 7], WHITE, 5, 3)
        assert PipRaceEnv.winner(state) == WHITE
        assert PipRaceEnv.outcome(state, turns_played=10) == 1.0
        state = self._state([3], [], BLACK, 2, 5)
        assert PipRaceEnv.winner(state) == BLACK
        assert PipRaceEnv.outcome(state, turns_played=10) == -1.0

    def test_move_cap_tiebreak(self) -> None:
        state = self._state([10], [11], WHITE, 2, 1)
        assert PipRaceEnv.declared_winner(state, MAX_TURNS) == WHITE  # more borne off
        state = self._state([10], [11], WHITE, 1, 1)
        # White pips: 14, Black pips: 12 -> Black wins on lower pip count.
        assert PipRaceEnv.declared_winner(state, MAX_TURNS) == BLACK

    def test_blocked_point_cannot_be_landed_on(self) -> None:
        # Two black pieces on point 3: White may never land there. The 1-die
        # is still playable (point 1), and playing 1 then 3 reaches point 4.
        state = self._state([0], [3, 3, 20, 21, 22], WHITE)
        plays = PipRaceEnv.legal_dice_plays(state, WHITE, (3, 1))
        white_ends = {p[0] for p in plays}
        assert all(3 not in p for p in white_ends)
        assert (1,) in white_ends  # 3-die forfeited, 1-die played
        assert (4,) in white_ends  # 1-die then 3-die
        assert all(p[2] == BLACK for p in plays)  # turn flipped after full play

    def test_blot_is_hit_back_to_start(self) -> None:
        # A single black piece on point 5 is a blot; landing on it sends it
        # back to Black's start (point 23), replacing backgammon's bar.
        state = self._state([0], [5, 10, 12, 14, 16], WHITE)
        plays = PipRaceEnv.legal_dice_plays(state, WHITE, (5, 1))
        hit = [p for p in plays if p[1] == (10, 12, 14, 16, 23)]
        assert hit, f"expected a play hitting the blot, got {plays}"

    def test_overshoot_bearing_off(self) -> None:
        state = self._state([22], [0, 2, 4, 6, 8], WHITE)
        plays = PipRaceEnv.legal_dice_plays(state, WHITE, (6, 1))
        borne = [p for p in plays if p[3] == 1]  # white_borne incremented
        assert borne, "overshoot bear-off must be legal"

    def test_single_piece_double_dice_play(self) -> None:
        # One White piece, dice (3, 4): both orders just move it to point 7.
        state = self._state([0], [23, 22, 21, 20, 19], WHITE)
        plays = PipRaceEnv.legal_dice_plays(state, WHITE, (3, 4))
        assert plays == [((7,), (19, 20, 21, 22, 23), BLACK, 0, 0)]

    def test_simulate_game_terminates(self) -> None:
        outcome = simulate_game(
            RandomPipRaceAgent(seed=1), RandomPipRaceAgent(seed=2), seed=0
        )
        assert outcome in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Features, linear value function and TD(lambda) update (exercise_16.1)
# ---------------------------------------------------------------------------

class TestLinearValueFunction:
    """Hand-designed features + linear v(s). Book Sec 9.x / 16.1."""

    def _fixture_state(self) -> tuple:
        # White: 2 on point 2, 8 on point 8, 1 borne off.
        # Black: 1 on point 15, 2 borne off. White to move.
        return ((2, 8), (15,), WHITE, 1, 2)

    def test_feature_vector_known_values(self) -> None:
        state = self._fixture_state()
        phi = pip_race_features(state)
        total = 5 * 24
        expected = np.zeros(N_FEATURES)
        expected[0] = 1.0
        expected[1] = (16 - 38) / total
        expected[2] = 1 / 5
        expected[3] = 2 / 5
        expected[4] = 0.0 / 5
        expected[5] = 0.0 / 5
        expected[6] = 2 / 5  # two lone white pieces -> two blots
        expected[7] = 1 / 5
        expected[8] = 0.0 / 24
        expected[9] = 0.0 / 24
        expected[10] = 38 / total
        expected[11] = 16 / total
        expected[12] = 1.0
        np.testing.assert_allclose(phi, expected, atol=1e-9)

    def test_feature_names_match_vector_length(self) -> None:
        assert len(PIP_RACE_FEATURES) == N_FEATURES

    def test_linear_value_known_answer(self) -> None:
        weights = np.arange(1.0, N_FEATURES + 1.0)
        phi = pip_race_features(self._fixture_state())
        assert linear_value(weights, phi) == pytest.approx(float(np.dot(weights, phi)))


class TestTDLambdaUpdate:
    """Semi-gradient TD(lambda) with accumulating traces. Eq. 12.10/12.11."""

    def test_single_update_matches_hand_computation(self) -> None:
        weights = np.zeros(N_FEATURES)
        eligibility = np.zeros(N_FEATURES)
        features = np.zeros(N_FEATURES)
        features[0], features[3] = 2.0, 0.5
        delta, alpha, gamma, lam = 1.5, 0.1, 1.0, 0.5

        new_weights, new_eligibility = semi_gradient_td_lambda_update(
            weights, eligibility, features, delta, alpha, gamma, lam
        )
        # e = gamma * lam * e_old + phi = phi (e_old = 0)
        np.testing.assert_allclose(new_eligibility, features, atol=1e-12)
        # w_new = w + alpha * delta * e
        expected = weights + alpha * delta * features
        np.testing.assert_allclose(new_weights, expected, atol=1e-12)
        assert new_weights[0] == pytest.approx(0.3)
        assert new_weights[3] == pytest.approx(0.075)

    def test_traces_accumulate_across_steps(self) -> None:
        weights = np.zeros(N_FEATURES)
        eligibility = np.zeros(N_FEATURES)
        features = np.zeros(N_FEATURES)
        features[0] = 1.0
        delta, alpha, gamma, lam = 0.5, 0.1, 1.0, 0.5

        _, e1 = semi_gradient_td_lambda_update(
            weights.copy(), eligibility, features, delta, alpha, gamma, lam
        )
        # Second identical step: e = gamma * lam * phi + phi = 1.5 * phi
        _, e2 = semi_gradient_td_lambda_update(
            weights.copy(), e1.copy(), features, delta, alpha, gamma, lam
        )
        assert e2[0] == pytest.approx(1.0 + 1.0 * 0.5 * 1.0)

    def test_lam_zero_has_no_trace_memory(self) -> None:
        weights = np.zeros(N_FEATURES)
        eligibility = np.full(N_FEATURES, 100.0)
        features = np.ones(N_FEATURES)
        delta, alpha, gamma, lam = 1.0, 0.1, 1.0, 0.0
        _, e_new = semi_gradient_td_lambda_update(
            weights, eligibility, features, delta, alpha, gamma, lam
        )
        np.testing.assert_allclose(e_new, features, atol=1e-12)

    def test_invalid_alpha_and_lambda_rejected(self) -> None:
        with pytest.raises(ValueError):
            semi_gradient_td_lambda_update(
                np.zeros(N_FEATURES), np.zeros(N_FEATURES), np.zeros(N_FEATURES),
                1.0, 0.0, 1.0, 0.5,
            )
        with pytest.raises(ValueError):
            semi_gradient_td_lambda_update(
                np.zeros(N_FEATURES), np.zeros(N_FEATURES), np.zeros(N_FEATURES),
                1.0, 0.1, 1.0, 1.5,
            )


@pytest.fixture(scope="module")
def trained_weights() -> np.ndarray:
    # Train once for all tests in this module (a few seconds).
    return train_td_gammon(n_games=400, alpha=0.05, lam=0.7, epsilon=0.05, seed=0)


class TestTrainedAgent:
    """Self-play training learns to beat a random player. Book Sec 16.1.

    Training is deliberately small (a few hundred games) so the suite stays
    fast; the book's TD-Gammon used ~300,000 games.
    """

    def test_training_produces_finite_weights(self, trained_weights: np.ndarray) -> None:
        assert trained_weights.shape == (N_FEATURES,)
        assert np.all(np.isfinite(trained_weights))

    def test_trained_agent_beats_random_player(self, trained_weights: np.ndarray) -> None:
        agent = TDGammonAgent(trained_weights)
        win_rate = evaluate_match(agent, n_games=200, seed=123)
        # TD-Gammon-style self-play should dominate uniform random play.
        assert win_rate >= 0.65

    def test_greedy_agent_is_deterministic(self, trained_weights: np.ndarray) -> None:
        agent = TDGammonAgent(trained_weights)
        state = PipRaceEnv.initial_state()
        dice = (3, 4)
        assert agent.choose_play(state, dice) == agent.choose_play(state, dice)
