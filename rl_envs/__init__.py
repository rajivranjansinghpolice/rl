"""Shared environments and agents for the Sutton & Barto reference implementations.

Centralizes environments/agents reused across chapters to avoid duplication.
Chapter modules re-export the names they use so existing imports and tests
keep working.
"""

from rl_envs.blackjack import ACTIONS, N_ACTIONS, BlackjackEnv
from rl_envs.four_rooms import (
    DEFAULT_GRID,
    Option,
    OptionResult,
    bfs_action_toward,
    bfs_distances,
    default_options,
    is_wall,
    make_subgoal_option,
    run_option,
)
from rl_envs.four_rooms import GridWorld as FourRoomsGridWorld
from rl_envs.gridworld import (
    Example41GridWorld,
    GreedyAgent,
    GridWorld,
    RandomAgent,
    run_episode,
    run_multiple_episodes,
)
from rl_envs.mountain_car import MountainCarEnv
from rl_envs.pip_race import (
    BLACK,
    MAX_TURNS,
    N_FEATURES,
    PIECES_PER_PLAYER,
    PIP_RACE_FEATURES,
    TRACK_LENGTH,
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
from rl_envs.racetrack import ACTIONS as RACETRACK_ACTIONS
from rl_envs.racetrack import MAX_VELOCITY, N_ACTIONS as RACETRACK_N_ACTIONS, RacetrackEnv
from rl_envs.random_walk import RandomWalkEnv, random_walk_true_values
from rl_envs.tictactoe import (
    GreedyPlayer,
    MCTSPlayer,
    RandomPlayer,
    TicTacToe,
    play_game,
    random_rollout,
    uct_score,
)
from rl_envs.windy_gridworld import WindyGridWorld

__all__ = [
    # Blackjack (Ch.5)
    "ACTIONS",
    "N_ACTIONS",
    "BlackjackEnv",
    # Four-room GridWorld + options (Ch.17)
    "DEFAULT_GRID",
    "FourRoomsGridWorld",
    "Option",
    "OptionResult",
    "bfs_action_toward",
    "bfs_distances",
    "default_options",
    "is_wall",
    "make_subgoal_option",
    "run_option",
    # GridWorld (Ch.1, Ch.3/4)
    "Example41GridWorld",
    "GridWorld",
    "GreedyAgent",
    "RandomAgent",
    "run_episode",
    "run_multiple_episodes",
    # Mountain Car (Ch.10/11)
    "MountainCarEnv",
    # PipRace / TD-Gammon (Ch.12, Sec.16.1)
    "BLACK",
    "MAX_TURNS",
    "N_FEATURES",
    "PIECES_PER_PLAYER",
    "PIP_RACE_FEATURES",
    "TRACK_LENGTH",
    "WHITE",
    "PipRaceEnv",
    "RandomPipRaceAgent",
    "TDGammonAgent",
    "evaluate_match",
    "linear_value",
    "pip_race_features",
    "semi_gradient_td_lambda_update",
    "simulate_game",
    "train_td_gammon",
    # Racetrack (Ch.5)
    "MAX_VELOCITY",
    "RACETRACK_ACTIONS",
    "RACETRACK_N_ACTIONS",
    "RacetrackEnv",
    # Random walk (Ch.6)
    "RandomWalkEnv",
    "random_walk_true_values",
    # Tic Tac Toe + MCTS (Ch.8, Ch.12)
    "GreedyPlayer",
    "MCTSPlayer",
    "RandomPlayer",
    "TicTacToe",
    "play_game",
    "random_rollout",
    "uct_score",
    # Windy gridworld (Ch.6)
    "WindyGridWorld",
]
