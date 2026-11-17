"""Solution implementation for Chapter 12 - Case Studies (Applications of RL).

Reference: Sutton & Barto (2nd Ed), final part "Frontiers and Case Studies".
The case studies live in Chapter 16, "Applications and Case Studies":

- Section 16.1 (TD-Gammon): Tesauro's backgammon player, self-play with
  TD(lambda) and a value function over backgammon positions. This module
  reimplements the *idea* on an honest, heavily documented approximation of
  backgammon (see below).
- Section 16.6 (AlphaGo / AlphaGo Zero): MCTS guided by learned policy and
  value networks. MCTS itself is Section 8.11.
- Section 8.11 (Monte Carlo Tree Search): the four steps Selection,
  Expansion, Simulation, Backup.

Implements:

- algorithm_12.1 (optional in the project backlog): Monte Carlo Tree Search with UCT
  selection (UCB1: Q + c * sqrt(ln N_parent / N_child)) for Tic Tac Toe.
  This is ONLY the tabular/UCT core of AlphaGo's search. AlphaGo's deep
  policy and value networks, supervised pre-training, asynchronous
  policy-and-value MCTS (APV-MCTS), and the reinforcement-learning pipeline
  that produced those networks are deliberately NOT reimplemented; see the
  README for what was simplified and why.
- exercise_12.1: a TD-Gammon-like game player for "PipRace", an honest,
  documented approximation of backgammon:

    * shared linear track of 24 points (not a real backgammon board with
      15 pieces/player and 24 *directed* points per side);
    * 5 pieces per player, stacked at opposite ends ("stack and run");
    * two dice per turn (doubles play four moves, as in backgammon);
    * a piece may not land on a point holding >= 2 opponent pieces
      (the "made point" block), but may pass over it;
    * landing on a point holding exactly one opponent piece (a "blot")
      hits it back to that player's start (a "hit to start", replacing
      backgammon's bar-re-entry rule);
    * pieces bear off by moving past the end; overshoot allowed;
    * a move that is impossible to make is forfeited, as in backgammon;
    * games are capped at ``MAX_TURNS`` turns; the winner is then decided by
      pieces borne off, then pip count, else a draw.

  A small set of hand-designed board features (pip counts, borne-off counts,
  blots, made points, home-board occupancy) feeds a LINEAR value function,
  trained with TD(lambda) (semi-gradient, accumulating eligibility traces,
  Chapter 12 / Eq. 12.10-12.11 of the function approximation chapter) via
  self-play with afterstate move selection (Section 6.8 / TD-Gammon's use of
  afterstates). This mirrors TD-Gammon 0.0's structure (raw features, no
  backgammon strategy knowledge) but with a linear instead of a sigmoidal
  network, 5 pieces instead of 15, and rewards in [-1, +1] instead of a
  0/1 "probability of winning" output. No gammon/backgammon bonuses are
  used; the terminal reward is +1/-1/0 for win/loss/draw from White's
  fixed perspective (the sign convention is documented on ``Env.step``).

The full 15-piece backgammon game with bar re-entry, exact bear-off rules,
and the doubling cube is intentionally NOT implemented here (see README).

The shared environments and agents themselves now live in the
``rl_envs`` package (``rl_envs.tictactoe``, ``rl_envs.pip_race``); they are
re-exported here so existing imports keep working.
"""

from __future__ import annotations

import numpy as np

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
from rl_envs.tictactoe import (
    GreedyPlayer,
    MCTSPlayer,
    RandomPlayer,
    TicTacToe,
    play_game,
    random_rollout,
    uct_score,
)

__all__ = [
    # Re-exported from rl_envs.tictactoe / rl_envs.pip_race
    "BLACK",
    "GreedyPlayer",
    "MAX_TURNS",
    "MCTSPlayer",
    "N_FEATURES",
    "PIECES_PER_PLAYER",
    "PIP_RACE_FEATURES",
    "PipRaceEnv",
    "RandomPipRaceAgent",
    "RandomPlayer",
    "TDGammonAgent",
    "TRACK_LENGTH",
    "TicTacToe",
    "WHITE",
    "evaluate_match",
    "linear_value",
    "pip_race_features",
    "play_game",
    "random_rollout",
    "semi_gradient_td_lambda_update",
    "simulate_game",
    "train_td_gammon",
    "uct_score",
    # Chapter 12 figure reproduction
    "plot_td_gammon_learning",
]

# The environments and agents used here live in the shared
# ``rl_envs`` package; they are re-exported so existing imports
# (``from chapter_12.solution import TicTacToe, ...``) keep working.

# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 16.1 / repo Ch. 12)
# ---------------------------------------------------------------------------

def plot_td_gammon_learning(
    save_path: str | None = None,
    show: bool = False,
    n_games: int = 400,
    eval_every: int = 50,
    eval_games: int = 80,
    alpha: float = 0.05,
    lam: float = 0.7,
    epsilon: float = 0.05,
    seed: int = 0,
):
    """TD-Gammon-like self-play learning curve on PipRace (Sec. 16.1 demo).

    Sutton & Barto (2nd ed.), Sec. 16.1 (TD-Gammon): self-play temporal-
    difference learning with a function approximator learns strong play from
    sparse terminal rewards. This demo trains the linear PipRace value function
    (documented backgammon approximation in this repo) by semi-gradient TD(lambda)
    self-play and periodically evaluates greedy play vs a random opponent
    (`evaluate_match`). PipRace, features, and scale are simplified relative to
    TD-Gammon/backgammon; the plot illustrates learning progress, not book-level
    performance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, N_FEATURES)
    eval_points = []
    scores = []

    def maybe_eval(games_done: int):
        if games_done % eval_every == 0 or games_done == n_games:
            agent = TDGammonAgent(weights.copy())
            score = evaluate_match(agent, n_games=eval_games, seed=seed + games_done + 1)
            eval_points.append(games_done)
            scores.append(score)

    for g in range(1, n_games + 1):
        env = PipRaceEnv(seed=int(rng.integers(0, 2**31 - 1)))
        state = env.reset()
        eligibility = np.zeros(N_FEATURES, dtype=np.float64)
        while not PipRaceEnv.is_over(state, env.turn_count):
            player = state[2]
            dice = env.roll_dice()
            plays = PipRaceEnv.legal_dice_plays(state, player, dice)
            if rng.random() < epsilon:
                play = plays[int(rng.integers(0, len(plays)))]
            else:
                scored = [linear_value(weights, pip_race_features(p)) for p in plays]
                index = int(np.argmax(scored)) if player == WHITE else int(np.argmin(scored))
                play = plays[index]
            features = pip_race_features(state)
            v_state = linear_value(weights, features)
            next_turns = env.turn_count + 1
            if PipRaceEnv.is_over(play, next_turns):
                reward = PipRaceEnv.outcome(play, next_turns)
                v_next = 0.0
            else:
                reward = 0.0
                v_next = linear_value(weights, pip_race_features(play))
            delta = reward + 1.0 * v_next - v_state
            weights, eligibility = semi_gradient_td_lambda_update(
                weights, eligibility, features, delta, alpha, 1.0, lam
            )
            state = env.play_turn(dice, play)
        maybe_eval(g)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eval_points, scores, marker="o")
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Self-play training games")
    ax.set_ylabel("Score vs random (wins + 0.5*draws)")
    ax.set_ylim(0, 1)
    ax.set_title("S&B (2nd ed.), Sec. 16.1 (TD-Gammon-like): PipRace self-play")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
