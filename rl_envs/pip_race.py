"""PipRace: a documented approximation of backgammon (S&B, Sec. 16.1 TD-Gammon).

Ported from ``chapter_12/solution.py`` (Part B, exercise_12.1). PipRace
reimplements the *idea* of TD-Gammon on an honest, heavily documented
approximation of backgammon:

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
Eq. 12.10-12.11 of the function approximation chapter) via self-play with
afterstate move selection (Section 6.8 / TD-Gammon's use of afterstates).
This mirrors TD-Gammon 0.0's structure (raw features, no backgammon
strategy knowledge) but with a linear instead of a sigmoidal network,
5 pieces instead of 15, and rewards in [-1, +1] instead of a 0/1
"probability of winning" output. No gammon/backgammon bonuses are used;
the terminal reward is +1/-1/0 for win/loss/draw from White's fixed
perspective (the sign convention is documented on ``PipRaceEnv.step`` /
``PipRaceEnv.outcome``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

WHITE = "W"
BLACK = "B"
TRACK_LENGTH = 24
PIECES_PER_PLAYER = 5
MAX_TURNS = 400
PIP_RACE_FEATURES = (
    "bias",
    "pip_lead_normalized",      # (black_pips - white_pips) / (PIECES * TRACK)
    "white_borne_off",          # white borne off / PIECES_PER_PLAYER
    "black_borne_off",          # black borne off / PIECES_PER_PLAYER
    "white_in_home",            # white pieces on points 18..23 / PIECES_PER_PLAYER
    "black_in_home",            # black pieces on points 0..5 / PIECES_PER_PLAYER
    "white_blots",              # white points with exactly 1 piece / PIECES_PER_PLAYER
    "black_blots",              # black points with exactly 1 piece / PIECES_PER_PLAYER
    "white_made_points",        # white points with >= 2 pieces / TRACK_LENGTH
    "black_made_points",        # black points with >= 2 pieces / TRACK_LENGTH
    "white_pips_normalized",    # white pip count / (PIECES_PER_PLAYER * TRACK_LENGTH)
    "black_pips_normalized",    # black pip count / (PIECES_PER_PLAYER * TRACK_LENGTH)
    "turn_white",               # 1.0 if White to move else 0.0
)
N_FEATURES = len(PIP_RACE_FEATURES)


def pip_race_features(state: tuple) -> np.ndarray:
    """Hand-designed board features for PipRace. Book Sec 16.1 (representation).

    Features are computed from White's fixed perspective (the same network
    plays both sides, with ``turn_white`` as an input, mirroring TD-Gammon's
    turn indicators). All values are scaled to roughly [0, 1].
    """
    white, black, turn, white_borne, black_borne = state
    white_pips = PipRaceEnv.pip_count(white, WHITE)
    black_pips = PipRaceEnv.pip_count(black, BLACK)
    total_pips = PIECES_PER_PLAYER * TRACK_LENGTH

    def blots(pieces: tuple[int, ...]) -> int:
        unique = set(pieces)
        count = 0
        for point in unique:
            if pieces.count(point) == 1:
                count += 1
        return count

    def made(pieces: tuple[int, ...]) -> int:
        unique = set(pieces)
        count = 0
        for point in unique:
            if pieces.count(point) >= 2:
                count += 1
        return count

    features = np.zeros(N_FEATURES, dtype=np.float64)
    features[0] = 1.0  # bias
    features[1] = (black_pips - white_pips) / total_pips
    features[2] = white_borne / PIECES_PER_PLAYER
    features[3] = black_borne / PIECES_PER_PLAYER
    features[4] = sum(1 for p in white if p >= TRACK_LENGTH - 6) / PIECES_PER_PLAYER
    features[5] = sum(1 for p in black if p < 6) / PIECES_PER_PLAYER
    features[6] = blots(white) / PIECES_PER_PLAYER
    features[7] = blots(black) / PIECES_PER_PLAYER
    features[8] = made(white) / TRACK_LENGTH
    features[9] = made(black) / TRACK_LENGTH
    features[10] = white_pips / total_pips
    features[11] = black_pips / total_pips
    features[12] = 1.0 if turn == WHITE else 0.0
    return features


class PipRaceEnv:
    """PipRace: an honest, documented approximation of backgammon. Book Sec 16.1.

    Simplifications versus real backgammon (all deliberate, see module
    docstring and README):

    * 5 pieces per player (backgammon: 15) on a shared linear track of
      24 points (backgammon: 24 directional points plus bar and off);
    * "stack and run": all White pieces start on point 0 moving right, all
      Black pieces start on point 23 moving left (backgammon has a fixed
      scattered setup);
    * two dice per turn; doubles play four moves of the same value
      (backgammon rule kept);
    * a point with >= 2 opponent pieces is blocked for landing (backgammon
      "made point" rule kept); pieces may pass over blocked points;
    * landing on a point with exactly one opponent piece (a "blot") hits it
      back to that player's start (replacing backgammon's bar re-entry);
    * bearing off by overshoot is allowed from anywhere on the track
      (backgammon requires a home-board and has exact-roll rules);
    * a die that cannot be played is forfeited (backgammon: "use as much of
      the roll as possible");
    * no doubling cube, no gammon/backgammon bonuses: terminal reward is
      +1 (White wins), -1 (Black wins), 0 (draw or move-cap deadlock);
    * at most ``MAX_TURNS`` turns; then the winner is decided by pieces
      borne off, then lower pip count, else a draw.

    State layout (immutable tuple, canonical and hashable):

        (sorted white positions, sorted black positions, player_to_move,
         white_borne, black_borne)

    Positions are ints in [0, 24); a piece's "pip count" is its distance to
    bearing off (24 - position for White, position + 1 for Black).
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.state: tuple | None = None
        self.turn_count = 0

    @staticmethod
    def initial_state() -> tuple:
        """Return the stacked start position (White to move)."""
        white = tuple(0 for _ in range(PIECES_PER_PLAYER))
        black = tuple(TRACK_LENGTH - 1 for _ in range(PIECES_PER_PLAYER))
        return (white, black, WHITE, 0, 0)

    def reset(self) -> tuple:
        """Reset to the initial state and return it."""
        self.state = PipRaceEnv.initial_state()
        self.turn_count = 0
        return self.state

    @staticmethod
    def _count_at(pieces: tuple[int, ...], point: int) -> int:
        return pieces.count(point)

    @staticmethod
    def pip_count(pieces: tuple[int, ...], player: str) -> int:
        """Total distance to bear-off for all on-track pieces of ``player``."""
        total = sum(pieces)
        if player == WHITE:
            return TRACK_LENGTH * len(pieces) - total
        return total + len(pieces)

    @staticmethod
    def winner(state: tuple) -> str | None:
        """Return 'W', 'B', or None (only for a normal win, not the cap)."""
        _, _, _, white_borne, black_borne = state
        if white_borne >= PIECES_PER_PLAYER:
            return WHITE
        if black_borne >= PIECES_PER_PLAYER:
            return BLACK
        return None

    @staticmethod
    def declared_winner(state: tuple, turns_played: int) -> str | None:
        """Return the winner including the move-cap tiebreak, or None for a draw.

        At ``MAX_TURNS`` the player with more borne pieces wins; ties go to
        the lower pip count; equal everything is a draw.
        """
        normal = PipRaceEnv.winner(state)
        if normal is not None:
            return normal
        if turns_played < MAX_TURNS:
            return None
        white, black, _, white_borne, black_borne = state
        if white_borne != black_borne:
            return WHITE if white_borne > black_borne else BLACK
        white_pips = PipRaceEnv.pip_count(white, WHITE)
        black_pips = PipRaceEnv.pip_count(black, BLACK)
        if white_pips != black_pips:
            return WHITE if white_pips < black_pips else BLACK
        return None

    @staticmethod
    def is_over(state: tuple, turns_played: int) -> bool:
        """True if the game is decided (normal win or move-cap tiebreak)."""
        return (
            PipRaceEnv.winner(state) is not None
            or turns_played >= MAX_TURNS
        )

    @staticmethod
    def outcome(state: tuple, turns_played: int = MAX_TURNS) -> float:
        """Terminal reward from White's perspective: +1 W win, -1 B win, 0 draw."""
        win = PipRaceEnv.declared_winner(state, turns_played)
        if win == WHITE:
            return 1.0
        if win == BLACK:
            return -1.0
        return 0.0

    @staticmethod
    def _flip_turn(state: tuple) -> tuple:
        white, black, turn, white_borne, black_borne = state
        return (white, black, BLACK if turn == WHITE else WHITE, white_borne, black_borne)

    @staticmethod
    def _apply_submove(
        state: tuple, player: str, piece_index: int, die: int
    ) -> tuple | None:
        """Apply one die to one piece. Return the new state or None if illegal.

        The player to move is NOT flipped here; a full dice play flips the
        turn only at the end (see ``legal_dice_plays``).
        """
        white, black, turn, white_borne, black_borne = state
        if player == WHITE:
            if piece_index >= len(white):
                return None
            start = white[piece_index]
            end = start + die
            if end >= TRACK_LENGTH:  # overshoot bear-off
                new_white = list(white)
                new_white.pop(piece_index)
                return (tuple(sorted(new_white)), black, turn, white_borne + 1, black_borne)
            if PipRaceEnv._count_at(black, end) >= 2:
                return None  # blocked point
            new_white = list(white)
            new_white[piece_index] = end
            new_black = list(black)
            if new_black.count(end) == 1:
                new_black[new_black.index(end)] = TRACK_LENGTH - 1  # hit back to Black's start
            return (tuple(sorted(new_white)), tuple(sorted(new_black)), turn,
                    white_borne, black_borne)
        # Black moves from high points toward point 0.
        if piece_index >= len(black):
            return None
        start = black[piece_index]
        end = start - die
        if end < 0:  # overshoot bear-off
            new_black = list(black)
            new_black.pop(piece_index)
            return (white, tuple(sorted(new_black)), turn, white_borne, black_borne + 1)
        if PipRaceEnv._count_at(white, end) >= 2:
            return None  # blocked point
        new_black = list(black)
        new_black[piece_index] = end
        new_white = list(white)
        if new_white.count(end) == 1:
            new_white[new_white.index(end)] = 0  # hit back to White's start
        return (tuple(sorted(new_white)), tuple(sorted(new_black)), turn,
                white_borne, black_borne)

    @staticmethod
    def legal_submoves(state: tuple, player: str, die: int) -> list[tuple]:
        """All distinct resulting states from applying one die. One per piece."""
        results: dict[tuple, None] = {}
        pieces = state[0] if player == WHITE else state[1]
        for index in range(len(pieces)):
            after = PipRaceEnv._apply_submove(state, player, index, die)
            if after is not None:
                results[after] = None
        return list(results)

    @staticmethod
    def legal_dice_plays(state: tuple, player: str, dice: tuple[int, int]) -> list[tuple]:
        """Enumerate distinct end positions reachable by a full dice roll.

        Both die orders are tried when the dice differ; doubles are played as
        four moves. Dice that cannot be used are forfeited, as in backgammon.
        The turn is flipped in every returned state (the roll is over).
        """
        sequences: list[list[int]] = []
        d1, d2 = dice
        if d1 == d2:
            sequences.append([d1, d1, d1, d1])
        else:
            sequences.append([d1, d2])
            sequences.append([d2, d1])

        results: dict[tuple, None] = {}

        def recurse(current: tuple, index: int) -> None:
            if index == len(sequence):
                results[PipRaceEnv._flip_turn(current)] = None
                return
            die = sequence[index]
            options = PipRaceEnv.legal_submoves(current, player, die)
            if not options:  # forfeit this die and move on
                recurse(current, index + 1)
                return
            for after in options:
                recurse(after, index + 1)

        for sequence in sequences:
            recurse(state, 0)
        return list(results)

    def roll_dice(self) -> tuple[int, int]:
        """Roll two dice (seeded)."""
        d1, d2 = (int(self.rng.integers(1, 7)) for _ in range(2))
        return (d1, d2)

    def play_turn(self, dice: tuple[int, int], play: tuple) -> tuple:
        """Apply a chosen full-turn play ``play`` (a resulting state).

        ``play`` must be one of the states returned by ``legal_dice_plays``
        for the rolled dice; it is generated in-process but checked anyway.
        """
        if self.state is None:
            raise RuntimeError("call reset() first")
        player = self.state[2]
        expected = PipRaceEnv.legal_dice_plays(self.state, player, dice)
        if play not in expected:
            raise ValueError("play is not a legal outcome of the dice roll")
        self.state = play
        self.turn_count += 1
        return self.state


def linear_value(weights: np.ndarray, features: np.ndarray) -> float:
    """v(s) = w . phi(s). The linear value function of Sec 9.x / TD-Gammon."""
    return float(np.dot(weights, features))


def semi_gradient_td_lambda_update(
    weights: np.ndarray,
    eligibility: np.ndarray,
    features: np.ndarray,
    delta: float,
    alpha: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One semi-gradient TD(lambda) update step (Eq. 12.10/12.11, Sec 12.2).

    Accumulating traces:  e <- gamma * lam * e + phi(s)
                          w <- w + alpha * delta * e
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    eligibility[:] = gamma * lam * eligibility + features
    weights = weights + alpha * delta * eligibility
    return weights, eligibility


class TDGammonAgent:
    """Greedy (or epsilon-greedy) player driven by the linear value function.

    Mirrors TD-Gammon's move selection (Sec 16.1): each legal full-turn play
    produces an afterstate; the value of the afterstate is looked up, and the
    best play is chosen (White maximizes, Black minimizes, since the features
    are White-perspective). Weights are shared by both sides, exactly as in
    self-play TD-Gammon.
    """

    def __init__(self, weights: np.ndarray | None = None) -> None:
        if weights is None:
            weights = np.zeros(N_FEATURES, dtype=np.float64)
        self.weights = np.array(weights, dtype=np.float64)
        if self.weights.shape != (N_FEATURES,):
            raise ValueError(f"weights must have shape ({N_FEATURES},)")

    def value(self, state: tuple) -> float:
        """Estimated White win score for ``state``."""
        return linear_value(self.weights, pip_race_features(state))

    def choose_play(
        self,
        state: tuple,
        dice: tuple[int, int],
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple:
        """Choose a full-turn play: greedy on value, or epsilon-greedy."""
        player = state[2]
        plays = PipRaceEnv.legal_dice_plays(state, player, dice)
        if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
            return plays[int(rng.integers(0, len(plays)))]
        scored = [(linear_value(self.weights, pip_race_features(p)), i, p)
                  for i, p in enumerate(plays)]
        if player == WHITE:
            best = max(scored, key=lambda item: (item[0], -item[1]))[2]
        else:
            best = min(scored, key=lambda item: (item[0], item[1]))[2]
        return best


class RandomPipRaceAgent:
    """Baseline opponent: picks a uniformly random legal full-turn play."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def choose_play(self, state: tuple, dice: tuple[int, int], **kwargs) -> tuple:
        """Return a uniformly random legal play."""
        plays = PipRaceEnv.legal_dice_plays(state, state[2], dice)
        return plays[int(self.rng.integers(0, len(plays)))]


def simulate_game(
    white_agent: "TDGammonAgent | RandomPipRaceAgent",
    black_agent: "TDGammonAgent | RandomPipRaceAgent",
    seed: int | None = None,
) -> float:
    """Play one PipRace game with the given agents. Return the White outcome.

    +1 White wins, -1 Black wins, 0 draw / move-cap deadlock.
    """
    env = PipRaceEnv(seed=seed)
    state = env.reset()
    while not PipRaceEnv.is_over(state, env.turn_count):
        player = state[2]
        dice = env.roll_dice()
        if player == WHITE:
            play = white_agent.choose_play(state, dice)
        else:
            play = black_agent.choose_play(state, dice)
        state = env.play_turn(dice, play)
    return PipRaceEnv.outcome(state, env.turn_count)


def evaluate_match(
    agent: "TDGammonAgent | RandomPipRaceAgent",
    n_games: int = 200,
    seed: int | None = None,
    opponent_factory: Callable[[int | None], "RandomPipRaceAgent"] | None = None,
) -> float:
    """Agent win rate vs a baseline (default: random), colors alternated.

    Returns the mean outcome from ``agent``'s perspective: +1 win, 0 draw,
    -1 loss, averaged over ``n_games`` games (half as White, half as Black),
    mapped to a win rate in [0, 1] as (wins + 0.5 * draws) / n_games. The
    agent is used as passed; with ``epsilon = 0.0`` its play is deterministic.
    """
    if n_games <= 0:
        raise ValueError("n_games must be positive")
    opponent_factory = opponent_factory or (lambda s: RandomPipRaceAgent(seed=s))
    points = 0.0
    for game in range(n_games):
        env_seed = None if seed is None else seed + 1000 + game
        agent_is_white = (game % 2) == 0
        if agent_is_white:
            agent_outcome = simulate_game(agent, opponent_factory(env_seed), seed=env_seed)
        else:
            agent_outcome = -simulate_game(opponent_factory(env_seed), agent, seed=env_seed)
        points += 1.0 if agent_outcome > 0 else (0.5 if agent_outcome == 0 else 0.0)
    return points / n_games


def train_td_gammon(
    n_games: int = 2000,
    alpha: float = 0.05,
    lam: float = 0.7,
    gamma: float = 1.0,
    epsilon: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Train the linear PipRace value function by TD(lambda) self-play.

    Book Sec 16.1 (TD-Gammon self-play) and Sec 12.2 (semi-gradient TD).
    Both sides are played by the same improving weights; on each turn the
    position after the full dice play is treated as an afterstate and chosen
    greedily (epsilon-greedy with probability ``epsilon`` for exploration).
    Rewards: +1/-1/0 for win/loss/draw from White's perspective, and the
    same fixed-perspective convention is used by the features, so one shared
    value function serves both players. The terminal transition uses a zero
    bootstrap (V(terminal) = 0) with the game outcome as the immediate reward.

    Returns the trained weight vector of shape (N_FEATURES,).
    """
    if n_games <= 0:
        raise ValueError("n_games must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, N_FEATURES)

    for _ in range(n_games):
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
            delta = reward + gamma * v_next - v_state
            weights, eligibility = semi_gradient_td_lambda_update(
                weights, eligibility, features, delta, alpha, gamma, lam
            )
            state = env.play_turn(dice, play)

    return weights
