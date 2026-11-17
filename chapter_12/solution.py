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
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

# ============================================================================
# Part A - algorithm_12.1: Monte Carlo Tree Search (UCT) for Tic Tac Toe
# ============================================================================


class TicTacToe:
    """Tic Tac Toe, a two-player zero-sum perfect-information game.

    Book Section 8.11 (MCTS) uses Go as the running example; Tic Tac Toe is
    the smallest robust perfect-information game, so it is used here as the
    test-bed for the MCTS algorithm (algorithm_12.1).

    The board is an immutable tuple of 9 cells, each 'X', 'O' or ' ' (empty).
    The tuple is hashable and is used directly as the canonical state
    representation (see ``canonical_key``): two states compare and hash equal
    iff they are the same position with the same player to move implied by
    the piece counts. 'X' always moves first.
    """

    X = "X"
    NOUGHT = "O"
    EMPTY = " "
    PLAYERS = (X, NOUGHT)
    WIN_LINES = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6),             # diagonals
    )

    @staticmethod
    def initial_state() -> tuple[str, ...]:
        """Return the empty board (9 empty cells, 'X' to move)."""
        return tuple(TicTacToe.EMPTY for _ in range(9))

    @staticmethod
    def canonical_key(state: tuple[str, ...]) -> tuple[str, ...]:
        """Return the canonical (hashable) representation of a board.

        The board tuple itself is canonical for this game: it is immutable,
        hashable, and identical for identical positions. The player to move
        is a function of the piece counts (``current_player``) and therefore
        needs no extra bits.
        """
        board = tuple(state)
        if len(board) != 9:
            raise ValueError("board must have exactly 9 cells")
        return board

    @staticmethod
    def current_player(state: tuple[str, ...]) -> str:
        """Return the player to move ('X' first; counts differ by at most 1)."""
        TicTacToe.canonical_key(state)
        x_count = state.count(TicTacToe.X)
        o_count = state.count(TicTacToe.NOUGHT)
        if x_count - o_count not in (0, 1):
            raise ValueError("invalid board: piece counts differ by more than one")
        return TicTacToe.X if x_count == o_count else TicTacToe.NOUGHT

    @staticmethod
    def legal_moves(state: tuple[str, ...]) -> list[int]:
        """Return the indices of the empty cells, in order."""
        TicTacToe.canonical_key(state)
        return [i for i, cell in enumerate(state) if cell == TicTacToe.EMPTY]

    @staticmethod
    def apply_move(state: tuple[str, ...], move: int) -> tuple[str, ...]:
        """Return the board after the current player plays ``move``.

        Raises ValueError for an out-of-range or occupied cell.
        """
        TicTacToe.canonical_key(state)
        if not 0 <= move < 9:
            raise ValueError("move must be an index in [0, 9)")
        if state[move] != TicTacToe.EMPTY:
            raise ValueError("cell is not empty")
        player = TicTacToe.current_player(state)
        cells = list(state)
        cells[move] = player
        return tuple(cells)

    @staticmethod
    def winner(state: tuple[str, ...]) -> str | None:
        """Return 'X' or 'O' if that player has three in a row, else None."""
        TicTacToe.canonical_key(state)
        for a, b, c in TicTacToe.WIN_LINES:
            if state[a] != TicTacToe.EMPTY and state[a] == state[b] == state[c]:
                return state[a]
        return None

    @staticmethod
    def is_terminal(state: tuple[str, ...]) -> bool:
        """Return True if the game is over (someone won, or the board is full)."""
        return TicTacToe.winner(state) is not None or not TicTacToe.legal_moves(state)

    @staticmethod
    def outcome(state: tuple[str, ...], player: str) -> float:
        """Terminal outcome for ``player``: +1 win, -1 loss, 0 draw.

        Raises ValueError if called on a non-terminal state.
        """
        if not TicTacToe.is_terminal(state):
            raise ValueError("outcome is only defined for terminal states")
        win = TicTacToe.winner(state)
        if win is None:
            return 0.0
        return 1.0 if win == player else -1.0


def random_rollout(
    state: tuple[str, ...],
    rng: np.random.Generator,
    rollout_policy: Callable | None = None,
) -> float:
    """Simulate a complete game from ``state`` to a terminal state.

    Roll back the outcome from the perspective of the player to move at
    ``state``. Actions are sampled from ``rollout_policy`` if given, else
    uniformly at random. The simulation stops at a terminal state and never
    plays past the end of the game (Book Section 8.11, Simulation step).
    """
    player = TicTacToe.current_player(state)
    current = tuple(state)
    while not TicTacToe.is_terminal(current):
        moves = TicTacToe.legal_moves(current)
        if rollout_policy is not None:
            move = int(rollout_policy(current, moves, rng))
        else:
            move = moves[int(rng.integers(0, len(moves)))]
        current = TicTacToe.apply_move(current, move)
    return TicTacToe.outcome(current, player)


def uct_score(
    value_mean: float,
    child_visits: int,
    parent_visits: int,
    c_uct: float,
) -> float:
    """UCB1 / UCT selection score for one child edge. Book Section 2.7 + 8.11.

    score = value_mean + c_uct * sqrt(ln(parent_visits) / child_visits)

    ``value_mean`` must be the mean action value from the perspective of the
    player choosing at the parent (for alternating zero-sum games, the
    negated child mean). An unvisited child (``child_visits <= 0``) scores
    +inf so every child is tried at least once before re-selection.
    """
    if child_visits <= 0:
        return math.inf
    if parent_visits <= 0:
        parent_visits = 1
    exploration = c_uct * math.sqrt(math.log(parent_visits) / child_visits)
    return value_mean + exploration


class _MCTSNode:
    """One node of the search tree (a board position). Internal use only.

    ``value_sum`` accumulates returns from the perspective of the player to
    move at *this* node, so the value of a move from the parent's perspective
    is the negated child mean. ``untried_moves`` holds moves not yet expanded.
    """

    def __init__(
        self,
        state: tuple[str, ...],
        parent: "_MCTSNode | None",
        move: int | None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[_MCTSNode] = []
        self.untried_moves: list[int] = TicTacToe.legal_moves(state)
        self.visits = 0
        self.value_sum = 0.0


def _select_uct_child(node: _MCTSNode, c_uct: float) -> _MCTSNode:
    """Pick the child whose UCT score (from node's perspective) is highest."""
    best_score = -math.inf
    best_child = node.children[0]
    for child in node.children:
        # Negate: child's stored value is from the child's player-to-move
        # perspective, i.e. the opponent of the player choosing here.
        value_mean = -(child.value_sum / child.visits) if child.visits else 0.0
        score = uct_score(value_mean, child.visits, node.visits, c_uct)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _tree_policy_path(root: _MCTSNode, c_uct: float) -> list[_MCTSNode]:
    """Selection: descend from root by UCT until a terminal or expandable leaf."""
    path = [root]
    node = root
    while True:
        if TicTacToe.is_terminal(node.state) or node.untried_moves:
            return path
        node = _select_uct_child(node, c_uct)
        path.append(node)


def _backup(path: list[_MCTSNode], value: float) -> None:
    """Backup: propagate the return up the path, flipping sign each ply.

    ``value`` must be the return from the perspective of the player to move
    at the last node of ``path`` (the leaf where the simulation started or
    the terminal node reached by selection).
    """
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


class MCTSPlayer:
    """MCTS agent wrapper: picks one move per game state with a fixed budget.

    Book Section 8.11 (Monte Carlo Tree Search). Each call to ``search``
    rebuilds the tree from the root and runs ``n_simulations`` iterations of
    Selection -> Expansion -> Simulation -> Backup. The returned move is the
    root child with the most visits (the book notes the largest action value
    or the largest visit count are both legitimate final selection rules;
    max visits is the more robust one).
    """

    def __init__(
        self,
        n_simulations: int = 200,
        c_uct: float = math.sqrt(2.0),
        seed: int | None = None,
        rollout_policy: Callable | None = None,
    ) -> None:
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if c_uct < 0.0:
            raise ValueError("c_uct must be non-negative")
        self.n_simulations = int(n_simulations)
        self.c_uct = float(c_uct)
        self.seed = seed
        self.rollout_policy = rollout_policy

    def search(self, state: tuple[str, ...]) -> int:
        """Run the full MCTS and return the chosen move (cell index)."""
        state = TicTacToe.canonical_key(state)
        if TicTacToe.is_terminal(state):
            raise ValueError("search requires a non-terminal state")
        rng = np.random.default_rng(self.seed)
        root = _MCTSNode(state, parent=None, move=None)

        for _ in range(self.n_simulations):
            # 1. Selection (expandable or terminal leaf reached).
            path = _tree_policy_path(root, self.c_uct)
            leaf = path[-1]

            # 2. Expansion: add one untried child, if any.
            if not TicTacToe.is_terminal(leaf.state) and leaf.untried_moves:
                index = int(rng.integers(0, len(leaf.untried_moves)))
                move = leaf.untried_moves.pop(index)
                child_state = TicTacToe.apply_move(leaf.state, move)
                child = _MCTSNode(child_state, parent=leaf, move=move)
                leaf.children.append(child)
                path.append(child)
                leaf = child

            # 3. Simulation: rollout from the leaf unless it is terminal.
            if TicTacToe.is_terminal(leaf.state):
                # Outcome from the leaf player's perspective (they lost or
                # drew); _backup flips the sign as it ascends.
                value = TicTacToe.outcome(leaf.state, TicTacToe.current_player(leaf.state))
            else:
                value = random_rollout(leaf.state, rng, self.rollout_policy)

            # 4. Backup from the leaf toward the root, flipping perspective.
            _backup(path, value)

        if not root.children:
            # Degenerate case (e.g. n_simulations=1 with immediate terminal):
            # fall back to a uniformly random legal move.
            moves = TicTacToe.legal_moves(state)
            return moves[int(rng.integers(0, len(moves)))]
        best = max(root.children, key=lambda child: (child.visits, child.value_sum))
        assert best.move is not None  # every non-root node carries its move
        return best.move

    def select_move(self, state: tuple[str, ...]) -> int:
        """Alias for ``search`` (uniform player interface used in tests)."""
        return self.search(state)


# ============================================================================
# Part B - exercise_12.1: TD-Gammon-like player (PipRace) with TD(lambda)
# ============================================================================

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


class RandomPlayer:
    """Baseline opponent: plays a uniformly random legal move."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_move(self, state: tuple[str, ...]) -> int:
        """Return a uniformly random legal move."""
        moves = TicTacToe.legal_moves(state)
        return moves[int(self.rng.integers(0, len(moves)))]


class GreedyPlayer:
    """Simple heuristic opponent: win > block > center > corner > any.

    This is the "reasonable heuristic/greedy player" used in the tests. It is
    deterministic apart from seeded tie-breaking and is clearly beatable by a
    search agent, but it is not a strawman: it takes immediate wins and
    blocks immediate losses.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_move(self, state: tuple[str, ...]) -> int:
        """Return the greedy-heuristic move for the current player."""
        player = TicTacToe.current_player(state)
        opponent = TicTacToe.NOUGHT if player == TicTacToe.X else TicTacToe.X
        moves = TicTacToe.legal_moves(state)
        for move in moves:
            if TicTacToe.winner(TicTacToe.apply_move(state, move)) == player:
                return move
        for move in moves:
            board = list(state)
            board[move] = opponent
            if TicTacToe.winner(tuple(board)) == opponent:
                return move
        if state[4] == TicTacToe.EMPTY:
            return 4
        for corner in (0, 2, 6, 8):
            if state[corner] == TicTacToe.EMPTY:
                return corner
        return moves[int(self.rng.integers(0, len(moves)))]


def play_game(player_x: MCTSPlayer | RandomPlayer | GreedyPlayer,
              player_o: MCTSPlayer | RandomPlayer | GreedyPlayer) -> str | None:
    """Play one full Tic Tac Toe game. Returns 'X', 'O', or None for a draw."""
    state = TicTacToe.initial_state()
    while not TicTacToe.is_terminal(state):
        current = TicTacToe.current_player(state)
        mover = player_x if current == TicTacToe.X else player_o
        move = mover.select_move(state)
        state = TicTacToe.apply_move(state, move)
    return TicTacToe.winner(state)


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
