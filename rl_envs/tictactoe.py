"""Tic Tac Toe + Monte Carlo Tree Search (Sutton & Barto, 2nd ed., Sec. 8.11).

Ported from ``chapter_12/solution.py`` (Part A, algorithm_12.1). MCTS is
the tabular/UCT core of AlphaGo's search: the four steps Selection,
Expansion, Simulation, Backup. AlphaGo's deep policy/value networks are
deliberately not implemented here (see the chapter README).

Book Section 8.11 uses Go as the running example; Tic Tac Toe is the
smallest robust perfect-information game, so it is used as the test-bed.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


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
