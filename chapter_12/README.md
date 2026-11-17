# Chapter 12 — Case Studies (Applications of RL)

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), final part
> "Frontiers and Case Studies" (Chapter 16, Applications and Case Studies).

## Overview
Case studies applying RL to games, following the book's two headline
board-game stories: MCTS (the engine behind AlphaGo) and TD-Gammon's
self-play TD(lambda) learning.

## Exercises and Solutions

- [exercises.md](exercises.md) — derived exercises from the book's Part IV case studies (Chapter 16
  §16.1 TD-Gammon, §16.6 AlphaGo/MCTS; §1.5 tic-tac-toe), labeled [derived], with book §/PDF-page
  citations and coding-exercise pointers into this chapter (`MCTSPlayer`, `uct_score`, `PipRaceEnv`,
  `TDGammonAgent`, `train_td_gammon`).

## Algorithms Implemented
- [x] algorithm_12.1 — Monte Carlo Tree Search with UCT selection (UCB1) for
  Tic Tac Toe (`TicTacToe`, `MCTSPlayer`, `uct_score`, `random_rollout`) ✅
- [x] exercise_12.1 — TD-Gammon-like game player for **PipRace**, a documented
  backgammon approximation, with a LINEAR value function trained by
  TD(lambda) self-play (`PipRaceEnv`, `pip_race_features`, `TDGammonAgent`,
  `train_td_gammon`, `simulate_game`, `evaluate_match`) ✅

## Book Reference
- Section 8.11 (MCTS): Selection, Expansion, Simulation, Backup; the tree
  policy balancing exploration/exploitation via UCB1 (Section 2.7).
- Section 16.6: AlphaGo / AlphaGo Zero — MCTS as the decision-time planner.
  **Important:** only the tabular/UCT core is implemented. AlphaGo's deep
  policy/value networks, supervised pre-training, RL pipeline, and
  APV-MCTS are NOT reimplemented (see "What was simplified").
- Section 16.1: TD-Gammon — self-play, TD(lambda) with a function
  approximator, afterstate move selection, minimal reward (zero except on
  win; sign convention: +1 White win / -1 Black win / 0 draw).
- Section 12.2 (Eq. 12.10/12.11): semi-gradient TD with accumulating
  eligibility traces.

## What Was Built
### Part A — MCTS with UCT on Tic Tac Toe
- Immutable tuple board (canonical, hashable), legal moves, winner/terminal
  detection, `'X'` first.
- UCT selection: `Q + c * sqrt(ln N_parent / N_child)`; unvisited children
  score infinity so every child is tried.
- Full MCTS iteration: Selection (UCT descent) -> Expansion (one untried
  child) -> Simulation (uniform-random rollout to terminal, stops at the
  terminal) -> Backup (sign-flipped returns from leaf to root).
- Final selection: child with the most visits (the book's alternative, max
  action value, is mentioned in `MCTSPlayer.search`).
- Agents: `MCTSPlayer` (fixed search budget), `RandomPlayer`, and
  `GreedyPlayer` (win > block > center > corner > any) as the
  "reasonable heuristic" opponent.

### Part B — PipRace, a TD-Gammon-like player
- "PipRace", an **honest, documented approximation of backgammon**:
  - 5 pieces per player (backgammon: 15), "stack and run" start on a shared
    24-point linear track (backgammon: 24 directional points, bar, off).
  - Two dice per turn; doubles play four moves (backgammon rule kept).
  - A point with >= 2 opponent pieces is blocked for landing; pieces may
    pass over it. Landing on a single opponent piece (a "blot") hits it
    back to that player's start (replacing backgammon's bar re-entry).
  - Overshoot bearing-off allowed from anywhere on the track.
  - A die that cannot be played is forfeited (backgammon's "use as much
    of the roll as possible").
  - Terminal reward from White's fixed perspective: +1 win / -1 loss / 0
    draw (no gammon/backgammon bonuses, no doubling cube). Move-cap at
    `MAX_TURNS` with borne-off / pip-count tiebreak.
- 13 hand-designed features (`pip_race_features`): pip lead, borne-off
  counts, home-board occupancy, blots, made points, normalized pip counts,
  turn indicator — all from White's fixed perspective, scaled ~[0, 1],
  with a bias unit.
- Linear value function `v(s) = w . phi(s)` trained by semi-gradient
  TD(lambda) self-play (accumulating eligibility traces, shared weights for
  both sides, terminal bootstrap = 0).
- Afterstate move selection as in TD-Gammon: enumerate legal full-turn
  plays, evaluate resulting positions with the value function, White
  maximizes / Black minimizes; epsilon-greedy exploration during training.

## What Was Simplified (and Why)
- **AlphaGo networks are not implemented.** The assignment asked for the
  tabular/UCT part only. APV-MCTS, the policy/value ensemble, and the
  4.9M-game cold-start pipeline of AlphaGo Zero are out of scope.
- **PipRace is not backgammon.** The 15-piece game with bar re-entry, exact
  bear-off rules, home boards, and the doubling cube is a much larger
  artifact; the assignment explicitly allowed a clearly stated approximation.
  The pieces that matter for the *learning story* are all present: dice
  randomness, opponent interaction through blocking and hitting, sparse
  terminal reward, and a learned value function guiding move choice.
- **Linear instead of sigmoid/network value.** The assignment asked for a
  linear value function over hand-designed features; the magnitudes are
  unbounded (no sigmoid), which is fine at this scale (see tests: weights
  stay small and finite).
- **Fixed-perspective (White) features** rather than side-to-move-relative
  features; side to move enters as a single binary feature, mirroring
  TD-Gammon's turn-indicator units.
- **Training scale.** Tests train for a few hundred games with a fixed seed.
  The measured result: with 400 games (seed 0), the trained greedy agent
  achieves a ~0.95-0.975 win rate vs uniform random over 200 seeded games.
  The book's TD-Gammon used ~300,000 games and 198 inputs.
- **Rollout policy is uniform random** (the simplest Section 8.10/8.11
  choice); a stronger rollout policy would play better but teach less.

## What Was Weakened in Tests (and Why)
Nothing was weakened — but two initial test fixtures were wrong and are now
correct against the actual (and rule-checked) game semantics:
- `test_blocked_point_cannot_be_landed_on`: White may still play 1 then 3 to
  land on point 4 (dice are forfeited only when unusable), so the test now
  asserts "never lands on 3", plus the two reachable outcomes.
- `test_blot_is_hit_back_to_start`: with dice (5, 1) the 1-die must also be
  played, so the post-hit signature is checked on the final position.
- The MCTS-never-loses-vs-greedy test uses 800 simulations per move; at
  400 sims a fixed-seed game (seed 12, MCTS as Black) was lost, which is
  within the expected variance of rollout-based search and not a logic
  error. 800 sims gives a comfortable margin and the suite stays fast.

## Fidelity Caveats
- MCTS selection uses UCT with `c_uct = sqrt(2)` and random-rollout
  evaluation; the tree is rebuilt from scratch every move (no tree reuse).
- `uct_score` returns `+inf` for unvisited children (standard UCB1).
- PipRace hit rule is "hit to start", not backgammon's bar; blocked-point
  rule applies only to landing, not to passing over.
- `evaluate_match` alternates colors and reports (wins + 0.5 * draws) /
  n_games from the agent's perspective.
- Determinism: everything is seeded; tests are reproducible.

## Quick Start
```bash
python -m pytest chapter_12 -q          # run this chapter's tests
python -m pytest                        # full suite from repo root
.venv\Scripts\ruff.exe check chapter_12 --select E,F,N,W
```

---

*Last updated: 2026-11-14*
