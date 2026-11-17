# Chapter 9 — Planning and Learning with Tabular Methods

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 8
> (repo Chapter 9)

## Overview
Model-based reinforcement learning: the Dyna architecture (direct RL,
model-learning, planning), prioritized sweeping, expected vs sample
updates, and world models for stochastic environments. All experiments use
a small gridworld built on chapter_03's `FiniteMDP`.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 8.1–8.4 with from-scratch solutions, book §/PDF-page citations, and coding-exercise pointers into this chapter (`dyna_q`, `TabularModel`, `prioritized_sweeping`, `expected_dyna_q`, `make_gridworld`).

## Algorithms Implemented
- [x] algorithm_9.1 — Dyna-Q planner (`dyna_q`) ✅
- [x] exercise_9.1 — Prioritized sweeping (`prioritized_sweeping`, `PriorityQueue`) ✅
- [x] exercise_9.2 — Expectation vs update Dyna (`expected_dyna_q`, `expected_sweep`, `compare_sample_vs_expected_dyna`) ✅
- [x] exercise_9.3 — World model with stochastic environments (`TabularModel`, `make_gridworld`) ✅

Also included: `expected_q_target` (eq. 8.1), `optimal_q_values` (exact
q* via chapter_03's Bellman optimality backup), `PriorityQueue` (max-priority
queue with lazy deletion of stale entries).

## Book Reference
- Section 8.1: Models and planning — distribution models vs sample models
- Section 8.2: Dyna-Q — Tabular Dyna-Q box (Example 8.1, maze)
- Section 8.4: Prioritized sweeping (Moore & Atkeson 1993); stochastic
  environments use expected updates
- Section 8.5: Expected vs sample updates — eq. (8.1) and (8.2)
- Section 8.6/8.7: trajectory sampling / RTDP (not implemented)

## Implementation Notes
- `TabularModel` generalizes the book's deterministic sample model to a
  **distribution model** built from per-outcome counts; it serves both as a
  sample model (`sample`) and a distribution model (`distribution`). This is
  the Exercise 9.3 world model: stochastic transitions are represented as
  outcome distributions, not collapsed to an average.
- `make_gridworld(rows, cols, slip=0.0)`: states `'r,c'`, actions
  up/down/left/right, +1 on entering the terminal goal, blocked moves stay
  in place. `slip > 0` sends the two perpendicular moves `slip/2` each.
  `slip = 0` recovers the deterministic world.
- `dyna_q` implements the book's "Tabular Dyna-Q" steps (a)-(f): acting with
  an epsilon-greedy policy, a direct Q-learning update, model learning, and
  `n_planning` planning updates per real step sampled from previously
  observed state-action pairs (eq. 8.2 sample updates).
- `prioritized_sweeping` implements the "Prioritized sweeping" box with the
  book's stochastic-environment extension: updates use the model's expected
  update (eq. 8.1); predecessors are re-prioritized with the same priority
  measure; the priority queue never holds duplicate stale entries for a pair.
- `expected_dyna_q` is Dyna-Q with expected one-step q* updates (eq. 8.1,
  direct assignment); `expected_sweep` performs full distribution sweeps over
  all known pairs — exact in the limit (verified against `optimal_q_values`).
- `compare_sample_vs_expected_dyna(env_factory, ...)` trains both variants
  on the same stochastic MDP and reports sample/expected max error on pairs
  each model actually knows (`known_max_q_error`) plus total `max_q_error`.

## Quick Start
```bash
python -m pytest chapter_09/tests/ -v
python -m pytest
ruff check chapter_09/
```

---

*Last updated: 2026-11-14*
