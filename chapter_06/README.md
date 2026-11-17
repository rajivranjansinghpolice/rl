# Chapter 6 — Temporal-Difference Learning

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 6

## Overview
Temporal-difference (TD) methods: TD prediction, on-policy control (Sarsa),
off-policy control (Q-learning), and TD(lambda) with accumulating traces —
and the classic random-walk comparison between TD(0) and constant-alpha MC.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 6.1–6.12 with from-scratch solutions, book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`td0_prediction`, `td_lambda_prediction`, `sarsa`, `q_learning`, `mc_prediction_incremental`, `RandomWalkEnv`, `run_random_walk_comparison`).

## Algorithms Implemented
- [x] algorithm_6.1 — Sarsa, prediction/control (`sarsa`) ✅
- [x] algorithm_6.2 — Q-learning (`q_learning`) ✅
- [x] algorithm_6.3 — TD(lambda) prediction (`td_lambda_prediction`) ✅
- [x] exercise_6.1 — Random walk with TD(0) and MC comparison (`RandomWalkEnv`, `run_random_walk_comparison`) ✅

Also included: `td0_prediction` (Algorithm 6.1 TD(0)) and
`mc_prediction_incremental` (sample-average MC baseline).

## Book Reference
- Section 6.1: TD prediction — TD(0) box; constant-alpha MC (Eq. 6.1/6.2)
- Section 6.2: Advantages of TD; Example 6.2 — random-walk MRP comparison
- Section 6.4: Sarsa (on-policy TD control)
- Section 6.5: Q-learning (off-policy TD control, max over next-state actions)
- Section 6.3 / 12.1: Optimality of TD(0); TD(lambda) traces

## Implementation Notes
- `RandomWalkEnv` implements Example 6.2 exactly: states A-E, start at C,
  +1 at the right terminal, undiscounted, v*(s) = (position+1)/6.
- `run_random_walk_comparison` reproduces Figure 6.2 with both methods
  initialized to V(s) = 0.5 and sharing the step size `alpha`: TD(0) learns
  faster (verified by test).
- `mc_prediction_incremental` implements the textbook sample-average MC.
- Import as a package: `from chapter_06.solution import ...`.

## Quick Start
```bash
python -m pytest chapter_06/tests/ -v
python -m pytest
ruff check chapter_06/
python scripts/run_quality_gates.py 06
```

---

*Last updated: 2026-11-12*
