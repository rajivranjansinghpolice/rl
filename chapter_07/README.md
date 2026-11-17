# Chapter 7 — n-step Bootstrapping

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 7

## Overview
Generalizes one-step TD and Monte Carlo into a spectrum of n-step methods:
on-policy n-step Sarsa control, the tree-backup algorithm (off-policy without
importance sampling), and SARSA(lambda) with eligibility traces.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 7.1–7.8 with paraphrased statements, §/Eq.
  citations, derivations/pseudocode, and repo-accurate coding pointers. Off-policy
  importance-sampling / per-decision exercises are marked theory-only where no API exists.

## Algorithms Implemented
- [x] n-step Sarsa control (`n_step_sarsa`) ✅
- [x] n-step Tree Backup (`tree_backup`) ✅
- [x] SARSA(lambda) control with accumulating traces (`sarsa_lambda`) ✅

> Note: no `td0_prediction` or off-policy n-step IS/per-reward API is implemented in this chapter.

## Book Reference
- Section 7.1: n-step TD prediction (19-state random walk, Example 7.1)
- Section 7.2: n-step Sarsa (box, Eq. 7.4-7.5)
- Section 7.5: n-step Tree Backup (box; delta/Z recurrence, Eq. 7.12-7.13)
- Section 12.2: SARSA(lambda) traces (adapting Sarsa-lambda control)

## Implementation Notes
- Reuses `RandomWalkEnv` from `chapter_06.solution` (Example 6.2's 5-state walk).
- `n_step_sarsa` updates every time tau = t - n + 1 with the n-step return;
  `tree_backup` implements the book's exact delta/Z backup with the target as
  the epsilon-greedy policy; both target q* semantics like the epsilon-greedy
  on-policy boxes.
- `sarsa_lambda` uses accumulating eligibility traces (step size alpha,
  trace decay gamma * lambda).

## Quick Start
```bash
python -m pytest chapter_07/tests/ -v
python -m pytest
ruff check chapter_07/
python scripts/run_quality_gates.py 07
```

---

*Last updated: 2026-11-12*
