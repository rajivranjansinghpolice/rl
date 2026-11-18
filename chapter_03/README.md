# Chapter 3 — Finite Markov Decision Processes

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 3

## Overview
This chapter implements finite MDPs end to end: the environment definition,
discounted returns, softmax preference policies, and the Bellman equations
(expectation + optimality), including exact solving and residual-based verification.

## Algorithms Implemented
- [x] algorithm_3.1 — MDP definition and environment (`FiniteMDP`, recycling robot factory `recycling_robot`, Example 3.5 5x5 gridworld factory `example_3_5_gridworld`) ✅
- [x] algorithm_3.2 — Return and reward-to-go (`compute_return`, `compute_returns_to_go`) ✅
- [x] algorithm_3.3 — Policy representation, softmax (`SoftmaxPolicy`) ✅
- [x] algorithm_3.4 — Bellman equations (`bellman_backup_expectation`, `bellman_backup_optimality`, `solve_bellman_expectation`, `verify_bellman`) ✅
- [x] Example 3.5 — 5x5 gridworld MDP, value iteration / policy evaluation (`example_3_5_gridworld`,
  `example_3_5_gridworld_random_policy_values`), `fig_3_5_gridworld.png` (random-policy and optimal
  value heatmaps). v*(A) = 24.4194 (book: 24.4); v_random(A) = 8.789 (book: ~8.8). ✅

## Book Reference
- Section 3.1–3.2: The Agent–Environment Interface and MDP dynamics `p(s', r | s, a)`
- Section 3.3: The Reinforcement Learning Problem (episodic/continuing tasks)
- Section 3.4: Policies and Value Functions (soft actions, Example 3.3 recycling robot)
- Section 3.5: Bellman Equations (expectation and optimality) and Example 3.5 (5x5 gridworld; figure numbering differs across editions — 2018 edition: Figs 3.2/3.5; 2015 in-progress draft PDF: the local reference figures)

## Exercises and Solutions
- [exercises.md](exercises.md) — Exercises 3.1–3.26 with from-scratch solutions, book §/equation
  citations, and coding-exercise pointers into this chapter (`FiniteMDP`, returns, Bellman backups).
- Figures: `figures/fig_3_3_recycling_robot.png` (Example 3.3 demonstration), `figures/fig_3_5_gridworld.png`
  (Example 3.5 5x5 gridworld — v_random heatmap and v* heatmap). Both are illustrative
  reproductions; figure identities cannot be verified against the local PDF summary text.
  See `solution.py` for the MDP builders and `plot_example_3_5_gridworld`.

## Environment Setup
All algorithms in this chapter use only `numpy`.

## Quick Start
```bash
# Run all tests in this chapter
python -m pytest chapter_03/tests/ -v

# Lint (E/F/N/W)
ruff check chapter_03/
```

## Implementation Notes
- `FiniteMDP` validates dynamics at construction time (probabilities sum to 1,
  known states/actions, terminal states need no transitions).
- Terminal states have value 0 by convention; Bellman backups and exact solves
  operate on the non-terminal block only.
- `verify_bellman()` with no arguments iterates optimality backups to a fixed
  point and checks the residual, so it works as a standalone verification check.

---

*Last updated: 2026-11-12*
