# Chapter 4 — Dynamic Programming

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 4

## Overview
Dynamic programming methods for solving finite MDPs given complete knowledge
of the environment: policy evaluation, policy improvement, policy iteration,
value iteration, and the classic gambler's problem.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 4.1–4.10 with from-scratch solutions, book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`policy_evaluation`, `policy_improvement`, `policy_iteration`, `value_iteration`, `gamblers_problem`).

## Algorithms Implemented
- [x] algorithm_4.1 — Policy evaluation, iterative (`policy_evaluation`, in-place and Jacobi variants) ✅
- [x] algorithm_4.2 — Policy improvement (`policy_improvement`) ✅
- [x] algorithm_4.3 — Policy iteration (`policy_iteration`) ✅
- [x] algorithm_4.4 — Value iteration (`value_iteration`) ✅
- [x] exercise_4.1 — Gambler's problem (`gamblers_problem`, Example 4.3 / Exercise 4.9) ✅

## Book Reference
- Section 4.1: Iterative policy evaluation (converges as `Δ ≤ θ`)
- Section 4.2: Policy improvement theorem — greedy policy is as good as or better
- Section 4.3: Policy iteration (Jack's car rental is Example 4.2)
- Section 4.4: Value iteration (Section 4.4 box, Eq. 4.10)
- Example 4.3 / Exercise 4.9: Gambler's problem, solved by value iteration, p_h ∈ {0.25, 0.4, 0.55}

## Implementation Notes
- Reuses `FiniteMDP` dynamics from `chapter_03.solution`; test modules import
  it via the package (`from chapter_04.solution import ...`).
- The Example 4.1 four-by-four gridworld itself (terminals on the main
  diagonal, `r = -1`, `gamma = 1`, converged random-policy values in
  Figure 4.1) is available as `rl_envs.Example41GridWorld` with unit tests in
  `tests/test_rl_envs.py`; the DP solvers in this chapter are MDP-generic and
  do not hard-code it.
- Terminal states are forced to value 0 everywhere (dummy terminal convention).
- `gamblers_problem` models the gambler's MDP directly with dummy terminals
  valued 0 and 1 (Exercise 4.9), since stakes are state-dependent — a
  dedicated vectorized value iteration is used instead of a generic MDP file.
- Greedy ties are split uniformly when apportioning probability among
  maximal actions.

## Quick Start
```bash
python -m pytest chapter_04/tests/ -v
python -m pytest                     # full suite
ruff check chapter_04/
python scripts/run_quality_gates.py 04
```

---

*Last updated: 2026-11-12*
