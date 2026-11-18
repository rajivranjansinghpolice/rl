# Chapter 4 — Dynamic Programming

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 4

## Overview
Dynamic programming methods for solving finite MDPs given complete knowledge
of the environment: policy evaluation, policy improvement, policy iteration,
value iteration, Jack's car rental (Example 4.2), and the classic gambler's
problem (Example 4.3).

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 4.1–4.10 with from-scratch solutions, book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`policy_evaluation`, `policy_improvement`, `policy_iteration`, `value_iteration`, `gamblers_problem`).

## Algorithms Implemented
- [x] algorithm_4.1 — Policy evaluation, iterative (`policy_evaluation`, in-place and Jacobi variants) ✅
- [x] algorithm_4.2 — Policy improvement (`policy_improvement`) ✅
- [x] algorithm_4.3 — Policy iteration (`policy_iteration`) ✅
- [x] algorithm_4.4 — Value iteration (`value_iteration`) ✅
- [x] exercise_4.1 — Gambler's problem (`gamblers_problem`, Example 4.3 / Exercise 4.9) ✅
- [x] Example 4.2 — Jack's car rental (`JacksCarRentalMDP` + `policy_iteration`, full Poisson/return dynamics with exact tails; Fig. 4.2) ✅
- [x] Exercise 4.5 (Programming) — Jack's car rental with modifications (free first 1→2 shuttle, parking fee for >10 cars overnight) ✅

## Book Reference
- Section 4.1: Iterative policy evaluation (converges as \(Δ \le \theta\))
- Section 4.2: Policy improvement theorem — greedy policy is as good as or better
- Section 4.3: Policy iteration (Jack's car rental is Example 4.2)
- Section 4.4: Value iteration (Section 4.4 box, Eq. 4.10)
- Example 4.2 / Example 4.3: Jack's car rental; gambler's problem (Exercise 4.9)
  - In the 2nd (2018) edition, Figure 4.2 is the Jack's car rental figure
    and Figure 4.3 is the gambler's problem figure. The in-progress
    2nd-edition draft (and the 1st edition) number them differently;
    this chapter reproduces both as illustrative demonstrations.
  - Jack's car rental is solved exactly (linear solve per policy-iteration
    round over the 441-state / 11-action MDP; Poisson rentals λ1=3, λ2=4,
    Poisson returns λ1=3, λ2=2, $10/rental, $2/car moved, γ=0.9). Converges
    in ~5 rounds from the zero-move policy; v*(0,0) ≈ 421.
  - The gambler's problem optimal policy is highly non-unique; ties are
    broken toward the SMALLEST stake, reproducing the book's Figure 4.3
    staircase (spikes at capitals 25/50/75). Do not expect pixel-identity
    with any printed edition.

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
  Ties between optimal stakes break toward the smallest stake, which
  reproduces the book's Figure 4.3 pyramid/staircase shape.
- `JacksCarRentalMDP` (Example 4.2) is a dedicated vectorized implementation
  (dense per-action transition matrices, exact Poisson tails folded in)
  rather than the generic `FiniteMDP` solver, because exact policy
  evaluation over 441 states needs direct 441×441 linear solves. The
  transitions are shared by the base problem (Figure 4.2 reproduction) and
  the Exercise 4.5 modifications (`free_first_shuttle=True`,
  `second_parking_fee=4.0`), which are off by default.
- Greedy ties are split uniformly when apportioning probability among
  maximal actions (DP solvers); the gambler's problem uses the smallest-stake
  tie-break instead, matching its printed figure's shape.

## Figures
- `figures/fig_4_2_jacks_car_rental.png` — policy improvement sequence
  (π₀ … π₄) and the final value function of Jack's car rental (Example 4.2,
  Fig. 4.2 in the 2nd edition; illustrative demonstration). Regenerate with:

  ```bash
  python chapter_04/figures/generate_figures.py
  ```

- `figures/fig_4_3_gamblers_problem.png` — gambler's problem value function
  and optimal policy (Example 4.3, Fig. 4.3 in the 2nd edition; policy spikes
  are a valid optimal behavior — ties broken toward the smallest stake).

## Quick Start
```bash
python -m pytest chapter_04/tests/ -v
python -m pytest                     # full suite
ruff check chapter_04/
python scripts/run_quality_gates.py 04
```

---

*Last updated: 2026-11-18*
