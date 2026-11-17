# Chapter 5 — Monte Carlo Methods

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 5

## Overview
Monte Carlo (MC) methods for estimating value functions and discovering
optimal policies from sample episodes — no model required.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 5.1–5.9 with from-scratch solutions, book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`mc_first_visit_prediction`, `mc_every_visit_prediction`, `mc_control_epsilon_greedy`, `mc_off_policy_prediction`, `mc_control_off_policy`, `BlackjackEnv`).

## Algorithms Implemented
- [x] algorithm_5.1 — MC prediction, first-visit (`mc_first_visit_prediction`) ✅
- [x] algorithm_5.2 — MC prediction, every-visit (`mc_every_visit_prediction`) ✅
- [x] algorithm_5.3 — MC control with epsilon-greedy (`mc_control_epsilon_greedy`) ✅
- [x] algorithm_5.4 — MC control with ordinary importance sampling (`mc_control_off_policy`, `weighted=False`) ✅
- [x] algorithm_5.5 — MC control with weighted importance sampling (`mc_control_off_policy`, `weighted=True`) ✅
- [x] exercise_5.1 — Blackjack (21) environment and MC agent (`BlackjackEnv`, `blackjack_stick_on_threshold`) ✅

Off-policy prediction for both estimators is also included
(`mc_off_policy_prediction`).

## Book Reference
- Section 5.1: MC prediction — two box algorithms (first-visit), every-visit variant
- Section 5.3: Monte Carlo ES and the GPI pattern for MC control
- Section 5.4: On-policy MC control, epsilon-soft (epsilon-greedy) policies
- Section 5.5: Off-policy prediction via importance sampling (ordinary Eq. 5.4, weighted Eq. 5.5)
- Section 5.6: Incremental weighted-IS update (C/W recurrences, Eq. 5.6-5.7)
- Section 5.7: Off-policy MC control (epsilon-soft behavior, greedy target)
- Example 5.1 / 5.3: Blackjack environment and solving it with MC control

## Implementation Notes
- Environments implement `reset() -> observation` and
  `step(action) -> (reward, observation, done)`. `BlackjackEnv` returns
  `(player_sum, dealer_showing, usable_ace)` and serves as both the
  prediction/control workhorse and Exercise 5.1's environment.
- Policy functions are callables `policy_fn(state) -> probability vector`
  over `ACTIONS`; uniform exploration = `[0.5, 0.5]`.
- Off-policy prediction is `every-visit` by default (`first_visit=False` for
  first-visit) — matching the book's every-visit off-policy algorithm.
- `BlackjackEnv` implements infinite-deck cards, natural-win handling, the
  fixed dealer rule (hit below 17), and usable-ace accounting.

## Quick Start
```bash
python -m pytest chapter_05/tests/ -v
python -m pytest
ruff check chapter_05/
python scripts/run_quality_gates.py 05
```

---

*Last updated: 2026-11-12*
