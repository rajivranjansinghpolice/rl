# Chapter 5 — Monte Carlo Methods

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 5

## Overview
Monte Carlo (MC) methods for estimating value functions and discovering
optimal policies from sample episodes — no model required.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 5.1–5.9 with from-scratch solutions, book §/equation/PDF-page citations, and coding-exercise pointers into this chapter and `rl_envs` (`mc_first_visit_prediction`, `mc_every_visit_prediction`, `mc_control_epsilon_greedy`, `mc_exploring_starts`, `mc_off_policy_prediction`, `mc_control_off_policy`, `plot_importance_sampling`, `plot_ordinary_is_unstable`, `BlackjackEnv`, `RacetrackEnv`).

## Algorithms Implemented
Book algorithm boxes in Chapter 5 are referred to by name and section (the
printed box numbers differ between the 2nd edition and the in-progress
draft), not by an invented index. Implemented algorithms:
- [x] MC prediction, first-visit — §5.1 (`mc_first_visit_prediction`) ✅
- [x] MC prediction, every-visit — §5.1 (`mc_every_visit_prediction`) ✅
- [x] Monte Carlo ES (exploring starts), optimal-control algorithm — §5.3 (`mc_exploring_starts`) ✅
- [x] On-policy MC control for ε-soft policies — §5.4 (`mc_control_epsilon_greedy`) ✅
- [x] Off-policy MC control, weighted importance sampling — §5.7/§5.6 (`mc_control_off_policy`, `weighted=True`) ✅
- [x] Off-policy MC control, ordinary importance sampling — §5.7/§5.6 (`mc_control_off_policy`, `weighted=False`) ✅
- [x] Example 5.5 one-state infinite-variance IS problem — §5.5 (`OneStateIsEnv`, `_one_state_is_samples`) ✅
- [x] Blackjack (21) environment — Example 5.1 (`BlackjackEnv`, `blackjack_stick_on_threshold`) ✅
- [x] Racetrack environment — Exercise 5.8 (`RacetrackEnv` in `rl_envs`) ✅

Off-policy prediction for both estimators is also included
(`mc_off_policy_prediction`), in both first-visit and every-visit variants.

## Figures
All figures are regenerable via `python chapter_05/figures/generate_figures.py`
(run from the repository root). Figure identities refer to the 2nd (2018)
edition; the in-progress draft and 1st edition number them differently.
All are illustrative demonstrations, not pixel-identical redraws:
- `figures/fig_5_1_blackjack_value.png` — Fig. 5.1: stick-on-20 value functions (3D surfaces)
- `figures/fig_5_2_blackjack_optimal.png` — Fig. 5.2: optimal policy and v* via MC ES (500k episodes)
- `figures/fig_5_3_importance_sampling.png` — Fig. 5.3: ordinary vs weighted IS MSE on one blackjack state
- `figures/fig_5_4_ordinary_is_unstable.png` — Fig. 5.4: ordinary IS in the one-state example (10 runs, vectorized)

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
- `RacetrackEnv` (Exercise 5.8) lives in the shared `rl_envs` package;
  chapter code and tests import it as `from rl_envs.racetrack import RacetrackEnv`.
  Observation:
  `(row, col, vx, vy)`; 9 acceleration actions indexed 0–8 (`ACTIONS`);
  reward −1 per step; boundary hits teleport to a random start cell and
  zero the velocity; finish line ends the episode; `max_steps` optionally
  caps episode length (needed so control episodes terminate).
- Action spaces are resolved per environment: `_actions_for(env)` reads
  `getattr(env, "ACTIONS", blackjack_ACTIONS)`, so the same MC prediction /
  control / ES machinery runs on both blackjack and the racetrack.
- Policy functions are callables `policy_fn(state) -> probability vector`
  over the environment's action table; uniform exploration for blackjack is
  `[0.5, 0.5]`.
- Off-policy prediction is `every-visit` by default (`first_visit=False` for
  first-visit) — matching the book's every-visit off-policy algorithm.
- `BlackjackEnv` implements infinite-deck cards, natural-win handling, the
  fixed dealer rule (hit below 17), and usable-ace accounting.
- `mc_exploring_starts` implements the ES algorithm (first action uniform;
  the greedy policy is followed afterwards; ties split uniformly).
- Figure 5.3's reference value is estimated with a 200,000-episode
  first-visit MC prediction (not hardcoded); book reports ≈ −0.277.

## Quick Start
```bash
python -m pytest chapter_05/tests/ -v
python -m pytest
ruff check chapter_05/
python scripts/run_quality_gates.py 05
```

---

*Last updated: 2026-11-18*
