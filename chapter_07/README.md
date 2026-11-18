# Chapter 7 — n-step Bootstrapping

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 7

## Overview
Generalizes one-step TD and Monte Carlo into a spectrum of n-step methods:
n-step TD prediction (Example 7.1 / Figure 7.2), on-policy n-step Sarsa control,
the tree-backup algorithm (off-policy without importance sampling), and
SARSA(lambda) with eligibility traces.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 7.1–7.8 with paraphrased statements, §/Eq.
  citations, derivations/pseudocode, and repo-accurate coding pointers. Off-policy
  importance-sampling / per-decision exercises are marked theory-only where no API exists.

## Algorithms Implemented
- [x] n-step TD prediction (`n_step_td_prediction`, Example 7.1) ✅
- [x] n-step TD prediction experiment / Figure 7.2 (`run_n_step_prediction_experiment`) ✅
- [x] n-step Sarsa control (`n_step_sarsa`) ✅
- [x] n-step Tree Backup (`tree_backup`) ✅
- [x] SARSA(lambda) control with accumulating traces (`sarsa_lambda`) ✅

> Note: no off-policy importance-sampling / per-decision n-step API is implemented
> in this chapter.

## Figures

- `figures/fig_7_2_n_step_td_alpha.png` — Performance of n-step TD methods as a function
  of α, for various values of n, on a 19-state random walk task (Example 7.1 / Figure 7.2).
  Reproduction: 19 states (A..S), left outcome −1, right +1, all values initialized to 0,
  equiprobable policy; error = RMS vs. the true values at the end of each of the first
  10 episodes, averaged over 10 episodes × 100 repetitions (the same walk sets are used
  for all settings). α grid [0, 1] in steps of 0.1 is the repo's choice (the book's printed
  x-axis range is not legible from the draft images). Intermediate n (2–16) wins the
  minima; large n degrades at large α; all curves coincide at α = 0. Deterministic (seed 0).
  Regenerate with `python chapter_07/figures/generate_figures.py`. Illustrative
  reproduction (book plot is at full scale).

## Book Reference
- Section 7.1: n-step TD prediction; Example 7.1 / Figure 7.2 (19-state random walk)
- Section 7.2: n-step Sarsa (box titled "State-action n-step Sarsa", Eq. 7.4-7.5)
- Section 7.5: n-step Tree Backup (box titled "State-action n-step Tree Backup")
- Section 12.2: SARSA(lambda) traces (adapting Sarsa-lambda control)

## Implementation Notes
- Reuses the shared `RandomWalkEnv` (`rl_envs/random_walk.py`, Example 6.2's 5-state walk);
  the 19-state Example 7.1 walk is the same class parameterized with
  `n_states=19, left_reward=-1.0, right_reward=1.0` (start state 'J'), with
  `env.true_values()` returning the linear terminal-interpolated values
  (−0.9 … +0.9).
- The n-step TD prediction box is implemented as `_n_step_td_sweep` (the update for one
  episode) and reused by the per-walk trainer and the Figure 7.2 experiment, which
  pre-samples shared walk sets (`_sample_walks`) so all (n, α) settings see identical
  episodes, as in the book. The α = 0 point is included for completeness
  (`n_step_td_prediction` itself requires α > 0).
- `n_step_sarsa` updates every time tau = t - n + 1 with the n-step return;
  `tree_backup` implements the book's exact backup with the target as
  the epsilon-greedy policy; both target q* semantics like the epsilon-greedy
  on-policy boxes.
- `sarsa_lambda` uses accumulating eligibility traces (step size alpha,
  trace decay gamma * lambda).

## Quick Start
```bash
python -m pytest chapter_07/tests/ -v
python -m pytest
python chapter_07/figures/generate_figures.py
ruff check chapter_07/
python scripts/run_quality_gates.py 07
```

---

*Last updated: 2026-11-18*
