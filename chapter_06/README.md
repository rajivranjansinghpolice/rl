# Chapter 6 — Temporal-Difference Learning

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 6

## Overview
Temporal-difference (TD) methods: TD prediction, on-policy control (Sarsa),
off-policy control (Q-learning), Expected Sarsa, Double Q-learning, TD(lambda)
with accumulating traces — and the classic random-walk comparison between TD(0)
and constant-alpha MC (Example 6.2), the batch-updating experiment (Example 6.3 /
Figure 6.2), Sarsa on the windy gridworld (Example 6.5 / Figure 6.3), Sarsa vs
Q-learning on cliff walking (Example 6.6 / Figure 6.4), and Double Q-learning on
the maximization-bias MDP (Example 6.7 / Figure 6.7).

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 6.1–6.12 with from-scratch solutions, book §/equation citations, and coding-exercise pointers into this chapter and `rl_envs` (`td0_prediction`, `td_lambda_prediction`, `sarsa`, `q_learning`, `sarsa_control`, `q_learning_control`, `expected_sarsa_control`, `double_q_learning`, `maximization_bias_experiment`, `run_batch_updating`, `RandomWalkEnv`, `CliffWalkingEnv`, `MaximizationBiasMDP`, `WindyGridWorld`).

## Algorithms Implemented
Book algorithm boxes are referred to by name and section (the printed box
numbers differ between the 2nd edition and the in-progress draft), not by an
invented index. Implemented algorithms:
- [x] TD(0) prediction — §6.1 (`td0_prediction`, box named "Tabular TD(0)") ✅
- [x] Constant-α MC prediction — §6.1/§6.3 (comparison baseline) (`mc_prediction_incremental`, `run_random_walk_comparison`) ✅
- [x] Sarsa (on-policy TD control) — §6.4 (`sarsa` for `RandomWalkEnv`; `sarsa_control` generic) ✅
- [x] Q-learning (off-policy TD control) — §6.5 (`q_learning` for `RandomWalkEnv`; `q_learning_control` generic) ✅
- [x] Expected Sarsa — §6.6 (`expected_sarsa_control`) ✅
- [x] Double Q-learning — §6.7 (`double_q_learning`, `maximization_bias_experiment`) ✅
- [x] TD(lambda) prediction — §12.1-style, accumulating traces (`td_lambda_prediction`) ✅
- [x] Batch update (Example 6.3): batch TD(0) and batch constant-α MC, solved exactly (`batch_td_fixed_point`, `batch_mc_values`, `run_batch_updating`) ✅
- [x] Example 6.2 random walk — mechanics and true values (`RandomWalkEnv`, `random_walk_true_values`) ✅
- [x] Example 6.5 / Exercises 6.9–6.10 windy gridworld — Sarsa with standard, king's-moves (9 actions) and stochastic-wind variants (`WindyGridWorld(king_moves=..., stochastic_wind=...)` in `rl_envs`) ✅
- [x] Example 6.6 cliff walking — Sarsa vs Q-learning vs Expected Sarsa (`CliffWalkingEnv`) ✅
- [x] Example 6.7 maximization-bias MDP — Q-learning vs Double Q-learning (`MaximizationBiasMDP`) ✅

Also included: `_epsilon_greedy_probs` (shared ε-greedy helper) and
`plot_figure_6_*` / `plot_example_6_2_*` figure builders.

## Figures
All figures are regenerable via `python chapter_06/figures/generate_figures.py`
(run from the repository root). Figure identities refer to the 2nd (2018)
edition; the draft and the 1st edition number figures differently. All are
illustrative demonstrations, not pixel-identical redraws:
- `figures/example_6_2_random_walk_td_mc.png` — Example 6.2 (2nd ed.): TD(0) vs constant-α MC
  RMS error vs episodes (unnumbered result panel in the 2nd ed.; was Figs. 6.6–6.7 in the 1st ed.)
- `figures/fig_6_2_batch_updating.png` — **Fig. 6.2** (2nd ed.), Example 6.3: RMS error vs walks
  seen under batch updating, batch TD(0) vs batch MC
- `figures/fig_6_3_windy_sarsa.png` — **Fig. 6.3**, Example 6.5: Sarsa steps per episode on the
  windy gridworld (ε-greedy ε=0.1, α=0.5; raw and 100-episode moving average)
- `figures/fig_6_4_cliff_walking.png` — **Fig. 6.4**, Example 6.6: sum of rewards per episode,
  Sarsa vs Q-learning, single run ε-greedy ε=0.1, smoothed over 10 successive episodes
- `figures/fig_6_6_td_control_performance.png` — **Fig. 6.6**: interim (first 100 episodes) and
  asymptotic (last 1,000 of 5,000) average return vs α for Sarsa / Q-learning / Expected Sarsa on
  cliff walking, ε=0.1 — adapted from van Seijen et al. (2009); heavily reduced runs (5) vs the
  book's 50,000/10, labeled illustrative
- `figures/fig_6_7_double_q_learning.png` — **Fig. 6.7**, Example 6.7: % of runs taking the
  suboptimal left action from state A vs episode, Q-learning vs Double Q-learning (500 runs vs the
  book's 10,000, labeled illustrative)
- `figures/fig_6_9_king_moves.png` — **Exercise 6.9** (no numbered book figure): windy gridworld
  with king's moves vs the 4-action baseline, Sarsa steps per episode
- `figures/fig_6_10_stochastic_wind.png` — **Exercise 6.10** (no numbered book figure): windy
  gridworld with stochastic wind (equal thirds over wind−1/wind/wind+1, king's moves) vs
  deterministic king's moves, Sarsa steps per episode

## Book Reference
- Section 6.1: TD prediction — TD(0) box; constant-α MC (Eq. 6.1/6.2)
- Section 6.2: Advantages of TD; Example 6.2 — random-walk MRP comparison
- Section 6.3: Optimality of TD(0) (certainty-equivalence); Example 6.3 batch updating (Figure 6.2)
- Section 6.4: Sarsa (on-policy TD control); Example 6.5 — windy gridworld (Figure 6.3);
  Exercises 6.9 (king's moves) and 6.10 (stochastic wind)
- Section 6.5: Q-learning (off-policy TD control); Example 6.6 — cliff walking (Figure 6.4)
- Section 6.6: Expected Sarsa (Figures 6.5–6.6; Fig. 6.6 adapted from van Seijen et al. 2009)
- Section 6.7: Maximization bias and Double Learning; Example 6.7 — Double Q-learning (Figure 6.7)
- Section 12.1 / Chapter 7: TD(lambda) traces (implemented in §12.1 backward-view form)

## Implementation Notes
- `RandomWalkEnv` implements Example 6.2 exactly: states A-E, start at C,
  +1 at the right terminal, undiscounted, v*(s) = (position+1)/6.
- `run_random_walk_comparison` reproduces the Example 6.2 result panel with both methods
  initialized to V(s) = 0.5 and sharing the step size `alpha`: TD(0) learns outperforms
  constant-α MC (verified by test).
- `batch_td_fixed_point` solves the converged batch TD(0) answer exactly
  (the 5×5 fixed-point linear system; under batch updating this is deterministic and
  independent of α for small α). `batch_mc_values` computes the batch constant-α MC
  answer (sample averages of first-visit returns). `run_batch_updating` reproduces
  Figure 6.2 (book: averaged over 100 repetitions; TD < MC consistently).
- The generic control family (`sarsa_control` / `q_learning_control` /
  `expected_sarsa_control`) works on any tabular indexed-action env (`ACTIONS` or an
  `actions_for(state)` method, e.g. `MaximizationBiasMDP`), Q keyed
  `(state, action_index)`, with a `max_steps` truncation cap for non-terminating
  gridworld episodes; it returns `(q_table, steps_per_episode, returns_per_episode)`.
- `WindyGridWorld` (`rl_envs/windy_gridworld.py`): standard 7×10 grid, wind strengths
  `[0,0,0,1,1,1,2,2,1,0]` (departure column, pushes upward); `king_moves=True` adds 8
  directions, `allow_stay=True` adds a no-op (Exercise 6.9), `stochastic_wind=True`
  draws the effective wind uniformly from `{w-1, w, w+1}` when `w >= 1`
  (Exercise 6.10, "a third of the time", king's moves included).
- `CliffWalkingEnv` (Example 6.6): 4×12, start (3,0), goal (3,11), cliff row 3 cols 1–10,
  reward −100 and sent back to start on a cliff cell, −1 elsewhere, deterministic.
- `MaximizationBiasMDP` (Example 6.7): A → left/right; right → terminal r=0; left → B r=0;
  B has `n_b_actions` (10) actions, each → terminal with reward ~ N(−0.1, 1). The optimal
  action from A is 'right' (left's value is −0.1), but maximization bias favors 'left'.
- `expected_sarsa_control` implements Eq. (6.9): bootstrap with
  `sum_a pi(a|s') * Q(s', a)` under the ε-greedy policy (no sampling variance).
- `double_q_learning` implements Eq. (6.10): fair coin picks which table updates; the
  other table evaluates the greedy action; behavior is ε-greedy over Q1 + Q2.
  `maximization_bias_experiment` reproduces Figure 6.7 (book: 10,000 runs; we use 500).
- Import as a package: `from chapter_06.solution import ...`.

## Quick Start
```bash
python -m pytest chapter_06/tests/ -v
python -m pytest
ruff check chapter_06/
python scripts/run_quality_gates.py 06
```

---

*Last updated: 2026-11-18*
