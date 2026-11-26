# Chapter 8 — Planning and Learning with Tabular Methods

> Reference implementation of Sutton & Barto,
> _Reinforcement Learning: An Introduction_ (2nd edition), Chapter 8.
> Figure numbers below were verified against the local book PDF.

## Overview

Model-based RL: models and planning (§8.1), the Tabular Dyna-Q box (§8.2),
Dyna-Q+ and changing environments (§8.3), prioritized sweeping (§8.4),
expected vs sample updates (§8.5), trajectory sampling (§8.6), RTDP (§8.7),
and decision-time planning / heuristic search (§8.8–§8.9).

## Implemented APIs (`solution.py` unless noted)

| Book item | API | Notes |
|---|---|---|
| §8.1 models | `TabularModel` | distribution model from per-outcome counts; sample model; incremental predecessor index |
| §8.2 Tabular Dyna-Q | `dyna_q` | direct Q-learning + model learning + `n_planning` sample backups |
| §8.3 Dyna-Q+ | `dyna_q_plus` | `r + κ√τ` on *modelled* (stale) pairs; untried pairs use the zero-reward self-loop default model without a bonus |
| §8.3 blocking/shortcut | `ChangingGridworld`, `changing_maze_experiment` | barrier swap mid-run; agents retain Q/model; variants `dyna_q`, `dyna_q_plus`, `selection_bonus` |
| Ex. 8.4 selection-only bonus | `_epsilon_greedy_action_with_bonus` | `argmax_a [Q + κ√τ]` with unchanged updates |
| §8.4 prioritized sweeping | `PriorityQueue`, `prioritized_sweeping` | expected-update extension for stochastic models |
| §8.4 Example 8.4 | `scaling_maze_experiment`, `plot_example_8_4_scaling` | scaled Dyna mazes; "solved" = shortest greedy path from S; runs without a solution are reported as `None` |
| §8.4 Example 8.5 | `chapter_08/rod.py` | 20×20 rod MDP (36 orientations, 4 actions); prioritized sweeping with a known model vs exhaustive sweeps |
| §8.5 expected updates | `expected_q_target`, `expected_sweep`, `expected_dyna_q`, `compare_sample_vs_expected_dyna` | eq. (8.1) |
| §8.6 trajectory sampling | `trajectory_sampling_experiment` | on-policy vs uniform backups on random tasks |
| §8.7 Example 8.6 | `chapter_08/rtdp.py` | RTDP vs Gauss–Seidel value iteration on the shared racetrack |
| §8.8 heuristic search | `plot_figure_8_10_heuristic_search` | one-step backup sequence toward the root |

## Figures (`figures/`)

Regenerate with:

```bash
python chapter_08/figures/generate_figures.py
```

Verified numbered book figures: `fig_8_3_dyna_q.png`,
`fig_8_4_policies.png`, `fig_8_7_backup_diagrams.png`,
`fig_8_8_update_efficiency.png`, `fig_8_9_trajectory_sampling.png`,
`fig_8_10_heuristic_search.png`.

The book's Figs. 8.5/8.6 (blocking/shortcut mazes) are **not** reproduced:
our small reconstructed maze reverses the phase-1 ordering, so those plots are
honestly named `example_8_2_blocking_maze.png` and
`example_8_3_shortcut_maze.png`. Likewise, the book prints no numbered figure
for Examples 8.4–8.6; their outputs are `example_8_4_scaling.png`,
`example_8_5_rod.png` and `example_8_6_rtdp_racetrack.png`.

All plots are reduced-scale reproductions or **illustrative demonstrations**,
not pixel-identical copies. In particular:

- The exact cell coordinates of the blocking/shortcut mazes and the rod
  obstacles are not printed in the book; the layouts here are reconstructions.
- The trajectory-sampling reward model is not printed; we use `r(s,a) ~ U[0,1]`
  with action-independent termination, which reproduces the book's qualitative
  pattern (on-policy faster initially, relatively slower in the long run).
- Our κ (1e-4) and maze scale differ from the book's tuning. On this small
  maze the bonus *slows* initial learning (planning on never-tried pairs
  dilutes goal-reward propagation), so Dyna-Q+ is worse in phase 1 — the
  opposite of the book. These plots are demonstrations only, never cited as
  Fig. 8.5/8.6.
- Example 8.6 uses a small illustrative racetrack (707 reachable states) with
  `noise_prob=0`, versus the book's 9,115-state track.

## Exercises

`exercises.md` covers book exercises 8.1–8.4 with student-friendly explanations
and coding pointers into the APIs above.

## Verification

```bash
python -m pytest chapter_08/tests/ -q   # 54 tests
python -m pytest -q                      # full suite from repo root
```

---

*Last updated: 2026-11-26*
