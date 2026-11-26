# Chapter 10 — On-policy Control with Approximation

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 10

## Overview
On-policy control with function approximation: episodic semi-gradient Sarsa on
the Mountain Car task (tile coding) and average-reward differential
semi-gradient Sarsa on the continuing Access-Control Queuing problem.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 10.1–10.5 derived from Sections
  10.1–10.3 with solutions, derivations, and coding pointers
  (`EpisodicSemiGradientSarsaAgent`, `mountain_car_features`, `run_mountain_car`,
  `DifferentialSarsaAgent`, `AccessControlQueue`, `run_access_control`).

## Algorithms Implemented
- [x] Episodic semi-gradient Sarsa, linear, tile coding (`EpisodicSemiGradientSarsaAgent`) ✅
- [x] Mountain Car training loop (`run_mountain_car`) ✅
- [x] Average-reward differential semi-gradient Sarsa (`DifferentialSarsaAgent`) ✅
- [x] Access-Control Queuing environment and training (`AccessControlQueue`,
       `run_access_control`) ✅

## Book Reference
- Section 10.1: Episodic semi-gradient control (Example 10.1 — Mountain Car)
- Sections 10.2/10.3: The average-reward setting (Example 10.2 — Access-Control
  Queuing)
- Section 9.4.3: Tile coding (shared; implemented in `chapter_09.solution.TileCoder`)

## Not implemented (with justification)
- **Continuing (non-episodic) n-step differential Sarsa beyond 1 step** — the
  differential target with n-step returns is the same idea unrolled; the 1-step
  version here is enough to demonstrate the setting and keeps the code exact.
- **Nonlinear approximation (neural networks) for Mountain Car** — descoped with
  the same justification as Chapter 9.

## Figures
- `figures/example_10_2_mountain_car.png` — Mountain Car learning curve
  (illustrative; Sec. 10.1).
- `figures/example_10_1_access_control.png` — average reward per step vs time on
  access-control queuing (illustrative; Sec. 10.2/10.3, Example 10.2).

## Implementation Notes
- Mountain Car dynamics live in `rl_envs.mountain_car.MountainCarEnv` (book
  dynamics, Eq. in Sec. 10.1). Tile coding is imported from Chapter 9 (DRY).
- `mountain_car_features` produces the 512-feature sparse vector for
  (position, velocity) in 8 tilings.
- `run_mountain_car` uses epsilon-greedy exploration (ε = 0.1) — a purely greedy
  policy from zero weights never escapes the valley (see Exercise 10.5).
- Import as a package: `from chapter_10.solution import ...`.

## Quick Start
```bash
python -m pytest chapter_10/tests/ -v
python -m pytest
ruff check chapter_10/
python scripts/run_quality_gates.py 10
```

---

*Last updated: 2026-11-18*
