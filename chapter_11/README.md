# Chapter 11 — Off-policy Methods with Approximation

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 11

## Overview
Off-policy prediction with function approximation: the deadly triad (function
approximation + bootstrapping + off-policy), the semi-gradient instability,
and gradient-corrected objectives (MSPBE) on an exactly-enumerable counterexample.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 11.1–11.5 derived from Sections
  11.1–11.4 with derivations and coding pointers (`SemiGradientOffPolicyAgent`,
  `behavior_stationary`, `projected_bellman_solution`, `mspbe`,
  `mspbe_gradient_descent`).

## Algorithms Implemented
- [x] Semi-gradient off-policy TD(0) with importance sampling (`SemiGradientOffPolicyAgent`) ✅
- [x] Baird-style asymmetric hub counterexample (MDP, exact eigen-solve) (`baird_*`) ✅
- [x] Behavior stationary distribution (exact) (`behavior_stationary`) ✅
- [x] Projected Bellman solution (exact linear solve) (`projected_bellman_solution`) ✅
- [x] MSPBE value and gradient descent (`mspbe`, `mspbe_gradient_descent`) ✅

## Book Reference
- Section 11.1/11.2: The deadly triad; semi-gradient methods off-policy
- Example 11.1: Baird's counterexample (implemented as an exactly-verifiable
  hub-star variant; verbatim book example not verifiable from the local PDF summary)
- Section 11.3: The projected Bellman equation
- Section 11.4: Gradient-correction methods / MSPBE

## Not implemented (with justification)
- **TDC/GTD recursions verbatim** — descoped; instead the chapter implements the
  exact MSPBE gradient (the objective those methods minimize), which is fully
  verifiable. Half-remembered recursions would violate this repo's
  "never fabricated" fidelity rule.
- **Emphatic TD (ETD)** — descoped; documented in Exercise 11.5 with a clear
  justification and a pointer to the section.
- **Nonlinear (neural network) counterexample** — descoped with Chapter 9's
  justification.

## Figures
- `figures/example_11_1_divergence.png` — semi-gradient off-policy TD, ||w|| on log scale,
  diverging (illustrative demonstration).
- `figures/example_11_2_mspbe.png` — MSPBE gradient descent: monotone (conditioning-limited)
  decrease of MSPBE and ||w|| (illustrative demonstration).

## Implementation Notes
- The hub star is intentionally asymmetric under action B (outer → outer_1) so the
  behavior distribution concentrates off-center and the expected semi-gradient map
  is indefinite; this is computed exactly (lstsq eigen-solve), not sampled.
- All functions operate on NumPy arrays; import as a package:
  `from chapter_11.solution import ...`.
- Exact book figure identity cannot be verified against the local PDF summary;
  figures are labelled illustrative.

## Quick Start
```bash
python -m pytest chapter_11/tests/ -v
python -m pytest
ruff check chapter_11/
python scripts/run_quality_gates.py 11
```

---

*Last updated: 2026-11-18*
