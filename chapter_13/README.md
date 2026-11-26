# Chapter 13 — Policy Gradient Methods

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 13

## Overview
Policy gradient methods: learn a parameterized policy pi(a|s, theta) directly
rather than deriving it from a value function. This chapter implements the
policy gradient theorem (numeric check), REINFORCE, REINFORCE with baseline,
one-step actor-critic, the average-reward (continuing) variant, and a Gaussian
policy for continuous actions.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 13.1–13.3 with from-scratch solutions (derivation of p* = 2 − √2 ≈ 0.586 for Example 13.1), book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`PolicyGradientAlgorithm`, `PolicyGradientWithBaseline`, `ActorCriticAlgorithm`, `GaussianPolicy`, `ShortCorridorEnv`).
- Figures: `figures/fig_13_1_short_corridor_value.png` (closed-form V1(p) curve + MC cross-checks, p* marker),
  `figures/fig_13_2_reinforce_short_corridor.png` (REINFORCE learning curve + exact value-ascent).
  See `figures/generate_figures.py` to regenerate.

## Algorithms Implemented
- [x] algorithm_13.1 — Policy gradient theorem: numeric verification (`check_policy_gradient_theorem`) ✅
- [x] algorithm_13.2 — REINFORCE: Monte Carlo policy gradient (`PolicyGradientAlgorithm`) ✅
- [x] algorithm_13.3 — REINFORCE with (learned state-value) baseline (`PolicyGradientWithBaseline`) ✅
- [x] algorithm_13.4 — Actor-critic methods, one-step episodic (`ActorCriticAlgorithm`, `ActorCriticAgent`) ✅
- [x] algorithm_13.5 — Policy gradient for continuing problems (`ContinuingActorCritic`, average-reward delta) ✅
- [x] algorithm_13.6 — Policy parameterization for continuous actions (`GaussianPolicy`, tabular mean + learnable log-sigma) ✅
- [x] Example 13.1 — Short corridor MDP (`ShortCorridorEnv`, `ShortCorridorREINFORCE`,
  `run_short_corridor_reinforce`, `run_short_corridor_value_ascent`) with closed-form
  V1(p) = 2(p-2)/(p(1-p)) maximized at p* = 2 - √2 ≈ 0.586; REINFORCE and exact
  value-ascent, plus Monte-Carlo cross-checks. Illustrative demonstration; figure
  identities cannot be verified against the local PDF summary. ✅

## Book Reference
- Section 13.2: The Policy Gradient Theorem (episodic and continuing forms)
- Section 13.3: REINFORCE: Monte Carlo Policy Gradient
- Section 13.4: REINFORCE with Baseline
- Section 13.5: Actor-Critic Methods
- Section 13.6: Policy Gradient for Continuing Problems
- Section 13.7: Policy Parameterization for Continuous Actions
- Section 13.1: Policy Approximation and its Advantages (Example 13.1 — short corridor)

## Implementation Notes
- Policies are tabular softmax over action preferences (the book's running
  formulation); test-compatible sklearn-style `fit`/`predict` interface with
  `learning_rate`/`gamma` attributes.
- `run_policy_gradient_simulation(algorithm, env, ...)` trains on a bandit
  `PolicyEnvironment` and returns per-episode total rewards.
- `main(seed=...)` runs a REINFORCE-with-baseline demo.
- **Example 13.1 (short corridor):** `ShortCorridorEnv` implements the 3-state
  switched-action MDP; episodes are capped by `max_steps` (5000) to bound the
  self-loop at state 1. `short_corridor_value_gradient` gives the analytic
  dV1/dp (numerically stable from p in (0,1)); `run_short_corridor_value_ascent`
  converges deterministically to p*, while `run_short_corridor_reinforce`
  demonstrates empirical REINFORCE (high-variance with all-negative rewards).
  Figures 13.1 (REINFORCE on the short-corridor gridworld) and 13.2 (REINFORCE
  with vs without a baseline) were verified against the local 2nd-edition PDF.
  The plots are reduced-scale illustrative reproductions, not pixel-level copies.
- Import as a package (`from chapter_13.solution import ...`); see
  `chapter_13/__init__.py` for the package marker.

## Quick Start
```bash
python -m pytest chapter_13/tests/ -v
python -m pytest
ruff check chapter_13/
python scripts/run_quality_gates.py 13
```

---

*Last updated: 2026-11-12*
