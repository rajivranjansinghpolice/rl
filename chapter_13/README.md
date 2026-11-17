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

- [exercises.md](exercises.md) — Exercises 13.1–13.3 with from-scratch solutions (derivation of p* = 2 − √2 ≈ 0.586 for Example 13.1), book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`PolicyGradientAlgorithm`, `PolicyGradientWithBaseline`, `ActorCriticAlgorithm`, `GaussianPolicy`).

## Algorithms Implemented
- [x] algorithm_13.1 — Policy gradient theorem: numeric verification (`check_policy_gradient_theorem`) ✅
- [x] algorithm_13.2 — REINFORCE: Monte Carlo policy gradient (`PolicyGradientAlgorithm`) ✅
- [x] algorithm_13.3 — REINFORCE with (learned state-value) baseline (`PolicyGradientWithBaseline`) ✅
- [x] algorithm_13.4 — Actor-critic methods, one-step episodic (`ActorCriticAlgorithm`, `ActorCriticAgent`) ✅
- [x] algorithm_13.5 — Policy gradient for continuing problems (`ContinuingActorCritic`, average-reward delta) ✅
- [x] algorithm_13.6 — Policy parameterization for continuous actions (`GaussianPolicy`, tabular mean + learnable log-sigma) ✅

## Book Reference
- Section 13.2: The Policy Gradient Theorem (episodic and continuing forms)
- Section 13.3: REINFORCE: Monte Carlo Policy Gradient
- Section 13.4: REINFORCE with Baseline
- Section 13.5: Actor-Critic Methods
- Section 13.6: Policy Gradient for Continuing Problems
- Section 13.7: Policy Parameterization for Continuous Actions

## Implementation Notes
- Policies are tabular softmax over action preferences (the book's running
  formulation); test-compatible sklearn-style `fit`/`predict` interface with
  `learning_rate`/`gamma` attributes.
- `run_policy_gradient_simulation(algorithm, env, ...)` trains on a bandit
  `PolicyEnvironment` and returns per-episode total rewards.
- `main(seed=...)` runs a REINFORCE-with-baseline demo.
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
