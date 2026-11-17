# Chapter 12 — Eligibility Traces

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 12
> (Eligibility Traces).

## Overview
Backward-view eligibility-trace methods: state-value TD(λ) prediction with
accumulating/replacing traces, Sarsa(λ) control with replacing traces and
with clearing traces, and True online TD(λ) (the exact online λ-return
algorithm). λ = 0 recovers one-step TD; λ = 1 recovers Monte-Carlo-like
behavior.

## Exercises and Solutions

- [exercises.md](exercises.md) — book Chapter 12 (Eligibility Traces) exercises 12.1–12.16 with from-scratch
  solutions, tagged [traces]/[Q(lambda)], book §/equation/PDF-page citations, and coding-exercise
  pointers into this chapter (`td_lambda_prediction`, `sarsa_lambda_replacing`, `true_online_td_lambda`,
  `watkins_q_lambda`, `watkins_lambda_return`).

## Algorithms Implemented
- [x] algorithm_12.1 — TD(λ) with replacing traces (`td_lambda_prediction(trace="replacing")`, `sarsa_lambda_replacing`) ✅
- [x] algorithm_12.2 — TD(λ) with clearing traces (`sarsa_lambda_clearing`) ✅
- [x] algorithm_12.3 — True online TD(λ) (`true_online_td_lambda`) ✅
- [x] algorithm_12.4 — Watkins's Q(λ) (`watkins_q_lambda`, `watkins_lambda_return`) ✅

Also included: `td_lambda_prediction` with `trace="accumulating"`
(Section 12.2, box on p. 240).

## Book Reference
- Section 12.2 / box p. 240: semi-gradient TD(λ); accumulating traces Eq. 12.5
- Eq. 12.17: replacing traces (tabular special case); deprecated for function
  approximation in favor of dutch traces
- Section 12.5 (Eq. 12.15-12.16, box p. 245): True online TD(λ) — exactly
  implements the on-line λ-return algorithm (Section 12.4); van Seijen et al.,
  2016
- Section 12.7 (box p. 250, Eq. 12.22-12.23): Sarsa(λ), replacing traces
- Section 12.7, Figure 12.11: Sarsa(λ) with clearing of non-selected actions'
  traces

## Implementation Notes
- Tabular implementations on the Example 6.2 random-walk environment
  (`RandomWalkEnv`, imported from `rl_envs.random_walk`).
- Clearing (algorithm_12.2) means: on each step, after the replacing-trace
  bump of the chosen action to 1, the traces of the *non-selected actions of
  the visited state* are set to 0; traces of unvisited states are untouched.
  It has no state-value analog, so it is implemented only for Sarsa(λ).
- True online TD(λ) uses one-hot features (x = e_S), so w is exactly the
  table of state values and z·x reduces to z[S]; V_old is the scalar
  bookkeeping variable of the book's box. λ = 0 reduces exactly to TD(0).
- Import as a package: `from chapter_12.solution import ...`.
- Validation against chapter_06: at λ = 0 the replacing/clearing Sarsa(λ)
  and the replacing/accumulating TD(λ) all reduce to their one-step
  counterparts (checked by tests); at λ → 1 learning approaches
  Monte-Carlo behavior.

## Quick Start
```bash
python -m pytest chapter_12/tests/ -v
python -m pytest
ruff check chapter_12/
```

---

*Last updated: 2026-11-14*
