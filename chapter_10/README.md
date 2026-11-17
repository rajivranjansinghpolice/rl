# Chapter 10 — Q(λ)

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition); directory
> `chapter_10` follows this repository's TODO.md numbering.

## Overview
Watkins's Q(λ) (algorithm_10.1): the Q-learning (off-policy, max-bootstrap)
control algorithm extended with eligibility traces, with the traces cut to
zero after the first non-greedy action. Implemented backward view (the
practical algorithm) plus the forward-view trace-cut λ-return definition.

## Book Chapter 12 — Exercises and Solutions

Book Chapter 12 has one reader-eligible exercise set. All 16 exercises are
answered in [`../chapter_08/exercises.md`](../chapter_08/exercises.md),
which also maps this implementation (`watkins_q_lambda`, `watkins_lambda_return`)
to the relevant exercises.

## Algorithms Implemented
- [x] algorithm_10.1 — Watkins's Q(λ) (`watkins_q_lambda`) ✅

Also included: `watkins_lambda_return` — the forward-view trace-cut
λ-return used by the conceptual algorithm (Section 12.10, Figure 12.12).

## Book Reference
- Section 12.10 (2nd ed., p. 257): "Watkins's Q(λ) to Tree-Backup(λ)"
  — Figure 12.12: the series of component updates ends either with the end
  of the episode or with the first non-greedy action, whichever comes first
- Q-learning (Algorithm 6.2, Section 6.5): the max-bootstrap target
- TODO.md lists this as "book Section 12.6"; in the 2nd edition Section 12.6
  is "Dutch Traces in Monte Carlo Learning" — the Watkins's Q(λ) material
  lives in Section 12.10

## Implementation Notes
- `watkins_q_lambda` (backward view): epsilon-greedy behavior; Q-learning
  target `R + gamma * max_a Q(S', a)`; traces are **replacing** by default
  (z(S,A) ← 1, keeps z bounded; `trace="accumulating"` also available).
- Trace-cutting: after the Q-update at a step whose behavior action was
  **non-greedy** w.r.t. Q(s, ·) at that state (not in the argmax set;
  ties count as greedy, matching the ε-greedy tie-breaking), ALL traces
  are set to zero. The non-greedy step's TD error still propagates through
  pre-existing traces at that same step, but no TD error can propagate
  backward across a non-greedy action. Greediness is evaluated with the
  Q-values from before the update at the current step.
- Stability caveat (fidelity note): Watkins's Q(λ) is semi-gradient
  off-policy control with a max bootstrap; with `gamma=1.0` it diverges in
  practice (observed: Q blows up after ~2000–2500 episodes at α=0.1 on the
  random walk). The default is therefore `gamma=0.99`, which keeps the
  backups contracting. Pass `gamma=1.0` only with small `lam`/`alpha`.
  The book motivates Tree-Backup(λ) (Section 12.10) as the stable
  successor; it is intentionally not implemented here (out of scope).
- Forward view: `watkins_lambda_return` computes, per time t,
  G^λ_t = (1−λ) Σ_{n=1..n_max} λ^{n−1} G_{t:t+n} + λ^{n_max} G_{t:t+n_max}
  with the horizon n_max cut at the first non-greedy action; λ = 0 reduces
  to the one-step Q-learning return.
- Tabular, on the Example 6.2 random-walk environment (`RandomWalkEnv`
  from `chapter_06.solution`). λ = 0 reduces exactly to one-step Q-learning.
- Import as a package: `from chapter_10.solution import ...`.

## Quick Start
```bash
python -m pytest chapter_10/tests/ -v
python -m pytest
ruff check chapter_10/
python scripts/run_quality_gates.py 10
```

---

*Last updated: 2026-11-14*
