# Chapter 11 — Exploration

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition); repo Chapter 11
> covers exploration strategies, implemented against the bandit material of
> book Chapter 2 (Sections 2.3, 2.6, 2.7) and applied to one-step tabular
> Q-learning on a small MDP.

## Overview
Two exploration strategies from the bandit literature, plus their use in RL:
UCB (upper-confidence-bound) action selection, and optimism in the face of
uncertainty via optimistic initialization and a descending optimism bonus.

## Exercises and Solutions

- [exercises.md](exercises.md) — derived exercises from the book's §2.6–2.7 (optimistic
  initialization, UCB action selection / OFU), labeled [derived], with book §/PDF-page citations and
  coding-exercise pointers into this chapter (`ucb_bonus`, `ucb_select`, `TenArmedBandit`,
  `run_optimistic_greedy_bandit`, `q_learning_ucb`).

## Algorithms Implemented
- [x] algorithm_11.1 — UCB exploration strategy (`ucb_bonus`, `ucb_select`,
  `run_ucb_bandit`, `q_learning_ucb`) ✅
- [x] exercise_11.1 — Optimism in the face of uncertainty (`run_optimistic_greedy_bandit`,
  `q_learning_optimistic`, `descending_bonus`, `run_descending_bonus_bandit`,
  `q_learning_descending_bonus`) ✅

Also included: `TenArmedBandit` (the 10-armed testbed, book Sec 2.3),
`run_epsilon_greedy_bandit` (the ε-greedy baseline of Fig 2.4),
`make_small_gridworld` (local deterministic 2x4 gridworld on chapter_03's
`FiniteMDP`).

## Book Reference
- Section 2.3: The 10-armed testbed
- Section 2.4: Incremental sample-average implementation
- Section 2.6: Optimistic initial values (Fig 2.3)
- Section 2.7: Upper-confidence-bound action selection (eq. 2.8, Fig 2.4,
  "UCB1" in the literature)

## Implementation Notes
- **UCB (Alg 11.1)**: bonus `c * sqrt(ln t / N(s, a))` added to Q, t the
  number of selections so far, untried actions treated as maximizing
  (infinite bonus). c = 2 follows the book's Figure 2.4 default. In the MDP
  setting, UCB replaces the epsilon-greedy action selection inside tabular
  Q-learning (same one-step backups; counts maintained per (s, a)).
  Bandit runners use sample-average Q values, matching Figure 2.4.
- **Optimistic initialization (Ex 11.1)**: Q1(a) = +5 on the 10-armed testbed
  (the book's Fig 2.3 setting, q*(a) ~ N(0, 1)); the same trick is exposed
  for Q-learning on an MDP via `initial_q`. `tie_break='first'` breaks greedy
  ties by lowest action index. Tests verify the agent cycles through all
  actions before settling: all 10 arms are tried within 40 steps on every
  seed, versus ~1/20 for zero-init greedy; on the deterministic gridworld,
  first-index tie-breaking locks zero-init greedy onto the blocked 'up'
  action forever (3000 visits, goal never reached) while the optimistic
  agent balances visits across all four actions (~49-53 each after 10
  episodes).
- **Descending optimism bonus (Ex 11.1)**: "optimistic greedy in the face of
  uncertainty" via `b(N) = b0 / (1 + N)` added to Q(s, a). The bonus is
  maximal for untried actions and vanishes with visit count — a local,
  self-terminating analogue of optimistic initialization. It is *not* UCB
  (no log t term, no confidence-interval semantics); tests check monotonic
  descent, vanishing bonus, and that it forces broad early exploration.
- Fidelity caveats: the repo has no book chapter for "exploration"; book
  Chapter 2 is used as the reference. Figure 2.3/2.4 experiments are scaled
  down (fewer testbed tasks/steps, 50 seeds instead of 2000 tasks) so the
  suite runs in seconds; thresholds are set with seeded runs.

## Quick Start
```bash
python -m pytest chapter_11/tests/ -v
python -m pytest
ruff check chapter_11/
```

---

*Last updated: 2026-11-14*
