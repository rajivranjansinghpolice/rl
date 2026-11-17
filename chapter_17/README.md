# Chapter 17 — Frontiers

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 17 (Frontiers)

## Overview
Chapter 17 surveys open problems: general value functions (17.1),
**temporal abstraction via options** (17.2), observations and state (17.3),
**designing reward signals** (17.4, including intrinsic motivation),
remaining issues (17.5) and RL and the future of AI (17.6). No boxed
algorithms are given. This chapter implements small, honest demonstrations
of the two sections with executable content:

1. **Algorithm 17.1** — **Options / hierarchical RL** in a 7x7 four-room
   GridWorld (Section 17.2): temporally extended options with learned
   option values (SMDP Q-learning), plus a primitive-only baseline.
2. **Algorithm 17.2** — **Count-based intrinsic motivation** (Section 17.4):
   a novelty bonus R+ = R + beta / sqrt(N(s) + 1) added to Q-learning.

## Exercises and Solutions

- [exercises.md](exercises.md) — derived exercises from the frontier sections (§17.1 GVFs, §17.2 options/SMDP
  Q-learning, §17.3 partial observability, §17.4 reward design/intrinsic motivation, §17.5–17.6 open
  problems), labeled [derived], with book §/PDF-page citations and coding-exercise pointers into this
  chapter (`Option`, `smdp_q_learning`, `default_options`, `GridWorld`).

## Algorithms Implemented
- [x] algorithm_17.1 — Options / hierarchical RL demo
  (`GridWorld`, `Option`, `make_subgoal_option`, `default_options`,
  `run_option`, `smdp_q_learning`, `rollout_smdp`, `primitive_q_learning`,
  `rollout_primitive`)
- [x] algorithm_17.2 — Intrinsic motivation (count-based novelty bonus)
  (`novelty_bonus`, `q_learning_novelty`)

## Book Reference
- Section 17.2: Temporal abstraction via options —
  omega = <pi_omega, gamma_omega>, options extend the action space,
  option-value functions, SMDP planning/learning. The four-room GridWorld
  is the canonical options testbed (Sutton et al. 1999).
- Section 17.4: Designing reward signals — intrinsic motivation /
  curiosity; reward signals as a means to transfer learning across tasks.
  Count-based novelty (R + beta / sqrt(N)) is the textbook tabular
  instantiation of "reward signals that depend on ... novelty".
- Chapter 6 (Q-learning): the primitive-only baseline and the inner
  learning loops.

## What Is a Model vs. an Algorithm
Both demos are **small simulations**, not re-implementations of any
experiment in the book — the book describes neither a GridWorld options
run nor a novelty run with hyperparameter specifics. Choices made for
honest, testable behaviour are listed below, including one non-obvious
one for the novelty demo.

## Implementation Notes / Simplifications
### Options (17.1)
- Grid: 7x7 four-room layout, start (0,0), goal (6,6), walls in row 3
  (gaps at col 2 and col 4) and col 3 (rows 4-5). Reward +1 at the goal,
  step cost -0.01, wall bump -0.25, cap of 50 option steps.
- Options are **hand-crafted** macro-actions with BFS-greedy subgoal
  policies (a standard options demonstration setup): gap-left, gap-right,
  and goal. Their *values* are learned with **SMDP Q-learning**
  (`gamma**tau` bootstrap with option duration tau).
- A primitive-only Q-learner (4 actions) is trained on the same grid as
  the comparison baseline. Typical seeded result: the SMDP agent reaches
  the goal in 1-2 option decisions / 12 primitive steps; the primitive
  agent reaches it in 12 primitive decisions. (Move counts equal
  (near-optimal both ways); the win is in *decisions*.)
- The options demo is trained with fixed start (0,0), so its greedy policy
  is validated from the training start — not claimed to generalize to all
  starts.

### Intrinsic motivation (17.2)
- Bonus: `novelty_bonus(visits, beta) = beta / sqrt(visits + 1)`; counts
  accumulate over the whole run; the bonus is added to the extrinsic reward
  (R+ = R + bonus).
- No-op (bump) transitions harvest **no** intrinsic bonus — a self-loop
  would otherwise be an infinite reward source.
- Default `gamma = 0.0` (bandit-style): with discounting near 1, the
  bootstrapped bonus re-inflates values around a revisitation cycle and the
  tabular agent gets trapped. `gamma = 0` makes the bonus a one-step
  frontier signal, which is the cleanest tabular form.
- Default `q_init = 0.5` (**optimistic initialization** at the never-visited
  bonus value): without it, untried actions (Q = 0) lose to already-visited
  inflated Q-values and the agent can never reach the frontier. This is the
  standard partner of count-based exploration.
- Measured (seeded, 20 runs, 8 episodes x 50 steps each):
  novelty agent ~41.8 / 42 distinct states visited; zero-bonus control
  ~28. The bonus is environment-sensitive: with bootstrapping near gamma = 1
  or pessimistic Q-init, it can fail or even invert — this is documented
  here rather than tuned away.

## Scope
- Not implemented (no executable spec in the book, or out of scope):
  option discovery, intra-option learning, option-critic methods, function
  approximation, RND-style prediction bonuses, general value functions,
  model-based intrinsic motivation (Schmidhuber-style curiosity),
  hierarchy over multiple levels.

## How to Run
```bash
python -m pytest chapter_17 -q
python -m pytest chapter_17/tests/ -v
.venv\Scripts\ruff.exe check chapter_17 --select E,F,N,W
```

---
*Last updated: 2026-11-14*
