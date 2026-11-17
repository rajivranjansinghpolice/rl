# Chapter 16 — Applications and Case Studies

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 16

## Overview
The book's Chapter 16 is a survey: TD-Gammon (16.1), Samuel's checkers player
(16.2), Watson's Daily-Double wagering (16.3), memory controller optimization
(16.4), deep RL for video games (16.5), AlphaGo/AlphaGo Zero (16.6),
personalized web services (16.7) and thermal soaring (16.8). It contains no
boxed algorithms, and the applications described are far too large to
reproduce here.

## Exercises and Solutions

- [exercises.md](exercises.md) — derived exercises from the applied-RL/control case studies (§16.1–16.8
  survey + Chapter 4 DP methodology as it underlies the applied formulations), labeled [derived], with
  book §/PDF-page citations and coding-exercise pointers into this chapter (`InventoryEnv`,
  `sarsa_inventory`, `QueueEnv`, `q_learning_queue`).

This chapter therefore implements two **small classic control applications**
in the same spirit — sequential decision problems deployed against a real
cost model, solved with the tabular methods of Chapters 6–7:

1. **Algorithm 16.1** — multi-period **inventory management** with Sarsa
   (order-up-to action, Poisson demand, order/holding/stockout costs),
2. **Algorithm 16.2** — a **single-server queue** MDP with an
   expedited-service decision, waiting-time costs, and Q-learning.

## Algorithms Implemented
- [x] algorithm_16.1 — Inventory management controlled with Sarsa
  (`InventoryEnv`, `sarsa_inventory`, `evaluate_inventory_policy`,
  `greedy_inventory_action`)
- [x] algorithm_16.2 — Simplified resource-control application: single-server
  queue scheduling with a service-speed action (`QueueEnv`,
  `q_learning_queue`, `evaluate_queue_policy`, `greedy_queue_action`,
  `bellman_target`)

## Book Reference
- Section 16.1–16.8: applications surveyed (no executable pseudocode — this
  chapter's implementations are independent small MDPs, not book algorithms)
- Section 6.4 (Sarsa), Section 6.2 (TD(0) error), Section 6.5 (Q-learning):
  the learning algorithms used
- Section 3.x (finite-horizon episodic tasks): time-limit truncation is a
  truncation, not termination — handled by bootstrapping, see `bellman_target`

## What Is Modeled vs. What Is Optimized
Both implementations are genuine sequential **decision problems with a cost
model**: the agent picks order quantities / service effort to minimize
long-run cost. There is no application-domain fidelity claim beyond these
cost models; they are synthetic-but-honest sandbox MDPs standing in for the
class of problems in the chapter (operations research-style control with
stochastic demand/arrivals).

## Implementation Notes / Simplifications
- **Common simplifications**: tabular state (inventory level; customers in
  system), Poisson demand/arrivals, fixed period cost models, finite-horizon
  episodes (12 periods / 40 periods), constant-α updates, ε-greedy behavior,
  undiscounted true problem treated with γ = 0.95 in training.
- **Truncated episodes**: the time limit is a truncation, not a terminal
  state. Targets bootstrap at truncation (`bellman_target(terminal=False)`);
  treating it as terminal biases early-period values toward pure immediate
  cost. The queue Q-learning target is constructed through the explicit
  `bellman_target` helper, which is also unit-tested for both branches.
- **Order-up-to action**: action a means "raise position to a"; choosing
  a ≤ current inventory orders nothing (equivalent to not ordering).
- **Learned queue policy**: with the default waiting cost h = 2.0, waiting
  dominates the service-cost premium, and the learned policy essentially
  always expedites — this is the near-optimal behaviour for these
  parameters, verified by evaluation against an all-expedited reference.
  The tests only assert improvement vs. random and vs. never-expediting.
- **Evaluation protocol**: environment parameters are explicit; evaluation
  uses a disjoint seeded demand/arrival stream (`episodes=200–300`), so all
  compared policies face identical scenario sequences.

## Results (seeded, this machine)
- Inventory (seed 0, 600 training episodes): learned ≈ 6.6 cost/period vs.
  zero-order ≈ 13.4 and random ≈ 7.4 (200 eval episodes).
- Queue (seed 0, 1500 training episodes): learned ≈ 8.0 cost/period vs.
  random ≈ 9.3 and never-expediting ≈ 11.0 (300 eval episodes).

## Scope
- Not implemented: TD-Gammon, checkers, Watson wagering, memory control,
  Atari/Go deep RL, web services, thermal soaring — none have executable
  specifications in the book and all are research-scale.

## How to Run
```bash
python -m pytest chapter_16 -q
python -m pytest chapter_16/tests/ -v
.venv\Scripts\ruff.exe check chapter_16 --select E,F,N,W
```

---
*Last updated: 2026-11-14*
