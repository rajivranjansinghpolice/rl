# Chapter 2 — Bandits

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition)

## Overview
This chapter covers multi-armed bandits and the exploration strategies the
literature developed for them (optimistic initial values, upper-confidence-bound
action selection, descending optimism bonus), plus their application to
tabular Q-learning on small MDPs.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 2.1–2.9 with from-scratch solutions, book §/equation/PDF-page citations, and coding-exercise pointers into this chapter (`BanditProblem`, `ActionValueEstimator`, `select_epsilon_greedy`, `ucb_bonus`, `select_ucb`, `GradientBanditAgent`, `ContextualBandit`), followed by derived exploration exercises from book §§2.6–2.7.

## Algorithms Implemented
- [x] algorithm_2.1 — k-armed bandit problem (`BanditProblem`)
- [x] algorithm_2.2 — action-value estimation / sample averages (`ActionValueEstimator`)
- [x] algorithm_2.3 — epsilon-greedy (`select_epsilon_greedy`)
- [x] algorithm_2.4 — UCB (`ucb_bonus`, `select_ucb`)
- [x] algorithm_2.5 — gradient bandit (`GradientBanditAgent`, `select_softmax`)
- [x] exercise_2.3 — non-stationary bandits (`simulate_nonstationary`)
- [x] exercise_2.4 — optimistic initial values (`simulate_optimistic_initial_values`)
- [x] exercise_2.5 — UCB with uncertainty (`simulate_ucb_uncertainty`)
- [x] exercise_2.6 — contextual bandits (`ContextualBandit`)
- [x] exploration — UCB (`run_ucb_bandit`, `q_learning_ucb`), optimism in the face of
  uncertainty (`run_optimistic_greedy_bandit`, `q_learning_optimistic`, descending bonus
  (`run_descending_bonus_bandit`, `q_learning_descending_bonus`), small MDP
  (`make_small_gridworld`)

## Book Reference
- Section 2.2: The k-armed bandit problem; Section 2.3: Action values and sample averages / ε-greedy;
  Section 2.4: Incremental implementation / non-stationary step sizes; Section 2.5: Tracking a
  non-stationary problem; Section 2.6: Optimistic initial values; Section 2.7: Upper-confidence-bound
  action selection; Section 2.8: Gradient bandit algorithms; Section 2.9: Associative search /
  contextual bandits.
- Exploration implementations reference book Section 2.7 (UCB, Eq. 2.8), Section 2.6 (optimistic
  initial values, Fig. 2.3), and Exercise 11.1 (descending optimism bonus / OFU).

## Environment Setup
All bandit algorithms in this chapter use only `numpy`; the Q-learning exploration runs use the
shared `FiniteMDP` from `chapter_03`.

## Quick Start
```bash
# Run all tests in this chapter
pytest chapter_02/tests/ -v

# Lint
ruff check chapter_02/
```
