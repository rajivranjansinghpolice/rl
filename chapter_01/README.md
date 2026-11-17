# Chapter 1 — What is Reinforcement Learning?

This chapter introduces the reinforcement learning problem through the **Gridworld** example from Sutton & Barto, Section 1.3.

## Contents

| File | Description |
|------|-------------|
| `pseudocode.py` | Pseudocode mapping to book's Gridworld MDP steps |
| `exercise_1_1.py` | Gridworld environment and agent simulation |
| `tests/` | Unit tests for exercise implementations |

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 1.1–1.5 with from-scratch solutions, book §/PDF-page
  citations, and coding-exercise pointers into this chapter (GridWorld, agents, episode runs).

## The Gridworld MDP (Figure 1.2)

A 4x4 grid world with terminal states A and B:

- **State A**: Best action → reward +10 (A+), second best → state transition to B (reward +5)
- **State B**: Actions lead to C (up, right) or D (down, left), all reward +5
- **Terminal states**: A+ and D (absorbing states)

### Key Concepts Introduced
1. **Agent** — The learner and decision maker
2. **Environment** — Everything the agent interacts with
3. **State (s)** — Current situation
4. **Action (a)** — What the agent can do
5. **Reward (r)** — Numeric feedback signal
6. **Return (G_t)** — Discounted sum of future rewards

---

*Last updated: 2026-10-13*
