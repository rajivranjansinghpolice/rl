# Chapter 14 — Psychology

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 14 (Psychology)

## Overview
Chapter 14 surveys RL's connections to the psychology of animal learning.
The core implementable content is the theory of **classical (Pavlovian)
conditioning**: what the book calls the Rescorla–Wagner model (a trial-level
error-correction rule) and the **TD model** of conditioning (a real-time
model equivalent to semi-gradient TD(λ) with linear function approximation).
This chapter implements both models, plus a runner for the Kamin **blocking**
paradigm (compound-cue conditioning) that they are meant to explain.

## Exercises and Solutions

- [exercises.md](exercises.md) — derived exercises from the book's Chapter 14 (psychology of learning;
  §14.2 Rescorla-Wagner & TD conditioning, §14.3–14.6), labeled [derived], with book §/PDF-page
  citations and coding-exercise pointers into this chapter (`RescorlaWagnerModel`,
  `TDConditioningModel`, `run_blocking_experiment`).

Chapter 14 also covers instrumental conditioning, goal-directed vs. habit
control, and higher-order conditioning; those are discussed in the book with
almost no additional implementable pseudocode, so they are listed under
"Scope" below as not implemented.

## Algorithms Implemented
- [x] algorithm_14.1 — Rescorla–Wagner model (`RescorlaWagnerModel`, `excitation_curve`) — Eqs. 14.1–14.3
- [x] algorithm_14.2 — TD model of classical conditioning (`TDConditioningModel`, `presence_trial`, `csc_trial`, `run_trial`, `train_presence`) — Eqs. 14.4–14.6
- [x] algorithm_14.3 — Kamin blocking experiment runner (`run_blocking_experiment`) — Sections 14.2.1–14.2.2

## Book Reference
- Section 14.2.1: Blocking and higher-order conditioning (Kamin, 1969)
- Section 14.2.2: The Rescorla–Wagner model — Eqs. (14.1)–(14.3)
- Section 14.2.3: The TD model — Eqs. (14.4)–(14.6)
- Section 14.2.4: TD model simulations; presence and CSC representations (Figure 14.2); prediction profiles (Figure 14.7)
- Section 14.2.1 Figure 14.3/14.4: Egger–Miller / facilitation phenomena (not implemented, see Scope)

## What Is a Model vs. an Algorithm
The book presents these as **theories of animal learning**, not numbered
algorithms. Nothing is executed against a reward-optimizing agent; the "task"
is reproducing the regularities of conditioning data (asymptotic growth,
blocking, extinction, ISI effects). Where the book describes a phenomenon
(blocking) rather than an algorithm, the code simulates the standard
experimental paradigm on top of the Rescorla–Wagner model and says so in the
docstring.

## Implementation Notes / Simplifications
- **Trials are episodic** (Section 15.6 assumption): the trial ends right
  after the US prediction, so `n_steps == reward_time` in the helpers'
  intended use. The rewarding state itself never predicts reward, matching
  the book ("the value of this state would come to predict the return over
  all the following trials, which here we are assuming to be zero").
- **Presence representation** (Figure 14.2, right): one feature per CS,
  valued 1 while present. Converges to a *constant* prediction whose level
  is set by US intensity and CS–US length — exactly what the book reports
  for this representation.
- **CSC representation** (Figure 14.2, left): one feature per
  (stimulus, time-since-onset); the US prediction rises through the interval
  and peaks at the US time (verified by test).
- **Blocking tests** use λ=0 (TD(0), trace-free) or λ=0.5 with γ∈{0.97, 1.0};
  with the book's default λ=0.95 + presence representation the compound and
  single-cue dynamics differ enough to blur the classic signature, so the
  property tests pin λ ≤ 0.5. The phenomenon is identical; only the trace
  bookkeeping changes.
- `run_blocking_experiment` implements the classic phase-1/phase-2 design on
  the **Rescorla–Wagner model** (the model the book created to explain
  blocking), with seeded Gaussian US-magnitude jitter for replicate-level
  statistics. TD-model blocking is tested separately in the 14.2 suite.
- US magnitude defaults to λ_US = 1.0; "noise" applies to the US per trial.
- Extinction is modeled as trials with target 0 (US omitted), as in the book.

## Scope
- Not implemented (book gives no executable algorithm; not requested):
  instrumental conditioning law of effect, Thorndike puzzle boxes,
  habitual vs. goal-directed control (Section 14.3/14.4), higher-order
  conditioning dynamics, serial-compound facilitation (Egger–Miller),
  stimulus representation microstimuli (MS), response generation/CR timing.
- The δ used here is an action-free prediction error (US prediction), as in
  the book's TD model of classical conditioning.

## How to Run
```bash
python -m pytest chapter_14 -q
python -m pytest chapter_14/tests/ -v
.venv\Scripts\ruff.exe check chapter_14 --select E,F,N,W
```

---
*Last updated: 2026-11-14*
