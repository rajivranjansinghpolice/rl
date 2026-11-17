# Chapter 15 — Neuroscience

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 15 (Neuroscience)

## Overview
Chapter 15 relates reinforcement learning to the neuroscience of reward
processing. The core implementable content is the **reward prediction error
hypothesis of dopamine neuron activity** (Sections 15.3 and 15.6): the phasic
activity of dopamine neurons tracks the TD error δ. This chapter simulates
the idealized conditioning task of Section 15.6 with a TD(λ) learner over a
complete serial compound (CSC) layout and reads δ as a phasic dopamine
response.

## Exercises and Solutions

- [exercises.md](exercises.md) — derived exercises from the book's Chapter 15 (neuroscience of learning;
  §15.3–15.6 RPE/TD correspondence, §15.7–15.8 actor-critic, §15.12 addiction), labeled [derived], with
  book §/PDF-page citations and coding-exercise pointers into this chapter (`PhasicDopamineModel`,
  `simulate_conditioning_trace`).

The chapter also discusses dopamine anatomy, D1/D2 receptor pathways,
habit-vs-goal-directed control, drug addiction models, collective RL, and
model-based vs. model-free distinctions — none of which come with executable
algorithms, so they are out of scope (see Scope).

## Algorithms Implemented
- [x] algorithm_15.1 — Phasic dopamine / TD-error model (`PhasicDopamineModel`,
  `simulate_conditioning`) — Sections 15.3, 15.6; Eqs. 14.4–14.6 (tabular)
- [x] algorithm_15.2 — Cue-reward conditioning trace (`simulate_conditioning_trace`) —
  Sections 15.5–15.6; acquisition/extinction/reinsertion of the phasic response

## Book Reference
- Section 15.3: The reward prediction error hypothesis of dopamine neuron activity
- Section 15.5: Experimental support (Romo & Schultz 1990; Ljungberg et al. 1992;
  Schultz et al. 1993; Schultz, Dayan & Montague 1997)
- Section 15.6: The TD error/dopamine correspondence; Figure 15.5 (the
  idealized task simulated here); Montague, Dayan & Sejnowski (1996)
- Figure 15.3/15.4: dopamine responses shift to earliest predictor; dip below
  baseline at omitted reward — both reproduced by the simulation
- Section 14.2.3 / Figure 14.2 (CSC representation): used as the state layout

## What Is a Model vs. an Algorithm
These are **computational models of a neural signal**, not algorithms. δ is
literally the TD error; the hypothesis (Sections 15.3/15.6) is that phasic
dopamine activity *is* δ (Montague et al. 1996). Nothing optimizes a
behavioural objective. The ITI state is frozen at 0 following the book's
episodic assumption ("predictions are confined to individual trials",
Section 15.6); the book's observation that ITI responses can emerge with
overtraining is therefore out of scope, not a bug.

## Implementation Notes / Simplifications
- **CSC layout, tabular TD(λ)**: one state per time step (complete serial
  compound, Figure 14.2 left), eligibility traces λ = 0.95 (book default),
  γ = 1.0 (book assumes ~1), α = 0.1, one shared ITI state with value/trace
  frozen at 0.
- **Backward-view update**: δ propagates through eligibility traces (Eqs.
  14.4–14.6), the tabular special case of semi-gradient TD(λ) (Chapter 12).
  With TD(0) alone (λ = 0) the cue shift is far slower (reward spreads back
  one state per trial), which is why the tests use λ = 0.95.
- **A negative δ is read as a dip below a neuron's background firing rate**,
  following Montague et al.'s assumption that dopamine activity corresponds
  to δ + b (b = background rate); only δ is simulated.
- **Extinction** is implemented as CS trials with the US omitted (reward 0),
  matching the "predicted reward omitted" condition of Figure 15.5.
- **Pattern verified** (see tests): early trials — large δ at the US and
  ~0 at the CS; late trials — δ at the CS and ~0 at the US; omission after
  learning — δ ≈ −R* at the expected US time; extinction fades the CS δ;
  reinsertion restores positive δ at the US.
- Deterministic simulation (no RNG needed); `seed`-like jitter is not used.

## Scope
- Not implemented (no executable algorithm in the book): actor–critic /
  dopamine pathway details (Section 15.7), ventral/dorsal striatum
  distinctions, D1/D2 receptor models, drug addiction simulations
  (Section 15.8), collective RL (Section 15.9), model-based dopamine
  accounts (Daw et al. 2006; Section 15.6 note), microstimulus (MS)
  representations (Ludvig et al. 2008).

## How to Run
```bash
python -m pytest chapter_15 -q
python -m pytest chapter_15/tests/ -v
.venv\Scripts\ruff.exe check chapter_15 --select E,F,N,W
```

---
*Last updated: 2026-11-14*
