# Chapter 9 — On-policy Prediction with Approximation

> Reference implementation of Sutton & Barto's
> _Reinforcement Learning: An Introduction_ (2nd Edition), Chapter 9

## Overview
Function-approximation prediction: linear value functions, the prediction
objective (VE), semi-gradient TD(0), constant-alpha MC prediction, Fourier
basis, tile coding, and RMS error measurement.

## Exercises and Solutions

- [exercises.md](exercises.md) — Exercises 9.1–9.5 derived from Sections 9.2–9.6
  with solutions, derivations, and coding pointers (`semi_gradient_td0_update`,
  `constant_alpha_mc_update`, `fourier_features`, `TileCoder`,
  `mean_squared_value_error`, `rms_error`, `run_wavelength_prediction`).

## Algorithms Implemented
- [x] — Linear semi-gradient TD(0) for value prediction (`semi_gradient_td0_update`) ✅
- [x] — Constant-alpha MC linear prediction (`constant_alpha_mc_update`) ✅
- [x] — Fourier cosine basis, order 0..m (`fourier_features`) ✅
- [x] — Tile coding for 2D boxes (`TileCoder`) ✅
- [x] — Prediction objective VE / RMS error (`mean_squared_value_error`, `rms_error`) ✅
- [x] — Wavelength demonstration of the prediction objective (`run_wavelength_prediction`,
  `plot_figure_9_wavelength`) ✅

## Book Reference
- Section 9.2: The prediction objective — mean squared value error (VE)
- Section 9.3: Stochastic-gradient and semi-gradient methods
- Section 9.4: Linear methods; feature construction for linear methods
  (Fourier basis, tile coding)
- Section 9.6: Selecting step-size parameters manually

## Not implemented (with justification)
- **Nonlinear/neural-network approximation and kernel methods** — descoped; the
  linear theory here carries the same prediction objective, and NN training
  infrastructure is out of this repo's pedagogical scope.
- **Episodic semi-gradient control** — this is Chapter 10 and is implemented there.

## Figures
- `figures/example_9_1_wavelength_fourier.png` — Fourier-basis fit of cos(2*pi*x) for
  orders 0–5 (illustrative demonstration of Sections 9.2/9.4.5; not a claimed
  numbered book figure — see the book-identity note in `exercises.md`).

## Implementation Notes
- All functions operate on NumPy arrays; import as a package:
  `from chapter_09.solution import ...`.
- The linear semi-gradient TD(0) update is the algorithmic core; unit tests pin
  down its exact arithmetic including the zero-bootstrap terminal case.
- `run_wavelength_prediction` fits the known wave by minimising VE over uniform
  samples; `evaluate_approximation` gives held-out RMS error per order.

## Quick Start
```bash
python -m pytest chapter_09/tests/ -v
python -m pytest
ruff check chapter_09/
python scripts/run_quality_gates.py 09
```

---

*Last updated: 2026-11-18*
