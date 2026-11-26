# Chapter 9 — On-policy Prediction with Approximation: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd edition). Repo chapter: `chapter_09`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_09` — On-policy prediction with function approximation
  (linear semi-gradient TD(0), constant-alpha MC linear prediction, the prediction
  objective VE, Fourier cosine basis, tile coding).
- **Book chapter:** Chapter 9 — On-policy Prediction with Approximation.
- **Book sections covered by the implementation:**
  - **§9.2** The prediction objective — mean squared value error (VE)
    (`mean_squared_value_error`, `rms_error`).
  - **§9.3** Stochastic-gradient and semi-gradient methods
    (`semi_gradient_td0_update`, `constant_alpha_mc_update`).
  - **§9.4** Linear methods, feature construction for linear methods
    (`fourier_features`, `TileCoder`).
  - **§9.6** Selecting step-size parameters manually (see Exercise 9.3).
- **Not implemented in this repo chapter (with justification):**
  - Nonlinear (neural network) function approximation and kernel-based
    approximation are not implemented. The book treats them in Sections 9.6+ / 9.7,
    but they require training infrastructure (deep nets) beyond this repo's
    pedagogical scope; the linear methods here carry the same learning-objective
    theory. Descoped with justification.
  - Episodic semi-gradient control is the subject of **Chapter 10** and is not
    duplicated here.
- **Book-identity note:** the local PDF text ships only a summary of Chapter 9
  (no exact exercise/figure numbers can be verified here), so the exercises below
  are *derived from the chapter's sections* rather than quoted verbatim, and the
  figure is an illustrative demonstration, not a claim about a numbered book figure.

---

## Key book equations (plausible citations, S&B 2nd ed.)

- Eq. (9.1) — Linear value function: \(\hat v(w,s) = w^\top x(s)\).
- Eq. (9.3) — Stochastic-gradient update for VE(w) (before the semi-gradient caveat).
- Eq. (9.5) — Semi-gradient TD(0) update: \(w \leftarrow w + \alpha \delta x(S)\) with
  \(\delta = R + \gamma \hat v(w, S') - \hat v(w, S)\).
- Eq. (9.6)/(9.7) — Constant-alpha MC and TD(0) updates on the linear model.

---

## Exercise 9.1 — Why is the TD(0) update "semi"-gradient?

- **Book location:** Section 9.3 (derivation of the semi-gradient method).
- **Type:** derivation/explanation.
- **Problem (paraphrased):** The true gradient of the prediction objective VE(w)
  with TD targets involves the derivative of the next-state value \(V(S')\). Show
  why the book ignores that derivative and what changes as a result.

**Solution.** Write VE(w) = \(\sum_s \mu(s) [v_\pi(s) - \hat v(w, s)]^2\). The gradient is

\[
\nabla_w \text{VE}(w) = -2 \sum_s \mu(s) [v_\pi(s) - \hat v(w, s)] \nabla_w \hat v(w, s).
\]

In tabular TD we cannot compute the sum exactly, so we take a sample and use a
*target* \(U_t\) (e.g., \(R_{t+1} + \gamma \hat v(w, S_{t+1})\)) in place of \(v_\pi(S_t)\):

\[
w \leftarrow w + \alpha \,[U_t - \hat v(w, S_t)] \,\nabla_w \hat v(w, S_t).
\]

A true gradient of the sample squared TD error (\(U_t - \hat v(w,S_t)\))^2 would additionally
differentiate \(U_t\) with respect to \(w\) (through the \(\hat v(w, S_{t+1})\) term). The book
**drops that term** — it is usually small and always opposes the correct direction,
and dropping it is what makes the method stable and still convergent to the TD fixed
point for linear features. Hence "semi-gradient": we follow half the gradient — the
part through the current-state value, not the part through the bootstrap target.
Because the target depends on \(w\) only through the next state, the missing factor is
\(\nabla_w \hat v(w, S_{t+1})\), scaled by \(\gamma\).

**Explanation (student-friendly).** Think of it as "chase the moving target, don't
push the carrot." The TD error says how far off your estimate is; the semi-gradient
steps the current estimate toward the bootstrap target. If you also differentiated
the target, you would be trying to minimize how much the target would change, which
is a different (and often destabilizing) objective.

- **Coding pointer.** `semi_gradient_td0_update` implements exactly this update;
  the bootstrap-derivative drop is visible because `x_next` is never differentiated
  in the code. Unit tests `TestSemiGradientTD0` pin down its exact arithmetic.

---

## Exercise 9.2 — Why does the VE objective differ from MSVE on the true value?

- **Book location:** Section 9.2 (on-policy distribution \(\mu\)).
- **Type:** conceptual/derivation.
- **Problem (paraphrased):** The book minimizes \(\text{VE}(w) = \sum_s \mu(s)(v_\pi(s) - \hat v(w,s))^2\)
  where \(\mu\) is the on-policy distribution, not the policy's value directly. Show
  that this focuses the approximation on states the policy actually visits, and
  explain the practical importance.

**Solution.** VE weights each state's squared error by \(\mu(s)\), the discounted
fraction of time spent in \(s\) under \(\pi\) and the environment's start distribution.
An approximation error in a state the policy never visits contributes nothing to
VE and nothing to behavior. In practice this means the fit is good where it
matters and can be poor (even arbitrarily bad) elsewhere — acceptable, because
those states are never evaluated behaviorally. This is why two function classes
with the same capacity can produce identical prediction behavior yet very different
VE; conversely, minimizing uniform-state squared error (as in regression on a
uniform grid) is a different objective and generally worse for sequential prediction.

**Explanation (student-friendly).** You are not required to model the whole world —
only the part of the world you actually walk through. Optimizing for visited states
is why on-policy prediction can be far cheaper than "solve everywhere."

- **Coding pointer.** `mean_squared_value_error` implements VE over an arbitrary
  sample set; pass the states the policy visits to evaluate behavioral quality.

---

## Exercise 9.3 — Fourier basis orders for the wave \(v(x) = \cos(2\pi x)\)

- **Book location:** Section 9.4.5 (Fourier basis) / the wavelength illustration.
- **Type:** experiment (implemented in this repo).
- **Problem (paraphrased):** Using the Fourier cosine basis
  \(\phi_k(x) = \cos(k\pi x)\), \(k = 0, \dots, m\), fit \(v(x) = \cos(2\pi x)\) on
  \(x \in [0,1]\) for several orders \(m\) and explain which orders suffice.

**Solution.** With the basis \(\cos(k\pi x)\), the true wave \(\cos(2\pi x)\) is a
single basis member (\(k = 2\)). Hence orders \(\ge 2\) can represent it exactly,
and the fitted \(\hat v\) should essentially reproduce the wave (residual mass
≈ numerical error). Order 0 keeps only the constant feature \(\cos(0) = 1\); the
best it can do is the mean of the wave (which is 0, so it predicts 0). Order 1
adds \(\cos(\pi x)\), still misaligned with \(\cos(2\pi x)\). So:

- \(m = 0\): constant predictor, RMS error is the wave's RMS value (~0.707).
- \(m = 1\): improves a little but cannot capture the 2nd harmonic.
- \(m \ge 2\): near-exact fit.

Note the prediction objective VE is what is being minimized; this is why the
fitted solution is exactly the projection of the wave onto the chosen subspace,
not an interpolation.

**Explanation (student-friendly).** Think of fitting a wave as "give the learner a
set of building blocks" (0th, 1st, 2nd, ... cosine harmonics). The wave is built
from block #2 exactly — so hand the learner at least block #2 and it can build the
wave; without it, no combination of block #0 and #1 works. This is the whole
Fourier-analysis idea, applied to value functions.

- **Coding pointer.** `fourier_features(x, order)` and `WavelengthTask` implement
  this; `run_wavelength_prediction` fits and returns the weights;
  `plot_figure_9_wavelength()` draws the per-order fit (orders 0–5) stored at
  `chapter_09/figures/example_9_1_wavelength_fourier.png`. Unit test
  `test_higher_order_fits_wave_better` verifies orders 5 fit strictly better than
  order 0 in RMS error.

---

## Exercise 9.4 — Tile coding: how many features and why generalize?

- **Book location:** Section 9.4.3 (tile coding).
- **Type:** conceptual/design.
- **Problem (paraphrased):** Given a 2D box \([0,1]^2\), 8 tiles per axis per
  tiling and 8 tilings, how many features are active per state? Explain how tile
  coding generalizes and why it beats a single tiling.

**Solution.** Each tiling activates exactly one tile (one feature) per point, so
with 8 tilings exactly **8** features are active per state out of
\(8 \times 8 \times 8 = 512\) total. Tile coding generalizes *locally* (nearby
points share many tiles) and *non-locally* (shifted tilings let distant points
share some), and each feature is strictly local to a tile. A single tiling has
the "box-corner problem": a point near a corner activates one tile while a point
barely across the corner boundary activates a completely different tile, giving
poor generalization across the boundary. Staggering 3–8 tilings offset by a fraction
of a tile width removes that arbitrariness and gives a smoother approximation.

**Explanation (student-friendly).** It is like paving a floor with shifted grids of
colored tiles. Any point touches one tile in each grid; points close together
touch the same tiles in most grids, so the learner updates the same features.
Shifted, overlapping grids avoid the "unlucky boundary" where a tiny step in the
world flips almost every active feature.

- **Coding pointer.** `TileCoder(tiles_per_tiling=8, n_tilings=8)` implements this;
  unit tests `TestTileCoder` assert the binary vector has exactly `n_tilings`
  ones and that off-grid points are clipped into the boundary tiles.

---

## Exercise 9.5 — Semi-gradient TD(0) on a known two-state MDP

- **Book location:** Section 9.3 (the prediction architecture).
- **Type:** implementation (verified in tests).
- **Problem (paraphrased):** Consider the two-state continuing MDP that stays in
  state A with probability 1 and pays reward 0 (terminal in infinity). Show that
  semi-gradient TD(0) converges. More usefully: verify on a single-step transition
  that the arithmetic matches the book's update exactly.

**Solution.** For one transition \(S \to S'\) with reward \(R\), semi-gradient TD(0)
makes the one-step change

\[
w \leftarrow w + \alpha \,[R + \gamma \hat v(w, S') - \hat v(w, S)] \phi(S).
\]

With a terminal \(S'\) the bootstrap value is 0, so the target is \(R\). For linear
features and \(\alpha\) in a suitable range, w converges (on average, in continuing
or properly discounted tasks) to the TD fixed point, which is the closest vector to
\(v_\pi\) in the feature subspace as weighted by the on-policy distribution — a
projection under the \(\mu\)-weighted norm, matching Exercise 9.2.

**Explanation (student-friendly).** After each step the estimate is nudged toward
"what would have been better": the difference between the immediate reward plus
discounted guess-after-transition and the guess-before-transition. Repeat this a lot
and the estimate settles into the prediction the data actually supports.

- **Coding pointer.** Unit tests: `TestSemiGradientTD0` covers the exact one-step
  arithmetic (including a zero bootstrap for terminals) and that `alpha = 0` leaves
  the weights unchanged.

---

## How to run the chapter tests

```bash
python -m pytest chapter_09/tests/ -v
```
