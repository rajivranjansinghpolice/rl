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

**Hook.** Let's follow the gradient of the prediction objective until a bootstrap target gets in the way. That obstruction is precisely why the update is called "semi."

**Formal bridge.** The objective weights squared prediction errors by the on-policy state distribution:

\[
\mathrm{VE}(w)=\sum_s\mu(s)\bigl[v_\pi(s)-\hat v(w,s)\bigr]^2,
\tag{9.1.1}
\]

whose full gradient is \(-2\sum_s\mu(s)[v_\pi(s)-\hat v(w,s)]\nabla_w\hat v(w,s)\). Replacing \(v_\pi(S_t)\) by a sampled TD target \(U_t=R_{t+1}+\gamma\hat v(w,S_{t+1})\) gives the sampled update

\[
w\leftarrow w+\alpha\bigl[U_t-\hat v(w,S_t)\bigr]\nabla_w\hat v(w,S_t).
\tag{9.1.2}
\]

where \(w\) is the weight vector, \(\mu(s)\) the on-policy state distribution, \(v_\pi(s)\) the true state value, \(\hat v(w,s)\) the approximation, \(U_t\) the bootstrap target, \(R_{t+1}\) the reward, \(\gamma\) the discount, and \(\alpha\) the step size.

**Reasoning chain.** A true gradient of the sample squared TD error \((U_t-\hat v(w,S_t))^2\) would additionally differentiate \(U_t\) with respect to \(w\), through the \(\hat v(w,S_{t+1})\) term. The book drops that term — the missing factor is \(\nabla_w\hat v(w,S_{t+1})\), scaled by \(\gamma\). It is usually small and always opposes the correct direction; dropping it is what makes the method stable and still convergent to the TD fixed point for linear features. Hence "semi-gradient": we follow half the gradient — the part through the current-state value, not the part through the bootstrap target.

In the tabular case the sum in (9.1.1) cannot be computed exactly, so we take a sample and use \(U_t\) in place of \(v_\pi(S_t)\), which leads to (9.1.2). The update nudges the current estimate toward the sampled target using only the current state's feature vector — `x_next` is never differentiated.

**Why it makes sense.** Think of it as "chase the moving target, don't push the carrot." The TD error says how far off your estimate is; the semi-gradient steps the current estimate toward the bootstrap target. If you also differentiated the target, you would be trying to minimize how much the target would change — a different, and often destabilizing, objective. The extreme case makes the direction clear: with \(\gamma=0\) the target is just the reward and has no \(w\)-dependence at all, so dropping the derivative costs nothing; the "semi" caveat only bites because bootstrapping makes the target move with \(w\).

**Check your understanding.** Prediction: raise \(\gamma\) toward 1 so the bootstrap target depends more strongly on \(w\); the omitted derivative in (9.1.2) grows, which is exactly why long-horizon off-policy bootstrapping becomes unstable, while at \(\gamma=0\) the semi-gradient is the true gradient.

**Where this shows up in the code.** `semi_gradient_td0_update` implements exactly this update; the bootstrap-derivative drop is visible because `x_next` is never differentiated in the code. Unit tests `TestSemiGradientTD0` pin down its exact arithmetic.

---

## Exercise 9.2 — Why does the VE objective differ from MSVE on the true value?

- **Book location:** Section 9.2 (on-policy distribution \(\mu\)).
- **Type:** conceptual/derivation.
- **Problem (paraphrased):** The book minimizes \(\text{VE}(w) = \sum_s \mu(s)(v_\pi(s) - \hat v(w,s))^2\)
  where \(\mu\) is the on-policy distribution, not the policy's value directly. Show
  that this focuses the approximation on states the policy actually visits, and
  explain the practical importance.

**Hook.** Approximating a value function everywhere is a luxury. On-policy prediction asks a cheaper question: how well do we do in the states we actually see?

**Formal bridge.** The objective is

\[
\mathrm{VE}(w)=\sum_s\mu(s)\bigl[v_\pi(s)-\hat v(w,s)\bigr]^2,
\tag{9.2.1}
\]

with the on-policy distribution factorised as

\[
\mu(s)=(1-\gamma)\sum_{s_0}d(s_0)\sum_{t=0}^{\infty}\gamma^t P(S_t=s\mid S_0=s_0,\pi),
\tag{9.2.2}
\]

where \(\mu(s)\) is the discounted fraction of time spent in state \(s\) under policy \(\pi\) and the environment's start distribution \(d\), \(\gamma\) is the discount, \(v_\pi\) the true value, and \(\hat v\) the approximation with weights \(w\).

**Reasoning chain.** VE weights each state's squared error by \(\mu(s)\), the discounted fraction of time spent in \(s\) under \(\pi\) and the start distribution. An approximation error in a state the policy never visits has \(\mu(s)=0\), so it contributes nothing to (9.2.1) and nothing to behaviour. In practice this means the fit is good where it matters and can be poor — even arbitrarily bad — elsewhere, which is acceptable because those states are never evaluated behaviourally.

This is why two function classes with the same capacity can produce identical prediction behaviour yet very different VE. Conversely, minimizing uniform-state squared error, as in regression on a uniform grid, is a different objective from (9.2.2) and generally worse for sequential prediction: it spends capacity on states the agent will never walk through.

**Why it makes sense.** Imagine a robot that only ever moves along one corridor of a large building. Fitting the whole floor plan accurately wastes sensors; fitting the corridor accurately is what lets it predict returns. VE formalises that: weight by frequency of visitation. If a state has \(\mu(s)=0\), no amount of error there can ever affect the agent's decisions, so ignoring it is not laziness but the right allocation of effort.

**Check your understanding.** Prediction: if the policy becomes uniformly random over states so that \(\mu\) flattens, VE approaches an unweighted average of squared errors and the fit must spread its capacity thin; if \(\mu\) concentrates on one state, the fit will nearly nail that state and ignore the rest.

**Where this shows up in the code.** `mean_squared_value_error` implements VE over an arbitrary sample set; pass the states the policy visits to evaluate behavioural quality.

---

## Exercise 9.3 — Fourier basis orders for the wave \(v(x) = \cos(2\pi x)\)

- **Book location:** Section 9.4.5 (Fourier basis) / the wavelength illustration.
- **Type:** experiment (implemented in this repo).
- **Problem (paraphrased):** Using the Fourier cosine basis
  \(\phi_k(x) = \cos(k\pi x)\), \(k = 0, \dots, m\), fit \(v(x) = \cos(2\pi x)\) on
  \(x \in [0,1]\) for several orders \(m\) and explain which orders suffice.

**Hook.** Think of the basis functions as building blocks for a wave. The question is simply: which blocks do you need before the target can be assembled exactly?

**Formal bridge.** With linear features \(\hat v(x,w)=\sum_{k=0}^{m}w_k\phi_k(x)\) and \(\phi_k(x)=\cos(k\pi x)\), minimising VE over the sampled states gives the least-squares projection

\[
w^*=\arg\min_w\sum_{x\in\mathcal X}\bigl[v(x)-\phi(x)^\top w\bigr]^2
=\bigl(\Phi^\top\Phi\bigr)^{-1}\Phi^\top v,
\tag{9.3.1}
\]

where \(w_k\) is the weight on cosine harmonic \(k\), \(m\) is the basis order, \(\phi(x)\) is the vector of features at \(x\), \(\Phi\) is the matrix of those features over the sample, and \(v\) is the target wave.

**Reasoning chain.** With the basis \(\cos(k\pi x)\), the true wave \(\cos(2\pi x)\) is a single basis member, \(k=2\). Hence orders \(\ge 2\) can represent it exactly: the fitted \(\hat v\) essentially reproduces the wave, with residual mass ≈ numerical error. Order 0 keeps only the constant feature \(\cos(0)=1\); the best it can do is the mean of the wave, which is 0, so it predicts 0. Order 1 adds \(\cos(\pi x)\), still misaligned with \(\cos(2\pi x)\).

The concrete outcomes are: \(m=0\) gives a constant predictor whose RMS error is the wave's RMS value (~0.707); \(m=1\) improves a little but cannot capture the 2nd harmonic; \(m\ge 2\) gives a near-exact fit. Note that the prediction objective VE is what is being minimised in (9.3.1), so the fitted solution is exactly the projection of the wave onto the chosen subspace, not an interpolation through selected points.

**Why it makes sense.** Fitting a wave means handing the learner a set of harmonics. The target is built from block #2 alone, so give the learner at least block #2 and it can reproduce the wave; without it, no combination of block #0 and #1 works. This is Fourier analysis applied to value functions: missing harmonics cannot be synthesised by adding the ones you have.

**Check your understanding.** Prediction: switching the target to \(\cos(4\pi x)\) would require order \(m\ge 4\), so a fit at \(m=2\) that was exact before would now fail; adding higher-order blocks never hurts the projection but costs more features.

**Where this shows up in the code.** `fourier_features(x, order)` and `WavelengthTask` implement this; `run_wavelength_prediction` fits and returns the weights; `plot_figure_9_wavelength()` draws the per-order fit (orders 0–5) stored at `chapter_09/figures/example_9_1_wavelength_fourier.png`. Unit test `test_higher_order_fits_wave_better` verifies orders 5 fit strictly better than order 0 in RMS error.

---

## Exercise 9.4 — Tile coding: how many features and why generalize?

- **Book location:** Section 9.4.3 (tile coding).
- **Type:** conceptual/design.
- **Problem (paraphrased):** Given a 2D box \([0,1]^2\), 8 tiles per axis per
  tiling and 8 tilings, how many features are active per state? Explain how tile
  coding generalizes and why it beats a single tiling.

**Hook.** Tile coding turns a continuous point into a short binary feature vector. Let's count what a single point lights up, then see why overlapping grids matter more than grid size.

**Formal bridge.** With \(n_{\text{tile}}\) tiles per axis and \(n_{\text{tilings}}\) tilings, the feature count is

\[
\#\text{features}=n_{\text{tilings}}\cdot n_{\text{tile}}^d,
\qquad
\#\text{active per state}=n_{\text{tilings}},
\tag{9.4.1}
\]

with the linear approximation

\[
\hat v(s,w)=w^\top x(s)=\sum_{i=1}^{n_{\text{tilings}}}w_{i,\text{tile}_i(s)}.
\tag{9.4.2}
\]

where \(d=2\) is the state-space dimension, \(n_{\text{tile}}=8\) tiles per axis, \(n_{\text{tilings}}=8\) staggered tilings, \(x(s)\) the binary feature vector, and \(w\) the weights.

**Reasoning chain.** Each tiling activates exactly one tile — one feature — per point, so with 8 tilings exactly **8** features are active per state out of \(8\times 8\times 8=512\) total features in (9.4.1). Tile coding generalizes *locally*, because nearby points share many tiles, and *non-locally*, because shifted tilings let distant points share some features; each individual feature is strictly local to one tile.

A single tiling has the "box-corner problem": a point near a corner activates one tile, while a point barely across the corner boundary activates a completely different tile, giving poor generalization across that boundary. Staggering 3–8 tilings offset by a fraction of a tile width removes the arbitrariness of where the boundaries fall and yields a smoother approximation — nearby points differ in only a fraction of their 8 active features in (9.4.2).

**Why it makes sense.** It is like paving a floor with shifted grids of coloured tiles. Any point touches one tile in each grid; points close together touch the same tiles in most grids, so the learner updates the same features. Overlapping grids avoid the unlucky boundary where a tiny step in the world flips almost every active feature — with one grid that is exactly what happens at a corner, and with eight staggered grids it happens at most in one or two of them.

**Check your understanding.** Prediction: reducing the number of tilings to 1 while keeping 8 tiles per axis would cut active features from 8 to 1 and reintroduce sharp boundary jumps; increasing tilings to 16 would double the active features and smooth the approximation further at the cost of more weights.

**Where this shows up in the code.** `TileCoder(tiles_per_tiling=8, n_tilings=8)` implements this; unit tests `TestTileCoder` assert the binary vector has exactly `n_tilings` ones and that off-grid points are clipped into the boundary tiles.

---

## Exercise 9.5 — Semi-gradient TD(0) on a known two-state MDP

- **Book location:** Section 9.3 (the prediction architecture).
- **Type:** implementation (verified in tests).
- **Problem (paraphrased):** Consider the two-state continuing MDP that stays in
  state A with probability 1 and pays reward 0 (terminal in infinity). Show that
  semi-gradient TD(0) converges. More usefully: verify on a single-step transition
  that the arithmetic matches the book's update exactly.

**Hook.** A tiny MDP lets us check the update arithmetic without any sampling noise hiding mistakes. Convergence is the easy part; exact one-step agreement is the real test.

**Formal bridge.** For one transition \(S\to S'\) with reward \(R\), the semi-gradient TD(0) update on linear features \(\hat v(w,s)=\phi(s)^\top w\) is

\[
w\leftarrow w+\alpha\bigl[R+\gamma\hat v(w,S')-\hat v(w,S)\bigr]\phi(S),
\tag{9.5.1}
\]

and at a terminal successor \(S'\),

\[
\hat v(w,S')=0\quad\Longrightarrow\quad w\leftarrow w+\alpha\bigl[R-\hat v(w,S)\bigr]\phi(S).
\tag{9.5.2}
\]

where \(w\) is the weight vector, \(\alpha\) the step size, \(R\) the reward, \(\gamma\) the discount, \(\phi(S)\) the feature vector of the current state, and \(\delta=R+\gamma\hat v(w,S')-\hat v(w,S)\) the TD error.

**Reasoning chain.** After each step the estimate is nudged toward "what would have been better": the bracket in (9.5.1) is the difference between the immediate reward plus the discounted guess after the transition and the guess before it. With a terminal \(S'\) the bootstrap value is 0, so the target is just \(R\) and (9.5.2) applies.

For linear features and \(\alpha\) in a suitable range, \(w\) converges — on average, in continuing or properly discounted tasks — to the TD fixed point, which is the closest vector to \(v_\pi\) in the feature subspace as weighted by the on-policy distribution: a projection under the \(\mu\)-weighted norm, matching Exercise 9.2. Repeating (9.5.1) enough times therefore settles the estimate into the prediction the data actually supports. In the degenerate two-state MDP that stays in A with probability 1 and pays 0, the target is 0 throughout and the weights decay toward the corresponding fixed point.

**Why it makes sense.** A step size of \(\alpha=0\) leaves (9.5.1) unchanged, so the weights never move — the extreme case that confirms the update is doing all the work. For small positive \(\alpha\), each correction is a fraction of the observed error, so repeated small nudges contract toward the fixed point rather than overshooting; too large an \(\alpha\) and the same correction can overshoot and diverge, which is why convergence needs a suitable range.

**Check your understanding.** Prediction: setting \(\alpha=0\) makes the one-step change exactly zero; doubling a small \(\alpha\) doubles the one-step change but also doubles the risk of overshoot on later steps.

**Where this shows up in the code.** Unit tests `TestSemiGradientTD0` cover the exact one-step arithmetic (including a zero bootstrap for terminals) and that `alpha = 0` leaves the weights unchanged.

---

## How to run the chapter tests

```bash
python -m pytest chapter_09/tests/ -v
```
