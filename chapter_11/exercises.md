# Chapter 11 — Off-policy Methods with Approximation: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd edition). Repo chapter: `chapter_11`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_11` — off-policy prediction with function
  approximation, the deadly triad (function approximation + bootstrapping +
  off-policy), and gradient-corrected objectives (MSPBE).
- **Book chapter:** Chapter 11 — Off-policy Methods with Approximation.
- **Book sections covered by the implementation:**
  - **§11.1/§11.2** The semi-gradient off-policy setting and why it can diverge
    (`SemiGradientOffPolicyAgent`).
  - **Example 11.1** A Baird-style counterexample — implemented exactly on a
    small, enumerable MDP (`baird_*`, `behavior_stationary`,
    `projected_bellman_solution`, `mspbe`, `mspbe_gradient_descent`).
  - **§11.3/§11.4** Gradient-corrected methods / the projected Bellman equation
    (`mspbe_gradient_descent`).
- **Book-identity note:** the local PDF text ships only a summary of Chapter 11
  (no exact example/figure numbers can be verified), so both figures are
  labelled as illustrative demonstrations of the *instability mechanism*, not
  verbatim reproductions of a numbered book figure. The asymmetric hub MDP here
  is an original, exactly-verifiable counterexample of the same phenomenon
  (distribution mismatch makes the expected semi-gradient map indefinite); it
  is not claimed to be the book's Example 11.1 verbatim.
- **Structure choices:** the on-policy distribution in this MDP (uniform, since
  it is symmetric for the target-policy chain) is stable; the off-policy hub
  behavior makes it indefinite — both are computed exactly, not sampled.

---

## Key book equations (plausible citations, S&B 2nd ed.)

- Semi-gradient off-policy update:
  \(w \leftarrow w + \alpha \rho (r + \gamma \phi(S')^\top w - \phi(S)^\top w)\phi(S)\),
  \(\rho = \pi(A|S)/b(A|S)\).
- The expected update direction is \(-H w\) with
  \(H = \sum_s \mu(s) \phi(s)(\phi(s) - \gamma \sum_{s'}P^\pi(s'|s)\phi(s'))^\top\);
  iteration is stable iff the real parts of the eigenvalues of H are positive.
- Projected Bellman equation (linear features):
  \((\Phi^\top D(\mathbb 1 - \gamma P^\pi)\Phi)\,w = \Phi^\top D r\), \(D = \mathrm{diag}(\mu)\).
- MSPBE(w) = \(\|\Pi T(\Phi w) - \Phi w\|^2_D\) with \(\Pi\) the \(D\)-orthogonal
  projector onto \(\mathrm{span}(\Phi)\); quadratic in w.

---

## Exercise 11.1 — Anatomy of the deadly triad

- **Book location:** Section 11.1/11.2 (the deadly triad).
- **Type:** conceptual.
- **Problem (paraphrased):** Name the three ingredients of the deadly triad and
  explain why removing any one of them stabilizes the update.

**Hook.** Three harmless-looking ingredients become dangerous only in combination. Let's name them, then remove each one in turn to see what breaks the instability.

**Formal bridge.** The expected semi-gradient map is governed by

\[
H=\sum_s\mu(s)\,\phi(s)\bigl(\phi(s)-\gamma\sum_{s'}P^\pi(s'|s)\phi(s')\bigr)^\top,
\qquad
w\leftarrow(I-\alpha H)w,
\tag{11.1.1}
\]

which is stable when the real parts of the eigenvalues of \(H\) are positive; here \(\mu\) is the state distribution induced by the behavior policy and importance sampling, \(\phi(s)\) the feature vector, \(P^\pi\) the target-policy transition matrix, \(\gamma\) the discount, and \(\alpha\) the step size.

**Reasoning chain.** The three ingredients are (i) **function approximation** — the value is a parametric \(\hat v_w\), not a table; (ii) **bootstrapping** — targets computed from the current \(\hat v_w\), as in TD; and (iii) **off-policy training** — data collected under a behavior distribution different from the target's. Remove any one and the update in (11.1.1) becomes well-behaved:

- **Tabular + bootstrap + off-policy:** converges (Q-learning) — the tabular projection is exact, so the semi-gradient is the true projected gradient.
- **FA + Monte Carlo + off-policy:** converges toward the least-squares solution of the projected Bellman equation with respect to the *behavior* distribution — stable because MC targets do not depend on \(w\), so the map in (11.1.1) has no self-referential term.
- **FA + bootstrap + on-policy:** converges for linear TD — on-policy the expected update matrix \(H=\Phi^\top D(I-\gamma P^\pi)\Phi\) is positive (semi)definite: it is symmetric in the \(D\)-inner-product norm and satisfies a \(w^\top H w\) VE-like bound.

The instability is that none of the three alone breaks, but the combination can produce an indefinite expected update map: the update chases a moving target computed from its own current answer, and data under a different policy says nothing directly about the target's Bellman operator. If the approximation cannot represent the fixed point well, repeated bootstrapping amplifies errors instead of contracting them.

**Why it makes sense.** Picture a blindfolded student (function approximation) grading their own essay (bootstrapping) using a rubric from a different class (off-policy data). Any one of those flaws is survivable; together, errors feed back with no grounding and can grow. Removing bootstrapping means an external answer key; removing off-policy means the rubric matches the class; removing approximation means the student can memorise the exact answer. Each repair restores a contraction.

**Check your understanding.** Prediction: replacing the learned \(\hat v_w\) bootstrap with a fixed Monte-Carlo target removes the \(w\)-dependence from (11.1.1), so divergence disappears even with a mismatched behavior distribution; restoring bootstrapping while keeping the behaviour on-policy also stays stable.

**Where this shows up in the code.** `TestExpectedUpdateMatrix::test_h_has_negative_real_eigenvalue` verifies numerically that H is indefinite for the hub behavior distribution (`H = Phi.T @ D @ (Phi - gamma P Phi)`; numpy finds a negative real part).

---

## Exercise 11.2 — Why does the symmetric star stay stable but the hub star diverge?

- **Book location:** Example 11.1 (counterexample construction).
- **Type:** analysis of the implemented counterexample (this repo).
- **Problem (paraphrased):** Show with the exact matrices that the behavior
  distribution\(\mu\) being concentrated on outer_1 makes H indefinite, and confirm
  the update matrix has a negative-real-part eigenvalue.

**Hook.** This is the deadly triad made arithmetic. Two behavior distributions, one MDP, and a decisive eigenvalue sign difference.

**Formal bridge.** The expected update under behavior \(b\) with importance-sampling weights is

\[
w\leftarrow(I-\alpha H_\mu)w,
\qquad
H_\mu=\Phi^\top D_\mu(\Phi-\gamma P^\pi\Phi),
\tag{11.2.1}
\]

where \(H_\mu\) is the expected-update matrix under state distribution \(\mu\), \(D_\mu=\mathrm{diag}(\mu)\), \(\Phi\) the feature matrix, \(P^\pi\) the target-policy transition matrix, \(\gamma\) the discount, and \(\alpha\) the step size.

**Reasoning chain.** For the uniform/on-policy \(\mu\), \(H_\mu\) has positive real parts, so for small enough \(\alpha\) all eigenvalues of \(I-\alpha H_\mu\) in (11.2.1) lie inside the unit disc — the iteration contracts. For the hub behavior (\(b(A)=0.02\)), \(\mu_{\text{outer\_1}}\approx0.96\) makes \(H_\mu\) indefinite: its eigenvalues have real parts \(-0.271\) with multiplicity 2. Then \(I-\alpha H_\mu\) has spectral radius \(>1\), so the iterates grow without bound; the norm \(\|w\|\) on a log scale climbs from 1 to about \(10^6\).

The reward-free MDP has true value 0 everywhere — divergence here is not a feature of the environment's rewards but purely of the update map. The off-policy behaviour visits essentially only outer_1, so the learner constantly sees a one-sided slice of the star. Its compensating importance-sampling weights are huge (\(\rho_A=50\)) and act as an unstable amplifier on the components that the slice cannot see.

**Why it makes sense.** Information from one corner of the star is being extrapolated to the whole thing. The enormous \(\rho\) is trying to make up for the missing visits, but amplifying a one-sided sample does not restore the missing directions — it makes them oscillate and grow. The sign matters: a negative real part in \(H_\mu\) means the "correction" points away from the fixed point, so each step makes things worse rather than better. If the behaviour were uniform, every direction would be observed directly and no amplification would be needed.

**Check your understanding.** Prediction: nudging \(b(A)\) upward from 0.02 toward the uniform value would shrink \(\rho_A\) and \(\mu_{\text{outer\_1}}\), moving the negative eigenvalue toward/or past zero; once all real parts are positive again, small \(\alpha\) gives contraction.

**Where this shows up in the code.** Call `behavior_stationary()` and `baird_features()` directly, or use `TestExpectedUpdateMatrix` and `TestBehaviorStationary`. Figure: `figures/example_11_1_divergence.png` (\(\|w\|\) on log scale grows from 1 to ~\(10^6\)).

---

## Exercise 11.3 — The projected Bellman equation and why its solution is finite

- **Book location:** Section 11.3 (projected Bellman equation).
- **Type:** derivation.
- **Problem (paraphrased):** Derive the linear system for the projected Bellman
  fixed point, show that for zero reward it is \(w^* = 0\), and explain why
  the fixed point exists even though the semi-gradient diverges.

**Hook.** The target the algorithm is chasing exists and is perfectly benign. The failure is in the *route*, not the destination.

**Formal bridge.** For linear \(\hat v_w=\Phi w\) and target \(\pi\), the Bellman operator is

\[
(T^\pi\hat v_w)(s)=r(s)+\gamma\sum_{s'}P^\pi(s'|s)\phi(s')^\top w,
\tag{11.3.1}
\]

and projecting it into \(\mathrm{span}(\Phi)\) under the \(D\)-norm gives the linear system

\[
\bigl(\Phi^\top D(\mathbb 1-\gamma P^\pi)\Phi\bigr)\,w=\Phi^\top D\,r,
\qquad D=\mathrm{diag}(\mu).
\tag{11.3.2}
\]

where \(w\) is the weight vector, \(\Phi\) the feature matrix (full column rank here), \(D\) the diagonal matrix of state weights, \(P^\pi\) the target-policy transitions, \(r\) the reward vector, and \(\gamma<1\) the discount.

**Reasoning chain.** Projecting (11.3.1) onto \(\mathrm{span}(\Phi)\) in the \(D\)-norm means applying \(\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D\) to \(r+\gamma P^\pi\Phi w\). Equating that to \(\Phi w\) and cancelling \(\Phi\) (full column rank here) yields (11.3.2).

With \(r=0\) the right side is 0. Since \(I-\gamma P^\pi\) is non-singular for \(\gamma<1\), the unique solution is \(w^*=0\): the true value function is exactly representable, so the projection is trivial. The semi-gradient nevertheless diverges despite this exact and benign fixed point, because semi-gradient is **not** the gradient of MSPBE — it omits the derivative of the bootstrap target, which is precisely the term that makes the map indefinite in Exercise 11.2. The projected Bellman equation is the *correct* fixed-point condition, and it is a clean linear system that can be solved exactly in one shot.

**Why it makes sense.** There is a single, knowable answer (all zeros), and a direct solve finds it immediately. The dangerous algorithm is simply the *wrong* iterative way to get there — it bounces off the wall instead of sliding down to the answer. When the target is representable, no amount of approximation error can justify divergence, so the culprit must be the update rule itself.

**Check your understanding.** Prediction: raising \(\gamma\) toward 1 makes \(I-\gamma P^\pi\) nearly singular, so the direct solve becomes ill-conditioned even though semi-gradient divergence is unchanged in sign; introducing nonzero rewards would move \(w^*\) away from zero but leave the existence of the solution intact.

**Where this shows up in the code.** `projected_bellman_solution()` solves exactly this system and returns ~0 (test `TestProjectedBellman`).

---

## Exercise 11.4 — MSPBE and gradient correction, and why convergence can be slow

- **Book location:** Section 11.4 (gradient-correction methods).
- **Type:** experiment (implemented in this repo).
- **Problem (paraphrased):** Define MSPBE for linear \(\hat v_w\), show it is
  quadratic in w, and analyze the convergence rate of gradient descent on it
  for the counterexample.

**Hook.** When the semi-gradient points the wrong way, minimise an objective that actually has a bottom. The question becomes how fast we can descend it.

**Formal bridge.** With \(\Pi u=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D u\) the \(D\)-orthogonal projector onto \(\mathrm{span}(\Phi)\),

\[
\mathrm{MSPBE}(w)=\|\Pi T(\Phi w)-\Phi w\|_D^2
=w^\top\underbrace{(B-\Phi)^\top D(B-\Phi)}_{Q}w,
\quad
B=\gamma\,\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D P^\pi\Phi.
\tag{11.4.1}
\]

where \(w\) is the weight vector, \(\Pi\) the projector, \(T\) the Bellman operator, \(\Phi\) the feature matrix, \(D\) the state-weight matrix, \(P^\pi\) the target transitions, \(\gamma\) the discount, and \(Q\) the PSD quadratic form.

**Reasoning chain.** \(Q\) is PSD — it is a Gram matrix — so MSPBE is a PSD quadratic minimized at \(w=0\). Gradient descent \(w\leftarrow w-2\alpha Q w\) converges from any start iff \(0<\alpha<1/\lambda_{\max}(Q)\).

For this counterexample the eigenvalues of \(Q\) are approximately \(0.0033\) (×4), \(0.0128\), and \(20\). The step size is capped by the largest eigenvalue, \(20\), leaving the small-eigenvalue directions decaying at rate \(\approx0.9997\) per step — monotone but very slow. This is the conditioning problem, and the motivation for second-order/preconditioned and true gradient-TD methods such as GTD/TDC. Contrast this with the semi-gradient, which does not converge at all.

**Why it makes sense.** MSPBE is a bowl-shaped quadratic, so walking downhill always works; but this bowl is long and narrow, so "downhill" meanders in the flat directions. It converges — slowly — while the unsafe method diverges. The curved direction is exactly the difference between "a correct objective" and "a diverging update": correctness guarantees a direction of descent, conditioning only governs how long the descent takes.

**Check your understanding.** Prediction: if all eigenvalues of \(Q\) were equal (a round bowl), the largest eigenvalue would no longer throttle the step and convergence would be fast; enlarging the gap between \(0.0033\) and \(20\) would make the flat directions slower without changing monotone descent.

**Where this shows up in the code.** `mspbe` computes the value; `mspbe_gradient_descent` performs the descent; figure `figures/example_11_2_mspbe.png` shows the monotone (conditioning-limited) decrease, and `TestMSPBE` checks non-negativity and descent. The conditioning caveat is documented in the figure title/README.

---

## Exercise 11.5 — Emphatic weighting: the other principled fix (sketch)

- **Book location:** Section 11.4 (emphatic temporal-difference methods).
- **Type:** conceptual/theory.
- **Problem (paraphrased):** Explain how emphatic TD fixes semi-gradient off-policy
  instability without a second vector, and when it is preferable to gradient-correction.

**Hook.** There are two honest ways to tame off-policy divergence: fix the objective, or fix the weighting of the samples. Let's compare them.

**Formal bridge.** Emphatic TD reweights each update by an along-the-trajectory emphasis factor of the form

\[
M(s)=\mathbb 1+\gamma\lambda\,\rho(s')\,M(s'),
\tag{11.5.1}
\]

with importance ratio

\[
\rho(s')=\frac{\pi(A_{t+1}\mid S_{t+1})}{b(A_{t+1}\mid S_{t+1})},
\tag{11.5.2}
\]

where \(M(s)\) is the emphasis on state \(s\), \(\rho\) the per-step importance ratio, \(\pi\) the target policy, \(b\) the behavior policy, \(\gamma\) the discount, and \(\lambda\) the trace parameter.

**Reasoning chain.** Emphatic TD uses (11.5.1) to compensate the \(\mu\) versus \(d^\pi\) mismatch exactly, so that the reweighted expected update is a contraction for the same reasons the on-policy case is. Gradient-correction (MSPBE/TDC/GTD) instead minimises the *correct objective* — projected Bellman error — and pays for it with an auxiliary vector/hessian; it is more general and can handle nonlinear approximation.

Emphatic TD needs no second vector, but is only valid for this "emphasis" reweighting form and has high variance when \(\rho\) varies much across steps. Both are honest fixes. The repo implements the MSPBE family exactly and does not claim an unverified TDC recursion from memory.

**Why it makes sense.** Both fixes stop the learner from amplifying its worst blind spot: either by explicitly minimising the right error (MSPBE), or by giving more vote to the rare transitions that actually matter (emphasis). The emphasis recursion accumulates importance backwards along the trajectory, so a rare-but-important path segment receives a larger weight — which is the right direction, since under-sampled states are precisely where the data are least trustworthy.

**Check your understanding.** Prediction: if \(\rho\approx1\) everywhere (behavior close to target), the emphasis in (11.5.1) reduces to a plain discount-and-λ accumulation and both fixes become nearly redundant with ordinary TD; if \(\rho\) swings wildly, Emphatic TD's variance grows while MSPBE's auxiliary-vector estimate stays comparatively steady.

**Where this shows up in the code.** Not implemented (emphatic machinery plus verification would be a chunk of new code; descoped with justification, see README "Not implemented"). The MSPBE path above demonstrates the gradient-correction family.

---

## How to run the chapter tests

```bash
python -m pytest chapter_11/tests/ -v
```
