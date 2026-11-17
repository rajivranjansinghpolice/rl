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

**Solution.** The three ingredients are (i) **function approximation** (the value
is a parametric \(\hat v_w\), not a table), (ii) **bootstrapping** (targets computed
from the current \(\hat v_w\), as in TD), and (iii) **off-policy training** (data
collected under a behavior distribution different from the target's). Remove any one:

- **Tabular + bootstrap + off-policy:** converges (Q-learning) — the tabular
  projection is exact, so the semi-gradient is the true projected gradient.
- **FA + Monte Carlo + off-policy:** converges toward the least-squares solution of
  the projected Bellman equation with respect to the *behavior* distribution —
  stable because MC targets don't depend on \(w\).
- **FA + bootstrap + on-policy:** converges for linear TD — on-policy the expected
  update matrix \(H = \Phi^\top D (I - \gamma P^\pi)\Phi\) is positive (semi)definite
  (it is symmetric in the \(D\)-inner-product norm and satisfies
  \(w^\top H w = \mathrm{VE}\)-like bound).

The instability is that **none of the three alone breaks**, and the combination
can produce an indefinite expected update map.

**Student-friendly explanation.** The update chases a moving target computed from
its own current answer, and the data under a different policy tells you nothing
directly about the target's Bellman operator. If the approximation can't represent
the fixed point well, repeated bootstrapping amplifies errors instead of
contracting them.

- **Coding pointer.** `TestExpectedUpdateMatrix::test_h_has_negative_real_eigenvalue`
  verifies numerically that H is indefinite for the hub behavior distribution
  (`H = Phi.T @ D @ (Phi - gamma P Phi)`; numpy finds a negative real part).

---

## Exercise 11.2 — Why does the symmetric star stay stable but the hub star diverge?

- **Book location:** Example 11.1 (counterexample construction).
- **Type:** analysis of the implemented counterexample (this repo).
- **Problem (paraphrased):** Show with the exact matrices that the behavior
  distribution\(\mu\) being concentrated on outer_1 makes H indefinite, and confirm
  the update matrix has a negative-real-part eigenvalue.

**Solution.** The expected update under behavior b with IS weights implements
\(w \leftarrow (I - \alpha H_\mu)w\), \(H_\mu = \Phi^\top D_\mu(\Phi - \gamma P^\pi\Phi)\).
For the uniform/on-policy \(\mu\), \(H_\mu\) has positive real parts, so for small
enough \(\alpha\) all eigenvalues of \(I - \alpha H_\mu\) are inside the unit disc —
the iteration contracts. For the hub behavior (b(A) = 0.02), \(\mu_{outer\_1} \approx 0.96\)
makes \(H_\mu\) indefinite (eigenvalues with real parts \(-0.271\) with multiplicity 2),
so \(I - \alpha H_\mu\) has spectral radius \(> 1\) and the iterates grow without bound.
The reward-free MDP has the true value 0 everywhere — divergence here is not a
feature of the environment's rewards but purely of the update map.

**Student-friendly explanation.** The off-policy policy visits essentially only
outer_1, so the learner constantly sees a one-sided slice of the star. Its
compensating IS weights are huge (\(\rho_A = 50\)) and act as a random, unstable
amplifier on the components that the slice cannot see.

- **Coding pointer.** Run `python -m pytest chapter_11/tests/ -v`
  (`TestExpectedUpdateMatrix`, `TestBehaviorStationary`), or call
  `behavior_stationary()` and `baird_features()` directly. Figure:
  `figures/fig_11_divergence.png` (||w|| on log scale grows from 1 to ~10^6).

---

## Exercise 11.3 — The projected Bellman equation and why its solution is finite

- **Book location:** Section 11.3 (projected Bellman equation).
- **Type:** derivation.
- **Problem (paraphrased):** Derive the linear system for the projected Bellman
  fixed point, show that for zero reward it is \(w^* = 0\), and explain why
  the fixed point exists even though the semi-gradient diverges.

**Solution.** For linear \(\hat v_w = \Phi w\) and target \(\pi\),
\((T^\pi \hat v_w)(s) = r(s) + \gamma \sum_{s'} P^\pi(s'|s)\phi(s')^\top w\).
The projection onto \(\mathrm{span}(\Phi)\) in the \(D\)-norm gives
\(\Phi w = \Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D(r + \gamma P^\pi\Phi w)\).
Canceling \(\Phi\) (\(\Phi\) full column rank here) yields the linear system

\[
(\Phi^\top D(\mathbb 1 - \gamma P^\pi)\Phi)\, w = \Phi^\top D\, r .
\]

With \(r = 0\) the right side is 0, and since \(I - \gamma P^\pi\) is non-singular
for \(\gamma < 1\), \(w^* = 0\) — the true value function is exactly representable.
The semi-gradient diverges despite the exact and benign fixed point because
semi-gradient is **not** the gradient of MSPBE; it omits the derivative of the
bootstrap target. The projected Bellman equation is the *correct* fixed-point
condition, and it is a clean linear system — computable exactly.

**Student-friendly explanation.** There is a single, knowable answer (all zeros),
and a direct solve finds it in one shot. The dangerous algorithm is simply the
*wrong* iterative way to get there — it bounces off the wall instead of sliding
down to the answer.

- **Coding pointer.** `projected_bellman_solution()` solves exactly this system
  and returns ~0 (test `TestProjectedBellman`).

---

## Exercise 11.4 — MSPBE and gradient correction, and why convergence can be slow

- **Book location:** Section 11.4 (gradient-correction methods).
- **Type:** experiment (implemented in this repo).
- **Problem (paraphrased):** Define MSPBE for linear \(\hat v_w\), show it is
  quadratic in w, and analyze the convergence rate of gradient descent on it
  for the counterexample.

**Solution.** With \(\Pi u = \Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D u\) the
\(D\)-orthogonal projector onto \(\mathrm{span}(\Phi)\),

\[
\text{MSPBE}(w) = \|\Pi T(\Phi w) - \Phi w\|^2_D = w^\top \underbrace{(B-\Phi)^\top D (B-\Phi)}_{Q} w,
\quad B = \gamma\, \Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D P^\pi \Phi .
\]

\(Q\) is PSD (it is a Gram matrix), so MSPBE is a PSD quadratic minimized at \(w=0\).
Gradient descent w \(\leftarrow w - 2\alpha Q w\) converges from any start iff
\(0 < \alpha < 1/\lambda_{\max}(Q)\). Here the eigenvalues of \(Q\) are
\(\approx 0.0033\) (×4), \(0.0128\), and \(20\); the step size is capped by the
largest one, leaving the small-eigenvalue directions decaying at rate
\(\approx 0.9997\) per step — monotone but very slow (this is the conditioning
problem and the motivation for second-order/preconditioned and true gradient-TD
methods like GTD/TDC). Contrast with the semi-gradient, which does not converge
at all.

**Student-friendly explanation.** MSPBE is a bowl-shaped quadratic, so walking
downhill always works; but the bowl is long and narrow, so "downhill" meanders
in the flat directions. It converges — slowly — while the unsafe method
diverges. The curved direction is exactly the difference between "a correct
objective" and "a diverging update."

- **Coding pointer.** `mspbe` computes the value; `mspbe_gradient_descent`
  performs the descent; figure `figures/fig_11_mspbe.png` shows the monotone
  (conditioning-limited) decrease, and `TestMSPBE` checks non-negativity and
  descent. The conditioning caveat is documented in the figure title/README.

---

## Exercise 11.5 — Emphatic weighting: the other principled fix (sketch)

- **Book location:** Section 11.4 (emphatic temporal-difference methods).
- **Type:** conceptual/theory.
- **Problem (paraphrased):** Explain how emphatic TD fixes semi-gradient off-policy
  instability without a second vector, and when it is preferable to gradient-correction.

**Solution.** Emphatic TD reweights each update by the *emphasis*
\(M(s) = \mathbb 1 + \gamma \lambda \rho(s')\, M(s')\)-style along-the-trajectory
factor, which compensates the \(\mu\) vs. \(d^\pi\) mismatch exactly so that the
reweighted expected update is a contraction for the same reasons the on-policy
case is. Gradient-correction (MSPBE/TDC/GTD) instead minimizes the *correct
objective* (projected Bellman error) and pays for it with an auxiliary vector/hessian;
it is more general (nonlinear). Emphatic needs no second vector but is only valid
for this "emphasis" reweighting form and has high variance when \(\rho\) varies
much across steps. Both are honest fixes; the repo implements the MSPBE family
exactly and does not claim an unverified TDC recursion from memory.

**Student-friendly explanation.** Both fixes stop the learner from amplifying
its worst blind spot: either by explicitly minimizing the right error (MSPBE), or
by giving more vote to the rare transitions that actually matter (emphasis).

- **Coding pointer.** Not implemented (emphatic machinery + verification would
  be a chunk of new code; descoped with justification, see README "Not
  implemented"). The MSPBE path above demonstrates the gradient-correction family.

---

## How to run the chapter tests

```bash
python -m pytest chapter_11/tests/ -v
```
