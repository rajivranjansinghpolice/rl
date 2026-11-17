# Chapter 7 — n-Step Bootstrapping: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd ed., 2018 complete draft). Repo chapter: `chapter_07`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_07` — n-step bootstrapping.
- **Implemented APIs (actual):** `n_step_sarsa`, `tree_backup`, `sarsa_lambda` in
  `chapter_07/solution.py`.
- **Implemented tests (actual):** `TestNStepSarsa`, `TestTreeBackup`, `TestSarsaLambda` in
  `chapter_07/tests/test_solution.py`.
- **Book chapter:** Chapter 7, *n-step Bootstrapping*.
- **Book sections covered by this repo implementation:**
  - §7.2 — n-step Sarsa / on-policy n-step control (Eq. 7.4–7.5), via `n_step_sarsa`.
  - §7.5 — n-step Tree Backup, off-policy control without importance sampling
    (Eq. 7.8/7.11–7.12 style backup recurrences), via `tree_backup`.
  - Eligibility-trace control related to §12.7 / SARSA(λ) (accumulating traces), via
    `sarsa_lambda` (included here for trace-based bootstrapping continuity; not a Chapter 7
    n-step IS algorithm).
- **Not implemented in this repo (theory-only where exercises ask for them):** n-step TD
  prediction API, off-policy n-step prediction/control with importance-sampling ratios
  (ordinary or per-decision/per-reward forms), and Expected Sarsa n-step variants. Exercises
  that request such algorithms are answered with derivations/pseudocode but are labeled
  **theory-only (no repo API)**.

**Key equations (paraphrased notation):**
- n-step return for state values (Eq. 7.1): \(G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})\), with the usual terminal convention.
- n-step TD update (Eq. 7.2): \(V(S_t) \leftarrow V(S_t) + \alpha\,[G_{t:t+n} - V(S_t)]\).
- n-step return/action-value update for Sarsa (Eq. 7.4–7.5): \(G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n Q(S_{t+n},A_{t+n})\), \(Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[G_{t:t+n}-Q(S_t,A_t)]\).
- Off-policy n-step importance-sampling ratio (Eq. 7.9/7.10): \(\rho_{t+1:t+n} = \prod_{k=t+1}^{\min(t+n,T-1)} \frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}\).
- Off-policy n-step state-value update (Eq. 7.7/7.10): \(V(S_t)\leftarrow V(S_t)+\alpha\,\rho_{t+1:t+n-1}[G_{t:t+n}-V(S_t)]\) (ratio horizon per the book’s convention).
- Off-policy n-step action-value/Sarsa form (Eq. 7.11): \(Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,\rho_{t+1:t+n}[G_{t:t+n}-Q(S_t,A_t)]\).
- Per-decision/Expected-Sarsa control-variate ideas appear in §7.4 (per-decision methods).

Citations above are to Sutton & Barto (2nd ed.), Chapter 7 (§/Eq. numbers as used in the
standard 2018 draft). Page numbers vary by PDF build; section/equation labels are the stable
references.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2nd ed.), Chapter 7. Exercise statements are
  **paraphrased** (no long verbatim quotes).
- **Cross-check:** public solution discussions were used only as a sanity check; all
  derivations/pseudocode here are written from scratch.
- **Repo truth rule:** coding pointers and “implemented” claims match `chapter_07/solution.py`
  and `chapter_07/tests/test_solution.py` exactly.

## How to Use This Document

- Each entry gives **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem (paraphrase)**, **Solution**, and **Coding pointer / status**.
- “Theory-only” means no matching function exists in `solution.py`; do not expect a runnable
  API for it in this chapter.

---

## Exercises

### Exercise 7.1 — n-step error as a sum of TD errors

- **Book location:** §7.1 (n-step TD prediction), Eq. 7.1–7.2; cf. Eq. 6.6.
- **Type:** Math / Proof.
- **Problem (paraphrase):** Assuming the value estimates are held fixed over the interval, show
  that the n-step TD error \(G_{t:t+n}-V(S_t)\) can be written as a sum of one-step TD errors.
- **Solution.** With fixed \(V\), define the one-step TD error
  \(\delta_k = R_{k+1} + \gamma V(S_{k+1}) - V(S_k)\) for \(k < T-1\) and treat \(V\) as 0 at
  terminals as usual. From Eq. 7.1,
  \(G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})\). Substitute
  \(R_{t+k+1} = \delta_{t+k} + V(S_{t+k}) - \gamma V(S_{t+k+1})\):
  \[
  G_{t:t+n}-V(S_t)
  = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}
    + \sum_{k=0}^{n-1} \gamma^k\bigl(V(S_{t+k})-\gamma V(S_{t+k+1})\bigr)
    + \gamma^n V(S_{t+n}) - V(S_t).
  \]
  The middle telescoping sum collapses to \(V(S_t)-\gamma^n V(S_{t+n})\), so all value terms
  cancel and
  \[
  G_{t:t+n}-V(S_t)=\sum_{k=0}^{n-1} \gamma^k \delta_{t+k}\,\quad (\text{fixed }V,\ \text{Eq. 7.1–7.2}).
  \]
- **Coding pointer / status:** **Theory-only (no repo API).** `solution.py` does not implement
  n-step TD prediction/V updates; implemented control methods are `n_step_sarsa`, `tree_backup`,
  `sarsa_lambda`.

### Exercise 7.2 — Is the sum-of-TD-errors form better when values change?

- **Book location:** §7.1, discussion following Eq. 7.2.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** If values change online, replacing the n-step error in Eq. 7.2 by the
  fixed-\(V\) sum \(\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}\) (using current/changing estimates inside
  each \(\delta\)) yields a different algorithm. Would it be better or worse? Suggest a small
  experiment.
- **Solution.** The identity in Exercise 7.1 is exact only when \(V\) is frozen over the window.
  With online updates, the “sum of TD errors” form uses bootstrap targets that have already been
  modified by earlier updates in the same episode; it no longer equals the n-step error of a
  single target and can introduce bias/credit misallocation relative to the standard n-step
  backup. It is not generally better; it is a different semi-online approximation.
  **Experiment (theory proposal):** on a random-walk prediction task (e.g. the 19-state walk of
  §7.1), compare RMS value error vs. episodes/steps for (a) standard n-step TD (Eq. 7.2) and
  (b) the sum-of-changing-\(\delta\) variant, sweeping \(n\) and \(\alpha\) with multiple seeds.
- **Coding pointer / status:** **Theory-only (no repo API).** No n-step prediction or TD-error-sum
  variant is implemented; related control code is `n_step_sarsa` (on-policy, Eq. 7.5 form).

### Exercise 7.3 — Why a larger random walk task?

- **Book location:** §7.1, Example 7.1 (19-state random walk).
- **Type:** Theory / Computation (qualitative reasoning).
- **Problem (paraphrase):** Why use a larger random walk (19 states) instead of a small one (5
  states)? Would a smaller walk shift the best \(n\)? Did changing the left terminal outcome in
  the larger walk affect the best \(n\)?
- **Solution.** A larger chain makes delayed consequences and the bias–variance tradeoff across
  \(n\) visible: with only 5 states episodes are very short, so most \(n\)-step returns collapse
  to near-MC/full-episode targets and intermediate \(n\) offer little separation. A smaller walk
  would compress horizons and tend to shift the apparent advantage toward smaller \(n\) (or make
  differences hard to detect). Changing the left terminal outcome (e.g. 0 vs. -1 in variants of
  the task) changes target values and error magnitudes but, for the same dynamics/horizon, does
  not fundamentally change which \(n\) best balances bootstrapping bias and sampling variance;
  the optimal \(n\) is driven mainly by timescale/horizon and step size rather than by a uniform
  shift in terminal reward sign/magnitude.
- **Coding pointer / status:** **Theory-only (no repo API).** This repo’s control experiments run
  on `chapter_06.solution.RandomWalkEnv` via `n_step_sarsa` / `tree_backup` / `sarsa_lambda`; no
  19-state prediction benchmark script is included.

### Exercise 7.4 — Pseudocode for off-policy n-step state-value prediction

- **Book location:** §7.3 (n-step off-policy learning), Eq. 7.1, 7.7, 7.9–7.10.
- **Type:** Programming / Theory (pseudocode).
- **Problem (paraphrase):** Give pseudocode for off-policy n-step state-value prediction using the
  off-policy n-step return and TD update.
- **Solution (theory-only pseudocode).** Behavior policy \(b\) (coverage over \(\pi\)), target
  \(\pi\), step size \(\alpha\in(0,1]\), \(n\ge 1\):

```text
Initialize V(s) arbitrarily for nonterminals, V(terminal)=0
Loop for each episode:
  Choose S_0 (nonterminal), T <- ∞, t <- 0
  While t - n + 1 < T (or until updates done):
    If t < T:
      Take A_t ~ b(.|S_t); observe R_{t+1}, S_{t+1}
      If S_{t+1} terminal: T <- t+1
    tau <- t - n + 1
    If tau >= 0:
      rho <- Π_{i=tau+1}^{min(tau+n-1, T-1)} π(A_i|S_i) / b(A_i|S_i)   (Eq. 7.9/7.10)
      G <- Σ_{k=0}^{n-1} γ^k R_{tau+1+k} (truncated at T)
      If tau+n < T: G <- G + γ^n V(S_{tau+n})                        (Eq. 7.1)
      V(S_tau) <- V(S_tau) + α * rho * (G - V(S_tau))                (Eq. 7.7/7.10)
    t <- t+1
```

- **Coding pointer / status:** **Theory-only (no repo API).** No off-policy importance-sampling
  n-step prediction function exists in `solution.py`. Implemented off-policy-without-IS control is
  `tree_backup` (different algorithm/target).

### Exercise 7.5 — Pseudocode for off-policy n-step action-value prediction (Sarsa / Expected Sarsa)

- **Book location:** §7.3 (off-policy n-step action values), Eq. 7.4–7.5, 7.9, 7.11; Expected
  Sarsa bootstrapping idea as in Chapter 6/7 discussion.
- **Type:** Programming / Theory (pseudocode).
- **Problem (paraphrase):** Give pseudocode for off-policy n-step action-value prediction using
  the off-policy action-value return/update; specify Sarsa and Expected Sarsa bootstrap targets.
- **Solution (theory-only pseudocode).** Maintain \(Q\); use behavior \(b\), target \(\pi\).

**Sarsa-style bootstrap (action sampled/known at \(S_{t+n}\)):**
```text
G_{t:t+n} <- Σ_{k=0}^{n-1} γ^k R_{t+1+k} + γ^n Q(S_{t+n}, A_{t+n})   if t+n < T else truncated sum
rho <- Π_{i=t+1}^{min(t+n, T-1)} π(A_i|S_i) / b(A_i|S_i)            (Eq. 7.11 ratio)
Q(S_t,A_t) <- Q(S_t,A_t) + α * rho * (G_{t:t+n} - Q(S_t,A_t))
```

**Expected Sarsa-style bootstrap (expectation under target at the bootstrap step):**
```text
G_{t:t+n} <- Σ_{k=0}^{n-1} γ^k R_{t+1+k} + γ^n Σ_a π(a|S_{t+n}) Q(S_{t+n},a)   if t+n < T
rho <- Π_{i=t+1}^{min(t+n-1, T-1)} π(A_i|S_i) / b(A_i|S_i)   (no A_{t+n} factor; bootstrap is expectation)
Q(S_t,A_t) <- Q(S_t,A_t) + α * rho * (G_{t:t+n} - Q(S_t,A_t))
```
(The ratio horizon differs because the Expected Sarsa target does not require sampling
\(A_{t+n}\) from \(\pi\); this matches the per-decision/control-variate reasoning of §7.4.)

- **Coding pointer / status:** **Theory-only (no repo API).** No off-policy IS action-value or
  Expected Sarsa n-step implementation exists in `solution.py`. On-policy n-step Sarsa control is
  implemented as `n_step_sarsa` (no \(\rho\) ratios, behavior=target in effect).

### Exercise 7.6 — Off-policy n-step state-value return as sum of state-based TD errors

- **Book location:** §7.3–7.4, Eq. 7.7/7.10 and TD-error form (cf. Eq. 6.5).
- **Type:** Math / Proof.
- **Problem (paraphrase):** Show that, if \(V\) is fixed, the off-policy n-step error can be
  written compactly via state-based TD errors (with the appropriate importance-sampling weights
  as in the per-decision decomposition).
- **Solution.** With fixed \(V\), Exercise 7.1 gives the unweighted identity
  \(G_{t:t+n}-V(S_t)=\sum_{k=0}^{n-1}\gamma^k \delta_{t+k}\), \(\delta_k=R_{k+1}+\gamma V(S_{k+1})-V(S_k)\).
  The off-policy update (Eq. 7.7/7.10) multiplies this error by \(\rho_{t+1:t+n-1}\), hence
  \(\rho_{t+1:t+n-1}(G_{t:t+n}-V(S_t))=\rho_{t+1:t+n-1}\sum_{k=0}^{n-1}\gamma^k \delta_{t+k}\).
  In the per-decision/per-reward view (§7.4), each reward/TD term can instead be weighted by the
  ratio up to the step where it is realized, yielding an equivalent-in-expectation decomposition
  that is the off-policy state-value analogue of Eq. 6.5/6.6: a sum of (importance-weighted)
  state-based TD errors under fixed estimates.
- **Coding pointer / status:** **Theory-only (no repo API).**

### Exercise 7.7 — Off-policy action version and Expected Sarsa TD errors

- **Book location:** §7.3–7.4, Eq. 7.11; Expected Sarsa TD error (cf. Eq. 6.9).
- **Type:** Math / Proof.
- **Problem (paraphrase):** Repeat the decomposition for the off-policy n-step action-value
  return, expressing it in terms of Expected Sarsa–style TD errors when \(Q\) is fixed.
- **Solution.** For fixed \(Q\), define the Expected Sarsa TD error
  \(\delta^E_t = R_{t+1} + \gamma \sum_a \pi(a\mid S_{t+1})Q(S_{t+1},a) - Q(S_t,A_t)\) (and the
  analogous form when a sampled next action is used for Sarsa). Telescoping the n-step
  action-value return \(G_{t:t+n}\) (Eq. 7.4/7.11 target form) yields
  \(G_{t:t+n}-Q(S_t,A_t)=\sum_{k=0}^{n-1}\gamma^k \tilde{\delta}_{t+k}\) for fixed \(Q\), where
  each \(\tilde{\delta}\) is the corresponding one-step (Expected) Sarsa TD error; multiplying by
  the appropriate \(\rho\) factors gives the off-policy weighted/per-decision decomposition
  analogous to Exercise 7.6 but in action-value space.
- **Coding pointer / status:** **Theory-only (no repo API).**

### Exercise 7.8 — Data efficiency: per-decision/off-policy n-step vs simpler off-policy

- **Book location:** §7.4 (per-decision methods / control variates), comparing per-decision
  estimators to ordinary importance-sampling n-step updates.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** Devise a small off-policy prediction problem showing that the more
  refined per-decision/off-policy algorithm is more data-efficient (lower variance / faster
  learning) than a simpler off-policy algorithm using coarser importance weighting.
- **Solution.** **Proposed experiment (theory-only):** choose a small episodic MDP with known
  \(v_\pi\), a behavior policy \(b\) that wanders but covers \(\pi\)’s actions, and compare
  learning curves of (a) ordinary off-policy n-step TD with full-product \(\rho\) over the window
  (Eq. 7.7/7.10 style) versus (b) a per-decision/per-reward variant that weights each reward/TD
  term only by ratios up to when it occurs (§7.4). Measure MSE \(\|V - v_\pi\|^2\) vs. episodes
  over many seeds. Per-decision weighting typically reduces variance because irrelevant future
  action mismatches do not downweight already-realized rewards, improving data efficiency.
- **Coding pointer / status:** **Theory-only (no repo API).** Neither off-policy IS variant is
  implemented; `tree_backup` is off-policy *without* IS (a different approach) and is not the
  algorithm pair requested by this exercise.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section / topic | Repo code (actual) | Repo tests (actual) | Status |
|---|---|---|---|---|
| 7.2, 7.3 (control intuition; not prediction APIs) | §7.2 n-step Sarsa control | `n_step_sarsa` | `TestNStepSarsa` | Implemented (on-policy control) |
| 7.4–7.8 (off-policy IS / per-decision / Expected Sarsa) | §7.3–7.4 theory | — | — | **Theory-only (no repo API)** |
| Off-policy without IS (related concept) | §7.5 Tree Backup | `tree_backup` | `TestTreeBackup` | Implemented (different from 7.4–7.8) |
| Trace-based control (continuity) | SARSA(λ), accumulating traces | `sarsa_lambda` | `TestSarsaLambda` | Implemented |

- **Full chapter command:** `python -m pytest chapter_07/tests/ -v`
- **Lint:** `ruff check chapter_07/`
- No standalone `exercise_*.py` scripts exist for this chapter. Do not claim `td0_prediction` or
  off-policy importance-sampling APIs; they are not present in `chapter_07/solution.py`.

---

## Verification and Quality Notes

- Derivations (7.1, 7.6, 7.7) are algebraic/telescoping arguments under the stated fixed-estimate
  assumptions, consistent with Chapter 7 equations.
- 7.4–7.5 provide explicit theory-only pseudocode because the requested IS-based APIs are not in
  this repo; 7.8 specifies a reproducible comparison experiment without asserting an implementation.
- Exercise statements are paraphrased; citations use stable §/Eq. labels.

*End of Chapter 7 exercises document (`chapter_07/exercises.md`).*
