# Chapter 7 — n-Step Bootstrapping: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd ed., 2018 complete draft). Repo chapter: `chapter_07`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_07` — n-step bootstrapping.
- **Implemented APIs (actual):** `n_step_td_prediction`, `run_n_step_prediction_experiment`,
  `n_step_sarsa`, `tree_backup`, `sarsa_lambda` in `chapter_07/solution.py`.
- **Implemented tests (actual):** `TestNStepTDPrediction`, `TestNStepSarsa`, `TestTreeBackup`,
  `TestSarsaLambda` in `chapter_07/tests/test_solution.py`.
- **Book chapter:** Chapter 7, *n-step Bootstrapping*.
- **Book sections covered by this repo implementation:**
  - §7.1 — n-step TD prediction (Eq. 7.1–7.2) and Example 7.1 / Figure 7.2: prediction on the
    19-state random walk (A..S, left outcome −1, right +1, values initialized to 0).
    `n_step_td_prediction` runs the box; `run_n_step_prediction_experiment` reproduces
    Figure 7.2 (error vs. α for n = 1..512, averaged over the first 10 episodes and
    repetitions, same walks for every setting).
  - §7.2 — n-step Sarsa / on-policy n-step control (Eq. 7.4–7.5), via `n_step_sarsa`.
  - §7.5 — n-step Tree Backup, off-policy control without importance sampling
    (backup recurrences), via `tree_backup`.
  - Eligibility-trace control related to §12.2 / SARSA(λ) (accumulating traces), via
    `sarsa_lambda` (included here for trace-based bootstrapping continuity; not a Chapter 7
    n-step IS algorithm).
- **Not implemented in this repo (theory-only where exercises ask for them):** off-policy n-step
  prediction/control with importance-sampling ratios (ordinary or per-decision/per-reward forms),
  and Expected Sarsa n-step variants. Exercises that request such algorithms are answered with
  derivations/pseudocode but are labeled **theory-only (no repo API)**.

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
- **Source and scope:** Sutton & Barto (2nd ed.) Chapter 7; exercise statements are paraphrased and derivations are worked from the book's equations, with section/equation/page citations for traceability.
- **Repo truth rule:** coding pointers and “implemented” claims match `chapter_07/solution.py`
  and `chapter_07/tests/test_solution.py` exactly.

## How to Use This Document

- Each entry gives **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem (paraphrase)**, and a worked answer in the five-part house style:
  a conversational **Hook**, a **Formal bridge** that states the key result as a numbered
  equation with a glossary of its symbols, a **Reasoning chain** that derives it step by step,
  a **Why it makes sense** intuition check at the extremes, and a **Check your understanding**
  prediction-style self-check; a short **Where this shows up in the code / status** pointer
  follows where relevant.
- “Theory-only” means no matching function exists in `solution.py`; do not expect a runnable
  API for it in this chapter.

---

## Exercises

### Exercise 7.1 — n-step error as a sum of TD errors

- **Book location:** §7.1 (n-step TD prediction), Eq. 7.1–7.2; cf. Eq. 6.6.
- **Type:** Math / Proof.
- **Problem (paraphrase):** Assuming the value estimates are held fixed over the interval, show
  that the n-step TD error \(G_{t:t+n}-V(S_t)\) can be written as a sum of one-step TD errors.

**Hook.** Let's pull the n-step return apart into the one-step signals it is built from. The trick only works while the value function is frozen, and seeing exactly where that assumption enters is the point of the exercise.

**Formal bridge.** With \(V\) fixed, the one-step TD error is defined by

\[
\delta_k = R_{k+1}+\gamma V(S_{k+1})-V(S_k),
\tag{7.1.1}
\]

with \(V=0\) at terminals, and the n-step return is

\[
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^k R_{t+k+1}+\gamma^n V(S_{t+n}).
\tag{7.1.2}
\]

where \(R_{k+1}\) is the reward after step \(k\), \(S_k\) is the state at step \(k\), \(\gamma\) is the discount, \(V\) is the held-fixed value estimate, and \(n\) is the backup horizon.

**Reasoning chain.** Start from (7.1.2) and solve (7.1.1) for the reward: \(R_{t+k+1}=\delta_{t+k}+V(S_{t+k})-\gamma V(S_{t+k+1})\). Substituting that expression for every reward in the window gives

\[
G_{t:t+n}-V(S_t)
=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}
+\sum_{k=0}^{n-1}\gamma^k\bigl(V(S_{t+k})-\gamma V(S_{t+k+1})\bigr)
+\gamma^n V(S_{t+n})-V(S_t).
\tag{7.1.3}
\]

The second sum in (7.1.3) telescopes: writing it out, \(V(S_t)-\gamma V(S_{t+1})+\gamma V(S_{t+1})-\gamma^2 V(S_{t+2})+\cdots\), every interior value cancels, leaving \(V(S_t)-\gamma^n V(S_{t+n})\). Those two surviving terms cancel against the final two terms of (7.1.3), so

\[
G_{t:t+n}-V(S_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}.
\tag{7.1.4}
\]

This is the fixed-\(V\) identity of Eq. 7.1–7.2: the n-step error is just the discounted sum of the one-step errors along the window.

**Why it makes sense.** Check the extreme \(n=1\): the sum in (7.1.4) has a single term \(\delta_t\), which is exactly the one-step TD error. At the other extreme, large \(n\) reaching a terminal, the bootstrap value vanishes and we recover the full-return error — still a sum of all the one-step errors along the episode. The cancellation works because each reward is accounted for twice in a controlled way: once as an error and once as a value difference, and the value differences form a chain with no loose ends. If \(V\) were changing between steps, those interior terms would no longer be the same number at both ends of the chain, and the cancellation would fail.

**Check your understanding.** Prediction: if the estimates are updated after every step inside the window, the interior value differences no longer match, so (7.1.4) stops being exact; the discrepancy grows with \(n\), and vanishes when \(n=1\).

**Where this shows up in the code / status.** **Theory (identity); exercised by implementation.** `n_step_td_prediction` implements the standard (changing-\(V\)) form of the box in §7.1; the identity above is verified for the fixed-\(V\) case in the derivation. See `TestNStepTDPrediction` and the Figure 7.2 experiment (`run_n_step_prediction_experiment`).

### Exercise 7.2 — Is the sum-of-TD-errors form better when values change?

- **Book location:** §7.1, discussion following Eq. 7.2.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** If values change online, replacing the n-step error in Eq. 7.2 by the
  fixed-\(V\) sum \(\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}\) (using current/changing estimates inside
  each \(\delta\)) yields a different algorithm. Would it be better or worse? Suggest a small
  experiment.

**Hook.** Here the trap is treating an identity derived for frozen values as if it survived online learning. It does not, and the reason tells us what kind of algorithm the substitution actually produces.

**Formal bridge.** The identity from Exercise 7.1 holds only for fixed estimates:

\[
G_{t:t+n}-V(S_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k},\qquad V\ \text{frozen}.
\tag{7.2.1}
\]

The proposed online replacement instead uses the estimates as they stand at the moment of summation,

\[
V(S_t)\leftarrow V(S_t)+\alpha\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}^{(\text{now})},
\tag{7.2.2}
\]

where \(G_{t:t+n}\) is the standard n-step return target, \(\delta_k\) is a one-step TD error, \(V\) is the value estimate, \(\alpha\) is the step size, and the superscript “now” marks that every \(\delta\) is computed with estimates already altered by earlier updates in the episode.

**Reasoning chain.** In (7.2.2) the bootstrap targets inside each \(\delta_{t+k}^{(\text{now})}\) have already been modified by earlier updates in the same episode. The value differences that made (7.2.1) telescope are therefore no longer evaluated at a single consistent \(V\), and the sum no longer equals the n-step error of any single, coherent target. Consequently this is not generally an improvement: it is a different semi-online approximation that can introduce bias and misallocate credit relative to the standard n-step backup of Eq. 7.2.

The honest way to judge it is a small controlled experiment. On a random-walk prediction task — for instance the 19-state walk of §7.1 — compare RMS value error against episodes (or steps) for (a) standard n-step TD using Eq. 7.2 and (b) the sum-of-changing-\(\delta\) variant of (7.2.2). Sweep \(n\) and \(\alpha\), and average over multiple seeds so that a single lucky walk does not decide the comparison.

**Why it makes sense.** At \(n=1\), (7.2.2) reduces to the ordinary one-step TD update, so no difference can appear; the divergence between the two algorithms only has room to grow as \(n\) increases, because that is where more stale-estimate errors accumulate. Equivalently, the standard n-step return asks "what is my error against one target?", whereas the sum form adds several targets computed from different versions of \(V\), which is not the same question.

**Check your understanding.** Prediction: with a very small step size the estimates barely move during a window, so (7.2.2) behaves almost like (7.2.1) and the learning curves nearly coincide; raising \(\alpha\) or \(n\) makes the two diverge and typically penalises the sum form.

**Where this shows up in the code / status.** **Implemented (prediction API).** `n_step_td_prediction` in `chapter_07/solution.py` implements the online variant exactly as written; the sum-of-changing-δ variant is not implemented (it is not the algorithm the book's box describes). Related control code: `n_step_sarsa` (on-policy, Eq. 7.5 form); tests in `TestNStepTDPrediction`.

### Exercise 7.3 — Why a larger random walk task?

- **Book location:** §7.1, Example 7.1 (19-state random walk).
- **Type:** Theory / Computation (qualitative reasoning).
- **Problem (paraphrase):** Why use a larger random walk (19 states) instead of a small one (5
  states)? Would a smaller walk shift the best \(n\)? Did changing the left terminal outcome in
  the larger walk affect the best \(n\)?

**Hook.** Before sweeping \(n\), make sure the task can actually distinguish the settings. A walk that is too short collapses every interesting horizon into the same near-Monte-Carlo backup.

**Formal bridge.** What matters is the horizon relative to the remaining episode length:

\[
G_{t:t+n}=\sum_{k=0}^{\min(n,T-t)-1}\gamma^k R_{t+k+1}
+\gamma^{\min(n,T-t)}V(S_{T}),
\tag{7.3.1}
\]

where \(T\) is the terminal time, \(V(S_T)=0\), \(n\) is the backup horizon, and the effective horizon is \(\min(n,T-t)\).

**Reasoning chain.** With only 5 states, episodes are very short, so for most \(n\) the effective horizon \(\min(n,T-t)\) in (7.3.1) is already the whole remaining episode. The n-step returns collapse to near-MC/full-episode targets and intermediate values of \(n\) offer little separation; a smaller walk compresses horizons and tends to shift the apparent advantage toward smaller \(n\), or makes the differences hard to detect at all.

The 19-state walk makes delayed consequences visible and gives the bias–variance tradeoff across \(n\) room to show up: small \(n\) bootstraps a lot (bias), large \(n\) waits a long time for real rewards (variance), and intermediate \(n\) can beat both extremes. Changing the left terminal outcome — for example 0 versus \(-1\) in variants of the task — changes the target values and the error magnitudes, but for the same dynamics and horizon it does not fundamentally change which \(n\) best balances bootstrapping bias against sampling variance. The best \(n\) is driven mainly by timescale/horizon and step size, not by a uniform shift in the terminal reward's sign or magnitude.

**Why it makes sense.** Picture \(n\) as a reach: once your arm is longer than the corridor, extending it further changes nothing. A five-state corridor is short enough that nearly every arm length reaches the end, so every setting looks like Monte Carlo. Widening the corridor separates the reaches — and moving the goalposts changes what you find there, not how far you must reach.

**Check your understanding.** Prediction: lengthening the walk to 39 states would push the best \(n\) upward, because more of the useful reward information arrives later; changing only the left-end reward would rescale the error curves but leave the location of their minimum roughly put.

**Where this shows up in the code / status.** **Implemented (Example 7.1).** The 19-state walk is `RandomWalkEnv(n_states=19, left_reward=-1.0, right_reward=1.0)` in `rl_envs/random_walk.py` (shared with Chapter 6's `RandomWalkEnv`); `n_step_td_prediction` predicts on it, and `run_n_step_prediction_experiment` reproduces Figure 7.2 (intermediate n, e.g. 2–16, beats the extremes at moderate α). Reproduction: `chapter_07/figures/fig_7_2_n_step_td_alpha.png` (`chapter_07/figures/generate_figures.py`). See also `TestNStepTDPrediction`.

### Exercise 7.4 — Pseudocode for off-policy n-step state-value prediction

- **Book location:** §7.3 (n-step off-policy learning), Eq. 7.1, 7.7, 7.9–7.10.
- **Type:** Programming / Theory (pseudocode).
- **Problem (paraphrase):** Give pseudocode for off-policy n-step state-value prediction using the
  off-policy n-step return and TD update.

**Hook.** Now the data come from a behavior policy \(b\) while we evaluate a target policy \(\pi\). Let's build the algorithm that pays for that mismatch with importance-sampling ratios.

**Formal bridge.** The correction is a product of per-step ratios over the window, and it scales the ordinary n-step error:

\[
\rho_{t+1:t+n-1}=\prod_{k=t+1}^{\min(t+n-1,T-1)}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},
\qquad
V(S_t)\leftarrow V(S_t)+\alpha\rho_{t+1:t+n-1}\bigl[G_{t:t+n}-V(S_t)\bigr].
\tag{7.4.1}
\]

where \(\pi\) is the target policy, \(b\) is the behavior policy with coverage over \(\pi\), \(\rho\) is the importance-sampling ratio, \(G_{t:t+n}\) is the n-step return of Eq. 7.1, \(\alpha\in(0,1]\) is the step size, \(n\ge 1\) is the horizon, and \(V\) is the estimated state value (zero at terminals).

**Reasoning chain.** The n-step return is the same as in the on-policy case; what changes is the multiplicative \(\rho\) in (7.4.1). Each factor asks how much more (or less) likely the taken action was under the target than under the behavior. Multiplying them over the window reweights the whole trajectory segment so that, in expectation under \(b\), the update behaves as if the data had come from \(\pi\). A longer window multiplies more ratios together — more correction, but also more noise, since unlikely actions compound.

The pseudocode below follows that structure. The update at time \(\tau\) uses ratios indexed from \(\tau+1\), matching the book's convention for the state-value form:

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

**Why it makes sense.** If \(b=\pi\), every ratio in (7.4.1) is 1 and the algorithm collapses exactly to on-policy n-step TD — a good sanity check on the indexing. If \(b\) rarely takes an action that \(\pi\) favours, \(\rho\) becomes large: one rare but important transition is given a big voice, which is the right compensation but also the source of the variance we flagged above.

**Check your understanding.** Prediction: give \(b\) a tiny probability of choosing one action that \(\pi\) chooses often; that single factor makes \(\rho\) spike and the update for that window becomes high-variance. Move to \(n=1\) and the spike affects only one step instead of a whole product.

**Where this shows up in the code / status.** **Theory-only (no repo API).** No off-policy importance-sampling n-step prediction function exists in `solution.py`. Implemented off-policy-without-IS control is `tree_backup` (different algorithm/target).

### Exercise 7.5 — Pseudocode for off-policy n-step action-value prediction (Sarsa / Expected Sarsa)

- **Book location:** §7.3 (off-policy n-step action values), Eq. 7.4–7.5, 7.9, 7.11; Expected
  Sarsa bootstrapping idea as in Chapter 6/7 discussion.
- **Type:** Programming / Theory (pseudocode).
- **Problem (paraphrase):** Give pseudocode for off-policy n-step action-value prediction using
  the off-policy action-value return/update; specify Sarsa and Expected Sarsa bootstrap targets.

**Hook.** Action values add one choice to the last exercise: at the bootstrap step, do we use the action that was actually sampled, or average over what the target policy would do?

**Formal bridge.** Both answers are n-step returns, but with different bootstrap terms and different ratio horizons:

\[
G_{t:t+n}^{\text{Sarsa}}=\sum_{k=0}^{n-1}\gamma^kR_{t+1+k}+\gamma^n Q(S_{t+n},A_{t+n}),
\quad
\rho^{\text{Sarsa}}=\prod_{i=t+1}^{\min(t+n,T-1)}\frac{\pi(A_i\mid S_i)}{b(A_i\mid S_i)},
\tag{7.5.1}
\]

\[
G_{t:t+n}^{\text{ES}}=\sum_{k=0}^{n-1}\gamma^kR_{t+1+k}
+\gamma^n\sum_a\pi(a\mid S_{t+n})Q(S_{t+n},a),
\quad
\rho^{\text{ES}}=\prod_{i=t+1}^{\min(t+n-1,T-1)}\frac{\pi(A_i\mid S_i)}{b(A_i\mid S_i)}.
\tag{7.5.2}
\]

where \(Q\) is the action-value estimate, \(\pi\) the target policy, \(b\) the behavior policy, \(\rho\) the importance-sampling ratio, \(\gamma\) the discount, and \(n\) the horizon; the update in both cases is \(Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\rho\,[G_{t:t+n}-Q(S_t,A_t)]\).

**Reasoning chain.** Maintain \(Q\), sample actions from \(b\), and evaluate \(\pi\). For the Sarsa-style bootstrap the action \(A_{t+n}\) is known (it was sampled), so the target uses \(Q(S_{t+n},A_{t+n})\) and the ratio product runs through \(i=t+n\): that last action also needs correcting from \(b\) to \(\pi\).

For the Expected Sarsa-style bootstrap the target averages over all \(a\) under \(\pi\), so no \(A_{t+n}\) is sampled and the product stops at \(t+n-1\). Removing the final ratio matters: an expectation needs no importance correction, because it is already a \(\pi\)-quantity. This is the per-decision/control-variate reasoning of §7.4 applied at the bootstrap step.

```text
Sarsa-style bootstrap (action sampled/known at S_{t+n}):
G_{t:t+n} <- Σ_{k=0}^{n-1} γ^k R_{t+1+k} + γ^n Q(S_{t+n}, A_{t+n})   if t+n < T else truncated sum
rho <- Π_{i=t+1}^{min(t+n, T-1)} π(A_i|S_i) / b(A_i|S_i)            (Eq. 7.11 ratio)
Q(S_t,A_t) <- Q(S_t,A_t) + α * rho * (G_{t:t+n} - Q(S_t,A_t))

Expected Sarsa-style bootstrap (expectation under target at the bootstrap step):
G_{t:t+n} <- Σ_{k=0}^{n-1} γ^k R_{t+1+k} + γ^n Σ_a π(a|S_{t+n}) Q(S_{t+n},a)   if t+n < T
rho <- Π_{i=t+1}^{min(t+n-1, T-1)} π(A_i|S_i) / b(A_i|S_i)   (no A_{t+n} factor; bootstrap is expectation)
Q(S_t,A_t) <- Q(S_t,A_t) + α * rho * (G_{t:t+n} - Q(S_t,A_t))
```

**Why it makes sense.** Expected Sarsa removes one source of sampling noise — the choice of \(A_{t+n}\) — at the price of computing an average over \(a\). In the extreme where \(\pi\) is deterministic, that average is just the value of one action, so the two targets coincide except for the ratio horizon; with a random \(\pi\), the Expected Sarsa target is smoother because it does not bet everything on one sampled next action.

**Check your understanding.** Prediction: if \(\pi\) becomes deterministic greedy, the Expected Sarsa bootstrap in (7.5.2) collapses to a single \(Q\) value and the two variants differ only by the final \(\rho\) factor; if \(\pi\) becomes uniform, the averaged bootstrap stays stable while the Sarsa bootstrap jumps between actions.

**Where this shows up in the code / status.** **Theory-only (no repo API).** No off-policy IS action-value or Expected Sarsa n-step implementation exists in `solution.py`. On-policy n-step Sarsa control is implemented as `n_step_sarsa` (no \(\rho\) ratios, behavior=target in effect).

### Exercise 7.6 — Off-policy n-step state-value return as sum of state-based TD errors

- **Book location:** §7.3–7.4, Eq. 7.7/7.10 and TD-error form (cf. Eq. 6.5).
- **Type:** Math / Proof.
- **Problem (paraphrase):** Show that, if \(V\) is fixed, the off-policy n-step error can be
  written compactly via state-based TD errors (with the appropriate importance-sampling weights
  as in the per-decision decomposition).

**Hook.** We already have two ingredients separately — a telescoping error identity and an importance-sampling correction. This exercise puts them together.

**Formal bridge.** With fixed \(V\), Exercise 7.1 gives the unweighted identity, and the off-policy update multiplies it by the ratio:

\[
\rho_{t+1:t+n-1}\bigl(G_{t:t+n}-V(S_t)\bigr)
=\rho_{t+1:t+n-1}\sum_{k=0}^{n-1}\gamma^k\delta_{t+k},
\qquad
\delta_k=R_{k+1}+\gamma V(S_{k+1})-V(S_k).
\tag{7.6.1}
\]

where \(\rho\) is the product of importance ratios over the window, \(G_{t:t+n}\) is the n-step return, \(\delta_k\) is the one-step state TD error, \(V\) is the fixed estimate, and \(\gamma\) discounts later errors.

**Reasoning chain.** Because \(V\) is fixed, the telescoping argument of Exercise 7.1 applies unchanged and \(G_{t:t+n}-V(S_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}\). The off-policy update of Eq. 7.7/7.10 multiplies this whole error by \(\rho_{t+1:t+n-1}\), which simply carries the correction into each term of (7.6.1).

In the per-decision/per-reward view of §7.4, one can instead weight each reward/TD term by the ratio only up to the step where that term is realized. That yields a decomposition equivalent in expectation to (7.6.1), and it is the off-policy state-value analogue of Eq. 6.5/6.6: a sum of importance-weighted state-based TD errors under fixed estimates. This is why per-decision implementations can accumulate discounted TD errors online — each term gets its own, shorter weight rather than waiting for the whole window to finish.

**Why it makes sense.** Set \(\rho=1\) (the on-policy case) and (7.6.1) is exactly Exercise 7.1. When \(\rho\) grows late in the window, the whole-return form applies that large correction to early rewards too, whereas per-decision weighting leaves those early, already-realized rewards alone. That is the same variance story as in Exercise 7.8, seen through the error decomposition rather than the return.

**Check your understanding.** Prediction: with \(b=\pi\) all ratios equal 1 and the per-decision and whole-return forms are identical; as \(b\) diverges from \(\pi\), their weights diverge, but their expectations remain the same.

**Where this shows up in the code / status.** **Theory-only (no repo API).**

### Exercise 7.7 — Off-policy action version and Expected Sarsa TD errors

- **Book location:** §7.3–7.4, Eq. 7.11; Expected Sarsa TD error (cf. Eq. 6.9).
- **Type:** Math / Proof.
- **Problem (paraphrase):** Repeat the decomposition for the off-policy n-step action-value
  return, expressing it in terms of Expected Sarsa–style TD errors when \(Q\) is fixed.

**Hook.** Same argument as the state-value case, one level up: the state TD error is replaced by an Expected Sarsa error.

**Formal bridge.** For fixed \(Q\), define the Expected Sarsa one-step error

\[
\delta_t^{E}=R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)-Q(S_t,A_t),
\tag{7.7.1}
\]

and the corresponding telescoped n-step error

\[
G_{t:t+n}-Q(S_t,A_t)=\sum_{k=0}^{n-1}\gamma^k\tilde{\delta}_{t+k},
\tag{7.7.2}
\]

where \(\tilde{\delta}\) is the appropriate one-step (Expected) Sarsa error at each step, \(Q\) is held fixed, \(\pi\) is the target policy, \(A_t\) is the taken action, and \(\gamma\) is the discount.

**Reasoning chain.** With \(Q\) frozen, substitute the Expected Sarsa error of (7.7.1) into the n-step action-value return \(G_{t:t+n}\) (the Eq. 7.4/7.11 target form). The value terms telescope exactly as they did for state values, and (7.7.2) follows. A sampled-next-action version uses the analogous Sarsa error instead. Multiplying by the appropriate \(\rho\) factors from Eq. 7.11 then gives the off-policy weighted/per-decision decomposition, the action-value analogue of Exercise 7.6.

The one structural difference from the state-value case is at the bootstrap: the expectation \(\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)\) already lives under the target policy, so no importance ratio is needed for the next action there. The correction applies to the actions actually taken earlier in the window.

**Why it makes sense.** At \(n=1\), (7.7.2) is just one Expected Sarsa error — the familiar one-step target. As \(n\) grows, more such errors are added, discounted by \(\gamma^k\). If \(\pi\) is deterministic, the sum in (7.7.1) has a single nonzero term and Expected Sarsa reduces to ordinary Sarsa, so the decomposition collapses consistently to the sampled-action case.

**Check your understanding.** Prediction: make \(\pi\) uniform over the actions; the bootstrap term in (7.7.1) becomes an average, so the per-step errors shrink in variance compared with using one sampled action, while the importance products still carry the off-policy correction.

**Where this shows up in the code / status.** **Theory-only (no repo API).**

### Exercise 7.8 — Data efficiency: per-decision/off-policy n-step vs simpler off-policy

- **Book location:** §7.4 (per-decision methods / control variates), comparing per-decision
  estimators to ordinary importance-sampling n-step updates.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** Devise a small off-policy prediction problem showing that the more
  refined per-decision/off-policy algorithm is more data-efficient (lower variance / faster
  learning) than a simpler off-policy algorithm using coarser importance weighting.

**Hook.** Importance sampling pays for every mismatch, but *when* you pay makes a big difference. Let's set up a task where coarse weighting wastes good data.

**Formal bridge.** The two estimators differ in how far each weight reaches:

\[
\rho^{\text{whole}}_{t+1:t+n}=\prod_{i=t+1}^{t+n}\rho_i,
\qquad
G^{\text{per-decision}}_{t:t+n}=\sum_{k=0}^{n-1}\gamma^k\Bigl(\prod_{i=t+1}^{t+1+k}\rho_i\Bigr)R_{t+1+k},
\tag{7.8.1}
\]

where \(\rho_i=\pi(A_i\mid S_i)/b(A_i\mid S_i)\) is the per-step importance ratio, \(\pi\) is the target policy, \(b\) the behavior policy that wanders but covers \(\pi\)'s actions, \(\gamma\) the discount, and \(n\) the window length.

**Reasoning chain.** Proposed experiment (theory-only): choose a small episodic MDP with known \(v_\pi\), and a behavior policy \(b\) that wanders but covers \(\pi\)'s actions. Compare the learning curves of (a) ordinary off-policy n-step TD with the full-product weight \(\rho^{\text{whole}}\) of Eq. 7.7/7.10 style against (b) a per-decision/per-reward variant of §7.4 that weights each reward/TD term by (7.8.1), i.e. only by ratios up to when that term occurs. Measure MSE \(\|V-v_\pi\|^2\) against episodes over many seeds.

The expected benefit comes from variance. In the coarse estimator, one unlikely action anywhere in the window multiplies the entire return, including rewards that were already realized before that action. In the per-decision estimator, those early rewards keep their smaller weights; only later terms inherit the unlikely action's ratio. Irrelevant future action mismatches therefore stop downweighting (or blowing up) earlier, useful data, which typically lowers variance and improves data efficiency.

**Why it makes sense.** Think of an extreme window where the last action is rare under \(b\). The whole-product weight is tiny (or huge) for the entire trajectory, throwing away a perfectly good early reward. The per-decision weight leaves that early reward at full strength and applies the extreme factor only to the tail. Since the early reward is real data, keeping it cannot hurt in expectation — it only reduces the noise of the estimator.

**Check your understanding.** Prediction: if \(b=\pi\), every \(\rho_i=1\), so both estimators coincide and their learning curves overlap; the gap appears only when \(b\) sometimes takes actions \(\pi\) would not, and widens with window length \(n\).

**Where this shows up in the code / status.** **Theory-only (no repo API).** Neither off-policy IS variant is implemented; `tree_backup` is off-policy *without* IS (a different approach) and is not the algorithm pair requested by this exercise.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section / topic | Repo code (actual) | Repo tests (actual) | Status |
|---|---|---|---|---|
| 7.1 (n-step return identity; fixed-V view) | §7.1 n-step TD prediction | `n_step_td_prediction` (implementation of the box) | `TestNStepTDPrediction` | Implemented |
| Example 7.1 / Figure 7.2 (19-state random walk) | §7.1 | `run_n_step_prediction_experiment`, `plot_figure_7_2` | `TestNStepTDPrediction` | Implemented (illustrative reproduction) |
| 7.2, 7.3 (control intuition; not prediction APIs) | §7.2 n-step Sarsa control | `n_step_sarsa` | `TestNStepSarsa` | Implemented (on-policy control) |
| 7.4–7.8 (off-policy IS / per-decision / Expected Sarsa) | §7.3–7.4 theory | — | — | **Theory-only (no repo API)** |
| Off-policy without IS (related concept) | §7.5 Tree Backup | `tree_backup` | `TestTreeBackup` | Implemented (different from 7.4–7.8) |
| Trace-based control (continuity) | SARSA(λ), accumulating traces | `sarsa_lambda` | `TestSarsaLambda` | Implemented |

- **Full chapter command:** `python -m pytest chapter_07/tests/ -v`
- **Lint:** `ruff check chapter_07/`
- No standalone `exercise_*.py` scripts exist for this chapter. Do not claim off-policy
  importance-sampling APIs; they are not present in `chapter_07/solution.py`.
  (`n_step_td_prediction` **is** present and powers the Example 7.1 / Figure 7.2 reproduction:
  `chapter_07/figures/fig_7_2_n_step_td_alpha.png`; error vs. α for
  n = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 on the 19-state walk, averaged over the first
  10 episodes and 100 repetitions, same walk sets for all settings. The plotted x-axis
  runs α ∈ [0, 1], which is the repo's choice: the book's printed x-axis range is not
  legible from the draft images. The figure is an illustrative reproduction; the
  deterministic seed is 0.)

---

## What to take away

- The n-step return is a dial between one-step TD (Chapter 6) and Monte Carlo: small n bootstraps from a possibly biased estimate, large n waits for real rewards and inherits their variance, and the useful middle is where those two costs balance. The λ-return of Chapter 12 is the next step — a weighted average across all horizons at once.
- The sum-of-TD-errors identity is an algebraic convenience that depends on the estimates standing still. Update inside the window and the telescoping chain has mismatched ends; the same trap returns with the offline/online λ-return distinction in Chapter 12.
- Importance sampling pays for the gap between behavior and target policies, but *when* you pay matters: a whole-window product taxes early, already-realized rewards for a late unlikely action, while per-decision weighting stops each ratio at its own reward. Both are unbiased; the per-decision form is quieter.
- An expectation is already a target-policy quantity, so bootstrapping with Expected Sarsa drops the final importance factor that a sampled Sarsa action still needs. That same rule reappears in the off-policy action-value traces of Chapter 12.
- Off-policy control need not use importance ratios at all: tree backup averages over untried actions to stay usable under any behavior policy, buying lower variance at the cost of a wider backup.

- **Where the theory meets the code.** Exercises 7.1–7.3 are backed by the implemented prediction and n-step Sarsa code, including the illustrative Figure 7.2 reproduction on a fixed seed with the repo's own choice of α-axis. Exercises 7.4–7.8 remain theory-only: no off-policy importance-sampling or Expected Sarsa n-step API exists in this repo, so those answers give pseudocode and a reproducible experiment design rather than a runnable implementation; tree backup is implemented but is a different off-policy-without-IS method.

*End of Chapter 7 exercises document (`chapter_07/exercises.md`).*
