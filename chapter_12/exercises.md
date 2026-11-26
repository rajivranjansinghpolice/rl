# Chapter 12 — Eligibility Traces: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd ed., 2018 / complete draft 2017).
> Repo chapter: `chapter_12` (primary) with `chapter_12` — both map to **Book Chapter 12 (Eligibility Traces)**.

---

## Scope and Chapter Mapping

- **Repo `chapter_12`** (`chapter_12/solution.py`): `td_lambda_prediction` (accumulating/replacing traces), `sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `true_online_td_lambda`.
- **Repo `chapter_12`** (`chapter_12/solution.py`): `watkins_q_lambda` (backward-view Watkins's Q(λ), replacing/accumulating traces, trace-cut on non-greedy action) and `watkins_lambda_return` (forward-view trace-cut λ-return). This corresponds to the book's Watkins's Q(λ) material in §12.10 (the repo's `TODO.md` historically labeled it "book Section 12.6"; in the 2nd ed. §12.6 is "Dutch Traces in Monte Carlo Learning" while Watkins's Q(λ) appears in §12.10).
- **Book Chapter 12 sections (2nd ed.):** §12.1 The λ-return, §12.2 TD(λ), §12.3 n-step truncated λ-return methods, §12.4 Redoing updates (online λ-return), §12.5 True online TD(λ), §12.6 Dutch traces in Monte Carlo learning (context), §12.7 Sarsa(λ), §12.8 True online Sarsa(λ) and true online TD(λ), §12.9–§12.11 off-policy traces / stable off-policy methods / implementation issues, §12.10 Watkins's Q(λ) and Tree-Backup(λ), §12.12/§12.13 summary/conclusions.
- **Key equations (plausible, 2nd ed.):** (12.1) n-step return, (12.2)–(12.3) λ-return and recursion, (12.4) offline λ-return update, (12.5)–(12.6) TD(λ)/TD error, (12.7)–(12.8) traces/updates, (12.9) truncated λ-return, (12.10)–(12.11) truncated λ-return as sum of TD errors, (12.12) online λ-return idea, (12.14)–(12.16) true-online/dutch trace recurrences, (12.21) forward-view semi-gradient form, (12.24)/(12.26)/(12.27)/(12.28) off-policy/action λ-returns, (12.29)–(12.36) off-policy trace recurrences and TD errors, (12.38) truncated tree-backup return.
- **PDF pages (plausible, `Barto_Sutton_Book.pdf` extraction):** Chapter 12 ≈ pp. 233–279; exercises 12.1–12.16 appear around pp. 255–256, 260–261, 268, 272–274, 277, 279.
- **Tags:** `[traces]` = eligibility-trace/on-policy TD(λ)/Sarsa(λ) material; `[Q(lambda)]` = off-policy λ-returns, truncation, Watkins-style cutoffs / tree-backup family.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2018, 2nd ed.), Chapter 12. Exercise statements below are **paraphrased** (no long verbatim quotes); equations/section numbers are cited as plausible 2nd-ed. references.
- **Cross-check (sanity only, never copied):** public repos `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, `vojtamolda/reinforcement-learning-an-introduction` and public Ch.12 solution notes. Disagreements resolved by independent derivation.
- **Repo-grounded mapping:** "Coding pointer" entries only reference functions/tests that exist in `chapter_12`; items without a direct implementation are marked **theory-only / not implemented in this repo**.

## How to Use This Document

- Each exercise lists **Book location**, **Type**, **Paraphrased problem**, and a worked answer in the five-part house style (**Hook**, **Formal bridge**, **Reasoning chain**, **Why it makes sense**, **Check your understanding**), plus a **Where this shows up in the code** pointer (or theory-only note).
- Verify mapped code with:
  - `python -m pytest chapter_12/tests/ -v`
  - `python -m pytest chapter_12/tests/ -v`

---

## Exercises

### Exercise 12.1 — Recursive form of the λ-return [traces]

- **Book location:** §12.1 (The λ-return), pp. ~255–256; uses (12.1), (12.2), (12.3).
- **Type:** Math.
- **Paraphrased problem:** The ordinary return satisfies a one-step recursion (as in Ch. 3, Eq. 3.9). Derive the analogous recursion for the λ-return defined by (12.1)–(12.2).

**Hook.** The λ-return is an average of infinitely many n-step returns. Peeling off the first one should leave the same kind of average starting one step later — that is the recursion we are after.

**Formal bridge.** The n-step return and the λ-return are

\[
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n}),
\qquad
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}.
\tag{12.1.1}
\]

where \(G_{t:t+n}\) is the n-step return, \(G_t^\lambda\) the λ-return, \(V\) the bootstrap value, \(\gamma\) the discount, \(\lambda\in[0,1]\) the mixing parameter, and \(R_{t+k+1}\) the reward \(k+1\) steps ahead.

**Reasoning chain.** Note \(G_{t:t+1}=R_{t+1}+\gamma V(S_{t+1})\) and, for \(n\ge2\), \(G_{t:t+n}=R_{t+1}+\gamma G_{t+1:t+n}\). Substituting into (12.1.1) gives

\[
G_t^\lambda=(1-\lambda)G_{t:t+1}
+(1-\lambda)\sum_{n=2}^{\infty}\lambda^{n-1}\bigl(R_{t+1}+\gamma G_{t+1:t+n}\bigr).
\tag{12.1.2}
\]

Since \((1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}=1\), the reward terms collect to \(R_{t+1}\). With the index shift \(m=n-1\),

\[
(1-\lambda)\sum_{n=2}^{\infty}\lambda^{n-1}G_{t+1:t+n}
=\lambda(1-\lambda)\sum_{m=1}^{\infty}\lambda^{m-1}G_{t+1:t+1+m}
=\lambda G_{t+1}^\lambda .
\tag{12.1.3}
\]

The \(n=1\) piece contributes \((1-\lambda)\gamma V(S_{t+1})\), so the λ-return recursion is

\[
G_t^\lambda=R_{t+1}+\gamma\Bigl[(1-\lambda)V(S_{t+1})+\lambda G_{t+1}^\lambda\Bigr].
\tag{12.1.4}
\]

**Why it makes sense.** Check the two ends of the \(\lambda\) dial. At \(\lambda=0\), (12.1.4) gives \(G_t^\lambda=R_{t+1}+\gamma V(S_{t+1})=G_{t:t+1}\), the one-step return. At \(\lambda=1\), it gives \(G_t^\lambda=R_{t+1}+\gamma G_{t+1}^\lambda\), which unfolds to the full Monte-Carlo return with the usual episodic terminal convention. The recursion interpolates between these extremes one step at a time, which is exactly what a weighted average of all n-step returns should do.

**Check your understanding.** Prediction: setting \(\lambda=0\) collapses (12.1.4) to a single bootstrap step; nudging \(\lambda\) upward gives the later, recursive term more weight, so information from farther ahead enters sooner — but at \(\lambda=1\) the recursion must wait until the episode terminates.

**Where this shows up in the code.** `chapter_12/solution.py:td_lambda_prediction` is the backward-view TD(λ) prediction algorithm whose forward-view target is the λ-return; see `TestTDLambdaPrediction` (`test_lambda_zero_matches_td0`, `test_high_lambda_close_to_true_values`) for the \(\lambda=0\) and high-\(\lambda\) behavior implied by this recursion.

### Exercise 12.2 — λ and the half-life of trace weighting [traces]

- **Book location:** §12.1, p. ~256 (Figure 12.2 weighting discussion).
- **Type:** Computation / Math.
- **Paraphrased problem:** The weighting on later n-step returns in the λ-return decays geometrically in \(\lambda\). Relate \(\lambda\) to a half-life \(\tau_{1/2}\) (or \(\lambda_h\)), the horizon after which the geometric factor has fallen to half its initial size.

**Hook.** \(\lambda\) is an abstract number between 0 and 1. Translating it into "how many steps ago still matters half as much" makes it much easier to set in practice.

**Formal bridge.** In (12.2) the n-step component is weighted proportionally to \(\lambda^{n-1}\). Relative to \(n=1\), after \(k=n-1\) additional steps the factor is \(\lambda^k\), so the half-life satisfies

\[
\lambda^{\tau_{1/2}}=\frac12,
\qquad
\tau_{1/2}=\frac{\ln(1/2)}{\ln\lambda}=\frac{-\ln 2}{\ln\lambda}.
\tag{12.2.1}
\]

where \(\lambda\in(0,1)\) is the decay parameter, \(k\) counts steps beyond \(n=1\), and \(\tau_{1/2}\) is the half-life in the same step units as the decay.

**Reasoning chain.** Solving (12.2.1) for the desired half-life gives

\[
\lambda=2^{-1/\tau_{1/2}}=\exp\!\left(-\frac{\ln 2}{\tau_{1/2}}\right).
\tag{12.2.2}
\]

So a smaller \(\lambda\) means a shorter half-life and a more myopic λ-return; a \(\lambda\) very close to 1 means a long half-life and a far-looking return. The constant factor \((1-\lambda)\) in (12.2) only normalises the weights — the decay from one n-step component to the next is entirely governed by \(\lambda^{n-1}\), which is why (12.2.1) captures the effective horizon.

**Why it makes sense.** Think of \(\lambda\) as the retention rate of a memory. If each step retains 90% of the previous weight, it takes a few steps to fall to half; if it retains 99%, it takes many. The direction is unavoidable: \(\lambda\to1\) makes \(\ln\lambda\to0^-\), so \(\tau_{1/2}\to\infty\). A near-one \(\lambda\) therefore behaves almost like Monte Carlo, while small \(\lambda\) behaves almost like one-step TD.

**Check your understanding.** Prediction: doubling the desired half-life in (12.2.2) requires taking the square root of \(\lambda\), moving it closer to 1; halving it squares \(\lambda\), moving it much closer to 0.

**Where this shows up in the code.** Theory-only (parameter-interpretation). Relevant knobs: `lam` in `td_lambda_prediction`, `sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `true_online_td_lambda`.

### Exercise 12.3 — Offline λ-return error as a sum of TD errors [traces]

- **Book location:** §12.2, p. ~260; offline λ-return algorithm (12.4) and TD error (12.6), by analogy with (6.6).
- **Type:** Math / Proof.
- **Paraphrased problem:** For a fixed value/weight vector over an episode, show that the offline λ-return error \(G_t^\lambda - V(S_t)\) can be written as a discounted sum of TD errors (cf. Exercise 12.1 and Ch. 6’s n-step/MC↔TD identities).

**Hook.** The same telescoping trick as for n-step returns works for the λ-return, provided the value function stands still while we do the algebra.

**Formal bridge.** With \(V\) (or \(\hat v(\cdot,w)\)) held fixed, define the one-step TD error

\[
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t),
\tag{12.3.1}
\]

with terminal bootstrap \(0\). From Exercise 12.1's recursion,

\[
G_t^\lambda-V(S_t)=\delta_t+\gamma\lambda\bigl(G_{t+1}^\lambda-V(S_{t+1})\bigr).
\tag{12.3.2}
\]

where \(G_t^\lambda\) is the λ-return, \(V\) the fixed value estimate, \(\delta_t\) the TD error, \(R_{t+1}\) the reward, and \(\gamma\lambda\) the effective per-step decay of the λ-return error.

**Reasoning chain.** Subtract \(V(S_t)\) from the recursion (12.1.4) and regroup the first terms: \(R_{t+1}+\gamma V(S_{t+1})-V(S_t)\) is exactly \(\delta_t\), leaving (12.3.2). Unrolling that recurrence to the terminal time \(T\), with \(G_T^\lambda=V(S_T)=0\) at termination, gives

\[
G_t^\lambda-V(S_t)=\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\,\delta_k.
\tag{12.3.3}
\]

Every intermediate \(V(S_k)\) cancels, just as in the n-step case, because each reward is a TD error plus a difference of neighbouring values. With changing weights this identity underlies the forward/backward-view relationship; it becomes an exact per-episode algebraic identity when \(V\) and \(\delta_k\) are evaluated consistently with the fixed estimates assumed here.

**Why it makes sense.** A λ-return update is just a discounted collection of one-step TD signals. At \(\lambda=0\), (12.3.3) keeps only \(\delta_t\) — ordinary one-step TD. At \(\lambda=1\) it adds every later error without the extra \(\lambda\) decay, matching the Monte-Carlo-style full return. The decay \(\gamma\lambda\) shrinking the contribution of later errors is the right direction: later errors rest on more bootstrap estimates, so they should carry less weight when λ is small.

**Check your understanding.** Prediction: holding \(\gamma\) fixed and lowering \(\lambda\) makes the sum in (12.3.3) concentrate on early \(\delta_k\); raising \(\lambda\) toward 1 gives late errors increasing influence, until at \(\lambda=1\) all remaining steps count equally apart from discounting.

**Where this shows up in the code.** Theory-only derivation; motivates `td_lambda_prediction` (`chapter_12/solution.py`) and the forward/backward-view equivalence ideas exercised in `TestTDLambdaPrediction`.

### Exercise 12.4 — Modified online TD(λ) using a pre-update baseline [traces]

- **Book location:** §12.2, p. ~260.
- **Type:** Theory / Math.
- **Paraphrased problem:** Consider redefining the TD error with a pre-update baseline, `\hatδ_t = R_{t+1} + γV_t(S_{t+1}) - V_{t-1}(S_t)`, and argue/show that a correspondingly modified online TD(λ) can achieve the exact per-state increment `ΔV_t(S_t)=α[G_t^λ - V_{t-1}(S_t)]` even with online updating/large α. Discuss tradeoffs and sketch an experiment to compare with conventional online TD(λ).

**Hook.** Online TD(λ) changes the value before the error is formed, which breaks the clean offline identity. Measuring the error against the pre-update value is a way to restore it.

**Formal bridge.** The pre-update-baseline error and the target increment are

\[
\hat\delta_t=R_{t+1}+\gamma V_t(S_{t+1})-V_{t-1}(S_t),
\qquad
\Delta V_t(S_t)=\alpha\bigl[G_t^\lambda-V_{t-1}(S_t)\bigr].
\tag{12.4.1}
\]

where \(V_{t-1}\) is the value before the update at step \(t\), \(V_t\) the value after it, \(G_t^\lambda\) the λ-return computed with the pre-episode values, \(R_{t+1}\) the reward, \(\gamma\) the discount, and \(\alpha\) the step size.

**Reasoning chain.** Conventional online TD(λ) changes \(V\) during the episode, so later TD errors and targets are computed against already-updated values. That breaks exact equality to the offline λ-return target based on \(V_{t-1}\). By anchoring the "error being corrected" at the pre-update baseline \(V_{t-1}(S_t)\) as in (12.4.1), and carrying the effect of within-episode updates into the trace/target accounting — as in the online λ-return / true-online line of derivations in §12.4–§12.5 — each visited state's net increment can be arranged to match \(\alpha(G_t^\lambda-V_{t-1}(S_t))\) for the λ-return computed with the pre-episode values.

The possible advantages are exact forward-view/λ-return equivalence at any \(\alpha\), and clearer target semantics. The costs are more bookkeeping, memory, or computation than standard TD(λ); different sensitivity and behaviour on continuing tasks or with function approximation; and practical gains that may be small compared with simpler true-online variants.

An experiment sketch: on a prediction benchmark such as a random walk (cf. the repo's `chapter_06.solution.RandomWalkEnv`, used by chapter 8 tests), run conventional TD(λ), the modified method, and true online TD(λ) across a sweep of \(\alpha\) and \(\lambda\). Compare RMS value error against known or estimated true values over episodes (learning curves), and compare sensitivity to \(\alpha\).

**Why it makes sense.** The identity in Exercise 12.3 holds when every value in the sum is the same fixed number. Online updating replaces those numbers mid-stream, so the telescoping chain has mismatched ends. Measuring every error against the pre-update baseline re-anchors the chain: the increment each state receives is precisely the one the offline λ-return would have given it, even with a large step size. The tradeoff is bookkeeping — remembering the old baseline is what buys exactness.

**Check your understanding.** Prediction: if \(\alpha\) is made tiny, conventional online TD(λ) barely changes \(V\) during an episode, so the modified method's advantage shrinks; with a large \(\alpha\), the conventional update drifts far from the baseline and the exactness guaranteed by (12.4.1) matters most.

**Where this shows up in the code.** No direct implementation of this modified baseline in the repo. Closest implemented methods: `chapter_12/solution.py:td_lambda_prediction` (conventional backward-view TD(λ)) and `chapter_12/solution.py:true_online_td_lambda` (true online TD(λ), dutch-trace/online λ-return lineage, §12.5/Eq. 12.14–12.16 tabular form), with `TestTrueOnlineTDLambda`.

### Exercise 12.5 — Truncated λ-return as a sum of TD errors, Eq. (12.11) [traces]

- **Book location:** §12.3 (TTD(λ)), p. ~261; Eq. (12.11).
- **Type:** Math / Proof.
- **Paraphrased problem:** For the (horizon-truncated/ n-step) λ-return, show it can be written in the form of a bootstrapped value plus a discounted sum of TD errors, i.e. `G_t^{λ|h} = V(S_t) + \sum_{i=t}^{h-1} (γλ)^{i-t} δ_i` with \(\delta_i = R_{i+1}+\gamma V(S_{i+1})-V(S_i)\), when `V` is held fixed. Prove (12.11).

**Hook.** Truncation just stops the telescoping early and bootstraps from the value at the cutoff. The proof is Exercise 12.3 with a finite endpoint.

**Formal bridge.** The truncated λ-return identity to prove is

\[
G_t^{\lambda|h}=V(S_t)+\sum_{i=t}^{h-1}(\gamma\lambda)^{i-t}\delta_i,
\qquad
\delta_i=R_{i+1}+\gamma V(S_{i+1})-V(S_i),
\tag{12.5.1}
\]

where \(h\) is the truncation horizon, \(V\) is held fixed, \(G_t^{\lambda|h}\) is the λ-return stopped at \(h\), \(\delta_i\) the one-step TD error, and \(\gamma\lambda\) the effective decay.

**Reasoning chain.** Using the truncated λ-return (cf. (12.9)/(12.10)), or directly the recursion of Exercise 12.3 stopped at horizon \(h\) where we bootstrap from \(V(S_h)\), we have stepwise

\[
G_t^{\lambda|h}-V(S_t)
=\delta_t+\gamma\lambda\bigl(G_{t+1}^{\lambda|h}-V(S_{t+1})\bigr)=\cdots
=\sum_{i=t}^{h-1}(\gamma\lambda)^{i-t}\delta_i
+(\gamma\lambda)^{h-t}\bigl(V(S_h)-V(S_h)\bigr).
\tag{12.5.2}
\]

The final bracket vanishes, so (12.5.2) is exactly (12.5.1). This is the same telescoping identity as (6.6)/Exercise 12.3, now truncated at \(h\): each intermediate value appears once with a positive and once with a negative sign and cancels.

**Why it makes sense.** Cutting the λ-return at \(h\) replaces all later, noisier information with a single bootstrap estimate \(V(S_h)\). In (12.5.1) that shows up as a finite sum: the sum is cheap to accumulate incrementally, which is precisely why truncated λ-return methods are practical. At \(h=t+1\) the sum has one term and we recover one-step TD; as \(h\to\infty\) we recover Exercise 12.3.

**Check your understanding.** Prediction: setting \(h=t+1\) reduces (12.5.1) to \(V(S_t)+\delta_t=R_{t+1}+\gamma V(S_{t+1})\), the one-step target; moving \(h\) later adds more TD errors and makes the target less reliant on the bootstrap at any single state.

**Where this shows up in the code.** Theory-only identity behind truncated/offline/online λ-return views; implemented learning rules in `chapter_12` are backward-view (`td_lambda_prediction`, `sarsa_lambda_*`, `true_online_td_lambda`).

### Exercise 12.6 — Sarsa(λ) pseudocode modified to use dutch traces [traces]

- **Book location:** §12.8 (True online TD(λ)/Sarsa(λ) context) and dutch traces (12.14)–(12.16); exercise p. ~268.
- **Type:** Programming / Theory (pseudocode modification).
- **Paraphrased problem:** Modify the standard accumulating-trace Sarsa(λ) pseudocode to use dutch traces (12.16) instead, assuming linear function approximation with binary features, without adding the other true-online corrections.

**Hook.** Accumulating traces let repeated visits pile credit onto the same feature. Dutch traces temper that pile-up with a correction term — the rest of Sarsa(λ) stays put.

**Formal bridge.** The modified trace and weight updates are

\[
z\leftarrow\gamma\lambda z+x(S,A)-\alpha\gamma\lambda\bigl(z^\top x(S,A)\bigr)x(S,A),
\qquad
w\leftarrow w+\alpha\delta\,z,
\tag{12.6.1}
\]

with the Sarsa TD error

\[
\delta=R+\gamma\,w^\top x(S',A')-w^\top x(S,A),
\tag{12.6.2}
\]

where \(z\) is the eligibility vector, \(x(s,a)\) the binary feature vector, \(w\) the weight vector, \(\alpha\) the step size, \(\gamma\lambda\) the trace decay, \(R\) the reward, and \(S',A'\) the next state and action (terminal handling as usual).

**Reasoning chain.** Standard Sarsa(λ), on transition \((S,A,R,S',A')\), computes \(\delta=R+\gamma q(S',A',w)-q(S,A,w)\), updates \(z\leftarrow\gamma\lambda z+x(S,A)\), and applies \(w\leftarrow w+\alpha\delta z\). The dutch-trace modification replaces only the trace update with (12.6.1), which blends the accumulating effect with a correction proportional to the component already present in the direction of \(x(S,A)\). The weight update (12.6.1) is unchanged, as is action selection.

For purely binary one-hot features, the correction tempers the visited component's increment by \(\alpha\gamma\lambda z_i\), interpolating between accumulating and replacing-like behaviour (cf. §12.6/§12.8). The TD error in (12.6.2) is the ordinary Sarsa error — we are deliberately *not* adding the other true-online corrections.

**Why it makes sense.** With accumulating traces, revisiting the same feature ten times can give it ten times the credit, even if the later visits add no new information. The \(-\alpha\gamma\lambda(z^\top x)x\) term subtracts a piece proportional to what that direction already holds, so repeated visits cannot over-count. If that correction were missing, the update would be the naive accumulating rule; if it were made complete with the other true-online terms, it would be a different (more elaborate) algorithm.

**Check your understanding.** Prediction: with a very small step size \(\alpha\), the dutch correction in (12.6.1) nearly vanishes and the rule approaches plain accumulating traces; a large \(\alpha\) makes the trace of an already-visited feature much smaller, pushing behaviour toward replacing traces.

**Where this shows up in the code.** `chapter_12/solution.py:sarsa_lambda_replacing` and `sarsa_lambda_clearing` implement Sarsa(λ) with **replacing/clearing** traces (not dutch). `chapter_12/solution.py:true_online_td_lambda` implements the **true online (dutch-trace) TD(λ)** prediction update in tabular/one-hot form (Section 12.5, Eq. 12.14–12.16 style), tested by `TestTrueOnlineTDLambda`. A dutch-trace Sarsa(λ) control variant itself is **not implemented** here (theory/pseudocode answer).

### Exercise 12.7 — Truncated versions of off-policy λ-return recursions [Q(lambda)]

- **Book location:** §12.9 (off-policy traces), p. ~272; by analogy with (12.9).
- **Type:** Math.
- **Paraphrased problem:** Generalize the (untruncated) off-policy/state- and action-value λ-return recursions given in §12.9 to horizon-\(h\) truncated forms \(G_t^{\lambda ,h}\) for (i) state values and (ii) action values in Sarsa-like and Expected-Sarsa-like forms.

**Hook.** Truncation is a single structural change: stop mixing n-step returns at \(h\) and put the leftover weight on the \(h\)-step return. The off-policy details sit inside the n-step targets.

**Formal bridge.** The truncation template is

\[
G_t^{\lambda,h}=(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_{t:t+n}
+\lambda^{h-t-1}G_{t:h},
\qquad
G_t^{\lambda,h}=G_{t:h}\ \text{if }h\le t+1,
\tag{12.7.1}
\]

where \(h\) is the truncation horizon, \(\lambda\) the mixing parameter, \(n\) the n-step horizon, and \(G_{t:t+n}\) the appropriate off-policy n-step return from §12.9.

**Reasoning chain.** Apply the template (12.7.1) with the right n-step targets. For the state-value/per-decision off-policy case, use the per-decision importance-sampling n-step state return of (12.29)/(12.31). For the action-value Sarsa form, use the n-step action returns of (12.26)/(12.27), with the \(\rho\) factors as defined there. For the action-value Expected Sarsa form, use the n-step returns of (12.27)/(12.28), bootstrapping from \(\bar Q\) (the expectation under the target policy) where specified.

In all three cases the only structural change is truncating the geometric mixture at \(h\), exactly as in (12.9); the off-policy weighting lives in the choice of \(G_{t:t+n}\) and does not alter the truncation pattern. The boundary convention in (12.7.1) handles the case where the horizon is closer than one step away.

**Why it makes sense.** The λ-return is a weighted average of all n-step returns. Truncation simply says "stop at \(h\) and trust the bootstrap there instead of waiting for the rest of the episode." The weights still sum to one — the leftover geometric mass is placed on the \(h\)-step return — so the truncated target is a proper average, just a shorter one. If the leftover mass were dropped instead, the target would shrink systematically and be biased downward.

**Check your understanding.** Prediction: setting \(h=t+1\) collapses (12.7.1) to the single \(G_{t:t+1}\) target; pushing \(h\) to the episode end recovers the full off-policy λ-return, with the \(\rho\) factors unchanged.

**Where this shows up in the code.** **Theory-only / not directly implemented.** `chapter_12/solution.py:watkins_lambda_return` implements a *different* forward-view truncated/cut λ-return (Watkins/Q-learning max-bootstrap with a horizon cut at the first non-greedy action, §12.10/Fig. 12.12 style), a useful related reference but **not** the same as the §12.9 off-policy IS/Expected-Sarsa recursions asked for here. See `TestWatkinsLambdaReturn`.

### Exercise 12.8 — Exactness of Eq. (12.31) for a fixed value function [Q(lambda)]

- **Book location:** §12.9, p. ~272; Eq. (12.31).
- **Type:** Math / Proof.
- **Paraphrased problem:** Prove that the off-policy state-value λ-return expansion (12.31), `G_t^λ ≈ V(S_t,w_t) + ρ_t \sum_{k=t}^{\infty} (...) δ_k`, becomes an exact equality when the approximate value function is held fixed (take `t=0`, write `V_k = v(S_k,w)` constant in `w`).

**Hook.** The "approximately equal" in the book comes from \(w\) changing mid-episode. Freeze it and the same expression becomes an algebraic identity.

**Formal bridge.** With fixed \(V_k=v(S_k,w)\) and the TD error \(\delta_k=R_{k+1}+\gamma V_{k+1}-V_k\), the per-decision off-policy expansion at \(t=0\) is

\[
G_0^\lambda-V_0=\rho_0\delta_0+\gamma\lambda\rho_0(G_1^\lambda-V_1),
\tag{12.8.1}
\]

which unrolls to

\[
G_0^\lambda-V_0=\sum_{k=0}^{\infty}(\gamma\lambda)^k\Bigl(\prod_{i=0}^{k}\rho_i\Bigr)\delta_k,
\tag{12.8.2}
\]

where \(V_k\) is the fixed approximate value at step \(k\), \(\delta_k\) the TD error, \(\rho_k=\pi(A_k\mid S_k)/b(A_k\mid S_k)\) the per-step importance ratio, \(\gamma\lambda\) the effective decay, and \(w\) is held constant.

**Reasoning chain.** The per-decision off-policy λ-return of §12.9 can be expanded recursively exactly as in Exercises 12.3/12.5, but carrying the per-step importance factors \(\rho_k\) of (12.29)–(12.31). At \(t=0\), (12.8.1) follows by substituting \(R_{k+1}+\gamma V_{k+1}=\delta_k+V_k\) and cancelling successive \(V_k\) terms. Unrolling gives (12.8.2) in the precise per-decision/product form of (12.31).

Because \(V\) does not change, no term is dropped: every value in the expansion is a fixed number, so the algebraic rewriting is an identity rather than an approximation. The approximation in (12.31) comes from within-episode/online changes to \(w\), which are absent under the fixed-value assumption.

**Why it makes sense.** Telescoping requires that the quantity subtracted at one step is the same quantity added at the next. With fixed \(V_k\), that is literally true, so the chain closes exactly. Once \(w\) updates mid-episode, the "same" value is evaluated at two different weights and the chain has a gap — hence only \(\approx\). This is the same distinction as offline versus online λ-return in Exercise 12.4.

**Check your understanding.** Prediction: if \(\rho_i=1\) for all \(i\) (behavior equals target), (12.8.2) reduces to the on-policy sum-of-TD-errors identity; if \(w\) is allowed to change each step, the equality in (12.8.2) degrades to the book's approximation by exactly those within-episode changes.

**Where this shows up in the code.** Theory-only; see also Ex. 12.12/12.13 for the backward-view recurrences. No direct IS-weighted off-policy evaluator in the repo beyond the Watkins/Q-learning-style control in `chapter_12`.

### Exercise 12.9 — Truncated off-policy state-value return (guess from 12.31) [Q(lambda)]

- **Book location:** §12.9, p. ~272.
- **Type:** Math / Theory.
- **Paraphrased problem:** By analogy with (12.31) and (12.9), write the horizon-\(h\) truncated off-policy state-value λ-return \(G_t^{\lambda ,h}\).

**Hook.** Take the exact off-policy expansion from the last exercise and cut the sum at \(h\); the importance product is accumulated along the way.

**Formal bridge.** The truncated off-policy state-value return is

\[
G_t^{\lambda,h}=V(S_t,w)+\sum_{k=t}^{h-1}(\gamma\lambda)^{k-t}
\Bigl(\prod_{i=t}^{k}\rho_i\Bigr)\delta_k^{(V)},
\tag{12.9.1}
\]

with the state TD error

\[
\delta_k^{(V)}=R_{k+1}+\gamma V(S_{k+1},w)-V(S_k,w),
\tag{12.9.2}
\]

where \(h\) is the horizon, \(V(\cdot,w)\) the fixed approximate value, \(\rho_i=\pi(A_i\mid S_i)/b(A_i\mid S_i)\) the importance ratio, \(\gamma\lambda\) the effective decay, and \(\delta_k^{(V)}\) the state-value TD error.

**Reasoning chain.** Truncate the infinite discounted sum in (12.31) at \(k=h-1\), which gives (12.9.1). The TD error \(\delta_k^{(V)}\) is the ordinary state TD error under the fixed \(w\), and the product \(\prod_{i=t}^{k}\rho_i\) follows the book's convention for any modified \(\rho\) factors inside the product as specified in §12.9.

This matches the "value + truncated weighted sum of TD errors" pattern of (12.11) and Exercise 12.5, with the importance-sampling product accumulated step by step so the truncated return remains off-policy-correct. When \(\rho_i=1\), it reduces exactly to the on-policy truncated form.

**Why it makes sense.** Cutting at \(h\) replaces the remaining tail of the infinite sum with the bootstrapped value \(V(S_h,w)\). Keeping the product \(\prod_{i=t}^{k}\rho_i\) up to each term is essential: a reward realized at step \(k\) should be corrected only for the importance mismatch up to that step, not for mismatches later in the window. That is the same per-decision logic as in Chapter 7, and it is why the truncated return stays unbiased under \(b\).

**Check your understanding.** Prediction: if the horizon \(h\) is moved later, more TD terms enter (12.9.1) and the reliance on the single bootstrap \(V(S_h,w)\) decreases; if \(b=\pi\), all \(\rho_i=1\) and the expression collapses to the on-policy truncated λ-return.

**Where this shows up in the code.** Theory-only. Related truncation/cutoff behaviour (different target: max-bootstrap, non-greedy cut) is illustrated by `watkins_lambda_return` in `chapter_12/solution.py` / `TestWatkinsLambdaReturn`.

### Exercise 12.10 — Exactness of Eq. (12.34) for a fixed action-value function [Q(lambda)]

- **Book location:** §12.9, p. ~273; Eqs. (12.33)–(12.35).
- **Type:** Math / Proof.
- **Paraphrased problem:** Prove the action-value analogue (12.34), `G_t^{λ,a} = Q(S_t,A_t,w) + \sum ... δ_k^a`, becomes exact when `q(\cdot,\cdot,w)` is held fixed (take `t=0`, `Q_k = q(S_k,A_k,w)`).

**Hook.** Repeat the fixed-value argument in action space. Freeze \(q\), and the sum-of-TD-errors expansion stops being approximate.

**Formal bridge.** With fixed \(Q_k=q(S_k,A_k,w)\), define an action TD error of §12.9, e.g.

\[
\delta_k^a=R_{k+1}+\gamma\bar V(S_{k+1})-Q_k,
\tag{12.10.1}
\]

or the exact form given in (12.35) including its expectation/correction terms. The expansion to prove is

\[
G_0^{\lambda,a}=Q_0+\sum_{k=0}^{\infty}(\gamma\lambda)^k\,\Pi_k\,\delta_k^a,
\tag{12.10.2}
\]

where \(Q_k\) is the fixed action value, \(\bar V(S_{k+1})\) the expected next-state value under the target policy, \(\delta_k^a\) the action TD error, \(\Pi_k\) the appropriate product of action/importance factors from (12.34), and \(\gamma\lambda\) the effective decay.

**Reasoning chain.** Starting from (12.33)/(12.27)–(12.28), write \(G_0^{\lambda,a}-Q_0\) and substitute \(R_1+\cdots\) in terms of \(\delta_0^a+Q_0\). That gives \(G_0^{\lambda,a}-Q_0=\delta_0^a+\gamma\lambda(\cdots)(G_1^{\lambda,a}-Q_1)\) in the factorized form of (12.34). Unrolling produces (12.10.2), with \(\Pi_k\) collecting the relevant action/importance factors.

As in Exercise 12.8, fixed \(q\) makes the expansion exact because no approximation from within-episode updates remains: each \(Q_k\) is the same number every time it appears, so the telescoping cancellation is complete. The \(\lambda\)-return itself is a weighted average of all n-step returns; peeling off the first leaves the same kind of average starting one step later, which is the recursive form behind the unrolling.

**Why it makes sense.** The action-value case mirrors the state-value case because the algebra only cares that the bootstrapped quantity is fixed. The extra \(\Pi_k\) factors are the price of off-policy sampling — they reweight each action TD error for the behaviour/target mismatch. With \(\Pi_k=1\), (12.10.2) is the familiar on-policy sum; with fixed \(q\), that weighted sum is an identity rather than a target to be chased.

**Check your understanding.** Prediction: allowing \(w\) to change during the episode breaks the equality in (12.10.2) by exactly the within-episode changes; setting \(\lambda=0\) leaves only the \(k=0\) term, the one-step action TD error.

**Where this shows up in the code.** Theory-only.

### Exercise 12.11 — Truncated off-policy action-value return (guess from 12.34) [Q(lambda)]

- **Book location:** §12.9, p. ~274.
- **Type:** Math / Theory.
- **Paraphrased problem:** Give the horizon-\(h\) truncated form of the off-policy action-value λ-return \(G_t^{\lambda ,h}\) consistent with (12.34).

**Hook.** Same cut as Exercise 12.9, now in action space: stop the infinite sum at \(h-1\) and keep the cumulative factor.

**Formal bridge.** The truncated off-policy action-value return is

\[
G_t^{\lambda,h}=q(S_t,A_t,w)+\sum_{k=t}^{h-1}(\gamma\lambda)^{k-t}\,\Pi_{t,k}\,\delta_k^a,
\tag{12.11.1}
\]

with \(\delta_k^a\) as in (12.35) and \(\Pi_{t,k}\) the same cumulative product of action/importance factors over \([t,k]\) as in (12.34), \(q(\cdot,\cdot,w)\) the fixed action-value estimate, and \(\gamma\lambda\) the effective decay.

**Reasoning chain.** Truncate (12.34) at \(k=h-1\), which yields (12.11.1). This is the direct action-value analogue of Exercise 12.9 and of (12.9): the value at the start plus a finite, weighted sum of TD errors up to the horizon. As before, the λ-return behind the expansion is a weighted average of n-step returns, and peeling off the first n-step return leaves the same kind of average starting one step later — the recursion that makes the truncation pattern work.

**Why it makes sense.** Stopping at \(h\) trades the remaining tail of TD errors for a bootstrap at the horizon. The cumulative factor \(\Pi_{t,k}\) still carries the off-policy correction for each term, so each realized reward is weighted only by the mismatches up to its own step. At \(\lambda=0\) only the first term survives; at \(h\) reaching termination the bootstrap disappears and the full off-policy return is recovered.

**Check your understanding.** Prediction: moving the horizon \(h\) earlier makes (12.11.1) depend more on \(q(S_h,A_h,w)\) and less on later TD errors; if all importance factors become 1, the expression becomes the on-policy truncated action λ-return.

**Where this shows up in the code.** Theory-only. (See note in Ex. 12.7/12.9 regarding `watkins_lambda_return` being a different cut/target.)

### Exercise 12.12 — Deriving the action-value eligibility trace, Eq. (12.36) [traces]

- **Book location:** §12.9, p. ~274; Eqs. (12.21), (12.33), (12.34), (12.36).
- **Type:** Math / Proof.
- **Paraphrased problem:** Derive the backward-view action-value trace recurrence (12.36), \(z_t = \gamma \lambda _t \rho _t z_{t-1} + \nabla q(S_t,A_t,w_t)\) (up to the book’s exact notation/conventions), from the forward-view/semi-gradient update (12.21) by substituting the λ-return expansion (12.33)/(12.34), paralleling the state-value derivation of (12.32).

**Hook.** Forward views sum a long return; backward views keep a running trace. This derivation shows they are the same sum reordered.

**Formal bridge.** Start from the forward-view semi-gradient update

\[
w_{t+1}=w_t+\alpha\bigl(G_t^\lambda-q(S_t,A_t,w_t)\bigr)\nabla q(S_t,A_t,w_t),
\tag{12.12.1}
\]

and aim for the trace recurrence

\[
z_k=\gamma\lambda_k\rho_k z_{k-1}+\nabla q(S_k,A_k,w_k),
\qquad
w\leftarrow w+\alpha\delta_k^a z_k.
\tag{12.12.2}
\]

where \(\nabla q\) is the gradient of the action value, \(z_k\) the eligibility trace, \(\delta_k^a\) the action TD error, \(\rho_k\) the importance ratio, \(\lambda_k\) the trace parameter, \(\gamma\) the discount, \(\alpha\) the step size, and \(z_{-1}=0\).

**Reasoning chain.** First, substitute \(G_t^\lambda-q_t\) using the sum-of-weighted-TD-errors form (12.34)/(12.33): \(G_t^\lambda-q_t=\sum_{k\ge t}(\gamma\lambda)^{k-t}\Pi_{t,k}\delta_k^a\). Next reindex the total gradient contribution over time exactly as in (12.32):

\[
\sum_t\alpha(G_t^\lambda-q_t)\nabla q_t
=\sum_k\alpha\delta_k^a\sum_{t\le k}(\gamma\lambda)^{k-t}\Pi_{t,k}\nabla q_t.
\tag{12.12.3}
\]

The inner sum in (12.12.3) is the eligibility trace, \(z_k=\sum_{t\le k}(\gamma\lambda)^{k-t}\Pi_{t,k}\nabla q_t\), which satisfies the one-step recurrence (12.12.2) with \(z_{-1}=0\), matching (12.36) under the book's \(\rho\)/\(\lambda\) conventions. This yields the backward-view update \(w\leftarrow w+\alpha\delta_k^a z_k\), equivalent in the idealized/offline sense. The \(\lambda\)-return behind (12.12.1) is a weighted average of all n-step returns, and peeling off the first leaves the same average one step later — the recursion that makes the reindexing valid.

**Why it makes sense.** A forward view must remember a whole trajectory to compute \(G_t^\lambda\); a backward view compresses that memory into one vector \(z_k\) that decays with \(\gamma\lambda\rho\) and is refreshed by the current gradient. The \(\rho_k\) factor in (12.12.2) is there because off-policy credit must be reweighted as it travels backward. When the trace is zero, no past state is eligible, and the update reduces to the one-step case — the same limiting behaviour as λ=0.

**Check your understanding.** Prediction: setting \(\lambda_k=0\) kills the decayed past term in (12.12.2), leaving only the current gradient and one-step TD; raising \(\lambda_k\) keeps older gradients eligible longer, but large \(\rho_k\) can make them high-variance.

**Where this shows up in the code.** Theory-only derivation. Related implemented backward-view traces: `td_lambda_prediction` (state, accumulating/replacing), `sarsa_lambda_replacing/clearing` (action, replacing/clearing) in `chapter_12/solution.py`.

### Exercise 12.13 — Expected Sarsa λ-return yields same trace with a different TD error [traces]

- **Book location:** §12.9, p. ~274; Eqs. (12.27), (12.28), and resulting TD error form.
- **Type:** Math / Proof.
- **Paraphrased problem:** Repeat the derivation idea of Ex. 12.12 starting from the Expected Sarsa form of the off-policy action λ-return ((12.27),(12.28)) to obtain the same eligibility-trace recurrence as (12.36) but with a different TD error `\hatδ_t^{(a),ES}` that uses the target expectation `\bar Q(S_{t+1})=\sum_a π(a|S_{t+1}) q(S_{t+1},a,w)` and applies \(\rho _{t+1}\) only to the deviation from that expectation.

**Hook.** Changing the bootstrap from a sampled action to an expectation does not change the trace — it changes the scalar driving it.

**Formal bridge.** In the Expected Sarsa off-policy target, the bootstrap splits into an expectation under the target policy and a sampled-action correction:

\[
\bar Q(S_{t+1})=\sum_a\pi(a\mid S_{t+1})\,q(S_{t+1},a,w),
\tag{12.13.1}
\]

\[
\delta_t^{ES}=R_{t+1}+\gamma\bar Q(S_{t+1})-q(S_t,A_t,w)
+\gamma\lambda_{t+1}\Bigl[\rho_{t+1}q(S_{t+1},A_{t+1},w)-\bar Q(S_{t+1})\Bigr].
\tag{12.13.2}
\]

where \(\bar Q\) is the expected next-state action value under \(\pi\), \(q\) the action-value estimate, \(\rho_{t+1}\) the importance ratio on the sampled next action, \(\lambda_{t+1}\) the trace parameter, \(\gamma\) the discount, and \(R_{t+1}\) the reward.

**Reasoning chain.** In (12.27)/(12.28), the next-state bootstrap is partly \(\bar Q(S_{t+1})\) — no importance ratio is needed for that expectation component — and partly the sampled-action correction weighted by \(\rho_{t+1}\). Substituting this λ-return into (12.21) and regrouping as in Exercise 12.12 leaves the eligibility vector \(z_t\) with the same structural recurrence \(z_t=\gamma\lambda_t\rho_t z_{t-1}+\nabla q(S_t,A_t,w_t)\): the trace still follows the sampled action's gradient and the per-decision weighting. What changes is the scalar driving it, which becomes (12.13.2) — the usual Expected Sarsa error plus a trace/IS correction only for the off-policy sampled-action part beyond \(\bar Q\), as in the book's stated form (up to \(\gamma_{t+1}/\lambda_{t+1}\) conventions).

**Why it makes sense.** Expectation components are already under the target policy, so importance-sampling them would double-count the correction; only the deviation of the actually sampled action from that expectation needs \(\rho_{t+1}\). This is the same per-decision logic as Chapter 7's Expected Sarsa bootstrap. When \(\rho_{t+1}=1\) or the sampled action equals the expectation, the correction term in (12.13.2) vanishes and the ordinary Expected Sarsa TD error remains.

**Check your understanding.** Prediction: making \(\pi\) deterministic turns \(\bar Q(S_{t+1})\) into a single action value, so the correction bracket in (12.13.2) shrinks and the error approaches the sampled-action form; making \(\pi\) uniform leaves a larger expected component and a different correction magnitude, but the trace recurrence is unchanged.

**Where this shows up in the code.** Theory-only (no Expected Sarsa(λ) implementation in repo).

### Exercise 12.14 — Dutch and replacing traces for off-policy state/action methods [traces]

- **Book location:** §12.9, p. ~277.
- **Type:** Theory.
- **Paraphrased problem:** Describe how to form dutch-trace and replacing-trace variants of off-policy eligibility traces for (a) state-value methods and (b) action-value methods.

**Hook.** Trace type and off-policy correction are independent choices. Let's combine them carefully so the correction factors survive the change of trace.

**Formal bridge.** The two variants modify the accumulating trace increment:

\[
\text{replacing:}\quad z_i\leftarrow x_i\ \text{for visited }i,
\tag{12.14.1}
\]

\[
\text{dutch:}\quad
z_t\leftarrow\gamma\lambda_t\rho_t z_{t-1}
+\alpha_t\bigl(1-\gamma\lambda_t\rho_t z_{t-1}^\top x_t\bigr)x_t,
\tag{12.14.2}
\]

where \(z_t\) is the eligibility vector, \(x_t\) the current (state or state–action) feature vector, \(\rho_t\) the importance ratio, \(\lambda_t\) the trace parameter, \(\gamma\) the discount, and \(\alpha_t\) the step size. Both keep the off-policy decay/product factors of (12.32)/(12.36).

**Reasoning chain.** For replacing traces in the off-policy case, set the trace component(s) for the visited feature(s)/state(-action) to 1 (or the feature value) at visit time, while retaining the off-policy decay/product factors from (12.32)/(12.36) — multiply by \(\gamma\lambda\) and the appropriate \(\rho\)/policy factors on non-visited steps. For action values, replace or raise the trace of \((S_t,A_t)\); for state values, of \(S_t\). Clearing variants, as in Fig. 12.11 style for on-policy control, zero other actions' traces in the same state when applicable.

For dutch traces, replace the accumulating update by the dutch correction (12.14.2) adapted to state or state–action features and the book's off-policy factors. This stabilises or interpolates trace magnitude relative to accumulating traces and is the mechanism underlying true-online/off-policy true-online methods.

The interaction with \(\rho\) matters: importance ratios can make accumulating traces high-variance; replacing or dutch traces bound or temper growth, but they also change the effective credit assignment, so behaviour and stability differ by method (on-policy vs IS vs tree-backup-like targets).

**Why it makes sense.** Accumulating traces add a little credit every time a feature is visited, so a long, off-policy trajectory with large \(\rho\) can produce enormous eligibility. Replacing traces cap a component at one visit's worth; dutch traces subtract the component already present via the \(z_{t-1}^\top x_t\) term. Both are ways of saying "revisiting the same feature should not multiply its credit without bound" — the same intuition as in Exercise 12.6, now with the \(\rho\) factor carried through.

**Check your understanding.** Prediction: if \(\rho_t\approx1\), (12.14.2) reduces to the ordinary on-policy dutch form; if \(\rho_t\) is large, the decay term grows and the correction must work harder to keep the trace from exploding, which is why off-policy dutch variants are higher-variance than on-policy ones.

**Where this shows up in the code.** On-policy implementations in `chapter_12/solution.py`: replacing/clearing for Sarsa(λ) (`sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `TestSarsaLambdaReplacing`, `TestSarsaLambdaClearing`), accumulating/replacing for TD(λ) prediction (`td_lambda_prediction`), and dutch/true-online for prediction (`true_online_td_lambda`, `TestTrueOnlineTDLambda`). Off-policy dutch/replacing trace variants per §12.9 are **not implemented** in this repo.

### Exercise 12.15 (Programming) — Rigid Tree Backup (proposed) [Q(lambda)]

- **Book location:** §12.11 (stable off-policy methods / implementation issues, near tree-backup discussion), p. ~277; uses (12.36) and (12.38).
- **Type:** Programming.
- **Paraphrased problem:** Implement the proposed "Rigid Tree Backup" method, which combines the truncated tree-backup λ-return target (12.38) with the usual backward-view eligibility-trace/action update mechanism (cf. 12.36), and compare its behavior to ordinary (one-step or n-step as specified by the reader’s setup) Tree Backup on a small task.

**Hook.** Tree backup avoids importance-sampling ratios by averaging over untried actions. Let's give it eligibility traces and see what that combination buys.

**Formal bridge.** The target mixes tree-backup n-step returns over horizons:

\[
G_t^{\lambda,\mathrm{TB}}=(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_{t:t+n}^{\mathrm{TB}}
+\lambda^{h-t-1}G_{t:h}^{\mathrm{TB}},
\tag{12.15.1}
\]

with the TD error driving the backward view

\[
z\leftarrow\gamma\lambda z+\nabla q(s,a),
\qquad
w\leftarrow w+\alpha\,\delta^{\mathrm{TB}}\,z.
\tag{12.15.2}
\]

where \(G^{\mathrm{TB}}\) is the tree-backup return built from expected values under the target policy, \(z\) the eligibility trace, \(q\) the action value, \(\delta^{\mathrm{TB}}\) the tree-backup TD error, \(\gamma\lambda\) the trace decay, and \(\alpha\) the step size.

**Reasoning chain.** The implementation plan has four parts. First, use the tree-backup n-step/λ-return target (12.38): for action values, bootstrap with expected values under the target policy and **do not** cut traces on exploratory actions via importance ratios in the Watkins manner; the truncated tree-backup λ-return mixes these targets over horizons as in (12.15.1).

Second, maintain eligibility traces over \((s,a)\) updated in the form of (12.36) appropriate to the chosen approximation — e.g. \(z\leftarrow\gamma\lambda z+\nabla q(s,a)\) with the tree-backup TD/error signal; in tabular form, \(z(s,a)\) accumulates and decays per step (12.15.2).

Third, on each step compute the tree-backup TD error for the chosen horizon/mixing and update \(q\) (or \(w\)) by \(\alpha\delta z\).

Fourth, compare with ordinary Tree Backup on a small control task — for example a gridworld/random-walk-style control setup or windy gridworld — plotting learning curves or average return. Expect no Watkins-style trace-cut instability from off-policy sampling, because tree backup uses expectations over actions.

**Why it makes sense.** Watkins's Q(λ) cuts the trace when an exploratory action is taken, because a max-bootstrap cannot safely credit a non-greedy action. Tree backup instead averages over all actions under the target policy, so no cut is needed — the expectation already accounts for the possibility that the sampled action was exploratory. Combining that expectation-based target with traces (12.15.2) should therefore retain credit assignment without the cutoff instability, which is exactly what the proposed experiment checks.

**Check your understanding.** Prediction: setting \(\lambda=0\) should make Rigid Tree Backup behave like ordinary one-step Tree Backup, since the trace in (12.15.2) then carries no past credit; raising \(\lambda\) should speed up credit propagation but also expose any instability the expectation-based target fails to suppress.

**Where this shows up in the code.** **Not implemented in this repo.** There is **no** Tree Backup / Tree-Backup(λ) / Rigid Tree Backup implementation in `chapter_12` or `chapter_12`. The closest off-policy trace control code is `chapter_12/solution.py:watkins_q_lambda` and `watkins_lambda_return` (Watkins's Q(λ), trace-cut on non-greedy actions, max-bootstrap target) with tests `TestWatkinsQLambda`, `TestWatkinsLambdaReturn` — **this is not the same algorithm as Rigid/Tree Backup(λ)** and must not be used as its implementation mapping.

### Exercise 12.16 — Double Expected Sarsa with eligibility traces [traces]

- **Book location:** §12.13 (conclusions / extensions context), p. ~279.
- **Type:** Theory / Math.
- **Paraphrased problem:** Propose how to extend Double Expected Sarsa to use eligibility traces.

**Hook.** Double learning already decouples action selection from evaluation. Adding traces should assign credit backward without undoing that decoupling.

**Formal bridge.** Maintain two estimators \(Q_1,Q_2\); the double Expected Sarsa target uses one to choose and the other to evaluate:

\[
\bar Q_2(S')=\sum_{a'}\pi(a'\mid S';Q_1)\,Q_2(S',a'),
\qquad
\delta=R+\gamma\bar Q_2(S')-Q_1(S,A),
\tag{12.16.1}
\]

with the backward view

\[
z\leftarrow\gamma\lambda z+\nabla Q_1(S,A),
\qquad
Q_1\leftarrow Q_1+\alpha\delta z.
\tag{12.16.2}
\]

where \(\pi(\cdot\mid S';Q_1)\) is the greedy policy derived from \(Q_1\), \(\bar Q_2\) the expected next-state value under \(Q_2\), \(\delta\) the TD error, \(z\) the eligibility trace, \(\gamma\lambda\) the trace decay, and \(\alpha\) the step size.

**Reasoning chain.** On a transition \((S,A,R,S')\), first choose which estimator to update — for example, swap or update \(Q_1\) or \(Q_2\) with probability 1/2, or use a symmetric scheme; say we update \(Q_1\). Form the Double Expected Sarsa target using the **other** estimator for evaluation: \(\bar Q_2(S')=\sum_{a'}\pi(a'|S';Q_1)Q_2(S',a')\), with policy/greedy probabilities derived from \(Q_1\) while values come from \(Q_2\), per the double/expected-Sarsa bias-reduction idea.

Compute the TD error \(\delta=R+\gamma\bar Q_2(S')-Q_1(S,A)\) and apply the eligibility-trace backward view \(z\leftarrow\gamma\lambda z+\nabla Q_1(S,A)\) (decaying or clearing per the chosen trace type), followed by \(Q_1\leftarrow Q_1+\alpha\delta z\). If updates are asymmetric, keep separate traces per estimator — a simple design ties the trace to the estimator being updated.

This preserves double-estimator decoupling — reducing maximisation bias / overestimation — while assigning credit backward in time via traces. The λ-return reasoning still applies: the weighted average of n-step returns can be peeled into a recursion starting one step later, which is what lets the trace carry the multi-step credit.

**Why it makes sense.** Overestimation comes from using the same noisy estimate both to pick the best action and to score it. Splitting those jobs makes the target conservative without shrinking values by hand. Traces then propagate that honest target to earlier states; if the trace instead used \(Q_2\)'s gradient while updating \(Q_1\), the decoupling would blur and the bias reduction would be compromised. At \(\lambda=0\) the scheme is ordinary Double Expected Sarsa — a useful limiting check.

**Check your understanding.** Prediction: using a single estimator for both selection and evaluation would restore maximisation bias; setting \(\lambda=0\) removes the trace and gives plain Double Expected Sarsa, while a large \(\lambda\) spreads the conservatively evaluated target further back at the cost of more variance.

**Where this shows up in the code.** **Theory-only / not implemented** in `chapter_12`.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Topic | Repo code | Repo tests | Notes / Pytest command |
|---|---|---|---|---|
| 12.1–12.3, 12.5 | λ-return recursion, TD(λ) sums/identities, prediction targets | `chapter_12/solution.py` (`td_lambda_prediction`) | `chapter_12/tests/test_solution.py::TestTDLambdaPrediction` | Theory-backed; `python -m pytest chapter_12/tests/test_solution.py -k TestTDLambdaPrediction -v` |
| 12.4 | Modified online TD(λ) (pre-update baseline) | — | — | **Not implemented.** Closest: `chapter_12/solution.py:true_online_td_lambda` (`TestTrueOnlineTDLambda`). |
| 12.6 | Sarsa(λ) with dutch traces (pseudocode) | `chapter_12/solution.py` (`sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `true_online_td_lambda`) | `TestSarsaLambdaReplacing`, `TestSarsaLambdaClearing`, `TestTrueOnlineTDLambda` | Dutch-trace **control** not implemented; dutch/true-online **prediction** implemented as `true_online_td_lambda`. |
| 12.7, 12.9, 12.11 | Truncated off-policy/state & action λ-returns | `chapter_12/solution.py` (`watkins_lambda_return`) | `TestWatkinsLambdaReturn` | **Partial/related only:** `watkins_lambda_return` is the forward-view Watkins trace-cut (max-bootstrap, non-greedy cutoff, §12.10), not the §12.9 IS/Expected-Sarsa truncated returns derived in these exercises. |
| 12.8, 12.10, 12.12, 12.13 | Off-policy return/trace derivations | — | — | **Theory-only.** |
| 12.14 | Dutch/replacing traces off-policy | `chapter_12/solution.py` (on-policy trace variants) | `TestTDLambdaPrediction`, `TestSarsaLambdaReplacing`, `TestSarsaLambdaClearing`, `TestTrueOnlineTDLambda` | **Off-policy** dutch/replacing variants not implemented. |
| 12.15 | Rigid Tree Backup (programming) | — | — | **Not implemented.** Not the same as `watkins_q_lambda` / `watkins_lambda_return`. |
| 12.16 | Double Expected Sarsa(λ) | — | — | **Theory-only / not implemented.** |

**Verification commands:**
- `python -m pytest chapter_12/tests/ -v`
- `python -m pytest -q` (full suite, from the repository root)
- `ruff check chapter_12/` (if Ruff is installed)

---

## Pointer from Chapter 10

Watkins's Q(λ) material from the former `chapter_10/` was merged into this
chapter. `chapter_10/README.md` points here for the book Chapter 12 exercises
that cover that material.

---

## Verification and Quality Notes

- Derivations (esp. Ex. 12.1, 12.3, 12.5 and the truncation identities) were re-derived independently; public solutions were used only for sanity-checking.
- Citations (§/Eq./page) are plausible for Sutton & Barto 2nd ed. and should be treated as reader aids; resolve any numbering ambiguity against the primary text.
- Exercise-to-code mappings were checked against actual symbols/tests: `td_lambda_prediction`, `sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `true_online_td_lambda` in `chapter_12/solution.py` and `watkins_q_lambda`, `watkins_lambda_return` in `chapter_12/solution.py`. Ex. 12.15 (Rigid Tree Backup) has **no** implementation in this repo and is not mapped to Watkins’s Q(λ).

---

*Exercises document for Chapter 12 — part of the per-chapter exercises-and-solutions series (`chapter_XX/exercises.md`).*
