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

- Each exercise lists **Book location**, **Type**, **Paraphrased problem**, **Solution/derivation**, and **Coding pointer** (or theory-only note).
- Verify mapped code with:
  - `python -m pytest chapter_12/tests/ -v`
  - `python -m pytest chapter_12/tests/ -v`

---

## Exercises

### Exercise 12.1 — Recursive form of the λ-return [traces]

- **Book location:** §12.1 (The λ-return), pp. ~255–256; uses (12.1), (12.2), (12.3).
- **Type:** Math.
- **Paraphrased problem:** The ordinary return satisfies a one-step recursion (as in Ch. 3, Eq. 3.9). Derive the analogous recursion for the λ-return defined by (12.1)–(12.2).
- **Solution/derivation:**
  Let the n-step return be
  `G_{t:t+n} = \sum_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})` ((12.1), with `V` the bootstrap value), and for the continuing/infinite-horizon form
  `G_t^λ = (1-λ) \sum_{n=1}^{∞} λ^{n-1} G_{t:t+n}` ((12.2)).
  Note \(G_{t:t+1}=R_{t+1}+\gamma V(S_{t+1})\) and, for `n>=2`, \(G_{t:t+n}=R_{t+1}+\gamma G_{t+1:t+n}\). Substitute:
  `G_t^λ = (1-λ)G_{t:t+1} + (1-λ)\sum_{n=2}^{∞} λ^{n-1}(R_{t+1}+γG_{t+1:t+n})`.
  Since `(1-λ)\sum_{n=1}^{∞}λ^{n-1}=1`, the reward terms collect to `R_{t+1}`. With index shift `m=n-1`,
  `(1-λ)\sum_{n=2}^{∞} λ^{n-1}G_{t+1:t+n} = λ (1-λ)\sum_{m=1}^{∞} λ^{m-1}G_{t+1:t+1+m} = λ G_{t+1}^λ`.
  Therefore the λ-return recursion is
  `G_t^λ = R_{t+1} + γ\left[(1-λ)\,V(S_{t+1}) + λ\,G_{t+1}^λ\right]`  (cf. (12.3)).
  **Checks:** at \(\lambda =0\), \(G_t^\lambda = R_{t+1}+\gamma V(S_{t+1})=G_{t:t+1}\) (one-step return). At \(\lambda =1\), \(G_t^\lambda = R_{t+1}+\gamma G_{t+1}^\lambda\), which unfolds to the full Monte Carlo return (with the usual episodic terminal convention).
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** `chapter_12/solution.py:td_lambda_prediction` is the backward-view TD(λ) prediction algorithm whose forward-view target is the λ-return; see `TestTDLambdaPrediction` (`test_lambda_zero_matches_td0`, `test_high_lambda_close_to_true_values`) for the \(\lambda =0\) and high-\(\lambda\) behavior implied by this recursion.

### Exercise 12.2 — λ and the half-life of trace weighting [traces]

- **Book location:** §12.1, p. ~256 (Figure 12.2 weighting discussion).
- **Type:** Computation / Math.
- **Paraphrased problem:** The weighting on later n-step returns in the λ-return decays geometrically in \(\lambda\). Relate \(\lambda\) to a half-life `τ_{1/2}` (or \(\lambda _h\)), the horizon after which the geometric factor has fallen to half its initial size.
- **Solution/derivation:**
  In (12.2) the n-step component is weighted proportionally to \(\lambda ^{n-1}\). Taking the factor relative to `n=1` as \(\lambda ^{k}\) after `k=n-1` additional steps, the half-life satisfies \(\lambda ^{τ_{1/2}} = 1/2\) (for \(0<\lambda <1\)).
  Solving gives `τ_{1/2} = \ln(1/2) / \ln(λ) = -\ln 2 / \ln(λ)` (steps, in the same step units as the decay). Equivalently, to achieve a desired half-life `τ_{1/2}` choose `λ = 2^{-1/τ_{1/2}} = exp(-\ln 2 / τ_{1/2})`.
  **Sanity:** \(\lambda \to 1\) ⇒ very long half-life; small \(\lambda\) ⇒ short half-life (near one-step).
**Explanation (student-friendly).** The extra n-step returns are weighted (1-lambda) lambda^(n-1). Halving that weight takes about log(1/2)/log(lambda) steps, so small lambda gives a short effective horizon and lambda near 1 looks far ahead. This converts an abstract parameter into 'how many steps ago still matters half as much'.

- **Coding pointer:** Theory-only (parameter-interpretation). Relevant knobs: `lam` in `td_lambda_prediction`, `sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `true_online_td_lambda`.

### Exercise 12.3 — Offline λ-return error as a sum of TD errors [traces]

- **Book location:** §12.2, p. ~260; offline λ-return algorithm (12.4) and TD error (12.6), by analogy with (6.6).
- **Type:** Math / Proof.
- **Paraphrased problem:** For a fixed value/weight vector over an episode, show that the offline λ-return error \(G_t^\lambda - V(S_t)\) can be written as a discounted sum of TD errors (cf. Exercise 12.1 and Ch. 6’s n-step/MC↔TD identities).
- **Solution/derivation:**
  Hold `V` (or `\hat v(\cdot,w)`) fixed. Define \(\delta _t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\) ((12.6), with terminal bootstrap 0). From Ex. 12.1,
  `G_t^λ - V(S_t) = R_{t+1}+γV(S_{t+1})-V(S_t) + γλ\,(G_{t+1}^λ - V(S_{t+1})) = δ_t + γλ\,(G_{t+1}^λ - V(S_{t+1}))`.
  Unrolling to the terminal time `T` (with \(G_T^\lambda = V(S_T)\) = 0 at termination) yields
  `G_t^λ - V(S_t) = \sum_{k=t}^{T-1} (γλ)^{k-t}\, δ_k`.
  (With changing weights this is the identity underlying the forward/backward-view relationship; it becomes an exact per-episode algebraic identity when \(V/\delta _k\) are evaluated consistently/fixed as assumed.)
**Explanation (student-friendly).** With the value function held still, the same telescoping trick used for n-step returns applies: each reward is a TD error plus a difference of neighbouring values, and the intermediate differences cancel. The remaining sum shows that lambda-return updates are just a discounted collection of one-step TD signals.

- **Coding pointer:** Theory-only derivation; motivates `td_lambda_prediction` (`chapter_12/solution.py`) and the forward/backward-view equivalence ideas exercised in `TestTDLambdaPrediction`.

### Exercise 12.4 — Modified online TD(λ) using a pre-update baseline [traces]

- **Book location:** §12.2, p. ~260.
- **Type:** Theory / Math.
- **Paraphrased problem:** Consider redefining the TD error with a pre-update baseline, `\hatδ_t = R_{t+1} + γV_t(S_{t+1}) - V_{t-1}(S_t)`, and argue/show that a correspondingly modified online TD(λ) can achieve the exact per-state increment `ΔV_t(S_t)=α\,[G_t^λ - V_{t-1}(S_t)]` even with online updating/large \(\alpha\). Discuss tradeoffs and sketch an experiment to compare with conventional online TD(λ).
- **Solution/derivation (sketch, as asked):**
  - **Why equivalence can hold:** Conventional online TD(λ) changes `V` during the episode so later TD errors/targets are computed against already-updated values, breaking exact equality to the offline λ-return target based on `V_{t-1}`. By anchoring the “error being corrected” at the pre-update baseline `V_{t-1}(S_t)` (and carrying the effect of within-episode updates into the trace/target accounting, as in the online λ-return / true-online line of derivations in §12.4–§12.5), each visited state’s net increment can be arranged to match \(\alpha (G_t^\lambda - V_{t-1}(S_t))\) for the λ-return computed with the pre-episode values.
  - **Possible advantages:** exact forward-view/λ-return equivalence at any \(\alpha\); clearer target semantics.
  - **Possible disadvantages:** more bookkeeping/memory or computation than standard TD(λ); sensitivity/behavior differs on continuing tasks or with function approximation; practical gains may be small versus simpler true-online variants.
  - **Experiment sketch:** On a prediction benchmark (e.g. random walk, cf. repo’s `chapter_06.solution.RandomWalkEnv` used by ch.8 tests), run conventional TD(λ), the modified method, and true online TD(λ) across \(\alpha , \lambda\); compare RMS value error vs. the known/estimated true values over episodes (learning curves) and sensitivity to \(\alpha\).
**Explanation (student-friendly).** The online update normally changes V(S_t) before the error is formed, so the identity breaks. Measuring the error against the pre-update value restores exactness: each state then receives precisely the increment the offline lambda-return would have given it, even with a large step size.

- **Coding pointer:** No direct implementation of this modified baseline in repo. Closest implemented methods: `chapter_12/solution.py:td_lambda_prediction` (conventional backward-view TD(λ)) and `chapter_12/solution.py:true_online_td_lambda` (true online TD(λ), dutch-trace/online λ-return lineage, §12.5/Eq. 12.14–12.16 tabular form), with `TestTrueOnlineTDLambda`.

### Exercise 12.5 — Truncated λ-return as a sum of TD errors, Eq. (12.11) [traces]

- **Book location:** §12.3 (TTD(λ)), p. ~261; Eq. (12.11).
- **Type:** Math / Proof.
- **Paraphrased problem:** For the (horizon-truncated/ n-step) λ-return, show it can be written in the form of a bootstrapped value plus a discounted sum of TD errors, i.e. `G_t^{λ|h} = V(S_t) + \sum_{i=t}^{h-1} (γλ)^{i-t} δ_i` with \(\delta _i = R_{i+1}+\gamma V(S_{i+1})-V(S_i)\), when `V` is held fixed. Prove (12.11).
- **Solution/derivation:**
  Using the truncated λ-return (cf. (12.9)/(12.10)) or directly the same recursion as Ex. 12.3 but stopped at horizon `h` (where we bootstrap from `V(S_h)`): from Ex. 12.1’s form applied stepwise,
  `G_t^{λ|h} - V(S_t) = δ_t + γλ (G_{t+1}^{λ|h} - V(S_{t+1})) = ... = \sum_{i=t}^{h-1} (γλ)^{i-t} δ_i + (γλ)^{h-t}(V(S_h)-V(S_h))`,
  hence `G_t^{λ|h} = V(S_t) + \sum_{i=t}^{h-1} (γλ)^{i-t} δ_i`, which is (12.11). This is the same telescoping identity as (6.6)/Ex. 12.3, now truncated at `h`.
**Explanation (student-friendly).** Truncating simply stops the telescoping at the chosen horizon and bootstraps from the value there. Writing the result as a value plus a finite sum of TD errors makes the truncated method cheap to implement incrementally.

- **Coding pointer:** Theory-only identity behind truncated/offline/online λ-return views; implemented learning rules in `chapter_12` are backward-view (`td_lambda_prediction`, `sarsa_lambda_*`, `true_online_td_lambda`).

### Exercise 12.6 — Sarsa(λ) pseudocode modified to use dutch traces [traces]

- **Book location:** §12.8 (True online TD(λ)/Sarsa(λ) context) and dutch traces (12.14)–(12.16); exercise p. ~268.
- **Type:** Programming / Theory (pseudocode modification).
- **Paraphrased problem:** Modify the standard accumulating-trace Sarsa(λ) pseudocode to use dutch traces (12.16) instead, assuming linear function approximation with binary features, without adding the other true-online corrections.
- **Solution (modified pseudocode, linear binary features `x(s,a)`):**
  Standard Sarsa(λ): on transition `(S,A,R,S',A')` compute `δ = R + γ\,q(S',A',w) - q(S,A,w)` (or terminal form), \(z \leftarrow \gamma \lambda z + x(S,A)\), \(w \leftarrow w + \alpha \delta z\). **Dutch-trace modification** replaces the trace update by (12.16, tabular/binary-feature friendly form):
  `δ = R + γ\,w^T x(S',A') - w^T x(S,A)` (with terminal handling)
  \(z \leftarrow \gamma \lambda z + x(S,A) - \alpha \gamma \lambda (z^T x(S,A)) x(S,A)\)  (dutch trace: blends accumulating effect with a correction proportional to the component already present in the direction of `x(S,A)`)
  \(w \leftarrow w + \alpha \delta z\)
  and then proceed as usual (choose `A'` etc.). For purely binary one-hot features this reduces so that the visited component’s increment is tempered by \(\alpha \gamma \lambda z_i\), interpolating between accumulating and replacing-like behavior (cf. discussion in §12.6/§12.8).
**Explanation (student-friendly).** Dutch traces replace the naive eligibility increment with one corrected by the current dot product z·x, so repeated visits to the same feature cannot over-count credit. The action-value update is otherwise unchanged: the same TD error is distributed over the corrected traces.

- **Coding pointer:** `chapter_12/solution.py:sarsa_lambda_replacing` and `sarsa_lambda_clearing` implement Sarsa(λ) with **replacing/clearing** traces (not dutch). `chapter_12/solution.py:true_online_td_lambda` implements the **true online (dutch-trace) TD(λ)** prediction update in tabular/one-hot form (Section 12.5, Eq. 12.14–12.16 style), tested by `TestTrueOnlineTDLambda`. A dutch-trace Sarsa(λ) control variant itself is **not implemented** here (theory/pseudocode answer).

### Exercise 12.7 — Truncated versions of off-policy λ-return recursions [Q(lambda)]

- **Book location:** §12.9 (off-policy traces), p. ~272; by analogy with (12.9).
- **Type:** Math.
- **Paraphrased problem:** Generalize the (untruncated) off-policy/state- and action-value λ-return recursions given in §12.9 to horizon-`h` truncated forms \(G_t^{\lambda ,h}\) for (i) state values and (ii) action values in Sarsa-like and Expected-Sarsa-like forms.
- **Solution/derivation:**
  By the same truncation pattern as (12.9), stop mixing n-step returns at horizon `h` and place the remaining weight on the `h`-step return:
  `G_t^{λ,h} = (1-λ)\sum_{n=1}^{h-t-1} λ^{n-1} G_{t:t+n} + λ^{h-t-1} G_{t:h}` (with the convention that if `h <= t+1` the sum vanishes and \(G_t^{\lambda ,h}=G_{t:h}\)).
  Apply this template with the appropriate off-policy n-step targets/returns from §12.9:
  - **State-value/per-decision off-policy:** use `G_{t:t+n}` as the per-decision importance-sampling (or related) n-step state return appearing in (12.29)/(12.31).
  - **Action-value Sarsa form:** use the n-step action returns of (12.26)/(12.27) (with \(\rho\) factors as defined there).
  - **Action-value Expected Sarsa form:** use the n-step returns of (12.27)/(12.28) (bootstrapping from `\bar Q`/expectation under target policy where specified).
  In all cases the only structural change is truncation of the geometric mixture at `h` exactly as in (12.9).
**Explanation (student-friendly).** Truncation means the same recursion stops at h and bootstraps from the estimated value there instead of continuing to termination. The structure of the recursion is unchanged; only the terminal condition and the final bootstrap term differ.

- **Coding pointer:** **Theory-only / not directly implemented.** `chapter_12/solution.py:watkins_lambda_return` implements a *different* forward-view truncated/cut λ-return (Watkins/Q-learning max-bootstrap with a horizon cut at the first non-greedy action, §12.10/Fig. 12.12 style), which is a useful related reference but **not** the same as the §12.9 off-policy IS/Expected-Sarsa recursions asked for here. See `TestWatkinsLambdaReturn`.

### Exercise 12.8 — Exactness of Eq. (12.31) for a fixed value function [Q(lambda)]

- **Book location:** §12.9, p. ~272; Eq. (12.31).
- **Type:** Math / Proof.
- **Paraphrased problem:** Prove that the off-policy state-value λ-return expansion (12.31), `G_t^λ \approx V(S_t,w_t) + ρ_t \sum_{k=t}^{\infty} (...) δ_k`, becomes an exact equality when the approximate value function is held fixed (take `t=0`, write `V_k = v(S_k,w)` constant in `w`).
- **Solution/derivation:**
  With fixed `V_k=v(S_k,w)` and TD error \(\delta _k = R_{k+1}+\gamma V_{k+1}-V_k\) ((12.6), terminal-adjusted), the per-decision off-policy λ-return of §12.9 can be expanded recursively (same telescoping as Ex. 12.3/12.5 but carrying the per-step importance ratio \(\rho _k = \pi (A_k|S_k)/b(A_k|S_k)\) factors as in (12.29)–(12.31)). For `t=0`,
  \(G_0^\lambda - V_0 = \rho _0 \delta _0 + \gamma \lambda \rho _0 (G_1^\lambda - V_1)\) and, unrolling, `G_0^λ - V_0 = \sum_{k=0}^{\infty} (γλ)^{k} (\prod_{i=0}^{k} ρ_i^{(form)}) δ_k` in the precise per-decision/product form given by (12.31). Because no term is dropped when `V` does not change (the approximation in (12.31) comes from within-episode/online changes to `w`), the identity is exact under a fixed value function. (Algebra mirrors the book’s derivation: expand \(G_0^\lambda\), substitute \(R_{k+1}+\gamma V_{k+1}=\delta _k+V_k\), and cancel successive `V_k` terms.)
**Explanation (student-friendly).** When the approximation is frozen, every value in the expansion is a fixed number, so the algebraic rewriting is an identity rather than an approximation. The 'approximately equal' in the book only appears because in a real algorithm w changes during the episode.

- **Coding pointer:** Theory-only; see also Ex. 12.12/12.13 for the backward-view recurrences. No direct IS-weighted off-policy evaluator in repo beyond the Watkins/Q-learning-style control in `chapter_12`.

### Exercise 12.9 — Truncated off-policy state-value return (guess from 12.31) [Q(lambda)]

- **Book location:** §12.9, p. ~272.
- **Type:** Math / Theory.
- **Paraphrased problem:** By analogy with (12.31) and (12.9), write the horizon-`h` truncated off-policy state-value λ-return \(G_t^{\lambda ,h}\).
- **Solution/derivation:**
  Truncate the infinite discounted sum in (12.31) at `k = h-1`:
  `G_t^{λ,h} = V(S_t,w) + \sum_{k=t}^{h-1} (γλ)^{k-t}\left(\prod_{i=t}^{k} ρ_i\right) δ_k^{(V)}` (in the per-decision/state-value form consistent with 12.31), with \(\delta _k^{(V)} = R_{k+1}+\gamma V(S_{k+1},w)-V(S_k,w)\) and the book’s convention for any modified \(\rho\) factors inside the product as specified in §12.9. This matches the “value + truncated weighted sum of TD errors” pattern and reduces to the on-policy truncated form when \(\rho _i=1\).
**Explanation (student-friendly).** Take the infinite sum in the exact off-policy expansion and cut it at the horizon h, replacing the remaining tail by the bootstrapped value. The importance-sampling product is accumulated step by step so the truncated return remains off-policy-correct.

- **Coding pointer:** Theory-only. Related truncation/cutoff behavior (different target: max-bootstrap, non-greedy cut) is illustrated by `watkins_lambda_return` in `chapter_12/solution.py` / `TestWatkinsLambdaReturn`.

### Exercise 12.10 — Exactness of Eq. (12.34) for a fixed action-value function [Q(lambda)]

- **Book location:** §12.9, p. ~273; Eqs. (12.33)–(12.35).
- **Type:** Math / Proof.
- **Paraphrased problem:** Prove the action-value analogue (12.34), `G_t^{λ,a} = Q(S_t,A_t,w) + \sum ... δ_k^a`, becomes exact when `q(\cdot,\cdot,w)` is held fixed (take `t=0`, `Q_k = q(S_k,A_k,w)`).
- **Solution/derivation:**
  With fixed `Q_k=q(S_k,A_k,w)`, define the action TD error of §12.9, e.g. `δ_k^a = R_{k+1} + γ \bar V(S_{k+1}) - Q_k` or the exact form given in (12.35) (including expectation/correction terms as specified). Starting from (12.33)/(12.27)-(12.28), write \(G_0^{\lambda ,a} - Q_0\) and substitute `R_{1}+...` in terms of \(\delta _0^a + Q_0\):
  \(G_0^{\lambda ,a} - Q_0 = \delta _0^a + \gamma \lambda (…)(G_1^{\lambda ,a} - Q_1)\) in the factorized form of (12.34); unrolling gives
  `G_0^{λ,a} = Q_0 + \sum_{k=0}^{\infty} (γλ)^{k}\,\Pi_k\, \delta_k^a`, where `\Pi_k` denotes the appropriate product of action/importance factors from (12.34). As in Ex. 12.8, fixed `q` makes the expansion exact because no approximation-from-updates remains.
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** Theory-only.

### Exercise 12.11 — Truncated off-policy action-value return (guess from 12.34) [Q(lambda)]

- **Book location:** §12.9, p. ~274.
- **Type:** Math / Theory.
- **Paraphrased problem:** Give the horizon-`h` truncated form of the off-policy action-value λ-return \(G_t^{\lambda ,h}\) consistent with (12.34).
- **Solution/derivation:**
  Truncate (12.34) at `h-1`:
  `G_t^{λ,h} = q(S_t,A_t,w) + \sum_{k=t}^{h-1} (γλ)^{k-t}\,\Pi_{t,k}\,\delta_k^a`, with `\delta_k^a` as in (12.35) and `\Pi_{t,k}` the same cumulative factor/product over `[t,k]` as in (12.34). This is the direct action-value analogue of Ex. 12.9 and of (12.9).
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** Theory-only. (See note in Ex. 12.7/12.9 regarding `watkins_lambda_return` being a different cut/target.)

### Exercise 12.12 — Deriving the action-value eligibility trace, Eq. (12.36) [traces]

- **Book location:** §12.9, p. ~274; Eqs. (12.21), (12.33), (12.34), (12.36).
- **Type:** Math / Proof.
- **Paraphrased problem:** Derive the backward-view action-value trace recurrence (12.36), \(z_t = \gamma \lambda _t \rho _t z_{t-1} + \nabla q(S_t,A_t,w_t)\) (up to the book’s exact notation/conventions), from the forward-view/semi-gradient update (12.21) by substituting the λ-return expansion (12.33)/(12.34), paralleling the state-value derivation of (12.32).
- **Solution/derivation (step outline):**
  1. Start from the forward-view update (12.21): \(w_{t+1} = w_t + \alpha (G_t^\lambda - q(S_t,A_t,w_t)) \nabla q(S_t,A_t,w_t)\) (per-step form).
  2. Substitute \(G_t^\lambda - q(S_t,A_t,w_t)\) by the sum-of-weighted-TD-errors form (12.34)/(12.33): `G_t^λ - q_t = \sum_{k>=t} (γλ)^{k-t}\Pi_{t,k} δ_k^a`.
  3. Reindex/sum the total gradient contribution over time as in (12.32): `\sum_t α (G_t^λ-q_t)∇q_t = \sum_k α δ_k^a \sum_{t<=k} (γλ)^{k-t}\Pi_{t,k} ∇q_t`.
  4. Identify the inner sum as an eligibility trace `z_k = \sum_{t<=k} (γλ)^{k-t}\Pi_{t,k} ∇q_t`, which satisfies the one-step recurrence \(z_k = \gamma \lambda _k \rho _k z_{k-1} + \nabla q(S_k,A_k,w_k)\) with `z_{-1}=0` (matching 12.36 under the book’s \(\rho\)/\(\lambda\) conventions). This yields the backward-view update \(w \leftarrow w + \alpha \delta _k^a z_k\) equivalent in the idealized/offline sense.
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** Theory-only derivation. Related implemented backward-view traces: `td_lambda_prediction` (state, accumulating/replacing), `sarsa_lambda_replacing/clearing` (action, replacing/clearing) in `chapter_12/solution.py`.

### Exercise 12.13 — Expected Sarsa λ-return yields same trace with a different TD error [traces]

- **Book location:** §12.9, p. ~274; Eqs. (12.27), (12.28), and resulting TD error form.
- **Type:** Math / Proof.
- **Paraphrased problem:** Repeat the derivation idea of Ex. 12.12 starting from the Expected Sarsa form of the off-policy action λ-return ((12.27),(12.28)) to obtain the same eligibility-trace recurrence as (12.36) but with a different TD error `\hatδ_t^{(a),ES}` that uses the target expectation `\bar Q(S_{t+1})=\sum_a π(a|S_{t+1}) q(S_{t+1},a,w)` and applies \(\rho _{t+1}\) only to the deviation from that expectation.
- **Solution/derivation:**
  In the Expected Sarsa off-policy target ((12.27),(12.28)), the next-state bootstrap is partly the expected value `\bar Q(S_{t+1})` (no importance ratio needed for that expectation component) and partly the sampled-action correction weighted by \(\rho _{t+1}\). Substituting this λ-return into (12.21) and regrouping as in Ex. 12.12 leaves the eligibility vector `z_t` with the same structural recurrence \(z_t = \gamma \lambda _t \rho _t z_{t-1} + \nabla q(S_t,A_t,w_t)\) (the trace still follows the sampled action’s gradient and the per-decision weighting), but the scalar driving it becomes
  `δ_t^{ES} = R_{t+1} + γ\,\bar Q(S_{t+1}) - q(S_t,A_t,w) + γλ_{t+1}\left[ρ_{t+1}\,q(S_{t+1},A_{t+1},w) - \bar Q(S_{t+1})\right]` (as in the book’s stated form, up to \(\gamma _{t+1}/\lambda _{t+1}\) conventions), i.e. the usual Expected Sarsa error plus a trace/IS correction only for the off-policy sampled-action part beyond `\bar Q`.
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** Theory-only (no Expected Sarsa(λ) implementation in repo).

### Exercise 12.14 — Dutch and replacing traces for off-policy state/action methods [traces]

- **Book location:** §12.9, p. ~277.
- **Type:** Theory.
- **Paraphrased problem:** Describe how to form dutch-trace and replacing-trace variants of off-policy eligibility traces for (a) state-value methods and (b) action-value methods.
- **Solution:**
  - **Replacing traces (off-policy):** in place of the accumulating increment, set the trace component(s) for the visited feature(s)/state(-action) to 1 (or the feature value) at visit time, while retaining the off-policy decay/product factors from (12.32)/(12.36) (e.g. multiply by \(\gamma \lambda\) and the appropriate \(\rho\)/policy factors on non-visited steps). For action values, replace/raise the trace of `(S_t,A_t)`; for state values, of `S_t`. Clearing variants (as in Fig. 12.11 style for on-policy control) zero other actions’ traces in the same state when applicable.
  - **Dutch traces (off-policy):** replace the accumulating update by the dutch correction (cf. 12.16): \(z_t \leftarrow \gamma \lambda _t \rho _t z_{t-1} + \alpha _t(1 - \gamma \lambda _t \rho _t z_{t-1}^T x_t) x_t\) in the linear/feature form (adapting notation to state or state-action features and the book’s off-policy factors). This stabilizes/interpolates trace magnitude relative to accumulating traces and is the trace mechanism underlying true-online/off-policy true-online methods.
  - **Interaction with \(\rho\):** in off-policy control/prediction, importance ratios can make accumulating traces high-variance; replacing or dutch traces bound/temper growth but change the effective credit assignment, so behavior and stability differ by method (on-policy vs IS vs tree-backup-like targets).
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** On-policy implementations in `chapter_12/solution.py`: replacing/clearing for Sarsa(λ) (`sarsa_lambda_replacing`, `sarsa_lambda_clearing`, `TestSarsaLambdaReplacing`, `TestSarsaLambdaClearing`), accumulating/replacing for TD(λ) prediction (`td_lambda_prediction`), and dutch/true-online for prediction (`true_online_td_lambda`, `TestTrueOnlineTDLambda`). Off-policy dutch/replacing trace variants per §12.9 are **not implemented** in this repo.

### Exercise 12.15 (Programming) — Rigid Tree Backup (proposed) [Q(lambda)]

- **Book location:** §12.11 (stable off-policy methods / implementation issues, near tree-backup discussion), p. ~277; uses (12.36) and (12.38).
- **Type:** Programming.
- **Paraphrased problem:** Implement the proposed "Rigid Tree Backup" method, which combines the truncated tree-backup λ-return target (12.38) with the usual backward-view eligibility-trace/action update mechanism (cf. 12.36), and compare its behavior to ordinary (one-step or n-step as specified by the reader’s setup) Tree Backup on a small task.
- **Solution (what to do):**
  1. Use the tree-backup n-step/λ-return target (12.38): for action-values, bootstrapping with expected values under the target policy and **not** cutting traces on exploratory actions via IS ratios in the Watkins manner; the truncated tree-backup λ-return mixes these targets over horizons.
  2. Maintain eligibility traces over `(s,a)` updated as in (12.36)’s form appropriate to the chosen approximation (e.g. \(z \leftarrow \gamma \lambda z + \nabla q(s,a)\) with the tree-backup TD/error signal; in tabular form `z(s,a)` accumulates/decays per step).
  3. On each step compute the tree-backup TD error for the chosen horizon/mixing and update `q` (or `w`) by \(\alpha * \delta * z\).
  4. **Experiment:** compare to ordinary Tree Backup on a small control task (e.g. a gridworld/random-walk-style control setup or windy gridworld), plotting learning curves / average return; expect no Watkins-style trace-cut instability from off-policy sampling because tree-backup uses expectations over actions.
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** **Not implemented in this repo.** There is **no** Tree Backup / Tree-Backup(λ) / Rigid Tree Backup implementation in `chapter_12` or `chapter_12`. The closest off-policy trace control code is `chapter_12/solution.py:watkins_q_lambda` and `watkins_lambda_return` (Watkins’s Q(λ), trace-cut on non-greedy actions, max-bootstrap target) with tests `TestWatkinsQLambda`, `TestWatkinsLambdaReturn` — **this is not the same algorithm as Rigid/Tree Backup(λ)** and must not be used as its implementation mapping.

### Exercise 12.16 — Double Expected Sarsa with eligibility traces [traces]

- **Book location:** §12.13 (conclusions / extensions context), p. ~279.
- **Type:** Theory / Math.
- **Paraphrased problem:** Propose how to extend Double Expected Sarsa to use eligibility traces.
- **Solution:**
  Maintain two independent action-value estimators `Q1, Q2` (double learning). On a transition `(S,A,R,S')`:
  - With some rule (e.g. swap/update `Q1` or `Q2` with probability 1/2, or a symmetric scheme), choose which estimator to update, say `Q1`.
  - Form the Double Expected Sarsa target using the **other** estimator for evaluation: `\bar Q_2(S') = \sum_{a'} π(a'|S'; Q1)\, Q2(S',a')` (policy/greedy probabilities can be derived from `Q1` while values come from `Q2`, per the double/expected-sarsa bias-reduction idea).
  - Compute TD error `δ = R + γ \bar Q_2(S') - Q1(S,A)` and apply an eligibility-trace backward view: \(z \leftarrow \gamma \lambda z + \nabla Q1(S,A)\) (and decay/clear per chosen trace type), \(Q1 \leftarrow Q1 + \alpha \delta z\) (or keep separate traces per estimator if updating asymmetrically — a simple design is a trace associated with the estimator being updated).
  This preserves double-estimator decoupling (reducing maximization bias / overestimation) while assigning credit backward in time via traces.
**Explanation (student-friendly).** The lambda-return is a weighted average of all n-step returns. Peeling off the first n-step return leaves the same kind of average starting one step later, which is exactly the recursive form — lambda = 0 keeps only the first piece, lambda = 1 keeps only the full return.

- **Coding pointer:** **Theory-only / not implemented** in `chapter_12`.

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
