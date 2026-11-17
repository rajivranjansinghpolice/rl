# Chapter 13 — Policy Gradient Methods: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_13`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_13` — Policy Gradient Methods (REINFORCE `PolicyGradientAlgorithm`,
  REINFORCE with baseline `PolicyGradientWithBaseline`, actor–critic `ActorCriticAlgorithm`,
  continuing actor–critic `ContinuingActorCritic`, Gaussian policies `GaussianPolicy`, simulation
  harness `run_policy_gradient_simulation`).
- **Book chapter:** Chapter 13 — Policy Gradient Methods.
- **Book sections covered by the chapter implementation:**
  - Section 13.1 — Policy approximation and its advantages (Example 13.1).
  - Section 13.2 — The policy gradient theorem (Eq. 13.5).
  - Section 13.3 — REINFORCE: Monte Carlo policy gradient (Eq. 13.6–13.7).
  - Section 13.4 — REINFORCE with baseline.
  - Section 13.5–13.7 — Actor–critic methods; Gaussian policies (continuous actions).
- **Key book equations referenced by the exercises:**
  - Eq. (13.2) — softmax action preferences: `π(a|s,θ) = exp(h(s,a,θ)) / Σ_b exp(h(s,b,θ))`.
  - Eq. (13.5) — policy gradient theorem.
  - Eq. (13.6) — REINFORCE update: `θ ← θ + α·G·∇_θπ(A|S,θ) / π(A|S, θt)`.
  - Eq. (13.7) — softmax eligibility: `∇_θ ln π(a|s,θ) = x(s,a) − Σ_b π(b|s,θ) x(s,b)`.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Chapter 13 content spans
  roughly PDF pages 283–300; the exercises were extracted from PDF pages 284, 290, and 296.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 13. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-13 material, plus the public
  chapter-13 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem**, **Solution**, and — where a companion implementation exists — a
  **coding pointer** into `chapter_13/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations; Programming
  exercises describe an experiment to run.
- The final section maps exercises to the repository code that already implements (or supports) the
  corresponding machinery; exercises that exercise a specific implementation also carry a **Coding
  pointer** inline.

---

## Exercises

### Exercise 13.1 — Optimal probability of "right" in the short-corridor gridworld

- **Book location:** Section 13.1 / Example 13.1, PDF page 284 (Example on PDF page 285).
- **Type:** Computation.
- **Problem:** Example 13.1 (short corridor, three nonterminal states, switched actions in the
  middle state; every state looks identical to the agent: `x(s, right) = [1, 0]ᵀ`,
  `x(s, left) = [0, 1]ᵀ`; reward −1 per step; the ε-greedy right/left policies achieve values of
  about −44 and −82 respectively). Determine the exact symbolic expression for the optimal probability
  `p = P(right)` of selecting the right action in this example.
- **Solution.** Because the features are state-independent, the best the agent can do is pick one
  probability `p` for all states. Let `V1, V2, V3` be the expected returns starting from states
  1, 2, 3 (state 1 = start; state 3 next to the terminal cell):
  - State 1: `V1 = −1 + p·V2 + (1−p)·V1`  ⇒  `V1·p = −1 + p·V2`  ⇒  `V2 = V1 + 1/p`.
  - State 2 (actions reversed): `V2 = −1 + p·V1 + (1−p)·V3`.
  - State 3: `V3 = −1 + (1−p)·V2` (right → terminal with reward −1).
  Substitute `V3` into `V2`: `V2 = −1 + p·V1 + (1−p)(−1 + (1−p)·V2) = (p − 2) + p·V1 + (1−p)²·V2`,
  so `V2(1 − (1−p)²) = p − 2 + p·V1` ⇒ `V2 = (p·V1 + p − 2)/(p(2−p))`.
  Using `V2 = V1 + 1/p` gives `V1 + 1/p = (p·V1 + p − 2)/(p(2−p))`, which simplifies to
  `V1(p) = (2p − 4) / (p(1 − p)) = 2(p − 2) / (p − p²)`.
  To maximize `V1(p)` (least negative expected return), differentiate `V1(p)=2(p−2)/(p−p²)`:
  `dV1/dp = 2[p² − 4p + 2]/(p−p²)² = 0` ⇒ `p² − 4p + 2 = 0` ⇒ `p = 2 ± √2`.
  Only `p* = 2 − √2 ≈ 0.586` is feasible. Substituting back gives
  `V1(p*) = −2√2 / ((2 − √2)(√2 − 1)) = −6 − 4√2 ≈ −11.66`.
- **Why this works / takeaway:** function approximation forces a single stochastic policy across
  states; optimizing that one parameter gives the best achievable performance
  (`p ≈ 0.59`, `v ≈ −11.6` as reported in the book).

### Exercise 13.2 — Prove Eq. (13.7) (softmax eligibility vector)

- **Book location:** Section 13.3, PDF page 289–290.
- **Type:** Math / Proof.
- **Problem:** Prove Eq. (13.7) — `∇_θ ln π(a|s,θ) = x(s,a) − Σ_b π(b|s,θ) · x(s,b)` — using the
  definitions and elementary calculus (exponential softmax policies with linear action preferences).
- **Solution.**
  `π(a|s,θ) = exp(h(s,a,θ)) / Σ_b exp(h(s,b,θ))` with `h(s, a, θ) = θᵀx(s,a)`.
  `ln π(a|s,θ) = θᵀx(s,a) − ln Σ_b exp(θᵀx(s,b))`.
  Differentiate with respect to `θ`:
  `∇_θ ln π(a|s,θ) = x(s,a) − ∇_θ ln Σ_b exp(θᵀx(s,b))`.
  `∇_θ ln Σ_b exp(θᵀx(s,b)) = [Σ_b exp(θᵀx(s,b)) · x(s,b)] / Σ_b exp(θᵀx(s,b)) = Σ_b π(b|s,θ) x(s,b)`.
  Hence `∇_θ ln π(a|s,θ) = x(s,a) − Σ_b π(b|s,θ) x(s,b)`. ∎
- **Why this works / takeaway:** the softmax eligibility is the chosen action's feature minus the
  expected feature under the current policy — a centered gradient.

### Exercise 13.3 — Bernoulli-logistic unit

- **Book location:** Section 13.3 / 13.5, PDF page 296.
- **Type:** Math / Computation.
- **Problem (paraphrased):** A Bernoulli-logistic unit is a stochastic, neuron-like unit. Its input at time `t` is
  a feature vector `x(S_t)`; its output `A_t` is random, with `Pr{A_t = 1} = P_t` and `Pr{A_t = 0} =
  1 − P_t`. Let `h(s, 0, θ)` and `h(s, 1, θ)` be the preferences for the unit's two actions given
  policy parameter `θ`; assume the preference difference is a weighted sum of the input:
  `h(s, 1, θ) − h(s, 0, θ) = θᵀx(s)`.
  (a) Show that if the exponential softmax distribution (13.2) is used to convert preferences into
  policies, then `P_t = π(1|S_t, θ_t) = 1 / (1 + exp(−θ_tᵀx(S_t)))` (the logistic function).
  (b) What is the Monte-Carlo REINFORCE update of `θ_t` to `θ_{t+1}` upon receipt of return `G_t`?
  (c) Express the eligibility `∇_θ ln π(a|s,θ)` for a Bernoulli-logistic unit in terms of `a`,
  `x(s)`, and `π(a|s,θ)` by calculating the gradient (separately for each action, then combined).
- **Solution.**
  (a) From the softmax:
  `P = π(1|s) = exp(h1) / (exp(h0) + exp(h1)) = 1 / (1 + exp(h0 − h1)) = 1 / (1 + exp(−z))`
  where `z = θᵀx(s) = h1 − h0`. This is the logistic function `σ(z)`. ∎
  (b) REINFORCE update: `θ ← θ + α·G_t ∇_θ ln π(A_t|S_t, θ)`; for this parameterization
  (see part (c)): `θ ← θ + α·G_t (A_t − P_t) x(S_t)` (or with `γ^t` per the boxed REINFORCE
  algorithm: `θ ← θ + α·γ^t·G_t (A_t − P_t) x(S_t)`).
  (c) For `a = 1`:  `∇_θ ln P = (1/P)·∇_θ P = (1/P)·P(1−P)·x(s) = (1−P)·x(s)`.
      For `a = 0`:  `∇_θ ln(1−P) = (1/(1−P))·(−P(1−P))·x(s) = −P·x(s)`.
      Combined, for `a ∈ {0, 1}`:
      `∇_θ ln π(a|s,θ) = (a − π(1|s,θ))·x(s)`.
- **Why this works / takeaway:** the Bernoulli-logistic eligibility is `(a − P)·x(s)` — positive on
  a "1" sample, negative on a "0" (response-driven, as in Eq. 15.3 in Chapter 15).

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 13.1 | §13.1 (Example 13.1, short corridor) | *(no direct implementation in this repo; derivation-only in this document — the repo’s `PolicyEnvironment` is a multi-armed bandit, not the short-corridor MDP)* | *(none; not covered by current tests)* | — |
| 13.2 | §13.3 (Eq. 13.7, softmax eligibility) | `chapter_13/solution.py` uses the softmax score form `onehot(a) − π(·|s)` in tabular updates (supporting usage, not an implementation of the Eq. 13.7 proof) | `TestPolicyGradientTheorem` (numeric policy-gradient-theorem check, not the Eq. 13.7 derivation) | `python -m pytest chapter_13/tests/ -v` |
| 13.3 | §13.3 / §13.5 (Bernoulli-logistic unit) | *(no direct implementation in this repo; repo policies are tabular softmax / Gaussian, not a sigmoid Bernoulli-logistic unit)* | *(none)* | — |

- Full chapter command: `python -m pytest chapter_13/tests/ -v`
- Lint: `ruff check chapter_13/`
- No standalone `exercise_*.py` file exists for this chapter; the policy-gradient machinery lives in
  `solution.py` and the tests group it by algorithm. Note: 13.1 and 13.3 are derivation-only here;
  13.2 is supported conceptually by the tabular softmax score form but its proof is not directly tested.

---

## Verification and Quality Notes

- The worked arithmetic in this document (e.g., Ex. 13.1: `p* = 2 − √2 ≈ 0.586`, `V(p*) ≈ −11.66`)
  was re-derived by hand; the public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`)
  was used only as a cross-check.
- The short-corridor MDP in Ex. 13.1 is reconstructed from the book's description (Example 13.1,
  PDF pages 284–285); the derived optimal probability `p* ≈ 0.586` matches the book's reported
  "about 0.59" and optimal value "about −11.6".

---

*Exercises document for Chapter 13 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
