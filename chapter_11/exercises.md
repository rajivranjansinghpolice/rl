# Chapter 11 — Exploration: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_11`
> Book sections: **Section 2.6** — Optimistic Initial Values · **Section 2.7** — Upper-Confidence-Bound Action Selection

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_11` — Exploration (UCB action selection `ucb_bonus` / `ucb_select`,
  optimism in the face of uncertainty, optimistic initialization `run_optimistic_greedy_bandit`,
  descending bonuses `run_descending_bonus_bandit`, Q-learning with exploration variants
  `q_learning_ucb`, `q_learning_optimistic`, `q_learning_descending_bonus`).
- **Book chapter:** Sections 2.6 and 2.7 of Chapter 2 — Multi-Armed Bandits.
- **Book sections covered:**
  - Section 2.6 — Optimistic initial values (bias introduced/removed by initialization; the trick
    and its limits).
  - Section 2.7 — Upper-confidence-bound action selection (Eq. 2.8, Figure 2.4).
- **Key book equations referenced:**
  - Eq. (2.1)/(2.6) — sample-average estimator and constant-step-size weighting (bias decay).
  - Eq. (2.8) — UCB index: `At = argmax_a [Q_t(a) + c·sqrt(ln t / N_t(a))]`.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Sections 2.6–2.7 are on PDF
  pages 44–46.
- **Exercise origin:** the book has **no formal numbered exercises in Sections 2.6–2.7**; the
  prompts below are derived from the sections themselves and are labeled **[derived]**.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Sections 2.6–2.7. Section statements below are paraphrased; short verbatim references only.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-2 material, plus the public
  chapter-2 solution PDFs. Any disagreement is resolved by independent derivation.
- All worked derivations are **rewritten from scratch**. All derived prompts in this document are
  paraphrased formulations of the cited sections (not verbatim book exercise text).

## How to Use This Document

- Each derived prompt lists: **Book location**, **Type** (`Theory`, `Math`, `Proof`,
  `Computation`, `Programming`), **Problem**, **Solution**, and — where a companion
  implementation exists — a **coding pointer** into `chapter_11/`.
- All derived answers are self-contained; no content from the public solution PDFs is copied.
- The final section maps derived prompts to repository code.

---

## Derived Exercises

### Derived 1 — Bias of sample-average vs constant-step-size estimators [derived]

- **Book location:** Section 2.6, PDF pages 44–45.
- **Type:** Math / Theory.
- **Problem:** Sample-average returns treat the past uniformly; the constant-step-size estimator
  weights recent rewards more. Show that sample-average estimates are unbiased once every action has
  been selected at least once, while constant-step-size estimates carry bias that decreases with time.
- **Solution.** `Q_n(a) = (1/n)·Σ_{i=1}^{n} R_i` is an unbiased estimator of `q*(a)` once
  `n ≥ 1` under stationary rewards. With constant step size `α` and initial estimate `Q1`, the
  incremental update `Q_{k+1} = Q_k + α(R_k − Q_k)` unrolls to
  `Q_{n+1} = (1−α)^n Q1 + Σ_{i=1}^{n} α(1−α)^{n−i} R_i`. The reward weights sum to
  `Σ_{i=1}^{n} α(1−α)^{n−i} = 1 − (1−α)^n`, so the remaining weight `(1−α)^n` stays on the
  initial estimate. For stationary rewards with `E[R_i]=q*(a)`,
  `E[Q_{n+1}] − q*(a) = (1−α)^n (Q1 − q*(a))`: the initial bias decays geometrically as `(1−α)^n`
  but is nonzero for finite `n` (and never fully “forgets” in the sample-mean sense, since recent
  rewards are weighted more). This recency weighting is why constant `α` tracks nonstationary
  problems better than sample averages.
- **Why this works / takeaway:** sample averages are unbiased in the stationary case; constant-step
  estimates trade a decaying initial bias for responsiveness to recent changes.

### Derived 2 — Experiment isolating optimistic initialization [derived]

- **Book location:** Section 2.6, PDF page 44.
- **Type:** Programming / Theory.
- **Problem:** Design an experiment to separate the effect of optimistic initialization from other
  exploration mechanisms. Predict which early steps' performance will differ between the optimistic
  greedy method and an ε-greedy method on the 10-armed testbed.
- **Why this works / takeaway:** the two mechanisms act at different times — optimism is front-loaded.

### Derived 3 — UCB index from a confidence argument [derived]

- **Book location:** Section 2.7, PDF pages 45–46.
- **Type:** Math / Proof.
- **Problem:** Derive the UCB index `Q_t(a) + c·sqrt(ln t / N_t(a))` from a Chernoff/Hoeffding-style
  confidence bound. Explain why `ln t` (and not `t`) appears and why untried actions are preferred.
- **Solution.** With bounded rewards in [0,1], Hoeffding's inequality gives
  `P(|Q̂_t(a) − q*(a)| > ε) ≤ 2·exp(−2N_t(a)ε²)`. Set a per-(action,time) tail budget
  `δ_t` with `2·exp(−2N_t(a)ε²) = δ_t`, so `ε_t(a) = sqrt(ln(2/δ_t) / (2N_t(a)))`.
  Choosing a summable schedule (e.g. `δ_t ∝ 1/t²`) yields `ε_t(a) = O(sqrt(ln t / N_t(a)))`,
  i.e. the UCB bonus `c·sqrt(ln t / N_t(a))`. Because `Σ_t δ_t < ∞`, a union bound over time
  (and actions) implies the true value stays below `Q̂_t(a)+ε_t(a)` with high probability uniformly
  often enough that suboptimal actions are pulled only `O(log t)` times. `ln t` (not `t`) is the
  right scale: it grows slowly enough to keep exploration efficient while still being summable as a
  failure budget. Untried actions (`N_t(a)=0`) are treated as maximizing to force initial sampling.
- **Why this works / takeaway:** UCB commits to exploring each action `O(log t)` times, giving
  near-optimal regret on stationary problems.

### Derived 4 — Every action is explored infinitely often under UCB, but rarely on average [derived]

- **Book location:** Section 2.7, PDF page 46.
- **Type:** Math / Theory.
- **Problem:** Show that under the UCB rule every action is selected infinitely often, but the index
  gap for suboptimal actions grows so that their selection frequency decreases to zero.
- **Why this works / takeaway:** the `sqrt(ln t / N)` term is unbounded as t → ∞, so any action eventually
  gets tried; but each pull of a suboptimal action raises its `N`, shrinking the bonus.

### Derived 5 — Optimistic initialization vs UCB: practical trade-offs [derived]

- **Book location:** Sections 2.6–2.7, PDF pages 44–46.
- **Type:** Theory.
- **Problem:** Compare optimistic initialization and UCB on (a) stationary vs nonstationary problems,
  (b) simplicity/intuition, (c) when each fails. Which would you use, and when?
- **Why this works / takeaway:** optimism helps early exploration; UCB manages exploration provably in
  stationary settings. Neither is suited, without modification, to nonstationary tasks.

### Derived 6 — Why UCB does not extend trivially to large state spaces / function approximation [derived]

- **Book location:** Section 2.7, PDF page 46.
- **Type:** Theory / Math.
- **Problem:** Explain why UCB1 is hard to extend beyond bandits (large state spaces, function
  approximation). Describe the general principle of optimistic value functions (OFU — optimism in the
  face of uncertainty) and its common tabular/linear instantiation (LinUCB); give its confidence set.
- **Why this works / takeaway:** UCB counts samples per (state,action); with function approximation,
  confidence must be carried by a weight-vector covariance matrix instead (the OFU/LinUCB principle).

### Derived 7 — The "disappointment" dynamic of optimistic initialization [derived]

- **Book location:** Section 2.6, PDF pages 44–45.
- **Type:** Math / Theory.
- **Problem:** Quantify how many steps optimistic initialization drives exploration: with `Q1 = +5`
  and `q*(a) ~ N(0, 1)`, every greedy pick starts above the true value, so the learner switches until
  rewards reveal lower values; estimate how long this lasts and why.
- **Why this works / takeaway:** optimism is self-terminating on stationary tasks; the drive for
  exploration disappears once all estimates drop below the initial optimism.

### Derived 8 — The tie-breaking rule for untried actions [derived]

- **Book location:** Section 2.7, PDF page 45–46.
- **Type:** Theory.
- **Problem:** In the book's UCB rule, actions with `N_t(a) = 0` are considered maximizing actions.
  Explain why this rule is necessary and what could go wrong without it.
- **Why this works / takeaway:** untried actions carry no information; mandating their selection
  removes the `N = 0` division ambiguity and guarantees every action is tried once.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 1–2, 7–8 | §2.6 | `chapter_11/solution.py` (`TenArmedBandit`, `run_optimistic_greedy_bandit`, `run_epsilon_greedy_bandit`) | `TestTenArmedBandit`, `TestOptimisticInitialization` (behavioral support for optimism effects; D1, D7, D8 are primarily theory/derivation in this doc) | `python -m pytest chapter_11/tests/ -v` |
| Derived 3–4 | §2.7 | `chapter_11/solution.py` (`ucb_bonus`, `ucb_select`, `run_ucb_bandit`) | `TestUCBBonus`, `TestUCBBandit` | `python -m pytest chapter_11/tests/ -v` |
| Derived 5–6 | §2.6–2.7 | `chapter_11/solution.py` (`q_learning_ucb`, `q_learning_optimistic`, `q_learning_descending_bonus`) | `TestUCBMDP`, `TestDescendingBonus` (related exploration mechanics; D5–D6 trade-offs/OFU extensions are largely theory-only) | `python -m pytest chapter_11/tests/ -v` |

- Full chapter command: `python -m pytest chapter_11/tests/ -v`
- Lint: `ruff check chapter_11/`

---

## Verification and Quality Notes

- Derived answers were re-derived from first principles; public solution material used only as a
  cross-check. Derivations of the Hoeffding bound and the bias decay follow standard concentration
  inequalities (not copied).

---

*Chapter 11 (Exploration) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
