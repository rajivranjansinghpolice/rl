# Chapter 2 — Multi-Armed Bandits: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_02`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_02` — Multi-Armed Bandits (Algorithms 2.1–2.5: bandit simulation
  environment, action-value estimation / sample averages, ε-greedy selection, UCB selection, gradient
  bandit preferences; plus simulations for non-stationary and optimistic-initial-value settings).
- **Book chapter:** Chapter 2 — Multi-Armed Bandits.
- **Book sections covered by the chapter implementation:**
  - Section 2.1 — The k-armed bandit problem.
  - Section 2.2 — Action values and the sample-average method (estimator algorithms).
  - Section 2.3 — ε-greedy action selection.
  - Section 2.4 — Incremental implementation of sample averages (and non-stationary step sizes);
    Section 2.5 — Tracking a non-stationary problem.
  - Section 2.6 — Optimistic initial values; Section 2.7 — Upper-confidence-bound action selection;
    Section 2.8 — Gradient bandit algorithms; Section 2.9 — Associative search / contextual bandits.
- **Key book equations referenced by the exercises:**
  - Eq. (2.1) — sample average: `Q_t(a) = (Σ_{i=1}^{t} R_i · I_i(a)) / (Σ_{i=1}^{t} I_i(a))` with
    `I_i(a) = 1/n` for the sample-average case.
  - Eq. (2.2) — greedy: `A_t = argmax_a Q_t(a)`.
  - Eq. (2.8) — UCB1: `A_t = argmax_a [Q_t(a) + c·√(ln(t) / N_t(a))]`.
  - Eq. (2.10) — gradient bandit preference updates.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Chapter 2 content spans
  roughly PDF pages 30–52; the exercises were extracted from PDF pages 39, 41, 44, 45, 47, 50, and 52.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 2. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`,
  `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-2 material, plus the public
  chapter-2 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem**, **Solution**, and — where a companion implementation exists — a
  **coding-exercise pointer** into `chapter_02/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations; Programming
  exercises describe an experiment to run.
- The final section maps exercises to the repository code that already implements (or supports) the
  corresponding machinery; exercises that exercise a specific implementation also carry a
  **Coding pointer** inline.

---

## Exercises

### Exercise 2.1 — Probability of selecting the greedy action under ε-greedy

- **Book location:** Section 2.2, PDF page 39.
- **Type:** Computation.
- **Problem:** In ε-greedy action selection, for the case of two actions and ε = 0.5, what is the
  probability that the greedy action is selected?
- **Solution.** With `k = 2` actions: with probability `1 − ε` the greedy action is selected; with
  probability `ε` an action is drawn uniformly at random from all `k` actions, so the random choice is
  the greedy action with probability `1/k`. Hence
  `P(greedy) = (1 − ε)·1 + ε·(1/k) = 1 − ε + ε/k = 1 − 0.5 + 0.25 = 0.75`.
- **Why this works / takeaway:** ε-greedy explores rarely when ε is small and mostly exploits when ε
  is large.

### Exercise 2.2 — Which steps involved the ε-case?

- **Book location:** Sections 2.2–2.3, PDF page 39.
- **Type:** Computation.
- **Problem:** k = 4 bandit, ε-greedy, sample-average estimates, initial `Q(a) = 0` for all actions.
  Observed sequence: `A1 = 1, R1 = 1; A2 = 2, R2 = 1; A3 = 2, R3 = 2; A4 = 2, R4 = 2; A5 = 3, R5 = 0`.
  On which time steps did the ε-case definitely occur? On which could it possibly have occurred?
- **Solution.** The ε-case definitely occurred on any step where the action actually taken differs
  from the currently greedy action. Working through the sample averages:
  - Step 1: all `Q = 0` → tie; arbitrary pick (action 1).
  - Step 2: `Q(1) = 1 > Q(2) = 0` → greedy is action 1; action 2 was taken → **ε-case definitely occurred**.
  - Step 3: after step 2, `Q(1) = 1, Q(2) = 1` → tie; the pick of action 2 may or may not have been
    greedy → **ambiguous**.
  - Step 4: after step 3, `Q(1) = 1, Q(2) = 1.5` → greedy is action 2, and action 2 was taken → **no ε-case occurred**.
  - Step 5: `Q(2) = 2 > Q(3) = 0` → greedy is action 2; action 3 was taken → **ε-case definitely occurred**.
  The definite ε-case selections are steps 2 and 5; step 4 involved no exploration.
- **Why this works / takeaway:** whenever the realised action deviates from the greedy choice, an
  exploration draw must have happened.

### Exercise 2.3 — Best method in the 10-armed testbed (Figure 2.2)

- **Book location:** Section 2.3, PDF page 41.
- **Type:** Theory / Computation.
- **Problem:** In the comparison shown in Figure 2.2, which method will perform best in the long run in
  terms of cumulative reward and probability of selecting the best action? How much better? Express
  your answer quantitatively.
- **Solution.** Over the 2000-task 10-armed testbed: the greedy method performs substantially worse
  than ε-greedy and UCB; with sample averages, ε-greedy recovers quickly once the optimal action is
  found, while greedy never explores and plateaus. UCB achieves high probability of selecting the best
  action with a deterministic exploration strategy. Quantitative comparison uses the average reward /
  learning curves (Figure 2.2): greedy's average reward over 1000 steps is well below the optimal;
  ε-greedy (with moderate ε) reaches near-optimal reward; UCB shows the best asymptotic performance.
  In quantitative terms, the average reward over the 1000-step horizon (the area under the learning
  curve) shows greedy well below optimal, ε-greedy near optimal, and UCB best among the compared
  methods.
- **Why this works / takeaway:** exploration strategy determines the trade-off between learning speed
  and asymptotic optimality.

### Exercise 2.4 — General step-size weighting

- **Book location:** Section 2.4, PDF page 44.
- **Type:** Math.
- **Problem:** If the step-size parameters `I_n` are not constant, then the estimate `Q_n` is a weighted
  average of previously received rewards with a weighting different from that given by Eq. (2.1). What
  is the weighting on each prior reward for the general case, analogous to Eq. (2.1)?
- **Solution.** With a constant step size `I_n = I`, the incremental recursion is
  `Q_n ← (1 − I)·Q_{n−1} + I·R_n`, whose solution gives the geometric weighting
  `Q_n = Σ_{i=1}^{n} (1 − I)^{n−i} · I · R_i` — each reward `R_i` is weighted by
  `(1 − I)^{n−i}·I`.
  The sample-average estimator corresponds to step size `I_i = δ / N_i` (δ = 1 here), which yields
  equal weights `1/n`. In the general case, the weighting on each prior reward is determined by the
  sequence of step sizes `I_i`.
- **Why this works / takeaway:** step-size sequences control both the weighting and (via the second
  convergence condition) whether estimates converge.

### Exercise 2.5 — (Programming) Non-stationary experiment

- **Book location:** Section 2.5, PDF page 44.
- **Type:** Programming.
- **Problem:** Design and conduct an experiment to demonstrate the difficulties that sample-average
  methods have for non-stationary problems. Use a modified version of the 10-armed testbed in which all
  `Q(a)` start equal and then take independent random walks (e.g., add a N(0, 0.01) increment to all
  `Q(a)` each step). Prepare plots like Figure 2.2 for an action-value method using sample averages,
  incrementally computed, and another using a constant ε = 0.1 (for the ε-greedy method). Use ε = 0.1
  and longer runs, e.g., 10,000 steps.
- **Why this works / takeaway:** sample averages assume stationarity; constant step sizes handle
  non-stationarity better.

### Exercise 2.6 — Mysterious spikes in optimistic method

- **Book location:** Section 2.6, PDF page 45.
- **Type:** Theory.
- **Problem:** The results in Figure 2.3 should be reliable (averages over 2000 tasks), so why are
  there oscillations and spikes in the early part of the curve for the optimistic method?
- **Solution.** Optimistic initial values (`Q(a) = +5`) cause even greedy methods to explore heavily
  early on; with few samples the estimates are noisy, producing spikes and dips. The exploration drive
  is temporary, so performance can oscillate before settling.
- **Why this works / takeaway:** optimism induces exploration; early estimates are high-variance.

### Exercise 2.7 — Softmax equals logistic

- **Book location:** Section 2.8, PDF page 47.
- **Type:** Math / Proof.
- **Problem:** Show that in the case of two actions, the softmax distribution is the same as the
  logistic/sigmoid function used in statistics and neural networks.
- **Solution.** `p(a1) = e^{H(a1)} / (e^{H(a1)} + e^{H(a2)}) = σ(H(a1) − H(a2))`, where `σ(x) = 1/(1+e^{-x})`.
- **Why this works / takeaway:** softmax with two actions reduces exactly to the sigmoid.

### Exercise 2.8 — Associative search / contextual bandits

- **Book location:** Section 2.9, PDF page 50.
- **Type:** Theory.
- **Problem:** 2-armed bandit whose true action values change randomly (case A: actions 1/2 = 0.1/0.2;
  case B: 0.9/0.8). (a) Without cues: best expectation of success and how to behave? (b) With case
  labels each step: best expectation of success and how to behave?
- **Solution.** (a) Without cues, the cases cannot be distinguished; the agent cannot identify which
  case it faces, so the best achievable expectation of success is limited by the prior uncertainty;
  behaviour cannot exploit case information. (b) With cues, the agent tracks the case and selects the
  best action per case (action 2 in case A, action 1 in case B), achieving the optimal expected
  success.
- **Why this works / takeaway:** cues convert a non-stationary bandit into a contextual bandit.

### Exercise 2.9 — (Programming) Figure for the non-stationary case

- **Book location:** Section 2.9, PDF page 52.
- **Type:** Programming.
- **Problem:** Make a figure analogous to Figure 2.6 for the non-stationary case outlined in Exercise
  2.5. Include the constant-step-size ε-greedy algorithm with ε = 0.1. Use runs of 200,000 steps and,
  as the performance measure for each algorithm and parameter setting, use the average reward over the
  last 100,000 steps.
- **Why this works / takeaway:** non-stationary problems favour constant step sizes over sample
  averages.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 2.1–2.2 | §2.2–2.3 | `chapter_02/solution.py` (`BanditProblem`, `ActionValueEstimator`) | `TestBanditProblem`, `TestActionValueEstimator` | `python -m pytest chapter_02/tests/ -v` |
| 2.3 | §2.3 | `select_epsilon_greedy` | `TestEpsilonGreedy` | `python -m pytest chapter_02/tests/ -v` |
| 2.4 | §2.4 | `ActionValueEstimator`, `simulate_nonstationary` | `TestActionValueEstimator`, `TestNonstationary` | `python -m pytest chapter_02/tests/ -v` |
| 2.5 | §2.5 | `chapter_02/solution.py` (`simulate_nonstationary`) | `TestNonstationary` | `python -m pytest chapter_02/tests/ -v` |
| 2.6 | §2.6 | `simulate_optimistic_initial_values` | `TestOptimisticInitialValues` | `python -m pytest chapter_02/tests/ -v` |
| 2.7 | §2.8 | `chapter_02/solution.py` (`select_softmax`, `GradientBanditAgent`) | `TestGradientBandit` | `python -m pytest chapter_02/tests/ -v` |
| 2.8 | §2.9 | `ContextualBandit`, `simulate_*` | `TestContextualBandit` | `python -m pytest chapter_02/tests/ -v` |
| 2.9 | §2.9 | `chapter_02/solution.py` (`simulate_nonstationary`) | `TestNonstationary` | `python -m pytest chapter_02/tests/ -v` |

- Full chapter command: `python -m pytest chapter_02/tests/ -v`
- Lint: `ruff check chapter_02/`
- No standalone `exercise_*.py` bandit file exists; the chapter implements the bandit machinery in
  `solution.py` (Algorithms 2.1–2.5).

---

## Verification and Quality Notes

- The worked arithmetic in this document (e.g., Ex. 2.1: `P(greedy) = 0.75`) was re-derived by hand;
  the public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`) was used only as a
  cross-check.
- Programming-exercise answers (2.5, 2.9) describe experiments to run; the underlying code paths are
  mapped to the repository simulations.

---

*Exercises document for Chapter 2 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
