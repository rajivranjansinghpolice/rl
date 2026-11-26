# Chapter 16 — Applications and Case Studies: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_16`
> Book chapter: **Chapter 16** — Applications and Case Studies

*Book contains no formal numbered exercises for this chapter.* The prompts below are derived from
the applied case studies and the application-relevant sections, labeled **[derived]**.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_16` — Applications (inventory control `InventoryEnv` with `sarsa_inventory`
  / `evaluate_inventory_policy`, and queue control `QueueEnv` with `q_learning_queue` /
  `evaluate_queue_policy`).
- **Book chapter:** Chapter 16 — Applications and Case Studies (applied RL in service/management
  problems; see also the DP/MDP formulations of Chapters 3–4 and the applied agents of Chapter 16).
- **Book sections covered:**
  - §16.1 TD-Gammon; §16.2 Samuel's Checkers Player; §16.3 Watson's Daily-Double Wagering;
    §16.4 Optimizing Memory Control; §16.5 Human-level Video Game Play; §16.6 Mastering Go;
    §16.7 Personalized Web Services; §16.8 Thermal Soaring.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Chapter 16 starts on PDF page
  367 (§16.1); §16.4 on PDF page 376; §16.7 on PDF page 390; §16.8 on PDF page 393.
- **Mapping note:** the repo implements inventory and queue control — classical operations-research
  applied RL problems that illustrate the same formulation/DP methodology as the book's case studies.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 16. Section statements paraphrased; short verbatim references only.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction`.
- All analyses are **rewritten from scratch**; none of the public PDFs are copied.

---

## Derived Exercises

### Derived 1 — Formulating an applied problem as a finite MDP [derived]

- **Book location:** Section 16.1, PDF page 367 (case-study methodology).
- **Type:** Math / Theory.
- **Problem (paraphrased):** Formulate the repo’s inventory control problem as a finite MDP: give state,
  action, reward, write the Bellman optimality equation, and justify tabular vs function approximation.
- **Solution.** **State:** inventory level `s ∈ {0,…,s_max}` at period start. **Action:** order-up-to
  level `a ∈ {0,…,s_max}` (order \(max(0,a-s)\) units). **Dynamics:** demand \(D ~ Poisson(\lambda )\); leftover
  \(L=max(0, s+o-D)\), unmet \(U=max(0, D-(s+o))\), next state `s'=min(L,s_max)`. **Reward:**
  \(R = -(c_o\cdot o + h\cdot L + p_u\cdot U)\). **Bellman optimality:**
  \(Q*(s,a)=E[R + \gamma max_{a'} Q*(s',a') | s,a]\), `V*(s)=max_a Q*(s,a)`.
  **Representation:** with small discrete `s_max` the problem is tabular (exact DP/TD feasible); large
  catalogs or continuous inventory would motivate function approximation.
- **Explanation (student-friendly).** a clear (S,A,R,P,γ) formulation enables DP/RL methods.

### Derived 2 — Optimal order-up-to structure (inventory) [derived]

- **Book location:** Sections 4.3–4.4, PDF pages 76–83; Chapter 16 (applications).
- **Type:** Math / Theory.
- **Problem (paraphrased):** For convex holding/order costs, derive the Bellman optimality form and
  explain why an order-up-to (base-stock) policy is optimal; describe how to verify empirically.
- **Solution.** The one-period cost in inventory position is convex in many standard settings (linear/convex
  holding and shortage costs), and dynamic programming preserves convexity of the cost-to-go. This yields
  a **base-stock / order-up-to** structure: there exists `S*` with optimal action `a*(s)=max(s,S*)`
  (order up to `S*` when below, otherwise order nothing). In the finite-horizon/discounted Bellman
  equation, \(V_t(s)=max_a E[-c_o(a-s)^+ - h(L) - p(U) + \gamma V_{t+1}(s')]\), the minimizer in inventory
  position is state-independent under these conditions. **Empirical check in this repo:** train/evaluate
  `sarsa_inventory` and inspect/evaluate the greedy policy via `evaluate_inventory_policy` (a full
  policy-iteration routine is not implemented in the repo API).
- **Explanation (student-friendly).** DP can characterize optimal policy *structure*, not just values.

### Derived 3 — Single-server queue control MDP (service speed) [derived]

- **Book location:** Sections 4.3–4.4, PDF pages 76–83; Chapter 16 (system applications).
- **Type:** Math / Theory.
- **Problem (paraphrased):** Formulate the **implemented** queue control MDP (`QueueEnv`) as a finite MDP,
  write its Bellman equation, and explain when expediting becomes preferable as congestion grows.
- **Solution.** **State:** queue length `q ∈ {0,…,q_max}` (customers in system). **Actions:**
  `a=0` normal service, `a=1` expedited service. **Dynamics:** if `q>0`, a completion occurs with
  probability `p(a)` (\(p_expedited \ge p_normal\)) at service cost `c(a)` (\(c_expedited \ge c_normal\));
  arrivals \(A ~ Poisson(\lambda _arr)\); \(q' = min(q_max, max(0,q-served)+A)\). **Reward:**
  \(R = -(h\cdot q + c(a,q))\) (waiting cost plus service cost). **Bellman optimality:**
  \(Q*(q,a)=E[R + \gamma max_{a'} Q*(q',a') | q,a]\).
  As `q` grows, holding/waiting cost \(h\cdot q\) dominates, so the marginal value of faster service
  increases; this typically induces a **threshold-like** policy (expedite when `q` is high enough),
  depending on cost/probability parameters.
- **Explanation (student-friendly).** control actions trade immediate service cost against congestion cost.

### Derived 4 — Comparing DP and TD on applied problems [derived]

- **Book location:** Section 4.5, PDF page 86–87; Chapter 16.
- **Type:** Theory / Computation.
- **Problem (paraphrased):** Compare DP (value iteration/policy iteration) and TD/Sarsa for moderate
  inventory/queue MDPs; when is asynchronous DP preferable?
- **Solution.** **DP** requires a known model `p(s',r|s,a)` and performs sweeps/backups over states; it is
  sample-efficient in terms of data (uses expectations exactly) but can be expensive when the state/action
  space is large. **TD/Sarsa** is model-free, learning from sampled transitions; it handles unknown or
  simulated dynamics but needs many samples and exploration. **Asynchronous DP** updates selected states
  in arbitrary order (e.g. prioritized or real-time), focusing computation where values are inaccurate or
  relevant—preferable when full sweeps are wasteful or states have uneven importance.
- **Explanation (student-friendly).** choose methods based on model availability, state-space size, and compute.

### Derived 5 — Case-study design patterns across applications [derived]

- **Book location:** Sections 16.2–16.7, PDF pages 371–390.
- **Type:** Theory.
- **Problem (paraphrased):** Identify common design decisions across book applications (self-play,
  features, search/planning, prior learning) and state which apply to inventory/queue problems.
- **Solution.** Common patterns: (1) **domain/state design** (state abstraction/features), (2) **model use
  or simulation** (self-play/simulators), (3) **search/planning** (lookahead, MCTS), (4) **prior/learned
  knowledge** (supervised pretraining, value/policy approximators), and (5) **reward/objective shaping**
  aligned with operational goals. Inventory/queue control primarily uses (1) careful MDP formulation and
  (5) cost-based rewards; simulation may provide samples (like TD), but self-play and tree search are less
  central than in games. Feature engineering/function approximation matters when scaling beyond tabular size.
- **Explanation (student-friendly).** successful applications decompose problems into representation, objective,
  and computation choices.

### Derived 6 — Risk-sensitive applied RL [derived]

- **Book location:** Section 16.8, PDF page 393.
- **Type:** Theory / Math.
- **Problem (paraphrased):** Explain why average/expected return can be inadequate for rare but severe
  events and give risk-aware metrics (e.g. CVaR) relevant to inventory stockouts or soaring.
- **Solution.** Optimizing `E[G]` can hide tail risk: two policies with equal mean return may differ in
  catastrophic failures (stockouts, crashes). Risk-sensitive objectives include variance penalties,
  worst-case/robust criteria, and **Conditional Value at Risk (CVaR)**:
  \(CVaR_\alpha (X)=E[X | X \le VaR_\alpha (X)]\) for losses/returns in the worst \(\alpha\) tail (definitions vary by sign
  convention), which explicitly optimizes tail performance. In inventory control, CVaR-like criteria
  penalize severe stockouts more than mean-cost optimization; in thermal soaring, they penalize rare
  land-out events. Such metrics require estimating tail distributions, not just expectations.
- **Explanation (student-friendly).** operational reliability often depends on tail behavior, not averages.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 1–2, 4 | §4.3–4.5 / §16.1 | `chapter_16/solution.py` (`InventoryEnv`, `sarsa_inventory`, `evaluate_inventory_policy`, `greedy_inventory_action`) — MDP formulation and learned-policy evaluation/TD (no policy-iteration routine) | `TestInventoryEnv`, `TestSarsaInventory` | `python -m pytest chapter_16/tests/ -v` |
| Derived 3–4 | §4.3–4.5 / §16.7 | `chapter_16/solution.py` (`QueueEnv`, `q_learning_queue`, `evaluate_queue_policy`, `greedy_queue_action`, `bellman_target`) — service-speed control MDP (not admission control) | `TestQueueEnv`, `TestQLearningQueue` | `python -m pytest chapter_16/tests/ -v` |
| Derived 5–6 | §16.2–16.8 | *Theory/design discussion grounded in book applications; no direct dedicated repo implementation* | — | — |

- Full chapter command: `python -m pytest chapter_16/tests/ -v`
- Lint: `ruff check chapter_16/`

- Full chapter command: `python -m pytest chapter_16/tests/ -v`
- Lint: `ruff check chapter_16/`

---

## Verification and Quality Notes

- Derived analyses were re-derived independently; public case-study material is used only as a
  cross-check. The inventory `(s, S)` structure and queue threshold follow the Bellman/policy
  iteration method documented in Chapter 4, which the repo code validates.

---

*Chapter 16 (Applications) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
