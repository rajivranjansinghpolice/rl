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

**Hook.** Before any algorithm can be applied, the operations problem has to be written as an MDP.
Once the state, action, and cost are pinned down, the Bellman equation follows mechanically.

**Formal bridge.** The inventory problem becomes a finite MDP with states, actions, stochastic
demand, and a cost-based reward:

\[
D\sim\text{Poisson}(\lambda),\quad o=\max(0,a-s),\quad L=\max(0,s+o-D),\quad U=\max(0,D-(s+o)),\quad s'=\min(L,s_{\max}),
\tag{16.1.1}
\]

\[
R = -(c_o\,o + h\,L + p_u\,U), \qquad
Q^*(s,a)=\mathbb{E}\!\left[R + \gamma\max_{a'} Q^*(s',a') \,\middle|\, s,a\right], \qquad V^*(s)=\max_a Q^*(s,a).
\tag{16.1.2}
\]

where \(s\in\{0,\dots,s_{\max}\}\) is the inventory level at the period start, \(a\in\{0,\dots,s_{\max}\}\)
is the order-up-to level, \(o\) is the number of units ordered, \(D\) is demand, \(L\) is leftover
stock, \(U\) is unmet demand, \(c_o\) is unit order cost, \(h\) the holding cost, \(p_u\) the shortage
cost, \(\gamma\) the discount, and \(s'\) the next state.

**Reasoning chain.** **State:** the inventory level \(s\in\{0,\dots,s_{\max}\}\) at the start of a
period. **Action:** the order-up-to level \(a\in\{0,\dots,s_{\max}\}\), so the amount actually ordered
is \(\max(0,a-s)\). **Dynamics:** demand \(D\sim\text{Poisson}(\lambda)\); leftover is
\(L=\max(0,s+o-D)\), unmet demand is \(U=\max(0,D-(s+o))\), and the next state is
\(s'=\min(L,s_{\max})\). **Reward:** \(R=-(c_o o + h L + p_u U)\), the negative of ordering,
holding, and shortage costs.

With those pieces, the Bellman optimality equation is (16.1.2): take the expectation of immediate
reward plus the discounted value of the best future action, and maximize over the current action.

For the **representation**: with a small discrete \(s_{\max}\) the state and action sets are finite,
so the problem is tabular and exact DP or TD is feasible. Large catalogs or continuous inventory
levels would motivate function approximation, because a table can no longer cover the state space
and visited states must generalize to unvisited ones.

**Why it makes sense.** A finite \(s_{\max}\) and finitely many order-up-to levels make the Bellman
equations a finite system that can be backed up exactly; there is no approximation error in the
representation. As \(s_{\max}\) grows, the table grows with it and each state receives fewer visits,
so learning becomes slow. In the limit of a continuous inventory level a table is impossible —
there are infinitely many states — and features are the only workable representation.

**Check your understanding.** Prediction: double \(s_{\max}\). The table roughly doubles in size, so
each DP sweep costs more and each state needs more visits to be learned accurately. Push the state
to a continuous range and tabular methods fail outright; function approximation becomes mandatory.

### Derived 2 — Optimal order-up-to structure (inventory) [derived]

- **Book location:** Sections 4.3–4.4, PDF pages 76–83; Chapter 16 (applications).
- **Type:** Math / Theory.
- **Problem (paraphrased):** For convex holding/order costs, derive the Bellman optimality form and
  explain why an order-up-to (base-stock) policy is optimal; describe how to verify empirically.

**Hook.** Dynamic programming can tell us the *shape* of a good policy before we compute any
numbers. For inventory, that shape is a single target level to which we top up.

**Formal bridge.** The discounted Bellman equation for inventory position and the resulting
base-stock policy are

\[
V_t(s)=\max_a \mathbb{E}\!\left[-c_o(a-s)^+ - h(L) - p(U) + \gamma V_{t+1}(s')\right], \qquad a^*(s)=\max(s,S^*).
\tag{16.2.1}
\]

where \(s\) is the current inventory, \(a\) the chosen order-up-to level, \((a-s)^+\) the units
ordered, \(L\) leftover stock, \(U\) unmet demand, \(h\) and \(p\) the convex holding and shortage
costs, \(\gamma\) the discount, \(s'\) the next state, and \(S^*\) the state-independent base-stock
target.

**Reasoning chain.** Under standard settings the one-period cost in inventory position is convex —
linear or convex holding and shortage costs — and dynamic programming preserves convexity of the
cost-to-go. That preservation is the key step: if \(V_{t+1}\) is convex, the right-hand side of
(16.2.1) is a convex function of the order-up-to level, so its minimizer is characterized by a
single target.

This yields a **base-stock / order-up-to** structure: there exists \(S^*\) with optimal action
\(a^*(s)=\max(s,S^*)\) — order up to \(S^*\) when below it, and order nothing when already at or
above it. In the finite-horizon/discounted Bellman equation in (16.2.1), the minimizer in inventory
position is state-independent under these conditions.

**Empirical check in this repo:** train and evaluate `sarsa_inventory` and inspect/evaluate the
greedy policy via `evaluate_inventory_policy`. A full policy-iteration routine is not implemented
in the repo API, so the structural prediction is checked by looking at the learned greedy action
rather than by exact iteration.

**Why it makes sense.** With convex costs, being below the target gets progressively more painful
as the gap grows, and so does being above it; the optimum is therefore to close the gap to one
number. Look at the extremes: if \(h\) is huge, holding is so costly that \(S^*\) falls and the agent
orders less; if \(p\) is huge, stockouts dominate and \(S^*\) rises. If the cost were concave (say a
discount for large orders), the optimum could be lumpy instead of a single target — the structure
depends on convexity, which is why that assumption matters.

**Check your understanding.** Prediction: raise the holding cost \(h\) in (16.2.1). The target
\(S^*\) moves downward because surplus inventory is more expensive. Raise the shortage cost \(p\)
instead and \(S^*\) moves upward, because running out is now the bigger penalty.

### Derived 3 — Single-server queue control MDP (service speed) [derived]

- **Book location:** Sections 4.3–4.4, PDF pages 76–83; Chapter 16 (system applications).
- **Type:** Math / Theory.
- **Problem (paraphrased):** Formulate the **implemented** queue control MDP (`QueueEnv`) as a finite MDP,
  write its Bellman equation, and explain when expediting becomes preferable as congestion grows.

**Hook.** Expediting service costs more right now but reduces waiting. The Bellman equation tells us
how that trade changes as the queue grows.

**Formal bridge.** The implemented queue MDP has queue length as state and a binary service-speed
action:

\[
A\sim\text{Poisson}(\lambda_{\text{arr}}),\qquad q'=\min\!\bigl(q_{\max},\max(0,q-\text{served})+A\bigr),\qquad R=-(h\,q + c(a,q)),
\tag{16.3.1}
\]

\[
Q^*(q,a)=\mathbb{E}\!\left[R + \gamma\max_{a'} Q^*(q',a') \,\middle|\, q,a\right].
\tag{16.3.2}
\]

where \(q\in\{0,\dots,q_{\max}\}\) is the number of customers in the system, \(a=0\) is normal and
\(a=1\) expedited service with \(p_{\text{expedited}}\ge p_{\text{normal}}\) completion probability
and \(c_{\text{expedited}}\ge c_{\text{normal}}\) cost, \(A\) is the arrival process, \(h\,q\) is the
waiting cost, \(\gamma\) the discount, and \(q'\) the next queue length.

**Reasoning chain.** **State:** the queue length \(q\in\{0,\dots,q_{\max}\}\). **Actions:** \(a=0\)
normal service, \(a=1\) expedited service. **Dynamics:** if \(q>0\), a completion occurs with
probability \(p(a)\), where \(p_{\text{expedited}}\ge p_{\text{normal}}\), at service cost \(c(a)\)
with \(c_{\text{expedited}}\ge c_{\text{normal}}\); arrivals are \(A\sim\text{Poisson}(\lambda_{\text{arr}})\),
and \(q'=\min(q_{\max},\max(0,q-\text{served})+A)\). **Reward:**
\(R=-(h q+c(a,q))\), the waiting cost plus the service cost. The Bellman optimality equation is
then (16.3.2).

As \(q\) grows, the holding/waiting cost \(h q\) dominates, so the marginal value of faster service
increases. Faster service clears more customers per step, reducing the large waiting penalty, so
expediting eventually pays for its higher \(c(a)\). This typically induces a **threshold-like**
policy — expedite when \(q\) is high enough — depending on the cost and probability parameters.

**Why it makes sense.** At small \(q\) the waiting cost is small, so paying the extra expedited fee
is not worth it; at large \(q\) the per-step waiting penalty is large and the extra completion more
than repays the fee. That is why the answer is a threshold rather than "always expedite" or "never
expedite." If expedited service were both cheaper and faster, the threshold would fall to zero and
the agent would always expedite; if it were no faster, expediting would never be attractive.

**Check your understanding.** Prediction: raise the holding coefficient \(h\). The threshold moves
lower, so the agent expedites at shorter queues. Raise \(c_{\text{expedited}}\) instead and the
threshold moves higher, because faster service now costs more.

### Derived 4 — Comparing DP and TD on applied problems [derived]

- **Book location:** Section 4.5, PDF page 86–87; Chapter 16.
- **Type:** Theory / Computation.
- **Problem (paraphrased):** Compare DP (value iteration/policy iteration) and TD/Sarsa for moderate
  inventory/queue MDPs; when is asynchronous DP preferable?

**Hook.** Both methods solve the same Bellman equations, but they differ in where the expectation
comes from: a known model or sampled experience. That choice drives cost, data needs, and where
computation is best spent.

**Formal bridge.** DP backs up with the model's full expectation, while TD backs up from one
sample:

\[
V_{k+1}(s)=\sum_{s',r} p(s',r\mid s,a)\bigl[r+\gamma V_k(s')\bigr], \qquad
V(S_t)\leftarrow V(S_t)+\alpha\bigl[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\bigr].
\tag{16.4.1}
\]

where \(p(s',r\mid s,a)\) is the known dynamics model used by DP, \(V_k\) is the value estimate at
sweep \(k\), \(R_{t+1}\) and \(S_{t+1}\) are a sampled reward and next state, \(\gamma\) is the
discount, and \(\alpha\) is the TD step size.

**Reasoning chain.** **DP** requires a known model \(p(s',r|s,a)\) and performs sweeps or backups
over states, as in (16.4.1). It is sample-efficient in terms of data because it uses expectations
exactly, but it can be expensive when the state/action space is large, since each backup sums over
all outcomes.

**TD/Sarsa** is model-free, learning from sampled transitions. It handles unknown or simulated
dynamics but needs many samples and exploration to approximate those same expectations.

**Asynchronous DP** updates selected states in an arbitrary order, for example prioritized or
real-time updates, focusing computation where values are inaccurate or relevant. It is preferable
when full sweeps are wasteful or states have uneven importance — for instance when only a few
inventory levels are actually reachable under the current policy.

**Why it makes sense.** The model-based backup spends computation to avoid sampling noise; the
sample-based backup spends samples to avoid needing the model. If the model is known and the state
space is small, DP converges in a few sweeps and TD would waste transitions. If the model is
unknown or the space is huge, DP is unavailable or too expensive and TD is the practical choice.
Asynchronous DP sits between them: it keeps exact backups but skips states whose values will not
change, which is sensible when importance is uneven.

**Check your understanding.** Prediction: give the agent a perfect model of a tiny inventory MDP.
DP converges in a handful of sweeps and TD is wasteful. Remove the model and enlarge the state
space, and TD/Sarsa becomes the only workable option, at the price of many more samples.

### Derived 5 — Case-study design patterns across applications [derived]

- **Book location:** Sections 16.2–16.7, PDF pages 371–390.
- **Type:** Theory.
- **Problem (paraphrased):** Identify common design decisions across book applications (self-play,
  features, search/planning, prior learning) and state which apply to inventory/queue problems.

**Hook.** Successful applications are not just algorithms; they are a bundle of representation,
objective, and computation choices. Let's name those choices and see which ones matter here.

**Formal bridge.** Every case study is an instance of maximizing expected return over a chosen
representation of the state:

\[
\max_\pi J(\pi)=\mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\right], \qquad \hat v(s,w)\ \text{or features } \phi(s)\ \text{replace tabular } v(s).
\tag{16.5.1}
\]

where \(\pi\) is the policy, \(J(\pi)\) its expected discounted return, \(R_{t+1}\) the reward shaped
to the operational goal, \(\gamma\) the discount, and \(\phi(s)\) or \(\hat v(s,w)\) the feature or
function-approximation choice that encodes the domain design.

**Reasoning chain.** The objective in (16.5.1) leaves five design decisions. The common patterns are: (1) **domain/state design** — state abstraction
and features; (2) **model use or simulation** — self-play and simulators; (3) **search/planning** —
lookahead and MCTS; (4) **prior/learned knowledge** — supervised pretraining and value/policy
approximators; and (5) **reward/objective shaping** aligned with operational goals.

Inventory and queue control primarily use (1) careful MDP formulation and (5) cost-based rewards.
Simulation may provide samples, as TD does, but self-play and tree search are less central than in
games. Feature engineering and function approximation matter when the problem scales beyond tabular
size.

**Why it makes sense.** Games have crisp rules and cheap self-contained simulators, so self-play
and search pay off; operations problems have stochastic demand and arrivals with explicit monetary
costs, so the leverage is in formulating the MDP correctly and pricing the outcomes. The direction
follows the model: where dynamics are known and cheap to simulate, planning and search help; where
they are not, representation and reward design dominate.

**Check your understanding.** Prediction: replace stochastic demand with a learned simulator.
Patterns (2) and (3) become more relevant because planning becomes possible. Move inventory to a
huge catalog instead and feature engineering under pattern (1) dominates, since the table no longer
fits.

### Derived 6 — Risk-sensitive applied RL [derived]

- **Book location:** Section 16.8, PDF page 393.
- **Type:** Theory / Math.
- **Problem (paraphrased):** Explain why average/expected return can be inadequate for rare but severe
  events and give risk-aware metrics (e.g. CVaR) relevant to inventory stockouts or soaring.

**Hook.** Two policies can have the same average return and very different worst days. In operations
that worst day — a stockout or a crash — can be the one that matters.

**Formal bridge.** Conditional Value at Risk focuses on the worst \(\alpha\) fraction of outcomes
instead of the mean:

\[
\mathrm{CVaR}_\alpha(X)=\mathbb{E}\bigl[X \mid X \le \mathrm{VaR}_\alpha(X)\bigr],
\tag{16.6.1}
\]

where \(X\) is the loss or return random variable, \(\mathrm{VaR}_\alpha(X)\) is its \(\alpha\)-quantile
threshold, and \(\alpha\) is the tail fraction; the inequality's direction depends on the sign
convention for losses versus returns.

**Reasoning chain.** Optimizing \(\mathbb{E}[G]\) can hide tail risk: two policies with equal mean
return may differ in catastrophic failures such as stockouts or crashes. Risk-sensitive objectives
include variance penalties, worst-case/robust criteria, and **Conditional Value at Risk (CVaR)** as
in (16.6.1), which explicitly optimizes tail performance over the worst \(\alpha\) tail (with
definitions varying by sign convention).

In inventory control, CVaR-like criteria penalize severe stockouts more than mean-cost
optimization; in thermal soaring, they penalize rare land-out events. Such metrics require
estimating tail distributions, not just expectations — you need to know about the rare bad outcomes,
which a sample mean smooths away.

**Why it makes sense.** A mean is a single number that can be held up by many ordinary good days
while hiding a tiny probability of ruin. Minimizing CVaR refuses that trade: it asks how bad things
get in the worst \(\alpha\) of cases and optimizes that. The direction is conservative on purpose —
a CVaR-optimal policy may accept a slightly worse mean to avoid the catastrophic tail, which is
exactly what an operator facing stockouts or land-outs wants.

**Check your understanding.** Prediction: make a failure rarer but much more costly while keeping
the same expected cost. The mean-optimal policy is unchanged, but the CVaR-optimal policy shifts to
avoid the catastrophe, because the tail got worse even though the average did not.

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
