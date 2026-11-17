# Chapter 4 — Dynamic Programming: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_04`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_04` — Dynamic Programming (Algorithms 4.1–4.4: iterative policy
  evaluation `policy_evaluation`, policy improvement `policy_improvement`, policy iteration
  `policy_iteration`, value iteration `value_iteration`; gambler's problem `gamblers_problem`).
- **Book chapter:** Chapter 4 — Dynamic Programming.
- **Book sections covered by the chapter implementation:**
  - Section 4.1 — Policy evaluation (prediction), Example 4.1 gridworld.
  - Section 4.2 — Policy improvement.
  - Section 4.3 — Policy iteration (Jack's car rental is Example 4.2).
  - Section 4.4 — Value iteration (boxed algorithm, Eq. 4.10; Example 4.3 gambler's problem).
  - Section 4.5 — Asynchronous dynamic programming.
- **Key book equations referenced by the exercises:**
  - Eq. (4.3) — Bellman expectation equation for `v_π`.
  - Eq. (4.4)/(4.5) — iterative policy evaluation: `v_{k+1}(s)=Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γ v_k(s')]`.
  - Eq. (4.9) — action-value Bellman expectation: `q_π(s,a)=Σ_{s',r} p(s',r|s,a)[r + γ Σ_{a'} π(a'|s') q_π(s',a')]`.
  - Eq. (4.10) — value iteration: `v_{k+1}(s)=max_a Σ_{s',r} p(s',r|s,a)[r + γ v_k(s')]`.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction, approximate):** Chapter 4 spans
  roughly PDF pages 60–87; the exercises were extracted from approximately PDF pages 79, 80, 83, 84, 86, and 87.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 4. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-4 material, plus public chapter-4
  solution PDFs. Any disagreement is resolved by independent derivation; discrepancies found during
  cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem**, **Solution**, and — where a companion implementation exists — a
  **coding pointer** into `chapter_04/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations; Programming
  exercises describe an experiment to run.

---

## Exercises

### Exercise 4.1 — Action values in the 4×4 gridworld

- **Book location:** Section 4.1 / Example 4.1, PDF p. ~79.
- **Type:** Computation.
- **Problem (paraphrased):** In Example 4.1, under the equiprobable random policy `π`, what are
  `q_π(11, down)` and `q_π(7, down)`?
- **Solution.** Use `q_π(s,a)=Σ_{s',r} p(s',r|s,a)[r + γ v_π(s')]` (Eq. 4.6). In this undiscounted
  episodic gridworld, `γ=1`, every transition gives reward `r=−1` unless already terminal, and the
  terminal state has value 0.
  - **State 11, action down:** moving down from state 11 goes directly to the terminal state, so
    `q_π(11, down)= −1 + γ·v_π(T)= −1 + 0 = **−1**`.
  - **State 7, action down:** moving down from state 7 leads to state 11, so
    `q_π(7, down)= −1 + γ·v_π(11)`. From Figure 4.1 (converged values for the random policy),
    `v_π(11) ≈ −14`, hence `q_π(7, down) ≈ −1 + (−14) = **−15**`.
- **Why this works / takeaway:** an action value is one-step reward plus the discounted value of the
  successor state; terminal transitions yield only the immediate reward.

### Exercise 4.2 — Value of a newly added state

- **Book location:** Section 4.1 / Example 4.1, PDF pp. ~79–80.
- **Type:** Computation.
- **Problem (paraphrased):** A new state 15 is added below state 13. In case (a), transitions from
  original states are unchanged and actions from state 15 go left/up/right/down to states
  12/13/14/15 respectively. In case (b), the dynamics of state 13 also change so that action down
  from state 13 leads to state 15. What is `v_π(15)` for the equiprobable random policy in each case?
- **Solution.** With equiprobable actions (`1/4` each), `r=−1`, `γ=1`:
  - **Case (a).** Original values are unchanged (Fig. 4.1): `v_π(12)=−22`, `v_π(13)=−20`,
    `v_π(14)=−14`. For state 15:
    `v_π(15)= (1/4)[(−1+v_π(12)) + (−1+v_π(13)) + (−1+v_π(14)) + (−1+v_π(15))]`
    `⇒ v_π(15)= −1 + (v_π(12)+v_π(13)+v_π(14)+v_π(15))/4`
    `⇒ (3/4)v_π(15)= −1 + (−22 − 20 − 14)/4 = −15`
    `⇒ **v_π(15)= −20**`.
  - **Case (b).** State 13 changes because down now goes to 15. Using unchanged neighbors of 13
    (up→9 with `v_π(9)=−20`, left→12, right→14):
    `v_π(13)= −1 + (v_π(12)+v_π(9)+v_π(14)+v_π(15))/4 = −15 + v_π(15)/4`.
    State 15’s equation is still `(3/4)v_π(15)= −1 + (v_π(12)+v_π(13)+v_π(14))/4`, so
    `3v_π(15)= −40 + v_π(13)`.
    Substituting gives `3v_π(15)= −40 − 15 + v_π(15)/4 ⇒ 11v_π(15)/4 = −55 ⇒ **v_π(15)= −20**`,
    and `v_π(13)= −20` as well.
- **Why this works / takeaway:** policy evaluation propagates value changes only through the states
  whose one-step successors are affected; you solve the resulting Bellman equations consistently.

### Exercise 4.3 — Action-value analogs of (4.3), (4.4), (4.5)

- **Book location:** Section 4.3, PDF p. ~83.
- **Type:** Math.
- **Problem (paraphrased):** Give equations analogous to (4.3), (4.4), and (4.5) for the action-value
  function `q_π` and its successive approximations `q_0, q_1, q_2, ...`
- **Solution.** The action-value Bellman expectation equation and its iterative form are:
  - **(4.3) analog — definition/fixed point:**
    `q_π(s,a)= Σ_{s',r} p(s',r|s,a)[r + γ Σ_{a'} π(a'|s') q_π(s',a')]`.
  - **(4.4)/(4.5) analog — iterative policy evaluation:**
    `q_{k+1}(s,a)= Σ_{s',r} p(s',r|s,a)[r + γ Σ_{a'} π(a'|s') q_k(s',a')]`, for all `s,a`,
    which converges to `q_π` under the usual policy-evaluation conditions.
  Policy improvement then becomes `π'(s)= argmax_a q_π(s,a)` (with the usual ε-soft/tie-breaking
  qualifications as needed).
- **Why this works / takeaway:** DP backups are identical in structure once expectations are written
  over action values instead of state values.

### Exercise 4.4 — Guaranteeing termination of policy iteration

- **Book location:** Section 4.3, PDF p. ~83.
- **Type:** Theory.
- **Problem (paraphrased):** Policy iteration may never terminate if the policy keeps switching among
  equally good policies. How should the algorithm be modified to guarantee convergence?
- **Solution.** Make policy improvement change an action only on a **strict** improvement. For each
  state `s`, let `a* ∈ argmax_a q_π(s,a)`; set `π'(s)=a*` only if `q_π(s,a*) > q_π(s,π(s))`,
  otherwise keep `π'(s)=π(s)`. (Equivalently, use a fixed deterministic tie-breaking rule and never
  replace an action by another action of equal value.) In a finite MDP, each actual change then yields
  a strictly better policy, policies cannot cycle among equally good ones, and policy iteration must
  terminate at an optimal policy.
- **Why this works / takeaway:** the policy improvement theorem needs strict improvement (or a
  cycle-free tie rule) to guarantee termination.

### Exercise 4.5 (Programming) — Jack's car rental with modifications

- **Book location:** Section 4.3 / Example 4.2, PDF p. ~84.
- **Type:** Programming.
- **Problem (paraphrased):** Implement policy iteration for Jack's car rental and re-solve it with the
  stated modifications (e.g. one car can be shuttled free from the first location to the second, and
  parking incurs extra cost when more than 10 cars are kept overnight at a location). Verify by
  replicating the original problem first.
- **Solution (experiment/design).** (1) **Validate:** encode the original Example 4.2 dynamics/rewards
  (Poisson rentals/returns, move cost $2, rental revenue $3, up to 20 cars/location, up to 5 moves)
  as an MDP and run `policy_iteration`/`value_iteration` to reproduce the known optimal policy.
  (2) **Modify rewards/dynamics:** make the first car moved from location 1→2 free (reduce move cost
  by $2 when that move count applies) and add a $4 overnight parking cost at any location holding more
  than 10 cars. (3) **Solve:** rerun policy iteration/value iteration and compare policies/values.
  If computation is slow, reduce state/action ranges (e.g. halve max cars) as the book suggests.
- **Coding pointer:** the repo provides the DP solvers `policy_iteration` and `value_iteration` in
  `chapter_04/solution.py` (usable on a `FiniteMDP`/custom dynamics model) with `TestPolicyIteration`
  and `TestValueIteration`. A full Jack's-car-rental MDP instance is **not** implemented as a named
  fixture in this repo’s tests.
- **Why this works / takeaway:** DP applies unchanged once the modified dynamics and reward model are
  specified.

### Exercise 4.6 — Policy iteration for `q_*`

- **Book location:** Section 4.3, PDF p. ~84.
- **Type:** Math / Algorithm.
- **Problem (paraphrased):** Give a complete policy-iteration algorithm for computing `q_*`, analogous
  to the algorithm for `v_*`.
- **Solution.** **Policy iteration (action values):**
  1. **Initialization:** set `q(s,a) ∈ ℝ` and `π(s) ∈ A(s)` arbitrarily for all `s,a`.
  2. **Policy evaluation:** repeat until `Δ < θ`:
     - `Δ ← 0`
     - for each `(s,a)`: `q_old ← q(s,a)`;
       `q(s,a) ← Σ_{s',r} p(s',r|s,a)[r + γ Σ_{a'} π(a'|s') q(s',a')]`;
       `Δ ← max(Δ, |q_old − q(s,a)|)`.
  3. **Policy improvement:** `policy-stable ← true`; for each `s`:
     - `old ← π(s)`; `π(s) ← argmax_a q(s,a)`;
     - if `old ≠ π(s)` then `policy-stable ← false` (apply the strict/tie-safe rule from Ex. 4.4).
  4. **Termination:** if `policy-stable` stop and return `q ≈ q_*`, `π ≈ π_*`; else go to 2.
- **Why this works / takeaway:** greedy improvement with respect to `q_π` yields a better policy, and
  evaluation/improvement alternate to the optimal action-value function.

### Exercise 4.7 — Policy iteration restricted to ε-soft policies

- **Book location:** Section 4.2/4.3, PDF p. ~84.
- **Type:** Theory.
- **Problem (paraphrased):** If only ε-soft policies are allowed, what changes are required in steps 3,
  2, and 1 (in that order) of the policy iteration algorithm for `v_π`?
- **Solution.** Considering the steps in the requested order:
  - **Step 3 (policy improvement):** instead of a fully greedy deterministic policy, form the ε-soft
    greedy policy with respect to `q_π(s,a)=Σ_{s',r} p(s',r|s,a)[r + γ v_π(s')]`. Let
    `A*(s)=argmax_a q_π(s,a)`. Set `π'(a|s)= 1−ε + ε/|A(s)|` for `a ∈ A*(s)` and
    `π'(a|s)= ε/|A(s)|` otherwise (splitting the greedy mass if there are ties). This preserves
    ε-softness and improves among ε-soft policies.
  - **Step 2 (policy evaluation):** unchanged in form; compute `v_π` for the current (now ε-soft)
    policy by iterative policy evaluation using its action probabilities (Eq. 4.5).
  - **Step 1 (initialization):** start from any ε-soft policy, e.g. uniform `π(a|s)=1/|A(s)|`.
  The algorithm converges to an ε-soft optimal policy.
- **Why this works / takeaway:** the policy-improvement step must satisfy the exploration constraint
  while still being greedy in the allowed (ε-soft) class.

### Exercise 4.8 — Why the gambler's optimal policy looks "curious"

- **Book location:** Section 4.4 / Example 4.3, PDF p. ~86.
- **Type:** Theory.
- **Problem (paraphrased):** Explain why the optimal policy for the gambler’s problem has its peculiar
  shape—in particular, why it bets everything at capital 50 but not at capital 51 (for `p_h < 0.5`,
  e.g. `p_h=0.4`).
- **Solution.** The goal is to maximize the probability of reaching capital 100 (reward 1 only on
  reaching the goal, 0 otherwise), not expected capital. When `p_h < 0.5` the game is unfavorable, so
  repeated small fair-sized bets lose in expectation; the optimal policy becomes “bold”/high-variance
  to have a chance of reaching the goal. At capital 50, staking 50 reaches the goal in one win, which
  is optimal under an unfavorable coin. At capital 51, an all-in-equivalent aggressive overbet (e.g.
  staking 49 to reach 100 on a win) risks dropping to 2 on a loss, which can be worse than a smaller
  stake that preserves a better chance of eventually reaching 100. More generally, optimal stakes
  reflect the Bellman optimality trade-off between immediate goal reach and the value of the losing
  successor state; this produces the non-monotone/triangular-looking policy seen in Figure 4.3 (and
  reproduced by `gamblers_problem(p_h=0.4)`).
- **Why this works / takeaway:** maximizing success probability can favor bold, state-dependent bets
  when the underlying wager is unfavorable.

### Exercise 4.9 (Programming) — Value iteration for the gambler's problem

- **Book location:** Section 4.4/4.5, Example 4.3, PDF p. ~87.
- **Type:** Programming.
- **Problem (paraphrased):** Implement value iteration for the gambler’s problem and solve it for
  `p_h = 0.25` and `p_h = 0.55` (introducing dummy terminal states for capital 0 and 100 with values 0
  and 1). Show results graphically as in Figure 4.3. Are the results stable as `θ → 0`?
- **Solution (experiment/design).** Model states as capital `s ∈ {1,…,99}` with actions (stakes)
  `a ∈ {0,…,min(s, 100−s)}`. Transitions: win (`p_h`) → `s+a` with reward `1` if reaching 100 else 0;
  lose (`1−p_h`) → `s−a` with reward 0 (capital 0 terminal). Apply value iteration:
  `V_{k+1}(s)=max_a [p_h (r_win + V_k(s+a)) + (1−p_h)(0 + V_k(s−a))]` until the max change `< θ`.
  Run for `p_h=0.25, 0.55` (and 0.4 for comparison), plotting the value function and the optimal
  stake/policy versus capital. As `θ → 0`, sweeps continue until updates are negligible, so the
  value/policy estimates stabilize to the fixed point (up to floating-point/tie effects).
- **Coding pointer:** `chapter_04/solution.py` provides `gamblers_problem(p_h, goal=100, theta=...)`
  and `value_iteration`; tests: `TestGamblersProblem`, `TestValueIteration`.
  Example: `python -m pytest chapter_04/tests/test_solution.py::TestGamblersProblem -v`
- **Why this works / takeaway:** value iteration directly applies the Bellman optimality backup to the
  known MDP dynamics.

### Exercise 4.10 — Value iteration update for action values

- **Book location:** Section 4.5, PDF p. ~87.
- **Type:** Math.
- **Problem (paraphrased):** What is the analog of the value iteration update (4.10) for action values,
  i.e. for `q_{k+1}(s,a)`?
- **Solution.** The action-value Bellman optimality backup is:
  `q_{k+1}(s,a)= Σ_{s',r} p(s',r | s,a) [r + γ max_{a'} q_k(s',a')]`, for all `s,a`
  (with the usual terminal-state convention `q(terminal,·)=0`). This is the direct analog of (4.10),
  replacing `v_k(s')` by `max_{a'} q_k(s',a')`.
- **Why this works / takeaway:** optimal state values are maxima over optimal action values; iterating
  this backup converges to `q_*`.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Type / mapping | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 4.1 | Computation (gridworld q-values) — no dedicated repo test | — | — | — |
| 4.2 | Computation (added-state values) — no dedicated repo test | — | — | — |
| 4.3 | Math (q-evaluation equations) — no dedicated repo test | `policy_evaluation` (state-value analog) | `TestPolicyEvaluation` (related) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyEvaluation -v` |
| 4.4 | Theory (termination/tie handling) — no dedicated repo test | `policy_iteration` (implements improvement loop) | `TestPolicyIteration` (related) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyIteration -v` |
| 4.5 | Programming (Jack's car rental) — **not implemented as a named instance in repo** | `policy_iteration`, `value_iteration` | `TestPolicyIteration`, `TestValueIteration` (solvers only) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyIteration chapter_04/tests/test_solution.py::TestValueIteration -v` |
| 4.6 | Math/Algorithm (policy iteration for `q_*`) — no dedicated repo test | — | — | — |
| 4.7 | Theory (ε-soft improvement) — no dedicated repo test | `policy_improvement` (greedy improvement, not ε-soft) | `TestPolicyImprovement` (related, not ε-soft) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyImprovement -v` |
| 4.8 | Theory (gambler's policy shape) | `gamblers_problem`, `value_iteration` | `TestGamblersProblem`, `TestValueIteration` | `python -m pytest chapter_04/tests/test_solution.py::TestGamblersProblem -v` |
| 4.9 | Programming (gambler’s problem for `p_h=0.25,0.55`, stability as `θ→0`) | `gamblers_problem`, `value_iteration` | `TestGamblersProblem`, `TestValueIteration` | `python -m pytest chapter_04/tests/test_solution.py::TestGamblersProblem -v` |
| 4.10 | Math (action-value value iteration) — no dedicated repo test | `value_iteration` (state-value analog) | `TestValueIteration` (related) | `python -m pytest chapter_04/tests/test_solution.py::TestValueIteration -v` |

- Full chapter command: `python -m pytest chapter_04/tests/ -v`
- Lint: `ruff check chapter_04/`
- No standalone `exercise_*.py` file exists for this chapter. Theory/derivation exercises are
  answered in this document; only 4.8–4.9 are directly exercised by the gambler’s-problem tests.

---

## Verification and Quality Notes

- Ex. 4.1 and 4.2 values were re-derived from the Bellman expectation equations using Figure 4.1’s
  converged gridworld values; the `q_π(11,down)` result is exact by the terminal transition.
- Ex. 4.4, 4.6, 4.7, 4.10 are standard DP derivations rewritten from the book’s definitions.
- Public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`) was used only as a
  cross-check. Discrepancies (if any) are resolved by the derivations above.
- Programming exercise 4.5 (Jack’s car rental modifications) is not implemented as a named repo
  instance; 4.9 is directly supported by `gamblers_problem`.

---

*Exercises document for Chapter 4 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
