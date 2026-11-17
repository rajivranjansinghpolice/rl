# Chapter 6 — Temporal-Difference Learning: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition). Repo chapter: `chapter_06`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_06` — Temporal-Difference Learning.
- **Implemented in repo:** TD(0) prediction `td0_prediction`, TD(λ) prediction with accumulating
  traces `td_lambda_prediction`, on-policy TD control `sarsa`, off-policy TD control `q_learning`,
  incremental (first-visit, sample-average) Monte Carlo prediction `mc_prediction_incremental`,
  random-walk environment `RandomWalkEnv`, true values `random_walk_true_values`, and comparison
  harness `run_random_walk_comparison` (Example 6.2 / Exercise 6.1 style).
- **Book sections covered by the implementation:** §6.1 (TD prediction, TD error, TD(0)), §6.2
  (advantages of TD; random-walk Example 6.2), §6.4 (Sarsa), §6.5 (Q-learning), and TD(λ) in the
  backward-view / eligibility-trace form used in `td_lambda_prediction`.
- **Not implemented in this repo chapter:** windy gridworld with king’s moves or stochastic wind
  (Exercises 6.9–6.10), explicit afterstates, and Double Expected Sarsa (Exercise 6.12). Those
  exercises are answered conceptually below with coding pointers only where a close analogue exists.
- **Key book equations (plausible citations, S&B 2nd ed.):**
  - Eq. (6.1) — constant-α Monte Carlo: `V(S_t) ← V(S_t) + α [G_t − V(S_t)]`.
  - Eq. (6.2) — TD(0): `V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) − V(S_t)]`.
  - Eq. (6.5) — TD error: `δ_t = R_{t+1} + γ V(S_{t+1}) − V(S_t)`.
  - Eq. (6.6) — MC error as sum of TD errors (fixed `V`): `G_t − V(S_t) = Σ_{k=t}^{T-1} γ^{k-t} δ_k`.
  - Eq. (6.7) — Sarsa: `Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)]`.
  - Eq. (6.8) — Q-learning: `Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a Q(S_{t+1},a) − Q(S_t,A_t)]`.
  - Eq. (5.3) / §6.7 — per-step importance-sampling ratio `ρ_{t:t} = π(A_t|S_t) / b(A_t|S_t)`.
- **Plausible PDF/page anchors (for orientation only):** §6.1–6.3 ~pp. 99–107, §6.4–6.5 ~pp. 107–112,
  §6.7 ~pp. 113–116, exercises ~pp. 117–130 in common printings of the 2nd edition draft.

## Sources and Ground Rules

- Exercise statements below are **paraphrased** in student-friendly language (no long verbatim
  quotes), with plausible section/equation citations.
- Solutions are derived from the Chapter 6 definitions and the repository implementation; claims
  about what is or is not implemented are checked against `chapter_06/solution.py` and
  `chapter_06/tests/test_solution.py`.
- When the repo does not implement an environment or algorithm asked for by an exercise, the answer
  states that explicitly and gives the correct book-style derivation/experiment description.

## How to Use This Document

- Each entry includes **Book location**, **Type**, **Problem (paraphrase)**, **Solution**, and a
  **Coding pointer** (or an explicit note when not implemented).
- Types used: `Theory`, `Computation`, `Math`, `Proof`, `Programming`.
- Run all Chapter 6 tests: `python -m pytest chapter_06/tests/ -v`.

---

## Exercises

### Exercise 6.1 — TD error vs. Monte Carlo error when `V` changes during the episode

- **Book location:** §6.1 (TD prediction), Eq. (6.5), Eq. (6.6), ~p. 117.
- **Type:** Math / Proof.
- **Problem (paraphrase):** Eq. (6.6) shows that, when the value estimates are held fixed during an
  episode, the Monte Carlo error `G_t − V(S_t)` equals the discounted sum of TD errors. If `V`
  changes during the episode (as in online TD), this identity holds only approximately. Let `V_t`
  denote the value array used at time `t`. Derive the exact difference and state what must be added
  to the sum of TD errors to recover the Monte Carlo error.
- **Solution.** Assume terminal `V_T = 0` and write `δ_t^t = R_{t+1} + γ V_t(S_{t+1}) − V_t(S_t)`
  for the TD error computed with the values available at time `t`. Expand the discounted sum:
  `Σ_{k=t}^{T-1} γ^{k-t} δ_k^t = (R_{t+1}+γR_{t+2}+⋯) + Σ_{k=t+1}^{T-1} γ^{k-t} V_k(S_k) − Σ_{k=t}^{T-1} γ^{k-t} V_k(S_k) = G_t − V_t(S_t) + Σ_{k=t+1}^{T-1} γ^{k-t} (V_k(S_k) − V_{k-1}(S_k))`.
  Therefore the exact identity is
  `G_t − V_t(S_t) = Σ_{k=t}^{T-1} γ^{k-t} δ_k^t − Σ_{k=t+1}^{T-1} γ^{k-t} (V_k(S_k) − V_{k-1}(S_k))`.
  The additional amount that must be **added** to the sum of TD errors to equal the Monte Carlo
  error is the correction term `− Σ_{k=t+1}^{T-1} γ^{k-t} (V_k(S_k) − V_{k-1}(S_k))`, i.e. minus
  the discounted accumulation of in-episode value changes. If `V` is fixed during the episode
  (`V_k = V_{k-1}`), this term vanishes and Eq. (6.6) is recovered exactly.
- **Why this matters:** Online TD bootstraps from values that may already have been updated within
  the same episode; Eq. (6.6) is exact only for batch/fixed estimates.
- **Coding pointer:** `td0_prediction` in `chapter_06/solution.py` performs online/in-place TD(0)
  updates (`values[state] += alpha * (reward + gamma * next_v - values[state])` each step). Covered
  by `TestTDPrediction` (`chapter_06/tests/test_solution.py`).

### Exercise 6.2 — Scenario where a TD update is better than a Monte Carlo update

- **Book location:** §6.2 (advantages of TD; driving-home example), ~p. 118.
- **Type:** Theory.
- **Problem (paraphrase):** Give a concrete everyday scenario in which a TD-style update is better on
  average than waiting for a Monte Carlo return, and explain why.
- **Solution.** Consider the driving-home prediction: you have long experience commuting and can
  predict total time remaining, but today traffic becomes unusually bad midway. A TD update can revise
  the estimate at the next state/time step immediately using the newly observed delay plus the
  predicted remainder, so later decisions along the same trip improve right away. A constant-α MC
  update `V ← V + α(G_t − V)` (Eq. 6.1) must wait until you arrive home to observe `G_t`, so it cannot
  correct the prediction mid-episode. TD is also preferable in continuing (non-episodic) tasks or very
  long episodes where waiting for a full return is impractical or high-variance.
- **Why this matters:** TD learns from incomplete sequences by bootstrapping; MC requires complete
  returns.
- **Coding pointer:** `td0_prediction` (online TD target `R+γV(S')`) vs.
  `mc_prediction_incremental` (first-visit, sample-average return target `G_t`) in
  `chapter_06/solution.py`. See `TestTDPrediction` and `TestExercise61`.

### Exercise 6.3 — First random-walk episode: why only `V(A)` changed

- **Book location:** §6.3 / Example 6.2 (random walk), ~p. 119.
- **Type:** Computation / Theory.
- **Problem (paraphrase):** In the left graph of Example 6.2, only the estimate for state `A` appears
  to change after the first episode. What does this imply about the first episode? Why was only `A`
  updated, and by how much?
- **Solution.** In Example 6.2 all nonterminal estimates start at `V(s)=0.5`, the walk starts at `C`,
  and rewards are `0` except `+1` on entering the right terminal. With TD(0) (Eq. 6.2,
  `γ=1`), `V(S_t) ← V(S_t) + α [R_{t+1} + V(S_{t+1}) − V(S_t)]`. For any nonterminal transition
  with `R_{t+1}=0`, the target is `V(S_{t+1})=0.5`, so the TD error is `0` and no update occurs.
  The only time a nonzero error occurs on that episode is when transitioning from `A` to the left
  terminal: reward `0`, terminal value `0`, so `δ = 0 + 0 − V(A) = −0.5` and
  `V(A) ← V(A) + α (−0.5) = 0.5 − 0.5α`. Therefore the first episode visited `A` and then terminated
  on the left (e.g. a path reaching `A` and stepping left). Only `A` changed because every earlier
  transition was to a nonterminal state whose value equaled `V(A)`’s initial value; the magnitude of
  the change was `0.5α` downward.
- **Why this matters:** Early TD learning propagates information only where a sampled transition’s
  target differs from the current estimate.
- **Coding pointer:** `td0_prediction`, `RandomWalkEnv`, `random_walk_true_values` in
  `chapter_06/solution.py`; `TestRandomWalkEnv` and `TestTDPrediction` in
  `chapter_06/tests/test_solution.py`.

### Exercise 6.4 — Effect of the step-size parameter α on TD vs. MC conclusions

- **Book location:** §6.3 / Example 6.2, ~p. 119.
- **Type:** Theory.
- **Problem (paraphrase):** The Example 6.2 comparison used particular step sizes (TD with constant α,
  MC with a decaying/sample-average or small constant step size). Would the conclusion that TD learns
  faster change for a wider range of α values? Could a different fixed α make MC significantly better
  than TD?
- **Solution.** The qualitative result depends on α, but the usual conclusion is robust over a broad
  reasonable range. For very small α, both methods learn slowly and their RMS curves stay close (TD
  still typically has lower early error because it bootstraps). For moderate α (like the values in the
  figure), TD(0) often achieves lower RMS error earlier on this task. For very large α, both become
  noisy/unstable; constant-α MC can fluctuate strongly because it uses high-variance full returns,
  while TD can also overshoot because of bootstrapping. A carefully tuned fixed α might let one
  method edge out the other over some horizon, but there is no fixed α that makes MC *uniformly and
  significantly* better than TD in general: TD’s lower variance from bootstrapping is advantageous
  early, and with appropriate (often decaying) step sizes both converge to correct predictions under
  standard conditions. Comparisons should therefore be made across matched/tuned α ranges rather than
  a single value.
- **Why this matters:** Step size controls the speed–stability (and bias–variance in practice) tradeoff.
- **Coding pointer:** `run_random_walk_comparison(repeats, episodes, alpha, seed)` uses the same
  `alpha` for TD(0) and constant-α MC (`chapter_06/solution.py`). Validated by
  `TestExercise61.test_td0_learned_faster_than_mc` and `TestExercise61.test_td0_error_decreases`
  (`chapter_06/tests/test_solution.py`).

### Exercise 6.5 — Why TD RMS error can go down then up (large α)

- **Book location:** §6.3 / Example 6.2 (right-hand plot), ~p. 119.
- **Type:** Computation / Theory.
- **Problem (paraphrase):** In the right graph, TD’s RMS error can decrease and then increase again
  for high α. What causes this? Does it happen for any choice of step size?
- **Solution.** With a large constant step size, TD(0) takes aggressive updates
  `V ← V + α (R + γV' − V)`. Early on, updates move estimates quickly toward the targets implied by
  sampled transitions, so RMS error often drops. Later, the same large α causes persistent overshoot
  and oscillation around the correct values (and, in the extreme, can prevent convergence or even
  diverge), so error rises again instead of settling near zero. This is **not** inevitable for all α:
  sufficiently small (or appropriately decreasing) step sizes satisfy stochastic-approximation
  conditions and yield stable convergence with RMS error trending down. The phenomenon also depends on
  initialization and the particular sample paths, but large constant α is the primary cause.
- **Why this matters:** Bootstrapping plus a too-large step size can destabilize learning.
- **Coding pointer:** Compare behavior conceptually to `run_random_walk_comparison(..., alpha=...)`
  (`chapter_06/solution.py`), which reproduces the TD-vs-MC RMS comparison over episodes; tests fix
  `alpha=0.1` (`TestExercise61`).

### Exercise 6.6 — Computing true values of the random walk

- **Book location:** §6.3 / Example 6.2, ~p. 120.
- **Type:** Math / Proof.
- **Problem (paraphrase):** The true state values for states A–E in the random walk are
  `1/6, 2/6, 3/6, 4/6, 5/6` (i.e. the probability of terminating on the right from each state under
  the equiprobable policy). Describe at least two ways to compute these values. Which is most direct?
- **Solution.** Let `v(A)..v(E)` be values of nonterminal states; terminals have values `v(L)=0`,
  `v(R)=1`, `γ=1`, reward `1` only on entering `R`. For the uniform random policy
  `π(left|s)=π(right|s)=1/2`:
  **Method 1 (Bellman expectation equations / linear system):**
  `v(A)=½·0 + ½·v(B)`, `v(B)=½·v(A)+½·v(C)`, `v(C)=½·v(B)+½·v(D)`, `v(D)=½·v(C)+½·v(E)`,
  `v(E)=½·v(D)+½·1`. Solving gives `v = (1/6, 2/6, 3/6, 4/6, 5/6)`.
  **Method 2 (iterative policy evaluation / DP):** start from any initial guess and repeatedly apply
  the Bellman expectation operator `v_{k+1}(s)=Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γ v_k(s')]` until
  convergence (Eq. 4.5 / §4.1), which converges to the same fixed point for this finite episodic MRP.
  The **most direct** method here is the small linear system (Method 1), since it yields the exact
  closed-form values in one solve; the book’s stated values match this result.
- **Why this matters:** Exact prediction targets can come from solving Bellman equations or from DP;
  TD/MC estimate them from samples.
- **Coding pointer:** `random_walk_true_values()` returns `{A:1/6,...,E:5/6}` in
  `chapter_06/solution.py`. Verified by `TestRandomWalkEnv.test_true_values`
  (`chapter_06/tests/test_solution.py`).

### Exercise 6.7 — Off-policy TD(0) with importance sampling

- **Book location:** §6.7 (off-policy TD via importance sampling), ~p. 123.
- **Type:** Math / Programming.
- **Problem (paraphrase):** Devise an off-policy version of the TD(0) update for estimating `v_π`
  from data generated by a behavior policy `b` (with `b(a|s)>0` whenever `π(a|s)>0`), using the
  per-step importance-sampling ratio (Eq. 5.3).
- **Solution.** For state-value TD(0), the sampled action is chosen by `b`, so the per-step
  importance-sampling ratio is `ρ_t = ρ_{t:t} = π(A_t|S_t) / b(A_t|S_t)`. The off-policy TD(0)
  update (ordinary importance sampling form) is
  `V(S_t) ← V(S_t) + α · ρ_t · [R_{t+1} + γ V(S_{t+1}) − V(S_t)]`,
  with `V(terminal)=0` (Eq. 6.2 combined with §6.7 importance sampling). This corrects the update for
  the fact that transitions were generated under `b` rather than `π`: actions likelier under `π` than
  `b` receive larger weight, and vice versa. (Related off-policy control, e.g. Q-learning Eq. 6.8,
  avoids explicit ρ for the greedy target by bootstrapping from `max_a Q(S_{t+1},a)`.)
- **Why this matters:** Importance sampling enables learning about a target policy from different
  behavior.
- **Coding pointer:** No explicit off-policy importance-sampling TD(0) prediction function is
  implemented in `chapter_06/solution.py`; the closest implemented off-policy TD control method is
  `q_learning` (Eq. 6.8), covered by `TestQLearning`. On-policy prediction/control analogues are
  `td0_prediction` / `sarsa`.

### Exercise 6.8 — Action-value version of the TD error identity (Eq. 6.6)

- **Book location:** §6.4 (Sarsa / action-value TD), Eq. (6.5)–(6.6) analogue, ~p. 124.
- **Type:** Math / Proof.
- **Problem (paraphrase):** Show that when action-value estimates `Q` are held fixed during an
  episode, the Monte Carlo action-value error equals the discounted sum of action-value TD errors
  `δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)`.
- **Solution.** With fixed `Q` and `Q(terminal,·)=0`, define
  `δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)` and let `G_t = Σ_{k=t}^{T-1} γ^{k-t} R_{k+1}`.
  Then
  `Σ_{k=t}^{T-1} γ^{k-t} δ_k = Σ_{k=t}^{T-1} γ^{k-t} R_{k+1} + Σ_{k=t}^{T-1} γ^{k-t+1} Q(S_{k+1},A_{k+1}) − Σ_{k=t}^{T-1} γ^{k-t} Q(S_k,A_k)`.
  The last two sums telescope to `−Q(S_t,A_t)`, hence
  `G_t − Q(S_t,A_t) = Σ_{k=t}^{T-1} γ^{k-t} δ_k`,
  the action-value analogue of Eq. (6.6). If `Q` changes within the episode, a correction term
  analogous to Exercise 6.1 appears.
- **Why this matters:** The return–TD-error relationship generalizes from `V` to `Q`, underpinning
  Sarsa and related methods.
- **Coding pointer:** `sarsa` in `chapter_06/solution.py` uses the action-value TD error/target
  `R_{t+1} + γ Q(S_{t+1},A_{t+1})` (Eq. 6.7). Covered by `TestSarsa`
  (`test_policy_sums_to_one`, `test_q_keys_cover_all_pairs`, `test_learns_right_preference`).

### Exercise 6.9 — Windy gridworld with king’s moves

- **Book location:** §6.4 (Sarsa) / Example 6.5 (windy gridworld), ~p. 125.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** Re-solve the windy gridworld using king’s moves (8 directions) instead of 4.
  How much faster can a goal be reached? What changes if a ninth “stay” action is added (still subject
  to wind)?
- **Solution.** With 8 actions the agent can move diagonally, allowing paths that counter wind more
  efficiently (e.g. moving down-and-right in an upward-wind column), so the optimal/learned path
  length is typically **fewer time steps** than with 4 actions. Adding a 9th action (no intended move,
  but wind still applies) lets the agent effectively trade horizontal progress for controlled vertical
  displacement in windy columns; in the standard stochastic-free windy gridworld this can sometimes
  help in specific columns but is often unnecessary because diagonal moves already suffice. A full
  empirical answer requires running Sarsa (or another TD control method) on the modified environment
  and comparing episodes-to-goal / steps-per-episode curves to the 4-action baseline.
- **Why this matters:** The action set defines reachable policies; TD control learns from whatever
  transitions the environment provides.
- **Coding pointer:** **Not implemented in this repo chapter.** There is no windy-gridworld
  environment here. The closest implemented on-policy TD control algorithm is `sarsa` (and off-policy
  `q_learning`) in `chapter_06/solution.py`, validated by `TestSarsa` and `TestQLearning` on
  `RandomWalkEnv`.

### Exercise 6.10 — Stochastic wind in the windy gridworld

- **Book location:** §6.4 / Example 6.5, ~p. 125.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** Re-solve windy gridworld when wind strength in each column varies
  stochastically (e.g. the mean strength plus `{-1,0,+1}` with equal probability), while the agent
  still chooses king’s moves as in Exercise 6.9.
- **Solution.** Stochastic wind makes transitions random even for a fixed action, so value estimates
  must average over possible wind outcomes. Sarsa’s expected update
  `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]` remains valid because it is a sample-based backup;
  learning will be noisier and typically slower, and optimal paths/step counts may differ (the agent
  must account for variability rather than a deterministic push). An empirical solution consists of
  implementing the stochastic-wind transition, running Sarsa with exploration (e.g. ε-greedy) for many
  episodes, and plotting learning curves/greedy trajectories.
- **Why this matters:** TD methods handle stochastic dynamics naturally through sampled backups.
- **Coding pointer:** **Not implemented in this repo chapter.** No windy-gridworld environment with
  stochastic wind exists in `chapter_06/solution.py`. See `sarsa` / `q_learning` for the TD control
  updates used in such experiments (`TestSarsa`, `TestQLearning`).

### Exercise 6.11 — Why Q-learning is off-policy control

- **Book location:** §6.5 (Q-learning), Eq. (6.8), ~p. 126.
- **Type:** Theory.
- **Problem (paraphrase):** Explain why Q-learning is considered an off-policy control method.
- **Solution.** Q-learning’s update is
  `Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a Q(S_{t+1},a) − Q(S_t,A_t)]` (Eq. 6.8). The
  bootstrapped target uses the **greedy** action at `S_{t+1}` (`max_a Q`), which corresponds to
  learning about a greedy target policy, regardless of which action `A_{t+1}` the agent actually takes
  next. Meanwhile the data can be generated by any sufficiently exploratory behavior policy (e.g.
  ε-greedy over `Q`), which may choose non-greedy actions. Because the learned/target policy differs
  from the behavior policy, Q-learning is off-policy. (Contrast with Sarsa, Eq. 6.7, which bootstraps
  from the actually sampled `Q(S_{t+1},A_{t+1})` and is on-policy with respect to that behavior.)
- **Why this matters:** Off-policy control can learn optimal action values while following an
  exploratory behavior policy.
- **Coding pointer:** `q_learning` in `chapter_06/solution.py` implements Eq. (6.8) with an
  ε-greedy behavior policy and `max` over next-state actions. Covered by `TestQLearning`
  (`test_policy_sums_to_one`, `test_learns_right_preference`, `test_converges`).

### Exercise 6.12 — Double Expected Sarsa with an ε-greedy target policy

- **Book location:** §6.7 (maximization bias, Double Learning) / §6.6 discussion context, ~p. 130.
- **Type:** Math / Programming.
- **Problem (paraphrase):** Give the update equations for Double Expected Sarsa when the target policy
  is ε-greedy.
- **Solution.** Maintain two independent action-value estimates `Q_1, Q_2`. On each transition
  `(S_t, A_t, R_{t+1}, S_{t+1})`, with probability 0.5 update `Q_1` using `Q_2` to evaluate the
  target, else update `Q_2` using `Q_1` (as in Double Q-learning, to decouple selection/evaluation
  and reduce maximization bias). For an ε-greedy target policy derived from the other estimate, let
  `A* = argmax_a Q_other(S_{t+1}, a)` (breaking ties arbitrarily/consistently) and define the
  expected target under ε-greedy:
  `E_{a~π_other}[Q_other(S_{t+1},a)] = (1-ε) Q_other(S_{t+1}, A*) + (ε/|A|) Σ_a Q_other(S_{t+1}, a)`.
  The corresponding Double Expected Sarsa update (e.g. updating `Q_1`) is
  `Q_1(S_t,A_t) ← Q_1(S_t,A_t) + α [R_{t+1} + γ ((1-ε) Q_2(S_{t+1}, argmax_a Q_2(S_{t+1},a)) + (ε/|A|) Σ_a Q_2(S_{t+1},a)) − Q_1(S_t,A_t)]`,
  and symmetrically when updating `Q_2` (swap `Q_1/Q_2` in the target). Behavior can be, for example,
  ε-greedy with respect to `Q_1+Q_2`.
- **Why this matters:** Double learning reduces maximization bias; the expected (over target-policy
  actions) form reduces target variance compared to sampling `A_{t+1}`.
- **Coding pointer:** **Not implemented in this repo chapter.** Neither Double Expected Sarsa nor
  double estimators are present in `chapter_06/solution.py`. Implemented single-estimator baselines
  are `sarsa` (on-policy, sampled next action) and `q_learning` (off-policy, greedy max target),
  covered by `TestSarsa` and `TestQLearning`.

---

## Coding Exercises Mapped to Repository Code

| Exercise(s) | Book section (plausible) | Repo API (`chapter_06/solution.py`) | Test classes (`chapter_06/tests/test_solution.py`) | Notes |
|---|---|---|---|---|
| 6.1, 6.2 | §6.1–§6.2 | `td0_prediction`, `td_lambda_prediction`, `mc_prediction_incremental` | `TestTDPrediction`, `TestTDLambda` | Online TD vs. (constant-α/first-visit MC) behavior |
| 6.3–6.6 | §6.2–§6.3 (Example 6.2) | `RandomWalkEnv`, `random_walk_true_values`, `td0_prediction`, `mc_prediction_incremental`, `run_random_walk_comparison` | `TestRandomWalkEnv`, `TestTDPrediction`, `TestExercise61` | Random-walk mechanics, true values, TD vs. MC comparison |
| 6.7 | §6.7 | `q_learning` (off-policy control analogue) | `TestQLearning` | Off-policy importance-sampling TD(0) prediction not implemented |
| 6.8 | §6.4 | `sarsa` | `TestSarsa` | Action-value TD/Sarsa update |
| 6.9, 6.10 | §6.4 / Example 6.5 | `sarsa`, `q_learning` (algorithms only) | `TestSarsa`, `TestQLearning` | Windy gridworld environments/experiments not implemented |
| 6.11 | §6.5 | `q_learning` | `TestQLearning` | Off-policy max-target update (Eq. 6.8) |
| 6.12 | §6.6–§6.7 | `sarsa`, `q_learning` (related baselines) | `TestSarsa`, `TestQLearning` | Double Expected Sarsa not implemented |

**Commands**
- Tests: `python -m pytest chapter_06/tests/ -v`
- Lint: `ruff check chapter_06/`

**API notes (signatures, from implementation):** `RandomWalkEnv(seed=None)` with `reset() -> str` and
`step(action) -> (reward, state, done)`; `random_walk_true_values() -> dict`;
`td0_prediction(env, policy_fn=None, alpha=0.1, n_episodes=5000, gamma=1.0, seed=None) -> dict`;
`td_lambda_prediction(env, policy_fn=None, alpha=0.1, lam=0.5, n_episodes=5000, gamma=1.0, seed=None) -> dict`;
`sarsa(env, n_episodes=5000, alpha=0.1, epsilon=0.1, gamma=1.0, seed=None) -> (policy, q_table)`;
`q_learning(env, n_episodes=5000, alpha=0.1, epsilon=0.1, gamma=1.0, seed=None) -> (policy, q_table)`;
`mc_prediction_incremental(env, policy_fn=None, n_episodes=5000, gamma=1.0, seed=None) -> dict`;
`run_random_walk_comparison(repeats=100, episodes=100, alpha=0.1, seed=0) -> {'td0': [...], 'mc': [...]}`.

---

## Verification and Quality Notes

- Derivations (TD/MC identity and correction, off-policy TD(0) with ρ, action-value identity,
  Sarsa/Q-learning/Double Expected Sarsa updates) were written to be consistent with standard S&B
  Chapter 6 results and with the actual behavior of `chapter_06/solution.py`.
- Mappings were checked against `chapter_06/solution.py` and `chapter_06/tests/test_solution.py` so
  that cited APIs, test class names (`TestRandomWalkEnv`, `TestTDPrediction`, `TestTDLambda`,
  `TestSarsa`, `TestQLearning`, `TestExercise61`), and commands match the repository.
- Exercises 6.9, 6.10, and 6.12 describe correct book-style results/experiments but note missing
  environment/algorithm implementations in this chapter (no false implementation claims).

---
*Exercises document for Chapter 6 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*