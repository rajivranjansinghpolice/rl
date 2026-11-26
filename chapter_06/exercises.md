# Chapter 6 — Temporal-Difference Learning: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition). Repo chapter: `chapter_06`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_06` — Temporal-Difference Learning.
- **Implemented in repo:** TD(0) prediction `td0_prediction`, TD(λ) prediction with accumulating
  traces `td_lambda_prediction`, on-policy TD control `sarsa` (random walk) and the generic
  indexed-action control family `sarsa_control` / `q_learning_control` / `expected_sarsa_control`
  (any env with `ACTIONS`, Q keyed `(state, action_index)`, per-episode step cap), off-policy TD
  control `q_learning` (random walk), incremental (first-visit, sample-average) Monte Carlo prediction
  `mc_prediction_incremental`, `double_q_learning` + `maximization_bias_experiment` (Example 6.7 /
  Figure 6.7), the batch-updating experiment `batch_td_fixed_point` / `batch_mc_values` /
  `run_batch_updating` (Example 6.3 / Figure 6.2), random-walk environment `RandomWalkEnv`, true values
  `random_walk_true_values`, comparison harness `run_random_walk_comparison` (Example 6.2 / Exercise 6.1
  style), `CliffWalkingEnv` (Example 6.6 / Figures 6.4 and 6.6) and `MaximizationBiasMDP`
  (Example 6.7 / Figure 6.7); the windy gridworld lives in `rl_envs/windy_gridworld.py`
  (`WindyGridWorld`, standard / king's moves (Ex. 6.9) / stochastic wind (Ex. 6.10) variants).
- **Book sections covered by the implementation:** §6.1 (TD prediction, TD error, TD(0)), §6.2
  (advantages of TD; random-walk Example 6.2), §6.3 (optimality of TD(0); batch updating, Example 6.3),
  §6.4 (Sarsa, windy gridworld, Example 6.5, and king's-moves / stochastic-wind variants, Exercises
  6.9–6.10), §6.5 (Q-learning; cliff walking, Example 6.6), §6.6 (Expected Sarsa), §6.7 (maximization
  bias, Double Learning; Example 6.7 / Double Q-learning), §12.1-style TD(λ) in `td_lambda_prediction`.
- **Figure identities (2nd edition, 2018):** Figure 6.2 is the batch-updating experiment (Example 6.3),
  **not** the random-walk RMS-error comparison. The random-walk panel in Example 6.2 is unnumbered in
  the 2nd edition (it was Figures 6.6–6.7 in the 1st edition). Other Chapter 6 figures of the 2nd
  edition: Figure 6.1 (TD(0) backup diagram, no data), Figure 6.3 (Sarsa applied to the windy
  gridworld, steps/episode vs episode), Figure 6.4 (cliff walking, Sarsa vs Q-learning, single run,
  smoothed over 10 successive episodes, sum of rewards per episode), Figure 6.5 (backup diagrams for
  Q-learning and Expected Sarsa, no data), Figure 6.6 (interim [n=100] and asymptotic [n=100,000]
  performance of TD control methods on cliff walking vs α, ε-greedy ε=0.1, adapted from van Seijen
  et al. 2009), Figure 6.7 (comparison of Q-learning and Double Q-learning, Example 6.7). Figures
  6.3/6.4/6.6/6.7 here are reduced-scale reproductions (fewer episodes/runs than the printed book);
  they are labeled illustrative demonstrations, not pixel-identical redraws.
- **Not implemented in this repo chapter:** explicit part-of-speech/afterstates (§6.8), Double Expected
  Sarsa (Exercise 6.12's specific algorithm), off-policy TD prediction with per-step importance sampling
  (Exercise 6.7). Those are answered conceptually below with coding pointers where a close analogue exists.
- **Key book equations (S&B 2nd ed.):**
  - Eq. (6.1) — constant-α Monte Carlo: \(V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]\).
  - Eq. (6.2) — TD(0): \(V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]\).
  - Eq. (6.5) — TD error: \(\delta _t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\).
  - Eq. (6.6) — MC error as sum of TD errors (fixed `V`): \(G_t - V(S_t) = \sum _{k=t}^{T-1} \gamma ^{k-t} \delta _k\).
  - Eq. (6.7) — Sarsa: \(Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]\).
  - Eq. (6.8) — Q-learning: \(Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma max_a Q(S_{t+1},a) - Q(S_t,A_t)]\).
  - Eq. (6.9) — Expected Sarsa: target \(R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1},a)\).
  - Eq. (6.10) — Double Q-learning: coin-flip update of one estimate using the other's greedy action.
- **PDF/page anchors (orientation; from the author's 2018 complete draft, March 16, 2018; printings differ):**
  §6.1–6.2 pp. 124–126, §6.3 pp. 126–128, §6.4 pp. 128–131, §6.5 pp. 131–133, §6.6 pp. 133–134,
  §6.7 pp. 134–136, §6.8–6.9 pp. 136–139.

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
  episode, the Monte Carlo error \(G_t - V(S_t)\) equals the discounted sum of TD errors. If `V`
  changes during the episode (as in online TD), this identity holds only approximately. Let `V_t`
  denote the value array used at time `t`. Derive the exact difference and state what must be added
  to the sum of TD errors to recover the Monte Carlo error.
**Hook.** The clean identity between a Monte Carlo error and a sum of TD errors assumes a frozen
value table. In online learning the table moves under our feet, so let's quantify exactly what the
movement costs.

**Formal bridge.** With terminal \(V_T=0\), let \(\delta_k = R_{k+1}+\gamma V_k(S_{k+1})-V_k(S_k)\)
be the TD error computed from the value array available at step \(k\). Then

\[
G_t - V_t(S_t)
= \sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
- \sum_{k=t+1}^{T-1}\gamma^{k-t}\left(V_k(S_k)-V_{k-1}(S_k)\right).
\tag{6.1.1}
\]

where \(G_t\) is the episode return from \(t\), \(V_t\) is the value array in force at time \(t\),
\(\delta_k\) is the per-step TD error, and the second sum is the correction for in-episode changes
to the values of states already visited.

**Reasoning chain.** Expand the discounted sum of TD errors. The reward terms give
\(R_{t+1}+\gamma R_{t+2}+\cdots = G_t\). The bootstrapped terms give a positive contribution
\(\sum_{k=t+1}^{T-1}\gamma^{k-t}V_k(S_k)\) after reindexing the successor states, and the subtracted
terms give \(-\sum_{k=t}^{T-1}\gamma^{k-t}V_k(S_k)\). The term at \(k=t\) is \(-V_t(S_t)\), and the
remaining pair at each \(k\ge t+1\) combines into
\(\gamma^{k-t}(V_k(S_k)-V_{k-1}(S_k))\). Hence the discounted TD errors reproduce the MC error only
after subtracting that accumulated change.

The missing piece has a concrete meaning: when the agent updates \(V\) during an episode, the value
it bootstrapped from a moment ago is no longer the value it would use now. Each such change at a
later step contributes its difference \(V_k(S_k)-V_{k-1}(S_k)\), discounted back to \(t\). If the
value array is frozen for the whole episode, every difference is zero and Eq. (6.6) is recovered
exactly — which is why the book states the identity in the batch/fixed-\(V\) setting.

**Why it makes sense.** Consider the extreme of one in-place update right after \(t\) that raises
\(V_{t+1}(S_{t+1})\). The TD error at \(t\) used the old, lower bootstrap value, so it understates the
Monte Carlo error; the correction term in (6.1.1) is negative, which lowers the right-hand side to
match. The sign is therefore right: an upward mid-episode revision means the earlier TD error was
too pessimistic. At the other extreme, no updates at all, the correction vanishes and the elegant
telescoping identity holds.

**Check your understanding.** Prediction: freeze \(V\) throughout each episode (batch style). The
correction sum disappears and the MC error equals the discounted TD-error sum exactly; if in-place
updates are made on every step, the correction grows and the two quantities can differ noticeably.

**Where this shows up in the code.** `td0_prediction` in `chapter_06/solution.py` performs online
in-place updates via `values[state] += alpha * (reward + gamma * next_v - values[state])`; test
`TestTDPrediction`.

### Exercise 6.2 — Scenario where a TD update is better than a Monte Carlo update

- **Book location:** §6.2 (advantages of TD; driving-home example), ~p. 118.
- **Type:** Theory.
- **Problem (paraphrase):** Give a concrete everyday scenario in which a TD-style update is better on
  average than waiting for a Monte Carlo return, and explain why.
**Hook.** Imagine learning the time it takes to drive home while stuck in an unexpected jam. The
choice is between revising your estimate now and waiting until you park the car.

**Formal bridge.** The two targets are

\[
\text{TD: } R_{t+1}+\gamma V(S_{t+1}),
\qquad
\text{MC: } G_t=\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}.
\tag{6.2.1}
\]

where \(S_{t+1}\) is the next state (a later point in the trip), \(V(S_{t+1})\) is the current
prediction of the remaining time, and \(G_t\) is the full observed duration of the trip.

**Reasoning chain.** You have years of commuting experience, so \(V\) is a decent prediction of total
remaining time from wherever you are. Halfway home today, traffic turns unusually bad. At the very
next step a TD update uses the newly observed delay \(R_{t+1}\) plus the old prediction of the
remainder \(V(S_{t+1})\) to revise the estimate for your current state — immediately, before the trip
ends. Later decisions on the same trip (take a side street? leave the motorway?) can use that better
prediction.

A constant-\(\alpha\) Monte Carlo update \(V\leftarrow V+\alpha(G_t-V)\) cannot do this: \(G_t\) is
only known when you arrive home, so the surprise stays unlearned until the episode terminates. TD is
also the natural choice in continuing tasks, where there is no \(T\) to wait for, and in very long
episodes, where a full return is both slow to obtain and high-variance because it sums many random
steps.

**Why it makes sense.** The information available at mid-episode is strictly larger than what you
had at the start, and TD exploits it by combining an observed reward with an existing prediction.
Waiting for \(G_t\) throws away the chance to learn *within* the episode. The extreme case is a
continuing task: MC would literally never update, while TD updates on every step — so the direction
of the comparison is unambiguous.

**Check your understanding.** Prediction: make the trip very short (few steps). The advantage of TD
shrinks because the wait for \(G_t\) is brief and MC's return is low-variance; make the trip
open-ended and TD's advantage grows.

**Where this shows up in the code.** `td0_prediction` (target \(R+\gamma V(S')\)) versus
`mc_prediction_incremental` (first-visit return target \(G_t\)) in `chapter_06/solution.py`; tests
`TestTDPrediction` and `TestExercise61`.

### Exercise 6.3 — First random-walk episode: why only `V(A)` changed

- **Book location:** §6.3 / Example 6.2 (random walk), ~p. 119.
- **Type:** Computation / Theory.
- **Problem (paraphrase):** In the left graph of Example 6.2, only the estimate for state `A` appears
  to change after the first episode. What does this imply about the first episode? Why was only `A`
  updated, and by how much?
**Hook.** A single TD update can only change the state it just left. Reading which state moved tells
us what the very first walk actually did.

**Formal bridge.** With \(\gamma=1\), the TD(0) error and update are

\[
\delta = R_{t+1}+V(S_{t+1})-V(S_t),
\qquad
V(S_t)\leftarrow V(S_t)+\alpha\,\delta .
\tag{6.3.1}
\]

where all nonterminal states start at \(V=0.5\), the terminal states have value \(0\) on the left and
reward \(+1\) on the right, and \(\alpha\) is the step size.

**Reasoning chain.** At the start of Example 6.2 every nonterminal state has \(V=0.5\). Along the
first episode, a transition between two nonterminal states gives reward \(0\), and the target is
\(V(S_{t+1})=0.5\), exactly the current estimate; the TD error in (6.3.1) is zero, so nothing changes.

A nonzero error appears only when the target differs from \(0.5\). The observed episode must have
reached state \(A\) and stepped left into the terminal state: reward \(0\), terminal value \(0\), so
\(\delta=0+0-0.5=-0.5\) and \(V(A)\leftarrow0.5-0.5\alpha\). If it had stepped right into the other
terminal, the reward would have been \(+1\) and the update would have been \(+0.5\alpha\) instead.

So the first episode visited \(A\) and then terminated on the left, and only \(A\) changed because
all earlier transitions were between uniformly valued nonterminal states. The magnitude of the
change is \(0.5\alpha\) downward.

**Why it makes sense.** TD propagates information one step at a time, and only where a sampled
transition contradicts the current estimate. With a flat \(0.5\) table, ordinary steps convey no
news — "the future looks exactly as I expected" — but falling off the left end is bad news worth
\(-0.5\). At the extreme \(\alpha=1\), \(V(A)\) would jump straight to \(0\), the observed terminal
value; at small \(\alpha\) it barely moves, which is why the figure shows a small single change.

**Check your understanding.** Prediction: if the first episode had instead ended by stepping right
from \(E\), the changed state would be \(E\) and its value would rise by \(0.5\alpha\); if the initial
values were \(0\) instead of \(0.5\), the left-terminal update would be zero and no state would move.

**Where this shows up in the code.** `td0_prediction`, `RandomWalkEnv` and `random_walk_true_values`
in `chapter_06/solution.py`; tests `TestRandomWalkEnv` and `TestTDPrediction`.

### Exercise 6.4 — Effect of the step-size parameter α on TD vs. MC conclusions

- **Book location:** §6.3 / Example 6.2, ~p. 119.
- **Type:** Theory.
- **Problem (paraphrase):** The Example 6.2 comparison used particular step sizes (TD with constant α,
  MC with a decaying/sample-average or small constant step size). Would the conclusion that TD learns
  faster change for a wider range of α values? Could a different fixed α make MC significantly better
  than TD?
**Hook.** A learning curve is always a joint story about the algorithm and its step size. Let's ask
whether "TD learns faster" survives when \(\alpha\) is swept over a wide range.

**Formal bridge.** Both methods move the same estimate toward different targets with the same step
size,

\[
V(S_t)\leftarrow V(S_t)+\alpha\left[\text{target}-V(S_t)\right],
\qquad
\text{target}_{\text{TD}}=R_{t+1}+\gamma V(S_{t+1}),
\quad
\text{target}_{\text{MC}}=G_t .
\tag{6.4.1}
\]

where \(\alpha\) is the step size shared by both methods in the comparison, and performance is measured
by RMS error against the true values of Exercise 6.6.

**Reasoning chain.** For very small \(\alpha\), both methods move slowly and their RMS curves stay
close; TD still typically shows lower early error because its target bootstraps from an estimate
rather than a full noisy return. At moderate \(\alpha\), the range used in the figure, TD(0) often
reaches lower RMS error earlier on the random walk.

At very large \(\alpha\), both become noisy and potentially unstable: constant-\(\alpha\) MC
fluctuates strongly because each full return \(G_t\) has high variance, while TD can overshoot
because its target contains the current estimate (the feedback loop behind Exercise 6.5). A
carefully tuned fixed \(\alpha\) can let one method edge out the other over a particular horizon, but
there is no fixed \(\alpha\) that makes MC *uniformly and significantly* better than TD in general:
TD's bootstrapping advantage is strongest early, and with appropriate — often decaying — step sizes
both converge to the correct predictions under standard stochastic-approximation conditions.

The practical conclusion is that comparisons should use matched or tuned \(\alpha\) ranges, not a
single arbitrary value; the qualitative ranking is robust over a broad reasonable range.

**Why it makes sense.** Think of \(\alpha\) as how much of a single noisy target you believe. At
\(\alpha\to0\) nothing is learned, so the methods tie at their initial error. At large \(\alpha\),
both chase noise, but MC's noise is the variance of a whole episode while TD's is the variance of
one transition plus a stable estimate — so MC has more to be unstable about. Since TD's advantage is
structural (lower-variance targets early), re-tuning \(\alpha\) cannot reverse it uniformly, only
shift the curves.

**Check your understanding.** Prediction: decrease \(\alpha\) by a factor of ten. Both curves drop
more slowly, the gap between them narrows, and eventual convergence improves in stability; raising
\(\alpha\) enough makes both curves oscillate, with MC typically the noisier of the two.

**Where this shows up in the code.** `run_random_walk_comparison(repeats, episodes, alpha, seed)`
uses the same `alpha` for TD(0) and constant-α MC in `chapter_06/solution.py`; tests
`TestExercise61.test_td0_learned_faster_than_mc` and `TestExercise61.test_td0_error_decreases`.

### Exercise 6.5 — Why TD RMS error can go down then up (large α)

- **Book location:** §6.3 / Example 6.2 (right-hand plot), ~p. 119.
- **Type:** Computation / Theory.
- **Problem (paraphrase):** In the right graph, TD’s RMS error can decrease and then increase again
  for high α. What causes this? Does it happen for any choice of step size?
**Hook.** A falling error curve is not always a promise of continued improvement. Let's see how the
same aggressive step size that helps early can hurt later.

**Formal bridge.** The TD(0) update is

\[
V(S_t)\leftarrow V(S_t)+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right],
\tag{6.5.1}
\]

where the target contains the current estimate \(V(S_{t+1})\), so a large \(\alpha\) amplifies a
self-referential feedback loop.

**Reasoning chain.** With a large constant \(\alpha\), each sampled transition moves the estimate a
long way toward its target. Early in learning the estimates start far from the truth, so those
aggressive moves quickly reduce RMS error. Later, when estimates are near their correct values, the
same step size overshoots on almost every update: a single transition into a low-valued state pulls
the estimate far down, the next one pulls it back up, and the system oscillates instead of settling.

In the extreme, a large enough \(\alpha\) prevents convergence or even causes divergence. This is not
fate for every step size: sufficiently small — or appropriately decreasing — step sizes satisfy the
stochastic-approximation conditions and give stable convergence with RMS error trending down. The
effect also depends on the initial values and the particular sample paths, but a large constant
\(\alpha\) is the primary cause.

**Why it makes sense.** Compare with MC, whose target \(G_t\) does not depend on the current
estimate: even a large \(\alpha\) there produces a noisy random walk, not a feedback loop. TD's
target in (6.5.1) *does* depend on \(V\), so an over-large step turns estimation into a loop that can
overshoot its own output. Think of a thermostat with an excessive gain: it corrects a small error by
a huge action, then corrects the resulting error just as violently. Reducing the gain restores
settling — exactly what a smaller \(\alpha\) does.

**Check your understanding.** Prediction: halve \(\alpha\) after the RMS curve has started to rise.
The oscillations should shrink and the curve should resume falling, because the overshoot in
(6.5.1) is scaled by \(\alpha\); raising \(\alpha\) again brings the rise back sooner.

**Where this shows up in the code.** `run_random_walk_comparison(..., alpha=...)` in
`chapter_06/solution.py` reproduces the TD-vs-MC RMS comparison; `TestExercise61` uses `alpha=0.1`.

### Exercise 6.6 — Computing true values of the random walk

- **Book location:** §6.3 / Example 6.2, ~p. 120.
- **Type:** Math / Proof.
- **Problem (paraphrase):** The true state values for states A–E in the random walk are
  `1/6, 2/6, 3/6, 4/6, 5/6` (i.e. the probability of terminating on the right from each state under
  the equiprobable policy). Describe at least two ways to compute these values. Which is most direct?
**Hook.** Before judging a learner's error we need exact targets. For this five-state walk there are
two clean ways to get them, and one is much shorter.

**Formal bridge.** For the equiprobable policy, the state values obey

\[
v(s)=\tfrac12 v(\text{left neighbour})+\tfrac12 v(\text{right neighbour}),
\qquad v(L)=0,\quad v(R)=1,\quad \gamma=1,
\tag{6.6.1}
\]

with reward \(1\) only on entering the right terminal \(R\).

where \(s\in\{A,B,C,D,E\}\) is a nonterminal walk state and \(v(L),v(R)\) are the terminal values.

**Reasoning chain.** Method 1: write (6.6.1) for each state:
\(v(A)=\tfrac12\cdot0+\tfrac12 v(B)\), \(v(B)=\tfrac12 v(A)+\tfrac12 v(C)\),
\(v(C)=\tfrac12 v(B)+\tfrac12 v(D)\), \(v(D)=\tfrac12 v(C)+\tfrac12 v(E)\),
\(v(E)=\tfrac12 v(D)+\tfrac12\cdot1\). This is a small linear system; solving it gives
\(v=(1/6,2/6,3/6,4/6,5/6)\), matching the book's values.

Method 2: iterative policy evaluation. Start from any initial guess and repeatedly apply
\(v_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_k(s')]\) until convergence (Eq. 4.5 /
§4.1). For this finite episodic MRP the operator converges to the same fixed point, so the iterated
values approach the same fractions.

The **most direct** route is Method 1: five linear equations solved once give exact closed-form
values, whereas DP approaches them asymptotically. These exact values are what RMS-error curves in
Example 6.2 measure against, while TD and MC estimate them from samples.

**Why it makes sense.** The answer \(k/6\) is just the probability of hitting the right terminal
before the left when moving randomly from position \(k\) on a line with absorbing ends — a classic
gambler's-ruin result. It increases linearly from \(A\) to \(E\), as it must: starting closer to the
right end makes winning more likely. The extreme cases check out: \(v(L)=0\) at the left terminal
and \(v(R)=1\) at the right.

**Check your understanding.** Prediction: add one more nonterminal state \(F\) between \(E\) and \(R\).
All six values become \(k/7\) instead of \(k/6\), so \(v(A)\) drops from \(1/6\) to \(1/7\); the linear
system grows by one equation but keeps the same structure.

**Where this shows up in the code.** `random_walk_true_values()` returns `{A:1/6, ..., E:5/6}` in
`chapter_06/solution.py`; test `TestRandomWalkEnv.test_true_values`.

### Exercise 6.7 — Off-policy TD(0) with importance sampling

- **Book location:** §6.7 (off-policy TD via importance sampling), ~p. 123.
- **Type:** Math / Programming.
- **Problem (paraphrase):** Devise an off-policy version of the TD(0) update for estimating \(v_\pi\)
  from data generated by a behavior policy `b` (with `b(a|s)>0` whenever \(\pi (a|s)>0\)), using the
  per-step importance-sampling ratio (Eq. 5.3).
**Hook.** The data come from one policy while the values we want belong to another. Importance
sampling is the correction that bridges the two.

**Formal bridge.** The per-step ratio and the weighted TD(0) update are

\[
\rho_t=\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)},\qquad
V(S_t)\leftarrow V(S_t)+\alpha\,\rho_t
\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right],
\tag{6.7.1}
\]

with \(V(\text{terminal})=0\) and the coverage condition \(b(a\mid s)>0\) whenever \(\pi(a\mid s)>0\).

where \(\pi\) is the target policy being evaluated, \(b\) is the behaviour policy generating the
transition, \(\rho_t\) is the one-step importance ratio, and \(\alpha\) is the step size.

**Reasoning chain.** Under \(b\), the sampled action \(A_t\) may be one that \(\pi\) would rarely
choose, so an ordinary TD update would estimate \(v_b\), not \(v_\pi\). Multiplying the TD error by
\(\rho_t\) reweights each transition by how much more or less likely the action is under \(\pi\):
actions likelier under \(\pi\) get larger weight, and vice versa, correcting for the mismatch.

The coverage condition is essential: if \(\pi\) would take an action that \(b\) never takes, there is
no data from which to learn about it. Terminal states keep \(V=0\), so transitions into termination
use the ordinary reward-only target. A related idea appears in off-policy control: Q-learning
(Eq. 6.8) avoids an explicit \(\rho\) for the greedy target by bootstrapping from
\(\max_a Q(S_{t+1},a)\), which effectively evaluates the greedy policy directly.

**Why it makes sense.** If \(\pi=b\), every ratio is \(1\) and (6.7.1) reduces exactly to on-policy
TD(0) — a necessary sanity check. If \(b\) is uniform over many actions while \(\pi\) is greedy, the
chosen greedy action has \(\rho>1\) and gets amplified, while exploratory actions (\(\pi=0\)) get
weight zero and drop out. The direction is right: reweight toward the policy whose values we want.
The cost is variance — ratios can be large — which is the familiar off-policy tradeoff from
Chapter 5.

**Check your understanding.** Prediction: make \(b\) identical to \(\pi\). Every \(\rho_t=1\), so the
update becomes ordinary TD(0); make \(b\) assign tiny probability to \(\pi\)'s greedy action and
\(\rho_t\) explodes, so learning becomes extremely noisy whenever that action is sampled.

**Where this shows up in the code.** An explicit importance-sampling TD(0) predictor is **not
implemented**; the closest off-policy TD control method is `q_learning` (Eq. 6.8) with test
`TestQLearning`, while `td0_prediction` and `sarsa` are the on-policy analogues.

### Exercise 6.8 — Action-value version of the TD error identity (Eq. 6.6)

- **Book location:** §6.4 (Sarsa / action-value TD), Eq. (6.5)–(6.6) analogue, ~p. 124.
- **Type:** Math / Proof.
- **Problem (paraphrase):** Show that when action-value estimates `Q` are held fixed during an
  episode, the Monte Carlo action-value error equals the discounted sum of action-value TD errors
  \(\delta _t = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\).
**Hook.** The telescoping argument behind Eq. 6.6 did not really need state values. Let's redo it with
state-action pairs and see the same identity emerge.

**Formal bridge.** With \(Q\) fixed during the episode and \(Q(\text{terminal},\cdot)=0\),

\[
G_t-Q(S_t,A_t)=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k,
\qquad
\delta_k=R_{k+1}+\gamma Q(S_{k+1},A_{k+1})-Q(S_k,A_k).
\tag{6.8.1}
\]

where \(G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}\) is the return and \(\delta_k\) is the action-value TD
error along the sampled trajectory.

**Reasoning chain.** Expand the discounted sum of TD errors into three sums: the rewards give
\(G_t\); the bootstrapped terms are \(\sum_{k=t}^{T-1}\gamma^{k-t+1}Q(S_{k+1},A_{k+1})\); and the
subtracted terms are \(\sum_{k=t}^{T-1}\gamma^{k-t}Q(S_k,A_k)\). Reindex the first of the two
\(Q\)-sums so it runs over the same pairs \((S_k,A_k)\) for \(k=t+1,\ldots,T\); the terminal term is
zero, so all interior contributions cancel pairwise, leaving only \(-Q(S_t,A_t)\). Therefore
\(G_t-Q(S_t,A_t)=\sum_k\gamma^{k-t}\delta_k\), the action-value analogue of Eq. 6.6.

If \(Q\) changes during the episode, telescoping is no longer exact and a correction term like the
one in Exercise 6.1 appears: the in-episode changes to \(Q\) at already-visited pairs must be
subtracted. With fixed \(Q\), the identity underpins Sarsa-style methods: each TD error is one piece
of the full action-value error.

**Why it makes sense.** The extreme \(T=t+1\) makes it obvious: one reward then termination, so
\(G_t=R_{t+1}\), \(\delta_t=R_{t+1}-Q(S_t,A_t)\), and the identity is immediate. Longer episodes just
add more discounted pieces that telescope away. If the \(Q\) used at step \(k\) were replaced by a
different \(Q\) at step \(k+1\), the cancellation would fail — exactly the online-learning caveat.

**Check your understanding.** Prediction: freeze \(Q\) but increase \(\gamma\) toward \(1\). More of
the later TD errors carry nearly full weight, so the identity still holds, but late transitions
contribute much more to the MC error; with \(\gamma=0\) only \(\delta_t\) matters.

**Where this shows up in the code.** `sarsa` in `chapter_06/solution.py` uses the target
\(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})\) (Eq. 6.7); tests `TestSarsa` include
`test_policy_sums_to_one`, `test_q_keys_cover_all_pairs` and `test_learns_right_preference`.

### Exercise 6.9 — Windy gridworld with king’s moves

- **Book location:** §6.4 (Sarsa) / Example 6.5 (windy gridworld) / Figure 6.3; Ex. 6.9 is printed after
  Example 6.5 in §6.4 (2018 draft pp. 131–132).
- **Type:** Programming / Theory.
- **Problem (paraphrase):** Re-solve the windy gridworld using king’s moves (8 directions) instead of 4.
  How much faster can a goal be reached? What changes if a ninth “stay” action is added (still subject
  to wind)?
**Hook.** Enlarging the action set changes which paths are even possible. Let's see what diagonal
moves — and a deliberate "do nothing" move — buy in a windy corridor.

**Formal bridge.** Sarsa's action-value update remains unchanged,

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right],
\tag{6.9.1}
\]

but the action set \(A_t\) now contains 8 king's moves (or 9 with "stay"), and the wind is applied to
the resulting position after the intended move.

where \(A_t\in\{\text{8 directions}\}\) (plus optional stay), and the next state \(S_{t+1}\) reflects
both the intended move and the column's wind.

**Reasoning chain.** With only four moves, counteracting an upward wind costs a dedicated downward
step that makes no horizontal progress. King's moves let the agent combine them: a down-and-right
diagonal in an upward-wind column can cancel the drift *while* advancing toward the goal. The
learned or optimal path therefore typically uses fewer time steps than the 4-action baseline — the
question "how much faster" is answered empirically by comparing steps-per-episode curves.

Adding a ninth "stay" action means no intended displacement, but the wind still applies. That lets
the agent trade horizontal progress for controlled vertical displacement in windy columns: staying
still in a strong upward-wind column drifts it up without spending a move toward the target. In the
standard deterministic grid this helps in specific columns but is often unnecessary, because
diagonal moves already provide the needed control. To settle the quantitative question, run Sarsa (or
another TD control method) on the modified environment and compare the resulting steps-per-episode
curve with the 4-action baseline of Figure 6.3.

**Why it makes sense.** A larger action set cannot make the optimal path longer: the old four moves
are still available, so the best achievable steps-per-episode can only improve or stay equal. The
gain comes from actions that do two jobs at once (diagonal progress plus wind compensation), which
is exactly what a axis-aligned action set lacks. The "stay" action is useful only when wind alone
moves you in a desirable direction — otherwise it wastes a step, which is why its benefit is
column-specific.

**Check your understanding.** Prediction: increase the wind strength in every column. The advantage
of king's moves grows because compensation is needed more often, and the stay action becomes useful
in more columns; with zero wind, diagonals still shorten paths geometrically but stay is strictly
dominated.

**Where this shows up in the code.** `WindyGridWorld(king_moves=True, allow_stay=True)` in
`rl_envs/windy_gridworld.py` (9 actions, wind applied after staying still too), solved with
`sarsa_control` (`alpha=0.5`, `epsilon=0.1`); reproduction `plot_figure_6_9_king_moves` in
`chapter_06/solution.py`; environment test `tests/test_rl_envs.py::test_windy_gridworld_king_moves`.

### Exercise 6.10 — Stochastic wind in the windy gridworld

- **Book location:** §6.4 / Example 6.5 / Figure 6.3; Ex. 6.10 is printed after Example 6.5 in §6.4
  (2018 draft p. 131). Note the book’s statement uses **king’s moves** as in Exercise 6.9.
- **Type:** Programming / Theory.
- **Problem (paraphrase):** Re-solve the windy gridworld task with king’s moves, assuming that the
  effect of the wind, if there is any, is stochastic, sometimes varying by 1 from the mean values
  given for each column. That is, a third of the time you move exactly according to these values, as
  in the previous exercise, but one cell above that a third of the time, and one cell below that a
  third of the time. (Book example: one cell to the right of the goal moving left lands one cell
  above the goal, two cells above the goal, or on the goal, with equal probability.)
**Hook.** Now the wind itself rolls dice. The backup equation does not change — only the distribution
of next states does — and that is precisely where TD methods shine.

**Formal bridge.** With column mean wind \(w\ge1\), the effective wind is drawn uniformly from

\[
\{w-1,\,w,\,w+1\}\quad\text{with probability }1/3\text{ each},
\tag{6.10.1}
\]

while columns with \(w=0\) stay zero-wind; Sarsa still updates from the sampled successor via
\(Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]\).

where \(w\) is the mean wind strength for a column and \(s'\) is the next state after applying the
sampled wind to the intended king's move.

**Reasoning chain.** In the book's example, a step to the right of the goal moving left is
subject to wind that lands the agent one cell above the goal, two cells above the goal, or on the
goal, each with probability \(1/3\). More generally, whenever a column's mean wind is \(w\ge1\), the
actual displacement is \(w-1\), \(w\) or \(w+1\) with equal probability; zero-wind columns remain
unchanged.

This makes transitions random even for a fixed action, so the value of an action must average over
the possible wind outcomes. Sarsa's update remains valid because it is a sample-based backup: each
observed transition is an unbiased sample from that distribution, and repeated sampling learns the
expectation. Learning is noisier and typically slower than in the deterministic case. The empirical
answer is to implement (6.10.1), run Sarsa with ε-greedy exploration, and compare learning curves and
trajectories against the deterministic windy gridworld.

**Why it makes sense.** The expected wind is still \(w\), so the *average* drift is unchanged; what
changes is the spread. A plan that relied on exactly cancelling the mean wind now sometimes
overshoots and sometimes undershoots, so cautious policies that stay away from the windy columns
become relatively more attractive. The direction of the slowdown is intuitive: extra outcome noise
means more samples are needed before action values settle. TD handles this naturally because it
learns from samples rather than needing to enumerate the distribution.

**Check your understanding.** Prediction: increase the variability from \(\pm1\) to \(\pm2\). The mean
drift stays \(w\), but outcomes spread further, so learning gets noisier and the optimal policy keeps
more distance from strong wind columns; if variability is zero, the curves revert to the
deterministic case of Exercise 6.9.

**Where this shows up in the code.** `WindyGridWorld(king_moves=True, allow_stay=True,
stochastic_wind=True)` in `rl_envs/windy_gridworld.py`, solved with `sarsa_control`; reproduction
`plot_figure_6_10_stochastic_wind` in `chapter_06/solution.py` (labeled illustrative).

### Exercise 6.11 — Why Q-learning is off-policy control

- **Book location:** §6.5 (Q-learning), Eq. (6.8), ~p. 126.
- **Type:** Theory.
- **Problem (paraphrase):** Explain why Q-learning is considered an off-policy control method.
**Hook.** Two policies are hiding in one update equation: the one gathering data and the one being
evaluated. Spotting the difference is the whole point of "off-policy".

**Formal bridge.** Q-learning's target uses a max,

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right],
\tag{6.11.1}
\]

whereas Sarsa uses the sampled next action \(A_{t+1}\) drawn from the behaviour policy.

where \(A_t\) is taken by the exploratory behaviour policy (e.g. ε-greedy over \(Q\)), while
\(\max_a Q(S_{t+1},a)\) is the value of the greedy action at \(S_{t+1}\).

**Reasoning chain.** The bootstrapped target in (6.11.1) uses the greedy action at the next state,
irrespective of which action \(A_{t+1}\) the agent actually takes. That means the values being learned
are those of a greedy target policy — the optimal action values \(q_*\) — even while the data come
from any sufficiently exploratory behaviour policy that may choose non-greedy actions. Because the
target policy differs from the behaviour policy, the method is off-policy.

Sarsa (Eq. 6.7) is the useful contrast: its target is \(Q(S_{t+1},A_{t+1})\), using the action that
the behaviour policy actually samples, so it evaluates the policy it follows — on-policy. Q-learning
needs coverage (the behaviour policy must eventually try all actions) but need not average over the
behaviour's probabilities: the max simply replaces that average with the greedy choice.

**Why it makes sense.** Imagine behaviour is ε-greedy. On a step where it explores, Q-learning still
backs up the value of the *best* next action as if the agent had been greedy thereafter. Learning is
therefore about "what would happen if I acted optimally", not "what happened because I explored".
At the extreme \(\epsilon=0\), behaviour is already greedy and the two methods coincide; increasing
\(\epsilon\) separates them, with Q-learning continuing to target greediness. That is why it can learn
the optimal policy while still exploring.

**Check your understanding.** Prediction: make the behaviour policy degenerate (always the same
action, no coverage). Sarsa's values describe that narrow policy, and Q-learning cannot discover
better actions either because their values are never sampled; restore ε-greedy behaviour and
Q-learning can still converge to the optimal values.

**Where this shows up in the code.** `q_learning` in `chapter_06/solution.py` implements Eq. (6.8)
with ε-greedy behaviour and a max over next-state actions; tests `TestQLearning` include
`test_policy_sums_to_one`, `test_learns_right_preference` and `test_converges`.

### Exercise 6.12 — Double Expected Sarsa with an ε-greedy target policy

- **Book location:** §6.7 (maximization bias, Double Learning) / §6.6 discussion context, ~p. 130.
- **Type:** Math / Programming.
- **Problem (paraphrase):** Give the update equations for Double Expected Sarsa when the target policy
  is ε-greedy.
**Hook.** Combining two ideas — double learning and Expected Sarsa — means separating the action
that is chosen from the values used to judge it, and then averaging over the target policy's actions
instead of sampling one.

**Formal bridge.** Maintain \(Q_1,Q_2\). With probability \(1/2\) update \(Q_1\), using \(Q_2\) for the
target; the ε-greedy expected target under \(Q_2\) is

\[
(1-\epsilon)\,Q_2\!\left(S_{t+1},\arg\max_a Q_2(S_{t+1},a)\right)
+ \frac{\epsilon}{|A|}\sum_a Q_2(S_{t+1},a),
\tag{6.12.1}
\]

and e.g. the \(Q_1\) update is

\[
Q_1(S_t,A_t)\leftarrow Q_1(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma\left((1-\epsilon)Q_2(S_{t+1},\arg\max_a Q_2(S_{t+1},a))
+\frac{\epsilon}{|A|}\sum_a Q_2(S_{t+1},a)\right)-Q_1(S_t,A_t)\right].
\tag{6.12.2}
\]

where \(\epsilon\) is the target policy's exploration rate, \(|A|\) is the number of actions, and the
\(Q_2\) update swaps the roles of \(Q_1\) and \(Q_2\).

**Reasoning chain.** Keep two independent estimates. On each transition \((S_t,A_t,R_{t+1},S_{t+1})\),
flip a fair coin: with probability \(0.5\) update \(Q_1\) using \(Q_2\) to evaluate the target, else
update \(Q_2\) using \(Q_1\) — exactly the decoupling that Double Q-learning uses to reduce
maximization bias.

For the selected update, derive the target policy from the *other* estimate. Let
\(A_*=\arg\max_a Q_{\text{other}}(S_{t+1},a)\), breaking ties arbitrarily but consistently. Under an
ε-greedy policy the expected value is \((1-\epsilon)Q_{\text{other}}(S_{t+1},A_*)+(\epsilon/|A|)
\sum_a Q_{\text{other}}(S_{t+1},a)\), which is (6.12.1). Substituting this expectation in place of a
sampled \(Q(S_{t+1},A_{t+1})\) gives (6.12.2), with the symmetric equation when the coin flip selects
\(Q_2\). The behaviour policy can be, for example, ε-greedy with respect to \(Q_1+Q_2\); it need not
match the target.

Using the expectation rather than a sampled next action reduces target variance compared with
sampling \(A_{t+1}\), while using the other estimate for both selection and evaluation removes the
maximization bias that comes from using the same noisy estimate for both jobs.

**Why it makes sense.** At \(\epsilon=0\), (6.12.1) reduces to the max of the other estimate — plain
Double Q-learning — so the new equations generalise it properly. At \(\epsilon=1\), the target is the
uniform average of the other estimate's action values and the argmax term drops out: there is no
selection bias left to remove, because nothing is being selected. For intermediate \(\epsilon\), the
two mechanisms work together, and the direction is right: more exploration in the target means less
weight on the (possibly biased) maximum.

**Check your understanding.** Prediction: raise \(\epsilon\) in the target policy. The max term's
weight \(1-\epsilon\) falls and the uniform-average term grows, so the target becomes less optimistic
and less sensitive to noisy maxima; at \(\epsilon=1\) the update is the same as Expected Sarsa with a
uniform target.

**Where this shows up in the code.** Double Expected Sarsa itself is **not implemented**. Available:
`double_q_learning` and `maximization_bias_experiment` (with `plot_figure_6_7_double_q_learning`),
plus single-estimator baselines `q_learning` and `expected_sarsa_control`; tests `TestQLearning` and
`TestControl`.

---

## Coding Exercises Mapped to Repository Code

| Exercise(s) | Book section (2nd ed.) | Repo API (`chapter_06/solution.py`) | Test classes (`chapter_06/tests/test_solution.py`) | Notes |
|---|---|---|---|---|
| 6.1, 6.2 | §6.1–§6.2 (Example 6.2) | `td0_prediction`, `td_lambda_prediction`, `mc_prediction_incremental`, `run_random_walk_comparison` | `TestTDPrediction`, `TestTDLambda`, `TestExercise61` | Online TD vs. (constant-α/first-visit) MC |
| 6.3–6.6 | §6.2–§6.3 (Example 6.2/6.3, Fig. 6.2) | `RandomWalkEnv`, `random_walk_true_values`, `batch_td_fixed_point`, `batch_mc_values`, `run_batch_updating` | `TestRandomWalkEnv`, `TestTDPrediction`, `TestExercise61`, `TestBatchUpdating` | Random walk; batch TD vs batch MC |
| 6.7 | §6.5 / §6.7 | `q_learning` (off-policy control analogue) | `TestQLearning` | Importance-sampling TD prediction not implemented |
| 6.8 | §6.4 | `sarsa`, `sarsa_control` | `TestSarsa`, `TestControl` | Action-value TD/Sarsa update |
| 6.9, 6.10 | §6.4 / Example 6.5, Ex. 6.9–6.10, Fig. 6.3 | `WindyGridWorld` (`rl_envs`), `sarsa_control`, `plot_figure_6_3_windy_gridworld`, `plot_figure_6_9_king_moves`, `plot_figure_6_10_stochastic_wind` | `tests/test_rl_envs.py::test_windy_gridworld_king_moves`, `TestControl`, `TestWindy` | King's moves (9 actions) and stochastic wind (equal thirds over wind−1/wind/wind+1) |
| 6.11 | §6.5 | `q_learning`, `q_learning_control` | `TestQLearning`, `TestControl` | Off-policy max-target update (Eq. 6.8) |
| 6.12 | §6.6–§6.7 (Example 6.7, Figs. 6.6–6.7) | `expected_sarsa_control`, `double_q_learning`, `maximization_bias_experiment`, `CliffWalkingEnv`, `MaximizationBiasMDP` | `TestControl`, `TestCliffWalking`, `TestMaximizationBias` | Expected Sarsa implemented; Double Q-learning implemented; Double Expected Sarsa not |
| (Fig. 6.4/6.6) | §6.5–§6.6, Example 6.6 | `CliffWalkingEnv`, `sarsa_control`, `q_learning_control`, `expected_sarsa_control`, `plot_figure_6_4_cliff_walking`, `plot_figure_6_6_td_control_performance` | `TestCliffWalking`, `TestControl` | Fig. 6.4 single run smoothed over 10 episodes; Fig. 6.6 reduced runs (illustrative) |

**Commands**
- Tests: `python -m pytest chapter_06/tests/ -v`
- Figures: `python chapter_06/figures/generate_figures.py`
- Lint: `ruff check chapter_06/`

**API notes (signatures, from implementation):** `RandomWalkEnv(seed=None)` with `reset() -> str` and
`step(action) -> (reward, state, done)`; `random_walk_true_values() -> dict`;
`td0_prediction(env, policy_fn=None, alpha=0.1, n_episodes=5000, gamma=1.0, seed=None) -> dict`;
`td_lambda_prediction(env, policy_fn=None, alpha=0.1, lam=0.5, n_episodes=5000, gamma=1.0, seed=None) -> dict`;
`sarsa(env, n_episodes=5000, alpha=0.1, epsilon=0.1, gamma=1.0, seed=None) -> (policy, q_table)`;
`q_learning(env, n_episodes=5000, alpha=0.1, epsilon=0.1, gamma=1.0, seed=None) -> (policy, q_table)`;
`mc_prediction_incremental(env, policy_fn=None, n_episodes=5000, gamma=1.0, seed=None) -> dict`;
`run_random_walk_comparison(repeats=100, episodes=100, alpha=0.1, seed=0) -> {'td0': [...], 'mc': [...]}`;
`CliffWalkingEnv()` with `reset() -> (row, col)` and `step(action) -> (reward, obs, done)`;
`MaximizationBiasMDP(n_b_actions=10, reward_mean=-0.1, reward_std=1.0, seed=None)`,
`actions_for(state) -> tuple`, `step(action) -> (reward, 'T', True|False)`;
`sarsa_control(env, n_episodes=1000, alpha=0.5, epsilon=0.1, gamma=1.0, seed=None, max_steps=10000) -> (q_table, steps, returns)`;
`q_learning_control(...)` and `expected_sarsa_control(...)` with identical signatures;
`double_q_learning(env, n_episodes=1000, alpha=0.1, epsilon=0.1, gamma=1.0, seed=None, max_steps=10000) -> (q1, q2, first_actions)`;
`maximization_bias_experiment(n_episodes=300, repeats=100, alpha=0.1, epsilon=0.1, gamma=1.0, seed=0) -> dict`;
`run_batch_updating(repeats=100, n_walks=100, seed=0) -> {'walks', 'td', 'mc'}`;
`batch_td_fixed_point(batch, gamma=1.0) -> dict` and `batch_mc_values(batch, gamma=1.0) -> dict`
(batch = list of episodes, each a list of `(state, reward, next_state, done)`).

---

## Verification and Quality Notes

- Derivations (TD/MC identity and correction, off-policy TD(0) with ρ, action-value identity,
  Sarsa/Q-learning/Double Expected Sarsa updates) were written to be consistent with standard S&B
  Chapter 6 results and with the actual behavior of `chapter_06/solution.py`.
- Mappings were checked against `chapter_06/solution.py` and `chapter_06/tests/test_solution.py` so
  that cited APIs, test class names (`TestRandomWalkEnv`, `TestTDPrediction`, `TestTDLambda`,
  `TestSarsa`, `TestQLearning`, `TestExercise61`, `TestBatchUpdating`, `TestCliffWalking`, `TestControl`,
  `TestMaximizationBias`, `TestWindy`), and commands match the repository.
- Exercises 6.9, 6.10, and 6.12 are now (partially) backed by implementations: `WindyGridWorld`
  covers the king's-moves (Ex. 6.9) and stochastic-wind (Ex. 6.10) experiments end to end with
  `sarsa_control`; plain Double Q-learning (Eq. 6.10) is implemented for Example 6.7 (Figure 6.7).
  Double Expected Sarsa (Ex. 6.12 strictly) remains unimplemented and is answered by derivation —
  no false implementation claims.
- Book-page anchors refer to the author's 2018 complete draft (March 16, 2018); final printings
  differ slightly. Figures were reproduced at reduced scale (fewer episodes/runs than printed)
  and are labeled illustrative; identities were taken from the 2nd edition, not the 1st or the
  2014–2015 draft (whose numbering differs).

---
*Exercises document for Chapter 6 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
