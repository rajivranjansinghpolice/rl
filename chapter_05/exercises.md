# Chapter 5 — Monte Carlo Methods: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd edition). Repo chapter: `chapter_05`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_05` — Monte Carlo prediction and control (first-visit/every-visit
  prediction, MC ES exploring starts, on-policy ε-greedy control, off-policy prediction and control
  with ordinary and weighted importance sampling), the Blackjack environment, the Racetrack
  environment (Exercise 5.8, in `rl_envs`), and scripted reproductions of Figures 5.2–5.4.
- **Book chapter:** Chapter 5 — Monte Carlo Methods.
- **Book sections covered by the implementation:**
  - **§5.1** Monte Carlo prediction (`mc_first_visit_prediction`, `mc_every_visit_prediction`).
  - **§5.2** Monte Carlo estimation of action values (action-value returns/averages).
  - **§5.3** Monte Carlo control / Monte Carlo ES (exploring starts idea).
  - **§5.4** On-policy first-visit MC control for ε-soft policies (`mc_control_epsilon_greedy`).
  - **§5.5** Off-policy prediction via importance sampling (`mc_off_policy_prediction`).
  - **§5.6** Incremental implementation (sample-average / weighted-average recurrences).
  - **§5.7** Off-policy MC control (`mc_control_off_policy`).
- **Key book equations used below (paraphrased notation):**
  - Eq. **(5.1)** value as expected return: \(v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]\).
  - Eq. **(5.2)** sample-average estimate \(\,V(s)\leftarrow \operatorname{average}(\text{Returns}(s))\).
  - Incremental sample average (cf. Eq. **(2.4)** / §5.6):
    \(V_{n+1}=V_n+\frac{1}{n}(G_n - V_n)\).
  - Eq. **(5.3)** importance-sampling ratio:
    \(\rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}\).
  - Eq. **(5.4)** ordinary importance-sampling estimator:
    \(V(s)\doteq \frac{\sum_{t\in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}\).
  - Eq. **(5.5)** weighted importance-sampling estimator:
    \(V(s)\doteq \frac{\sum_{t\in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}
    {\sum_{t\in \mathcal{T}(s)} \rho_{t:T(t)-1}}\) (or \(0\) if denominator is \(0\)).
  - Eq. **(5.6)** incremental weighted-average form:
    \(V_{n+1}=V_n+\frac{W_n}{C_n}(G_n - V_n),\quad C_{n}=C_{n-1}+W_n,\; C_0=0\).

## Sources and Ground Rules

- Exercise statements are **paraphrased** from Sutton & Barto (2nd ed., Ch. 5). No long verbatim
  quotes are included. Section/equation numbers are given so answers can be checked against the book.
- Solutions/derivations below are written for this repository (student-friendly, self-contained).
- Book Ch. 5 states there are **9 exercises** (5.1–5.9). All nine are covered below.

## How to Use This Document

- Labels: **Theory**, **Computation/intuition**, **Math/Derivation**, **Programming/experiment**.
- “Coding pointer” lists only APIs/tests that actually exist in `chapter_05/solution.py` and
  `chapter_05/tests/test_solution.py`.
- Theory-only items (no direct implementation in this repo) are marked **Not implemented**.

---

## Exercises

### Exercise 5.1 — Interpreting the Blackjack value estimates (Figure 5.1)

- **Book location:** §5.1, Figure 5.1 (Blackjack example).
- **Type:** Theory / intuition.

**Problem (paraphrased).** In Figure 5.1 (estimated state values for Blackjack under the policy
“stick on 20 or 21, otherwise hit”), explain: (1) why the estimated value jumps up for the last
two player-sum rows in the rear view, (2) why it drops off for the whole last row on the left
view, and (3) why the frontmost values are higher in the upper diagrams than in the lower.

**Hook.** The estimated Blackjack surface is not smooth, and every kink is a story about the fixed
policy or the dealer's cards. Let's read the three features one at a time.

**Formal bridge.** Monte Carlo prediction estimates

\[
v_\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t=s\right],
\tag{5.1.1}
\]

where \(s\) is the player sum, dealer showing card and usable-ace flag, \(\pi\) is the fixed policy
"stick on 20 or 21, otherwise hit", and \(G_t\) is the sampled return from that state.

**Reasoning chain.** First, the rear view's last two rows jump up. Those rows are player sums 20
and 21, where the policy sticks, so the player stops drawing cards and usually avoids busting. A
sum of 19 still hits under this policy and risks going over 21. The jump is therefore the policy's
threshold showing up in the value surface — it is not a plotting artifact (§5.1, Example 5.1).

Second, the left view's whole last row drops. That row corresponds to the dealer showing an ace: a
usable ace gives the dealer many chances to reach a high total, or even blackjack on the initial
deal. Compared with a weak dealer showing 4–6, the player's expected return is systematically lower
across all player sums, so the entire row sits lower.

Third, the frontmost values are higher in the upper diagrams than in the lower ones. The upper
diagrams are states with a usable ace, which the player can count as 11 without busting. That extra
flexibility means an extra safe hit option, so for the same player sum and dealer card the usable-ace
state has value at least as high as the corresponding no-usable-ace state. Because MC prediction
averages sampled returns from the actual dynamics and fixed policy (5.1.1), all three effects are
learned rather than imposed.

**Why it makes sense.** Push each effect to an extreme. If the policy hit on 21, the value there
would collapse, because a bust would be nearly certain — the jump exists because stopping is
valuable. If the dealer always showed an ace and had blackjack every time, the player's value would
fall to zero everywhere; a weak dealer card is the opposite extreme. And an ace that cannot be used
as 11 is just a 1, with no flexibility, so the lower surfaces must be no better than the upper
ones. The direction of each feature is forced by the rules of the game.

**Check your understanding.** Prediction: change the stick threshold from 20 to 19. The jump in
the rear view should move down one row to player sum 19, and values around 20 should shift because
the player now stops earlier; the dealer-ace and usable-ace patterns should remain.

**Where this shows up in the code.** `BlackjackEnv` and `blackjack_stick_on_threshold(stick_threshold=20)`
implement the Example 5.1 environment and policy; prediction can use `mc_first_visit_prediction` or
`mc_every_visit_prediction`; tests `TestBlackjackEnv` and `test_blackjack_stick_on_threshold`.

### Exercise 5.2 — Backup diagram for Monte Carlo estimation of \(q_\pi\)

- **Book location:** §5.2 (Monte Carlo estimation of action values).
- **Type:** Theory / diagram reasoning.

**Problem (paraphrased).** What is the backup diagram for Monte Carlo estimation of action values
\(q_\pi(s,a)\)?

**Hook.** A Monte Carlo "backup" is not a one-step branch at all. Let's describe the diagram and
see why it has no model or bootstrapping in it.

**Formal bridge.** The target of an action-value backup is the sampled return after a visit to
\((s,a)\),

\[
q_\pi(s,a) = \mathbb{E}_\pi\left[G_t \mid S_t=s, A_t=a\right],
\qquad
G_t = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}.
\tag{5.2.1}
\]

where \((s,a)\) is the visited state-action pair, \(G_t\) is the complete episode return following
that visit, \(\gamma\) is the discount factor, and \(T\) is termination.

**Reasoning chain.** Draw the diagram starting at the node \((S_t=s,A_t=a)\). From it there is a
single sampled trajectory \(S_t,A_t,R_{t+1},S_{t+1},A_{t+1},\ldots\) continuing all the way down to
termination, and the contribution carried back along that one path is \(G_t\). That is the entire
backup: one path, one return.

Contrast this with a dynamic-programming backup, which branches over all next states and actions
using the model and then bootstraps from estimated values. Here there is no branching over a model
and no bootstrap — the target is the actual observed return. Averaging those returns over all visits
to \((s,a)\) estimates \(q_\pi(s,a)\), the action-value analogue of Eq. 5.1. That is why action values
are the natural object for model-free control: they can be learned from trajectories without knowing
\(p\).

**Why it makes sense.** The extreme check is an episode of length one: the path is a single reward
and \(G_t=R_{t+1}\), so the action value is just the average reward received. As episodes get longer,
the same path simply carries more rewards back, each discounted by \(\gamma^k\). If the diagram
branched over all outcomes, it would be a DP backup and would require a model — the whole point of
Monte Carlo estimation is that the sampled path stands in for the expectation.

**Check your understanding.** Prediction: although the visit definition changes which returns are
counted, the diagram does not branch either way; every-visit adds more single paths from repeated
visits to \((s,a)\), while first-visit keeps only the first path per episode.

**Where this shows up in the code.** Action-value averaging appears inside
`mc_control_epsilon_greedy` and in off-policy control return processing; a dedicated standalone
\(q_\pi\) prediction API is **not implemented** — the prediction APIs in this chapter estimate
\(v_\pi\).

### Exercise 5.3 — Monte Carlo update/estimate for action values

- **Book location:** §5.2.
- **Type:** Math.

**Problem (paraphrased).** Give the action-value analogue of the state-value Monte Carlo estimate
(Eq. 5.2 / 5.1 idea), i.e. how to estimate \(q_\pi(s,a)\) from returns generated using policy \(\pi\)
(or behavior policy \(b\) in the off-policy sense, before correction).

**Hook.** Replacing states by state-action pairs is almost all it takes. Let's write the batch and
incremental forms so the bookkeeping is explicit.

**Formal bridge.** Let \(\mathcal{T}(s,a)\) be the set of visits to \((s,a)\) (first-visit or
every-visit, by choice) and \(G_t\) the return following \(t\). The estimate is

\[
Q(s,a)\;\doteq\;\frac{1}{|\mathcal{T}(s,a)|}\sum_{t\in \mathcal{T}(s,a)} G_t.
\tag{5.3.1}
\]

After the \(n\)-th observed return \(G_n\) for that pair, the same average can be maintained as

\[
Q_{n+1}(s,a)=Q_n(s,a)+\frac{1}{n}\bigl(G_n-Q_n(s,a)\bigr).
\tag{5.3.2}
\]

where \(Q(s,a)\) estimates \(q_\pi(s,a)\), \(n=|\mathcal{T}(s,a)|\), and \(G_t\) is the episode return
after the visit; in the off-policy case the returns are additionally weighted by importance ratios
(Ex. 5.7).

**Reasoning chain.** The batch form (5.3.1) is just Eq. 5.2 with \(s\) replaced by \((s,a)\): collect
every return that followed a visit to that pair and average them. Choosing first-visit or
every-visit determines which time steps enter \(\mathcal{T}(s,a)\), but not the form of the average.

The incremental form (5.3.2) is the sample-average recursion of §2.4 applied separately to each
state-action pair: the error \(G_n-Q_n\) is moved into the estimate by a step of size \(1/n\). This
avoids storing the full list of returns.

If the data came from a different behaviour policy \(b\), these raw averages estimate \(q_b\) until
correction; off-policy learning weights each \(G_t\) by an importance ratio (Eq. 5.3–5.6) so the
average targets \(q_\pi\) instead. Control then improves the policy by acting greedily or
ε-greedily with respect to \(Q\).

**Why it makes sense.** With a single visit, (5.3.2) sets \(Q=G_1\) exactly — the observed return is
the only evidence available. With many visits, each new return moves the estimate by \(1/n\), so
early samples matter more and later ones are progressively averaged out. The direction is right:
if a new return is higher than the current estimate, the estimate rises; a lower return pulls it
down.

**Check your understanding.** Prediction: replace the step size \(1/n\) by a constant \(\alpha\). The
estimate becomes a recency-weighted average that can track changing action values, but it no longer
equals the exact mean of the observed returns.

**Where this shows up in the code.** `mc_control_epsilon_greedy` maintains per-\((s,a)\) return
lists and sets `q_table[state][action] = mean(returns)`; test `TestOnPolicyControl`.

### Exercise 5.4 — Why weighted importance-sampling error can rise then fall (Fig. 5.3)

- **Book location:** §5.5, importance sampling examples/figures (e.g. off-policy evaluation of a
  blackjack state value).
- **Type:** Theory / math intuition.

**Problem (paraphrased).** In learning curves like Figure 5.3, the error of ordinary importance
sampling typically decreases with more episodes, but weighted importance sampling can show error
that first increases and then decreases. Why?

**Hook.** Both estimators chase the same target, but they normalise their samples in different
ways. That difference is invisible in the limit and dramatic in the first few episodes.

**Formal bridge.** Both target

\[
v_\pi(s)=\mathbb{E}_b\!\left[\rho_{t:T-1}G_t\mid S_t=s\right],
\qquad
\rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},
\tag{5.4.1}
\]

and the two estimators are the ordinary average \(\sum \rho G/n\) and the self-normalised ratio
\((\sum \rho G)/(\sum \rho)\).

where \(\rho_{t:T-1}\) is the importance-sampling ratio from step \(t\) to termination, \(\pi\) is
the target policy, \(b\) is the behaviour policy, \(G_t\) is the return, and \(n\) is the number of
returns for \(s\).

**Reasoning chain.** Ordinary IS (Eq. 5.4) divides the weighted return sum by the visit count \(n\),
not by the sum of weights. With few episodes it can be very noisy, and in problems like Example 5.5
its variance may be extremely high or even infinite; nevertheless its error curve can appear to
improve steadily in a simple mean-squared-error plot because its early estimates are not distorted
by a random denominator.

Weighted IS (Eq. 5.5) uses the self-normalised form \((\sum \rho G)/(\sum \rho)\). With very few
returns, both the numerator and denominator can be dominated by a handful of atypical trajectories
with moderate ratios, producing a biased estimate that can first move *away* from the true value —
the apparent rise in error. As more episodes arrive, the accumulated weights stabilise, the
self-normalisation bias shrinks, and the estimator converges to the correct value: weighted IS is
consistent (§5.5), and usually has far lower variance than ordinary IS.

**Why it makes sense.** Picture one early trajectory with a large ratio \(\rho\). Ordinary IS scales
its contribution by \(\rho/n\), so a single huge value can dominate — an infinite-variance risk.
Weighted IS divides by the sum of ratios, so that same trajectory is partly cancelled by its own
weight in the denominator; the estimate is dragged toward that trajectory's return when the sample
is small, which is a bias, but it cannot be multiplied without bound. As the denominator grows,
one trajectory's influence fades. This is the classic bias-variance trade: lower variance bought
with a temporary, small-sample bias — hence an error curve that rises before it falls.

**Check your understanding.** Prediction: start the experiment from a larger initial sample count.
The weighted-IS curve should skip most of its rising phase, because the self-normalising denominator
is already stable; ordinary IS remains noisy whenever a rare large-ratio trajectory appears.

**Where this shows up in the code.** `mc_off_policy_prediction(..., weighted=False)` implements
ordinary IS and `weighted=True` implements weighted IS; `mc_control_off_policy` supports both.
`plot_importance_sampling` reproduces Figure 5.3 (MSE of ordinary vs weighted IS on the single
blackjack state — player sum 13 / dealer 2 / usable ace — averaged over 100 runs, with the reference
value estimated by 200,000-episode first-visit MC prediction rather than hardcoded). Tests:
`TestOffPolicyPrediction` and `TestOffPolicyControl`.

### Exercise 5.5 — Every-visit MC and the infinite-variance example (Example 5.5 / Fig. 5.4)

- **Book location:** §5.5, Example 5.5 (single-state example illustrating infinite variance of
  ordinary importance sampling).
- **Type:** Math / theory.

**Problem (paraphrased).** Example 5.5/Figure 5.4 used first-visit MC. If every-visit MC were used
on the same problem, would the variance of the estimator still be infinite? Why or why not?

**Hook.** Switching to every-visit changes how often we count, but not how wildly the importance
weights grow. Let's check the tail that actually causes the divergence.

**Formal bridge.** In the one-state example, a trajectory with \(k\) self-loops has importance ratio
\(\rho=2^{k+1}\) (since \(\pi/b=2\) on each taken step), while a return of 1 occurs after \(k\) loops
with probability \(0.1\cdot 0.9^{k}\). The ordinary IS second moment therefore contains

\[
\sum_k \left(2^{k+1}\right)^2 \cdot 0.1\cdot 0.9^{k},
\tag{5.5.1}
\]

which diverges because \(4\cdot0.9>1\).

where \(k\) is the number of self-loops, \(2^{k+1}\) is the importance ratio along that trajectory,
and \(0.1\cdot0.9^{k}\) is the probability of the corresponding return.

**Reasoning chain.** The example has one nonterminal state \(s\). The target policy \(\pi\) always
selects the action that may stay in \(s\), so episodes can be long; the behaviour policy \(b\) picks
that action with probability \(1/2\), giving a ratio of \(2\) per step. Returns of 1 occur with the
geometric tail \(0.1\cdot0.9^{k}\) after \(k\) self-loops, while \(\rho=2^{k+1}\). The second moment of
\(\rho G\) scales as \(\sum_k 4^{k+1}\cdot0.1\cdot0.9^{k}\), and since \(4\cdot0.9>1\) the terms grow —
the variance of ordinary IS is infinite in the once-per-episode sense of the example.

Does every-visit fix this? No. Counting every visit to \(s\) within the same episode adds *more*
correlated high-variance terms, each carrying the same explosive weights; it does not remove the
rare long trajectories with huge \(\rho\). Those trajectories still occur with the same
probabilities, so ordinary IS does not become finite-variance by changing the visit definition.
Weighted IS, by contrast, normalises the contributions and has finite variance in this setting
(§5.5).

**Why it makes sense.** The divergence is a race between two exponentials: the ratio grows like
\(4^k\) in the second moment while the probability of such trajectories decays like \(0.9^k\). When
the growth wins (\(4\cdot0.9>1\)), the tail has infinite variance — no counting rule can change
that inequality. If the behaviour policy were closer to the target, say \(\pi/b=1\) on that action,
the ratio would stay at 1 and the same return distribution would have finite variance. The problem
is the mismatch between policies, not the definition of a visit.

**Check your understanding.** Prediction: raise the probability of continuing the self-loop so the
tail decays more slowly. The series in (5.5.1) diverges even faster; conversely, make \(b\) choose
the loop action with probability close to 1, so \(\pi/b\approx1\), and the variance becomes finite.

**Where this shows up in the code.** `plot_importance_sampling` (Figure 5.3) and
`plot_ordinary_is_unstable` (Figure 5.4, via `OneStateIsEnv` and `_one_state_is_samples`) in
`chapter_05/solution.py`; `mc_off_policy_prediction` supports `weighted` and `first_visit` flags,
used in `TestOffPolicyPrediction.test_weighted_is_infinite_variance_example` and
`test_ordinary_is_returns_entries`; the vectorized sampler is covered by `TestOneStateIs`. The
vectorized run uses \(10^6\) episodes per run (the book prints to \(10^8\)), so deep-tail spikes are
smaller; per-run means (about 0.78–0.92) typically sit below the true value, which is itself the
infinite-variance phenomenon at work.

### Exercise 5.6 — Incremental implementation of first-visit MC prediction

- **Book location:** §5.6 (Incremental implementation).
- **Type:** Math / programming (conceptual).

**Problem (paraphrased).** Modify first-visit MC policy evaluation (as in §5.1) to use the
incremental sample-average update from §2.4 instead of storing all returns.

**Hook.** MC prediction is an average, so it can be maintained incrementally. Let's replace the list
of returns with two numbers per state: a count and a running estimate.

**Formal bridge.** For each state keep \(N(s)\) and \(V(s)\); after an episode, for the first visit
only,

\[
N(s)\leftarrow N(s)+1,
\qquad
V(s)\leftarrow V(s)+\frac{1}{N(s)}\left(G_t-V(s)\right).
\tag{5.6.1}
\]

where \(G_t\) is the return following that first visit, \(N(s)\) counts first visits so far, and
\(V(s)\) is the running average; a constant step size \(\alpha\) may replace \(1/N(s)\) for tracking.

**Reasoning chain.** After each episode, compute returns \(G_t\) backwards as usual. When a state is
encountered for the first time in that episode, increment its count and move its estimate toward
the new return by a fraction \(1/N(s)\) of the error \(G_t-V(s)\). Ignore any later visits to the
same state in that episode — that is the first-visit rule. Every-visit MC would apply the same
update at each visit instead.

This is exactly the incremental mean of §2.4, Eq. (2.4), applied once per qualifying visit, so no
list of past returns is needed: the pair \((V,N)\) carries everything. Replacing \(1/N(s)\) by a
constant \(\alpha\) turns the estimator into a recency-weighted tracker for nonstationary problems.

**Why it makes sense.** On the first visit, \(N=1\) and the update sets \(V=G_t\) exactly. On the
second, the new return moves the estimate halfway; on the hundredth, only one percent. That is
precisely the behaviour of a running average — early data has large influence because little else
is known, and later data is averaged in gently. If the step size did not shrink, one noisy return
could keep overturning a well-established estimate.

**Check your understanding.** Prediction: switch from first-visit to every-visit. Each state then
receives more updates per episode, so \(N(s)\) grows faster and individual returns have less effect;
the estimates converge to a different (but still valid) average of returns.

**Where this shows up in the code.** `mc_first_visit_prediction` and `mc_every_visit_prediction`
compute the correct averages (batch mean in this implementation); a separate incremental-API variant
is **not implemented** — the recurrence above is the required incremental form.

### Exercise 5.7 — Derive the weighted-average incremental update (Eq. 5.6)

- **Book location:** §5.6.
- **Type:** Math / derivation.

**Problem (paraphrased).** Derive the incremental weighted-average update rule (Eq. 5.6) from the
weighted-average definition (Eq. 5.5-style cumulative form), following the pattern of the unweighted
derivation in §2.4.

**Hook.** The weighted average differs from an ordinary mean only by carrying a cumulative weight.
Let's unfold it and watch the update fall out.

**Formal bridge.** Define the cumulative weight and weighted return sum

\[
C_n \doteq \sum_{k=1}^{n} W_k,
\qquad
S_n \doteq \sum_{k=1}^{n} W_k G_k,
\qquad
V_n = \frac{S_n}{C_n}.
\tag{5.7.1}
\]

where \(W_k\) is the importance weight of the \(k\)-th return \(G_k\), \(C_n\) is their sum, and
\(V_n\) is the weighted-average estimate (arbitrary/unused when \(C_n=0\)).

**Reasoning chain.** Write the next estimate with the accumulated quantities:

\[
V_{n+1}=\frac{S_{n+1}}{C_{n+1}}
       =\frac{S_n+W_{n+1}G_{n+1}}{C_n+W_{n+1}}.
\tag{5.7.2}
\]

Now substitute \(S_n=C_nV_n\), which follows from the definition in (5.7.1):

\[
V_{n+1}=\frac{C_nV_n+W_{n+1}G_{n+1}}{C_{n+1}}
       =V_n+\frac{W_{n+1}}{C_{n+1}}\left(G_{n+1}-V_n\right),
\tag{5.7.3}
\]

which is Eq. (5.6), with the step size \(W_{n+1}/C_{n+1}\). When every weight is \(W_k\equiv1\),
\(C_n=n\) and (5.7.3) reduces to the ordinary sample-average update
\(V_{n+1}=V_n+\frac1n(G_{n+1}-V_n)\).

**Why it makes sense.** The step size is the new weight's *share* of all weight seen so far. A large
new weight means that return is trustworthy, so the estimate moves a lot; a tiny weight means it is
nearly ignored. If all weights are equal, the share is \(1/n\), recovering the familiar mean. Since
\(0\le W_{n+1}/C_{n+1}\le1\), the estimate always moves toward the new return, never past it — the
right direction for an average.

**Check your understanding.** Prediction: let the first ten returns have weight 1 and the eleventh
have weight 100. The step size for that return is \(100/110\approx0.91\), so the estimate jumps close
to \(G_{11}\); if the eleventh weight were 0.01, the step size would be about \(0.001\) and almost
nothing would change.

**Where this shows up in the code.** Weighted incremental updates appear in
`mc_control_off_policy(..., weighted=True)` via `q += (w/C)*(g-q)`; test `TestOffPolicyControl`.

### Exercise 5.8 — Racetrack (programming)

- **Book location:** §5.5/§5.7 programming exercise context in the book (Racetrack control task).
- **Type:** Programming / experiment.

**Problem (paraphrased).** Implement the Racetrack: a car moves on a grid with position and velocity
components, chooses acceleration-like actions from a small discrete set each step, receives \(-1\)
per time step until crossing the finish line, and must stay on the track (with a specified recovery
rule when it leaves). Apply a Monte Carlo control method to learn a policy to reach the finish from
start states.

**Hook.** Racetrack is where Monte Carlo control meets a task too large for a model-based sweep.
Let's pin down the rules and then the training setup.

**Formal bridge.** The agent maximises the return of a trajectory that ends at the finish line,

\[
G_t=\sum_{k=0}^{T-t-1}\gamma^k(-1)=-T+t \quad (\gamma=1),
\tag{5.8.1}
\]

where \(T\) is the step on which the car crosses the finish, so each step costs \(-1\) and the
shortest completed trajectory has the best return.

where the state is \((\text{row},\text{col},v_x,v_y)\) with velocity components clipped to
\(\{0,\ldots,4\}\), the actions are the 9 acceleration choices (each component up, stay or down by
1), and \(T\) is the terminal step.

**Reasoning chain.** `RacetrackEnv` in `rl_envs/racetrack.py` implements the book's rules: velocity
may not be \((0,0)\) off the start line; reward is \(-1\) per step; episodes start at a random
starting-line cell with \(v=(0,0)\). Before applying a move, the straight projected path is checked:
if it crosses the finish line the episode ends; otherwise, if it crosses off-track, the car is reset
to a random start cell with \(v=(0,0)\) and the episode continues — the boundary check is performed
*after* the finish check. With probability `noise_prob` (0.1 in the book) the velocity increments
are forced to zero that step; `max_steps` can cap episode length so control episodes terminate. The
tracks are ASCII maps, and the default is a small right-turn track in the spirit of the book's
Figure 5.5 — an illustrative demonstration, not a pixel-identical copy of an edition-specific grid.

Training uses the repository's MC control: `mc_control_epsilon_greedy` on
`RacetrackEnv(seed=0, noise_prob=0.0, max_steps=100)` runs and returns 9-dimensional ε-soft
policies; `mc_exploring_starts` also applies, covering every reachable \((s,a)\) pair. The state
space is \(46\times25=1150\) cells times 9 actions, roughly 10k \(Q\)-entries. A full racetrack solve
rivalling the book's tuned runs is not claimed here; what is implemented and tested is the
environment and the MC machinery needed to train it.

**Why it makes sense.** Because the reward is \(-1\) per step and zero otherwise, the objective
(5.8.1) is simply "finish as soon as possible", with off-track resets adding wasted steps. That is
exactly the kind of objective Monte Carlo control handles well: it needs complete episodes and no
model, whereas dynamic programming would need the transition probabilities of every acceleration on
every grid cell. The direction of learning is unambiguous — each reset or detour adds another
\(-1\), so the learned policy should drive more directly toward the finish.

**Check your understanding.** Prediction: turn on velocity noise. Episodes get longer and more
variable, so the value of careful, lower-speed control rises and the learned policy becomes more
conservative; turning noise off should produce faster, riskier trajectories.

**Where this shows up in the code.** `RacetrackEnv` in `rl_envs/racetrack.py` (exported from
`rl_envs/__init__.py`), with `mc_control_epsilon_greedy` and `mc_exploring_starts` in
`chapter_05/solution.py`; tests `tests/test_rl_envs.py::TestRacetrack` and
`chapter_05/tests/test_solution.py::TestRacetrackControl`.

### Exercise 5.9 — Off-policy MC control with discounted (truncated/discount-aware) weighting

- **Book location:** §5.7 (off-policy Monte Carlo control).
- **Type:** Math / programming (conceptual).

**Problem (paraphrased).** Modify the off-policy MC control algorithm to use a discounted/flat partial
return weighting idea (as introduced for off-policy estimation in §5.6 in the 2nd ed.), converting it
to action values and combining with greedy-target improvement.

**Hook.** Off-policy control already combines importance ratios with backward returns. The question
is how to fold a discount-aware weighting into that loop without breaking the target.

**Formal bridge.** The update targets \(q_\pi(s,a)\) for a greedy target \(\pi\) while behaving under
an ε-soft \(b\), using the ratio

\[
W \leftarrow W\cdot\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)},
\qquad
G\leftarrow \gamma G+R_{t+1},
\tag{5.9.1}
\]

with cumulative-weight updates such as Eq. 5.6 in the weighted case, and \(W=0\) stopping the
truncation when \(\pi(A_t\mid S_t)=0\).

where \(b\) is the behaviour policy with coverage of all actions, \(\pi\) is greedy with respect to
the current \(Q\), \(G\) is the backward-accumulated return, and \(W\) is the running importance
weight.

**Reasoning chain.** For each episode, process steps backward: accumulate the return with
\(G\leftarrow\gamma G+R_{t+1}\) and update \(Q(S_t,A_t)\) using the accumulated weight \(W\), either in
ordinary form or with the weighted incremental recurrence of Eq. 5.6. The book's discounted or flat
partial-return weighting replaces or augments the full-trajectory ratio by a factor that accounts
for discounting over the remaining horizon — a "discount-aware" or truncated contribution in that
presentation — reducing the impact of very long-horizon ratios while still targeting the desired
off-policy return.

Concretely, convert the state-value weighting expression to an action-value one, maintain cumulative
weights per \((s,a)\), and after each update apply (5.9.1), stopping when
\(\pi(A_t\mid S_t)=0\), as in standard off-policy MC control. Any such modification must preserve
two invariants: \(\pi\) remains greedy with respect to the current \(Q\), and \(b\) retains coverage
of the actions \(\pi\) might choose. Without coverage the ratio is undefined; without greediness the
target policy is not the one being improved.

**Why it makes sense.** Long trajectories multiply many ratios, so their weights can explode even
when their returns are heavily discounted — the same tail problem seen in Exercise 5.5. A
discount-aware or truncated weighting dampens that multiplication by recognising that rewards far in
the future contribute little to \(G\) anyway. The direction is right: reduce variance from distant
steps while keeping the one-step correction exact; pushing the truncation too far, however, biases
the estimate by discarding real returns.

**Check your understanding.** Prediction: shorten the truncation horizon further. Variance drops
because fewer long products of ratios enter, but the estimate becomes more biased toward short-term
returns; if behaviour \(b\) stops covering an action that \(\pi\) prefers, the ratio in (5.9.1)
cannot correct for it at all.

**Where this shows up in the code.** The specific discounted weighting variant is **not
implemented**. The available API is `mc_control_off_policy(env, ..., weighted=True|False, ...)`,
using standard greedy-target importance sampling with backward accumulation; test
`TestOffPolicyControl`.

---

## Coding Exercises Mapped to Repository Code

| Book exercise | Section | Repo API (`chapter_05/solution.py`) | Repo tests (`chapter_05/tests/test_solution.py`) | Status / notes |
|---:|---|---|---|---|
| 5.1 | §5.1 | `BlackjackEnv`, `blackjack_stick_on_threshold` | `TestBlackjackEnv`, `test_blackjack_stick_on_threshold` | Implemented (env/policy); value-estimation plots not included |
| 5.2 | §5.2 | (action-value backup used inside control) `mc_control_epsilon_greedy` | `TestOnPolicyControl` | Theory/diagram; no standalone \(q_\pi\) predictor |
| 5.3 | §5.2 | `mc_control_epsilon_greedy` (Q averaging) | `TestOnPolicyControl` | Implemented as control updates |
| 5.4 | §5.5 | `mc_off_policy_prediction(weighted=...)`, `mc_control_off_policy(weighted=...)`, `plot_importance_sampling` | `TestOffPolicyPrediction`, `TestOffPolicyControl` | Estimators implemented; Fig. 5.3 scripted |
| 5.5 | §5.5 | `OneStateIsEnv`, `plot_ordinary_is_unstable`, `mc_off_policy_prediction(weighted=..., first_visit=...)` | `TestOffPolicyPrediction` (incl. infinite-variance weighted case), `TestOneStateIs` | Estimators implemented; Fig. 5.4 scripted |
| 5.6 | §5.6 | `mc_first_visit_prediction`, `mc_every_visit_prediction` | `TestPrediction` | Correct MC averages; explicit incremental variant not separate |
| 5.7 | §5.6 | `mc_control_off_policy(weighted=True)` | `TestOffPolicyControl` | Weighted incremental recurrence used |
| 5.8 | Racetrack | `RacetrackEnv` (`rl_envs`), `mc_control_epsilon_greedy`, `mc_exploring_starts` | `tests/test_rl_envs.py::TestRacetrack`, `TestRacetrackControl` | **Implemented** (small illustrative right-turn track) |
| 5.9 | §5.7 | `mc_control_off_policy` (standard ordinary/weighted IS) | `TestOffPolicyControl` | **Specific variant not implemented** |

### Commands

- Full chapter tests: `python -m pytest chapter_05/tests/ -v`
- Lint: `ruff check chapter_05/`

### API quick reference (all exist in `solution.py` unless noted)

- Environment/helpers: `BlackjackEnv` (`reset`, `step`, `ACTIONS`), `blackjack_stick_on_threshold`
- Racetrack: `RacetrackEnv` in `rl_envs/racetrack.py` (`ACTIONS`, `RACETRACK_ACTIONS`, exports from `rl_envs/__init__.py`)
- Prediction: `mc_first_visit_prediction`, `mc_every_visit_prediction`, `mc_off_policy_prediction`
- Control: `mc_control_epsilon_greedy`, `mc_control_off_policy`, `mc_exploring_starts`
- Example 5.5: `OneStateIsEnv`, `plot_ordinary_is_unstable`, `_one_state_is_samples`
- Figure 5.3: `plot_importance_sampling`, `_log_step_list`

### Test inventory (all exist in `tests/test_solution.py`)

- Classes: `TestBlackjackEnv`, `TestPrediction`, `TestOnPolicyControl`, `TestOffPolicyPrediction`,
  `TestOffPolicyControl`, `TestExploringStarts`, `TestOneStateIs`, `TestRacetrackControl`
- Top-level: `test_blackjack_stick_on_threshold`
- Racetrack environment tests live in `tests/test_rl_envs.py::TestRacetrack`

---

## Verification Notes

- Derivations (Eqs. 5.1–5.6, incremental forms, Ex. 5.5 variance argument) are stated in standard
  Ch. 5 form and aligned with the implemented estimators where available.
- Items marked **Not implemented** are theory-only or programming tasks not present in this repo’s
  chapter 5 code; they are called out explicitly to avoid claiming nonexistent APIs/tests.

---

*Exercises document for Chapter 5 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
