# Chapter 5 — Monte Carlo Methods: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd edition). Repo chapter: `chapter_05`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_05` — Monte Carlo prediction and control (first-visit/every-visit
  prediction, on-policy ε-greedy control, off-policy prediction and control with ordinary and
  weighted importance sampling) plus the Blackjack environment.
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

**Solution.**
1. **Rear view, last two rows jump up (player sums 20 and 21).** Under the evaluated policy the
   player sticks on 20 or 21, so these states immediately stop drawing and usually avoid busting.
   That gives much higher expected return than states like 19 where the policy still hits and risks
   going over 21. The jump reflects the policy’s action change at the threshold, not a modeling
   artifact (§5.1, Example 5.1).
2. **Left view, whole last row drops (dealer showing an ace).** A dealer ace is strong: the dealer
   has a usable ace and many chances to reach a high total (or blackjack in the initial deal),
   which lowers the player’s expected return across player sums in that column compared with a
   weak dealer showing (e.g. 4–6). The plot shows this systematic disadvantage.
3. **Frontmost values higher in the upper diagrams than the lower (usable ace vs no usable ace).**
   A usable ace means the player can count an ace as 11 without busting, giving flexibility (an
   extra safe hit option). Therefore, for the same player sum/dealer showing, states with a usable
   ace have higher (or no worse) value than those without, which is why the upper surfaces sit
   above the lower ones in Figure 5.1.

**Why this matters.** Monte Carlo prediction estimates \(v_\pi\) directly from sampled returns
(Eq. 5.1–5.2); the learned shape comes from the environment dynamics and the fixed policy.

**Coding pointer.** `BlackjackEnv` and `blackjack_stick_on_threshold(stick_threshold=20)` implement
the Example 5.1 environment/policy idea; prediction can be run with `mc_first_visit_prediction` /
`mc_every_visit_prediction`. Tests: `TestBlackjackEnv`, `test_blackjack_stick_on_threshold`.

---

### Exercise 5.2 — Backup diagram for Monte Carlo estimation of \(q_\pi\)

- **Book location:** §5.2 (Monte Carlo estimation of action values).
- **Type:** Theory / diagram reasoning.

**Problem (paraphrased).** What is the backup diagram for Monte Carlo estimation of action values
\(q_\pi(s,a)\)?

**Solution.** For action-value MC, each estimate is based on a **sample episode return** following a
visit to the state–action pair \((S_t=s, A_t=a)\): the “backup” is a single sampled trajectory
\(S_t,A_t,R_{t+1},S_{t+1},A_{t+1},\dots\) down to termination, producing
\(G_t=\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}\). Unlike DP (Eq. 4.x backups), there is **no branching
over all next states/actions by a model and no bootstrapping**; the update target is the actual
return \(G_t\). Averaging these targets over visits to \((s,a)\) estimates
\(q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]\) (the action-value analogue of Eq. 5.1).

**Why this matters.** Without a model, action values are needed for model-free improvement
(§5.2–§5.3), and MC backs up full returns rather than one-step expectations.

**Coding pointer.** Action-value averaging/backups appear inside `mc_control_epsilon_greedy` (Q of
state–action pairs) and in off-policy control returns processing. **Not implemented:** a dedicated
standalone \(q_\pi\) prediction API; prediction APIs in this repo estimate \(v_\pi\).

---

### Exercise 5.3 — Monte Carlo update/estimate for action values

- **Book location:** §5.2.
- **Type:** Math.

**Problem (paraphrased).** Give the action-value analogue of the state-value Monte Carlo estimate
(Eq. 5.2 / 5.1 idea), i.e. how to estimate \(q_\pi(s,a)\) from returns generated using policy \(\pi\)
(or behavior policy \(b\) in the off-policy sense, before correction).

**Solution.** Let \(\mathcal{T}(s,a)\) be the set of time steps at which \((s,a)\) is visited (first-visit
or every-visit, by choice) and \(G_t\) the return following time \(t\). The Monte Carlo estimate is
the average return after those visits:
\[
Q(s,a)\;\doteq\;\frac{1}{|\mathcal{T}(s,a)|}\sum_{t\in \mathcal{T}(s,a)} G_t,
\qquad \text{(analogous to Eq. 5.2 for }V(s)\text{).}
\]
Incrementally (cf. §5.6 / Eq. 2.4), after the \(n\)-th observed return \(G_n\) for \((s,a)\):
\[
Q_{n+1}(s,a)=Q_n(s,a)+\frac{1}{n}\bigl(G_n - Q_n(s,a)\bigr).
\]
This is the sample-average action-value update; off-policy learning additionally weights \(G_t\) by
importance ratios (Eq. 5.3–5.6, Ex. 5.7).

**Why this matters.** Control improves by acting greedily/ε-greedily with respect to \(Q\) (§5.3–§5.4).

**Coding pointer.** `mc_control_epsilon_greedy` maintains per-(state,action) return lists and sets
`q_table[state][action] = mean(returns)` (batch form of the same average). Tests: `TestOnPolicyControl`.

---

### Exercise 5.4 — Why weighted importance-sampling error can rise then fall (Fig. 5.3)

- **Book location:** §5.5, importance sampling examples/figures (e.g. off-policy evaluation of a
  blackjack state value).
- **Type:** Theory / math intuition.

**Problem (paraphrased).** In learning curves like Figure 5.3, the error of ordinary importance
sampling typically decreases with more episodes, but weighted importance sampling can show error that
first increases and then decreases. Why?

**Solution.** Both estimators target \(v_\pi(s)=\mathbb{E}_b[\rho_{t:T-1}G_t\mid S_t=s]\)
(Eq. 5.3–5.5), but they behave differently at small sample sizes:
- **Ordinary IS (Eq. 5.4)** divides by the number of visits/episodes \(n\), so early estimates can
  be very noisy yet often move quickly in magnitude; the curve can appear to improve in a simple MSE
  trend while still having extremely high/possibly infinite variance in such problems (see Ex. 5.5).
- **Weighted IS (Eq. 5.5)** forms a self-normalized ratio
  \((\sum \rho_t G_t)/(\sum \rho_t)\). With very few returns, the denominator and numerator may be
  dominated by a small number of atypical trajectories with moderate ratios, producing a **biased**
  estimate that can move away from the true value (apparent error increase). As more episodes are
  observed, the ratio stabilizes and the bias/variance tradeoff improves, so error then decreases and
  the estimator converges to the correct value (weighted IS is consistent; §5.5).

**Why this matters.** Variance and small-sample bias differ: ordinary IS is unbiased but can have
huge variance; weighted IS is biased (especially early) but usually much lower variance (§5.5).

**Coding pointer.** `mc_off_policy_prediction(..., weighted=False)` implements ordinary IS (Eq. 5.4
behavior in code) and `weighted=True` implements weighted IS (Eq. 5.5). `mc_control_off_policy`
supports `weighted=True/False`. Tests: `TestOffPolicyPrediction`, `TestOffPolicyControl`.

---

### Exercise 5.5 — Every-visit MC and the infinite-variance example (Example 5.5 / Fig. 5.4)

- **Book location:** §5.5, Example 5.5 (single-state example illustrating infinite variance of
  ordinary importance sampling).
- **Type:** Math / theory.

**Problem (paraphrased).** Example 5.5/Figure 5.4 used first-visit MC. If every-visit MC were used
on the same problem, would the variance of the estimator still be infinite? Why or why not?

**Solution.** **Yes, for ordinary importance sampling the variance can still be infinite.** In the
example there is one nonterminal state \(s\). Target policy \(\pi\) always selects the action that
may stay in \(s\) (so the episode can be long) and behavior policy \(b\) selects that action with
probability 1/2. Let the episode length/termination be such that a return of 1 occurs with
probability \(0.1\cdot 0.9^{k}\) after \(k\) self-loops (geometric tail), while the importance
ratio for a trajectory that stays \(k\) steps is \(\rho = 2^{k+1}\) (since \(\pi/b=2\) on each taken
step). The ordinary IS contribution involves \(\rho G\); its **second moment** scales like
\(\sum_k (2^{k+1})^2 \cdot 0.1\cdot 0.9^k\), which diverges because \(4\cdot 0.9 > 1\). Therefore
\(\operatorname{Var}(\rho G)=\infty\) in the first-visit/once-per-episode sense of that example.

Switching to **every-visit** counting means we would include multiple \((s,t)\) time steps within the
same episode, adding *more* correlated high-variance terms rather than removing the explosive weight
growth. The rare long episodes that cause huge \(\rho\) still occur with the same probabilities, so
ordinary IS does **not** become finite-variance by changing visit definition. (Weighted IS, by
contrast, bounds/normalizes contributions and has finite variance in this setting — §5.5.)

**Why this matters.** Visit definition (first vs every) affects estimator details, but it does not
cure the fundamental tail/ratio explosion that makes ordinary IS variance infinite in Example 5.5.

**Coding pointer.** `mc_off_policy_prediction` supports `weighted` and `first_visit` flags and is
used in `TestOffPolicyPrediction::test_weighted_is_infinite_variance_example` and
`test_ordinary_is_returns_entries`. **Not implemented:** a dedicated experiment reproducing Fig. 5.4.

---

### Exercise 5.6 — Incremental implementation of first-visit MC prediction

- **Book location:** §5.6 (Incremental implementation).
- **Type:** Math / programming (conceptual).

**Problem (paraphrased).** Modify first-visit MC policy evaluation (as in §5.1) to use the
incremental sample-average update from §2.4 instead of storing all returns.

**Solution.** For each state \(s\) keep a count \(N(s)\) and value estimate \(V(s)\). After an episode,
compute returns \(G_t\) backward as usual, but for the **first** time \(t\) that \(s\) is visited in
that episode do:
\[
N(s) \leftarrow N(s)+1,\qquad V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl(G_t - V(s)\bigr),
\]
and ignore later visits to \(s\) in the same episode (first-visit rule). This is exactly Eq. (2.4) /
the incremental mean, applied once per first visit. Every-visit would apply the same update on every
visit. No list of past returns is required, only \((V, N)\).

**Why this matters.** It shows MC prediction is averaging; incremental form is memory-efficient and
generalizes to nonstationary tracking by replacing \(1/N(s)\) with a constant step size \(\alpha\).

**Coding pointer.** `mc_first_visit_prediction` and `mc_every_visit_prediction` compute correct
averages (batch mean in this implementation). **Not implemented:** a separate incremental-API variant;
the math above is the required incremental form.

---

### Exercise 5.7 — Derive the weighted-average incremental update (Eq. 5.6)

- **Book location:** §5.6.
- **Type:** Math / derivation.

**Problem (paraphrased).** Derive the incremental weighted-average update rule (Eq. 5.6) from the
weighted-average definition (Eq. 5.5-style cumulative form), following the pattern of the unweighted
derivation in §2.4.

**Solution.** Define cumulative weight and weighted return sum
\(C_n \doteq \sum_{k=1}^{n} W_k\), \(S_n \doteq \sum_{k=1}^{n} W_k G_k\), with \(V_n = S_n/C_n\)
(and \(V_n\) arbitrary/unused when \(C_n=0\)). Then
\[
V_{n+1}=\frac{S_{n+1}}{C_{n+1}}=\frac{S_n + W_{n+1}G_{n+1}}{C_n + W_{n+1}}.
\]
Substitute \(S_n = C_n V_n\):
\[
V_{n+1}=\frac{C_n V_n + W_{n+1}G_{n+1}}{C_{n+1}}
= V_n + \frac{W_{n+1}}{C_{n+1}}\left(G_{n+1}-V_n\right),
\]
which is Eq. (5.6). This reduces to the ordinary sample-average update when \(W_k \equiv 1\)
(\(C_n=n\)).

**Why this matters.** This recurrence is the basis of weighted importance sampling implementations and
off-policy control with incremental updates (§5.6–§5.7).

**Coding pointer.** Weighted incremental updates appear in `mc_control_off_policy(..., weighted=True)`
(update `q += (w/C)*(g-q)`). Tests: `TestOffPolicyControl`.

---

### Exercise 5.8 — Racetrack (programming)

- **Book location:** §5.5/§5.7 programming exercise context in the book (Racetrack control task).
- **Type:** Programming / experiment.

**Problem (paraphrased).** Implement the Racetrack: a car moves on a grid with position and velocity
components, chooses acceleration-like actions from a small discrete set each step, receives \(-1\)
per time step until crossing the finish line, and must stay on the track (with a specified recovery
rule when it leaves). Apply a Monte Carlo control method to learn a policy to reach the finish from
start states.

**Solution (what to do).** A standard solution models states as \((x,y,v_x,v_y)\) with bounded
velocities, actions as increments in \(\{-1,0,1\}^2\) (9 actions), stochastic zero-acceleration noise
optional per book variant, and episodes ending on finish-line crossing. Use on-policy MC control
(ε-soft, exploring starts or ε-greedy) or off-policy MC control over sampled episodes; update
\(Q(s,a)\) from returns \(G_t=-\text{steps-to-finish}\) (γ=1) and improve the policy toward greedy.
Because transitions are unknown/complex, Monte Carlo’s model-free sampling is a natural fit.

**Why this matters.** It demonstrates MC control on a larger trajectory task where DP is impractical
without an explicit model.

**Coding pointer.** **Not implemented in `chapter_05/` (no racetrack env/API/tests).** General MC
control machinery in this repo: `mc_control_epsilon_greedy`, `mc_control_off_policy`.

---

### Exercise 5.9 — Off-policy MC control with discounted (truncated/discount-aware) weighting

- **Book location:** §5.7 (off-policy Monte Carlo control).
- **Type:** Math / programming (conceptual).

**Problem (paraphrased).** Modify the off-policy MC control algorithm to use a discounted/flat partial
return weighting idea (as introduced for off-policy estimation in §5.6 in the 2nd ed.), converting it
to action values and combining with greedy-target improvement.

**Solution.** In off-policy control we want \(q_\pi(s,a)\) for a greedy target \(\pi\) while behaving
under an ε-soft \(b\). For each episode, process steps backward accumulating return \(G\leftarrow \gamma G + R_{t+1}\)
and update \(Q(S_t,A_t)\) with an importance weight \(W\) (ordinary or weighted incremental form,
Eq. 5.6 in the weighted case). The book’s discounted/alternative weighting replaces or augments the
full-trajectory ratio by a factor that accounts for discounting over the remaining horizon (a
“discount-aware”/truncated contribution in that presentation), reducing the impact of very long-horizon
ratios while still targeting the desired off-policy return. Concretely: convert the state-value
weighting expression to \(Q(s,a)\), maintain cumulative weights per \((s,a)\), and after each update do
\(W \leftarrow W \cdot \frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}\) (stopping when \(\pi(A_t\mid S_t)=0\),
as in standard off-policy MC control). Any such modification must preserve that \(\pi\) is greedy w.r.t.
current \(Q\) and that \(b\) has coverage.

**Why this matters.** Variants of weighting/discounting trade bias and variance in off-policy control;
all rely on importance ratios (Eq. 5.3) and incremental averaging (Eq. 5.6).

**Coding pointer.** **Not implemented:** the specific discounted weighting variant requested by Ex. 5.9.
Implemented off-policy control: `mc_control_off_policy(env, ..., weighted=True|False, ...)` using
standard greedy-target IS with backward accumulation. Tests: `TestOffPolicyControl`.

---

## Coding Exercises Mapped to Repository Code

| Book exercise | Section | Repo API (`chapter_05/solution.py`) | Repo tests (`chapter_05/tests/test_solution.py`) | Status / notes |
|---:|---|---|---|---|
| 5.1 | §5.1 | `BlackjackEnv`, `blackjack_stick_on_threshold` | `TestBlackjackEnv`, `test_blackjack_stick_on_threshold` | Implemented (env/policy); value-estimation plots not included |
| 5.2 | §5.2 | (action-value backup used inside control) `mc_control_epsilon_greedy` | `TestOnPolicyControl` | Theory/diagram; no standalone \(q_\pi\) predictor |
| 5.3 | §5.2 | `mc_control_epsilon_greedy` (Q averaging) | `TestOnPolicyControl` | Implemented as control updates |
| 5.4 | §5.5 | `mc_off_policy_prediction(weighted=...)`, `mc_control_off_policy(weighted=...)` | `TestOffPolicyPrediction`, `TestOffPolicyControl` | Implemented estimators |
| 5.5 | §5.5 | `mc_off_policy_prediction(weighted=..., first_visit=...)` | `TestOffPolicyPrediction` (incl. infinite-variance weighted case) | Estimators implemented; Fig. 5.4 experiment not scripted |
| 5.6 | §5.6 | `mc_first_visit_prediction`, `mc_every_visit_prediction` | `TestPrediction` | Correct MC averages; explicit incremental variant not separate |
| 5.7 | §5.6 | `mc_control_off_policy(weighted=True)` | `TestOffPolicyControl` | Weighted incremental recurrence used |
| 5.8 | Racetrack | — | — | **Not implemented** in this chapter |
| 5.9 | §5.7 | `mc_control_off_policy` (standard ordinary/weighted IS) | `TestOffPolicyControl` | **Specific variant not implemented** |

### Commands

- Full chapter tests: `python -m pytest chapter_05/tests/ -v`
- Lint: `ruff check chapter_05/`

### API quick reference (all exist in `solution.py`)

- Environment/helpers: `BlackjackEnv` (`reset`, `step`, `ACTIONS`), `blackjack_stick_on_threshold`
- Prediction: `mc_first_visit_prediction`, `mc_every_visit_prediction`, `mc_off_policy_prediction`
- Control: `mc_control_epsilon_greedy`, `mc_control_off_policy`

### Test inventory (all exist in `tests/test_solution.py`)

- Classes: `TestBlackjackEnv`, `TestPrediction`, `TestOnPolicyControl`, `TestOffPolicyPrediction`,
  `TestOffPolicyControl`
- Top-level: `test_blackjack_stick_on_threshold`

---

## Verification Notes

- Derivations (Eqs. 5.1–5.6, incremental forms, Ex. 5.5 variance argument) are stated in standard
  Ch. 5 form and aligned with the implemented estimators where available.
- Items marked **Not implemented** are theory-only or programming tasks not present in this repo’s
  chapter 5 code; they are called out explicitly to avoid claiming nonexistent APIs/tests.

---

*Exercises document for Chapter 5 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
