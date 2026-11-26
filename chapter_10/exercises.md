# Chapter 10 — On-policy Control with Approximation: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd edition). Repo chapter: `chapter_10`.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_10` — On-policy control with function approximation
  (episodic semi-gradient Sarsa on Mountain Car; average-reward differential
  semi-gradient Sarsa on Access-Control Queuing).
- **Book chapter:** Chapter 10 — On-policy Control with Approximation.
- **Book sections covered by the implementation:**
  - **§10.1** Episodic semi-gradient control — `EpisodicSemiGradientSarsaAgent`,
    `mountain_car_features`, `run_mountain_car`.
  - **§10.2/§10.3** The average-reward setting — `DifferentialSarsaAgent`.
  - **Example 10.1** Mountain Car — `run_mountain_car`.
  - **Example 10.2** Access-Control Queuing — `AccessControlQueue`,
    `run_access_control`.
- **Book-identity note:** the local PDF text ships only a summary of Chapter 10
  (no exact exercise/figure numbers can be verified), so the exercises below are
  *derived from the chapter's sections* rather than quoted verbatim, and both
  figures are labelled as illustrative demonstrations, not claimed numbered book
  figures.
- **Reuse:** tile coding is shared with Chapter 9 (`chapter_09.solution.TileCoder`)
  to keep the repo DRY; Mountain Car dynamics live in `rl_envs.mountain_car`.

---

## Key book equations (plausible citations, S&B 2nd ed.)

- Episodic semi-gradient Sarsa update:
  \(w_a \leftarrow w_a + \alpha \frac{1}{n} \delta \, \phi(S)\),
  \(\delta = R + \gamma \hat q_{A'}(S') - \hat q_A(S)\), with \(\hat q_a(s) = w_a^\top \phi(s)\).
- Differential setting: \(\delta = R - \bar R + \hat q_{A'}(S') - \hat q_A(S)\) and
  \(\bar R \leftarrow \bar R + \alpha_{\bar R} \delta\).

---

## Exercise 10.1 — Why episodic control needs the bootstrap value only at terminal

- **Book location:** Section 10.1 (algorithm for episodic semi-gradient control).
- **Type:** derivation/explanation.
- **Problem (paraphrased):** In the episodic case, the semi-gradient Sarsa target
  at a terminal transition is just the reward R. Show why and how the
  implementation handles it.

**Hook.** Nothing mysterious happens at the finish line — but if the code forgets to treat it specially, the whole value scale drifts. Let's see why the terminal case is actually the simple one.

**Formal bridge.** The action-value target at a transition \(S\to S'\) is

\[
U=R+\gamma\hat q(S',A'),
\qquad
\hat q(S_{\text{term}},\cdot)=0
\;\Longrightarrow\;
U_{\text{terminal}}=R,
\tag{10.1.1}
\]

so the TD error at the final step collapses to

\[
\delta=R-\hat q_A(S).
\tag{10.1.2}
\]

where \(R\) is the reward, \(\gamma\) the discount, \(\hat q\) the approximated action value, \(S_{\text{term}}\) the terminal state, and \(\delta\) the TD error; the value of a terminal state is identically 0, and the Q-values for terminal (no following action) are defined as 0.

**Reasoning chain.** In the episodic setting every episode ends in a terminal state \(S_{\text{term}}\) with \(v(S_{\text{term}})\equiv 0\). At the final step \(S\to S_{\text{term}}\), the bootstrap term \(\gamma\hat v(S_{\text{term}})\) or \(\gamma\hat q(S_{\text{term}},\cdot)\) is exactly 0, because there is no future to bootstrap from. Hence the last update in an episode is (10.1.2): the whole target collapses to the last reward.

Any implementation that bootstraps from \(\gamma\hat q(S')\) without special-casing terminals would inflate bootstrap values for states that lead to termination. The implementation handles this by taking `next_x = None` for terminal transitions and using `q_next = 0.0`; the Mountain Car loop in `run_mountain_car` passes `None` when `done`.

**Why it makes sense.** A terminal state is "the finish line with no credit to keep." When you are one step from the finish, the best you can say about the future is 0, so the target cannot be anything but \(R\). That is why episodes, unlike continuing tasks, do not need a bootstrap discount at their end: there is no estimated future at all. A positive bootstrap at termination would mean believing in rewards after the game is over.

**Check your understanding.** Prediction: if the terminal guard were removed and a stale nonzero \(\hat q(S_{\text{term}},\cdot)\) were used, states leading to termination would be overvalued by roughly \(\gamma\hat q(S_{\text{term}},\cdot)\); setting \(\gamma=0\) would hide the bug because the bootstrap is multiplied away.

**Where this shows up in the code.** `EpisodicSemiGradientSarsaAgent.update_sarsa` takes `next_x = None` for terminal transitions and uses `q_next = 0.0`; the Mountain Car loop in `run_mountain_car` passes `None` when `done`. Unit tests assert the exact one-step arithmetic on a terminal transition (`TestEpisodicSemiGradientSarsa`).

---

## Exercise 10.2 — Why divide the Mountain Car step size by the number of tilings?

- **Book location:** Section 10.1 / 9.4.3 (tile coding).
- **Type:** conceptual/design.
- **Problem (paraphrased):** In tile coding each state activates exactly one tile
  per tiling (8 tilings => 8 active features). Show that the weights move by the
  equivalent of one feature being stepped by \(8\times\) the per-feature step size,
  and derive the compensating factor.

**Hook.** One observation lights up eight features at once. If each of them takes a full step, the learner has effectively taken eight steps — let's quantify and fix that.

**Formal bridge.** With \(n_{\text{tilings}}=8\) active binary features, the semi-gradient update and the resulting change in the prediction are

\[
w\leftarrow w+\alpha\delta\,\phi(s),
\qquad
\Delta\hat v\approx\alpha\|\phi(s)\|^2\delta=\alpha\cdot 8\cdot\delta,
\tag{10.2.1}
\]

so matching a single-feature intended step \(\alpha'\delta\) requires

\[
\alpha=\frac{\alpha'}{8}\quad\Longleftrightarrow\quad \alpha'=8\alpha.
\tag{10.2.2}
\]

where \(\phi(s)\) is the feature vector with exactly 8 non-zero entries, \(w\) the weight vector, \(\delta\) the TD error, \(\alpha\) the per-feature step size, and \(\alpha'\) the equivalent single-feature step size.

**Reasoning chain.** Each tiling activates exactly one tile per state, so \(\phi(s)\) has exactly 8 non-zero entries. In the update (10.2.1), each active feature receives \(\alpha\delta\), so all 8 are updated equally. The *total* movement in the prediction is \(\Delta\hat v\approx\alpha\|\phi(s)\|^2\delta=\alpha\cdot 8\cdot\delta\), because the squared norm of the feature vector counts those eight ones.

If the intended movement is the single-feature step \(\alpha'\delta\), we need \(\alpha' = 8\alpha\), i.e. the per-feature step must be \(\alpha=\alpha'/8\). Equivalently the code updates each active weight by \((\alpha/8)\delta\), so the system as a whole hears each event exactly once.

**Why it makes sense.** Tile coding spreads one observation over 8 features, like 8 witnesses each recording a fraction of the same event. If you charged full price at each witness, the total would be 8× the intended correction and the agent would learn far too aggressively. Dividing by 8 keeps the aggregate step size independent of an arbitrary representational choice — a nice instance of the principle that changing your features should not silently change your learning rate.

**Check your understanding.** Prediction: switching to 4 tilings without adjusting \(\alpha\) would halve the total prediction change per step; to keep the same aggregate learning rate you would double the per-feature \(\alpha\) (or keep dividing by the active-feature count).

**Where this shows up in the code.** `EpisodicSemiGradientSarsaAgent.update_sarsa` uses `(self.alpha / 8.0) * delta * x`.

---

## Exercise 10.3 — Average-reward differential Sarsa on a continuing task

- **Book location:** Sections 10.2/10.3 (average-reward setting).
- **Type:** conceptual/derivation.
- **Problem (paraphrased):** Show why the discounted setting is equivalent, in the
  limit and under a uniform policy, to the average-reward setting, and why the
  differential update removes the discounting bias.

**Hook.** In an endless game, "total reward" is infinite and tells you nothing. Let's replace it with reward per step and see why the differential target follows naturally.

**Formal bridge.** The performance measure in a continuing ergodic task is

\[
r(\pi)=\lim_{h\to\infty}\mathbb E\Bigl[\frac{1}{h}\sum_{t=0}^{h-1}R_t\Bigr],
\tag{10.3.1}
\]

and the differential Sarsa update is

\[
\delta=R-\bar R+\hat q_{A'}(S')-\hat q_A(S),
\qquad
\bar R\leftarrow\bar R+\alpha_{\bar R}\delta.
\tag{10.3.2}
\]

where \(r(\pi)\) is the average reward under policy \(\pi\), \(h\) the horizon, \(R_t\) the reward at time \(t\), \(\bar R\) the running estimate of average reward, \(\hat q\) the action-value estimate, \(\alpha_{\bar R}\) the step size for \(\bar R\), and \(\delta\) the differential TD error.

**Reasoning chain.** In a continuing ergodic task, the total discounted return from any fixed start point is unbounded, so the fundamental value is the average reward \(r(\pi)\) of (10.3.1). Optimality is judged against every other policy's average, not against discounted return: a selfish "front-load" bias toward early rewards is inappropriate when there is no horizon.

The discounted Bellman equation \(v=\sum_a\pi(a|s)[r+\gamma\sum_{s'}P(s'|s,a)v(s')]\) in the \(\gamma\to1\) limit is undefined; continuing rewards make the discounted values diverge. The differential (average-reward) Bellman equation, built around the differential target \(R-\bar R\), is well-defined instead. The differential Sarsa update in (10.3.2) tracks both \(q\) and the average reward \(\bar R\) by stochastic approximation of that differential Bellman equation: each reward is judged relative to the running average rather than against a discounted future.

This is the sense in which the discounted and average-reward views line up in the limit under a uniform policy: the discount only adds a front-loading weighting that vanishes when every step is equally represented, and the differential form removes that bias explicitly.

**Why it makes sense.** In an endless game, "maximize total reward" is infinite with no meaning; the fair question is "maximize reward *per step*, forever." Differential Sarsa lives entirely on that average. The sign in (10.3.2) is informative: a reward above \(\bar R\) yields a positive contribution, pushing the policy toward actions that beat the current norm, while a reward below \(\bar R\) pushes it away — exactly the direction you want for maximizing the long-run average.

**Check your understanding.** Prediction: if the task becomes episodic with a short horizon, the average-reward formulation loses its advantage and the discounting bias becomes harmless; conversely, raising \(\gamma\) toward 1 in a continuing task makes discounted values blow up while \(\bar R\) stays finite.

**Where this shows up in the code.** `DifferentialSarsaAgent.update` implements the differential update with the running average `r_bar`; see `TestDifferentialSarsa`.

---

## Exercise 10.4 — Access-Control Queuing: why prioritize high-priority customers?

- **Book location:** Example 10.2 (Access-control queuing).
- **Type:** analysis of the implemented experiment.
- **Problem (paraphrased):** Explain why a differential Sarsa agent should learn
  to prefer admitting high-priority customers, and what the average reward per
  step approaches.

**Hook.** One serving slot, two queues, different payoffs. The differential target rewards whatever beats the running average — so the premium customers should win.

**Formal bridge.** The average reward per step under a prioritising policy is

\[
r(\pi)\approx 0.5\times 8+0.5\times 1=4.5,
\tag{10.4.1}
\]

where the first term serves (almost all) high-priority arrivals at 8 each and the second serves lows with leftover capacity at 1 each, and a naive "serve anyone" policy earns around 4.2 per step. The learning signal is

\[
\delta=R-\bar R+\hat q_{A'}(S')-\hat q_A(S).
\tag{10.4.2}
\]

where \(R\) is the recorded reward (8 for high priority, 1 for low), \(\bar R\) the running average, \(\hat q\) the action value, and \(\delta\) the differential TD error.

**Reasoning chain.** High-priority customers pay 8 and low-priority customers pay 1. If the two queues compete for one serving slot per step, the optimal policy admits a high-priority customer whenever one is waiting, and falls back to low-priority otherwise. The resulting average reward per step consists of serving almost all high-priority arrivals (\(\approx 0.5/\text{step}\times 8\)) plus serving lows with leftover capacity (up to \(0.5/\text{step}\times 1\)), giving an achievable ceiling around 4.5 per step (10.4.1), versus a naive "serve anyone" policy around 4.2.

Differential Sarsa discovers this without being told, because the differential target in (10.4.2) rewards contributors above the running \(\bar R\): admitting a high-priority customer produces a reward that beats the average and gets a positive differential signal, while admitting a low when a high is waiting fails to beat it. Over training, the running average climbs from roughly 3 toward 4.5 as the priority ordering is learned.

**Why it makes sense.** Like a bouncer at a two-line club: letting in the high-value customer whenever possible earns more per minute, because the slot is scarce and the premium is large. The direction is unambiguous — with one slot, serving a low while a high waits costs the difference of 7. Differential Sarsa learns that ordering by trying it and collecting the premium, not by being given a priority schedule.

**Check your understanding.** Prediction: raising the low-priority payment from 1 toward 8 would shrink the premium and flatten the learned preference, pulling the average reward ceiling toward the naive value; if the arrival rate of highs doubled, the ceiling would rise because more slots would be filled at 8.

**Where this shows up in the code.** `AccessControlQueue.step`, `run_access_control`, `DifferentialSarsaAgent`; run to see the moving-average reward improve from ~3 toward ~4.5. Figure: `figures/example_10_1_access_control.png`.

---

## Exercise 10.5 — Why Mountain Car is a hard exploration problem

- **Book location:** Section 10.1 (Example 10.1 background).
- **Type:** conceptual/environmental.
- **Problem (paraphrased):** Show that neither reversing, coasting, nor driving
  forward alone can solve Mountain Car if the policy is purely greedy from
  zero weights.

**Hook.** The obvious strategy — floor it — fails here. Seeing why reveals that the real problem is not credit assignment but finding the right behaviour at all.

**Formal bridge.** A greedy policy from zero weights selects

\[
A_t=\arg\max_a\hat q(S_t,a;w),
\qquad w=0\ \Rightarrow\ \text{ties broken toward action 0 (reverse)},
\tag{10.5.1}
\]

while the physics from the standard start gives a bounded forward speed

\[
v\le 0.07\quad\text{from position }-0.5\text{ with zero velocity},
\tag{10.5.2}
\]

which is insufficient to climb to the goal at \(+0.5\) under forward thrust alone.

where \(A_t\) is the chosen action, \(\hat q\) the action-value estimate, \(w\) the weights, \(v\) the car's velocity, and the hill's gradient at \(-0.5\) is strong enough that the car slides back.

**Reasoning chain.** From position \(-0.5\) with zero velocity, pushing forward gains at most \(v\le 0.07\), and the hill at position \(-0.5\) has a gradient strong enough that the car slides back; the goal at \(+0.5\) is unreachable by forward thrust alone. Also, greedy-from-zero-weights breaks ties toward action 0 (reverse) forever, per (10.5.1), so the agent never even experiments with a useful forward push timing.

The only way out is to first build reverse momentum — swing left — and then push forward at the right moment. That requires either (a) initial exploration with \(\epsilon>0\), or (b) optimistic initialization, so that the greedy policy is nudged into unexplored regions. Hence the learning curve begins at the 200-step cap and must climb down as the swinging manoeuvre is discovered.

**Why it makes sense.** The car cannot win by flooring it uphill; it must rock back and forth to build momentum, like pushing a playground swing. A brain that has only ever tried "go forward" never discovers this — and the zero-weight tie-break in (10.5.1) makes that trap especially sticky. That is why both humans and algorithms must explore the valley first, and why the initial performance sits at the step cap rather than near the goal.

**Check your understanding.** Prediction: setting \(\epsilon=0\) with zero-initialised weights would leave the agent stuck reversing forever at the 200-step cap; raising \(\epsilon\) would make early episodes more erratic but give it a chance to stumble onto the momentum strategy.

**Where this shows up in the code.** `run_mountain_car` uses `epsilon=0.1` exploration for this reason; unit test `test_returns_episode_lengths` uses `epsilon=0.5` to keep learning active in short runs. See also `example_10_2_mountain_car.png`.

---

## How to run the chapter tests

```bash
python -m pytest chapter_10/tests/ -v
```
