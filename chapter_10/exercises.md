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

**Solution.** In the episodic setting every episode ends in a terminal state
\(S_{term}\) with \(v(S_{term}) \equiv 0\). At the final step
\(S \to S_{term}\), the bootstrap term \(\gamma \hat v(S_{term})\) or \(\gamma \hat q(S_{term}, \cdot)\)
is exactly 0, because the value of a terminal state is 0 and the Q-values for
terminal (no-following-action) are defined as 0. Hence the last update in an
episode is \(\delta = R - \hat q_A(S)\). Any implementation that bootstraps from
\(\gamma \hat q(S')\) without special-casing terminals would inflate bootstrap
values for states that lead to termination.

- **Coding pointer.** `EpisodicSemiGradientSarsaAgent.update_sarsa` takes
  `next_x = None` for terminal transitions and uses `q_next = 0.0`; the Mountain
  Car loop in `run_mountain_car` passes `None` when `done`. Unit tests assert the
  exact one-step arithmetic on a terminal transition (`TestEpisodicSemiGradientSarsa`).

**Explanation (student-friendly).** A terminal state is "the finish line with no
given credit to keep." When you are one step from the finish, the best you can
say about the future is 0, so the entire target collapses to the last reward.
That is why episodes, unlike continuing tasks, do not need a bootstrap discount.

---

## Exercise 10.2 — Why divide the Mountain Car step size by the number of tilings?

- **Book location:** Section 10.1 / 9.4.3 (tile coding).
- **Type:** conceptual/design.
- **Problem (paraphrased):** In tile coding each state activates exactly one tile
  per tiling (8 tilings => 8 active features). Show that the weights move by the
  equivalent of one feature being stepped by \(8\times\) the per-feature step size,
  and derive the compensating factor.

**Solution.** With 8 tilings, \(\phi(s)\) has exactly 8 non-zero entries (one per
tiling). The semi-gradient update is \(w \leftarrow w + \alpha \delta \phi(s)\);
each active feature gets \(\alpha \delta\), so all 8 get updated equally. If we
want the *total* movement in the prediction (\(\Delta \hat v \approx \alpha
\|\phi(s)\|^2 \delta = \alpha \cdot 8 \cdot \delta\)) to be the same as a single
active feature's intended step \(\alpha' \delta\), we need the per-feature step
\(\alpha = \alpha'/8\), i.e. \(\alpha' = 8\alpha\). Equivalently the code updates
each active weight by \((\alpha/8)\delta\).

**Explanation (student-friendly).** Tile coding spreads one observation over 8
features, like 8 witnesses each recording a fraction of the same event. If you
charged full price at each witness, the total would be 8×; divide by 8 so the
whole system hears the event exactly once.

- **Coding pointer.** `EpisodicSemiGradientSarsaAgent.update_sarsa` uses
  `(self.alpha / 8.0) * delta * x`.

---

## Exercise 10.3 — Average-reward differential Sarsa on a continuing task

- **Book location:** Sections 10.2/10.3 (average-reward setting).
- **Type:** conceptual/derivation.
- **Problem (paraphrased):** Show why the discounted setting is equivalent, in the
  limit and under a uniform policy, to the average-reward setting, and why the
  differential update removes the discounting bias.

**Solution.** In a continuing ergodic task, the total discounted return from any
fixed start point is unbounded, so the fundamental value is the average reward
\(r(\pi) = \lim_{h\to\infty} \mathbb E[\frac{1}{h}\sum_t R_t]\). Optimality is
judged against every other policy's average, not against discounted return; a
selfish "front-load" bias toward early rewards is inappropriate when there is no
horizon. The discounted Bellman equation \(v = \sum_a \pi(a|s)[r + \gamma
\sum_{s'} P(s'|s,a) v(s')]\) in the \(\gamma \to 1\) limit is undefined; the
differential (average-reward) Bellman equation
\(v(s) - r(\pi) + r + \ldots\) (with a differential target \(R - \bar R\)) is well-defined.
The differential Sarsa update
\(\delta = R - \bar R + \hat q_{A'}(S') - \hat q_A(S)\) and
\(\bar R \leftarrow \bar R + \alpha_{\bar R} \delta\)
tracks both \(q\) and the average reward \(\bar R\) by stochastic approximation of
the differential Bellman equation.

**Explanation (student-friendly).** In an endless game, "maximize the total
reward" is infinite with no meaning; the fair question is "maximize reward *per
step*, forever." Differential Sarsa does this by living entirely on the average:
each step is judged against the running average, not against a discounted future.

- **Coding pointer.** `DifferentialSarsaAgent.update` implements the differential
  update with the running average `r_bar`; see `TestDifferentialSarsa`.

---

## Exercise 10.4 — Access-Control Queuing: why prioritize high-priority customers?

- **Book location:** Example 10.2 (Access-control queuing).
- **Type:** analysis of the implemented experiment.
- **Problem (paraphrased):** Explain why a differential Sarsa agent should learn
  to prefer admitting high-priority customers, and what the average reward per
  step approaches.

**Solution.** High-priority customers pay 8 and low-priority customers pay 1. If
the two queues compete for one serving slot per step, the optimal policy admits a
high-priority customer whenever one is waiting, and falls back to low-priority
otherwise. The average reward per step then consists of serving (almost all)
high-priority arrivals (\(\approx 0.5/step \times 8\)) plus serving lows with leftover
capacity (up to \(0.5/step \times 1\)), giving an achievable ceiling around 4.5
per step (versus a naive "serve anyone" policy around 4.2). Differential Sarsa
discovers this without being told, because the differential target rewards
contributors above the running r-bar.

**Explanation (student-friendly).** Like a bouncer at a two-line club: letting in
the high-value customer whenever possible earns more per minute. Differential
Sarsa learns that priority ordering by trying it and collecting the premium.

- **Coding pointer.** `AccessControlQueue.step`, `run_access_control`,
  `DifferentialSarsaAgent`; run to see the moving-average reward improve from ~3
  toward ~4.5. Figure: `figures/fig_10_access_control.png`.

---

## Exercise 10.5 — Why Mountain Car is a hard exploration problem

- **Book location:** Section 10.1 (Example 10.1 background).
- **Type:** conceptual/environmental.
- **Problem (paraphrased):** Show that neither reversing, coasting, nor driving
  forward alone can solve Mountain Car if the policy is purely greedy from
  zero weights.

**Solution.** From position -0.5 with zero velocity, pushing forward gains at
most \(v \leq 0.07\), and the hill at position \(-0.5\) has gradient strong enough
that the car slides back; the goal at +0.5 is unreachable by forward thrust alone.
Also, greedy-from-zero-weights breaks ties toward action 0 (reverse) forever.
The only way out is to first build reverse momentum (swing left), then push
forward at the right moment; that requires (a) initial exploration (epsilon > 0) or
(b) optimistic initialization, so the greedy policy is nudged into unexplored
regions. Hence the learning curve begins at the 200-step cap and must climb down.

**Explanation (student-friendly).** The car cannot win by flooring it uphill; it must
rock back and forth to build momentum. A brain that has only ever tried "go
forward" never discovers this, which is why everyone (humans and algorithms) must
explore the valley first.

- **Coding pointer.** `run_mountain_car` uses `epsilon=0.1` exploration for this
  reason; unit test `test_returns_episode_lengths` uses `epsilon=0.5` to keep
  learning active in short runs. See also `fig_10_mountain_car.png`.

---

## How to run the chapter tests

```bash
python -m pytest chapter_10/tests/ -v
```
