# Chapter 8 — Planning and Learning with Tabular Methods: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, 2018).
> Repo chapter: `chapter_08`
> Book chapter: **Chapter 8** — Planning and Learning with Tabular Methods

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_08` — Planning and Learning with Tabular Methods.
- **Book chapter:** Chapter 8 — Planning and Learning with Tabular Methods.
- **Book sections relevant to these exercises and to the implementation:**
  - §8.1 — Models and planning (distribution vs sample models; planning from a model).
  - §8.2 — Dyna: integrated planning, acting, and learning; **Tabular Dyna-Q** pseudocode (direct RL, model-learning, and n planning updates per real step).
  - §8.3 — When the model is wrong: blocking and shortcut maze experiments, and **Dyna-Q+** with an exploration bonus (see Eq. 8.12 in §8.3, 2nd ed.).
  - §8.4 — Prioritized sweeping (backward focusing via a priority queue; stochastic extension uses expected updates).
  - §8.5 — Expected vs sample updates (expected update target, Eq. 8.1, §8.5; comparison of computation and accuracy).
- **Verified citations (2nd ed.):** Chapter 8 spans pp. 131–157. Example 8.1 / Dyna maze and Fig. 8.3 in §8.2 (p. 136); policies in Fig. 8.4; blocking/shortcut mazes in Figs. 8.5–8.6 and Dyna-Q+ in §8.3 (pp. 137–138); prioritized sweeping and Examples 8.4–8.5 in §8.4 (pp. 139–141); expected vs sample updates in Fig. 8.8 (p. 144); trajectory sampling in Fig. 8.9 (p. 146); RTDP / Example 8.6 in §8.7 (pp. 146–148). Exercises 8.1–8.4 appear at p. 139.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2018), _Reinforcement Learning: An Introduction_, 2nd ed., Chapter 8. Exercise statements below are **paraphrased** (no long verbatim quotes); short phrases appear only with section/equation/page citations.
- **Cross-check (sanity only, not copied):** public repos `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and `vojtamolda/reinforcement-learning-an-introduction` (Ch. 8), plus common solution PDFs. Disagreements resolved by independent derivation.
- All solutions are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type**, **Problem (paraphrased)**, and a worked answer in the five-part house style (**Hook**, **Formal bridge**, **Reasoning chain**, **Why it makes sense**, **Check your understanding**), plus a **Where this shows up in the code** pointer that matches actual APIs/tests in `chapter_08/`.
- **Implementation note:** Dyna-Q+ (`dyna_q_plus`), the changing-maze harness (`ChangingGridworld`, `changing_maze_experiment`), the selection-only bonus variant (Ex. 8.4), the rod reconstruction (Ex. 8.5, `chapter_08/rod.py`) and RTDP (Ex. 8.6, `chapter_08/rtdp.py`) **are implemented**. Coding pointers below name the exact APIs and tests.

---

## Exercises

### Exercise 8.1 — Can multi-step bootstrapping match Dyna in the Dyna maze?

- **Book location:** §8.2, Exercise 8.1, Dyna maze / Fig. 8.3 (2nd ed., p. 136).
- **Type:** Theory.
- **Problem (paraphrased):** In the Dyna maze example, the non-planning agent is weak in part because it uses only one-step bootstrapping. A multi-step bootstrapping method (Chapter 7) would propagate rewards farther per real transition. Could such a multi-step method, using only real experience, perform as well as Dyna (with planning) on this task? Explain.

**Hook.** Let's separate two things that both speed up learning: reaching farther along one real trajectory, and generating extra simulated trajectories. They are not the same resource.

**Formal bridge.** The n-step backup uses only the experienced path,

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Bigl[\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nQ(S_{t+n},A_{t+n})-Q(S_t,A_t)\Bigr],
\tag{8.1.1}
\]

whereas after storing a transition in the model, Dyna performs \(n_{\text{plan}}\) additional backups on sampled past pairs:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr],
\quad (s,a,r,s')\sim\text{model}.
\tag{8.1.2}
\]

Here \(Q\) is the action-value estimate, \(\alpha\) the step size, \(\gamma\) the discount, \(n\) the multi-step horizon, \(r\) the recorded reward, \(s'\) the successor state, and \(n_{\text{plan}}\) the number of planning updates per real step.

**Reasoning chain.** In general, no — a pure multi-step method using only real experience cannot fully match Dyna here. What n-step TD and n-step Sarsa (Chapter 7, §§7.1–7.2) can do is propagate return information back \(n\) real steps along the actually experienced trajectory, so after a single successful episode they can update values along that path much faster than one-step TD. With large \(n\), or Monte-Carlo-like returns, they can credit the whole trajectory from real data. But every one of those updates needs the corresponding real transition to have happened.

Dyna-Q (§8.2, Tabular Dyna-Q) learns a model from real transitions and then performs \(n_{\text{plan}}\) planning updates per real step by sampling previously observed \((s,a)\) pairs from the model and applying Q-learning backups (8.1.2) to simulated transitions. Value information can spread to many state–action pairs — including ones not on the current trajectory — without any additional environment steps. With the same number of *real* steps, Dyna therefore performs far more total backups by reusing the model.

The caveat matters: if the multi-step method were given an enormous amount of real experience, or if planning updates were severely limited, the gap could shrink. But the key advantage in Fig. 8.3 is the extra computation on a learned model per real step, not merely multi-step credit assignment on real data.

**Why it makes sense.** Imagine one successful run through a maze. A multi-step method can credit that one path richly, but states off the path stay untouched until the agent physically revisits them. Dyna replays the same transition many times in simulation, letting the news spread sideways through the state–action graph. Replaying a cheap model is like rehearsing a route at night: the real walk does not get longer, but more of the map gets updated.

**Check your understanding.** Prediction: if planning is turned off (\(n_{\text{plan}}=0\)) while \(n\) stays large, Dyna-Q collapses to one-step Q-learning and the multi-step agent wins on credit assignment; if real experience is unlimited and planning is capped, the advantage of (8.1.2) shrinks toward zero.

**Where this shows up in the code.** `dyna_q(env, n_planning=..., ...)` in `chapter_08/solution.py` implements Tabular Dyna-Q with direct Q-learning, model learning via `TabularModel.observe`, and `n_planning` sample-model planning updates per real step (tests: `TestDynaQ`, e.g. `test_planning_accelerates_learning`, `test_learns_short_path`). `make_gridworld(...)` builds the small gridworld MDP used by tests. No multi-step (Ch. 7) agent is implemented in this chapter.

### Exercise 8.2 — Why did Dyna-Q+ outperform Dyna-Q in both phases?

- **Book location:** §8.3, Exercise 8.2, blocking and shortcut maze experiments / Figs. 8.5–8.6 (2nd ed., pp. 137–138).
- **Type:** Theory.
- **Problem (paraphrased):** In the blocking and shortcut experiments, the Dyna agent augmented with an exploration bonus (Dyna-Q+) performs better than Dyna-Q in *both* the first phase and the second phase (after the environment changes). Why is Dyna-Q+ better in both phases?

**Hook.** The bonus is often introduced as a fix for changing environments, so its benefit in the *stable* first phase is the surprising part. One mechanism explains both.

**Formal bridge.** Dyna-Q+ (§8.3, Eq. 8.12, p. 138) replaces the recorded reward \(r\) in the planning backup by an optimism-adjusted reward

\[
r^{\text{plan}}=r+\kappa\sqrt{\tau(s,a)},
\tag{8.2.1}
\]

with staleness

\[
\tau(s,a)=\text{steps since }(s,a)\text{ was last tried in the real environment}.
\tag{8.2.2}
\]

where \(s\) is the state, \(a\) the action, \(r\) the recorded model reward, \(\tau(s,a)\) the real-time staleness of the pair, and \(\kappa>0\) a small bonus weight. The square root makes the bonus grow with neglect but sub-linearly.

**Reasoning chain.** The bonus in (8.2.1) is added to the reward used in *planning* backups, not to the true environment reward; it biases simulated targets toward exploring neglected transitions. In the first phase, before any change, many \((s,a)\) pairs are rarely or never tried, so their \(\tau\) values grow and the κ√(τ) term drives broader exploration. This helps the agent discover good paths and refine the model more thoroughly than plain Dyna-Q, which can become complacent once it has a workable — but not necessarily best — policy.

In the second phase, the model becomes wrong: a passage closes in the blocking task, a shortcut opens in the shortcut task. Dyna-Q keeps planning from its stale model and may be slow to re-try actions whose recorded outcomes changed. Dyna-Q+'s growing \(\tau\) for untested or stale pairs produces pressure to revisit them, so it detects the blocked path sooner and, crucially in the shortcut task, eventually tries the actions that reveal the newly opened shorter route.

The net effect is the same mechanism — intrinsic reward for staleness — yielding useful extra exploration both initially and after non-stationary changes.

**Why it makes sense.** Think of \(\tau\) as a timer that only resets when you actually check a corridor. In a stable maze, unchecked corridors may hide a better route, so checking them is worthwhile. In a changed maze, unchecked corridors may hide the new reality, so checking them is *essential*. The direction of the bonus is positive because staleness means uncertainty, and uncertainty is worth resolving; a negative bonus would make the agent cling forever to the corners it already knows.

**Check your understanding.** Prediction: once the task is stable and every frequently used pair is fresh, \(\tau\) stays small, the κ√(τ) bonus nearly vanishes, and the two agents' curves move together again; raising \(\kappa\) keeps them apart longer but adds noisy exploratory steps.

**Where this shows up in the code.** `dyna_q_plus(env, ..., kappa=1e-4)` applies `κ√τ` to modelled (stale) pairs during planning; `changing_maze_experiment('blocking'|'shortcut', 'dyna_q'|'dyna_q_plus'|'selection_bonus', change_step, ...)` runs the two-phase experiment; `plot_changing_maze(...)` reproduces Figs. 8.5–8.6. Tests: `TestDynaQPlus`, `TestChangingMaze`.

### Exercise 8.3 — Why did the Dyna-Q+ vs Dyna-Q gap narrow early on?

- **Book location:** §8.3, Exercise 8.3, shortcut experiment / Fig. 8.6 (2nd ed., p. 138).
- **Type:** Theory.
- **Problem (paraphrased):** In the shortcut experiment (Fig. 8.6), the performance advantage of Dyna-Q+ over Dyna-Q becomes slightly smaller during the first part of the experiment (before the shortcut opens). What explains this narrowing?

**Hook.** A shrinking gap before any environmental change looks like the bonus "wearing out." That is exactly the right reading, but it is a feature, not a failure.

**Formal bridge.** The marginal pull of the bonus comes from its growth with staleness:

\[
\frac{\partial}{\partial\tau}\bigl(\kappa\sqrt{\tau}\bigr)=\frac{\kappa}{2\sqrt{\tau}},
\tag{8.3.1}
\]

so the bonus is strongest for freshly neglected pairs and its *rate of increase* falls as \(\tau\) accumulates — while frequently revisited pairs keep \(\tau\) small:

\[
\tau(s,a)=0\ \text{immediately after }(s,a)\ \text{is tried}.
\tag{8.3.2}
\]

where \(\tau(s,a)\) is the real-time staleness of pair \((s,a)\), \(\kappa>0\) is the bonus weight, and the derivative in (8.3.1) describes how much extra exploration pressure one more step of neglect buys.

**Reasoning chain.** In the first, stationary phase the environment does not change and the agent gradually learns an accurate model and good values for the commonly used routes. Initially, many pairs have recently grown \(\tau\), so the bonus \(\kappa\sqrt{\tau}\) of Eq. 8.12 makes exploratory actions and planning targets attractive; this extra exploration can find or improve behaviour faster than Dyna-Q, creating the early gap.

As learning stabilizes, the policy and visitation pattern settle. Frequently relevant \((s,a)\) pairs are revisited often, keeping their \(\tau\) small; at the same time Dyna-Q's own estimates converge toward good behaviour on the stable task. Consequently the *marginal* benefit of Dyna-Q+'s bonus-driven exploration — the derivative in (8.3.1) applied to pairs that are no longer neglected — diminishes, and the cumulative-reward curves move closer together.

The important contrast is that the gap widens again only later, when the shortcut opens. Then previously ignored transitions suddenly become informative, their \(\tau\) keeps growing, and Dyna-Q+'s continued sensitivity to staleness discovers the change sooner than Dyna-Q.

**Why it makes sense.** Exploration bonuses trade short-run exploitation for information. In a stationary phase there is less information left to buy, so both agents converge to similar behaviour and the bonus's advantage shrinks. When the world changes, the value of information jumps back up. If the gap instead kept widening in a stable maze, that would mean the agent was exploring for its own sake rather than because uncertainty demanded it.

**Check your understanding.** Prediction: delaying the shortcut change to a much later step would let the curves converge more closely before they separate; making the change earlier would catch the gap while it is still open, so Dyna-Q+'s recovery advantage would appear sooner.

**Where this shows up in the code.** `changing_maze_experiment('shortcut', agent, change_step=3000, ...)` with `agent='dyna_q'` or `'dyna_q_plus'`; see `TestChangingMaze`. In our reconstructed maze the early-phase curves differ from the book (documented in `README.md`); the post-change recovery pattern is reproduced.

### Exercise 8.4 (Programming) — Exploration bonus in action selection only

- **Book location:** §8.3, Exercise 8.4 (programming), blocking/shortcut context (2nd ed., ~p. 166).
- **Type:** Programming / Theory.
- **Problem (paraphrased):** The exploration bonus in Dyna-Q+ modifies the values used in *updates* (planning targets). Is that modification necessary? Consider an alternative: keep all Q-learning/planning updates unchanged (use only the recorded/model reward), but when choosing actions behave greedily with respect to \(Q(S_t,a) + \kappa \sqrt{\tau(S_t,a)}\) (i.e., add the bonus only for action selection). Design and carry out a gridworld experiment (in the spirit of the blocking/shortcut tasks) to test and illustrate the strengths and weaknesses of this selection-only bonus.

**Hook.** Optimism can be injected where learning happens or where deciding happens. Those two choices feel similar but lead to different value estimates and different exploration dynamics.

**Formal bridge.** The selection-only rule scores actions by

\[
a_t=\arg\max_a\bigl[Q(S_t,a)+\kappa\sqrt{\tau(S_t,a)}\bigr],
\tag{8.4.1}
\]

while the backup stays unbiased,

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[r+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\bigr],
\tag{8.4.2}
\]

where \(Q\) is the action value, \(\tau(S_t,a)\) the staleness of that state–action pair, \(\kappa>0\) the bonus weight, \(r\) the recorded or model reward, \(\alpha\) the step size, and \(\gamma\) the discount. Compare (8.4.1) with Dyna-Q+'s planning target \(r+\kappa\sqrt{\tau(s,a)}\).

**Reasoning chain.** The modification is not strictly necessary to obtain exploratory behaviour: adding the bonus only at selection time also encourages trying actions with large \(\tau\), so the agent can still be driven to revisit stale parts of the environment.

The strengths follow from leaving (8.4.2) untouched. First, value estimates stay unbiased: \(Q\) continues to estimate the return under the true recorded/model rewards rather than an optimism-inflated target. Second, there is a cleaner separation of concerns — exploration pressure is transient, at decision time, and does not permanently distort stored action values.

The weaknesses are the mirror image. Without optimistic planning targets, distant predecessors of a stale action do not look promising in simulated backups, so exploration can propagate information more slowly. The effective preference can also flip immediately after a visit, when \(\tau\) resets, causing dithering unless \(\kappa\) and the selection rule are tuned. And during exploration the executed action maximizes the augmented score rather than \(Q\) alone, so there is no strict greedy-optimality guarantee with respect to \(Q\).

A matching experiment mirrors §8.3. Build a deterministic gridworld/maze with a start and goal (`make_gridworld(rows, cols, start=..., goal=..., slip=0.0, gamma=...)`). In phase 1, run Dyna-Q (`dyna_q(..., n_planning=k, n_episodes=..., epsilon=..., seed=..., start_state=...)`) until a short path is learned, comparing episode lengths and rewards. In phase 2, emulate blocking by removing or penalising a corridor action, or a shortcut by adding a shorter passage, and continue training under three variants: plain Dyna-Q, `dyna_q_plus` with the bonus in planning targets (Eq. 8.12), and `selection_bonus` via `changing_maze_experiment(..., agent='selection_bonus')`. Track cumulative reward or steps-to-goal per episode, speed of recovery after blocking, and time to discover the shortcut. Selection-only should track true \(Q\) more faithfully, but may react slower or faster than Dyna-Q+ depending on whether planning propagation or immediate choice pressure dominates; fix seeds for reproducibility.

**Why it makes sense.** A bonus in the update is like raising the posted price of an unexplored road: every simulated traveller hears about it, so the incentive diffuses backward through the map. A bonus in selection is like a navigator who personally prefers unexplored roads but records the true travel time: the map stays honest, but other agents reading the map are not pulled toward the frontier. Neither is universally better — the first spreads exploration, the second protects accuracy.

**Check your understanding.** Prediction: raising \(\kappa\) in the selection-only rule makes the agent try stale actions more often and recover from a sudden shortcut sooner, but also increases dithering right after a visit because \(\tau\) resets to zero and the augmented score flips.

**Where this shows up in the code.** `changing_maze_experiment(..., agent='selection_bonus')` implements the augmented selection rule \(\arg\max_a[Q(s,a)+\kappa\sqrt{\tau(s,a)}]\) with unchanged updates; `dyna_q_plus` is the planning-time variant. `plot_changing_maze(..., agents=('dyna_q','dyna_q_plus','selection_bonus'))` plots all three. Tests: `TestDynaQPlus`, `TestChangingMaze`.

---

## Coding Exercises Mapped to Repository Code

| Book exercise | Book section / topic | Support in repo | Implemented? | Repo tests | Pytest command |
|---|---|---|---|---|---|
| 8.1 | §8.2, Dyna maze: multi-step vs Dyna planning | `dyna_q`, `TabularModel`, `make_gridworld` illustrate model-based planning; no Ch. 7 multi-step baseline in this chapter | Partial (Dyna/planning only) | `TestDynaQ`, `TestTabularModel`, `TestMakeGridworld` | `python -m pytest chapter_08/tests/ -v` |
| 8.2 | §8.3, Dyna-Q+ (Eq. 8.12) on blocking/shortcut | `dyna_q_plus`, `ChangingGridworld`, `changing_maze_experiment` | **Yes** | `TestDynaQPlus`, `TestChangingMaze` | `python -m pytest chapter_08/tests/ -v` |
| 8.3 | §8.3, narrowing gap in first phase (Fig. 8.6) | Same changing-maze harness; qualitative pattern only (see README) | **Partial (illustrative)** | `TestChangingMaze` | `python -m pytest chapter_08/tests/ -v` |
| 8.4 | §8.3 programming, bonus in selection only | `agent='selection_bonus'` in `changing_maze_experiment` | **Yes** | `TestChangingMaze` | `python -m pytest chapter_08/tests/ -v` |

**Additional implemented (not asked by 8.1–8.4 but present and tested):** `PriorityQueue`, `prioritized_sweeping` (§8.4) — `TestPriorityQueue`, `TestPrioritizedSweeping`; scaled-maze Example 8.4 and rod Example 8.5 (`chapter_08/rod.py`); RTDP Example 8.6 (`chapter_08/rtdp.py`); trajectory sampling (§8.6) and `compare_sample_vs_expected_dyna` (§8.5) — `TestTrajectorySampling`, `TestExpectedVsSampleDyna`; stochastic model tests `TestTabularModel`, `TestStochasticWorldModel`.

- Full chapter command: `python -m pytest chapter_08/tests/ -v`
- Lint: `ruff check chapter_08/`
- No standalone `exercise_*.py` scripts exist in `chapter_08/`.

---

## Verification and Quality Notes

- Reasoning and derivations were produced independently; public materials were used only for cross-checking.
- Exercises 8.2–8.4 are backed by runnable Dyna-Q+ / exploration-bonus experiments. Figure outputs are reduced-scale illustrations: exact maze coordinates and the book's κ tuning are not recoverable from the printed text, and the phase-1 ordering of our curves differs from the book (documented in `README.md`).
- Exercise statements are paraphrased; citations are given to plausible 2nd-edition sections/pages/equations/figures for traceability.

---

*Exercises document for Chapter 8 — part of the per-chapter exercises-and-solutions series (`chapter_XX/exercises.md`).*
