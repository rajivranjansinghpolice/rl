# Chapter 9 — Planning and Learning with Tabular Methods: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, 2018; online complete draft, Nov 5 2017).
> Repo chapter: `chapter_09`
> Book chapter: **Chapter 8** — Planning and Learning with Tabular Methods

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_09` — Planning and Learning with Tabular Methods.
- **Book chapter:** Chapter 8 — Planning and Learning with Tabular Methods.
- **Book sections relevant to these exercises and to the implementation:**
  - §8.1 — Models and planning (distribution vs sample models; planning from a model).
  - §8.2 — Dyna: integrated planning, acting, and learning; **Tabular Dyna-Q** pseudocode (direct RL, model-learning, and n planning updates per real step).
  - §8.3 — When the model is wrong: blocking and shortcut maze experiments, and **Dyna-Q+** with an exploration bonus (see Eq. 8.12, p. 166 in the 2nd ed.).
  - §8.4 — Prioritized sweeping (backward focusing via a priority queue; stochastic extension uses expected updates).
  - §8.5 — Expected vs sample updates (expected update target, Eq. 8.1, p. 168; comparison of computation and accuracy).
- **Plausible citations (2nd ed.):** Chapter 8 spans roughly pp. 159–177; Example 8.1 / Dyna maze and Fig. 8.2 around §8.2 (pp. 161–163); blocking/shortcut maze, Figs. 8.4–8.5 and Dyna-Q+ discussion around §8.3 (pp. 164–166, Eq. 8.12); prioritized sweeping §8.4 (pp. 166–168); expected vs sample updates §8.5 (pp. 168–170, Eq. 8.1). Exercises 8.1–8.4 appear in §8.2–§8.3 (roughly pp. 162–166).

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2018), _Reinforcement Learning: An Introduction_, 2nd ed., Chapter 8. Exercise statements below are **paraphrased** (no long verbatim quotes); short phrases appear only with section/equation/page citations.
- **Cross-check (sanity only, not copied):** public repos `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and `vojtamolda/reinforcement-learning-an-introduction` (Ch. 8), plus common solution PDFs. Disagreements resolved by independent derivation.
- All solutions are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type**, **Problem (paraphrased)**, **Solution**, **Derivation/experiment description** (as relevant), and a **Coding pointer** that matches actual APIs/tests in `chapter_09/`.
- **Implementation note:** **Dyna-Q+ (exploration bonus) is not implemented** in this repo. Exercises that depend on Dyna-Q+ are answered as **theory-only / experiment descriptions** and mapped to the closest implemented planning/model APIs (see coding pointers and the mapping table).

---

## Exercises

### Exercise 8.1 — Can multi-step bootstrapping match Dyna in the Dyna maze?

- **Book location:** §8.2, Exercise 8.1, Dyna maze / Fig. 8.2 (2nd ed., ~p. 162).
- **Type:** Theory.
- **Problem (paraphrased):** In the Dyna maze example, the non-planning agent is weak in part because it uses only one-step bootstrapping. A multi-step bootstrapping method (Chapter 7) would propagate rewards farther per real transition. Could such a multi-step method, using only real experience, perform as well as Dyna (with planning) on this task? Explain.
- **Solution.** In general, **no** — a pure multi-step method using only real experience cannot fully match Dyna here.
  - **What multi-step methods can do:** n-step TD / n-step Sarsa (Ch. 7, §§7.1–7.2) propagate return information back n real steps along the actually experienced trajectory, so after a single successful episode they can update values along that path much faster than one-step TD. With large n (or Monte Carlo-like returns) they can credit the whole trajectory from real data.
  - **What Dyna can do that real-experience multi-step cannot:** Dyna-Q (§8.2, Tabular Dyna-Q) learns a model from real transitions and then performs **n planning updates per real step** by sampling previously observed (s,a) pairs and applying Q-learning backups to *simulated* transitions. This lets value information spread to many state–action pairs (including those not on the current trajectory) without taking additional environment steps. Even with the same number of *real* steps, Dyna can perform far more total backups by reusing the model.
  - **Caveat:** If a multi-step method were given an enormous amount of real experience, or if planning updates were severely limited, the gap could shrink; but the key advantage in Fig. 8.2 is *extra computation on a learned model per real step*, not merely multi-step credit assignment on real data.
- **Why this works / takeaway:** Multi-step bootstrapping improves credit assignment over real trajectories; Dyna improves **sample efficiency** by generating simulated experience from a model, so planning adds value propagation that real-only methods cannot create.
- **Coding pointer (supporting machinery):** `dyna_q(env, n_planning=..., ...)` in `chapter_09/solution.py` implements Tabular Dyna-Q with direct Q-learning, model learning via `TabularModel.observe`, and `n_planning` sample-model planning updates per real step (tests: `TestDynaQ`, e.g. `test_planning_accelerates_learning`, `test_learns_short_path`). `make_gridworld(...)` builds the small gridworld MDP used by tests. No multi-step (Ch. 7) agent is implemented in this chapter.

### Exercise 8.2 — Why did Dyna-Q+ outperform Dyna-Q in both phases?

- **Book location:** §8.3, Exercise 8.2, blocking and shortcut maze experiments / Figs. 8.4–8.5 (2nd ed., ~pp. 164–165).
- **Type:** Theory.
- **Problem (paraphrased):** In the blocking and shortcut experiments, the Dyna agent augmented with an exploration bonus (Dyna-Q+) performs better than Dyna-Q in *both* the first phase and the second phase (after the environment changes). Why is Dyna-Q+ better in both phases?
- **Solution.** Dyna-Q+ (§8.3, Eq. 8.12, ~p. 166) adds a bonus to the *planning* update for a simulated transition: instead of using the recorded reward r, it uses
  \[
  r + \kappa \sqrt{\tau(s,a)},
  \]
  where \(\tau(s,a)\) is the (real-time) number of steps since pair (s,a) was last tried in the real environment and \(\kappa>0\) is small. This bonus treats long-untried pairs as potentially better than the stale model predicts, encouraging their simulated/selected visitation.
  - **First phase (before the change, stable environment):** Early on, many (s,a) pairs are rarely or never tried, so their \(\tau\) values grow and the bonus drives broader exploration. This helps the agent discover good paths and refine the model more thoroughly than plain Dyna-Q, which can become complacent once it has a workable (but not necessarily best-known) policy.
  - **Second phase (after the environment changes):** The model becomes wrong (a passage closes in the blocking task; a shortcut opens in the shortcut task). Dyna-Q tends to keep planning from its stale model and may be slow to re-try actions whose recorded outcomes changed. Dyna-Q+’s growing \(\tau\) for untested/stale pairs produces pressure to revisit them, so it detects the blocked path sooner and, crucially in the shortcut task, eventually tries actions that reveal the newly opened shorter route.
  - **Net effect:** The same mechanism — intrinsic reward for “staleness” — yields useful extra exploration both initially and after non-stationary changes.
- **Derivation note:** The bonus is added to the reward used in planning backups (Eq. 8.12), not to the true environment reward; it biases simulated targets to favor exploration of neglected transitions.
- **Coding pointer (theory-only for Dyna-Q+):** **`chapter_09/solution.py` does not implement Dyna-Q+, `κ`/`τ` bookkeeping, or the blocking/shortcut experiments.** The closest implemented APIs are `dyna_q` (sample-model Dyna-Q), `expected_dyna_q` / `expected_sweep` (expected updates, Eq. 8.1, §8.5), `TabularModel` (distribution/sample model, counts, `distribution`/`sample`/`predecessors`), and `make_gridworld(rows, cols, ..., slip=..., gamma=...)`. Tests covering these APIs: `TestDynaQ`, `TestExpectedVsSampleDyna`, `TestTabularModel`, `TestMakeGridworld`, `TestStochasticWorldModel`. A faithful Dyna-Q+ experiment would extend `dyna_q`/the model to track last-visit times and apply the Eq. 8.12 bonus during planning (and/or selection per Ex. 8.4).

### Exercise 8.3 — Why did the Dyna-Q+ vs Dyna-Q gap narrow early on?

- **Book location:** §8.3, Exercise 8.3, shortcut experiment / Fig. 8.5 (2nd ed., ~pp. 165–166).
- **Type:** Theory.
- **Problem (paraphrased):** In the shortcut experiment (Fig. 8.5), the performance advantage of Dyna-Q+ over Dyna-Q becomes slightly smaller during the first part of the experiment (before the shortcut opens). What explains this narrowing?
- **Solution.** In the first (stationary) phase the environment does not change and the agent gradually learns an accurate model and good values for the commonly used routes.
  - **Initially:** Dyna-Q+ explores more because many pairs have recently grown \(\tau\), so the bonus \(\kappa\sqrt{\tau}\) (Eq. 8.12) can make exploratory actions/planning targets attractive; this extra exploration can find/improve behavior faster than Dyna-Q, creating an early gap.
  - **As learning stabilizes:** The policy and visitation pattern settle. Frequently relevant (s,a) pairs are revisited often, keeping their \(\tau\) small, while the agent has less to gain from wandering; meanwhile Dyna-Q’s estimates also converge toward good behavior on the stable task. Consequently the *marginal* benefit of Dyna-Q+’s bonus-driven exploration diminishes and the cumulative-reward curves move closer together.
  - **Important contrast:** The gap widens again only later, when the shortcut opens and Dyna-Q+’s continued sensitivity to staleness helps it discover the change sooner than Dyna-Q.
- **Why this works / takeaway:** Exploration bonuses trade off short-run exploitation for information; in a stationary phase their advantage can shrink once both agents have learned good behavior, but they remain valuable when non-stationarity appears.
- **Coding pointer (theory-only for Dyna-Q+ / Fig. 8.5):** **No Dyna-Q+ or shortcut-experiment harness is implemented.** Related implemented comparisons are *sample vs expected* planning (a different §8.5 question), not Dyna-Q vs Dyna-Q+: `compare_sample_vs_expected_dyna(env_factory, n_planning=..., n_episodes=..., alpha=..., epsilon=..., gamma=..., seed=..., start_state=...)` returns per-variant `max_q_error`, `known_max_q_error`, `episode_steps`, `episode_rewards`, `q_table`, and `q_star_max` (tests: `TestExpectedVsSampleDyna`). Gridworld construction for experiments: `make_gridworld` (`TestMakeGridworld`).

### Exercise 8.4 (Programming) — Exploration bonus in action selection only

- **Book location:** §8.3, Exercise 8.4 (programming), blocking/shortcut context (2nd ed., ~p. 166).
- **Type:** Programming / Theory.
- **Problem (paraphrased):** The exploration bonus in Dyna-Q+ modifies the values used in *updates* (planning targets). Is that modification necessary? Consider an alternative: keep all Q-learning/planning updates unchanged (use only the recorded/model reward), but when choosing actions behave greedily with respect to
  \[
  Q(S_t,a) + \kappa \sqrt{\tau(S_t,a)},
  \]
  (i.e., add the bonus only for action selection). Design and carry out a gridworld experiment (in the spirit of the blocking/shortcut tasks) to test and illustrate the strengths and weaknesses of this selection-only bonus.
- **Solution (expected behavior and experiment design).**
  - **Is changing update values necessary?** Not strictly necessary to obtain exploratory behavior: adding the bonus only at selection time also encourages trying actions with large \(\tau\), so the agent can still be driven to revisit stale parts of the environment.
  - **Strengths of selection-only bonus:** (1) **Unbiased value estimates.** Because backups/planning targets are unchanged, \(Q\) continues to estimate the return under the true (recorded/model) rewards rather than an optimism-inflated target. (2) **Cleaner separation of concerns.** Exploration pressure is transient (at decision time) and does not permanently distort stored action values.
  - **Weaknesses / trade-offs:** (1) **No bonus-propagated exploration through planning.** In Dyna-Q+, optimistic planning targets can make distant predecessors look promising in simulated backups; selection-only optimism affects only the current choice and may propagate information more slowly. (2) **Policy instability / dithering.** The effective preference can flip as \(\tau\) changes immediately after a visit, so behavior may oscillate unless \(\kappa\) and the selection rule are tuned. (3) **No strict greedy-optimality guarantee w.r.t. \(Q\) alone** during exploration: the executed action maximizes the augmented score, not \(Q\) itself.
  - **Suggested experiment (blocking + shortcut, mirroring §8.3):**
    1. Build a deterministic gridworld/maze with a start and goal (e.g. `make_gridworld(rows, cols, start=..., goal=..., slip=0.0, gamma=...)`).
    2. **Phase 1 (learn):** run Dyna-Q (`dyna_q(..., n_planning=k, n_episodes=..., epsilon=..., seed=..., start_state=...)`) until a short path is learned (compare episode lengths/rewards).
    3. **Phase 2 (change):** emulate blocking (remove/penalize a corridor action by changing the environment/model) or shortcut (add a shorter passage). Continue training under three variants: (a) plain Dyna-Q, (b) Dyna-Q+ (bonus in planning targets, Eq. 8.12) — *not implemented here*, (c) selection-only bonus \(Q+\kappa\sqrt{\tau}\) with unchanged updates — *not implemented here*.
    4. **Metrics:** cumulative reward or steps-to-goal per episode over time (as in Figs. 8.4–8.5), speed of recovery after blocking, and time to discover the shortcut.
    5. **Report:** selection-only should track true \(Q\) more faithfully but may react slower/faster than (b) depending on whether planning propagation or immediate choice pressure dominates; include fixed seeds for reproducibility.
- **Why this works / takeaway:** Optimism can be injected at *decision time* or at *learning/planning time*; the former preserves value accuracy, the latter can spread exploratory incentives through backups — with different dynamics on non-stationary tasks.
- **Coding pointer (partially supported):** **Neither Dyna-Q+ nor selection-only bonus is implemented.** Implemented building blocks for such an experiment: `make_gridworld(rows, cols, start="0,0", goal=None, slip=0.0, gamma=0.95, seed=None)` (`TestMakeGridworld`), `dyna_q` / `expected_dyna_q` (signatures as in `solution.py`), `TabularModel` (`observe`, `is_known`, `known_pairs`, `visited_states`, `actions_taken`, `sample_known_pair`, `distribution`, `sample`, `observation_count`, `predecessors`), and comparison helpers `expected_sweep`, `optimal_q_values`, `compare_sample_vs_expected_dyna`. Relevant tests: `TestMakeGridworld`, `TestTabularModel`, `TestDynaQ`, `TestExpectedVsSampleDyna`, `TestStochasticWorldModel`. To run this programming exercise, add \(\tau\)-tracking and an augmented \(\arg\max_a [Q(s,a)+\kappa\sqrt{\tau(s,a)}]\) policy on top of these APIs.

---

## Coding Exercises Mapped to Repository Code

| Book exercise | Book section / topic | Support in repo | Implemented? | Repo tests | Pytest command |
|---|---|---|---|---|---|
| 8.1 | §8.2, Dyna maze: multi-step vs Dyna planning | `dyna_q`, `TabularModel`, `make_gridworld` illustrate model-based planning; no Ch. 7 multi-step baseline in this chapter | Partial (Dyna/planning only) | `TestDynaQ`, `TestTabularModel`, `TestMakeGridworld` | `python -m pytest chapter_09/tests/ -v` |
| 8.2 | §8.3, Dyna-Q+ (Eq. 8.12) on blocking/shortcut | Planning/model APIs available: `dyna_q`, `expected_dyna_q`, `expected_sweep`, `TabularModel`, `make_gridworld`; **no `κ`, `τ`, or env-change experiment** | **No (theory-only for Dyna-Q+)** | `TestDynaQ`, `TestExpectedVsSampleDyna`, `TestTabularModel`, `TestMakeGridworld`, `TestStochasticWorldModel` | `python -m pytest chapter_09/tests/ -v` |
| 8.3 | §8.3, narrowing gap in first phase (Fig. 8.5) | `compare_sample_vs_expected_dyna` is a different comparison (§8.5), but shows experiment-harness pattern; gridworld via `make_gridworld` | **No (theory-only for Dyna-Q+ dynamics)** | `TestExpectedVsSampleDyna`, `TestMakeGridworld` | `python -m pytest chapter_09/tests/ -v` |
| 8.4 | §8.3 programming, bonus in selection only | Base environment/agents/models: `make_gridworld`, `dyna_q`, `expected_dyna_q`, `TabularModel`, `optimal_q_values`, `expected_sweep`; **no \(\tau\)/bonus policy** | **No (experiment not implemented)** | `TestMakeGridworld`, `TestDynaQ`, `TestTabularModel`, `TestExpectedVsSampleDyna`, `TestStochasticWorldModel` | `python -m pytest chapter_09/tests/ -v` |

**Additional implemented (not asked by 8.1–8.4 but present and tested):** `PriorityQueue`, `prioritized_sweeping` (§8.4, Exercise 9.1 in-repo) — `TestPriorityQueue`, `TestPrioritizedSweeping`. `compare_sample_vs_expected_dyna` and expected/sample variants (§8.5, Exercise 9.2 in-repo) — `TestExpectedVsSampleDyna`. Stochastic distribution model behavior — `TestTabularModel`, `TestStochasticWorldModel`.

- Full chapter command: `python -m pytest chapter_09/tests/ -v`
- Lint: `ruff check chapter_09/`
- No standalone `exercise_*.py` scripts exist in `chapter_09/`.

---

## Verification and Quality Notes

- Reasoning and derivations were produced independently; public materials were used only for cross-checking.
- Exercises 8.2–8.4 involve Dyna-Q+ / exploration-bonus experiments that are **not implemented** in this repository; answers therefore provide theory, derivations (Eq. 8.12 / \(\tau\)-bonus), and experiment designs, with pointers to the closest implemented APIs/tests rather than claiming unsupported functionality.
- Exercise statements are paraphrased; citations are given to plausible 2nd-edition sections/pages/equations/figures for traceability.

---

*Exercises document for Chapter 9 (book Chapter 8) — part of the per-chapter exercises-and-solutions series (`chapter_XX/exercises.md`).*
