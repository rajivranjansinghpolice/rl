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

- Each exercise lists: **Book location**, **Type**, **Problem (paraphrased)**, **Solution**, **Derivation/experiment description** (as relevant), and a **Coding pointer** that matches actual APIs/tests in `chapter_08/`.
- **Implementation note:** Dyna-Q+ (`dyna_q_plus`), the changing-maze harness (`ChangingGridworld`, `changing_maze_experiment`), the selection-only bonus variant (Ex. 8.4), the rod reconstruction (Ex. 8.5, `chapter_08/rod.py`) and RTDP (Ex. 8.6, `chapter_08/rtdp.py`) **are implemented**. Coding pointers below name the exact APIs and tests.

---

## Exercises

### Exercise 8.1 — Can multi-step bootstrapping match Dyna in the Dyna maze?

- **Book location:** §8.2, Exercise 8.1, Dyna maze / Fig. 8.3 (2nd ed., p. 136).
- **Type:** Theory.
- **Problem (paraphrased):** In the Dyna maze example, the non-planning agent is weak in part because it uses only one-step bootstrapping. A multi-step bootstrapping method (Chapter 7) would propagate rewards farther per real transition. Could such a multi-step method, using only real experience, perform as well as Dyna (with planning) on this task? Explain.
- **Solution.** In general, **no** — a pure multi-step method using only real experience cannot fully match Dyna here.
  - **What multi-step methods can do:** n-step TD / n-step Sarsa (Ch. 7, §§7.1–7.2) propagate return information back n real steps along the actually experienced trajectory, so after a single successful episode they can update values along that path much faster than one-step TD. With large n (or Monte Carlo-like returns) they can credit the whole trajectory from real data.
  - **What Dyna can do that real-experience multi-step cannot:** Dyna-Q (§8.2, Tabular Dyna-Q) learns a model from real transitions and then performs **n planning updates per real step** by sampling previously observed (s,a) pairs and applying Q-learning backups to *simulated* transitions. This lets value information spread to many state–action pairs (including those not on the current trajectory) without taking additional environment steps. Even with the same number of *real* steps, Dyna can perform far more total backups by reusing the model.
  - **Caveat:** If a multi-step method were given an enormous amount of real experience, or if planning updates were severely limited, the gap could shrink; but the key advantage in Fig. 8.3 is *extra computation on a learned model per real step*, not merely multi-step credit assignment on real data.
- **Explanation (student-friendly).** Multi-step bootstrapping improves credit assignment over real trajectories; Dyna improves **sample efficiency** by generating simulated experience from a model, so planning adds value propagation that real-only methods cannot create.
- **Coding pointer (supporting machinery):** `dyna_q(env, n_planning=..., ...)` in `chapter_08/solution.py` implements Tabular Dyna-Q with direct Q-learning, model learning via `TabularModel.observe`, and `n_planning` sample-model planning updates per real step (tests: `TestDynaQ`, e.g. `test_planning_accelerates_learning`, `test_learns_short_path`). `make_gridworld(...)` builds the small gridworld MDP used by tests. No multi-step (Ch. 7) agent is implemented in this chapter.

### Exercise 8.2 — Why did Dyna-Q+ outperform Dyna-Q in both phases?

- **Book location:** §8.3, Exercise 8.2, blocking and shortcut maze experiments / Figs. 8.5–8.6 (2nd ed., pp. 137–138).
- **Type:** Theory.
- **Problem (paraphrased):** In the blocking and shortcut experiments, the Dyna agent augmented with an exploration bonus (Dyna-Q+) performs better than Dyna-Q in *both* the first phase and the second phase (after the environment changes). Why is Dyna-Q+ better in both phases?
- **Solution.** Dyna-Q+ (§8.3, Eq. 8.12, p. 138) adds a bonus to the *planning* update for a simulated transition: instead of using the recorded reward r, it uses
  \[
  r + \kappa \sqrt{\tau(s,a)},
  \]
  where \(\tau(s,a)\) is the (real-time) number of steps since pair (s,a) was last tried in the real environment and \(\kappa>0\) is small. This bonus treats long-untried pairs as potentially better than the stale model predicts, encouraging their simulated/selected visitation.
  - **First phase (before the change, stable environment):** Early on, many (s,a) pairs are rarely or never tried, so their \(\tau\) values grow and the bonus drives broader exploration. This helps the agent discover good paths and refine the model more thoroughly than plain Dyna-Q, which can become complacent once it has a workable (but not necessarily best-known) policy.
  - **Second phase (after the environment changes):** The model becomes wrong (a passage closes in the blocking task; a shortcut opens in the shortcut task). Dyna-Q tends to keep planning from its stale model and may be slow to re-try actions whose recorded outcomes changed. Dyna-Q+’s growing \(\tau\) for untested/stale pairs produces pressure to revisit them, so it detects the blocked path sooner and, crucially in the shortcut task, eventually tries actions that reveal the newly opened shorter route.
  - **Net effect:** The same mechanism — intrinsic reward for “staleness” — yields useful extra exploration both initially and after non-stationary changes.
- **Derivation note:** The bonus is added to the reward used in planning backups (Eq. 8.12), not to the true environment reward; it biases simulated targets to favor exploration of neglected transitions.
**Explanation (student-friendly).** The same staleness pressure that helps after a change also helps before one: early on many actions have never been tried, so the bonus pushes the agent to test them and build a better model. Once the task is stable the commonly used pairs are fresh, the bonus shrinks, and both agents settle down — which is why the gap can narrow again.

- **Coding pointer:** `dyna_q_plus(env, ..., kappa=1e-4)` applies `κ√τ` to modelled (stale) pairs during planning; `changing_maze_experiment('blocking'|'shortcut', 'dyna_q'|'dyna_q_plus'|'selection_bonus', change_step, ...)` runs the two-phase experiment; `plot_changing_maze(...)` reproduces Figs. 8.5–8.6. Tests: `TestDynaQPlus`, `TestChangingMaze`.

### Exercise 8.3 — Why did the Dyna-Q+ vs Dyna-Q gap narrow early on?

- **Book location:** §8.3, Exercise 8.3, shortcut experiment / Fig. 8.6 (2nd ed., p. 138).
- **Type:** Theory.
- **Problem (paraphrased):** In the shortcut experiment (Fig. 8.6), the performance advantage of Dyna-Q+ over Dyna-Q becomes slightly smaller during the first part of the experiment (before the shortcut opens). What explains this narrowing?
- **Solution.** In the first (stationary) phase the environment does not change and the agent gradually learns an accurate model and good values for the commonly used routes.
  - **Initially:** Dyna-Q+ explores more because many pairs have recently grown \(\tau\), so the bonus \(\kappa\sqrt{\tau}\) (Eq. 8.12) can make exploratory actions/planning targets attractive; this extra exploration can find/improve behavior faster than Dyna-Q, creating an early gap.
  - **As learning stabilizes:** The policy and visitation pattern settle. Frequently relevant (s,a) pairs are revisited often, keeping their \(\tau\) small, while the agent has less to gain from wandering; meanwhile Dyna-Q’s estimates also converge toward good behavior on the stable task. Consequently the *marginal* benefit of Dyna-Q+’s bonus-driven exploration diminishes and the cumulative-reward curves move closer together.
  - **Important contrast:** The gap widens again only later, when the shortcut opens and Dyna-Q+’s continued sensitivity to staleness helps it discover the change sooner than Dyna-Q.
- **Explanation (student-friendly).** Exploration bonuses trade off short-run exploitation for information; in a stationary phase their advantage can shrink once both agents have learned good behavior, but they remain valuable when non-stationarity appears.
- **Coding pointer:** `changing_maze_experiment('shortcut', agent, change_step=3000, ...)` with `agent='dyna_q'` or `'dyna_q_plus'`; see `TestChangingMaze`. In our reconstructed maze the early-phase curves differ from the book (documented in `README.md`); the post-change recovery pattern is reproduced.

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
    3. **Phase 2 (change):** emulate blocking (remove/penalize a corridor action by changing the environment/model) or shortcut (add a shorter passage). Continue training under three variants: (a) plain Dyna-Q, (b) `dyna_q_plus` (bonus in planning targets, Eq. 8.12), (c) `selection_bonus` via `changing_maze_experiment(..., agent='selection_bonus')`, which adds \(\kappa\sqrt{\tau}\) only at action selection.
    4. **Metrics:** cumulative reward or steps-to-goal per episode over time (as in Figs. 8.4–8.5), speed of recovery after blocking, and time to discover the shortcut.
    5. **Report:** selection-only should track true \(Q\) more faithfully but may react slower/faster than (b) depending on whether planning propagation or immediate choice pressure dominates; include fixed seeds for reproducibility.
- **Explanation (student-friendly).** Optimism can be injected at *decision time* or at *learning/planning time*; the former preserves value accuracy, the latter can spread exploratory incentives through backups — with different dynamics on non-stationary tasks.
- **Coding pointer:** `changing_maze_experiment(..., agent='selection_bonus')` implements the augmented selection rule \(\arg\max_a[Q(s,a)+\kappa\sqrt{\tau(s,a)}]\) with unchanged updates; `dyna_q_plus` is the planning-time variant. `plot_changing_maze(..., agents=('dyna_q','dyna_q_plus','selection_bonus'))` plots all three. Tests: `TestDynaQPlus`, `TestChangingMaze`.

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
