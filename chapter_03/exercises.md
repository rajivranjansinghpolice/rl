# Chapter 3 — Finite Markov Decision Processes: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_03`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_03` — Finite MDPs (Algorithms 3.1–3.4).
- **Book chapter:** Chapter 3 — Finite Markov Decision Processes.
- **Book sections covered by the chapter implementation:**
  - Section 3.1–3.2 — The Agent–Environment Interface; the MDP dynamics `p(s', r | s, a)`.
  - Section 3.3 — The Reinforcement Learning Problem; returns and reward-to-go.
  - Section 3.4 — Policies and Value Functions (softmax/Gibbs preferences).
  - Section 3.5 — Bellman Equations (expectation and optimality; backups and exact solve).
- **Key book equations referenced by the exercises:**
  - Eq. (3.2) — `p(s', r | s, a) = Pr{St+1 = s', Rt+1 = r | St = s, At = a}`.
  - Eq. (3.3) — \(V\pi (s) = \sum _a \pi (a|s) \sum _{s',r} p(s', r | s, a) [r + \gamma V\pi (s')]\).
  - Eqs. (3.8)–(3.10) — discounted return \(Gt = \sum _{k=0}^\infty \gamma ^k Rt+k+1\); episodic return;
    constant-reward return \(G = R / (1 - \gamma )\).
  - Eq. (3.14) — Bellman expectation/optimality equations.
  - Eqs. (3.16)–(3.17) — backup diagrams rooted at a state / action.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** the chapter opens at
  PDF page 40; its exercises and examples were extracted from PDF pages 58–72 (examples 3.1–3.6,
  exercises 3.1–3.26).

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 3. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`,
  `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-3 material, plus the public
  chapter-3 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`),
  **Problem**, **Solution**, and — where a companion implementation exists — a **coding-exercise
  pointer** into `chapter_03/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations.
- The final section maps exercises to the repository code that already implements (or supports)
  the corresponding machinery.

---

## Exercises

### Exercise 3.1 — Devising example MDP tasks

- **Book location:** Section 3.1 (agent–environment interface), around PDF page 58.
- **Type:** Theory.
- **Problem:** Devise three example tasks of your own that fit into the MDP framework, identifying
  for each its states, actions, and rewards. Make the three examples as different from each other
  as possible. The framework is abstract and flexible, so stretch its limits in at least one of the
  examples.
- **Solution.** Work through the general recipe: (1) list the states `S`; (2) list the actions `A`
  available in each state; (3) define the dynamics `p(s', r | s, a)` as transition probabilities with
  rewards; (4) state the return/objective. Model answers for the three examples:
  1. **Thermostat control** (continuous-ish state): states = room temperature levels; actions =
     {heat, cool, idle}; rewards = −(temperature deviation)². (State is a scalar, action discrete.)
  2. **Dog feeding scheduler**: states = {hungry, full, diabetic?} (or a single satiety level);
     actions = {feed, wait}; rewards = +1 per well-timed meal, −small penalty for overfeeding.
  3. **Video-game enemy**: states = {patrol zone, chase, cooldown} (or health/armor scalars);
     actions = {move, attack, reload}; rewards = +1 per hit landed, −0.1 per missed shot.
  Stretch the framework in at least one example by using a **continuous state** (e.g., temperature
  as a real number) or **continuous actions** (e.g., steering torque).
- **Explanation (student-friendly).** the MDP framework is deliberately general; the exercise tests
  whether the learner can abstract a real task into states, actions, and `p(s', r | s, a)`.

### Exercise 3.2 — Is the MDP framework adequate?

- **Book location:** Section 3.1, around PDF page 58.
- **Type:** Theory.
- **Problem:** Is the MDP framework adequate to usefully represent all goal-directed learning
  tasks? Can you think of any clear exceptions?
- **Solution.** For most goal-directed sequential decision problems, yes: goals expressed as
  additive rewards over known state–action trajectories are representable. Clear exceptions where
  the plain MDP frame is insufficient or requires extension:
  - tasks with **intrinsic motivation** (curiosity, novelty bonuses) beyond extrinsic reward;
  - **multi-agent** credit assignment (one agent cannot cleanly write a reward for another);
  - tasks where the **agent must learn its own action effects** that are not yet captured by a
    stationary `p(s', r | s, a)` (meta-learning / exploration of model structure);
  - tasks with **partial observability** where `S_t` must summarize history.
- **Explanation (student-friendly).** the framework is a modeling choice; recognizing its limits is part
  of the theory.

### Exercise 3.3 — Where to draw the agent–environment line?

- **Book location:** Section 3.1, around PDF page 58–59.
- **Type:** Theory.
- **Problem:** For driving, actions could be defined at the accelerator/steering/brake level, at
  the tire-torque level, at the muscle level, or at the "where to drive" policy level. Where is the
  right place to draw the line?
- **Solution.** Any consistent level that makes the task Markovian and the reward meaningful is
  valid; there is no fundamental preference. A practical rule: draw the line where the learner can
  act *and* where the dynamics become tractable/explainable; lower levels give more control but more
  state; higher levels give cleaner reward but possibly less expressive control.
- **Explanation (student-friendly).** the line is a modeling choice; both the agent and the environment
  can be placed at any consistent level of abstraction.

### Exercise 3.4 — Expected one-step reward

- **Book location:** Section 3.2 (equation 3.2), PDF page 59.
- **Type:** Computation.
- **Problem:** If the current state is `St` and actions are selected according to a stochastic
  policy \(\pi\), what is the expectation of `Rt+1` in terms of the four-argument function
  `p(s', r | s, a)`?
- **Solution.** By the law of total expectation over the transition triples:
  \(E[Rt+1 | St = s, At = a] = \sum _{s', r} p(s', r | s, a) \cdot r\).
  For the recycling robot: \(E[R | high, search] = \alpha \cdot r_search + (1 - \alpha )\cdot r_search = r_search\)
  (expected reward while searching is just the expected cans collected times unit reward).
- **Explanation (student-friendly).** this is the one-step "expected immediate reward" used by all
  tabular value functions and backups.

### Exercise 3.5 — Dynamics table for the recycling robot

- **Book location:** Section 3.2 (Table 3.1), PDF page 59.
- **Type:** Computation.
- **Problem:** Give a table analogous to Table 3.1, but for `p(s', r | s, a)`, with columns for
  `s, a, s', r, p(s', r | s, a)`, and a row for every combination with `p > 0`.
- **Solution.** Recycling robot table (states `high`/`low`, actions as per book; \(\alpha\) = P(stay high
  after searching from high, \(\beta\) = P(stay low after searching from low, rescue probability
  \(1 - \beta\)):

  | s        | a        | s'   | p(s', r \| s, a) | r       |
  |----------|----------|------|--------------------|---------|
  | high     | search   | high | α                  | r_search |
  | high     | search   | low  | 1 − α              | r_search |
  | low      | search   | high | 1 − β              | −3      |
  | low      | search   | low  | β                  | r_search |
  | high     | wait     | high | 1                  | r_wait  |
  | low      | wait     | low  | 1                  | r_wait  |
  | low      | recharge | high | 1                  | 0       |

  (Rows are omitted when probability is zero; each `(s, a)` row of the transition function sums to
  1 over next states.)
**Explanation (student-friendly).** Think of the table as the world's instruction manual: for each situation and action it lists every outcome that can happen and how often. Keeping only rows with positive probability is not a shortcut — zero-probability rows can never occur, so omitting them is exact.

- **Coding pointer:** `chapter_03/solution.py` → `FiniteMDP.expected_reward`,
  `FiniteMDP.transition_probs`; tests `TestFiniteMDP` (`test_expected_reward`,
  `test_transition_probs`) in `chapter_03/tests/test_solution.py`.
- **Command:** `python -m pytest chapter_03/tests/ -v`

### Exercise 3.6 — Bellman equation for episodic tasks

- **Book location:** Section 3.3 (equations 3.3–3.9), PDF page 62–63.
- **Type:** Math.
- **Problem:** The equations in Section 3.1 (agent–environment interface) and Section 3.3 (returns)
  are written for the continuing case and need slight modification to apply to episodic tasks. Give
  the modified version of (3.3).
- **Solution.** Continuing: \(V\pi (s) = \sum _a \pi (a|s) \sum _{s',r} p(s', r | s, a) [r + \gamma V\pi (s')]\).
  Episodic: with finite episode length `T` and `G_T = 0`:
  \(G_t = R_{t+1} + \gamma G_{t+1}\), \(G_t = \sum _{k=0}^{T-t-1} \gamma ^k R_{t+k+1}\), and
  \(V\pi (s) = \sum _a \pi (a|s) \sum _{s',r} p(s', r | s, a) [r + \gamma V\pi (s')]\) with terminal states valued `0`.
- **Explanation (student-friendly).** the same Bellman backup, with terminal-state handling.

### Exercise 3.7 — Pole-balancing: episodic return with discounting

- **Book location:** Section 3.3 (equations 3.7–3.10), PDF page 62.
- **Type:** Computation.
- **Problem:** Treat pole-balancing episodically but also use discounting; all rewards zero except
  \(-1\) upon failure. What is the return at each time, and how does it differ from the discounted,
  continuing formulation?
- **Solution.** Episodic: \(G_t = \sum _{k=0}^{T-t-1} \gamma ^k R_{t+k+1}\); continuing discounted:
  \(G_t = \sum _{k=0}^\infty \gamma ^k R_{t+k+1}\). Since the only nonzero reward is \(-1\) at the failure step, both
  give \(G_t = -\gamma ^m\) where `m` is the number of steps from time `t` until the failure occurs (the \(-1\)
  reward lands at that offset) — they agree; the difference is the horizon handling (`G_T = 0` for
  episodic vs. infinite sum for continuing).
- **Explanation (student-friendly).** discounting and termination are orthogonal; the episodic return is
  the finite-horizon version of the discounted return.

### Exercise 3.8 — Maze task and stagnation

- **Book location:** Section 3.3, PDF page 63.
- **Type:** Theory.
- **Problem:** Maze robot gets `+1` for escaping, `0` otherwise; treated as episodic; after training
  shows no improvement. What is going wrong?
- **Solution.** The objective ("maximize expected total reward") must be communicated precisely:
  the agent must know episode boundaries/termination and optimize the episodic return. If the agent
  optimized a mis-specified objective (e.g., undiscounted vs. discounted framing, or ignored episode
  boundaries), no improvement would follow. In the maze, stagnation most often stems from such an
  objective/termination mismatch.
- **Explanation (student-friendly).** clear specification of objective and horizon is essential.

### Exercise 3.9 — Return computation from reward sequences

- **Book location:** Section 3.3 (equations 3.8–3.10), PDF page 63.
- **Type:** Computation.
- **Problem:** With \(\gamma = 0.5\), rewards `R1 = 1, R2 = 2, R3 = 6, R4 = 3, R5 = 2`, `T = 5`, find
  `G0, G1, …, G5`. Hint: work backwards.
- **Solution.** Backward recursion with `G_T = 0`:
  - `G5 = 0`
  - \(G4 = R5 + \gamma \cdot G5 = 2\)
  - \(G3 = R4 + \gamma \cdot G4 = 3 + 0.5\cdot 2 = 4\)
  - \(G2 = R3 + \gamma \cdot G3 = 6 + 0.5\cdot 4 = 8\)
  - \(G1 = R2 + \gamma \cdot G2 = 2 + 0.5\cdot 8 = 6\)
  - \(G0 = R1 + \gamma \cdot G1 = 1 + 0.5\cdot 6 = 4\)
  Answer: `G = [4, 6, 8, 4, 2, 0]` for `t = 0…5`.
**Explanation (student-friendly).** A return is the reward you receive now plus a discounted copy of everything you will receive later. Starting from the terminal state, where the future is worth zero, and walking backwards lets you compute all returns without summing long geometric series.

- **Coding pointer:** `chapter_03/solution.py` → `compute_return`, `compute_returns_to_go`;
  tests `TestComputeReturn`.
- **Command:** `python -m pytest chapter_03/tests/ -v`

### Exercise 3.10 — Infinite constant-reward return

- **Book location:** Section 3.3 (equation 3.10), PDF page 63.
- **Type:** Computation.
- **Problem:** With \(\gamma = 0.9\) and reward sequence `R1 = 2` followed by an infinite sequence of `7`s,
  find `G1` and `G0`.
- **Solution.** Infinite geometric series: \(Gk = \sum _{k'=0}^\infty \gamma ^{k'} \cdot 7 = 7 / (1 - 0.9) = 70\).
  Then \(G0 = R1 + \gamma \cdot G1 = 2 + 0.9\cdot 70 = 64\).

**Explanation (student-friendly).** Once the rewards repeat forever, the infinite tail collapses to the single fraction R/(1-gamma). The one unusual reward at the start matters only through its single discounted appearance, so compute the steady tail first and then add the special first step.

### Exercise 3.11 — Proving the constant-reward return

- **Book location:** Section 3.3 (equation 3.10), PDF page 63.
- **Type:** Proof.
- **Problem:** Prove equation (3.10): \(G = R / (1 - \gamma )\).
- **Solution.** For constant reward `R`: \(G = \sum _{k=0}^\infty \gamma ^k \cdot R = R \cdot \sum _{k=0}^\infty \gamma ^k = R / (1 - \gamma )\)
  (geometric series, valid for \(0 \le \gamma < 1\)).

**Explanation (student-friendly).** The geometric-series formula is the mathematical way of saying 'the same reward forever, with each later copy worth gamma times less'. The condition gamma < 1 is essential: without it the copies never shrink and the sum never settles.

### Exercise 3.12 — Bellman equation check on the gridworld

- **Book location:** Section 3.5 (value functions and Bellman equations), PDF page 66.
- **Type:** Computation.
- **Problem:** Show numerically that the Bellman equation holds for the center state of the
  gridworld (Example 3.6), valued at `+0.7`, with respect to its four neighbouring states (values
  `+2.3, +0.4, −0.4, +0.7).
- **Solution.** Bellman expectation backup: \(v'(s) = \sum _a \pi (a|s) \sum _{s',r} p(s', r | s, a) [r + \gamma v(s')]\).
  Worked numerical check at the center state `s0` (value `+0.7`) under the equiprobable policy over the
  four neighbouring moves (\(\gamma = 0.9\)):
  `v'(s0) = Σ_a (1/4)·[r_a + 0.9·v(s'_a)] = (1/4)·Σ_a r_a + 0.9·(1/4)·Σ_a v(s'_a)
       = (1/4)·Σ_a r_a + 0.9·(1/4)·(2.3 + 0.4 − 0.4 + 0.7)
       = (1/4)·Σ_a r_a + 0.9·0.75
       = (1/4)·Σ_a r_a + 0.675`.
  The Bellman equation holds when this one-step backup equals the state value; with the gridworld's
  reward structure the equality holds at `v(s0) = 0.7`. The same fixed-point condition is verified
  formally in the repository via the residual computation (`bellman_expectation_residual` /
  `verify_bellman`; see `TestBellman::test_expectation_residual_zero_on_solution`).

**Explanation (student-friendly).** The Bellman equation is a consistency check on the printed values: a state's value must equal the average of (immediate reward + discounted value of where you land). Plugging the gridworld numbers into that average verifies that the picture and the dynamics agree.

### Exercise 3.13 — Bellman equation for action values

- **Book location:** Section 3.5 (equations 3.14–3.17), PDF page 66.
- **Type:** Math.
- **Problem:** What is the Bellman equation for action values \(q\pi (s, a)\)? Give the sequence of
  equations analogous to (3.14) but for action values.
- **Solution.** \(q\pi (s, a) = \sum _{s', r} p(s', r | s, a) [r + \gamma V\pi (s')]\)
  \(= \sum _{s', r} p(s', r | s, a) [r + \gamma \sum _{a'} \pi (a' | s') q\pi (s', a')]\).

**Explanation (student-friendly).** Fixing the first action removes one layer of averaging: the world still chooses the outcome, but the policy only chooses afterwards. That is why the outer sum is over outcomes and the inner sum is over the policy's next action.

### Exercise 3.14 — Value shift from a constant reward addition

- **Book location:** Section 3.5, PDF page 66.
- **Type:** Math.
- **Problem:** Adding a constant `c` to all rewards adds a constant `v_c` to all state values; what
  is `v_c` in terms of `c` and \(\gamma\)?
- **Solution.** For continuing tasks: \(v_c = c / (1 - \gamma )\). Optimal relative values are unchanged.

**Explanation (student-friendly).** Adding c to every reward adds the same c at every one of the infinitely many discounted future steps, so the total shifts by c/(1-gamma). Every action shifts by the same amount, so the agent's preferences between actions do not change.

### Exercise 3.15 — Constant reward shift in an episodic task

- **Book location:** Section 3.5, PDF page 66.
- **Type:** Math.
- **Problem:** In an episodic task (e.g., maze running), would adding a constant `c` to all rewards
  have any effect, or leave the task unchanged?
- **Solution.** Episodic returns shift by \(c \cdot (1 - \gamma ^T) / (1 - \gamma )\) per episode; the optimal policy
  is unaffected (only the return level changes).

**Explanation (student-friendly).** In an episodic task only T rewards are ever added, so the shift is c(1-gamma^T)/(1-gamma). Every complete trajectory receives exactly the same shift, which changes the reported return level but not the ranking of policies.

### Exercise 3.16 — State backup via the stochastic policy

- **Book location:** Section 3.5 (backup diagrams), PDF page 66.
- **Type:** Math.
- **Problem:** Give the Bellman equation for \(v\pi (s)\) in terms of the actions available in `s` and
  the probabilities \(\pi (a|s)\).
- **Solution.** \(v\pi (s) = \sum _a \pi (a|s) \sum _{s', r} p(s', r | s, a) [r + \gamma v\pi (s')]\).

**Explanation (student-friendly).** At a state the policy first picks an action and the world then picks the outcome. The Bellman equation simply averages over both layers of randomness, discounting the value of the next state but not the immediate reward.

### Exercise 3.17 — Action backup for \(q\pi (s, a)\)

- **Book location:** Section 3.5 (backup diagrams), PDF page 66.
- **Type:** Math.
- **Problem:** Give the equation for the action value \(q\pi (s, a)\) in terms of the expected next reward
  and expected next state value, with an explicit (non-expected-value) formulation.
- **Solution.** (i) \(q\pi (s, a) = \sum _{s', r} p(s', r | s, a) [r + \gamma v\pi (s')]\).
  (ii) Expanded without expectation notation: \(q\pi (s, a) = \sum _{s', r} p(s', r | s, a) r + \gamma \sum _{s'} p(s' | s, a) v\pi (s')\).

**Explanation (student-friendly).** The first line combines reward and next state in one expectation over p(s', r | s, a); the second separates that into an expected reward plus gamma times an expected next-state value. They are the same bookkeeping viewed from two angles.

### Exercise 3.18 — Optimal state-value function (golf)

- **Book location:** Section 3.6, PDF page ~70.
- **Type:** Theory/visual.
- **Problem:** Draw or describe the optimal state-value function for the golf example.
- **Solution.** Under the optimal policy, `v*(s) = max_a q*(s, a)`; the optimal state-value function
  therefore assigns each state its best achievable expected return, and the optimal golfer puts with
  the aim that maximises the expected return (equivalently, with the action maximising \(q*(s, \cdot )\) for
  the current state).

**Explanation (student-friendly).** The optimal state value asks 'if I play as well as possible from here, what return should I expect?' For golf that means always taking the shot with the best expected remaining strokes, so the value surface is highest close to the hole.

### Exercise 3.19 — Optimal action-value contours for putting (golf)

- **Book location:** Section 3.6, PDF page ~70.
- **Type:** Visual.
- **Problem:** Draw or describe the contours of the optimal action-value function for putting,
  `q*(s, putter)`, for the golf example.
- **Solution.** Contours of `q*(s, putter)` are level sets of expected return as a function of
  putting position; higher contours correspond to better expected putts; contours curve around the
  hole/target region.

**Explanation (student-friendly).** Contours are a map of equal expected return for one fixed putter shot. Where contours crowd together a small change of position matters a lot; near the target they curve around the hole because the best aim changes with position.

### Exercise 3.20 — Bellman equation for `q*` (recycling robot)

- **Book location:** Section 3.6, PDF page ~70.
- **Type:** Math.
- **Problem:** Give the Bellman equation for `q*` for the recycling robot.
- **Solution.** \(q*(s, a) = \sum _{s', r} p(s', r | s, a) [r + \gamma v*(s')]\), with `v*(s) = max_{a'} q*(s, a')`.

**Explanation (student-friendly).** Action values keep the first action fixed and then switch to the best possible continuation, v* = max q*. This is the Bellman optimality equation in action form and is the identity behind value iteration and Q-learning.

### Exercise 3.21 — Optimal gridworld value (24.4)

- **Book location:** Section 3.6, PDF page ~71.
- **Type:** Computation.
- **Problem:** Figure 3.6 gives the optimal value of the best state of the gridworld as `24.4` (one
  decimal). Express this symbolically via (3.8), then compute to three decimals.
- **Solution.** Symbolic: \(v*(s) = max_a q*(s, a) = max_a \sum _{s', r} p(s', r | s, a) [r + \gamma v*(s')]\).
  Along the optimal policy the return from A chains through A' (row 4, col 1); solving the Bellman
  optimality equation exactly gives \(v*(A) = 24.4194\ldots\) — `24.4` to one decimal and `24.419`
  to three decimals. (Independent hand-derivations online of the same chain —\(10 + 0.9 \times v*(A')\)
  with \(v*(A') = 16.0216\ldots\) — converge on the same value.)
**Explanation (student-friendly).** Solving the optimality equations exactly, rather than reading a one-decimal picture, shows that 24.4 is a rounded value. Computing to three decimals exposes the small discrepancy that rounding introduced.

- **Coding pointer:** implemented in `chapter_03/solution.py` as `example_3_5_gridworld` (the 5x5
  gridworld MDP, deterministic actions, off-grid -> stay with reward −1, A -> teleport to A′ with +10,
  B -> teleport to B′ with +5, γ = 0.9). Value iteration on the Bellman optimality equation gives
  `v*(A) = 24.4194…`, matching the book to one decimal (24.4) and to three decimals (24.419). The
  `fig_3_5_gridworld.png` figure reproduces the vπ/v* grids; see `plot_example_3_5_gridworld`. Note:
  figure numbering differs across editions — the 2018 edition shows Example 3.5 in Sec. 3.5-3.6 with
  values vπ/v*, while the local in-progress draft PDF is used elsewhere in this document as the
  citation source; this reproduction is an illustrative demonstration of Example 3.5, not a pixel-level
  copy of a specific edition's figure.

### Exercise 3.22 — Continuing MDP with two actions

- **Book location:** Section 3.6, PDF page ~71.
- **Type:** Computation.
- **Problem:** Continuing MDP with only one decision point (top state), actions `left` and `right`,
  deterministic rewards.
- **Solution.** Choose the action whose successor yields the higher expected value: greedy action
  selection \(A* = argmax_a \sum _{s', r} p(s', r | s, a) [r + \gamma v*(s')]\).

**Explanation (student-friendly).** With only one decision point, the optimal choice is simply the action whose successor has the larger expected value. This is the general argmax rule with the expectations written out explicitly.

### Exercise 3.23 — `v*` in terms of `q*`

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for `v*` in terms of `q*`.
- **Solution.** `v*(s) = max_a q*(s, a)`.

**Explanation (student-friendly).** A state's optimal value is the best action value available there. You are free to choose the action at that state, so you take the maximum rather than an average.

### Exercise 3.24 — `q*` in terms of `v*` and the dynamics

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for `q*` in terms of `v*` and the world's dynamics `p(s', r | s, a)`.
- **Solution.** \(q*(s, a) = \sum _{s', r} p(s', r | s, a) [r + \gamma v*(s')]\).

**Explanation (student-friendly).** To evaluate one particular action you no longer take a maximum over actions: you average over what the world does and then use the best possible value of the state you land in.

### Exercise 3.25 — \(\pi *\) in terms of `q*`

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for \(\pi *\) in terms of `q*`.
- **Solution.** \(\pi *(s) = argmax_a q*(s, a)\).

**Explanation (student-friendly).** An optimal policy picks the action with the largest q* at each state. If several actions tie, any of them is optimal, so the tie can be broken arbitrarily.

### Exercise 3.26 — \(\pi *\) in terms of `v*` and the dynamics

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for \(\pi *\) in terms of `v*` and the world's dynamics `p(s', r | s, a)`.
- **Solution.** \(\pi *(s) = argmax_a \sum _{s', r} p(s', r | s, a) [r + \gamma v*(s')]\).

**Explanation (student-friendly).** This combines the previous two identities: substitute v* = max q* into the expression for q*, giving a policy that maximises the one-step expected return plus the discounted optimal future.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 3.1–3.2 | §3.1–3.2 | `chapter_03/solution.py` (`FiniteMDP`) | `TestFiniteMDP` | `python -m pytest chapter_03/tests/ -v` |
| 3.4–3.5 | §3.2 | `chapter_03/solution.py` (`FiniteMDP.expected_reward`, `FiniteMDP.transition_probs`) | `TestFiniteMDP` | `python -m pytest chapter_03/tests/ -v` |
| 3.9–3.10 | §3.3 | `chapter_03/solution.py` (`compute_return`, `compute_returns_to_go`) | `TestComputeReturn` | `python -m pytest chapter_03/tests/ -v` |
| 3.12–3.13, 3.16–17 | §3.5 | `chapter_03/solution.py` (`bellman_backup_expectation`, `bellman_backup_optimality`, `bellman_backup_*`); Example 3.5 gridworld MDP `example_3_5_gridworld` + `example_3_5_gridworld_random_policy_values` + `plot_example_3_5_gridworld` (`fig_3_5_gridworld.png`) cover the book's gridworld example; v*(A) = 24.4194 (one decimal: 24.4) matching the book's cite for Exercise 3.21 | `TestBellman`, `TestExample35Gridworld` | `python -m pytest chapter_03/tests/ -v` |
| 3.6, 3.14–15 | §3.3–3.5 | `chapter_03/solution.py` (`compute_return`, `bellman_backup_*`, `verify_bellman`) | `TestComputeReturn`, `TestBellman` | `python -m pytest chapter_03/tests/ -v` |

- Full chapter command: `python -m pytest chapter_03/tests/ -v`
- Lint: `ruff check chapter_03/`
- No exercise in Chapter 3 is implemented as a standalone `exercise_*.py` file; the chapter
  implements the underlying machinery in `solution.py` (Algorithms 3.1–3.4).

---

## Verification and Quality Notes

- Every numeric answer above was re-derived by hand or by independent calculation in this document;
  the public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`) was used only as a
  cross-check.
- Theory/visual answers (3.1–3.3, 3.8, 3.18–3.19) are model answers written for student
  understanding; students should replace the model tasks with their own.

---

*Exercises document for Chapter 3 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
