# Chapter 17 — Frontiers: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_17`
> Book chapter: **Chapter 17** — Frontiers

*Book contains no formal numbered exercises for this chapter.* The prompts below are derived from
the frontier sections, labeled **[derived]**.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_17` — Frontiers (four-room `GridWorld`, hierarchical options
  `Option` / `default_options` / `run_option`, SMDP Q-learning `smdp_q_learning`, primitive
  Q-learning `primitive_q_learning`, and intrinsic novelty `novelty_bonus` / `q_learning_novelty`).
- **Book chapter:** Chapter 17 — Frontiers (where the field is going next).
- **Book sections covered:**
  - §17.1 General Value Functions and Auxiliary Tasks
  - §17.2 Temporal Abstraction via Options
  - §17.3 Observations and State
  - §17.4 Designing Reward Signals
  - §17.5 Remaining Issues
  - §17.6 Reinforcement Learning and the Future of Artificial Intelligence
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction, approximate):** §17.1 p. 397;
  §17.2 p. 398; §17.3 p. 401; §17.4 p. 404; §17.5 p. 406; §17.6 p. 408.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 17. Section statements/prompts below are paraphrased/derived (not verbatim book exercise
  text); short references only.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction`.
- All analyses are **rewritten from scratch**.

---

## Derived Exercises

### Derived 1 — General Value Functions (GVFs) and auxiliary tasks [derived]

- **Book location:** Section 17.1, PDF p. ~397.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Define a GVF-style auxiliary prediction (e.g. predicting a future signal or
  feature outcome under a behavior) and write its TD update; explain how auxiliary predictions can
  improve the representation/features used for a main task.
- **Solution.** A **General Value Function** predicts the expected discounted sum of a specified
  **cumulant** `c_{t+1}` (not necessarily extrinsic reward) under a target/behavior policy \(\pi\) with a
  continuation/discount signal \(\gamma _t ∈ [0,1]\): \(v_{\pi ,\gamma ,c}(s)=E[\sum _{k=0}^{\infty } (Π_{i=1}^{k} \gamma _{t+i}) c_{t+k+1} | S_t=s, \pi ]\).
  For off-policy auxiliary prediction, importance sampling (or other corrections) may be needed; the
  simplest on-policy TD(0) update for a parameterized predictor `\hat v(s,w)` is
  `δ_t = c_{t+1} + γ_{t+1}\hat v(S_{t+1},w) - \hat v(S_t,w)`, `w ← w + α δ_t ∇_w \hat v(S_t,w)`.
  **Why this helps the main task:** auxiliary predictions (next features, sensors, termination, etc.)
  force the agent to learn state features that capture environment dynamics and structure, providing a
  richer representation and learning signal even when extrinsic reward is sparse.
- **Why this works / takeaway:** predicting many auxiliary signals builds useful features without
  external reward.

### Derived 2 — SMDP Q-learning [derived]

- **Book location:** Section 17.2, PDF p. ~398.
- **Type:** Math / Programming.
- **Problem (paraphrased):** Derive the SMDP Q-learning update for an option
  \(o = (I, \pi , \beta )\): \(Q(s,o) \leftarrow Q(s,o) + \alpha [r + \gamma ^k max_{o'} Q(s',o') - Q(s,o)]\). Contrast it with
  ordinary one-step Q-learning and explain the variable-duration return.
- **Solution.** When option `o` is initiated in state `s_t ∈ I` and executes for `k` primitive steps
  until termination at `s_{t+k}`, it produces cumulative discounted reward
  \(r = \sum _{j=0}^{k-1} \gamma ^j R_{t+1+j}\) and next decision state `s' = s_{t+k}`. Treating the option as a
  temporally extended action in a Semi-Markov Decision Process gives the SMDP Bellman optimality form
  \(Q*(s,o)=E[r + \gamma ^k max_{o'} Q*(s',o') | s,o]\), so SMDP Q-learning updates
  \(Q(s,o) \leftarrow Q(s,o) + \alpha [r + \gamma ^k max_{o'} Q(s',o') - Q(s,o)]\).
  **Contrast:** ordinary Q-learning is the special case `k=1` (a primitive action). The option return
  discounts across the option’s duration via \(\gamma ^k\) and bootstraps only at option termination, enabling
  multi-step temporal abstraction and fewer high-level decisions.
- **Why this works / takeaway:** options fit the Bellman framework when rewards and discounting are
  aggregated over variable option durations.

### Derived 3 — Options as temporal abstraction that accelerates learning [derived]

- **Book location:** Section 17.2, PDF p. ~398.
- **Type:** Theory / Programming.
- **Problem (paraphrased):** (a) Give the formal option definition \((I, \pi , \beta )\). (b) Explain (and relate
  to the repo demo) how subgoal options can reduce decisions and speed learning versus primitives.
- **Solution.** (a) A Markov option is \(o = (I, \pi , \beta )\) where `I ⊆ S` is the initiation set, \(\pi (a|s)\) is
  the intra-option policy, and \(\beta (s) ∈ [0,1]\) is the termination condition. When `o` is executing, actions
  follow \(\pi\) and the option stops in state `s` with probability \(\beta (s)\).
  (b) Subgoal options (e.g. “go to hallway/door”) package useful multi-step behaviors. In the repo’s
  four-room gridworld, SMDP Q-learning over primitives+subgoal options (`smdp_q_learning`) can reach the
  goal with far fewer **macro decisions** than primitive-only control (see `TestOptions::test_options_solve_in_few_macro_decisions`
  and `test_options_agent_vs_primitive_only_agent`), because high-level choices jump between meaningful
  subgoals instead of replanning every primitive step.
- **Why this works / takeaway:** reusable subgoal skills compress planning/control depth.

### Derived 4 — Observations and state [derived]

- **Book location:** Section 17.3, PDF p. ~401.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Explain why observations are not states in POMDPs, derive the belief-state
  update (Bayes filter), and discuss memory/RNNs as state estimators.
- **Solution.** In a **POMDP**, the agent receives observation `o_t` drawn from `O(o_t | s_t, a_{t-1})`
  and the latent environment state `s_t` may not be fully revealed (different states can emit similar
  observations), so `o_t` is not Markov in general. A sufficient statistic is the **belief**
  `b_t(s)=P(s_t=s | h_t)` given history `h_t`. After action `a_t` and observation `o_{t+1}`, Bayes’
  rule gives:
  \(b_{t+1}(s') = η \cdot O(o_{t+1} | s', a_t) \cdot \sum _s P(s' | s, a_t) b_t(s)\),
  where `η = 1 / P(o_{t+1} | b_t, a_t)` normalizes so that \(\sum _{s'} b_{t+1}(s') = 1\). Because optimal
  control can depend on the full history, practical agents use **memory** (recurrent policies/RNNs,
  belief trackers) to approximate latent state from observation sequences.
- **Why this works / takeaway:** maintain/update a belief (or learned memory state) when observations are
  partial.

### Derived 5 — Designing reward signals [derived]

- **Book location:** Section 17.4, PDF p. ~404.
- **Type:** Theory.
- **Problem (paraphrased):** Discuss sparse vs dense rewards, derive potential-based shaping
  \(F(s,a,s') = \gamma Φ(s') - Φ(s)\) and show it preserves optimal policies, and note pitfalls of ad hoc
  shaping.
- **Solution.** **Sparse rewards** are easy to specify but provide little learning signal; **dense rewards**
  can accelerate learning but may encode the designer’s biases and create unintended incentives.
  **Potential-based shaping** augments reward as `R' = R + F` with \(F(s,a,s') = \gamma Φ(s') - Φ(s)\) for
  potential `Φ`. For any policy, the shaped return telescopes:
  \(\sum _t \gamma ^t F(s_t,a_t,s_{t+1}) = -Φ(s_0) + lim_{T\to \infty } \gamma ^T Φ(s_T)\) (constant/terminal-dependent offset
  independent of action choices in the standard discounted/episodic settings), so action-value
  advantages and optimal policies are preserved: `Q'*(s,a)` differs from `Q*(s,a)` by a state-dependent
  term that does not change `argmax_a`. **Pitfall:** non-potential shaping (or poorly chosen dense
  rewards) can change the optimal policy and produce reward hacking.
- **Why this works / takeaway:** shape with potentials to guide learning without altering optimality.

### Derived 6 — Intrinsic motivation and novelty bonuses [derived]

- **Book location:** Sections 17.1, 17.4, PDF pp. ~397–404.
- **Type:** Theory / Math.
- **Problem (paraphrased):** Define a novelty/intrinsic bonus (count- or prediction-error-based), show how
  it encourages visiting rare states, write the combined reward, and contrast with Dyna exploration
  bonuses (§8.3/8.4).
- **Solution.** A simple **count-based** novelty bonus is \(b(s)=\beta / sqrt(N(s)+1)\) (as in
  `novelty_bonus`) or similar decreasing functions of visit count `N(s)` (e.g. `∝ 1/N(s)`), giving
  larger intrinsic reward to rarely visited states. The agent optimizes modified reward
  `R'_t = R_t^{ext} + b_t^{int}` (e.g. \(q \leftarrow q + \alpha [(R_ext + b) + \gamma max_{a'} q(s',a') - q]\)), so the
  effective/modified optimal policy becomes
  \(\pi *_{mod} = argmax_\pi E_\pi [\sum _t \gamma ^t (R_t^{ext} + b_t^{int})]\), trading task return against exploration.
  **Contrast with Dyna bonuses (§8.3/8.4):** Dyna-Q+/model-based bonuses (e.g. \(κ\sqrt τ\)) are typically used
  to encourage trying actions/state–action pairs whose outcomes have not been revisited recently in the
  **model/planning** loop (useful under nonstationarity), whereas intrinsic novelty bonuses directly
  augment experienced rewards to drive visitation/coverage during interaction.
- **Why this works / takeaway:** intrinsic bonuses turn uncertainty/novelty into a learnable signal in
  sparse-reward settings.

### Derived 7 — Reinforcement learning and the future of AI [derived]

- **Book location:** Sections 17.5–17.6, PDF pp. ~406–410.
- **Type:** Theory.
- **Problem (paraphrased):** Discuss frontiers including (a) open-world/continual learning, (b) inductive
  bias and symmetry, (c) abstraction/unification, and (d) scaling; give one research direction each for
  representation, credit assignment, exploration, and scaling.
- **Solution.** (a) **Open-world/continual learning:** agents must handle nonstationarity, new objects,
  actions, and goals over long lifetimes without catastrophic forgetting. (b) **Inductive bias and
  symmetry:** priors (equivariance, compositionality, physics) can improve sample efficiency and
  generalization. (c) **Abstraction:** learning reusable temporal/state abstractions (options, GVFs,
  hierarchies) supports transfer and planning. (d) **Scaling:** larger models/data/compute interact with
  stability, alignment, and efficiency constraints.
  Example research directions (one per area): **representation** — learn disentangled/compositional state
  abstractions and GVF-based features; **credit assignment** — better long-horizon/causal credit
  assignment beyond local TD/eligibility limits; **exploration** — principled intrinsic motivation and
  information-seeking in vast state spaces; **scaling** — efficient, stable training and deployment of
  RL agents that generalize and adapt continually.
- **Why this works / takeaway:** frontier challenges focus on generality, abstraction, and robust
  long-term learning beyond current benchmarks.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 1 | §17.1 | *Theory only in this repo chapter (no GVF/auxiliary-task implementation exposed)* | — | — |
| Derived 2–3 | §17.2 | `chapter_17/solution.py` (`GridWorld`, `Option`, `default_options`, `make_subgoal_option`, `run_option`, `smdp_q_learning`, `greedy_smdp_action`, `primitive_q_learning`) | `TestGridWorld`, `TestOptions` | `python -m pytest chapter_17/tests/test_solution.py::TestGridWorld chapter_17/tests/test_solution.py::TestOptions -v` |
| Derived 4–5, 7 | §17.3–17.6 | *Theory/design discussion only in this repo chapter* | — | — |
| Derived 6 | §17.1, §17.4 | `chapter_17/solution.py` (`novelty_bonus`, `q_learning_novelty`) | `TestNoveltyBonus` | `python -m pytest chapter_17/tests/test_solution.py::TestNoveltyBonus -v` |

- Full chapter command: `python -m pytest chapter_17/tests/ -v`
- Lint: `ruff check chapter_17/`

---

## Verification and Quality Notes

- Derived analyses were re-derived independently; public frontier material was used only as a
  cross-check.
- Derived 4 uses the standard POMDP belief update
  \(b'(s') = η O(o_{t+1}|s',a_t) \sum _s P(s'|s,a_t) b_t(s)\).
- The repo demonstrates options/SMDP and count-based intrinsic novelty on a small gridworld; GVFs,
  belief-state/RNN estimators, and potential-based shaping are discussed theoretically here.

---

*Chapter 17 (Frontiers) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
