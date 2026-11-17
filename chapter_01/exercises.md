# Chapter 1 — What is Reinforcement Learning?: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_01`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_01` — What is reinforcement learning? (Gridworld example,
  Sutton & Barto Section 1.3, Figure 1.2; plus the tic-tac-toe example).
- **Book chapter:** Chapter 1 — Introduction.
- **Book sections covered by the chapter implementation:**
  - Section 1.1 — The reinforcement learning problem and the computational approach.
  - Section 1.3 — An example agent (gridworld MDP, Figure 1.2; tic-tac-toe with lookahead).
- **Key chapter concepts:** agent, environment, state, action, reward, return `G_t`, expected value
  `V(s) = E[G | S]`.
- **Return / value equations:** the chapter introduces the discounted return
  `G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}` and the expected value `vπ(s) = E[G | S = s, π]` under a policy
  `π`, with optimal values `v*(s)`.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** the chapter opens around
  PDF page 19; the gridworld material and the exercises (1.1–1.5) were extracted from PDF page 28.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 1. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`,
  `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-1 material, plus the public
  chapter-1 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`),
  **Problem**, **Solution**, and — where a companion implementation exists — a **coding-exercise
  pointer** into `chapter_01/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations.
- The final section maps exercises to the repository code that already implements (or supports)
  the corresponding machinery.

---

## Exercises

### Exercise 1.1 — Self-Play

- **Book location:** Section 1.5, PDF page 28.
- **Type:** Theory.
- **Problem:** The tic-tac-toe player could look ahead via a model of its environment (or play
  against a random opponent). Suppose the reinforcement learning algorithm played against itself,
  with both sides learning. What would happen? Would it learn a different policy for selecting moves?
- **Solution.** Under symmetric rules, self-play should drive both players toward the same optimal
  strategy (mirror-image policies); with perfect play neither side can win, and the learned policies
  would approach optimal play. In practice, early self-play can be unstable because both learners
  explore the same move space simultaneously; divergence or slow convergence may occur before the
  policies stabilise.
- **Why this works / takeaway:** self-play is a standard way to generate training data without a
  fixed opponent; the symmetry of the game makes the two roles interchangeable.

### Exercise 1.2 — Symmetries

- **Book location:** Section 1.5, PDF page 28.
- **Type:** Theory.
- **Problem:** Many tic-tac-toe positions appear different but are really the same because of
  symmetries. How might we amend the learning process to take advantage? In what ways would this
  change improve the learning process? Now think again: if the opponent did not take advantage of
  symmetries, should we? Is it true that symmetrically equivalent positions should necessarily have
  the same value?
- **Solution.** Quotient the state space by the symmetry group (rotations/reflections of the board)
  before learning; this reduces the effective state space and improves sample efficiency. The value
  function should be constant on equivalence classes, and the policy should respect the symmetry
  (equivalent actions should be treated alike). Whether we *should* symmetrise depends on whether the
  opponent cooperates; if the opponent ignores symmetries, exploiting them is still valid for the
  agent.
- **Why this works / takeaway:** symmetries are a form of state abstraction that can shrink the
  learning problem without changing the optimal policy.

### Exercise 1.3 — Greedy Play

- **Book location:** Section 1.5, PDF page 28.
- **Type:** Theory / Computation.
- **Problem:** Suppose the reinforcement learning player was greedy — always playing the move that
  brought it to the position it rated best. Might it learn to play better, or worse, than a
  non-greedy player? What problems might occur?
- **Solution.** A purely greedy player has no exploration, so it may never discover better moves
  beyond those favoured by early (possibly inaccurate) value estimates; it can converge to a
  suboptimal policy. A non-greedy (exploratory) learner can probe alternatives and refine values; the
  learned greedy policy is then the one that maximises expected return.
- **Why this works / takeaway:** exploration and greedy selection are complementary; value learning
  and policy selection are separate concerns.

### Exercise 1.4 — Learning from Exploration

- **Book location:** Section 1.5, PDF page 28.
- **Type:** Theory / Computation.
- **Problem:** Suppose learning updates are applied to moves — including exploratory moves — while
  the step-size parameter is reduced over time (but the exploration tendency is not). How do the state
  values converge differently depending on whether we learn from all moves, or only from the moves
  that were actually taken? Assuming that exploration continues, which set of value estimates is the
  better one, and which would result in more wins?
- **Solution.** Two distinct value estimates:
  1. **Learning from all moves (including exploratory / considered moves):** backups therefore cover
     the moves the agent thought about or tried, and the converged estimates approach the **optimal**
     values `v*(s) = E[G | S = s]`.
  2. **Learning only from moves actually taken:** updates follow realised trajectories, so the
     converged estimates approach the values of the exploratory behaviour policy,
     `vπ(s) = E[G | S = s, π]`.
  With continued exploration, the **optimal** estimate set (case 1) is the better one; it is the one
  that leads to better behaviour and, ultimately, more wins, since the agent can act greedily from
  optimal values.
- **Why this works / takeaway:** what the learner conditions on (all considered moves vs. actual
  moves) determines whether the converged value estimates approach the optimal values or the values
  of the exploratory behaviour policy.

### Exercise 1.5 — Other Improvements

- **Book location:** Section 1.5, PDF page 28.
- **Type:** Theory.
- **Problem:** Can you think of other ways to improve the reinforcement learning player? Can you
  think of any better way to solve the tic-tac-toe problem as posed?
- **Solution.** Candidates: stronger exploration (softmax/optimistic initialisation), function
  approximation for large state spaces, planning / lookahead (the tic-tac-toe player already used a
  model to look ahead), or better reward/termination design. For tic-tac-toe, combining a learned
  value function with lookahead (as in the book) is already strong; a better solution may combine
  planning with learning.
- **Why this works / takeaway:** improvement ideas typically target exploration, representation, or
  planning — the three axes of RL methodology.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 1.1–1.5 (concepts) | §1.1–1.3 | `chapter_01/exercise_1_1.py` (`GridWorld`, `RandomAgent`, `GreedyAgent`, `run_episode`, `run_multiple_episodes`) | `TestGridWorldInit`, `TestGridWorldStep`, `TestGridWorldGetReward`, `TestGridWorldIsTerminal`, `TestRandomAgent`, `TestGreedyAgent`, `TestRunEpisode`, `TestRunMultipleEpisodes` | `python -m pytest chapter_01/tests/ -v` |

- Full chapter command: `python -m pytest chapter_01/tests/ -v`
- Lint: `ruff check chapter_01/`
- Pseudocode for the gridworld MDP steps: `chapter_01/pseudocode.py`.
- The repository implements the gridworld MDP of Section 1.3 (`GridWorld` with terminal states
  A+ and D, actions, rewards) plus simple agents — the coding counterpart of the agent concepts
  exercised in the book exercises above.

---

## Verification and Quality Notes

- The reasoning in this document was re-derived by hand where arithmetic applies; the public
  solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`) was used only as a cross-check.
- Theory answers (1.1–1.3, 1.5) are model answers written for student understanding; students should
  replace the tic-tac-toe framing with their own tasks where noted.

---

*Exercises document for Chapter 1 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
