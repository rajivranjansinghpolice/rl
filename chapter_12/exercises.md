# Chapter 12 — Case Studies: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_12`
> Book sections: **Chapter 16** — Applications and Case Studies (§16.1 TD-Gammon, §16.5–16.6); plus §1.5 (Tic-Tac-Toe)

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_12` — Case Studies (Tic-Tac-Toe environment and MCTS player
  `MCTSPlayer`, UCT selection `uct_score`, Pip Race racing game `PipRaceEnv`, TD-Gammon-style agent
  `TDGammonAgent`, linear value learning `semi_gradient_td_lambda_update`, match evaluation
  `evaluate_match` / `train_td_gammon`).
- **Book chapter:** Sections 16.1 (TD-Gammon), 16.5–16.6 (MCTS, AlphaGo) of Chapter 16 —
  Applications and Case Studies; plus the tic-tac-toe case study from Section 1.5.
- **Book sections covered:**
  - Section 16.1 — TD-Gammon (backgammon, TD(λ), nonlinear function approximation).
  - Section 16.5–16.6 — MCTS and AlphaGo (policy networks, value networks, rollout policy, MCTS).
  - Section 1.5 — Tic-Tac-Toe case study (RL player, lookahead via a model).
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Chapter 16 starts on PDF page
  367; the TD-Gammon section is on pages 367–372; the AlphaGo discussion is on pages 374–386; the
  tic-tac-toe example appears throughout Chapter 1 (e.g., PDF page 25/28).
- **Exercise origin:** the book has **no formal numbered exercise list** for these case-study
  sections; the prompts below are derived from the sections themselves and labeled **[derived]**.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 16 + §1.5. Section statements paraphrased; short verbatim references only.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-16 material, plus public
  applications/case-study write-ups. Any disagreement resolved by independent reasoning.
- All worked derivations/analyses are **rewritten from scratch**. Derived prompts in this document
  are paraphrased formulations of the cited case-study material (not verbatim book exercise text).

## How to Use This Document

- Each derived prompt lists: **Book location**, **Type** (`Theory`, `Math`, `Computation`,
  `Programming`), **Problem**, **Solution**, and — where a companion implementation exists — a
  **coding pointer** into `chapter_12/`.
- The final section maps derived prompts to repository code.

---

## Derived Exercises

### Derived 1 — Why TD-Gammon worked [derived]

- **Book location:** Section 16.1, PDF pages 367–372.
- **Type:** Theory.
- **Problem:** Tesauro's TD-Gammon, trained only by self-play with TD(λ) and nonlinear function
  approximation, reached near-grandmaster play in backgammon. Explain why backgammon was a good fit
  for TD learning and what specifically made the method work (features, network, rewards, extras).
- **Why this works / takeaway:** stochastic, complete-state games reward "on-line" value learning
  from every game; the outcome (+1/−1) is a natural terminal reward.

### Derived 2 — Feature-based vs full-neural value functions [derived]

- **Book location:** Section 16.1, PDF pages 367–372.
- **Type:** Theory / Computation.
- **Problem:** Explain the trade-off between hand-constructed features (e.g., blots, made points,
  pip counts) and a full neural network for representing the backgammon value function.
- **Why this works / takeaway:** good features compress what must be learned; full networks trade
  interpretability for capacity.

### Derived 3 — Pip Race as a TD(λ) case study [derived]

- **Book location:** Section 16.1, PDF pages 367–372.
- **Type:** Theory / Programming.
- **Problem:** Pip Race (a simplified backgammon race) is used in the repo as a TD(λ) training task.
  Describe the rules, the linear value function, and how the linear agent compares to the random
  baseline after training.
- **Why this works / takeaway:** small stochastic races let a linear value function be effectively
  learned from relatively few games.

### Derived 4 — AlphaGo's component networks [derived]

- **Book location:** Section 16.6, PDF pages 374–386.
- **Type:** Theory.
- **Problem:** Describe AlphaGo's three models (SL policy network, RL policy network, value network,
  plus rollout policy), how each was trained, and what role MCTS played with them.
- **Why this works / takeaway:** supervised learning gave fast candidate moves; RL self-play improved
  them; value evaluation guided search; rollouts let MCTS explore cheaply.

### Derived 5 — Why MCTS beats minimax in Go-like games [derived]

- **Book location:** Section 16.6, PDF pages 374–386.
- **Type:** Theory / Computation.
- **Problem:** Compare MCTS+UCT vs full-width minimax search given Go's effective branching factor
  of ~250–400; explain why the policy/value network guidance made MCTS feasible.
- **Why this works / takeaway:** selective search with good prior evaluations beats exhaustive search
  when branching factor is high.

### Derived 6 — UCT score derivation [derived]

- **Book location:** Section 16.6, PDF pages 374–386.
- **Type:** Math.
- **Problem:** Derive the UCT selection score (`v + c·sqrt(ln N / n)`) used in MCTS from the UCB1
  principle (Section 2.7, PDF pages 45–46). Explain what changes when leaf evaluations come from
  stochastic Monte Carlo rollouts (e.g. a default random rollout policy) rather than stationary
  bounded bandit rewards.
- **Why this works / takeaway:** UCT lifts bandit-style optimism into tree search; the c constant
  plays the same role as in UCB1.

### Derived 7 — Self-play from the tic-tac-toe RL player to MCTS [derived]

- **Book location:** Sections 1.5–1.6, PDF pages 25–35 · Section 16.6, PDF pages 374–386.
- **Type:** Theory.
- **Problem:** Revisit the tic-tac-toe self-play RL player from Chapter 1 (a tabular lookahead agent
  trained by TD).
  (a) Explain how the book's tic-tac-toe player learned an optimal strategy by self-play.
  (b) Explain how MCTS could solve tic-tac-toe from scratch using only the rules.
  (c) Decide which would be superior, and why, on a small vs large state space.
- **Why this works / takeaway:** model-free self-play and model-based tree search are complementary
  strategies for perfect-information games.

### Derived 8 — When a task demands a model vs when a model-free method suffices [derived]

- **Book location:** Sections 1.5–1.6, PDF pages 25–35 · Section 5.2, Section 8.1.
- **Type:** Theory.
- **Problem:** TD-Gammon used a model-free TD(λ) approach; AlphaGo used MCTS with models. Give a
  principled characterization of the tasks where each is favored.
- **Why this works / takeaway:** model-free excels when simulation is expensive but sampling is cheap;
  model-based search excels when a perfect or fast model exists.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 6–8 (tic-tac-toe) | §1.5 / §16.6 | `chapter_12/solution.py` (`TicTacToe`, `MCTSPlayer`, `uct_score`, `RandomPlayer`, `GreedyPlayer`, `play_game`) | `TestUCTScore`, `TestMCTS`, `TestTicTacToeRules` (supporting implementations; some conceptual analysis in D7–D8 is theory-only) | `python -m pytest chapter_12/tests/ -v` |
| Derived 3 | §16.1 | `chapter_12/solution.py` (`PipRaceEnv`, `pip_race_features`) | `TestPipRaceRules` | `python -m pytest chapter_12/tests/ -v` |
| Derived 1–2 | §16.1 | `chapter_12/solution.py` (`TDGammonAgent`, `linear_value`, `semi_gradient_td_lambda_update`, `train_td_gammon`, `evaluate_match`) | `TestTDLambdaUpdate`, `TestLinearValueFunction`, `TestTrainedAgent` (supporting; D1–D2 case-study analysis is largely theory-only) | `python -m pytest chapter_12/tests/ -v` |

- Full chapter command: `python -m pytest chapter_12/tests/ -v`
- Lint: `ruff check chapter_12/`

---

## Verification and Quality Notes

- Derived answers were re-derived independently; public case-study write-ups and PDFs were used only
  as cross-checks. The UCT derivation (Derived 6) follows the UCB1 argument of Section 2.7.

---

*Chapter 12 (Case Studies) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
