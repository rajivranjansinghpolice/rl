# Implementation / Descope Audit

Sutton & Barto, _Reinforcement Learning: An Introduction_ (2nd ed.) reference
repository. Audit date: 2026-11-26.

This document is the single inventory of every item previously labelled
"not implemented", "theory-only" or lacking a coding pointer. Each item is
either **implemented** (with the API that now backs it) or **descoped** with
an explicit reason. The book is the sole source of truth: we never claim an
API reproduces an algorithm when it only illustrates part of it.

---

## Implemented during this audit

| Chapter | Item | API / evidence |
|---|---|---|
| Ch 8 §8.3 | Dyna-Q+ exploration bonus `r + κ√τ` | `dyna_q_plus` (bonus on modelled/stale pairs; untried pairs use the zero-reward self-loop default model without a bonus) |
| Ch 8 §8.3 | Blocking / shortcut maze experiments | `ChangingGridworld`, `changing_maze_experiment` |
| Ch 8 ex 8.4 | Bonus in action selection only | `agent="selection_bonus"` in `changing_maze_experiment` |
| Ch 8 §8.4 ex 8.5 | Rod maneuvering | `chapter_08/rod.py`: `RodWorld`, `prioritized_sweeping_known_model`, `value_iteration_known_model`, `shortest_path` |
| Ch 8 §8.6 | Trajectory sampling | `trajectory_sampling_experiment` |
| Ch 8 §8.7 ex 8.6 | RTDP vs value iteration on a racetrack | `chapter_08/rtdp.py` using the shared `rl_envs.racetrack` |
| Ch 8 figures | Verified Fig. 8.3/8.4/8.7–8.10; blocking & shortcut as `example_8_2_*`/`example_8_3_*` (not Fig. 8.5/8.6); Examples 8.4–8.6 | `chapter_08/figures/generate_figures.py` |
| Ch 12 figs 12.3/12.6/12.8/12.10/12.11/12.14 | Reduced-scale reproductions | `chapter_12/figure_experiments.py`, `chapter_12/figures/generate_figures.py` |
| All chapters | Student-friendly explanation after every exercise answer | `chapter_XX/exercises.md`, `**Explanation (student-friendly).**` markers |

---

## Descoped items (with justification)

### Chapter 5 — Monte Carlo Methods

| Item | Justification |
|---|---|
| Standalone `q_pi` first-visit prediction API | Prediction APIs estimate `v_pi`; action values are computed inside the control routines. Adding a redundant API would duplicate `mc_control_epsilon_greedy`'s averaging logic without new book content. |
| Separate incremental-API variant for weighted importance sampling | The incremental recurrence is used internally (`mc_off_policy_prediction`); exposing a second entry point adds no algorithm. |
| The exact discounted-weighting variant of Ex. 5.9 | The book asks for a derivation/experiment variant rather than a named algorithm; the standard ordinary/weighted IS estimators are implemented and the derivation is answered in `exercises.md`. |

### Chapter 6 — TD Learning

| Item | Justification |
|---|---|
| Part-of-speech / afterstates (§6.8) | Book gives a discussion, not an executable algorithm; no reference task is printed. |
| Double Expected Sarsa (§6.8) | Theory exercise with no pseudocode box; `double_q_learning` covers the related double-estimator idea. |
| Importance-sampling TD(0) prediction (Ex. 6.7) | The book asks for a derivation; off-policy IS *control* is in Ch. 5 and would be duplicated here. |

### Chapter 7 — n-step Methods

| Item | Justification |
|---|---|
| Off-policy n-step IS prediction/control APIs (Ex. 7.4–7.8) | Book supplies derivations/pseudocode only. The repo deliberately keeps its implemented scope to on-policy n-step methods; `exercises.md` gives complete pseudocode and a student-friendly explanation, and marks these as theory-only (no repo API). |
| Modified sum-of-δ online variant (Ex. 7.2) | Not the book's boxed algorithm; the exercise is a comparison question, answered analytically with a proposed experiment. |

### Chapter 9 — On-policy Prediction with Approximation

| Item | Justification |
|---|---|
| off-policy approximation, nonlinear/neural approximation, §9.6+ methods | Explicitly out of scope: those are Chapters 10–13 in the book. `chapter_09/README.md` records the scope decision. |

### Chapter 11 — Off-policy Methods with Approximation

| Item | Justification |
|---|---|
| Emphatic-TD verification machinery | The book discusses emphatic weighting without an executable algorithm suitable for this tabular/linear repository; the exercise answer derives the requirement and explains the verification idea. |

### Chapter 12 — Eligibility Traces

| Item | Justification |
|---|---|
| Dutch-trace **control** (Sarsa(λ)) | The book gives pseudocode discussion only; dutch/true-online **prediction** (`true_online_td_lambda`) is implemented, and the on-policy replacing/clearing Sarsa(λ) variants are implemented. |
| Off-policy IS/Expected-Sarsa trace evaluators (Ex. 12.7–12.13) | Pure algebraic derivations in the book. `watkins_lambda_return` / `watkins_q_lambda` are explicitly *not* presented as implementations of these algorithms. |
| Rigid Tree Backup (Ex. 12.15) | Book labels it a proposed programming exercise without a specification; implementing a guessed algorithm would violate the "never fabricate" rule. |
| Double Expected Sarsa(λ) (Ex. 12.16) | Theory extension with no pseudocode; the answer describes a sound construction. |
| True Online Sarsa(λ) on Mountain Car (Fig. 12.11) | Requires a full tile-coded actor implementation beyond the tabular/linear scope; the implemented variants and their comparison are labelled illustrative. |

### Chapters 14–17 — Psychology / Neuroscience / Applications / Frontiers

| Item | Justification |
|---|---|
| Figures 14.3–14.7 (Egger–Miller, facilitation, second-order conditioning) | The book reports experimental phenomena and reproduces figures from elsewhere; no executable algorithm or data is supplied. |
| Instrumental control, devaluation, cognitive maps beyond the implemented TD model | Book describes experiments qualitatively; no algorithm to implement. |
| Actor–critic / policy-gradient neural implementations in Ch. 15 | Book chapter is descriptive neuroscience; gradient algorithms are covered in Ch. 13. |
| TD-Gammon, checkers, Watson wagering, memory control (Ch. 16) | Case studies with no pseudocode; `rl_envs.pip_race` and the tic-tac-toe/MCTS demo provide the runnable analogues where the book gives one. |
| GVF / auxiliary tasks (Ch. 17) | Frontiers survey; no executable specification. |

---

## Known deviations from the printed figures

Documented per chapter in `chapter_XX/README.md`. Summary:

- Ch 8: maze/rod obstacle coordinates are not printed in the book, so layouts
  are illustrative reconstructions; κ and the trajectory-sampling reward model
  are not printed. On the small reconstructed blocking maze Dyna-Q+ is *worse*
  in phase 1 (the opposite of the book), so those outputs are named
  `example_8_2_*`/`example_8_3_*` and are never cited as Fig. 8.5/8.6.
  Example 8.6 uses a small racetrack (707 states) rather than the book's 9,115.
- Ch 12: reduced-scale random-walk and Mountain Car experiments (5 repetitions
  vs 100). Fig. 12.14 reproduces only two of the book's four panels; the
  Cart/Pole and Puddle World panels are empty with an explicit notice. Earlier
  formula-generated stand-ins were removed — no fabricated data is plotted.
- Ch 3/4/5/6/7/9/10/11/13: reduced episode/run counts; figures are labeled
  illustrative where exact book settings are unavailable.

## Verification

- `python -m pytest -q` — run from the repository root (all chapters and shared
  `rl_envs` tests).
- `python chapter_XX/figures/generate_figures.py` — regenerate a chapter's PNGs.
- Ruff is not available in this environment; style is kept PEP 8-compatible.
