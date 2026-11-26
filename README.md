# Reinforcement Learning — Reference Implementations

Reference implementations of selected algorithms and exercises from Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (MIT Press, 2018): <https://mitpress.mit.edu/9780262039246/reinforcement-learning/>.

This repository is a trimmed public snapshot containing only the chapter implementations and this README.

## Chapter index

Each `chapter_XX/` directory is a self-contained Python package with its own tests and notes. See the chapter `README.md` for details, book references, and implementation caveats.

| Chapter directory | Contents (book mapping / notes) |
| --- | --- |
| `chapter_01/` | Introduction (Book Ch. 1) |
| `chapter_02/` | Multi-armed bandits; exploration/optimism (Book Ch. 2, plus folded exploration content formerly in old Ch. 11) |
| `chapter_03/` | Finite Markov decision processes (Book Ch. 3) |
| `chapter_04/` | Dynamic programming (Book Ch. 4) |
| `chapter_05/` | Monte Carlo methods (Book Ch. 5) |
| `chapter_06/` | Temporal-difference learning (Book Ch. 6) |
| `chapter_07/` | n-step bootstrapping (Book Ch. 7) |
| `chapter_08/` | Planning and learning with tabular methods (**Book Chapter 8** — moved from old `chapter_09/`) |
| `chapter_09/` | On-policy prediction with approximation (Book Ch. 9) |
| `chapter_10/` | On-policy control with approximation (Book Ch. 10) |
| `chapter_11/` | Off-policy methods with approximation (Book Ch. 11) |
| `chapter_12/` | Eligibility traces (**Book Chapter 12** — merged old `chapter_08/` (traces) + old `chapter_10/` (Q(λ) Watkins)) |
| `chapter_13/` | Policy gradient methods (Book Ch. 13) |
| `chapter_14/` | Psychology (Book Ch. 14) |
| `chapter_15/` | Neuroscience (Book Ch. 15) |
| `chapter_16/` | Applications and case studies (**Book Chapter 16** — merged old `chapter_16/` + old `chapter_12/` case studies) |
| `chapter_17/` | Frontiers (Book Ch. 17) |

> **Note:** All book chapters 1–17 are now scaffolded as `chapter_XX/` directories matching the book's chapter numbering.

## Known gaps and reproducibility notes

Every plot that claims a book figure number was checked against the local
2nd-edition PDF. Where the book does not print enough detail (cell coordinates,
reward models, step sizes, random seeds) a figure is labelled an
** illustrative demonstration**, not a pixel-level reproduction. The most
important deviations are:

- `chapter_08/`: the blocking/shortcut mazes and rod obstacles are
  reconstructions; κ and the trajectory-sampling reward model are not printed
  in the book. The blocking/shortcut plots reverse the book's phase-1 ordering
  on our small maze, so they are named `example_8_2_*`/`example_8_3_*`, **not**
  Fig. 8.5/8.6. Example 8.6 uses a small racetrack (707 states) instead of the
  book's 9,115-state track.
- `chapter_12/`: random-walk figures use 5 repetitions instead of 100; Fig. 12.14
  reproduces only two of four panels — the Cart/Pole and Puddle World panels
  are empty with an explicit notice (no fabricated data).
- `chapter_09/`–`chapter_11/`: figures 9.1/9.2/9.5/9.10, 10.1–10.5 and
  11.2/11.6/11.7 are now implemented at reduced scale (100 vs 1000 walk
  states, fewer Mountain Car episodes, expected rather than sampled updates
  for Baird's counterexample).
- `chapter_03/`–`chapter_16/`: several experiments use fewer runs/episodes
  than the printed book; see each chapter `README.md` for its exact settings.

`AUDIT.md` is the complete inventory of items that are **not implemented** or
**theory-only**, with a justification for each; it also lists what was
implemented during the book-alignment work. Exercise writeups in
`chapter_XX/exercises.md` follow the house style in
[`STUDENT_ANSWER_TEMPLATE.md`](STUDENT_ANSWER_TEMPLATE.md):
a conversational hook, a numbered formal equation with a symbol glossary, a
reasoning chain, an intuition check, and a prediction-style self-check. The
same guide's code standard governs `chapter_XX/solution.py`: one readable
function per algorithm, comments that explain why loops exist, and named
constants instead of magic numbers.

## Chapter renumbering provenance

To make `chapter_XX/` map to *Sutton & Barto (2nd ed.)* Chapter XX, the repository's historical section-based numbering was reconciled with the book:

- Old `chapter_09/` (Planning and Learning — **Book Ch. 8**) → now `chapter_08/`.
- Old `chapter_08/` (Eligibility Traces — **Book Ch. 12**) merged with old `chapter_10/` (Watkins's Q(λ) — book §12.10) → now `chapter_12/`.
- Old `chapter_11/` (Exploration — derived from book §§2.6–2.7) folded into `chapter_02/`.
- Old `chapter_12/` (Case studies — **Book Ch. 16**) merged into `chapter_16/`.



## Per-chapter layout

Chapters typically include:

- `README.md` — chapter overview, implemented algorithms/exercises, book references, and caveats
- `exercises.md` — exercise writeups/solutions where provided
- `pseudocode.py` — pseudocode with book-step mapping where provided
- `solution.py` and/or `exercise_*.py` — Python implementations
- `figures/` — generated plots reproducing or demonstrating book experiments/examples where feasible
- `tests/` — pytest test suite (some chapters also include `__init__.py`)

Not every chapter contains every file above; consult the chapter directory for its exact contents.

Shared environments and agents reused across chapters live in the tracked `rl_envs/` package (e.g. `BlackjackEnv`, `RandomWalkEnv`, `WindyGridWorld`, `GridWorld`, `Example41GridWorld`, `MountainCarEnv`, `TicTacToe` + MCTS, `PipRaceEnv`, the four-room GridWorld and option utilities). Chapter modules re-export the names they use, so existing imports keep working. Unit tests for the shared package live in `tests/test_rl_envs.py`.

Plot-generating helpers (e.g. `plot_*` / `figure_*` functions) are provided in the chapter implementation modules where applicable. They save figures when called with a `save_path` and include docstring/comments referencing the relevant Sutton & Barto (2nd ed.) sections/figures and any simplifications.

## Installation

Requires Python 3. Install dependencies from the repository root:

```bash
pip install numpy pytest ruff matplotlib
```

To reproduce experiment/example demonstration plots, also install `matplotlib`:

```bash
pip install matplotlib
```

## Running tests and linting

Run commands from the repository root so `chapter_XX` packages import correctly.

Per-chapter tests:

```bash
python -m pytest chapter_XX/tests/ -v
```

Per-chapter lint (uses the tracked `ruff.toml`):

```bash
python -m ruff check chapter_XX/
```

Examples:

```bash
python -m pytest chapter_02/tests/ -v
python -m pytest chapter_04/tests/ -v
python -m ruff check chapter_02/
```
