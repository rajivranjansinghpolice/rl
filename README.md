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
| `chapter_12/` | Eligibility traces (**Book Chapter 12** — merged old `chapter_08/` (traces) + old `chapter_10/` (Q(λ) Watkins)) |
| `chapter_13/` | Policy gradient methods (Book Ch. 13) |
| `chapter_14/` | Psychology (Book Ch. 14) |
| `chapter_15/` | Neuroscience (Book Ch. 15) |
| `chapter_16/` | Applications and case studies (**Book Chapter 16** — merged old `chapter_16/` + old `chapter_12/` case studies) |
| `chapter_17/` | Frontiers (Book Ch. 17) |

> **Note:** Book Chapters 9, 10, and 11 (on-policy prediction/control with approximation; off-policy methods with approximation) are not yet scaffolded as dedicated directories; they will occupy `chapter_09/`, `chapter_10/`, and `chapter_11/` once implemented.

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
pip install numpy pytest ruff
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

Per-chapter lint:

```bash
ruff check chapter_XX/
```

Examples:

```bash
python -m pytest chapter_02/tests/ -v
python -m pytest chapter_04/tests/ -v
ruff check chapter_02/
```
