# Reinforcement Learning — Reference Implementations

Reference implementations of selected algorithms and exercises from Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (MIT Press, 2018): <https://mitpress.mit.edu/9780262039246/reinforcement-learning/>.

This repository is a trimmed public snapshot containing only the chapter implementations and this README.

## Chapter index

Each `chapter_XX/` directory is a self-contained Python package with its own tests and notes. See the chapter `README.md` for details, book references, and implementation caveats.

| Chapter directory | Contents (book mapping / notes) |
| --- | --- |
| `chapter_01/` | Introduction |
| `chapter_02/` | Multi-armed bandits |
| `chapter_03/` | Finite Markov decision processes |
| `chapter_04/` | Dynamic programming |
| `chapter_05/` | Monte Carlo methods |
| `chapter_06/` | Temporal-difference learning |
| `chapter_07/` | n-step bootstrapping |
| `chapter_08/` | Eligibility traces (hosts **Book Chapter 12** material in this repository's numbering) |
| `chapter_09/` | Planning and learning with tabular methods (hosts **Book Chapter 8** material in this repository's numbering) |
| `chapter_10/` | On-policy control with approximation |
| `chapter_11/` | Reserved for book Chapter 11 (off-policy methods with approximation); the former exploration content now lives in `chapter_02/` (bandits) |
| `chapter_12/` | Case studies / applications (derived exercises/notes; see `chapter_12/README.md`) |
| `chapter_13/` | Policy gradient methods |
| `chapter_14/` | Psychology (derived exercises/notes; see `chapter_14/README.md`) |
| `chapter_15/` | Neuroscience (derived exercises/notes; see `chapter_15/README.md`) |
| `chapter_16/` | Applications and case studies (derived exercises/notes; see `chapter_16/README.md`) |
| `chapter_17/` | Frontiers (derived exercises/notes; see `chapter_17/README.md`) |

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
