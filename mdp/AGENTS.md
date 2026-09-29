# RL GridWorld Solver - Agent Guide

## Project Purpose
Reinforcement learning environment and algorithms for solving MDPs using classical dynamic programming: **Policy Iteration** and **Value Iteration**. Finds optimal policies in a grid world from terminal state to terminal state.

## Package Boundaries
- `gridworld_env.py` - Only source of truth for the environment (`GridWorldEnv`)
- `agents/` - Modular implementations of agents (preferred)
  - `policy_iteration.py` - Policy iteration algorithm
  - `value_iteration.py` - Value iteration algorithm
  - `__init__.py` - Exports both agents as functions
- `rl_gridworld.py` - Main entry point; handles CLI args and interactive mode
- Ignored/obsolete: `agents.py` (empty), `*.bak`, `*_fixed*.py`, `run_policy_iteration.py`

## Developer Commands
### Run main (default value iteration)
```bash
python rl_gridworld.py [--size N] [--gamma G] [--theta T] [--stochastic]
# or non-interactive test:
python test_run.py
```

### Run with custom arguments
```bash
python rl_gridworld.py --size 8 --gamma 0.95 --theta 1e-8
```

### Run single agent type
Edit `rl_gridworld.py` args or use:
```bash
# Force policy iteration only (edit sys.argv for test_run.py)
python test_run.py --selected 1
# Force value iteration only (default):
python test_run.py --selected 2
```

## Key Implementation Notes
- **Deterministic vs Stochastic**: Set `stochastic=True` to enable noisy actions (~80% intended, 10% left/right turn)
- **Interactive Mode**: When run without args, prompts for inputs with 5s timeout per prompt; defaults available on Enter
- **Terminal States**: Top-left `(0,0)` and bottom-right `((size-1), (size-1))` are absorbing terminals with reward 0
- **Reward Structure**: -1 per step toward terminal, 0 at terminal

## Testing
```bash
# Quick sanity check (non-interactive)
python test_run.py
```

### Avoid interactive stdin issues
Use `test_run.py` wrapper which sets `sys.argv` before calling main.

## Environment Quirks
- Stochastic actions can unexpectedly turn left/right 10% each when enabled
- Terminal states have no outgoing transitions (absorbing)
- Value iteration may take longer than policy iteration for sparse rewards
