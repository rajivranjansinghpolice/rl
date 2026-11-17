"""
Solution for Chapter 1 — Gridworld MDP Implementation

Convenience wrapper that imports from exercise_1_1.py for compatibility.

Book Reference: Sutton & Barto (2nd Ed.), Section 1.3, Figure 1.2
"""

from __future__ import annotations
import sys
import os

# Ensure current directory is on path for relative import
sys.path.insert(0, os.path.dirname(__file__))

from exercise_1_1 import (
    GridWorld,
    RandomAgent,
    GreedyAgent,
    plot_gridworld_figure,
    run_episode,
    run_multiple_episodes,
    main,
)

__all__ = [
    "GridWorld",
    "RandomAgent",
    "GreedyAgent",
    "plot_gridworld_figure",
    "run_episode",
    "run_multiple_episodes",
    "main",
]
