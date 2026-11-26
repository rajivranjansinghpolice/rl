"""Regenerate the Chapter 5 figures (run from the repository root).

Usage:
    python chapter_05/figures/generate_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_05.solution import (
    plot_blackjack_optimal,
    plot_blackjack_value_functions,
    plot_importance_sampling,
    plot_ordinary_is_unstable,
)

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    plot_blackjack_value_functions(
        save_path=os.path.join(HERE, "fig_5_1_blackjack_value.png")
    )
    plot_blackjack_optimal(
        save_path=os.path.join(HERE, "fig_5_2_blackjack_optimal.png"),
        n_episodes=500_000,
    )
    plot_importance_sampling(
        save_path=os.path.join(HERE, "fig_5_3_importance_sampling.png"),
        n_runs=100,
    )
    plot_ordinary_is_unstable(
        save_path=os.path.join(HERE, "fig_5_4_ordinary_is_unstable.png"),
        n_runs=10,
        max_episodes=1_000_000,
    )
    print(
        "Figures generated: fig_5_1_blackjack_value.png, "
        "fig_5_2_blackjack_optimal.png, fig_5_3_importance_sampling.png, "
        "fig_5_4_ordinary_is_unstable.png"
    )
