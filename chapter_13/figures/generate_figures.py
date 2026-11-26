"""Regenerate the Chapter 13 figures (run from any directory)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_13.solution import plot_figure_13_1, plot_figure_13_2

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    plot_figure_13_1(save_path=os.path.join(HERE, "fig_13_1_short_corridor_value.png"))
    plot_figure_13_2(
        save_path=os.path.join(HERE, "fig_13_2_reinforce_short_corridor.png"),
        episodes=800, seed=0,
    )
    print("Figures generated in", HERE)
