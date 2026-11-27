"""Regenerate the Chapter 3 figures (run from the repository root).

Usage:
    python chapter_03/figures/generate_figures.py

Writes PNGs next to this script. Book figure identities refer to the
2nd (2018) edition. All figures are illustrative demonstrations, not pixel-identical redraws. """

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_03.solution import plot_recycling_robot_example

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    plot_recycling_robot_example(save_path=os.path.join(HERE, "fig_3_3_recycling_robot.png"))
    print("Figure generated: fig_3_3_recycling_robot.png")
