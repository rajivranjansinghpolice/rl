"""Regenerate all Chapter 7 figures.

Run from the repository root:

    python chapter_07/figures/generate_figures.py

Writes PNGs next to this script. Book figure identities refer to the
2nd (2018) edition. All figures are illustrative demonstrations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_07.solution import plot_figure_7_2

HERE = os.path.dirname(os.path.abspath(__file__))

FIGURES = [
    ("fig_7_2_n_step_td_alpha.png", plot_figure_7_2),
]

for filename, plot_fn in FIGURES:
    path = os.path.join(HERE, filename)
    plot_fn(save_path=path)
    print("wrote", path)
