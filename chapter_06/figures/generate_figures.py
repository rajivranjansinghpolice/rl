"""Regenerate all Chapter 6 figures.

Run from the repository root:

    python chapter_06/figures/generate_figures.py

Writes PNGs next to this script. Book figure identities refer to the
2nd (2018) edition; the in-progress draft numbers them differently.
All figures are illustrative demonstrations, not pixel-identical redraws.
"""

import os

from chapter_06.solution import (
    plot_example_6_2_random_walk,
    plot_figure_6_10_stochastic_wind,
    plot_figure_6_2_batch_updating,
    plot_figure_6_3_windy_gridworld,
    plot_figure_6_4_cliff_walking,
    plot_figure_6_6_td_control_performance,
    plot_figure_6_7_double_q_learning,
    plot_figure_6_9_king_moves,
)

HERE = os.path.dirname(os.path.abspath(__file__))

FIGURES = [
    ("example_6_2_random_walk_td_mc.png", plot_example_6_2_random_walk),
    ("fig_6_2_batch_updating.png", plot_figure_6_2_batch_updating),
    ("fig_6_3_windy_sarsa.png", plot_figure_6_3_windy_gridworld),
    ("fig_6_4_cliff_walking.png", plot_figure_6_4_cliff_walking),
    ("fig_6_6_td_control_performance.png", plot_figure_6_6_td_control_performance),
    ("fig_6_7_double_q_learning.png", plot_figure_6_7_double_q_learning),
    ("fig_6_9_king_moves.png", plot_figure_6_9_king_moves),
    ("fig_6_10_stochastic_wind.png", plot_figure_6_10_stochastic_wind),
]

for filename, plot_fn in FIGURES:
    path = os.path.join(HERE, filename)
    plot_fn(save_path=path)
    print("wrote", path)
