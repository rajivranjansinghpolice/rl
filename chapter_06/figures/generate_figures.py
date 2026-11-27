"""Regenerate all Chapter 6 figures.

Run from the repository root:

    python chapter_06/figures/generate_figures.py

Writes PNGs next to this script. Book figure identities refer to the
2nd (2018) edition; the in-progress draft numbers them differently.
All figures are illustrative demonstrations, not pixel-identical redraws.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_06.solution import (
    plot_example_6_2_random_walk,
    plot_figure_6_2_batch_updating,
    plot_figure_6_3_windy_gridworld,
    plot_figure_6_4_cliff_walking,
    plot_figure_6_6_td_control_performance,
    plot_figure_6_7_double_q_learning,
    plot_figure_6_9_king_moves,
    plot_figure_6_10_stochastic_wind,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Reduced-scale settings keep the regeneration script usable; the plotting
# functions themselves retain their book-scale defaults when called directly.
FIGURES = [
    ("example_6_2_random_walk_td_mc.png",
     lambda p: plot_example_6_2_random_walk(save_path=p, repeats=30)),
    ("fig_6_2_batch_updating.png",
     lambda p: plot_figure_6_2_batch_updating(save_path=p, repeats=30, n_walks=50)),
    ("fig_6_3_windy_sarsa.png",
     lambda p: plot_figure_6_3_windy_gridworld(save_path=p, n_episodes=2000)),
    ("fig_6_4_cliff_walking.png",
     lambda p: plot_figure_6_4_cliff_walking(save_path=p, n_episodes=300)),
    ("fig_6_6_td_control_performance.png",
     lambda p: plot_figure_6_6_td_control_performance(
         save_path=p, n_episodes=1000, repeats=3)),
    ("fig_6_7_double_q_learning.png",
     lambda p: plot_figure_6_7_double_q_learning(
         save_path=p, n_episodes=200, repeats=100)),
    ("fig_6_9_king_moves.png",
     lambda p: plot_figure_6_9_king_moves(save_path=p, n_episodes=1000)),
    ("fig_6_10_stochastic_wind.png",
     lambda p: plot_figure_6_10_stochastic_wind(save_path=p, n_episodes=1500)),
]

def main() -> None:
    for filename, plot_fn in FIGURES:
        path = os.path.join(HERE, filename)
        plot_fn(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
