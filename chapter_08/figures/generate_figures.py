"""Regenerate all Chapter 8 figures.

Run from the repository root:

    python chapter_08/figures/generate_figures.py

File names use the figure numbers of Sutton & Barto, 2nd ed. (verified
against the local book PDF):

    fig_8_3  Dyna maze learning curves (Example 8.1)
    fig_8_4  Policies halfway through episode 2
    fig_8_5  Blocking maze (Example 8.2)
    fig_8_6  Shortcut maze (Example 8.3)
    fig_8_7  Backup diagrams for one-step updates
    fig_8_8  Expected vs sample update efficiency
    fig_8_9  Trajectory sampling
    fig_8_10 Heuristic search as one-step updates

The book prints no numbered figure for Examples 8.4-8.6; those outputs are
named ``example_8_X_*``. Reproductions are reduced-scale illustrations unless
the book's settings are explicitly matched.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_08.rod import plot_example_8_5_rod
from chapter_08.rtdp import plot_example_8_6_rtdp
from chapter_08.solution import (
    plot_example_8_4_scaling,
    plot_figure_8_10_heuristic_search,
    plot_figure_8_3_dyna_maze,
    plot_figure_8_4_policies,
    plot_figure_8_7_backup_diagrams,
    plot_figure_8_8_update_efficiency,
    plot_figure_8_9_trajectory_sampling,
    plot_changing_maze,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    jobs = [
        ("fig_8_3_dyna_q.png", lambda path: plot_figure_8_3_dyna_maze(
            save_path=path, runs=10, n_episodes=30)),
        ("fig_8_4_policies.png", lambda path: plot_figure_8_4_policies(save_path=path)),
        ("example_8_2_blocking_maze.png", lambda path: plot_changing_maze(
            "blocking", 1000, 4000, save_path=path, runs=5)),
        ("example_8_3_shortcut_maze.png", lambda path: plot_changing_maze(
            "shortcut", 3000, 6000, save_path=path, runs=5)),
        ("fig_8_7_backup_diagrams.png", lambda path: plot_figure_8_7_backup_diagrams(
            save_path=path)),
        ("fig_8_8_update_efficiency.png", lambda path: plot_figure_8_8_update_efficiency(
            save_path=path)),
        ("fig_8_9_trajectory_sampling.png", lambda path:
         plot_figure_8_9_trajectory_sampling(save_path=path, tasks=10)),
        ("fig_8_10_heuristic_search.png", lambda path:
         plot_figure_8_10_heuristic_search(save_path=path)),
        ("example_8_4_scaling.png", lambda path: plot_example_8_4_scaling(
            save_path=path, repeats=3, max_steps=1200)),
        ("example_8_5_rod.png", lambda path: plot_example_8_5_rod(save_path=path)),
        ("example_8_6_rtdp_racetrack.png", lambda path: plot_example_8_6_rtdp(
            save_path=path)),
    ]
    for filename, plot_fn in jobs:
        path = os.path.join(HERE, filename)
        plot_fn(path)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
