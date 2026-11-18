"""Regenerate the Chapter 4 figures (run from the repository root).

Usage:
    python chapter_04/figures/generate_figures.py
"""
import matplotlib

matplotlib.use("Agg")

from chapter_04.solution import plot_gamblers_problem, plot_jacks_car_rental

if __name__ == "__main__":
    plot_gamblers_problem(save_path="chapter_04/figures/fig_4_3_gamblers_problem.png")
    plot_jacks_car_rental(save_path="chapter_04/figures/fig_4_2_jacks_car_rental.png")
    print(
        "Figures generated: fig_4_2_jacks_car_rental.png, "
        "fig_4_3_gamblers_problem.png"
    )
