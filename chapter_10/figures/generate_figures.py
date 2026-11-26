"""Regenerate Chapter 10 figures (book Chapter 10, On-policy Control).

Run from any directory:

    python chapter_10/figures/generate_figures.py

Figure numbers/captions verified against the local 2nd-edition PDF. These are
reduced-scale illustrative reproductions (fewer episodes/steps than the book).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib.pyplot as plt  # noqa: E402

from chapter_10.figure_experiments import (  # noqa: E402
    figure_10_1_data,
    figure_10_2_data,
    figure_10_3_data,
    figure_10_4_data,
    figure_10_5_data,
)
from chapter_10.solution import (  # noqa: E402
    plot_figure_10_access_control,
    plot_figure_10_mountain_car,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def fig_10_1() -> None:
    data = figure_10_1_data(episodes=300)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (episode, (positions, velocities, grid)) in zip(
            axes.ravel(), sorted(data["snapshots"].items())):
        im = ax.pcolormesh(positions, velocities, grid.T, shading="auto",
                           cmap="viridis")
        ax.set_title(f"Episode {episode}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("S&B (2nd ed.), Fig. 10.1: Mountain Car cost-to-go "
                 "$-max_a q(s,a)$\nillustrative reproduction (task diagram omitted)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "fig_10_1_cost_to_go.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_10_2() -> None:
    data = figure_10_2_data(episodes=300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha, lengths in data["curves"].items():
        ax.plot(range(1, len(lengths) + 1), lengths,
                label=f"$\\alpha={alpha / 8:.3f}$")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("S&B (2nd ed.), Fig. 10.2: Mountain Car, semi-gradient Sarsa\n"
                 "illustrative reproduction (300 episodes)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_10_2_mountain_car_alpha.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_10_3() -> None:
    data = figure_10_3_data(episodes=300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, lengths in data["curves"].items():
        ax.plot(range(1, len(lengths) + 1), lengths, label=f"n={n}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("S&B (2nd ed.), Fig. 10.3: one-step vs n-step semi-gradient Sarsa\n"
                 "illustrative reproduction (300 episodes)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_10_3_nstep_mountain_car.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_10_4() -> None:
    data = figure_10_4_data(episodes=50)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n in enumerate(data["ns"]):
        ax.plot([a * 8 for a in data["alphas"]], data["grid"][i],
                "o-", label=f"n={n}")
    ax.set_xlabel("$\\alpha \\times$ number of tilings (8)")
    ax.set_ylabel("Steps per episode (first 50)")
    ax.legend(fontsize=8)
    ax.set_title("S&B (2nd ed.), Fig. 10.4: effect of $\\alpha$ and $n$ on\n"
                 "early Mountain Car performance (illustrative, reduced)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_10_4_alpha_n_grid.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_10_5() -> None:
    data = figure_10_5_data(n_steps=20_000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    im0 = axes[0].imshow(data["policy"], origin="lower", cmap="Set1",
                         interpolation="nearest")
    axes[0].set_title("Learned policy (0=reject, 1=high, 2=low)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(data["value"], origin="lower", cmap="viridis",
                         interpolation="nearest")
    axes[1].set_title("Value of best action")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_xlabel("Low-priority waiting")
        ax.set_ylabel("High-priority waiting")
    fig.suptitle("S&B (2nd ed.), Fig. 10.5: access-control queuing, differential "
                 "Sarsa\nillustrative reproduction (20k steps; book: 2M)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(HERE, "fig_10_5_access_control.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_10_1()
    fig_10_2()
    fig_10_3()
    fig_10_4()
    fig_10_5()
    plot_figure_10_mountain_car(
        save_path=os.path.join(HERE, "example_10_2_mountain_car.png"))
    plot_figure_10_access_control(
        save_path=os.path.join(HERE, "example_10_1_access_control.png"))
    print("Chapter 10 figures written to", HERE, flush=True)


if __name__ == "__main__":
    main()
