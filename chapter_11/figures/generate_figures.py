"""Regenerate Chapter 11 figures (book Chapter 11, Off-policy Approximation).

Run from any directory:

    python chapter_11/figures/generate_figures.py

Figure numbers/captions verified against the local 2nd-edition PDF. The
Baird-counterexample panels use expected (full-distribution) updates rather
than sampled transitions, so they are labelled illustrative reproductions.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_11.figure_experiments import run_expected  # noqa: E402
from chapter_11.solution import (  # noqa: E402
    plot_figure_11_divergence,
    plot_figure_11_mspbe,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INIT = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])


def fig_11_2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode, label in (
        (axes[0], "semi_gradient_td", "semi-gradient off-policy TD"),
        (axes[1], "semi_gradient_dp", "semi-gradient DP (expected)"),
    ):
        traj = run_expected(mode, n_steps=300, alpha=0.01, init=INIT)
        ax.plot(np.linalg.norm(traj, axis=1))
        ax.set_yscale("log")
        ax.set_xlabel("Updates")
        ax.set_ylabel("$\\|w\\|$")
        ax.set_title(label)
    fig.suptitle("S&B (2nd ed.), Fig. 11.2: instability on Baird's counterexample\n"
                 "illustrative expected-update reproduction (book uses sampled updates)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(HERE, "fig_11_2_baird_divergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _component_plot(mode: str, filename: str, title: str) -> None:
    traj = run_expected(mode, n_steps=300, alpha=0.01, init=INIT)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i in range(traj.shape[1]):
        axes[0].plot(traj[:, i], label=f"$w_{i + 1}$")
    axes[0].set_xlabel("Updates")
    axes[0].set_ylabel("Weight component")
    axes[0].legend(fontsize=7)
    axes[1].plot(np.linalg.norm(traj, axis=1))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Updates")
    axes[1].set_ylabel("$\\|w\\|$")
    axes[1].set_title("Norm")
    fig.suptitle("S&B (2nd ed.), " + title + "\n"
                 "illustrative expected-update reproduction on Baird's star",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(HERE, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_11_2()
    _component_plot("tdc", "fig_11_6_tdc_baird.png", "Fig. 11.6: TDC on Baird's counterexample")
    _component_plot("emphatic", "fig_11_7_emphatic_td_baird.png",
                    "Fig. 11.7: one-step emphatic TD")
    plot_figure_11_divergence(
        save_path=os.path.join(HERE, "example_11_1_divergence.png"))
    plot_figure_11_mspbe(
        save_path=os.path.join(HERE, "example_11_2_mspbe.png"))
    print("Chapter 11 figures written to", HERE, flush=True)


if __name__ == "__main__":
    main()
