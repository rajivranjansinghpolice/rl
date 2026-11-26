"""Regenerate Chapter 9 figures (book Chapter 9, On-policy Prediction).

Run from any directory:

    python chapter_09/figures/generate_figures.py

Figure numbers and captions were verified against the local 2nd-edition PDF.
Outputs are reduced-scale illustrative reproductions (100 states instead of
1000, fewer episodes), labelled accordingly in each plot.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib.pyplot as plt  # noqa: E402

from chapter_09.figure_experiments import (  # noqa: E402
    figure_9_1_data,
    figure_9_2_data,
    figure_9_5_data,
    figure_9_10_data,
)
from chapter_09.solution import plot_figure_9_wavelength  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def fig_9_1() -> None:
    d = figure_9_1_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(d["states"], d["true"], label="true $v_\\pi$", color="k")
    axes[0].plot(d["states"], d["approx"], label="MC estimate (aggregation)")
    axes[0].set_xlabel("State")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8)
    axes[0].set_title("True vs approximate value")
    axes[1].plot(d["states"], d["mu"])
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("$\\mu(s)$")
    axes[1].set_title("State distribution (empirical)")
    fig.suptitle("S&B (2nd ed.), Fig. 9.1: state aggregation, gradient Monte Carlo\n"
                 "illustrative reproduction (100 states; book: 1000)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(HERE, "fig_9_1_state_aggregation_mc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_9_2() -> None:
    d = figure_9_2_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(d["states"], d["true"], label="true $v_\\pi$", color="k")
    axes[0].plot(d["states"], d["approx"], label="TD estimate")
    axes[0].set_xlabel("State")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Asymptotic values (best n={d['best_n']})")
    for n, curve in d["curves"].items():
        axes[1].plot(d["alphas"], curve, "o-", markersize=3, label=f"n={n}")
    axes[1].set_xlabel("$\\alpha$")
    axes[1].set_ylabel("RMS error")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Performance vs step size")
    fig.suptitle("S&B (2nd ed.), Fig. 9.2: state aggregation with bootstrapping\n"
                 "illustrative reproduction (100 states; book: 1000)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(HERE, "fig_9_2_state_aggregation_td.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_9_5() -> None:
    d = figure_9_5_data()
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"fourier": "-", "polynomial": "--"}
    for (family, order), curve in d["curves"].items():
        ax.plot(range(1, len(curve) + 1), curve, styles[family],
                markersize=2, label=f"{family}, order {order}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error in value estimate")
    ax.legend(fontsize=8)
    ax.set_title("S&B (2nd ed.), Fig. 9.5: Fourier basis vs polynomials\n"
                 "gradient Monte Carlo, illustrative reproduction (100 states)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_9_5_fourier_vs_polynomial.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_9_10() -> None:
    d = figure_9_10_data()
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_tilings, curve in d["curves"].items():
        ax.plot(range(1, len(curve) + 1), curve,
                label=f"{n_tilings} tiling(s)", linewidth=1.5)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMS error in value estimate")
    ax.legend(fontsize=8)
    ax.set_title("S&B (2nd ed.), Fig. 9.10: single vs multiple tilings\n"
                 "gradient Monte Carlo, illustrative reproduction (100 states)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_9_10_tilings.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_9_1()
    fig_9_2()
    fig_9_5()
    fig_9_10()
    plot_figure_9_wavelength(
        save_path=os.path.join(HERE, "example_9_1_wavelength_fourier.png"))
    print("Chapter 9 figures written to", HERE, flush=True)


if __name__ == "__main__":
    main()
