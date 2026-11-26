"""Regenerate all Chapter 12 (book Ch. 12, Eligibility Traces) figures.

Run from the repository root:

    python chapter_12/figures/generate_figures.py

Figure numbers follow the local 2nd-edition book PDF. All outputs are
reduced-scale illustrative reproductions (fewer repetitions / smaller tasks
than the book); the book prints no exact data files for these plots.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from chapter_12.figure_experiments import (  # noqa: E402
    LAMBDAS,
    mountain_car_lambda_curves,
    random_walk_lambda_curves,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _plot_random_walk_figures(path_prefix, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = np.array(curves["alphas"])
    for fig_no, name, label in (
        ("12_3", "offline", "off-line $\\lambda$-return algorithm"),
        ("12_6", "td", "TD($\\lambda$)"),
        ("12_8", "true_online", "on-line $\\lambda$-return / true online TD($\\lambda$)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for lam in LAMBDAS:
            err = np.clip(np.array(curves["curves"][name][lam]), 0, 1.2)
            ax.plot(alphas, err, "o-", markersize=3, label=f"$\\lambda={lam:g}$")
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("RMS error at the end of the episode")
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"S&B (2nd ed.), Fig. {fig_no.replace('_', '.')}: "
                     f"19-state random walk, {label}\n(illustrative, reduced scale)")
        fig.tight_layout()
        path = path_prefix + f"fig_{fig_no}_"
        path += {"12_3": "offline_lambda_return", "12_6": "td_lambda",
                 "12_8": "true_online_td_lambda"}[fig_no] + ".png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path, flush=True)


def _plot_mountain_car_figures(path_prefix, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(next(iter(curves.values()))) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, steps in curves.items():
        raw = np.array(steps)
        smooth = np.convolve(raw, np.ones(10) / 10, mode="valid")
        ax.plot(episodes, raw, color="0.85", linewidth=0.6)
        ax.plot(episodes[9:], smooth, label=name, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode")
    ax.legend(fontsize=8)
    ax.set_title("S&B (2nd ed.), Fig. 12.10: Sarsa($\\lambda$) with replacing traces\n"
                 "on Mountain Car (illustrative, tile-coded linear agent)")
    fig.tight_layout()
    path = path_prefix + "fig_12_10_sarsa_lambda_mountain_car.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(curves)
    final = [float(np.mean(list(curves[k])[-20:])) for k in labels]
    ax.bar(labels, final, color=["0.7", "0.5", "tab:blue", "tab:green"])
    ax.set_ylabel("Steps per episode (last 20)")
    ax.set_title("S&B (2nd ed.), Fig. 12.11: comparison of Sarsa($\\lambda$) variants\n"
                 "(illustrative; book includes True Online Sarsa($\\lambda$))")
    fig.tight_layout()
    path = path_prefix + "fig_12_11_sarsa_lambda_variants.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)


def _plot_lambda_summary(path_prefix, walk_curves, car_curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    best = []
    for lam in LAMBDAS:
        err = np.array(walk_curves["curves"]["td"][lam])
        best.append(np.nanmin(np.clip(err, 0, 1.2)))
    axes[0, 0].plot(LAMBDAS, best, "o-")
    axes[0, 0].set_title("19-state random walk: best TD($\\lambda$)")
    axes[0, 0].set_xlabel("$\\lambda$")
    axes[0, 0].set_ylabel("lowest RMS error")

    for name, steps in car_curves.items():
        axes[0, 1].plot(np.arange(1, len(steps) + 1), steps, label=name, linewidth=1)
    axes[0, 1].set_title("Mountain Car: steps per episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].legend(fontsize=7)

    # Two extra panels are illustrative stand-ins: the book's CartPole and
    # Puddle World tasks are not reproduced in this repository.
    axes[1, 0].plot(LAMBDAS,
                    [0.5 + 0.4 * (1 - lam) for lam in LAMBDAS], "s-",
                    color="tab:orange")
    axes[1, 0].set_title("Failures on a simple control task (illustrative)")
    axes[1, 0].set_xlabel("$\\lambda$")
    axes[1, 0].set_ylabel("failures / 100k steps")

    axes[1, 1].plot(LAMBDAS,
                    [0.3 + 0.5 * lam ** 2 for lam in LAMBDAS], "^-",
                    color="tab:purple")
    axes[1, 1].set_title("Cost on a simple grid task (illustrative)")
    axes[1, 1].set_xlabel("$\\lambda$")
    axes[1, 1].set_ylabel("cost per episode")

    fig.suptitle("S&B (2nd ed.), Fig. 12.14: effect of $\\lambda$ on performance\n"
                 "(only the random-walk and mountain-car panels are data-driven here)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = path_prefix + "fig_12_14_lambda_effects.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)


def main() -> None:
    prefix = HERE + os.sep
    walk = random_walk_lambda_curves(repeats=5, n_episodes=10, seed=0)
    _plot_random_walk_figures(prefix, walk)
    car = mountain_car_lambda_curves(episodes=200, seed=0)
    _plot_mountain_car_figures(prefix, car)
    _plot_lambda_summary(prefix, walk, car)


if __name__ == "__main__":
    main()
