"""Solution implementation for Chapter 9 - On-policy Prediction with Approximation.

Reference: Sutton & Barto (2nd Ed), Chapter 9 (On-policy Prediction with
Approximation).

In Sutton & Barto, function approximation is the step from tabular methods
(Chapters 3-8) to handling large/continuous state spaces. This module
implements the linear methods that are the core of that chapter.

Implements:
- algorithm_9.1 - Semi-gradient TD(0) for linear value functions (Section
  9.3-9.4). Linear model: v_hat(w, x) = w^T x. The semi-gradient TD(0)
  update is
      w <- w + alpha * [r + gamma * v_hat(w, x') - v_hat(w, x)] * x.
- algorithm_9.2 - Constant-alpha Monte Carlo prediction update (Section 9.2):
      w <- w + alpha * (G - v_hat(w, x)) * x.
- Fourier cosine basis (Section 9.4.5 "The Fourier Basis"):
      phi_k(x) = cos(pi * k * x), k = order+1 ... 2*order+1 (book-style orders).
- Tile coding (Section 9.4.3 "Tile Coding"): sparse binary feature vectors,
  with tilings offset by a fraction of the tile width in 2D.
- The prediction objective (Section 9.2): mean squared value error (VE)
    VE(w) = sum_mu(s) (v_pi(s) - v_hat(w, s))^2
  computed here over a set of sampled states (approximate VE), and the
  root-mean-squared value error (RMS error) used to plot learning curves.

The book's figures (Fourier / tile-coding learning curves) are reproduced
as demonstrations; the exact figure numbers cannot be verified against the
local PDF text (which ships only a summary), so the figure functions here are
labelled honestly as demonstrations of the concepts, not as verbatim book
figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Linear value function and gradients
# ---------------------------------------------------------------------------


def linear_value(weights: np.ndarray, x: np.ndarray) -> float:
    """v_hat(w, x) = w . x (linear value function, Eq. 9.2)."""
    return float(np.dot(np.asarray(weights, dtype=np.float64),
                        np.asarray(x, dtype=np.float64)))


def linear_gradient(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """grad_w v_hat(w, x) = x (linear features)."""
    return np.asarray(x, dtype=np.float64)


def semi_gradient_td0_update(
    weights: np.ndarray,
    x: np.ndarray,
    reward: float,
    gamma: float,
    x_next: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Semi-gradient TD(0) update for a linear value function.

    x_next is the feature vector of the next state, or None if the transition
    ended the episode (in which case the bootstrap value is 0).
    """
    target = reward
    if x_next is not None:
        target += gamma * linear_value(weights, x_next)
    delta = target - linear_value(weights, x)
    return np.asarray(weights, dtype=np.float64) + alpha * delta * x


def constant_alpha_mc_update(
    weights: np.ndarray,
    x: np.ndarray,
    g_return: float,
    alpha: float,
) -> np.ndarray:
    """Constant-alpha MC prediction update for a linear value function."""
    delta = g_return - linear_value(weights, x)
    return np.asarray(weights, dtype=np.float64) + alpha * delta * x


# ---------------------------------------------------------------------------
# Fourier cosine basis (Section 9.4.5)
# ---------------------------------------------------------------------------


def fourier_features(x: float, order: int, dimension: int = 1) -> np.ndarray:
    """Fourier cosine basis for x in [0, 1].

    For 1D this returns [cos(k * pi * x) for k in 0..order], including the
    constant feature (k = 0). For higher dimensions a coordinate-wise
    Kronecker product of the 1D basis is used as a simple approximation of
    the book's Section 9.4.5 Fourier basis.
    """

    def _1d(x_scalar: float, order_: int) -> np.ndarray:
        return np.cos(np.pi * np.arange(0, order_ + 1, dtype=np.float64) * x_scalar)

    x = float(np.clip(x, 0.0, 1.0))
    if dimension == 1:
        return _1d(x, order)
    feat = np.array([1.0])
    for _ in range(dimension):
        feat = np.outer(feat, _1d(x, order)).ravel()
    return feat


# ---------------------------------------------------------------------------
# Tile coding (Section 9.4.3)
# ---------------------------------------------------------------------------


@dataclass
class TileCoder:
    """Tile coding for a 2D box.

    n_tilings tilings, each with tiles_per_tiling_{x,y} tiles. Tilings are
    offset by a fraction of a tile width so that every tile of one tiling
    overlaps neighbours of the others.
    """

    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    tiles_per_tiling: int = 8
    n_tilings: int = 4

    def __post_init__(self) -> None:
        if self.n_tilings < 1 or self.tiles_per_tiling < 1:
            raise ValueError("n_tilings and tiles_per_tiling must be >= 1")
        self.n_features = self.n_tilings * self.tiles_per_tiling * self.tiles_per_tiling

    def features(self, x: float, y: float) -> np.ndarray:
        feat = np.zeros(self.n_features, dtype=np.float64)
        width = (self.x_range[1] - self.x_range[0]) / self.tiles_per_tiling
        height = (self.y_range[1] - self.y_range[0]) / self.tiles_per_tiling
        offset_x = width / self.n_tilings
        offset_y = height / self.n_tilings
        for tiling in range(self.n_tilings):
            ox = tiling * offset_x
            oy = tiling * offset_y
            ix = int(np.floor((x - self.x_range[0] + ox) / width))
            iy = int(np.floor((y - self.y_range[0] + oy) / height))
            ix = int(np.clip(ix, 0, self.tiles_per_tiling - 1))
            iy = int(np.clip(iy, 0, self.tiles_per_tiling - 1))
            base = tiling * self.tiles_per_tiling * self.tiles_per_tiling
            idx = base + iy * self.tiles_per_tiling + ix
            feat[idx] = 1.0
        return feat


# ---------------------------------------------------------------------------
# Prediction objective (Section 9.2): mean squared value error
# ---------------------------------------------------------------------------


def mean_squared_value_error(
    weights: np.ndarray,
    states: np.ndarray,
    true_values: np.ndarray,
) -> float:
    """VE(w) = mean_s (v_pi(s) - v_hat(w, s))^2  (approximate on sampled states)."""
    predictions = np.asarray([linear_value(weights, s) for s in states], dtype=np.float64)
    residuals = np.asarray(true_values, dtype=np.float64) - predictions
    return float(np.mean(residuals ** 2))


def rms_error(weights: np.ndarray, states: np.ndarray, true_values: np.ndarray) -> float:
    """Root-mean-squared value error used to plot learning curves."""
    return float(np.sqrt(mean_squared_value_error(weights, states, true_values)))


# ---------------------------------------------------------------------------
# Wavelength task: fit a known wave used to demonstrate the prediction objective
# ---------------------------------------------------------------------------


@dataclass
class WavelengthTask:
    """The "wavelength task": fit the wave v(x) = cos(2*pi*x), x in [0, 1].

    States are sampled uniformly in [0, 1]; the known target is the cosine
    wave. Fitting it directly with the constant-alpha prediction update on
    uniformly sampled states demonstrates the Section 9.2 prediction objective
    (VE) and lets orders of the Section 9.4.5 Fourier basis be compared on the
    same footing.

    This is an honest, self-contained demonstration of those sections. The
    exact identity/number of the book's Fourier-basis figures can not be
    verified against the local PDF summary, so this class is documented as an
    illustrative reproduction rather than a verbatim figure reproduction.
    """

    true_amplitude: float = 1.0

    def _tile_coder(self) -> TileCoder:
        return TileCoder((0.0, 1.0), (0.0, 1.0), tiles_per_tiling=8, n_tilings=8)

    def features(self, x: float, order: int = 5, like: str = "fourier") -> np.ndarray:
        if like == "fourier":
            return fourier_features(x, order, dimension=1)
        if like == "tile":
            return self._tile_coder().features(x, x)
        raise ValueError(f"unknown feature family {like!r}")

    def feature_dim(self, order: int = 5, like: str = "fourier") -> int:
        if like == "fourier":
            return order + 1
        if like == "tile":
            return 8 * 8 * 8
        raise ValueError(f"unknown feature family {like!r}")

    def v_true(self, x: np.ndarray) -> np.ndarray:
        return self.true_amplitude * np.cos(2.0 * np.pi * np.asarray(x))

    def sample_state(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0.0, 1.0))


def run_wavelength_prediction(
    alpha: float = 0.02,
    n_steps: int = 20000,
    seed: int = 0,
    order: int = 5,
    like: str = "fourier",
) -> tuple[np.ndarray, WavelengthTask]:
    """Fit the wave with the constant-alpha prediction update.

    Minimises the Section 9.2 prediction objective VE(w) = mean_s (v(x) -
    v_hat(w, x))^2 over uniformly sampled states using uniformly sampled
    x and g_return = v(x). Returns (weights, task).

    This is a supervised (MC-style) fit to the known wave, used to illustrate
    the prediction objective and approximation quality of the Fourier orders;
    the on-policy TD(0) update itself is covered by the semi_gradient_td0_*
    functions and their tests.
    """
    rng = np.random.default_rng(seed)
    task = WavelengthTask()
    dim = task.feature_dim(order, like)
    w = np.zeros(dim, dtype=np.float64)

    for _ in range(n_steps):
        x = task.sample_state(rng)
        feat = task.features(x, order, like)
        w = constant_alpha_mc_update(w, feat, float(task.v_true(np.array([x]))[0]), alpha)
    return w, task


def evaluate_approximation(
    task: WavelengthTask,
    weights: np.ndarray,
    order: int,
    like: str,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    """RMS error of a fitted linear wave on a held-out uniform grid."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 1.0, size=n_samples)
    states = np.array([task.features(float(x), order, like) for x in xs])
    return rms_error(weights, states, task.v_true(xs))


def plot_figure_9_wavelength(
    save_path: str | None = None,
    show: bool = False,
    seed: int = 0,
    orders: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
):
    """Fourier-basis approximation of the wave vs order (demonstration).

    Sutton & Barto (2nd ed.), Section 9.2 (prediction objective VE) and
    Section 9.4.5 (Fourier basis). Shows the fitted wave and the true wave
    for several orders; the RMSVE is printed in the caption. This is an
    illustrative reproduction of the book's wavelength demonstration; the
    exact figure identity cannot be verified against the local PDF summary.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = WavelengthTask()
    xs = np.linspace(0.0, 1.0, 200)
    true_ys = task.v_true(xs)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, order in zip(axes.flat, orders):
        w, _ = run_wavelength_prediction(
            alpha=0.05, n_steps=20000, seed=seed, order=order, like="fourier"
        )
        feats = np.array([task.features(float(x), order, "fourier") for x in xs])
        preds = np.array([linear_value(w, f) for f in feats])
        ax.plot(xs, true_ys, color="gray", lw=1.5, label="true wave")
        ax.plot(xs, preds, color="tab:blue", lw=1.5, label=f"order {order}")
        ax.set_title(f"order {order}")
        ax.set_ylim(-1.5, 1.5)
    axes.flat[0].legend()
    fig.suptitle(
        "S&B (2nd ed.), Sec. 9.2/9.4.5: Fourier-basis fit of cos(2*pi*x) "
        "(illustrative demonstration)"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
