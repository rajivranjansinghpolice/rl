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
    """v_hat(w, x) = w . x (linear value function, Eq. 9.2).

    Book: Sutton & Barto (2nd ed.), Equation 9.2 (Chapter 9).

    Args:
        weights: See Equation 9.2 and the implementation for its role.
        x: See Equation 9.2 and the implementation for its role.

    Returns:
        linear value as defined in Equation 9.2."""
    # carry out the linear value step (book Chapter 9).
    return float(np.dot(np.asarray(weights, dtype=np.float64),
                        np.asarray(x, dtype=np.float64)))


def linear_gradient(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """grad_w v_hat(w, x) = x (linear features).

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        weights: See Section 9.3 and the implementation for its role.
        x: See Section 9.3 and the implementation for its role.

    Returns:
        linear gradient as defined in Section 9.3."""
    # carry out the linear gradient step (book Chapter 9).
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

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        weights: See Section 9.3 and the implementation for its role.
        x: See Section 9.3 and the implementation for its role.
        reward: See Section 9.3 and the implementation for its role.
        gamma: See Section 9.3 and the implementation for its role.
        x_next: See Section 9.3 and the implementation for its role.
        alpha: See Section 9.3 and the implementation for its role.

    Returns:
        semi gradient td0 update as defined in Section 9.3."""
    # carry out the semi gradient td0 update step (book Chapter 9).
    target = reward
    if x_next is not None:
        # target: add value computed from its operands into the running value (book Chapter 9).
        target += gamma * linear_value(weights, x_next)
    # delta: TD error or update difference for this step (book Chapter 9).
    delta = target - linear_value(weights, x)
    return np.asarray(weights, dtype=np.float64) + alpha * delta * x


def constant_alpha_mc_update(
    weights: np.ndarray,
    x: np.ndarray,
    g_return: float,
    alpha: float,
) -> np.ndarray:
    """Constant-alpha MC prediction update for a linear value function.

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        weights: See Section 9.3 and the implementation for its role.
        x: See Section 9.3 and the implementation for its role.
        g_return: See Section 9.3 and the implementation for its role.
        alpha: See Section 9.3 and the implementation for its role.

    Returns:
        constant alpha mc update as defined in Section 9.3."""
    # carry out the constant alpha mc update step (book Chapter 9).
    # delta: TD error or update difference for this step (book Chapter 9).
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

    Book: Sutton & Barto (2nd ed.), Section 9.4 (Chapter 9).

    Args:
        x: See Section 9.4 and the implementation for its role.
        order: See Section 9.4 and the implementation for its role.
        dimension: See Section 9.4 and the implementation for its role.

    Returns:
        fourier features as defined in Section 9.4."""

    # carry out the fourier features step (book Chapter 9).
    def _1d(x_scalar: float, order_: int) -> np.ndarray:
        """1d (reference implementation).


        Args:
            x_scalar: See the relevant book section and function body.
            order_: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9)."""
        # internal helper for the 1d step (book Chapter 9).
        return np.cos(np.pi * np.arange(0, order_ + 1, dtype=np.float64) * x_scalar)

    # x: state feature vector (book Chapter 9).
    x = float(np.clip(x, 0.0, 1.0))
    if dimension == 1:
        return _1d(x, order)
    # feat: array built from the supplied values (book Chapter 9).
    feat = np.array([1.0])
    # Iterate over _: one pass per index (book algorithm).
    for _ in range(dimension):
        # feat: result of np.outer(feat, _1d(x, order)).ravel(...) (book Chapter 9).
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

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9)."""

    # x range: tuple[float, float] attribute/state used by the algorithm (book Chapter 9).
    x_range: tuple[float, float] = (0.0, 1.0)
    # y range: tuple[float, float] attribute/state used by the algorithm (book Chapter 9).
    y_range: tuple[float, float] = (0.0, 1.0)
    # tiles per tiling: int attribute/state used by the algorithm (book Chapter 9).
    tiles_per_tiling: int = 8
    # n tilings: int attribute/state used by the algorithm (book Chapter 9).
    n_tilings: int = 4

    def __post_init__(self) -> None:
        """post init (reference implementation).

        Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9)."""
        # derive cached state after dataclass initialization (book Chapter 9).
        if self.n_tilings < 1 or self.tiles_per_tiling < 1:
            raise ValueError("n_tilings and tiles_per_tiling must be >= 1")
        self.n_features = self.n_tilings * self.tiles_per_tiling * self.tiles_per_tiling

    def features(self, x: float, y: float) -> np.ndarray:
        """features (reference implementation).


        Args:
            x: See the relevant book section and function body.
            y: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9)."""
        # feat: zero-filled array (book Chapter 9).
        feat = np.zeros(self.n_features, dtype=np.float64)
        # width: value computed from its operands (book Chapter 9).
        width = (self.x_range[1] - self.x_range[0]) / self.tiles_per_tiling
        # height: value computed from its operands (book Chapter 9).
        height = (self.y_range[1] - self.y_range[0]) / self.tiles_per_tiling
        for tiling in range(self.n_tilings):
            # Overlapping tilings are why a single point lights up several
            # features instead of one hard-coded bucket.
            idx = _tiling_feature_index(self, tiling, x, y, width, height)
            feat[idx] = 1.0
        return feat


def _tiling_feature_index(
    coder: TileCoder, tiling: int, x: float, y: float, width: float, height: float
) -> int:
    """Index of the one active tile in a shifted tiling.

    Each tiling is offset by a fraction of a tile width so the same point falls
    into different tiles across tilings; that overlap is what makes tile coding
    generalise between nearby states.

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        coder: See Section 9.3 and the implementation for its role.
        tiling: See Section 9.3 and the implementation for its role.
        x: See Section 9.3 and the implementation for its role.
        y: See Section 9.3 and the implementation for its role.
        width: See Section 9.3 and the implementation for its role.
        height: See Section 9.3 and the implementation for its role.

    Returns:
        tiling feature index as defined in Section 9.3."""
    # internal helper for the tiling feature index step (book Chapter 9).
    # ox: value computed from its operands (book Chapter 9).
    ox = tiling * (width / coder.n_tilings)
    # oy: value computed from its operands (book Chapter 9).
    oy = tiling * (height / coder.n_tilings)
    # ix: integer conversion (book Chapter 9).
    ix = int(np.floor((x - coder.x_range[0] + ox) / width))
    # iy: integer conversion (book Chapter 9).
    iy = int(np.floor((y - coder.y_range[0] + oy) / height))
    # ix: integer conversion (book Chapter 9).
    ix = int(np.clip(ix, 0, coder.tiles_per_tiling - 1))
    # iy: integer conversion (book Chapter 9).
    iy = int(np.clip(iy, 0, coder.tiles_per_tiling - 1))
    # base: value computed from its operands (book Chapter 9).
    base = tiling * coder.tiles_per_tiling * coder.tiles_per_tiling
    return base + iy * coder.tiles_per_tiling + ix


# ---------------------------------------------------------------------------
# Prediction objective (Section 9.2): mean squared value error
# ---------------------------------------------------------------------------


def mean_squared_value_error(
    weights: np.ndarray,
    states: np.ndarray,
    true_values: np.ndarray,
) -> float:
    """VE(w) = mean_s (v_pi(s) - v_hat(w, s))^2  (approximate on sampled states).

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        weights: See Section 9.3 and the implementation for its role.
        states: See Section 9.3 and the implementation for its role.
        true_values: See Section 9.3 and the implementation for its role.

    Returns:
        mean squared value error as defined in Section 9.3."""
    # carry out the mean squared value error step (book Chapter 9).
    # predictions: result of np.asarray(...) (book Chapter 9).
    predictions = np.asarray([linear_value(weights, s) for s in states], dtype=np.float64)
    # residuals: value computed from its operands (book Chapter 9).
    residuals = np.asarray(true_values, dtype=np.float64) - predictions
    return float(np.mean(residuals ** 2))


def rms_error(weights: np.ndarray, states: np.ndarray, true_values: np.ndarray) -> float:
    """Root-mean-squared value error used to plot learning curves.

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        weights: See Section 9.3 and the implementation for its role.
        states: See Section 9.3 and the implementation for its role.
        true_values: See Section 9.3 and the implementation for its role.

    Returns:
        rms error as defined in Section 9.3."""
    # carry out the rms error step (book Chapter 9).
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

    Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9)."""

    # true amplitude: float attribute/state used by the algorithm (book Chapter 9).
    true_amplitude: float = 1.0

    def _tile_coder(self) -> TileCoder:
        """tile coder (reference implementation).


        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9)."""
        # internal helper for the tile coder step (book Chapter 9).
        return TileCoder((0.0, 1.0), (0.0, 1.0), tiles_per_tiling=8, n_tilings=8)

    def features(self, x: float, order: int = 5, like: str = "fourier") -> np.ndarray:
        """features (reference implementation).


        Args:
            x: See the relevant book section and function body.
            order: See the relevant book section and function body.
            like: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9)."""
        # build the feature vector for the given state/observation (book Chapter 9).
        if like == "fourier":
            return fourier_features(x, order, dimension=1)
        if like == "tile":
            return self._tile_coder().features(x, x)
        raise ValueError(f"unknown feature family {like!r}")

    def feature_dim(self, order: int = 5, like: str = "fourier") -> int:
        """feature dim (reference implementation).


        Args:
            order: See the relevant book section and function body.
            like: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9)."""
        # carry out the feature dim step (book Chapter 9).
        if like == "fourier":
            return order + 1
        if like == "tile":
            return 8 * 8 * 8
        raise ValueError(f"unknown feature family {like!r}")

    def v_true(self, x: np.ndarray) -> np.ndarray:
        """v true (reference implementation).


        Args:
            x: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9)."""
        # carry out the v true step (book Chapter 9).
        return self.true_amplitude * np.cos(2.0 * np.pi * np.asarray(x))

    def sample_state(self, rng: np.random.Generator) -> float:
        """sample state (reference implementation).


        Args:
            rng: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9)."""
        # carry out the sample state step (book Chapter 9).
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

    Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9).

    Args:
        alpha: See Section 9.2 and the implementation for its role.
        n_steps: See Section 9.2 and the implementation for its role.
        seed: See Section 9.2 and the implementation for its role.
        order: See Section 9.2 and the implementation for its role.
        like: See Section 9.2 and the implementation for its role.

    Returns:
        run wavelength prediction as defined in Section 9.2."""
    # run the experiment and collect the figure data (book Chapter 9).
    # rng: seeded random generator so runs are reproducible (book Chapter 9).
    rng = np.random.default_rng(seed)
    # task: result of WavelengthTask(...) (book Chapter 9).
    task = WavelengthTask()
    # dim: result of task.feature_dim(...) (book Chapter 9).
    dim = task.feature_dim(order, like)
    # w: zero-filled array (book Chapter 9).
    w = np.zeros(dim, dtype=np.float64)

    # Iterate over _: one pass per time step (book algorithm).
    for _ in range(n_steps):
        # x: state feature vector (book Chapter 9).
        x = task.sample_state(rng)
        # feat: result of task.features(...) (book Chapter 9).
        feat = task.features(x, order, like)
        # w: result of constant_alpha_mc_update(...) (book Chapter 9).
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
    """RMS error of a fitted linear wave on a held-out uniform grid.

    Book: Sutton & Barto (2nd ed.), Section 9.3 (Chapter 9).

    Args:
        task: See Section 9.3 and the implementation for its role.
        weights: See Section 9.3 and the implementation for its role.
        order: See Section 9.3 and the implementation for its role.
        like: See Section 9.3 and the implementation for its role.
        n_samples: See Section 9.3 and the implementation for its role.
        seed: See Section 9.3 and the implementation for its role.

    Returns:
        evaluate approximation as defined in Section 9.3."""
    # carry out the evaluate approximation step (book Chapter 9).
    # rng: seeded random generator so runs are reproducible (book Chapter 9).
    rng = np.random.default_rng(seed)
    # xs: result of rng.uniform(...) (book Chapter 9).
    xs = rng.uniform(0.0, 1.0, size=n_samples)
    # states: all states in the MDP (book Chapter 9).
    states = np.array([task.features(float(x), order, like) for x in xs])
    return rms_error(weights, states, task.v_true(xs))


# Fixed settings for the illustrative figure: the fit is deliberately short
# and evaluated on a dense grid so the curves are easy to read.
_PLOT_FIT_ALPHA = 0.05
_PLOT_FIT_STEPS = 20000
_PLOT_GRID_POINTS = 200


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

    Book: Sutton & Barto (2nd ed.), Section 9.2 (Chapter 9).

    Args:
        save_path: See Section 9.2 and the implementation for its role.
        show: See Section 9.2 and the implementation for its role.
        seed: See Section 9.2 and the implementation for its role.
        orders: See Section 9.2 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 9).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # task: result of WavelengthTask(...) (book Chapter 9).
    task = WavelengthTask()
    # xs: array of linearly spaced values (book Chapter 9).
    xs = np.linspace(0.0, 1.0, _PLOT_GRID_POINTS)
    # true ys: result of task.v_true(...) (book Chapter 9).
    true_ys = task.v_true(xs)

    # (fig, axes): result of plt.subplots(...) (book Chapter 9).
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    # Iterate over (ax, order): paired entries (book algorithm).
    for ax, order in zip(axes.flat, orders):
        # (w, _): result of run_wavelength_prediction(...) (book Chapter 9).
        w, _ = run_wavelength_prediction(
            alpha=_PLOT_FIT_ALPHA, n_steps=_PLOT_FIT_STEPS, seed=seed,
            order=order, like="fourier"
        )
        # feats: array built from the supplied values (book Chapter 9).
        feats = np.array([task.features(float(x), order, "fourier") for x in xs])
        # preds: array built from the supplied values (book Chapter 9).
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
