"""Tests for Chapter 9 - On-policy Prediction with Approximation."""
from __future__ import annotations

import numpy as np
import pytest

from chapter_09.solution import (
    TileCoder,
    WavelengthTask,
    constant_alpha_mc_update,
    fourier_features,
    linear_value,
    mean_squared_value_error,
    rms_error,
    run_wavelength_prediction,
    semi_gradient_td0_update,
)


class TestLinearValue:
    def test_dot_product(self) -> None:
        assert linear_value(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(11.0)

    def test_zero_weights(self) -> None:
        assert linear_value(np.zeros(4), np.ones(4)) == 0.0


class TestSemiGradientTD0:
    def test_known_answer_single_step(self) -> None:
        # w = [0.5, 0.0], x = [1, 0], r = 0, x' = [1, 0], gamma = 0.5
        # target = 0 + 0.5 * (w.x') = 0.5 * 0.5 = 0.25
        # delta = 0.25 - 0.5 = -0.25 ; w_new = w + alpha * delta * x
        w = np.array([0.5, 0.0])
        x = np.array([1.0, 0.0])
        out = semi_gradient_td0_update(w, x, reward=0.0, gamma=0.5, x_next=x, alpha=0.5)
        assert out[0] == pytest.approx(0.5 + 0.5 * (-0.25) * 1.0)

    def test_terminal_bootstrap_zero(self) -> None:
        w = np.array([1.0, 1.0])
        x = np.array([1.0, 0.0])
        out = semi_gradient_td0_update(w, x, reward=2.0, gamma=0.99, x_next=None, alpha=0.1)
        # target = 2.0 ; delta = 2.0 - 1.0 = 1.0 ; w = w + 0.1*1.0*[1,0]
        assert out[0] == pytest.approx(1.0 + 0.1 * 1.0 * 1.0)

    def test_alpha_zero_leaves_weights_unchanged(self) -> None:
        w = np.array([1.0, 1.0])
        x = np.array([1.0, 0.0])
        out = semi_gradient_td0_update(w, x, 5.0, 0.5, None, alpha=0.0)
        assert np.allclose(out, w)


class TestConstantAlphaMC:
    def test_known_answer(self) -> None:
        w = np.array([0.0, 0.0])
        x = np.array([1.0, 0.0])
        out = constant_alpha_mc_update(w, x, g_return=5.0, alpha=0.1)
        assert out[0] == pytest.approx(0.1 * 5.0)


class TestFourierFeatures:
    def test_shape_and_range(self) -> None:
        feat = fourier_features(0.5, order=4)
        assert feat.shape == (5,)
        assert np.all(feat <= 1.0)
        assert np.all(feat >= -1.0)

    def test_k_zero_is_one(self) -> None:
        assert fourier_features(0.3, order=3)[0] == pytest.approx(1.0)

    def test_evenly_spaced_wave(self) -> None:
        feat = fourier_features(0.0, order=6)
        assert np.allclose(feat, np.ones(7))


class TestTileCoder:
    def test_binary_and_dim(self) -> None:
        coder = TileCoder(tiles_per_tiling=4, n_tilings=2)
        feat = coder.features(0.5, 0.5)
        assert feat.shape == (coder.n_features,)
        assert set(feat) <= {0.0, 1.0}
        assert np.sum(feat) == coder.n_tilings

    def test_off_grid_clipped(self) -> None:
        coder = TileCoder(tiles_per_tiling=4, n_tilings=2)
        feat = coder.features(-1.0, 2.0)
        assert np.sum(feat) == coder.n_tilings


class TestMeanSquaredValueError:
    def test_zero_at_perfect_fit(self) -> None:
        states = np.array([[1.0, 0.0], [0.0, 1.0]])
        true_values = np.array([2.0, 3.0])
        w = np.array([2.0, 3.0])
        assert mean_squared_value_error(w, states, true_values) == pytest.approx(0.0)

    def test_rms_is_sqrt_of_mse(self) -> None:
        states = np.array([[1.0, 0.0]])
        true_values = np.array([0.0])
        w = np.array([1.0, 0.0])
        assert rms_error(w, states, true_values) == pytest.approx(1.0)


class TestWavelengthTask:
    def test_true_values_bounded(self) -> None:
        task = WavelengthTask()
        assert np.all(np.abs(task.v_true(np.linspace(0, 1, 50))) <= 1.0)

    def test_features_shapes(self) -> None:
        task = WavelengthTask()
        f = task.features(0.5, order=3, like="fourier")
        assert f.shape == (4,)
        t = task.features(0.5, like="tile")
        assert np.sum(t) == 8  # 8 tilings

    def test_run_prediction_finite(self) -> None:
        w, _ = run_wavelength_prediction(
            alpha=0.05, n_steps=2000, seed=0, order=3, like="fourier"
        )
        assert np.all(np.isfinite(w))

    def test_higher_order_fits_wave_better(self) -> None:
        low, _ = run_wavelength_prediction(
            alpha=0.05, n_steps=40000, seed=0, order=0, like="fourier"
        )
        high, task = run_wavelength_prediction(
            alpha=0.05, n_steps=40000, seed=0, order=5, like="fourier"
        )
        xs = np.linspace(0.0, 1.0, 200)
        low_feats = np.array([task.features(float(x), 0, "fourier") for x in xs])
        high_feats = np.array([task.features(float(x), 5, "fourier") for x in xs])
        true_values = task.v_true(xs)
        from chapter_09.solution import linear_value

        low_err = np.sqrt(np.mean(
            (true_values - np.array([linear_value(low, f) for f in low_feats])) ** 2))
        high_err = np.sqrt(np.mean(
            (true_values - np.array([linear_value(high, f) for f in high_feats])) ** 2))
        assert high_err < low_err
