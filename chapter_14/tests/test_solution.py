"""
Tests for Chapter 14: Psychology.

Reference: Sutton & Barto (2nd Ed), Chapter 14 (Psychology).

Tests for: Alg 14.1 (Rescorla-Wagner model, Eqs. 14.1-14.3),
Alg 14.2 (TD model of classical conditioning, Eqs. 14.4-14.6),
Alg 14.3 (Kamin blocking experiment runner).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_14.solution import (
    RescorlaWagnerModel,
    TDConditioningModel,
    csc_trial,
    excitation_curve,
    presence_trial,
    run_blocking_experiment,
    run_trial,
    train_presence,
)


class TestRescorlaWagner:
    """Tests for the Rescorla-Wagner model (Algorithm 14.1). Book Sec 14.2.2."""

    def test_excitation_curve_approaches_asymptote(self) -> None:
        model = RescorlaWagnerModel(1, alpha=0.1)
        history = excitation_curve(model, [1.0], 60)
        # Monotone approach from below, correct sign, asymptote at lambda = 1.
        assert np.all(np.diff(history) > 0.0)
        assert history[-1] == pytest.approx(1.0, abs=1e-2)
        assert abs(history[-1] - history[-2]) < abs(history[1] - history[0])

    def test_compound_cue_and_overshadowing(self) -> None:
        # More salient stimulus (3x alpha) captures more associative strength.
        model = RescorlaWagnerModel(2, alpha=0.2, saliences=[3.0, 1.0])
        for _ in range(80):
            model.trial([1.0, 1.0])
        assert model.strength(0) > model.strength(1)
        # Compound strength still respects the lambda asymptote.
        assert model.predict([1.0, 1.0]) == pytest.approx(1.0, abs=1e-2)

    def test_blocking_emerges(self) -> None:
        # Phase 1: A -> US until asymptote; Phase 2: A+B -> US.
        model = RescorlaWagnerModel(2, alpha=0.1)
        for _ in range(100):
            model.trial([1.0, 0.0])
        b_before = model.strength(1)
        for _ in range(100):
            model.trial([1.0, 1.0])
        assert model.strength(1) - b_before < 0.01

    def test_extinction_decays_to_zero(self) -> None:
        model = RescorlaWagnerModel(1, alpha=0.2)
        for _ in range(50):
            model.trial([1.0])
        conditioned = model.strength(0)
        assert conditioned > 0.9
        for _ in range(50):
            model.trial([1.0], 0.0)  # US omitted
        assert model.strength(0) == pytest.approx(0.0, abs=1e-2)

    def test_alpha_sensitivity(self) -> None:
        fast = RescorlaWagnerModel(1, alpha=0.4)
        slow = RescorlaWagnerModel(1, alpha=0.05)
        for _ in range(5):
            fast.trial([1.0])
            slow.trial([1.0])
        assert fast.strength(0) > slow.strength(0)

    def test_only_present_stimuli_update(self) -> None:
        model = RescorlaWagnerModel(2, alpha=0.1)
        model.trial([1.0, 0.0])
        assert model.strength(1) == 0.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            RescorlaWagnerModel(0)
        with pytest.raises(ValueError):
            RescorlaWagnerModel(1, alpha=0.0)
        with pytest.raises(ValueError):
            RescorlaWagnerModel(2, saliences=[1.0])
        model = RescorlaWagnerModel(1)
        with pytest.raises(ValueError):
            model.trial([1.0, 2.0])


class TestTDConditioningModel:
    """Tests for the TD model of classical conditioning (Algorithm 14.2).

    Book Section 14.2.3, Eqs. 14.4-14.6. Trials below follow the book's
    episodic assumption (Section 15.6): the trial ends right after the US
    prediction, so n_steps == reward_time and the terminal transition
    bootstraps v = 0.
    """

    def test_prediction_error_decreases_over_trials(self) -> None:
        model = TDConditioningModel(1, alpha=0.05, gamma=1.0, lam=0.0)
        history = train_presence(model, 100, 10, 0, 10, reward=1.0)
        early = float(np.abs(np.concatenate(history[:5])).mean())
        late = float(np.abs(np.concatenate(history[-5:])).mean())
        assert late < early

    def test_blocking_with_linear_features(self) -> None:
        # Phase 1: A -> US; Phase 2: compound A+B -> US.
        model = TDConditioningModel(2, alpha=0.05, gamma=1.0, lam=0.0)
        train_presence(model, 100, 10, 0, 10, reward=1.0, cs_index=0, n_stimuli=2)
        b_before = model.strength(1)
        train_presence(
            model, 100, 10, 0, 10, reward=1.0, cs_index=0, n_stimuli=2, compound=[1],
        )
        blocked_delta = model.strength(1) - b_before
        control = TDConditioningModel(2, alpha=0.05, gamma=1.0, lam=0.0)
        train_presence(
            control, 100, 10, 0, 10, reward=1.0, cs_index=0, n_stimuli=2, compound=[1],
        )
        assert blocked_delta < 0.05
        assert control.strength(1) > 10 * blocked_delta

    def test_csc_prediction_orders_toward_us(self) -> None:
        # With the CSC representation, the US prediction grows through the
        # interval and peaks at the time of the US (Figure 14.7, left).
        model = TDConditioningModel(11, alpha=0.05, gamma=0.97, lam=0.95)
        for _ in range(100):
            features, rewards, terminals = csc_trial(10, 0, 10, reward=1.0)
            run_trial(model, features, rewards, terminals)
        features, _, _ = csc_trial(10, 0, 10, reward=1.0)
        predictions = [model.predict(x) for x in features]
        assert predictions[9] > predictions[5] > predictions[0]
        assert int(np.argmax(predictions[:10])) == 9

    def test_kamin_blocking_case(self) -> None:
        # Kamin (1969) design: pre-trained A blocks B, vs. control (no
        # pre-training) learning both. Presence representation, lambda = 0
        # keeps the trace bookkeeping explicit (TD(0) special case).
        gamma, lam = 0.97, 0.5
        model = TDConditioningModel(2, alpha=0.05, gamma=gamma, lam=lam)
        train_presence(model, 100, 10, 0, 10, reward=1.0, cs_index=0, n_stimuli=2)
        b_before = model.strength(1)
        train_presence(
            model, 100, 10, 0, 10, reward=1.0, cs_index=0, n_stimuli=2, compound=[1],
        )
        blocked_delta = model.strength(1) - b_before
        control = TDConditioningModel(2, alpha=0.05, gamma=gamma, lam=lam)
        train_presence(
            control, 100, 10, 0, 10, reward=1.0, cs_index=0, n_stimuli=2, compound=[1],
        )
        assert blocked_delta < control.strength(1)
        assert control.strength(1) > 0.2

    def test_presence_trial_shapes_and_values(self) -> None:
        features, rewards, terminals = presence_trial(10, 0, 10, reward=1.0)
        assert len(features) == 11
        assert len(rewards) == len(terminals) == 10
        assert rewards[9] == pytest.approx(1.0)
        assert terminals[-1] is True
        assert all(terminals[k] is False for k in range(9))
        with pytest.raises(ValueError):
            presence_trial(10, 5, 5, reward=1.0)

    def test_only_present_features_get_traces(self) -> None:
        model = TDConditioningModel(2, alpha=0.05)
        features, rewards, terminals = presence_trial(4, 0, 4, reward=1.0, n_stimuli=2)
        run_trial(model, features, rewards, terminals)
        assert model.traces[1] == 0.0
        assert model.traces[0] > 0.0

    def test_gamma_zero_reduces_to_rescorla_wagner_style_update(self) -> None:
        # gamma = 0: delta = R - v (Rescorla-Wagner error, Eq. 14.3), so the
        # TD update collapses to the trial-level rule (modulo the one-step
        # lead in R and the trace bookkeeping).
        model = TDConditioningModel(1, alpha=0.1, gamma=0.0, lam=0.0)
        features, rewards, terminals = presence_trial(4, 0, 4, reward=1.0)
        deltas = run_trial(model, features, rewards, terminals)
        assert deltas[0] == pytest.approx(0.0)
        assert deltas[3] == pytest.approx(1.0)  # transition into US state

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            TDConditioningModel(0)
        with pytest.raises(ValueError):
            TDConditioningModel(1, alpha=-0.1)
        with pytest.raises(ValueError):
            TDConditioningModel(1, gamma=1.5)
        with pytest.raises(ValueError):
            TDConditioningModel(1, lam=-0.1)


class TestBlockingExperiment:
    """Tests for the Kamin blocking experiment runner (Algorithm 14.3).

    Book Sections 14.2.1-14.2.2. Wraps the Rescorla-Wagner model into a
    phase-1 / phase-2 paradigm producing the classic blocking profile.
    """

    def test_result_structure(self) -> None:
        result = run_blocking_experiment(phase1_trials=10, phase2_trials=10, replicates=4, seed=0)
        assert result["n_replicates"] == 4
        assert len(result["blocking_delta_b"]) == 4
        assert len(result["control_delta_b"]) == 4
        assert set(result) >= {
            "blocking_delta_b", "control_delta_b", "blocking_final_b", "control_final_b",
            "mean_blocking_delta", "mean_control_delta",
        }

    def test_blocking_delta_beats_control_delta(self) -> None:
        result = run_blocking_experiment(seed=0)
        assert result["mean_blocking_delta"] < result["mean_control_delta"]
        assert result["mean_control_delta"] > 0.4
        assert result["mean_blocking_delta"] < 0.1

    def test_blocking_scalic_statistically(self) -> None:
        # Seeded statistical check: blocking shows up in (almost) every
        # replicate, with US-magnitude noise on each trial.
        result = run_blocking_experiment(replicates=16, us_noise=0.1, seed=1)
        wins = sum(
            b < c for b, c in zip(result["blocking_delta_b"], result["control_delta_b"])
        )
        assert wins >= 14

    def test_deterministic_with_seed(self) -> None:
        first = run_blocking_experiment(replicates=4, seed=42)
        second = run_blocking_experiment(replicates=4, seed=42)
        assert first["blocking_delta_b"] == pytest.approx(second["blocking_delta_b"])
        assert first["control_delta_b"] == pytest.approx(second["control_delta_b"])

    def test_control_learns_b_without_phase1(self) -> None:
        result = run_blocking_experiment(phase1_trials=0, phase2_trials=40, seed=0)
        # No phase-1 pre-training: both groups are identical, both learn B.
        assert result["mean_blocking_delta"] == pytest.approx(
            result["mean_control_delta"], rel=1e-6,
        )

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            run_blocking_experiment(phase1_trials=-1)
        with pytest.raises(ValueError):
            run_blocking_experiment(replicates=0)
