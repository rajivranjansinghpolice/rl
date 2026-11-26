"""
Tests for Chapter 15: Neuroscience.

Reference: Sutton & Barto (2nd Ed), Chapter 15 (Neuroscience).

Tests for: Alg 15.1 (phasic dopamine / TD-error model of Section 15.6) and
Alg 15.2 (cue-reward conditioning trace: acquisition/extinction/reinsertion).
The TD error delta stands in for phasic dopamine activity
(Sections 15.3 and 15.6).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_15.solution import (
    PhasicDopamineModel,
    simulate_conditioning,
    simulate_conditioning_trace,
)


class TestPhasicDopamineModel:
    """Core model checks for the Section 15.6 task (Algorithm 15.1)."""

    def test_it_state_is_frozen(self) -> None:
        model = PhasicDopamineModel(5, alpha=0.1)
        model.run_trial(3, reward=1.0)
        assert model.values[0] == 0.0
        assert model.traces[0] == 0.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            PhasicDopamineModel(0)
        with pytest.raises(ValueError):
            PhasicDopamineModel(5, alpha=0.0)
        with pytest.raises(ValueError):
            PhasicDopamineModel(5, gamma=1.5)
        with pytest.raises(ValueError):
            PhasicDopamineModel(5, lam=-0.1)
        model = PhasicDopamineModel(5)
        with pytest.raises(ValueError):
            model.run_trial(6, reward=1.0)

    def test_untrained_prediction_error_at_us_only(self) -> None:
        model = PhasicDopamineModel(10, alpha=0.1)
        deltas = model.run_trial(5, reward=1.0)
        assert deltas[4] == pytest.approx(1.0)  # transition into US state
        assert np.all(np.delete(deltas, 4) == pytest.approx(0.0))


class TestPhasicDopamineTask:
    """Tests for Algorithm 15.1 — phasic dopamine / TD-error model.

    Book Section 15.6, Figure 15.5; Section 15.3 (reward prediction error
    hypothesis of dopamine neuron activity).
    """

    def test_error_peak_shifts_from_us_to_cs(self) -> None:
        result = simulate_conditioning(n_trials=50, us_index=15)
        # Early: the largest |delta| is at the US transition (index 14).
        assert result["peak_transition"][0] == result["us_index"] - 1
        # Late: the largest |delta| has moved to the CS transition (index 0).
        assert result["peak_transition"][-1] == 0

    def test_response_develops_at_cs_and_fades_at_us(self) -> None:
        result = simulate_conditioning(n_trials=50, us_index=15)
        assert result["cs_delta"][-1] > 0.8  # response at earliest CS
        assert abs(result["us_delta"][-1]) < 0.05  # predicted reward: slip of US response

    def test_reward_magnitude_coding(self) -> None:
        small = simulate_conditioning(n_trials=1, us_index=15, reward=1.0)
        large = simulate_conditioning(n_trials=1, us_index=15, reward=2.0)
        assert large["us_delta"][0] == pytest.approx(2.0 * small["us_delta"][0])
        assert large["cs_delta"][0] == 0.0

    def test_omitted_reward_gives_negative_error(self) -> None:
        result = simulate_conditioning(n_trials=60, us_index=15, omit_trials=(59,))
        assert result["us_delta"][-1] < -0.8
        assert result["deltas"][-1, result["us_index"] - 1] < 0.0

    def test_us_error_decreases_over_trials(self) -> None:
        result = simulate_conditioning(n_trials=50, us_index=15)
        early = float(np.abs(result["us_delta"][:5]).mean())
        late = float(np.abs(result["us_delta"][-5:]).mean())
        assert late < early
        assert late < 0.05

    def test_values_converge_to_reward_before_us(self) -> None:
        result = simulate_conditioning(n_trials=80, us_index=15)
        values = result["values"]
        # Every reward-predicting state ends up predicting the full return.
        assert values[1] > 0.8
        assert values[14] > 0.9
        # The rewarding state itself never predicts (Section 15.6).
        assert values[15] == pytest.approx(0.0, abs=0.1)


class TestConditioningTrace:
    """Tests for Algorithm 15.2 — cue-reward conditioning trace.

    Book Sections 15.5-15.6 (Figures 15.3-15.5). Verifies the event
    sequence: early learning, big response at the US; later learning,
    response at the CS and silence at the US; extinction, below-baseline
    dip at the expected US time; reinsertion, response reappears.
    """

    def test_event_sequence_acquisition(self) -> None:
        trace = simulate_conditioning_trace(us_index=15)
        boundary, _ = trace["boundaries"]
        # Early (first trial): big response at US, none at CS.
        assert trace["us_delta"][0] > 0.5
        assert abs(trace["cs_delta"][0]) < 0.05
        # Late: response at CS, none at US.
        late_us = trace["us_delta"][boundary - 5:boundary].mean()
        late_cs = trace["cs_delta"][boundary - 5:boundary].mean()
        assert late_cs > 0.5
        assert abs(late_us) < 0.1

    def test_sign_changes_at_both_events(self) -> None:
        trace = simulate_conditioning_trace(us_index=15)
        boundary, _ = trace["boundaries"]
        # CS response: near zero early, positive late (sign flip at CS event).
        assert trace["cs_delta"][:2].mean() < 0.1
        assert trace["cs_delta"][boundary - 5:boundary].mean() > 0.5
        # US response: positive early, near zero late (sign flip at US event).
        assert trace["us_delta"][:5].mean() > 0.5
        assert abs(trace["us_delta"][boundary - 5:boundary].mean()) < 0.1

    def test_extinction_inserts_negative_dip(self) -> None:
        trace = simulate_conditioning_trace(us_index=15)
        boundary, second = trace["boundaries"]
        extinction_us = trace["us_delta"][boundary:second]
        # First omitted-US trials dip clearly below baseline.
        assert extinction_us[:5].mean() < -0.3
        # CS response fades across extinction.
        assert trace["cs_delta"][boundary:second][-1] < trace["cs_delta"][boundary:second][0]

    def test_reinsertion_restores_response(self) -> None:
        trace = simulate_conditioning_trace(us_index=15)
        boundary, second = trace["boundaries"]
        reacquisition_us = trace["us_delta"][second:]
        # After extinction, CS -> US returns: US response goes positive again.
        assert reacquisition_us[0] > 0.3
        assert reacquisition_us[-1] >= 0.0

    def test_phase_labels_and_boundaries(self) -> None:
        n_acq, n_ext, n_reacq = 20, 10, 15
        trace = simulate_conditioning_trace(
            n_acquisition=n_acq, n_extinction=n_ext, n_reacquisition=n_reacq, us_index=15,
        )
        assert len(trace["phase"]) == n_acq + n_ext + n_reacq
        assert trace["phase"][:n_acq] == ["acquisition"] * n_acq
        assert trace["phase"][n_acq:n_acq + n_ext] == ["extinction"] * n_ext
        assert trace["phase"][n_acq + n_ext:] == ["reacquisition"] * n_reacq
        assert trace["boundaries"] == (n_acq, n_acq + n_ext)

    def test_invalid_phase_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            simulate_conditioning_trace(n_acquisition=0)
