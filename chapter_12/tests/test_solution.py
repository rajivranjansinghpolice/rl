"""
Tests for Chapter 12: Eligibility Traces (book Chapter 12).

Reference: Sutton & Barto (2nd Ed), Chapter 12.

Tests for: algorithm_8.1 (TD(lambda) with replacing traces),
algorithm_8.2 (clearing traces), algorithm_8.3 (true online TD(lambda)).
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter_06.solution import (
    RandomWalkEnv,
    q_learning,
    random_walk_true_values,
    td0_prediction,
)
from chapter_06.solution import (
    sarsa as sarsa_one_step,
)
from chapter_06.solution import (
    td_lambda_prediction as td_lambda_accumulating_ch06,
)
from chapter_12.solution import (
    sarsa_lambda_clearing,
    sarsa_lambda_replacing,
    td_lambda_prediction,
    true_online_td_lambda,
    watkins_lambda_return,
    watkins_q_lambda,
)


class TestTDLambdaPrediction:
    """State-value TD(lambda), Sections 12.2/12.5. Algorithm 8.1 (state side)."""

    def test_lambda_zero_matches_td0(self) -> None:
        # lambda = 0 reduces to TD(0) math (Eq. 12.5/12.17).
        lam_zero = td_lambda_prediction(RandomWalkEnv(seed=0), lam=0.0,
                                        n_episodes=500, seed=0)
        td0 = td0_prediction(RandomWalkEnv(seed=0), n_episodes=500, seed=0)
        for s in RandomWalkEnv.STATES:
            assert lam_zero[s] == pytest.approx(td0[s], abs=1e-6)

    def test_lambda_zero_replacing_matches_td0(self) -> None:
        lam_zero = td_lambda_prediction(RandomWalkEnv(seed=0), lam=0.0,
                                        n_episodes=500, seed=0,
                                        trace="replacing")
        td0 = td0_prediction(RandomWalkEnv(seed=0), n_episodes=500, seed=0)
        for s in RandomWalkEnv.STATES:
            assert lam_zero[s] == pytest.approx(td0[s], abs=1e-6)

    def test_replacing_matches_ch06_accumulating_at_zero(self) -> None:
        # At lambda = 0 both trace types degenerate to TD(0).
        mine = td_lambda_prediction(RandomWalkEnv(seed=0), lam=0.0,
                                    n_episodes=300, seed=0,
                                    trace="replacing")
        theirs = td_lambda_accumulating_ch06(RandomWalkEnv(seed=0), lam=0.0,
                                             n_episodes=300, seed=0)
        for s in RandomWalkEnv.STATES:
            assert mine[s] == pytest.approx(theirs[s], abs=1e-6)

    def test_high_lambda_close_to_true_values(self) -> None:
        # lambda -> 1 approaches Monte-Carlo-like behavior (Section 12.2).
        values = td_lambda_prediction(RandomWalkEnv(seed=0), lam=0.9,
                                      n_episodes=5000, seed=0,
                                      trace="replacing")
        true_values = random_walk_true_values()
        errs = [abs(values[s] - true_values[s]) for s in true_values]
        assert max(errs) < 0.12

    def test_returns_all_states(self) -> None:
        values = td_lambda_prediction(RandomWalkEnv(seed=0), n_episodes=50, seed=0)
        assert set(values) == set(RandomWalkEnv.STATES)

    def test_invalid_trace_raises(self) -> None:
        with pytest.raises(ValueError):
            td_lambda_prediction(RandomWalkEnv(seed=0), trace="dutch")

    def test_lam_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            td_lambda_prediction(RandomWalkEnv(seed=0), lam=1.5)

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            td_lambda_prediction(RandomWalkEnv(seed=0), alpha=0.0)


class TestSarsaLambdaReplacing:
    """algorithm_8.1: Sarsa(lambda), Section 12.7."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, _ = sarsa_lambda_replacing(env, n_episodes=100, lam=0.9,
                                           epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa_lambda_replacing(env, n_episodes=10, lam=0.9,
                                            epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * len(RandomWalkEnv.ACTIONS)

    def test_lambda_zero_matches_one_step_sarsa(self) -> None:
        # lambda = 0 degenerates to one-step Sarsa (Algorithm 6.4).
        _, q_trace = sarsa_lambda_replacing(RandomWalkEnv(seed=0),
                                            n_episodes=500, lam=0.0,
                                            epsilon=0.1, seed=0)
        _, q_one_step = sarsa_one_step(RandomWalkEnv(seed=0),
                                       n_episodes=500, epsilon=0.1, seed=0)
        for key in q_trace:
            assert q_trace[key] == pytest.approx(q_one_step[key], abs=1e-10)

    def test_learns_rightward_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, q_table = sarsa_lambda_replacing(env, n_episodes=5000,
                                                 lam=0.9, epsilon=0.1, seed=0)
        # Reward +1 at the right terminal, so the learned policy moves right.
        for s in RandomWalkEnv.STATES:
            assert policy[s][1] > 0.5
        assert q_table[("A", "right")] > q_table[("A", "left")]

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            sarsa_lambda_replacing(RandomWalkEnv(seed=0), epsilon=0.0)

    def test_lam_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            sarsa_lambda_replacing(RandomWalkEnv(seed=0), lam=2.0)


class TestSarsaLambdaClearing:
    """algorithm_8.2: Sarsa(lambda) with clearing traces. Section 12.7, Fig 12.11."""

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, _ = sarsa_lambda_clearing(env, n_episodes=100, lam=0.9,
                                          epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = sarsa_lambda_clearing(env, n_episodes=10, lam=0.9,
                                           epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * len(RandomWalkEnv.ACTIONS)

    def test_lambda_zero_matches_one_step_sarsa(self) -> None:
        # At lambda = 0 the trace starts empty each step, so clearing
        # the non-selected traces is a no-op: identical to one-step Sarsa.
        _, q_trace = sarsa_lambda_clearing(RandomWalkEnv(seed=0),
                                           n_episodes=500, lam=0.0,
                                           epsilon=0.1, seed=0)
        _, q_one_step = sarsa_one_step(RandomWalkEnv(seed=0),
                                       n_episodes=500, epsilon=0.1, seed=0)
        for key in q_trace:
            assert q_trace[key] == pytest.approx(q_one_step[key], abs=1e-10)

    def test_clearing_differs_from_replacing_at_high_lambda(self) -> None:
        # Clearing must actually change the updates once traces persist.
        _, q_repl = sarsa_lambda_replacing(RandomWalkEnv(seed=1),
                                           n_episodes=200, lam=0.9,
                                           epsilon=0.1, seed=1)
        _, q_clear = sarsa_lambda_clearing(RandomWalkEnv(seed=1),
                                           n_episodes=200, lam=0.9,
                                           epsilon=0.1, seed=1)
        assert any(abs(q_repl[k] - q_clear[k]) > 1e-9 for k in q_repl)

    def test_learns_rightward_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, q_table = sarsa_lambda_clearing(env, n_episodes=5000,
                                                lam=0.9, epsilon=0.1, seed=0)
        for s in RandomWalkEnv.STATES:
            assert policy[s][1] > 0.5
        assert q_table[("A", "right")] > q_table[("A", "left")]


class TestTrueOnlineTDLambda:
    """algorithm_8.3: True online TD(lambda), Section 12.5 (box on p. 245)."""

    def test_lambda_zero_matches_td0(self) -> None:
        # At lambda = 0 the dutch trace is x and the correction terms cancel:
        # true online TD(0) == TD(0).
        values = true_online_td_lambda(RandomWalkEnv(seed=0), lam=0.0,
                                       n_episodes=500, seed=0)
        td0 = td0_prediction(RandomWalkEnv(seed=0), n_episodes=500, seed=0)
        for s in RandomWalkEnv.STATES:
            assert values[s] == pytest.approx(td0[s], abs=1e-6)

    def test_converges_toward_true_values(self) -> None:
        values = true_online_td_lambda(RandomWalkEnv(seed=0), lam=0.9,
                                       n_episodes=5000, seed=0)
        true_values = random_walk_true_values()
        errs = [abs(values[s] - true_values[s]) for s in true_values]
        assert max(errs) < 0.12

    def test_returns_all_states(self) -> None:
        values = true_online_td_lambda(RandomWalkEnv(seed=0), n_episodes=50,
                                       seed=0)
        assert set(values) == set(RandomWalkEnv.STATES)

    def test_values_are_finite(self) -> None:
        values = true_online_td_lambda(RandomWalkEnv(seed=0), lam=1.0,
                                       n_episodes=100, seed=0)
        assert all(np.isfinite(v) for v in values.values())

    def test_reproducible_with_seed(self) -> None:
        v1 = true_online_td_lambda(RandomWalkEnv(seed=42), lam=0.5,
                                   n_episodes=100, seed=42)
        v2 = true_online_td_lambda(RandomWalkEnv(seed=42), lam=0.5,
                                   n_episodes=100, seed=42)
        for s in RandomWalkEnv.STATES:
            assert v1[s] == pytest.approx(v2[s])

    def test_lam_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            true_online_td_lambda(RandomWalkEnv(seed=0), lam=-0.1)

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            true_online_td_lambda(RandomWalkEnv(seed=0), alpha=0.0)


# ---------------------------------------------------------------------------
# Q(lambda) tests (Section 12.10, Figure 12.12)
# ---------------------------------------------------------------------------


def _reference_watkins(
    env: RandomWalkEnv,
    n_episodes: int,
    alpha: float,
    lam: float,
    epsilon: float,
    gamma: float,
    seed: int | None,
    cut: bool = True,
) -> dict:
    """Independent transcription of the documented algorithm (spec-level).

    Mirrors the backward-view algorithm from the docstring: epsilon-greedy
    behavior, Q-learning max bootstrap, replacing traces (z <- 1 for the
    visited pair), decay by gamma * lam, and (if cut) zeroing ALL traces
    after a non-greedy step. Used to validate the solution against a
    separate code path written directly from the specification.
    """
    rng = np.random.default_rng(seed)
    states = env.STATES
    actions = env.ACTIONS
    q_table = {(s, a): 0.0 for s in states for a in actions}

    def behavior_probs(state: str) -> np.ndarray:
        q = np.array([q_table[(state, a)] for a in actions])
        best = q == q.max()
        probs = np.full(len(actions), epsilon / len(actions), dtype=np.float64)
        probs[best] += (1.0 - epsilon) / np.sum(best)
        return probs

    for _ in range(n_episodes):
        traces = {p: 0.0 for p in q_table}
        state = env.reset()
        done = False
        while not done:
            q_before = np.array([q_table[(state, a)] for a in actions])
            probs = behavior_probs(state)
            action_index = int(rng.choice(len(actions), p=probs))
            greedy = q_before[action_index] == q_before.max()
            action = actions[action_index]
            pair = (state, action)
            reward, next_state, done = env.step(action)
            best_next = (
                0.0
                if done
                else max(q_table[(next_state, a)] for a in actions)
            )
            delta = reward + gamma * best_next - q_table[pair]
            traces[pair] = 1.0
            for p in q_table:
                q_table[p] += alpha * delta * traces[p]
            for p in q_table:
                traces[p] *= gamma * lam
            if not greedy and cut:
                for p in q_table:
                    traces[p] = 0.0
            state = next_state if not done else state
    return q_table


class TestWatkinsQLambda:
    """algorithm_12.4: Watkins's Q(lambda), Section 12.10, Figure 12.12."""

    def test_lambda_zero_matches_q_learning(self) -> None:
        # lambda = 0 degenerates to one-step Q-learning (Algorithm 6.2).
        _, q_trace = watkins_q_lambda(RandomWalkEnv(seed=0), n_episodes=500,
                                      lam=0.0, epsilon=0.1, gamma=0.99, seed=0)
        _, q_one_step = q_learning(RandomWalkEnv(seed=0), n_episodes=500,
                                   epsilon=0.1, gamma=0.99, seed=0)
        for key in q_trace:
            assert q_trace[key] == pytest.approx(q_one_step[key], abs=1e-10)

    def test_lambda_zero_accumulating_matches_q_learning(self) -> None:
        _, q_trace = watkins_q_lambda(RandomWalkEnv(seed=0), n_episodes=500,
                                      lam=0.0, epsilon=0.1, gamma=0.99,
                                      seed=0, trace="accumulating")
        _, q_one_step = q_learning(RandomWalkEnv(seed=0), n_episodes=500,
                                   epsilon=0.1, gamma=0.99, seed=0)
        for key in q_trace:
            assert q_trace[key] == pytest.approx(q_one_step[key], abs=1e-10)

    def test_matches_reference_simulation(self) -> None:
        # Trace-cutting correctness: same updates as an independent
        # spec-level transcription, on identical trajectories.
        env = RandomWalkEnv(seed=3)
        _, q_table = watkins_q_lambda(env, n_episodes=100, alpha=0.1,
                                      lam=0.9, epsilon=1.0, gamma=0.99,
                                      seed=3)
        reference = _reference_watkins(RandomWalkEnv(seed=3),
                                       n_episodes=100, alpha=0.1, lam=0.9,
                                       epsilon=1.0, gamma=0.99, seed=3,
                                       cut=True)
        for key in q_table:
            assert q_table[key] == pytest.approx(reference[key], abs=1e-10)

    def test_trace_cutting_has_effect(self) -> None:
        # With a non-greedy action in the trajectory, zeroing the traces
        # must change the final action values relative to no cutting.
        without_cut = _reference_watkins(RandomWalkEnv(seed=3),
                                         n_episodes=100, alpha=0.1, lam=0.9,
                                         epsilon=1.0, gamma=0.99, seed=3,
                                         cut=False)
        with_cut = _reference_watkins(RandomWalkEnv(seed=3),
                                      n_episodes=100, alpha=0.1, lam=0.9,
                                      epsilon=1.0, gamma=0.99, seed=3,
                                      cut=True)
        assert any(abs(without_cut[k] - with_cut[k]) > 1e-9 for k in with_cut)

    def test_learns_rightward_preference(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, q_table = watkins_q_lambda(env, n_episodes=5000, lam=0.9,
                                           epsilon=0.1, seed=0)
        for s in RandomWalkEnv.STATES:
            assert policy[s][1] > 0.5
        assert q_table[("A", "right")] > q_table[("A", "left")]
        assert q_table[("E", "right")] > q_table[("E", "left")]

    def test_policy_sums_to_one(self) -> None:
        env = RandomWalkEnv(seed=0)
        policy, _ = watkins_q_lambda(env, n_episodes=100, lam=0.9,
                                     epsilon=0.1, seed=0)
        for probs in policy.values():
            assert probs.sum() == pytest.approx(1.0)

    def test_q_keys_cover_all_pairs(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = watkins_q_lambda(env, n_episodes=10, lam=0.9,
                                      epsilon=0.1, seed=0)
        assert len(q_table) == len(RandomWalkEnv.STATES) * len(env.ACTIONS)

    def test_random_behavior_stays_finite(self) -> None:
        env = RandomWalkEnv(seed=0)
        _, q_table = watkins_q_lambda(env, n_episodes=300, lam=0.9,
                                      epsilon=1.0, seed=0)
        assert all(np.isfinite(v) for v in q_table.values())

    def test_true_values_used_as_sanity_baseline(self) -> None:
        # Reward only flows at the right terminal; sanity-check that the
        # learned values are consistent with an absorbing +1 right end.
        true_values = random_walk_true_values()
        env = RandomWalkEnv(seed=0)
        _, q_table = watkins_q_lambda(env, n_episodes=5000, lam=0.9,
                                      epsilon=0.1, seed=0)
        # The best action at the right end must lead to the rewarding end;
        # its value must be near the undiscounted return of 1 (gamma ~ 1).
        assert q_table[("E", "right")] == pytest.approx(1.0, abs=0.2)
        assert true_values["E"] == pytest.approx(5 / 6)

    def test_epsilon_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            watkins_q_lambda(RandomWalkEnv(seed=0), epsilon=0.0)

    def test_lam_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            watkins_q_lambda(RandomWalkEnv(seed=0), lam=1.5)

    def test_invalid_trace_raises(self) -> None:
        with pytest.raises(ValueError):
            watkins_q_lambda(RandomWalkEnv(seed=0), trace="dutch")


class TestWatkinsLambdaReturn:
    """Forward-view Watkins's Q(lambda)-return, Section 12.10, Fig 12.12."""

    def test_all_greedy_known_answer(self) -> None:
        # rewards 0, 0, 1; bootstraps max Q at s1, s2, s3 (0 at terminal).
        rewards = [0.0, 0.0, 1.0]
        bootstraps = [0.2, 0.3, 0.0]
        greedy = [True, True, True]
        result = watkins_lambda_return(rewards, bootstraps, greedy,
                                       gamma=1.0, lam=0.5)
        # t=0: G_{0:1} = .2, G_{0:2} = .3, G_{0:3} = 1.0
        #   G^λ = .5*(.2 + .5*.3 + .25*1.0) + .125*1.0 = .425
        # t=1: G_{1:2} = .3, G_{1:3} = 1.0
        #   G^λ = .5*(.3 + .5*1.0) + .25*1.0 = .65
        # t=2: G_{2:3} = 1.0
        #   G^λ = .5*1.0 + .5*1.0 = 1.0
        assert result == pytest.approx([0.425, 0.65, 1.0], abs=1e-12)

    def test_nongreedy_action_cuts_horizon(self) -> None:
        rewards = [0.0, 0.0, 1.0]
        bootstraps = [0.2, 0.3, 0.0]
        greedy = [True, False, True]
        result = watkins_lambda_return(rewards, bootstraps, greedy,
                                       gamma=1.0, lam=0.5)
        # t=0: first non-greedy at j=1 -> n_max=2
        #   G^λ = .5*(.2 + .5*.3) + .25*G_{0:2}(=.3) = .25
        # t=1: first non-greedy at j=1 -> n_max=1
        #   G^λ = .5*G_{1:2}(=.3) + .5*.3 = .3
        # t=2: all greedy -> n_max=1: G^λ = 1.0
        assert result == pytest.approx([0.25, 0.3, 1.0], abs=1e-12)

    def test_lambda_zero_is_one_step_return(self) -> None:
        rewards = [0.0, 0.0, 1.0]
        bootstraps = [0.2, 0.3, 0.0]
        greedy = [True, True, True]
        result = watkins_lambda_return(rewards, bootstraps, greedy,
                                       gamma=1.0, lam=0.0)
        # G_{t:t+1} = r_t + bootstrap_values[t]
        assert result == pytest.approx([0.2, 0.3, 1.0], abs=1e-12)

    def test_lambda_one_is_trimmed_return(self) -> None:
        rewards = [0.0, 0.0, 1.0]
        bootstraps = [0.2, 0.3, 0.0]
        greedy = [True, True, True]
        result = watkins_lambda_return(rewards, bootstraps, greedy,
                                       gamma=0.9, lam=1.0)
        # G^λ = G_{t:T}: t=0: 1.0*0.81, t=1: 1.0*0.9, t=2: 1.0
        assert result == pytest.approx([0.81, 0.9, 1.0], abs=1e-12)

    def test_output_shape_and_finiteness(self) -> None:
        rewards = [1.0, 2.0, 3.0]
        bootstraps = [1.0, 1.0, 1.0]
        greedy = [True, True, True]
        result = watkins_lambda_return(rewards, bootstraps, greedy,
                                       gamma=1.0, lam=0.5)
        assert len(result) == 3
        assert all(np.isfinite(v) for v in result)

    def test_bootstrap_length_validation(self) -> None:
        with pytest.raises(ValueError):
            watkins_lambda_return([1.0], [0.0, 0.0], [True], lam=0.5)

    def test_greedy_length_validation(self) -> None:
        with pytest.raises(ValueError):
            watkins_lambda_return([1.0], [0.0, 0.0], [True, True], lam=0.5)

    def test_lam_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            watkins_lambda_return([1.0], [0.0, 0.0], [True], lam=-0.5)


class TestFigureExperiments:
    """Reduced-scale data behind Figures 12.3, 12.6, 12.8, 12.10, 12.11."""

    def test_random_walk_curves_shape(self):
        from chapter_12.figure_experiments import (
            ALPHAS,
            LAMBDAS,
            random_walk_lambda_curves,
        )

        curves = random_walk_lambda_curves(repeats=2, n_episodes=3, seed=0)
        assert set(curves["curves"]) == {"offline", "td", "true_online"}
        for name in curves["curves"]:
            for lam in LAMBDAS:
                values = np.asarray(curves["curves"][name][lam])
                assert values.shape == (len(ALPHAS),)
                assert np.all(values[:len(ALPHAS)] >= 0)
        assert len(curves["alphas"]) == len(ALPHAS)

    def test_mountain_car_curves_shape(self):
        from chapter_12.figure_experiments import mountain_car_lambda_curves

        curves = mountain_car_lambda_curves(episodes=5, seed=0)
        assert len(curves) == 4
        for steps in curves.values():
            assert len(steps) == 5
            assert all(0 < s <= 200 for s in steps)
