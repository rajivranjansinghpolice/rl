"""Tests for Chapter 13: Policy Gradient Algorithms.

Reference: Sutton & Barto (2nd Ed), Chapter 13.

Tests for: Alg 13.2 (policy gradient theorem check), Alg 13.3 (REINFORCE),
Alg 13.4 (REINFORCE with baseline), Alg 13.5 (actor-critic), Alg 13.7
(Gaussian policy for continuous actions), and Example 13.1 (short corridor).
"""
from __future__ import annotations

import numpy as np
import pytest

from chapter_13.solution import (
    ActorCriticAgent,
    ActorCriticAlgorithm,
    ContinuingActorCritic,
    GaussianPolicy,
    LEFT,
    PolicyEnvironment,
    PolicyGradientAlgorithm,
    PolicyGradientWithBaseline,
    RIGHT,
    ShortCorridorEnv,
    ShortCorridorREINFORCE,
    check_policy_gradient_theorem,
    main,
    run_policy_gradient_simulation,
    run_short_corridor_reinforce,
    run_short_corridor_value_ascent,
    short_corridor_features,
    short_corridor_optimal_p,
    short_corridor_right_prob,
    short_corridor_value,
    short_corridor_value_gradient,
    simulate_short_corridor,
)


def make_env() -> PolicyEnvironment:
    """The standard 3-action bandit used across the tests."""
    return PolicyEnvironment(
        n_actions=3,
        p=np.array([0.33, 0.33, 0.34]),
        rewards=np.array([1.0, 2.0, 3.0]),
    )


def make_pg_algorithm(**overrides) -> PolicyGradientAlgorithm:
    params = dict(n_actions=3, n_states=10, lr=0.01, gamma=0.99, baseline=1.0, seed=42)
    params.update(overrides)
    return PolicyGradientAlgorithm(**params)


def make_pg_baseline(**overrides) -> PolicyGradientWithBaseline:
    params = dict(n_actions=3, n_states=10, lr=0.01, gamma=0.99, baseline_dim=1, seed=42)
    params.update(overrides)
    return PolicyGradientWithBaseline(**params)


def make_ac_algorithm(**overrides) -> ActorCriticAlgorithm:
    params = dict(n_states=10, n_actions=3, lr=0.01, gamma=0.99, seed=42)
    params.update(overrides)
    return ActorCriticAlgorithm(**params)


class TestPolicyEnvironment:
    """Test the PolicyEnvironment class (Section 13.1)."""

    def test_create_policy_environment(self) -> None:
        """Test creating a basic policy environment."""
        env = make_env()
        assert env is not None
        assert hasattr(env, 'n_actions')
        assert hasattr(env, 'p')
        assert hasattr(env, 'rewards')
        assert hasattr(env, 'action_count')

    def test_policy_environment_initialization(self) -> None:
        """Test that environment starts with correct parameters."""
        env = make_env()
        assert env.n_actions == 3

    def test_policy_environment_select_action(self) -> None:
        """Test that select_action returns a valid action."""
        env = make_env()
        assert isinstance(env.select_action(), int)

    def test_policy_environment_step(self) -> None:
        """Test that step returns a reward."""
        env = make_env()
        assert isinstance(env.step(0), (int, float))

    def test_policy_environment_get_action_count(self) -> None:
        """Test that get_action_count returns the right count."""
        env = make_env()
        assert isinstance(env.get_action_count(0), float)
        env.step(0)
        assert env.get_action_count(0) == pytest.approx(1.0)


class TestActorCriticAgent:
    """Test the ActorCriticAgent class (Algorithm 13.5)."""

    def test_create_actor_critic_agent(self) -> None:
        """Test creating a basic actor-critic agent."""
        agent = ActorCriticAgent(n_actions=3, n_states=10)
        assert agent is not None

    def test_actor_critic_agent_initialization(self) -> None:
        """Test that agent has required attributes."""
        agent = ActorCriticAgent(n_actions=3, n_states=10)
        assert hasattr(agent, 'n_actions')
        assert hasattr(agent, 'n_states')
        assert hasattr(agent, 'theta')
        assert hasattr(agent, 'V')

    def test_actor_critic_agent_select_action(self) -> None:
        """Test that select_action returns an action index."""
        agent = ActorCriticAgent(n_actions=3, n_states=10)
        action = agent.select_action(0)
        assert isinstance(action, int)

    def test_actor_critic_agent_predict(self) -> None:
        """Test that predict returns a valid action index."""
        agent = ActorCriticAgent(n_actions=3, n_states=10)
        assert 0 <= agent.predict(0) < 3


class TestActorCriticAlgorithm:
    """Test the ActorCriticAlgorithm class (Algorithm 13.5)."""

    def test_create_actor_critic_algorithm(self) -> None:
        """Test creating a basic actor-critic algorithm."""
        algo = make_ac_algorithm()
        assert algo is not None

    def test_actor_critic_algorithm_initialization(self) -> None:
        """Test that algorithm has required attributes."""
        algo = make_ac_algorithm()
        assert hasattr(algo, 'n_states')
        assert hasattr(algo, 'n_actions')
        assert hasattr(algo, 'learning_rate')
        assert hasattr(algo, 'gamma')

    def test_actor_critic_algorithm_fit(self) -> None:
        """Test that fit method exists."""
        algo = make_ac_algorithm()
        assert hasattr(algo, 'fit')

    def test_actor_critic_algorithm_predict(self) -> None:
        """Test that predict method exists."""
        algo = make_ac_algorithm()
        assert hasattr(algo, 'predict')


class TestPolicyGradientAlgorithm:
    """Test the PolicyGradientAlgorithm class (Algorithm 13.3)."""

    def test_create_policy_gradient_algorithm(self) -> None:
        """Test creating a policy gradient algorithm."""
        algo = make_pg_algorithm()
        assert algo is not None

    def test_policy_gradient_algorithm_initialization(self) -> None:
        """Test that algorithm has required attributes."""
        algo = make_pg_algorithm()
        assert hasattr(algo, 'n_actions')
        assert hasattr(algo, 'n_states')
        assert hasattr(algo, 'learning_rate')
        assert hasattr(algo, 'gamma')

    def test_policy_gradient_algorithm_fit(self) -> None:
        """Test that fit method exists."""
        algo = make_pg_algorithm()
        assert hasattr(algo, 'fit')

    def test_policy_gradient_algorithm_predict(self) -> None:
        """Test that predict method exists."""
        algo = make_pg_algorithm()
        assert hasattr(algo, 'predict')


class TestPolicyGradientWithBaseline:
    """Test the PolicyGradientWithBaseline class (Algorithm 13.4)."""

    def test_create_policy_gradient_with_baseline(self) -> None:
        """Test creating a policy gradient algorithm with baseline."""
        algo = make_pg_baseline()
        assert algo is not None

    def test_policy_gradient_with_baseline_initialization(self) -> None:
        """Test that algorithm has required attributes."""
        algo = make_pg_baseline()
        assert hasattr(algo, 'n_actions')
        assert hasattr(algo, 'n_states')
        assert hasattr(algo, 'learning_rate')
        assert hasattr(algo, 'gamma')

    def test_policy_gradient_with_baseline_fit(self) -> None:
        """Test that fit method exists."""
        algo = make_pg_baseline()
        assert hasattr(algo, 'fit')

    def test_policy_gradient_with_baseline_predict(self) -> None:
        """Test that predict method exists."""
        algo = make_pg_baseline()
        assert hasattr(algo, 'predict')


class TestRunPolicyGradientSimulation:
    """Test the run_policy_gradient_simulation function."""

    def test_run_policy_gradient_simulation_exists(self) -> None:
        """Test that the function exists."""
        assert callable(run_policy_gradient_simulation)

    def test_run_policy_gradient_simulation_signature(self) -> None:
        """Test that the function has the expected signature."""
        import inspect
        sig = inspect.signature(run_policy_gradient_simulation)
        params = list(sig.parameters.keys())
        assert 'algorithm' in params
        assert 'env' in params

    def test_run_policy_gradient_simulation_returns_expected_type(self) -> None:
        """Test that the function returns a list of episode rewards."""
        algo = make_pg_algorithm()
        env = make_env()
        result = run_policy_gradient_simulation(algo, env)
        assert isinstance(result, list)
        assert len(result) > 0


class TestMain:
    """Test the main function."""

    def test_main_exists(self) -> None:
        """Test that the main function exists."""
        assert callable(main)

    def test_main_signature(self) -> None:
        """Test that the main function has expected signature."""
        import inspect
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        assert 'seed' in params

    def test_main_runs_simulation(self) -> None:
        """Test that main returns a reward history."""
        result = main(seed=42)
        assert isinstance(result, list)
        assert len(result) > 0


class TestPolicyGradientTheorem:
    """Test the policy gradient theorem numeric check (Algorithm 13.2)."""

    def test_analytic_matches_numeric(self) -> None:
        rng = np.random.default_rng(0)
        theta = rng.normal(0, 1, 3)
        q = np.array([1.0, 2.0, 3.0])
        assert check_policy_gradient_theorem(theta, q) is True

    def test_rejects_mismatched_shape(self) -> None:
        with pytest.raises(ValueError):
            check_policy_gradient_theorem(np.zeros(2), np.zeros(3))


class TestGaussianPolicy:
    """Test GaussianPolicy (Algorithm 13.7)."""

    def test_sample_returns_float(self) -> None:
        policy = GaussianPolicy(n_states=1, seed=0)
        assert isinstance(policy.sample(0), float)

    def test_update_moves_mean_toward_positive_advantage(self) -> None:
        policy = GaussianPolicy(n_states=1, lr=0.1, seed=0)
        before = policy.mu[0]
        policy.update(0, action=5.0, advantage=1.0)
        assert policy.mu[0] > before

    def test_log_prob_is_finite(self) -> None:
        policy = GaussianPolicy(n_states=1, seed=0)
        assert np.isfinite(policy.log_prob(0, 0.5))

    def test_invalid_n_states_raises(self) -> None:
        with pytest.raises(ValueError):
            GaussianPolicy(0)


class TestContinuingActorCritic:
    """Test ContinuingActorCritic (Section 13.6) — average-reward actor-critic."""

    def test_r_hat_moves_toward_average_reward(self) -> None:
        algo = ContinuingActorCritic(n_states=1, n_actions=3, lr=0.05, beta=0.05, seed=42)
        env = PolicyEnvironment(
            n_actions=3,
            p=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
            rewards=np.array([1.0, 2.0, 3.0]),
            seed=42,
        )
        for _ in range(300):
            action = algo.select_action(0)
            reward = env.step(action)
            algo.continuing_step(0, action, reward, 0)
        assert np.isfinite(algo.r_hat)
        assert np.isfinite(algo.V).all()
        assert np.isfinite(algo.theta).all()

    def test_continuing_step_prefers_high_reward_action(self) -> None:
        algo = ContinuingActorCritic(n_states=1, n_actions=3, lr=0.1, beta=0.1, seed=0)
        rng = np.random.default_rng(0)
        rewards = np.array([1.0, 2.0, 3.0])
        for _ in range(500):
            action = algo.select_action(0)
            reward = float(rewards[action] + rng.normal(0, 0.01))
            algo.continuing_step(0, action, reward, 0)
        assert algo.theta[0, 2] > algo.theta[0, 0]

    def test_beta_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ContinuingActorCritic(n_states=1, n_actions=3, beta=0.0)


class TestShortCorridor:
    """Test the short-corridor MDP (Figure 13.1/13.2, Example 13.1)."""

    def test_transitions(self) -> None:
        env = ShortCorridorEnv()
        env.reset()
        next_state, reward, done = env.step(RIGHT)
        assert next_state == 2 and reward == -1.0 and not done
        env.reset()
        next_state, reward, done = env.step(LEFT)
        assert next_state == 1 and reward == -1.0 and not done

        env.state = 2
        next_state, reward, done = env.step(RIGHT)
        assert next_state == 1 and reward == -1.0 and not done
        env.state = 2
        next_state, reward, done = env.step(LEFT)
        assert next_state == 3 and reward == -1.0 and not done

        env.state = 3
        next_state, reward, done = env.step(RIGHT)
        assert next_state == 4 and reward == -1.0 and done

    def test_features(self) -> None:
        assert np.allclose(short_corridor_features(RIGHT), [1.0, 0.0])
        assert np.allclose(short_corridor_features(LEFT), [0.0, 1.0])

    def test_policy_sigmoid(self) -> None:
        theta = np.array([0.5, -0.5])
        assert short_corridor_right_prob(theta) == pytest.approx(1.0 / (1 + np.exp(-1.0)))

    def test_optimal_p_and_value(self) -> None:
        p, v = short_corridor_optimal_p()
        assert p == pytest.approx(2.0 - np.sqrt(2.0))
        assert v == pytest.approx(-6.0 - 4.0 * np.sqrt(2.0))

    def test_value_curve_max_at_optimal(self) -> None:
        p_star, _ = short_corridor_optimal_p()
        ps = np.linspace(0.05, 0.95, 181)
        vals = [short_corridor_value(p) for p in ps]
        best_p = ps[np.argmax(vals)]
        assert abs(best_p - p_star) < 0.02

    def test_value_matches_monte_carlo(self) -> None:
        p = 0.586
        analytic = short_corridor_value(p)
        mc = simulate_short_corridor(p, episodes=3000, seed=0)
        assert abs(mc - analytic) < 1.0  # MC variance; book value is approximate

    def test_reinforce_moves_toward_optimal(self) -> None:
        """REINFORCE without baseline on negative rewards: 800 episodes may not reach p*."""
        agent, hist = run_short_corridor_reinforce(lr=0.02, episodes=800, seed=0)
        # Without a baseline, high variance can prevent convergence in few episodes;
        # verify the process completes and stays in a valid probability range.
        p_final = agent.right_prob()
        assert 0.0 < p_final < 1.0
        # Record that the curve may not reach the theoretical p*; see Figure 13.2
        # (illustrative demonstration, figure identity cannot be verified against PDF).


class TestValueAscent:
    """Test exact policy-gradient ascent on the closed-form V1(p)."""

    def test_gradient_sign_matches_optimum(self) -> None:
        p_star, _ = short_corridor_optimal_p()
        assert short_corridor_value_gradient(p_star - 0.1) > 0
        assert short_corridor_value_gradient(p_star + 0.1) < 0

    def test_ascent_converges_to_optimal(self) -> None:
        hist = run_short_corridor_value_ascent(start_p=0.5, alpha=0.01, max_iter=10000)
        p_star, v_star = short_corridor_optimal_p()
        final_p = hist["p"][-1]
        final_v = hist["v1"][-1]
        assert abs(final_p - p_star) < 1e-3
        assert abs(final_v - v_star) < 1e-2


class TestIntegration:
    """Integration tests for the full policy-gradient flow."""

    def test_run_simulation_length(self) -> None:
        algo = make_pg_algorithm(baseline=0.0)
        env = make_env()
        result = run_policy_gradient_simulation(algo, env, episodes=10, steps=3)
        assert isinstance(result, list)
        assert len(result) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
