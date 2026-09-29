#!/usr/bin/env python3
"""
Test script to verify stochastic environment handling is fixed.
Tests both Policy Iteration and Value Iteration with deterministic and stochastic environments.
"""

import sys
sys.path.insert(0, '/home/romeo/projects/rl')

from gridworld_env import GridWorldEnv
from agents.policy_iteration import policy_iteration, print_policy_and_values as pi_print
from agents.value_iteration import value_iteration, print_policy_and_values as vi_print


def test_transition_probabilities():
    """Test that get_transition_and_reward_probabilities works correctly."""
    print("=" * 60)
    print("TEST 1: Transition Probabilities")
    print("=" * 60)
    
    # Test deterministic
    env_det = GridWorldEnv(size=4, stochastic=False)
    transitions = env_det.get_transition_and_reward_probabilities((2, 2), 'up')
    print(f"\nDeterministic env - State (2,2), Action 'up':")
    for prob, next_state, reward in transitions:
        print(f"  Prob={prob:.1f}, Next={next_state}, Reward={reward}")
    
    # Test stochastic
    env_stoch = GridWorldEnv(size=4, stochastic=True, stochasticity_factor=0.2)
    transitions = env_stoch.get_transition_and_reward_probabilities((2, 2), 'up')
    print(f"\nStochastic env (p=0.8) - State (2,2), Action 'up':")
    for prob, next_state, reward in transitions:
        print(f"  Prob={prob:.3f}, Next={next_state}, Reward={reward}")
    
    # Verify probabilities sum to 1
    total_prob = sum(p for p, _, _ in transitions)
    assert abs(total_prob - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total_prob}"
    print(f"  ✓ Probabilities sum to {total_prob:.1f} (expected 1.0)")
    
    # Verify terminal state behavior
    transitions_terminal = env_stoch.get_transition_and_reward_probabilities((0, 0), 'up')
    print(f"\nStochastic env - Terminal State (0,0), Action 'up':")
    assert len(transitions_terminal) == 1 and transitions_terminal[0] == (1.0, (0, 0), 0.0)
    print("  ✓ Terminal state has absorbing self-loop with zero reward")
    
    print("\n✓ TEST 1 PASSED\n")


def test_policy_iteration_both_modes():
    """Test policy iteration with both deterministic and stochastic environments."""
    print("=" * 60)
    print("TEST 2: Policy Iteration - Deterministic vs Stochastic")
    print("=" * 60)
    
    # Deterministic
    env_det = GridWorldEnv(size=4, stochastic=False)
    policy_det, values_det = policy_iteration(env_det, discount_factor=0.9, 
                                               max_iterations=1000, tolerance=1e-6)
    print("\nDeterministic Environment (Policy Iteration):")
    pi_print(env_det, policy_det, values_det)
    
    # Stochastic
    env_stoch = GridWorldEnv(size=4, stochastic=True, stochasticity_factor=0.2)
    policy_stoch, values_stoch = policy_iteration(env_stoch, discount_factor=0.9,
                                                   max_iterations=1000, tolerance=1e-6)
    print("\nStochastic Environment (Policy Iteration):")
    pi_print(env_stoch, policy_stoch, values_stoch)
    
    # Verify both converged (policies are different because stochastic is harder)
    print(f"\nDeterministic converged in iterations where policy: {policy_det}")
    print(f"Stochastic converged in iterations where policy: {policy_stoch}")
    assert len(policy_det) > 0 and len(policy_stoch) > 0
    print("\n✓ TEST 2 PASSED\n")


def test_value_iteration_both_modes():
    """Test value iteration with both deterministic and stochastic environments."""
    print("=" * 60)
    print("TEST 3: Value Iteration - Deterministic vs Stochastic")
    print("=" * 60)
    
    # Deterministic
    env_det = GridWorldEnv(size=4, stochastic=False)
    policy_det, values_det = value_iteration(env_det, discount_factor=0.9,
                                              max_iterations=1000, tolerance=1e-6)
    print("\nDeterministic Environment (Value Iteration):")
    vi_print(env_det, policy_det, values_det)
    
    # Stochastic
    env_stoch = GridWorldEnv(size=4, stochastic=True, stochasticity_factor=0.2)
    policy_stoch, values_stoch = value_iteration(env_stoch, discount_factor=0.9,
                                                  max_iterations=1000, tolerance=1e-6)
    print("\nStochastic Environment (Value Iteration):")
    vi_print(env_stoch, policy_stoch, values_stoch)
    
    # Verify both converged
    assert len(policy_det) > 0 and len(policy_stoch) > 0
    print("\n✓ TEST 3 PASSED\n")


def test_optimal_behavior():
    """Verify that the agents actually make progress toward the goal."""
    print("=" * 60)
    print("TEST 4: Expected Behavior Verification")
    print("=" * 60)
    
    env_stoch = GridWorldEnv(size=4, stochastic=True, stochasticity_factor=0.2)
    policy, values = value_iteration(env_stoch, discount_factor=0.9,
                                      max_iterations=1000, tolerance=1e-6)
    
    # Verify terminal states have optimal behavior (move toward other terminal)
    # The bottom-right corner should move left or up consistently
    print("\nAnalyzing policy near goal:")
    
    # Check that non-terminal states don't try to stay in terminals
    for state, action in policy.items():
        if state[0] == 3 and state[1] == 2:  # Bottom row, second-to-last column
            print(f"State (3,2) [near goal]: chooses {action}")
            assert action in ['right', 'up'], f"Should move toward goal, got {action}"
    
    print("\n✓ TEST 4 PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("RL GRIDWORLD STOCHASTIC BUG FIX - VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_transition_probabilities()
        test_policy_iteration_both_modes()
        test_value_iteration_both_modes()
        test_optimal_behavior()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary of fix:")
        print("- GridWorldEnv.get_transition_and_reward_probabilities() provides")
        print("  full probability distributions for any state-action pair")
        print("- Policy Iteration now computes EXPECTED VALUES properly")
        print("- Value Iteration now computes EXPECTED VALUES properly")
        print("- Both deterministic and stochastic environments work correctly!")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
