#!/usr/bin/env python3

# Direct test to bypass the interactive mode issues

import sys
sys.path.append('/home/romeo/projects/rl')

from agents.policy_iteration import policy_iteration
from agents.value_iteration import value_iteration
from gridworld_env import GridWorldEnv

def test_agents():
    # Create a simple environment for testing
    env = GridWorldEnv(size=4, stochastic=False)
    
    print("--- Testing Value Iteration ---")
    try:
        policy_vi, V_vi = value_iteration(env, discount_factor=0.99, tolerance=1e-6)
        print("Value Iteration completed successfully!")
        print(f"Found policy of length {len(policy_vi)}")
    except Exception as e:
        print(f"Value iteration failed: {e}")
        
    print("\n--- Testing Policy Iteration ---")
    try:
        policy_pi, V_pi = policy_iteration(env, discount_factor=0.99, tolerance=1e-6)
        print("Policy Iteration completed successfully!")
        print(f"Found policy of length {len(policy_pi)}")
    except Exception as e:
        print(f"Policy iteration failed: {e}")

if __name__ == "__main__":
    test_agents()