#!/usr/bin/env python3
"""
Run policy iteration on the GridWorld environment.
"""

from policy_iteration import policy_iteration, print_policy_and_values
from gridworld_env import GridWorldEnv

def main():
    # Create environment
    env = GridWorldEnv(size=4)
    
    print("Grid World Environment (Size: 4x4)")
    print("Terminal states at (0,0) and (3,3)")
    print()
    
    # Run policy iteration
    policy, values = policy_iteration(env)
    
    # Display results
    print_policy_and_values(env, policy, values)
    
    print("\n" + "="*50)
    print("Testing with stochastic environment:")
    
    # Test with stochastic environment
    env_stochastic = GridWorldEnv(size=4, stochastic=True, stochasticity_factor=0.2)
    policy_stochastic, values_stochastic = policy_iteration(env_stochastic)
    
    print_policy_and_values(env_stochastic, policy_stochastic, values_stochastic)

if __name__ == "__main__":
    main()