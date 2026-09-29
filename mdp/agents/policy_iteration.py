#!/usr/bin/env python3
"""
Policy Iteration Implementation for GridWorld Environment
"""

import sys; sys.path.insert(0, '/home/romeo/projects/rl')
import numpy as np
from gridworld_env import GridWorldEnv


def policy_iteration(env, discount_factor=0.9, max_iterations=1000, tolerance=1e-6):
    """
    Policy iteration algorithm for finding optimal policy in GridWorld.
    
    Args:
        env: GridWorld environment instance
        discount_factor (float): Discount factor for future rewards
        max_iterations (int): Maximum number of iterations to prevent infinite loops
        tolerance (float): Convergence threshold
        
    Returns:
        tuple: (policy, value_function)
    """
    # Initialize policy and value function
    policy = {state: env.actions[0] for state in env.states}  # Start with first action
    # Include terminal states with value 0
    value_function = {state: 0.0 for state in env.all_states}
    
    for iteration in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            for state in env.states:
                if state in env.terminal_states:
                    continue  # Skip terminal states (absorbing)
                
                old_value = value_function[state]
                action = policy[state]
                
                # Compute expected value over all possible outcomes
                expected_value = 0.0
                for prob, next_state, reward in env.get_transition_and_reward_probabilities(state, action):
                    next_state_value = value_function[next_state] if next_state in env.states else 0.0
                    expected_value += prob * (reward + discount_factor * next_state_value)
                
                value_function[state] = expected_value
                delta = max(delta, abs(old_value - value_function[state]))
            
            # Check for convergence
            if delta < tolerance:
                break
        
        # Policy Improvement
        policy_stable = True
        for state in env.states:
            if state in env.terminal_states:
                continue  # Skip terminal states
            
            old_action = policy[state]
            
            # Find best action based on current value function using expectations
            best_action = None
            best_value = float('-inf')
            
            for action in env.actions:
                action_value = 0.0
                for prob, next_state, reward in env.get_transition_and_reward_probabilities(state, action):
                    next_state_value = value_function[next_state] if next_state in env.states else 0.0
                    action_value += prob * (reward + discount_factor * next_state_value)
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            policy[state] = best_action
            
            # Check if policy improved
            if old_action != best_action:
                policy_stable = False
        
        # If policy didn't change, we have converged
        if policy_stable:
            break
    
    return policy, value_function


def print_policy_and_values(env, policy, values):
    """Utility function to display the policy and value function."""
    print("Policy:")
    for i in range(env.size):
        row = []
        for j in range(env.size):
            if (i, j) in env.terminal_states:
                row.append('T')  # Terminal state
            else:
                row.append(policy[(i, j)][0].upper())  # First letter of action
        print(' '.join(row))
    
    print("\nValue Function:")
    for i in range(env.size):
        row = []
        for j in range(env.size):
            if (i, j) in env.terminal_states:
                row.append("T")
            else:
                row.append(f"{values[(i, j)]:.1f}")
        print(' '.join(row))


if __name__ == "__main__":
    # Create environment
    grid_size = 4
    env = GridWorldEnv(size=grid_size)
    
    print(f"Grid World Environment (Size: {grid_size}x{grid_size})")
    print("Terminal states at (0,0) and ({},{})".format(grid_size-1, grid_size-1))
    print()
    
    # Run policy iteration
    policy, values = policy_iteration(env)
    
    # Display results
    print_policy_and_values(env, policy, values)