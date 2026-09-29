#!/usr/bin/env python3
"""
Value Iteration Implementation for GridWorld Environment
"""

import numpy as np
from gridworld_env import GridWorldEnv


def value_iteration(env, discount_factor=0.9, max_iterations=1000, tolerance=1e-6):
    """
    Value iteration algorithm for finding optimal policy in GridWorld.
    
    Args:
        env: GridWorld environment instance
        discount_factor (float): Discount factor for future rewards
        max_iterations (int): Maximum number of iterations to prevent infinite loops
        tolerance (float): Convergence threshold
        
    Returns:
        tuple: (policy, value_function)
    """
    # Initialize value function
    value_function = {state: 0.0 for state in env.states}
    
    # Set terminal states to zero value (they don't have transitions)
    for state in env.terminal_states:
        value_function[state] = 0.0
    
    for iteration in range(max_iterations):
        delta = 0
        
        # For each state, compute the new value using Bellman update with expectations
        for state in env.states:
            if state in env.terminal_states:
                continue  # Skip terminal states
            
            # Compute the maximum expected value over all actions
            max_value = float('-inf')
            
            for action in env.actions:
                action_value = 0.0
                for prob, next_state, reward in env.get_transition_and_reward_probabilities(state, action):
                    next_state_value = value_function[next_state] if next_state in env.states else 0.0
                    action_value += prob * (reward + discount_factor * next_state_value)
                
                max_value = max(max_value, action_value)
            
            # Update value function with the maximum value found
            old_value = value_function[state]
            value_function[state] = max_value
            delta = max(delta, abs(old_value - value_function[state]))
        
        # Check for convergence
        if delta < tolerance:
            break
    
    # Extract optimal policy from the final value function
    policy = {}
    for state in env.states:
        if state in env.terminal_states:
            continue  # Skip terminal states
        
        # Find the best action that maximizes the expected value function
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
    
    # Run value iteration
    policy, values = value_iteration(env)
    
    # Display results
    print_policy_and_values(env, policy, values)