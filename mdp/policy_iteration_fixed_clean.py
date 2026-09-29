#!/usr/bin/env python3
"""
Policy Iteration Implementation for GridWorld Environment (Fixed version)
"""

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
    policy = {state: env.actions[0] for state in env.states if state not in env.terminal_states}
    value_function = {state: 0.0 for state in env.states}
    
    # Set terminal states to zero value (they don't have transitions)
    for state in env.terminal_states:
        value_function[state] = 0.0
    
    for iteration in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            for state in env.states:
                if state in env.terminal_states:
                    continue  # Skip terminal states
                
                old_value = value_function[state]
                action = policy[state]
                next_state, reward, done = env.step(state, action)
                value_function[state] = reward + discount_factor * value_function[next_state]
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
            
            # Find best action based on current value function
            best_action = None
            best_value = float('-inf')
            
            for action in env.actions:
                next_state, reward, done = env.step(state, action)
                value = reward + discount_factor * value_function[next_state]
                
                if value > best_value:
                    best_value = value
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