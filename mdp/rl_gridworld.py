#!/usr/bin/env python3
"""
Reinforcement Learning Grid‑World Solver

This script implements a simple grid‑world environment and two classic
dynamic‑programming algorithms from Barto & Sutton:
    • Policy Iteration
    • Value Iteration

The environment is fully customizable via user input:
    • Grid size (e.g., 4 for a 4×4 grid)
    • Action model: deterministic or stochastic (80% intended, 10% left, 10% right)

After solving the MDP, the optimal policy and value function are printed
in a grid format with arrows indicating the best action at each state.
"""

import argparse
import sys
import signal
import time
import numpy as np
from gridworld_env import GridWorldEnv
from agents import policy_iteration, value_iteration


class TimeoutError(Exception):
    """Custom exception for timeout."""
    pass


def input_with_timeout(prompt, timeout=5):
    """Get user input with a timeout. Returns default if timeout or empty input."""
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        value = input(prompt).strip()
    except EOFError:
        value = ''
    except TimeoutError:
        print(f"\nTimeout after {timeout} seconds. Using default.")
        return None
    finally:
        signal.alarm(0)  # Cancel alarm
    
    return value if value else None


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def print_policy(policy, env):
    """Print the policy in a grid with arrows."""
    arrow_map = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}
    grid = []
    for i in range(env.size):
        row = []
        for j in range(env.size):
            s = (i, j)
            if s in env.terminal_states:
                row.append('T')
            else:
                row.append(arrow_map[policy[s]])
        grid.append(row)
    for row in grid:
        print(' '.join(row))
    print()

def print_values(values, env):
    """Print the value function in a grid."""
    grid = []
    for i in range(env.size):
        row = []
        for j in range(env.size):
            s = (i, j)
            row.append(f"{values[s]:6.2f}")
        grid.append(row)
    for row in grid:
        print(' '.join(row))
    print()

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Grid-World Solver"
    )
    parser.add_argument(
        "--size", "-s", type=int, default=4,
        help="Grid size (default: 4)"
    )
    parser.add_argument(
        "--stochastic", "-st", action="store_true", default=False,
        help="Use stochastic action model (default: deterministic)"
    )
    parser.add_argument(
        "--stochasticity-factor", type=float, default=0.2,
        help="Stochasticity factor s (default: 0.2)"
    )
    parser.add_argument(
        "--gamma", "-g", type=float, default=0.99,
        help="Discount factor gamma (default: 0.99)"
    )
    parser.add_argument(
        "--theta", "-t", type=float, default=1e-6,
        help="Convergence threshold theta (default: 1e-6)"
    )
    parser.add_argument(
        "--selected-agents", "-sa", type=str, default="2",
        help="Agents to run: '1' for Policy Iteration, '2' for Value Iteration, or '1,2' for both (default: 2)"
    )
    return parser.parse_args()


def main():
    # --- Parse command-line arguments or interactive input ----------------- #
    args = parse_args()
    
    # If no command-line arguments were provided, use interactive input
    if not sys.argv[1:]:
        print("\n=== Interactive Mode ===")
        print("Enter values or press Enter to use defaults (5 second timeout)\n")
        
        # Grid size
        size_input = input_with_timeout("Enter grid size (default 4): ")
        if size_input:
            try:
                args.size = int(size_input)
            except ValueError:
                print("Invalid input. Using default size 4.")
                args.size = 4
        
        # Action model
        action_input = input_with_timeout("Choose action model: (d)eterministic or (s)tochastic? [d/s] (default d): ")
        if action_input:
            args.stochastic = action_input.lower() == 's'
        
        # Stochasticity factor
        s_input = input_with_timeout("Enter stochasticity factor s (default 0.2): ")
        if s_input:
            try:
                args.stochasticity_factor = float(s_input)
                if not (0 <= args.stochasticity_factor <= 1):
                    print("Invalid value. Using default 0.2.")
                    args.stochasticity_factor = 0.2
            except ValueError:
                print("Invalid input. Using default 0.2.")
                args.stochasticity_factor = 0.2
        
        # Discount factor gamma
        gamma_input = input_with_timeout("Enter discount factor gamma (default 0.99): ")
        if gamma_input:
            try:
                args.gamma = float(gamma_input)
                if not (0 <= args.gamma <= 1):
                    print("Invalid value. Using default 0.99.")
                    args.gamma = 0.99
            except ValueError:
                print("Invalid input. Using default 0.99.")
                args.gamma = 0.99
        
        # Convergence threshold theta
        theta_input = input_with_timeout("Enter convergence threshold theta (default 1e-6): ")
        if theta_input:
            try:
                args.theta = float(theta_input)
                if args.theta <= 0:
                    print("Invalid value. Using default 1e-6.")
                    args.theta = 1e-6
            except ValueError:
                print("Invalid input. Using default 1e-6.")
                args.theta = 1e-6
        
        # Agent selection menu
        print("\n=== Select Agents to Run ===")
        print("Available agents:")
        print("  [1] Policy Iteration")
        print("  [2] Value Iteration (default)")
        print("  Enter numbers separated by commas (e.g., 1,2) or press Enter for default")
        
        agent_input = input_with_timeout("Select agents: ")
        if agent_input:
            try:
                selected_agents = [int(x.strip()) for x in agent_input.split(',')]
                # Validate selections
                valid_selections = []
                for sel in selected_agents:
                    if sel == 1:
                        valid_selections.append('policy_iteration')
                    elif sel == 2:
                        valid_selections.append('value_iteration')
                if not valid_selections:
                    print("Invalid selection. Using default (Value Iteration).")
                    args.selected_agents = ['value_iteration']
                else:
                    args.selected_agents = valid_selections
            except ValueError:
                print("Invalid input. Using default (Value Iteration).")
                args.selected_agents = ['value_iteration']
        else:
            args.selected_agents = ['value_iteration']
    
    # Validate inputs from command-line or interactive mode
    if args.size <= 0:
        print("Invalid grid size. Using default size 4.")
        args.size = 4
    if not (0 <= args.stochasticity_factor <= 1):
        print("Invalid stochasticity factor. Using default 0.2.")
        args.stochasticity_factor = 0.2
    if not (0 <= args.gamma <= 1):
        print("Invalid gamma value. Using default 0.99.")
        args.gamma = 0.99
    if args.theta <= 0:
        print("Invalid theta value. Using default 1e-6.")
        args.theta = 1e-6

    # Extract parameters for agents
    gamma = args.gamma
    theta = args.theta
    
    # Parse selected agents from CLI or interactive mode
    if not sys.argv[1:]:
        # Interactive mode - args.selected_agents already set by input prompts
        agent_selections = args.selected_agents
    else:
        # CLI mode - parse --selected-agents argument (value like "2", "1,2", "1")
        try:
            raw_value = str(args.selected_agents).strip()
            # Check if it's numeric or comma-separated integers
            agent_numbers = [int(x.strip()) for x in str(raw_value).split(',')]
            agent_selections = []
            for sel in agent_numbers:
                if sel == 1:
                    agent_selections.append('policy_iteration')
                elif sel == 2:
                    agent_selections.append('value_iteration')
            # Default to value_iteration if empty or invalid
            if not agent_selections:
                agent_selections = ['value_iteration']
        except (ValueError, TypeError):
            print(f"Invalid --selected-agents format. Using default (Value Iteration).")
            agent_selections = ['value_iteration']

    env = GridWorldEnv(
        size=args.size,
        stochastic=args.stochastic,
        stochasticity_factor=args.stochasticity_factor
    )

    # Run selected agents
    for agent_type in agent_selections:
        if agent_type == 'policy_iteration':
            print("\n--- Policy Iteration ---")
            policy_pi, V_pi = policy_iteration(env, discount_factor=gamma, tolerance=theta)
            print("Optimal Policy:")
            print_policy(policy_pi, env)
            print("Value Function:")
            print_values(V_pi, env)
        elif agent_type == 'value_iteration':
            print("\n--- Value Iteration ---")
            policy_vi, V_vi = value_iteration(env, discount_factor=gamma, tolerance=theta)
            print("Optimal Policy:")
            print_policy(policy_vi, env)
            print("Value Function:")
            print_values(V_vi, env)

if __name__ == "__main__":
    main()
    sys.exit(0)
