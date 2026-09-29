#!/usr/bin/env python3
"""
GridWorld Environment Module

Implements a simple grid-world environment for reinforcement learning.
"""

import numpy as np


class GridWorldEnv:
    """
    A simple grid-world environment for testing RL algorithms.
    
    Attributes:
        size (int): Size of the grid (size x size)
        states (list): List of all non-terminal states
        terminal_states (set): Set of terminal state positions
        actions (list): List of available actions
        stochastic (bool): Whether actions are stochastic
        stochasticity_factor (float): Probability of unintended action
    """

    def __init__(self, size=4, stochastic=False, stochasticity_factor=0.2):
        self.size = size
        self.stochastic = stochastic
        self.stochasticity_factor = stochasticity_factor
        
        # Define all possible states (including terminal)
        self.all_states = [(i, j) for i in range(size) for j in range(size)]
        
        # Terminal states at opposite corners (top-left and bottom-right)
        self.terminal_states = {(0, 0), (size - 1, size - 1)}
        
        # Non-terminal states exclude terminal states
        self.states = [s for s in self.all_states if s not in self.terminal_states]
        
        # Define actions
        self.actions = ['up', 'down', 'left', 'right']
        
        # Action transitions (deterministic)
        self.action_transitions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def get_transition_and_reward_probabilities(self, state, action):
        """
        Get full probability distribution of next states and rewards for a given state-action pair.
        
        Args:
            state (tuple): Current state (row, col)
            action (str): Intended action
            
        Returns:
            list: List of tuples [(prob, next_state, reward), ...] representing all possible outcomes
                  and their probabilities. For terminal states, returns [(1.0, state, 0)].
        """
        # Terminal states absorb with zero transitions
        if state in self.terminal_states:
            return [(1.0, state, 0.0)]
        
        results = []
        
        if self.stochastic and len(self.actions) > 1:
            s = self.stochasticity_factor
            num_other_actions = len(self.actions) - 1
            
            # Intended action probability (1-s)
            intended_prob = 1.0 - s
            dr, dc = self.action_transitions[action]
            next_row = max(0, min(self.size - 1, state[0] + dr))
            next_col = max(0, min(self.size - 1, state[1] + dc))
            next_state = (next_row, next_col)
            reward = -1.0  # Every step incurs a cost, including final transition to terminal
            results.append((intended_prob, next_state, reward))
            
            # Other actions get s/(n-1) each
            other_actions = [a for a in self.actions if a != action]
            prob_per_other = s / num_other_actions
            
            for other_action in other_actions:
                dr, dc = self.action_transitions[other_action]
                next_row = max(0, min(self.size - 1, state[0] + dr))
                next_col = max(0, min(self.size - 1, state[1] + dc))
                next_state = (next_row, next_col)
                reward = -1.0  # Every step incurs a cost, including final transition to terminal
                results.append((prob_per_other, next_state, reward))
        else:
            # Deterministic: single outcome with probability 1.0
            dr, dc = self.action_transitions[action]
            next_row = max(0, min(self.size - 1, state[0] + dr))
            next_col = max(0, min(self.size - 1, state[1] + dc))
            next_state = (next_row, next_col)
            reward = -1.0  # Every step incurs a cost, including final transition to terminal
            results.append((1.0, next_state, reward))
        
        return results

    def step(self, state, action):
        """
        Execute an action in the given state.
        
        Args:
            state (tuple): Current state (row, col)
            action (str): Action to take
            
        Returns:
            tuple: (next_state, reward, done)
        """
        if state in self.terminal_states:
            return state, 0.0, True
        
        # Calculate intended movement
        dr, dc = self.action_transitions[action]
        next_row = max(0, min(self.size - 1, state[0] + dr))
        next_col = max(0, min(self.size - 1, state[1] + dc))
        
        # Handle stochastic actions: (1-s) prob for intended, s/(n-1) for each remaining action
        if self.stochastic and len(self.actions) > 1:
            s = self.stochasticity_factor
            num_other_actions = len(self.actions) - 1
            
            rand = np.random.random()
            if rand >= s:
                # Execute intended action (probability 1-s)
                stochastic_action = action
            else:
                # Choose one of the other actions uniformly (each with probability s/(n-1))
                other_actions = [a for a in self.actions if a != action]
                stochastic_action = np.random.choice(other_actions)
            
            dr, dc = self.action_transitions[stochastic_action]
            next_row = max(0, min(self.size - 1, state[0] + dr))
            next_col = max(0, min(self.size - 1, state[1] + dc))
        else:
            # Deterministic actions
            dr, dc = self.action_transitions[action]
            next_row = max(0, min(self.size - 1, state[0] + dr))
            next_col = max(0, min(self.size - 1, state[1] + dc))
        
        next_state = (next_row, next_col)
        
        # Reward: -1 for each step, 0 at terminal state
        reward = -1.0  # Every step incurs a cost, including final transition to terminal
        
        done = next_state in self.terminal_states
        
        return next_state, reward, done

    def reset(self):
        """Reset the environment to initial state (non-terminal corner)."""
        # Start from a non-terminal state (avoiding terminal states)
        start_state = (0, 1) if self.size > 1 else (0, 0)
        return start_state
