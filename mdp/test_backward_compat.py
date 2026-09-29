#!/usr/bin/env python3
"""Backward compatibility test - ensure old usage patterns still work."""

import sys
sys.path.insert(0, '/home/romeo/projects/rl')

from gridworld_env import GridWorldEnv
from agents.policy_iteration import policy_iteration
from agents.value_iteration import value_iteration

# Test that the main functions still work with default arguments
print("Testing backward compatibility...")

env_det = GridWorldEnv(size=4, stochastic=False)
policy, values = policy_iteration(env_det)
assert len(policy) == 14 and len(values) == 16
print("✓ Policy Iteration works (deterministic)")

env_stoch = GridWorldEnv(size=4, stochastic=True)
policy, values = value_iteration(env_stoch)
assert len(policy) == 14 and len(values) == 16
print("✓ Value Iteration works (stochastic)")

# Verify no API changes - old code using these imports will still work
print("\n✓ Backward compatibility verified!")
