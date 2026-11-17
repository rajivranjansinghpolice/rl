"""
Pseudocode for Chapter 16 - Applications and Case Studies.

Mapping to Sutton & Barto book text:
- A mapping table at the top shows step -> code correspondence.
- Chapter 16 of the book surveys real-world applications (TD-Gammon,
  Samuel's checkers, Watson's wagering, memory control, deep RL, thermal
  soaring) and contains no numbered algorithms. The implementations here
  are classic small-scale control applications in the same spirit, built
  on the tabular algorithms of Chapters 6 and 7.

Reference: Sutton & Barto (2nd Ed), Chapter 16 (Applications and Case Studies)
"""

# Book Step Mapping Table
# | Book Step | Description                              | Code (chapter_16/solution.py)   |
# |-----------|------------------------------------------|----------------------------------|
# | 16.x      | Inventory MDP (order-up-to, Poisson dmnd)| InventoryEnv                     |
# | 6.4       | Sarsa on-policy TD control               | sarsa_inventory                  |
# | 6.2       | Constant-alpha bootstrap targets          | sarsa_inventory / q_learning_*   |
# | 16.x      | Single-server queue MDP w/ service speed | QueueEnv                         |
# | 6.5       | Q-learning off-policy TD control          | q_learning_queue                 |
# | 6.2       | Truncated episodes bootstrap (not zero)  | bellman_target                   |


# ============================================================================
# Algorithm 16.1: Inventory management with Sarsa
# Classic inventory-control MDP solved with the Sarsa algorithm
# (Book Chapter 6, Algorithm 6.4 flavor). Structure:
# 1. State: inventory level s in [0, S_max] at period start.
# 2. Action: order-up-to level a in [0, S_max]; order max(0, a - s) units.
# 3. Demand: Poisson(demand_rate); sales capped by position; unmet demand
#    lost. Per-period cost = c_order*units + h*leftover + p*unmet.
# 4. Reward = -cost. Sarsa (epsilon-greedy, constant alpha) learns
#    q(s, a). The time-limit truncation bootstraps (does not zero).
# ============================================================================


def pseudocode_inventory_sarsa():
    """Pseudocode for Sarsa-based inventory control. Book Step 6.4 flavor."""
    pass


# ============================================================================
# Algorithm 16.2: Single-server queue with service-speed decision
# Serve-based queue MDP with waiting-time targets; Q-learning
# (Book Chapter 6, Algorithm 6.2 flavor). Structure:
# 1. State: customers in system q in [0, Q_max] at period start.
# 2. Action: 0 = normal service (completion prob p_normal, cost c_normal),
#            1 = expedited service (prob p_expedited, cost c_expedited).
#    Serving an empty system is free and completes nothing.
# 3. Arrivals: Poisson(arrival_rate); overflow past Q_max is lost.
#    Waiting cost h per customer per period.
# 4. Q-learning (epsilon-greedy, constant alpha) learns q(q, action).
#    Truncation at the horizon bootstraps via bellman_target (terminal=
#    False) rather than treating the time limit as a true terminal state.
# ============================================================================


def pseudocode_queue_q_learning():
    """Pseudocode for the queue service-speed MDP. Book Step 6.2 flavor."""
    pass
