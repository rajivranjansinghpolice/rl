"""
Pseudocode for Chapter 1 — Gridworld MDP (Section 1.3, Figure 1.2)

This file maps each pseudocode step directly to Sutton & Barto (2nd Edition),
with clear line-by-line correspondence comments.

Book Reference: Sutton & Barto, Section 1.3 "An Example Agent" — The Gridworld Example
"""

# ============================================================================
# Section 1.3: The Gridworld Example
# ============================================================================
#
# From the book (Section 1.3):
#
# "In the gridworld example, the agent is concerned with selecting a sequence
#  of actions to maximize its long-term return. The agent receives observations
#  about its current location (its state) and the reward associated with each
#  transition. The agent's goal is to maximize the expected return."
#
# The gridworld is a 4×4 grid. States are (row, col) pairs from (0,0) to (3,3).
# States A and B are special: from A, best action → A+ (+10 reward); from B,
# actions go to C or D (+5 reward). A+ and D are terminal (absorbing) states.
# ============================================================================


# ----------------------------------------------------------------------------
# PSEUDOCODE: Gridworld MDP Definition
# ----------------------------------------------------------------------------
#
# Book Section 1.3, Figure 1.2
#
# Algorithm: Gridworld Environment
# --------------------------------
# 1.  Initialize the gridworld as a 4×4 grid of states
# 2.  Define the special states:
# 2a. State A: located at (0, 0)
# 2b. State B: located at (0, 3)
# 2c. State A+: terminal state (absorbing), reward +10
# 2d. State D: terminal state (absorbing), reward 0
# 3.  Define the action set: {up, down, left, right}
# 4.  For each non-terminal state s and action a:
# 4a.  Compute the next state s' based on deterministic transitions
# 4b.  Compute the reward r(s, a, s')
# 4c.  If s' is a terminal state, mark episode as complete
# 5.  Return (s', r) for the agent to observe
# ----------------------------------------------------------------------------

def pseudocode_gridworld_mdp():
    """
    Pseudocode mapping for Gridworld MDP.

    Step mapping:
        Step 1  → Gridworld.__init__(): Create 4x4 grid
        Step 2  → Gridworld._define_special_states(): Define A, B, A+, D
        Step 3  → Gridworld.ACTIONS: {up, down, left, right}
        Step 4  → Gridworld.step(action): Compute s', r
        Step 4a → Gridworld._get_next_state(s, a): Deterministic transition
        Step 4b → Gridworld._get_reward(s, a, s'): Reward computation
        Step 4c → Gridworld._is_terminal(s'): Check absorbing state
    """
    pass


# ----------------------------------------------------------------------------
# PSEUDOCODE: Agent Policy in Gridworld
# ----------------------------------------------------------------------------
#
# Book Section 1.3, Equation 1.1 (discounted return)
#
# Algorithm: Agent Policy Selection
# ----------------------------------
# 1.  Initialize the agent's policy π(a|s) as a stochastic policy
# 2.  At each time step t:
# 2a.  Receive observation s_t (current state)
# 2b.  If s_t is terminal, reset environment and continue
# 2c.  Select action a_t from π(·|s_t)
# 2d.  Execute action a_t, observe reward r_{t+1} and next state s_{t+1}
# 2e.  Update the value estimate V(s_t) ← V(s_t) + α [r_{t+1} + γV(s_{t+1}) - V(s_t)]
# 2f.  Set t ← t + 1 and repeat
# ----------------------------------------------------------------------------

def pseudocode_agent_policy():
    """
    Pseudocode mapping for Agent Policy in Gridworld.

    Step mapping:
        Step 1  → Agent.__init__(): Initialize stochastic policy (uniform)
        Step 2  → Agent.step(s_t): Main interaction loop
        Step 2a → Agent.step(s_t): Receive current state
        Step 2b → Agent._is_terminal(s_t): Check terminal state
        Step 2c → Agent._select_action(s_t): Sample from policy
        Step 2d → Agent.step(s_t): Execute action, observe r, s'
        Step 2e → Agent.update_value(s_t, r, s_t_next): TD update
        Step 2f → Agent.step(s_t): Increment timestep
    """
    pass


# ----------------------------------------------------------------------------
# PSEUDOCODE: Value Iteration for Gridworld (preview, Chapter 4)
# ----------------------------------------------------------------------------
#
# This is included as a forward reference. Full implementation in chapter_04.
#
# Algorithm: Value Iteration
# ---------------------------
# 1.  Initialize V(s) ← 0 for all states s (arbitrary initialization)
# 2.  Repeat until convergence:
# 2a.  Δ ← 0
# 2b.  For each state s in all_states:
# 2c.      v ← V(s)
# 2d.      V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
# 2e.      Δ ← max(Δ, |v - V(s)|)
# 2f.  If Δ < θ, stop
# ----------------------------------------------------------------------------

def pseudocode_value_iteration():
    """
    Pseudocode mapping for Value Iteration (Chapter 4, Section 4.1).

    Step mapping:
        Step 1  → ValueIteration.__init__(): Initialize V(s) = 0
        Step 2  → ValueIteration.solve(): Iteration loop
        Step 2a → ValueIteration.solve(): Δ = 0
        Step 2b → ValueIteration.solve(): For each state s
        Step 2c → ValueIteration.compute_q_values(s): Compute Q(s,a) for all a
        Step 2d → ValueIteration.solve(): V(s) ← max_a Q(s,a)
        Step 2e → ValueIteration.solve(): Δ = max(Δ, |ΔV|)
        Step 2f → ValueIteration.solve(): Check convergence θ
    """
    pass
