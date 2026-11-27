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

    Step mapping (rl_envs.gridworld.GridWorld):
        Step 1  → GridWorld.__init__(): Create 4x4 grid
        Step 2  → GridWorld.__init__(): Defines A, B, A+, C, D coordinates
        Step 3  → GridWorld.UP/DOWN/LEFT/RIGHT: Action constants
        Step 4  → GridWorld.step(action): Compute s', r, done
        Step 4a → GridWorld.get_next_state(s, a): Transition
        Step 4b → GridWorld.get_reward(s, a): Reward computation
        Step 4c → GridWorld.is_terminal(): Check absorbing state
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

    Step mapping (implemented pieces are split across real classes):
        Step 1  → rl_envs.gridworld.RandomAgent(n_actions=4): uniform policy
        Step 2  → chapter_01/exercise_1_1.py:main(): interaction loop
        Step 2a → GridWorld.reset()/step(): current state
        Step 2b → GridWorld.is_terminal(): terminal check
        Step 2c → RandomAgent.select_action(state, epsilon): sample action
        Step 2d → GridWorld.step(action): observe r, s', done
        Step 2e → GreedyAgent.update_value(state, action, value): value update
        Step 2f → chapter_01/exercise_1_1.py:main(): timestep loop

    Note: the discounted return (Eq. 1.1) and the full TD update are previewed
    here; the complete value-iteration implementation lives in chapter_04.
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

    Step mapping (implemented in chapter_04.solution, not a ValueIteration class):
        Step 1  → chapter_04.solution.value_iteration(): V(s) starts at 0
        Step 2  → chapter_04.solution.value_iteration(): sweep loop
        Step 2a → value_iteration(): delta = 0
        Step 2b → value_iteration(): for each state s
        Step 2c → value_iteration(): compute q for each action via MDP dynamics
        Step 2d → value_iteration(): V(s) ← max_a q(s,a)
        Step 2e → value_iteration(): delta = max(delta, |old - best|)
        Step 2f → value_iteration(): check delta < theta
        Final  → chapter_04.solution.policy_improvement(): greedy policy
    """
    pass
