"""
Pseudocode for Chapter 6 - Temporal-Difference Learning.

Mapping to Sutton & Barto book pseudocode:
- Each algorithm's steps are annotated with "Book Step X.Y" comments.
- A mapping table at the top shows step -> code correspondence.

Reference: Sutton & Barto (2nd Ed), Chapter 6
"""

# Book Step Mapping Table
# | Book Step | Description                 | Code (chapter_06/solution.py)        |
# |-----------|-----------------------------|--------------------------------------|
# | 6.1       | TD(0) prediction            | td0_prediction                      |
# | 6.4       | Sarsa prediction / control   | sarsa                               |
# | 6.5       | Q-learning (off-policy TD)   | q_learning                          |
# | 7-adj     | TD(lambda) prediction        | td_lambda_prediction                |
# | Ex 6.1    | Random walk TD(0) vs MC       | run_random_walk_comparison          |


# ============================================================================
# Algorithm 6.1: TD prediction (TD(0))
# ============================================================================
# Book Section 6.1: TD Prediction
# Pseudocode: Tabular TD(0) for estimating v ~= v_pi
# 1. Initialize V(s) arbitrarily, for all s in S, and V(terminal) = 0
# 2. Repeat (for each episode):
#    a. S <- initial state of episode
#    b. Repeat (for each step of episode):
#       A <- action given by pi for S
#       Take action A, observe R, S'
#       V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
#       S <- S'
#    until S is terminal
# ============================================================================


def pseudocode_td0_prediction():
    """Pseudocode for TD(0) prediction. Book Step 6.1."""
    pass


# ============================================================================
# Algorithm 6.1 / 6.4: Sarsa (on-policy control)
# ============================================================================
# Book Section 6.4: Sarsa: On-Policy TD Control
# Pseudocode: Sarsa (on-policy TD control) for estimating Q ~= q*
# 1. Initialize Q(s,a) arbitrarily, for all s in S, a in A(s); Q(terminal,.)= 0
# 2. Repeat (for each episode):
#    a. S <- initial state of episode
#    b. Choose A from S using policy derived from Q (e.g., epsilon-greedy)
#    c. Repeat (for each step of episode):
#       Take action A, observe R, S'
#       Choose A' from S' using policy derived from Q (epsilon-greedy)
#       Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
#       S <- S'; A <- A'
#    until S is terminal
# ============================================================================


def pseudocode_sarsa():
    """Pseudocode for Sarsa. Book Step 6.4."""
    pass


# ============================================================================
# Algorithm 6.2: Q-learning (off-policy control)
# ============================================================================
# Book Section 6.5: Q-learning: Off-policy TD Control
# Pseudocode: Q-learning (off-policy TD control) for estimating Q ~= q*
# 1. Initialize Q(s,a) arbitrarily, for all s in S, a in A(s); Q(terminal,.)= 0
# 2. Repeat (for each episode):
#    a. S <- initial state of episode
#    b. Repeat (for each step of episode):
#       Choose A from S using policy derived from Q (e.g., epsilon-greedy)
#       Take action A, observe R, S'
#       Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]
#       S <- S'
#    until S is terminal
# ============================================================================


def pseudocode_q_learning():
    """Pseudocode for Q-learning. Book Step 6.5."""
    pass


# ============================================================================
# Algorithm 6.3: TD(lambda) prediction
# ============================================================================
# Book Section 7 (traces) / 12.1: TD(lambda), lambda-return (forward view)
# and eligibility traces (backward view) — here: one-step eligibility traces.
# Pseudocode: TD(lambda) with accumulating traces for estimating v ~= v_pi
# 1. Initialize V(s) arbitrarily, for all s in S
# 2. Repeat (for each episode):
#    a. z(s) <- 0, for all s in S  (eligibility trace)
#    b. S <- initial state of episode
#    c. Repeat (for each step of episode):
#       A <- action given by pi for S
#       Take action A, observe R, S'
#       delta <- R + gamma * V(S') - V(S)
#       z(S) <- z(S) + 1
#       V(s) <- V(s) + alpha * delta * z(s),  for all s in S
#       z(s) <- gamma * lambda * z(s),      for all s in S
#       S <- S'
#    until S is terminal
# ============================================================================


def pseudocode_td_lambda():
    """Pseudocode for TD(lambda) prediction. Book Step 6.3."""
    pass


# ============================================================================
# Exercise 6.1: Random walk with TD(0) and MC comparison
# ============================================================================
# Book Section 6.2 (Example 6.2): Random Walk
# 1. States A, B, C, D, E between two terminals; start in C; equiprob left/right.
# 2. Reward +1 on reaching the right end (terminal), 0 elsewhere; gamma = 1.
# 3. True values under the random policy: v(s) = (position+1)/6.
# 4. Compare empirical RMS error of TD(0) (alpha=0.1) vs MC (sample average)
#    averaged over 100 repetitions / 100 episodes.
# ============================================================================


def pseudocode_random_walk():
    """Pseudocode for the random-walk comparison. Book Step Ex 6.1."""
    pass
