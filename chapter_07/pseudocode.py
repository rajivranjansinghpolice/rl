"""
Pseudocode for Chapter 7 - n-step Bootstrapping.

Mapping to Sutton & Barto book pseudocode:
- Each algorithm's steps are annotated with "Book Step X.Y" comments.
- A mapping table at the top shows step -> code correspondence.

Reference: Sutton & Barto (2nd Ed), Chapter 7
"""

# Book Step Mapping Table
# | Book Step | Description             | Code (chapter_07/solution.py)     |
# |-----------|-------------------------|-----------------------------------|
# | 7.1       | n-step TD prediction    | n_step_td_prediction              |
# | 7.2       | n-step Sarsa            | n_step_sarsa                      |
# | 7.5       | n-step Tree Backup      | tree_backup                       |
# | 12.2-acc  | SARSA(lambda) (Sarsa-lambda) | sarsa_lambda                   |


# ============================================================================
# Box: n-step TD prediction (Sec. 7.1)
# ============================================================================
# Book Section 7.1: n-step TD for estimating V ~= v_pi
# Pseudocode: n-step TD prediction for the equiprobable random-walk MRP.
# 1. Initialize V(s) arbitrarily for all s   (Example 7.1 initializes to 0)
# 2. Repeat (for each episode):
#    a. Initialize and store S_0 != terminal
#    b. T <- infinity
#    c. For t = 0, 1, ... :
#       | If t < T: take an action according to pi(.|S_t); observe and store
#       |   the next reward as R_{t+1} and the next state as S_{t+1}
#       |   If S_{t+1} is terminal: T <- t + 1
#       |   tau <- t - n + 1
#       |   If tau >= 0:
#       |     G <- sum_{i=tau+1}^{min(tau+n, T)} gamma^{i-tau-1} * R_i
#       |     if tau + n < T: G <- G + gamma^n * V(S_{tau+n})
#       |     V(S_tau) <- V(S_tau) + alpha * (G - V(S_tau))
#    until tau = T - 1
# ============================================================================


def pseudocode_n_step_td_prediction():
    """Pseudocode for n-step TD prediction. Book Sec. 7.1."""
    pass


# ============================================================================
# Box: n-step Sarsa (Sec. 7.2)
# ============================================================================
# Book Section 7.2: n-step Sarsa
# Pseudocode: n-step Sarsa for estimating Q ~= q*
# 1. Initialize Q(s,a) arbitrarily for all s, a; Q(terminal, .) = 0
# 2. Initialize pi to be epsilon-greedy w.r.t. Q, or a fixed policy
# 3. Parameters: step size alpha, small epsilon > 0, integer n
# 4. Repeat (for each episode):
#    a. Initialize and store S_0 != terminal; select and store A_0 ~ pi(.|S_0)
#    b. T <- infinity
#    c. For t = 0, 1, ... :
#       | If t < T: take action A_t; observe R_{t+1}, S_{t+1}
#       | If S_{t+1} is terminal: T <- t+1
#       | else: A_{t+1} <- sample from pi(.|S_{t+1})
#       | tau <- t - n + 1
#       | If tau >= 0:
#       |   G <- sum_{i=tau+1}^{min(tau+n, T)} gamma^{i-tau-1} * R_i
#       |   if tau + n < T: G <- G + gamma^n * Q(S_{tau+n}, A_{tau+n})
#       |   Q(S_tau, A_tau) <- Q(S_tau, A_tau) + alpha * (G - Q(S_tau, A_tau))
#       |   if pi is being learned: ensure pi(.|S_tau) is epsilon-greedy w.r.t. Q
#    until tau = T - 1
# ============================================================================


def pseudocode_n_step_sarsa():
    """Pseudocode for n-step Sarsa. Book Sec. 7.2."""
    pass


# ============================================================================
# Box: n-step Tree Backup algorithm (Sec. 7.5)
# ============================================================================
# Book Section 7.5: The n-step Tree Backup Algorithm (off-policy without IS)
# Pseudocode: n-step Tree Backup for estimating Q ~= q* (or q_pi for given pi)
# 1. Initialize Q(s,a) arbitrarily for all s, a
# 2. Initialize pi to be epsilon-greedy w.r.t. Q, or a fixed policy
# 3. Parameters: step size alpha, small epsilon > 0, integer n
# 4. Repeat (for each episode):
#    a. Initialize and store S_0 != terminal; select A_0 ~ pi(.|S_0)
#    b. Store Q(S_0, A_0) as Q_0
#    c. T <- infinity
#    d. For t = 0, 1, ... :
#       | If t < T:
#       |   Take action A_t; observe R; observe and store S_{t+1}
#       |   If S_{t+1} is terminal: T <- t+1
#       |   Store delta_t <- R - Q_t            (terminal)
#       |   else: delta_t <- R + gamma * sum_a pi(a|S_{t+1}) Q(S_{t+1}, a) - Q_t
#       |   Select and store an action A_{t+1}; store Q(S_{t+1}, A_{t+1}) as Q_{t+1}
#       |   Store pi(A_{t+1}|S_{t+1}) as pi_{t+1}
#       | tau <- t - n + 1
#       | If tau >= 0:
#       |   Z <- 1; G <- Q_tau
#       |   For k = tau ... min(tau+n-1, T-1):
#       |   |   G <- G + Z * delta_k
#       |   |   Z <- gamma * Z * pi_{k+1}
#       |   Q(S_tau, A_tau) <- Q(S_tau, A_tau) + alpha * (G - Q(S_tau, A_tau))
# ============================================================================


def pseudocode_tree_backup():
    """Pseudocode for n-step Tree Backup. Book Sec. 7.5."""
    pass


# ============================================================================
# Box: SARSA(lambda) control (Sec. 12.2-style)
# ============================================================================
# Book Section 12.2 (traces), used in Sec. 7.2 for bootstrapping control
# Pseudocode: SARSA(lambda), for estimating Q ~= q*, epsilon-greedy
# 1. Initialize Q(s,a) arbitrarily for all s, a
# 2. Repeat (for each episode):
#    a. z(s,a) <- 0, for all s, a                (eligibility trace)
#    b. S <- initial state; A <- epsilon-greedy w.r.t. Q(S, .)
#    c. Repeat (for each step of episode):
#       Take action A; observe R, S'
#       A' <- epsilon-greedy w.r.t. Q(S', .)
#       delta <- R + gamma * Q(S', A') - Q(S, A)
#       z(S, A) <- z(S, A) + 1
#       Q(s, a) <- Q(s, a) + alpha * delta * z(s, a)   for all s, a
#       z(s, a) <- gamma * lambda * z(s, a)            for all s, a
#       S <- S'; A <- A'
#    until S is terminal
# ============================================================================


def pseudocode_sarsa_lambda():
    """Pseudocode for SARSA(lambda). Book Sec. 12.2-style."""
    pass
