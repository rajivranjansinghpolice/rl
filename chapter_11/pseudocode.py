"""
Pseudocode for Chapter 11 - Off-policy Methods with Approximation.

Reference: Sutton & Barto (2nd Ed), Chapter 11
"""

# Book Step Mapping Table
# | Book Step | Description                        | Code (chapter_11/solution.py)       |
# |-----------|------------------------------------|-------------------------------------|
# | 11.2      | Semi-gradient off-policy TD(0)     | SemiGradientOffPolicyAgent          |
# | 11.1      | Baird-style hub counterexample     | baird_* / behavior_stationary       |
# | 11.3/11.4 | MSPBE gradient correction          | mspbe / mspbe_gradient_descent      |
# | 11.3      | Projected Bellman solution (r = 0) | projected_bellman_solution          |


# ============================================================================
# Algorithm 11.1: Semi-gradient off-policy TD(0) with importance sampling
# ============================================================================
# Book Section 11.2 / the deadly triad
# Pseudocode
# 1. Initialize w arbitrarily (non-zero)
# 2. S <- any starting state
# 3. Repeat (for each step, forever):
#    A <- behavior-policy action, observe R, S'
#    rho <- pi(A|S) / b(A|S)
#    delta <- R + gamma * (Phi S')^T w - (Phi S)^T w
#    w <- w + alpha * rho * delta * Phi(S)
#    S <- S'
# NOTE: NOT a true gradient of any objective; may diverge (deadly triad).
# ============================================================================


def pseudocode_semi_gradient_off_policy_td():
    """Pseudocode for semi-gradient off-policy TD(0). Book Step 11.2."""
    pass


# ============================================================================
# Example 11.1: Baird-style hub counterexample
# ============================================================================
# 1. 7-state asymmetric star: center state + 6 outer states.
# 2. Action A: outer -> center (target policy: always A).
# 3. Action B: outer -> outer_1 hub (behavior favors B strongly, b(A) = 0.02).
# 4. Reward is 0 everywhere; true value is 0.
# 5. The behavior state distribution concentrates on outer_1, making the
#    expected semi-gradient map indefinite -> iteration diverges.
# 6. The exact fixed point is computed via projected Bellman equations.
# ============================================================================


def pseudocode_baird_counterexample():
    """Pseudocode for the hub counterexample. Book Step 11.1."""
    pass


# ============================================================================
# MSPBE and gradient correction (Sections 11.3/11.4)
# ============================================================================
# MSPBE(w) = ||Pi T (Phi w) - Phi w||^2_{D_mu}
#          = (Bw - Phi w)^T D (Bw - Phi w),   B = gamma * Phi (Phi^T D Phi)^{-1} Phi^T D P^pi Phi
# Gradient descent: w <- w - lr * 2 * (B - Phi)^T D (B - Phi) w
# For the quadratic MSPBE this is guaranteed to decrease; conditioning
# (eigenvalues 0.0033 vs 20) makes convergence slow in the small directions.
# ============================================================================


def pseudocode_mspbe_gradient_descent():
    """Pseudocode for MSPBE gradient correction. Book Step 11.3/11.4."""
    pass
