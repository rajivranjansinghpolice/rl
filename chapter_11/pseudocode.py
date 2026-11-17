"""
Pseudocode for Chapter 11 - Exploration.

Mapping to Sutton & Barto book pseudocode:
- Each algorithm's steps are annotated with "Book Step X.Y" comments.
- A mapping table at the top shows step -> code correspondence.

Reference: Sutton & Barto (2nd Ed), Chapter 2 (multi-armed bandits) applied
in repo Chapter 11
"""

# Book Step Mapping Table
# | Book Step | Description                          | Code (chapter_11/solution.py) |
# |-----------|--------------------------------------|-------------------------------|
# | 2.3       | 10-armed bandit testbed              | TenArmedBandit                |
# | 2.4       | Sample-average action values         | (all *_bandit runners)        |
# | 2.6       | Optimistic initial values (Ex 11.1)  | run_optimistic_greedy_bandit  |
# | 2.6       | Optimistic Q-learning on an MDP      | q_learning_optimistic         |
# | 2.7       | UCB action selection (Alg 11.1)      | ucb_bonus, ucb_select         |
# | 2.7       | UCB bandit runner (Figure 2.4)       | run_ucb_bandit                |
# | 2.7       | UCB Q-learning on an MDP (Alg 11.1)  | q_learning_ucb                |
# | 2.2       | Epsilon-greedy baseline              | run_epsilon_greedy_bandit     |
# | Ex 11.1   | Descending optimistic (OFU) bonus    | descending_bonus              |
# | Ex 11.1   | Descending-bonus bandit runner       | run_descending_bonus_bandit   |
# | Ex 11.1   | Descending-bonus Q-learning          | q_learning_descending_bonus   |


# ============================================================================
# Algorithm 11.1 (repo) / Book Section 2.7: UCB action selection
# ============================================================================
# Book Section 2.7: Upper-Confidence-Bound Action Selection
# Eq. (2.8):  A_t = argmax_a [ Q_t(a) + c * sqrt(ln t / N_t(a)) ]
# - lnt is the natural logarithm of t (number of action selections so far)
# - N_t(a) is the number of times a was selected prior to time t
# - c > 0 controls the degree of exploration
# - If N_t(a) = 0, a is considered to be a maximizing action
# Bandit runner: keep sample-average Q(a) (book eq. (2.1) incrementally),
# matching Figure 2.4 (UCB c = 2 vs epsilon-greedy epsilon = 0.1).
# MDP runner: UCB replaces the epsilon-greedy step-size exploration of
# tabular Q-learning on the same small gridworld (book Section 6.5 update).
# ============================================================================


def pseudocode_ucb_exploration():
    """Pseudocode for UCB action selection. Book Step 2.7 (repo Alg 11.1)."""
    pass


# ============================================================================
# Exercise 11.1 (repo) / Book Section 2.6: Optimism in the face of uncertainty
# ============================================================================
# Book Section 2.6: Optimistic Initial Values
# - Set Q1(a) = +5 for all a on the 10-armed testbed (q*(a) ~ N(0, 1)):
#   wildly optimistic, so each greedy choice is "disappointing" and the
#   agent switches arms until all have been tried several times.
# - Greedy method with Q1 = +5 vs epsilon-greedy (epsilon = 0.1, Q1 = 0):
#   the optimistic method explores more early, then outperforms (Fig 2.3).
# - Optimistic Q-learning: initialize Q(s, a) optimistically high on an MDP
#   so greedy action selection tries every action before settling.
# - Optimistic greedy in the face of uncertainty with a *descending* bonus:
#   add b(N) = b0 / (1 + N) to Q(s, a) with N the visit count; the bonus is
#   maximal for untried actions and self-terminates as they are tried.
# ============================================================================


def pseudocode_optimistic_initialization():
    """Pseudocode for optimistic initialization / OFU. Book Step 2.6 (repo Ex 11.1)."""
    pass
