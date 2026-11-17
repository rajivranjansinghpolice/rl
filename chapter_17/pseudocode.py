"""
Pseudocode for Chapter 17 - Frontiers.

Mapping to Sutton & Barto book text:
- A mapping table at the top shows step -> code correspondence.
- Chapter 17 is a discussion of open problems; its two sections with
  implementable, demonstrable content are 17.2 (Temporal Abstraction via
  Options) and 17.4 (Designing Reward Signals, incl. intrinsic motivation).

Reference: Sutton & Barto (2nd Ed), Chapter 17 (Frontiers)
"""

# Book Step Mapping Table
# | Book Step | Description                              | Code (chapter_17/solution.py)   |
# |-----------|------------------------------------------|----------------------------------|
# | 17.2      | Option omega = <pi, termination>         | Option                           |
# | 17.2      | Options extend the action space          | smdp_q_learning                  |
# | 17.2      | SMDP Bellman update (gamma ** tau)       | smdp_q_learning                  |
# | Ch 6      | Primitive-only Q-learning baseline       | primitive_q_learning             |
# | 17.4      | Novelty bonus R+ = R + beta / sqrt(N+1)  | novelty_bonus                    |
# | 17.4      | Count-based exploration agent            | q_learning_novelty               |
# | 17.2/17.4 | Shared 4-room GridWorld                  | GridWorld                        |


# ============================================================================
# Algorithm 17.1: Options / hierarchical RL demo (Section 17.2)
# Book Section 17.2 (Temporal Abstraction via Options). Structure:
# 1. Define options omega = <pi_omega, gamma_omega>: pi_omega a policy over
#    primitive actions, gamma_omega a termination function. Here: BFS-greedy
#    navigation to a subgoal cell; terminates on arrival (or on a step cap).
# 2. Options extend the action space: action indices [0, 4) are primitive
#    moves, [4, 4 + len(options)) are options executed to termination.
# 3. SMDP Q-learning update:
#        Q(s, o) <- Q(s, o) + alpha * [R + gamma**tau * max_a Q(s', a)
#                                       - Q(s, o)]
#    where R is the discounted return over the option and tau its duration.
# 4. A primitive-only Q-learning agent is trained on the same grid as a
#    baseline; the comparison is decisions-to-goal and moves-to-goal.
# ============================================================================


def pseudocode_options():
    """Pseudocode for the options / hierarchical RL demo. Book Step 17.2."""
    pass


# ============================================================================
# Algorithm 17.2: Intrinsic motivation via count-based novelty bonus
# Book Section 17.4 (Designing Reward Signals). Structure:
# 1. Maintain state visit counts N(s) across the whole run.
# 2. Intrinsic reward: R+ = R + beta / sqrt(N(s') + 1) for transitions into
#    s' (no-op transitions harvest no bonus).
# 3. Q-learning with the shaped reward (gamma = 0 here); novelty-seeking
#    behaviour is greedy w.r.t. value including the bonus.
# ============================================================================


def pseudocode_novelty():
    """Pseudocode for intrinsic motivation via count-based novelty. Book Step 17.4."""
    pass
