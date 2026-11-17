"""Solution implementation for Chapter 10 - Q(lambda).

Reference: Sutton & Barto (2nd Ed), Section 12.10 "Watkins's Q(lambda)
to Tree-Backup(lambda)" (p. 257); the chapter-numbering follows this
repository's task list (algorithm_10.1, assigned as "book Section 12.6",
i.e. the pointer to the Watkins's Q(lambda) section of the 2nd edition).

Implements:
- algorithm_10.1 - Watkins's Q(lambda), backward view (Section 12.10,
  Figure 12.12), on the Example 6.2 random walk. Watkins's Q(lambda) is
  the Q-learning (off-policy, max-target) control algorithm extended with
  eligibility traces, with the traces cut to zero after the first
  non-greedy action.
- Forward-view Watkins's Q(lambda)-return definition, for completeness
  (the theoretical companion to the backward-view implementation).
"""

from __future__ import annotations

import numpy as np

from chapter_06.solution import RandomWalkEnv


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    n = len(q_values)
    best = np.flatnonzero(q_values == q_values.max())
    probs = np.full(n, epsilon / n, dtype=np.float64)
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def _sample_action(state, policy_fn, rng) -> int:
    probs = np.asarray(policy_fn(state), dtype=np.float64)
    if probs.shape != (len(RandomWalkEnv.ACTIONS),):
        raise ValueError("policy_fn must return a vector of length 2")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("policy probabilities must sum to 1")
    return int(rng.choice(len(probs), p=probs))


def _is_greedy(q_values: np.ndarray, action_index: int) -> bool:
    """True iff action_index is in the argmax set of q_values."""
    return bool(q_values[action_index] == q_values.max())


def watkins_q_lambda(
    env: RandomWalkEnv,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    lam: float = 0.9,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    seed: int | None = None,
    trace: str = "replacing",
) -> tuple[dict, dict]:
    """algorithm_10.1: Watkins's Q(lambda), backward view. Book Section 12.10.

    Off-policy control with eligibility traces (Figure 12.12). Like
    Q-learning (Algorithm 6.2), the bootstrap target is
    R + gamma * max_a Q(S', a). Eligibility traces are bumped for the
    visited (S, A) pair — z <- 1 for replacing traces (default, matching
    the tabular practice and keeping z bounded) or z += 1 for accumulating
    traces — and decayed by gamma * lam after every step.

    Trace-cutting behavior (Watkins's rule): after the Q-update at a step
    whose behavior action was non-greedy with respect to Q(s, .) at that
    state (i.e., not in the argmax set; ties count as greedy, matching the
    epsilon-greedy tie-breaking), ALL eligibility traces are set to zero.
    Thus the TD error of a non-greedy step still propagates through any
    pre-existing traces at that same step, but no TD error can propagate
    backward across a non-greedy action: the series of component updates
    ends either with the end of the episode or with the first non-greedy
    action, whichever comes first (Figure 12.12). Greediness is evaluated
    with the Q-values from before the update at the current step. At
    lambda = 0 the trace always starts empty at each step, so the method
    reduces exactly to one-step Q-learning (Algorithm 6.2). Note: with
    accumulating traces this off-policy-with-traces method can be unstable
    at larger alpha; replacing traces are the safer default.

    Watkins's Q-learning off-policy control with eligibility traces is
    semi-gradient and uses the max bootstrap; with an undiscounted (gamma=1)
    task it is known to be unstable and diverges in practice for any
    reasonable alpha once traces are lengthened. The default gamma=0.99
    (rather than 1.0) keeps the backups contracting; Section 12.10 of the
    book itself motivates Tree-Backup(lambda) as a stable successor. Pass
    gamma=1.0 only with small lam/alpha.

    Args:
        env: A RandomWalkEnv (or compatible env with STATES/ACTIONS).
        n_episodes: Number of episodes to sample.
        alpha: Step size.
        lam: Trace decay parameter lambda in [0, 1].
        epsilon: Exploration rate of the epsilon-greedy behavior policy.
        gamma: Discount factor (default 0.99; see stability note above).
        seed: Optional seed.
        trace: "replacing" (z <- 1) or "accumulating" (z += 1).

    Returns:
        (policy, Q): policy[state] = probability vector over ACTIONS;
        q_table[(state, action)] = action value.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if trace not in ("replacing", "accumulating"):
        raise ValueError("trace must be 'replacing' or 'accumulating'")
    rng = np.random.default_rng(seed)
    pairs = [(s, a) for s in env.STATES for a in env.ACTIONS]
    q_table = {pair: 0.0 for pair in pairs}

    def behavior_probs(state):
        q = np.array([q_table[(state, a)] for a in env.ACTIONS])
        return _epsilon_greedy_probs(q, epsilon)

    for _ in range(n_episodes):
        traces = {pair: 0.0 for pair in pairs}
        state = env.reset()
        done = False
        while not done:
            q_before = np.array([q_table[(state, a)] for a in env.ACTIONS])
            action_index = _sample_action(state, behavior_probs, rng)
            greedy = _is_greedy(q_before, action_index)
            action = env.ACTIONS[action_index]
            pair = (state, action)
            reward, next_state, done = env.step(action)
            best_next = (
                0.0
                if done
                else max(q_table[(next_state, a)] for a in env.ACTIONS)
            )
            delta = reward + gamma * best_next - q_table[pair]
            if trace == "accumulating":
                traces[pair] += 1.0
            else:  # replacing: z(S, A) <- 1 for the visited pair
                traces[pair] = 1.0
            for p in pairs:
                q_table[p] += alpha * delta * traces[p]
            for p in pairs:
                traces[p] *= gamma * lam
            if not greedy:
                # Watkins's cut (Section 12.10, Figure 12.12): after a
                # non-greedy action, all traces are zeroed.
                for p in pairs:
                    traces[p] = 0.0
            state = next_state if not done else state

    policy = {
        s: _epsilon_greedy_probs(
            np.array([q_table[(s, a)] for a in env.ACTIONS]), epsilon
        )
        for s in env.STATES
    }
    return policy, q_table


def watkins_lambda_return(
    rewards: list,
    bootstrap_values: list,
    greedy: list,
    gamma: float = 1.0,
    lam: float = 0.9,
) -> list:
    """Forward-view Watkins's Q(lambda)-return. Book Section 12.10.

    The theoretical companion to `watkins_q_lambda`: for each time t of one
    episode, computes the trace-cut lambda-return used to update (S_t, A_t)

        G^lambda_t = (1 - lam) * sum_{n=1}^{n_max} lam^(n-1) * G_{t:t+n}
                     + lam^n_max * G_{t:t+n_max},

    where G_{t:u} = sum_{i=t}^{u-1} gamma^(i-t) * rewards[i]
    + gamma^(u-t) * bootstrap_values[u - 1], and the horizon n_max is determined
    by the first non-greedy action: n_max = j - t + 1 with j the first index
    j >= t with greedy[j] False, or n_max = len(rewards) - t if all actions
    are greedy (Section 12.10, Figure 12.12: component updates end either at
    the episode's end or at the first non-greedy action, whichever comes
    first). For lam = 1 the result is the lambda-return to that horizon;
    for lam = 0 it reduces to the one-step Q-learning return G_{t:t+1}.

    Args:
        rewards: rewards R_{t+1}..R_T of one episode (0-based list).
        bootstrap_values: max_a Q(s_k, .) for k = 1..len(rewards);
            bootstrap_values[-1] should be 0.0 for a terminal final state.
        greedy: greedy[k] is True iff the behavior action A_k at step k was
            greedy w.r.t. Q(s_k, .).
        gamma: Discount factor.
        lam: Trace decay parameter lambda in [0, 1].

    Returns:
        list of G^lambda_t for t = 0..len(rewards)-1.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if len(bootstrap_values) != len(rewards):
        raise ValueError("bootstrap_values must have len(rewards) entries")
    if len(greedy) != len(rewards):
        raise ValueError("greedy must have the same length as rewards")
    n_steps = len(rewards)

    def n_step_return(t: int, u: int) -> float:
        total = 0.0
        for i in range(t, u):
            total += (gamma ** (i - t)) * rewards[i]
        total += (gamma ** (u - t)) * bootstrap_values[u - 1]
        return total

    returns = []
    for t in range(n_steps):
        first_nongreedy = next((j for j in range(t, n_steps)
                                if not greedy[j]), None)
        if first_nongreedy is None:
            n_max = n_steps - t
        else:
            n_max = first_nongreedy - t + 1
        g_lambda = (1.0 - lam) * sum(
            (lam ** (n - 1)) * n_step_return(t, t + n) for n in range(1, n_max + 1)
        ) + (lam ** n_max) * n_step_return(t, t + n_max)
        returns.append(g_lambda)
    return returns
