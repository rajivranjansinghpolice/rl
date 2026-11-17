"""Solution implementation for Chapter 5 - Monte Carlo Methods.

Reference: Sutton & Barto (2nd Ed), Chapter 5

Implements: Alg 5.1 (first-visit / every-visit MC prediction),
Alg 5.3 (on-policy MC control, epsilon-greedy),
Alg 5.4 (off-policy control, ordinary importance sampling),
Alg 5.5 (off-policy control, weighted importance sampling),
Exercise 5.1 (Blackjack (21) environment + MC control).

Policies are probability vectors over ACTIONS; the prediction and control
functions take a policy as a callable ``policy_fn(state) -> probs`` (a
probability vector) and any environment implementing ``reset() -> observation``
and ``step(action) -> (reward, observation, done)`` — see ``BlackjackEnv``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ACTIONS = ("hit", "stick")
N_ACTIONS = len(ACTIONS)


def _card_value(rank: int) -> int:
    """Map a drawn rank (1..13) to its blackjack value (face cards = 10)."""
    return 10 if rank > 10 else rank


class BlackjackEnv:
    """Blackjack (21) environment (Example 5.1). Book Section 5.1.

    Observation: (player_sum, dealer_showing, usable_ace).
    Actions: 'hit' or 'stick'. Reward +1 win, -1 loss, 0 draw; gamma = 1.
    Cards are drawn from an infinite deck. The dealer hits on < 17 and sticks
    on >= 17. Naturals (21 on the deal) end the episode immediately: a player
    natural wins unless the dealer also has a natural (draw).
    """

    ACTIONS = ACTIONS

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._player_cards: list[int] = []
        self._dealer_cards: list[int] = []
        self._forced_reward: float | None = None

    @staticmethod
    def _hand_value(cards: list[int]) -> tuple[int, bool]:
        """Return (best_sum, usable_ace) for a hand. Aces count as 11 or 1."""
        total = sum(_card_value(c) for c in cards)
        aces = sum(1 for c in cards if c == 1)
        usable = aces > 0 and total + 10 <= 21
        total += aces * 10 if usable else 0
        return int(min(total, 99)), bool(usable)

    def _draw(self) -> int:
        return int(self.rng.integers(1, 14))

    def reset(self) -> tuple:
        """Reset; resolve naturals immediately if they occur on the deal."""
        self._player_cards = [self._draw(), self._draw()]
        self._dealer_cards = [self._draw(), self._draw()]
        self._forced_reward = None

        player_sum, _ = self._hand_value(self._player_cards)
        dealer_sum, _ = self._hand_value(self._dealer_cards)
        if player_sum == 21 or dealer_sum == 21:
            player_wins = player_sum == 21 and dealer_sum != 21
            draw = player_sum == 21 and dealer_sum == 21
            self._forced_reward = 0.0 if draw else (1.0 if player_wins else -1.0)
        return self._observation()

    def _observation(self) -> tuple:
        player_sum, usable = self._hand_value(self._player_cards)
        dealer_showing = _card_value(self._dealer_cards[0])
        return (int(player_sum), int(dealer_showing), bool(usable))

    def step(self, action: str) -> tuple:
        """Take one action; returns (reward, observation, done).

        Args:
            action: 'hit' or 'stick'.
        """
        if action not in ("hit", "stick"):
            raise ValueError("action must be 'hit' or 'stick'")

        if self._forced_reward is not None:
            return self._forced_reward, self._observation(), True

        if action == "hit":
            self._player_cards.append(self._draw())
            player_sum, _ = self._hand_value(self._player_cards)
            if player_sum > 21:
                return -1.0, self._observation(), True
            return 0.0, self._observation(), False

        # 'stick' — dealer plays: hit until sum >= 17.
        dealer_sum, _ = self._hand_value(self._dealer_cards)
        while dealer_sum < 17:
            self._dealer_cards.append(self._draw())
            dealer_sum, _ = self._hand_value(self._dealer_cards)
        player_sum, _ = self._hand_value(self._player_cards)

        if dealer_sum > 21:
            return 1.0, self._observation(), True
        if player_sum > dealer_sum:
            return 1.0, self._observation(), True
        if player_sum < dealer_sum:
            return -1.0, self._observation(), True
        return 0.0, self._observation(), True


def _validate_policy_fn(policy_fn: Callable) -> None:
    if not callable(policy_fn):
        raise TypeError("policy_fn must be callable")


def _sample_action(state, policy_fn, rng: np.random.Generator) -> int:
    probs = np.asarray(policy_fn(state), dtype=np.float64)
    if probs.shape != (N_ACTIONS,):
        raise ValueError(f"policy_fn must return a vector of length {N_ACTIONS}")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("policy probabilities must sum to 1")
    return int(rng.choice(N_ACTIONS, p=probs))


def _run_episode(env, policy_fn, rng: np.random.Generator) -> list[tuple]:
    """Return [(state, action_index, reward), ...] for one episode."""
    state = env.reset()
    trajectory = []
    while True:
        action_index = _sample_action(state, policy_fn, rng)
        reward, observation, done = env.step(ACTIONS[action_index])
        trajectory.append((state, action_index, reward))
        if done:
            return trajectory
        state = observation


def _returns_from_rewards(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def mc_first_visit_prediction(
    env,
    policy_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """First-visit MC prediction (Algorithm 5.1). Book Section 5.1.

    Estimates v_pi(s) as the average of returns following FIRST visits to s.
    """
    _validate_policy_fn(policy_fn)
    rng = np.random.default_rng(seed)
    returns_by_state: dict = {}
    for _ in range(n_episodes):
        trajectory = _run_episode(env, policy_fn, rng)
        rewards = [r for _, _, r in trajectory]
        returns = _returns_from_rewards(rewards, gamma)
        seen: set = set()
        for t, (state, _, _) in enumerate(trajectory):
            if state in seen:
                continue
            seen.add(state)
            returns_by_state.setdefault(state, []).append(returns[t])
    return {s: float(np.mean(g)) for s, g in returns_by_state.items()}


def mc_every_visit_prediction(
    env,
    policy_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Every-visit MC prediction (Algorithm 5.1). Book Section 5.1.

    Estimates v_pi(s) as the average of returns following ALL visits to s.
    """
    _validate_policy_fn(policy_fn)
    rng = np.random.default_rng(seed)
    returns_by_state: dict = {}
    for _ in range(n_episodes):
        trajectory = _run_episode(env, policy_fn, rng)
        rewards = [r for _, _, r in trajectory]
        returns = _returns_from_rewards(rewards, gamma)
        for t, (state, _, _) in enumerate(trajectory):
            returns_by_state.setdefault(state, []).append(returns[t])
    return {s: float(np.mean(g)) for s, g in returns_by_state.items()}


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    """Epsilon-greedy probabilities (the closest epsilon-soft policy to greedy)."""
    q_values = np.asarray(q_values, dtype=np.float64)
    n = len(q_values)
    greedy = np.flatnonzero(q_values == q_values.max())
    probs = np.full(n, epsilon / n, dtype=np.float64)
    probs[greedy] += (1.0 - epsilon) / len(greedy)
    return probs


def mc_control_epsilon_greedy(
    env,
    n_episodes: int = 20_000,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    seed: int | None = None,
) -> tuple[dict, dict, list]:
    """On-policy first-visit MC control for epsilon-soft policies (Algorithm 5.3).

    Book Section 5.4. The behavior and target policy is the epsilon-greedy
    policy with respect to Q; the greedy action gets 1 - epsilon + epsilon/|A|,
    the others share epsilon/|A|.

    Returns:
        (policy, Q, episode_rewards): policy[state] = probability vector over
        ACTIONS; q_table[state] = action-value array of shape (2,).
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    rng = np.random.default_rng(seed)

    q_table: dict = {}
    returns_by_pair: dict = {}

    def behavior_probs(state):
        if state not in q_table:
            return np.ones(N_ACTIONS) / N_ACTIONS
        return _epsilon_greedy_probs(q_table[state], epsilon)

    episode_rewards: list[float] = []
    for _ in range(n_episodes):
        trajectory = _run_episode(env, behavior_probs, rng)
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))
        returns = _returns_from_rewards(rewards, gamma)
        seen: set = set()
        for t, (state, action, _) in enumerate(trajectory):
            if (state, action) in seen:
                continue
            seen.add((state, action))
            pair = (state, action)
            returns_by_pair.setdefault(pair, []).append(returns[t])
            q_table.setdefault(state, np.zeros(N_ACTIONS))
            q_table[state][action] = float(np.mean(returns_by_pair[pair]))

    policy = {s: _epsilon_greedy_probs(q, epsilon) for s, q in q_table.items()}
    return policy, q_table, episode_rewards


def _importance_ratios(
    trajectory: list[tuple],
    target_fn: Callable,
    behavior_fn: Callable,
) -> list[float]:
    """Importance sampling ratios rho_{t:T-1} for each timestep."""
    n = len(trajectory)
    ratios = [1.0] * n
    for t in range(n - 1, -1, -1):
        state, action, _ = trajectory[t]
        pi_prob = float(target_fn(state)[action])
        b_prob = float(behavior_fn(state)[action])
        ratio = pi_prob / b_prob if b_prob > 0.0 else 0.0
        ratios[t] = ratio * (ratios[t + 1] if t + 1 < n else 1.0)
    return ratios


def mc_off_policy_prediction(
    env,
    target_fn: Callable,
    behavior_fn: Callable,
    n_episodes: int = 10_000,
    gamma: float = 1.0,
    weighted: bool = True,
    first_visit: bool = True,
    seed: int | None = None,
) -> dict:
    """Off-policy MC prediction via importance sampling (Algorithms 5.4/5.5).

    Book Section 5.5 (ordinary IS, unbiased; Eq. 5.4) and Section 5.5/5.6
    (weighted IS; Eq. 5.5). Behavior policy b must cover the target policy pi.

    Args:
        env: Environment.
        target_fn: Target policy pi(state) -> probability vector.
        behavior_fn: Behavior policy b(state) -> probability vector.
        n_episodes: Number of episodes to sample.
        gamma: Discount factor.
        weighted: If True use weighted IS; otherwise ordinary IS.
        first_visit: If True, only first visits within an episode count.
        seed: Optional seed.

    Returns:
        dict state -> estimated v_pi(s).
    """
    _validate_policy_fn(target_fn)
    _validate_policy_fn(behavior_fn)
    rng = np.random.default_rng(seed)

    scaled: dict = {}
    weights_sum: dict = {}

    for _ in range(n_episodes):
        trajectory = _run_episode(env, behavior_fn, rng)
        rewards = [r for _, _, r in trajectory]
        returns = _returns_from_rewards(rewards, gamma)
        ratios = _importance_ratios(trajectory, target_fn, behavior_fn)
        seen: set = set()
        for t, (state, _, _) in enumerate(trajectory):
            if first_visit and state in seen:
                continue
            seen.add(state)
            scaled.setdefault(state, []).append((ratios[t], returns[t]))
            weights_sum[state] = weights_sum.get(state, 0.0) + ratios[t]

    estimates: dict = {}
    for state, pairs in scaled.items():
        if weighted:
            den = weights_sum[state]
            estimates[state] = (
                float(np.sum([w * g for w, g in pairs]) / den) if den > 0.0 else 0.0
            )
        else:
            estimates[state] = float(np.mean([w * g for w, g in pairs]))
    return estimates


def mc_control_off_policy(
    env,
    n_episodes: int = 20_000,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    weighted: bool = True,
    seed: int | None = None,
) -> tuple[dict, dict, list]:
    """Off-policy MC control (Algorithms 5.4/5.5). Book Section 5.7.

    Behavior policy is epsilon-soft (epsilon-greedy w.r.t. Q); the target
    policy is greedy w.r.t. Q. Uses weighted importance sampling (default) or
    ordinary importance sampling (visit-count-normalized scaled returns).

    Returns:
        (policy, Q, episode_rewards): policy[state] = probability vector over ACTIONS.
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    rng = np.random.default_rng(seed)

    q_table: dict = {}
    counts: dict = {}

    def behavior_probs(state):
        if state not in q_table:
            return np.ones(N_ACTIONS) / N_ACTIONS
        return _epsilon_greedy_probs(q_table[state], epsilon)

    def greedy_action(state) -> int:
        q = q_table.get(state)
        if q is None:
            return 0
        return int(np.argmax(q))

    episode_rewards: list[float] = []
    for _ in range(n_episodes):
        trajectory = _run_episode(env, behavior_probs, rng)
        rewards = [r for _, _, r in trajectory]
        episode_rewards.append(float(sum(rewards)))

        g = 0.0
        w = 1.0
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            g = gamma * g + reward
            q_table.setdefault(state, np.zeros(N_ACTIONS))
            pair_counts = counts.setdefault(state, np.zeros(N_ACTIONS))

            if weighted:
                pair_counts[action] += w
                pair_counts[action] = max(pair_counts[action], 1e-8)
                q_table[state][action] += (w / pair_counts[action]) * (g - q_table[state][action])
            else:
                # ordinary IS: simple average of w*g over visits
                pair_counts[action] += 1.0
                q_table[state][action] += (w * g - q_table[state][action]) / pair_counts[action]

            # target is greedy; stop bootstrapping from non-greedy suffixes
            if action != greedy_action(state):
                break
            b_prob = behavior_probs(state)[action]
            w = w * (1.0 / b_prob) if b_prob > 0.0 else 0.0

    policy = {}
    for state in q_table:
        probs = np.zeros(N_ACTIONS)
        greedy = np.flatnonzero(q_table[state] == q_table[state].max())
        probs[greedy] = 1.0 / len(greedy)
        policy[state] = probs
    return policy, q_table, episode_rewards


def blackjack_stick_on_threshold(stick_threshold: int = 20) -> Callable:
    """Deterministic policy: stick when player_sum >= stick_threshold.

    Book Example 5.1 used "stick on 20 or 21"; returned as a probability vector.
    """

    def policy_fn(state) -> np.ndarray:
        player_sum = state[0]
        if player_sum >= stick_threshold:
            return np.array([0.0, 1.0])
        return np.array([1.0, 0.0])

    return policy_fn
