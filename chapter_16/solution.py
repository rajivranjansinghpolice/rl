"""Solution implementation for Chapter 16 - Applications and Case Studies.

Reference: Sutton & Barto (2nd Ed), Chapter 16 (Applications and Case Studies).

The book's Chapter 16 surveys real-world applications (TD-Gammon, Samuel's
checkers, Watson's Jeopardy wagering, memory controllers, deep RL for Atari
and Go, personalized web services, thermal soaring) and gives no boxed
algorithms. This chapter implements two small classic control applications
in the spirit of that chapter, both tabular:

- Alg 16.1 — Inventory management controlled with Sarsa (classic
  order-up-to inventory MDP: stochastic Poisson demand, order-up-to
  action, holding/stockout/order costs)
- Alg 16.2 — Simplified resource-control application: a single-server
  queue MDP with a service-speed decision and waiting-time targets,
  solved with Q-learning

Both use finite-horizon episodes with bootstrapped (not zeroed) values at
the time-limit truncation — the correct Bellman handling for truncated
episodes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ============================================================================
# Algorithm 16.1: Inventory management with Sarsa
# Classic inventory-control MDP (in the spirit of Chapter 16's perpetually
# applied problems); Sutton & Barto Section 6.4 (Sarsa)
# ============================================================================


class InventoryEnv:
    """Single-product periodic-review inventory control MDP.

    State: inventory level at the start of a period, s in [0, s_max].
    Action: order-up-to level a in [0, s_max] (order a - s units if a > s,
    otherwise nothing is ordered). After ordering, Poisson(demand_rate)
    demand arrives; unmet demand is lost sales.

    Per-period cost (reward = -cost):
        c_order * units_ordered + h * leftover + p * unmet_demand

    Args:
        s_max: Maximum inventory / order-up-to level.
        demand_rate: Poisson demand mean per period.
        c_order: Per-unit ordering cost.
        h: Holding cost per unit left over.
        p: Stockout (lost-sale) cost per unit of unmet demand.
        periods: Periods per episode (finite horizon; the time limit is a
            truncation, not a true terminal state).
        seed: RNG seed for the demand stream.
    """

    def __init__(
        self,
        s_max: int = 10,
        demand_rate: float = 3.0,
        c_order: float = 1.0,
        h: float = 1.0,
        p: float = 5.0,
        periods: int = 12,
        seed: int | None = None,
    ) -> None:
        if s_max < 1:
            raise ValueError("s_max must be positive")
        if demand_rate <= 0.0:
            raise ValueError("demand_rate must be positive")
        if periods < 1:
            raise ValueError("periods must be positive")
        self.s_max = int(s_max)
        self.demand_rate = float(demand_rate)
        self.c_order = float(c_order)
        self.h = float(h)
        self.p = float(p)
        self.periods = int(periods)
        self.rng = np.random.default_rng(seed)
        self.state: int | None = None
        self._t = 0

    def reset(self, start_inventory: int | None = None) -> int:
        """Start a new episode; return the initial inventory level."""
        if start_inventory is None:
            self.state = int(self.rng.integers(0, self.s_max + 1))
        else:
            self.state = int(np.clip(start_inventory, 0, self.s_max))
        self._t = 0
        return self.state

    def step(self, order_up_to: int) -> tuple[int, float, bool]:
        """Advance one period. Returns (next_state, reward, done).

        Args:
            order_up_to: Order-up-to level a in [0, s_max].
        """
        if self.state is None:
            raise RuntimeError("call reset() first")
        if not 0 <= order_up_to <= self.s_max:
            raise ValueError(f"order_up_to must be in [0, {self.s_max}]")
        action = int(order_up_to)
        ordered = max(0, action - self.state)
        position = self.state + ordered
        demand = int(self.rng.poisson(self.demand_rate))
        leftover = max(0, position - demand)
        unmet = max(0, demand - position)
        cost = self.c_order * ordered + self.h * leftover + self.p * unmet
        self.state = min(leftover, self.s_max)
        self._t += 1
        done = self._t >= self.periods
        return self.state, -float(cost), done


def sarsa_inventory(
    n_episodes: int = 600,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    seed: int | None = None,
    s_max: int = 10,
    demand_rate: float = 3.0,
    c_order: float = 1.0,
    h: float = 1.0,
    p: float = 5.0,
    periods: int = 12,
) -> tuple[dict, InventoryEnv]:
    """Train Sarsa on the inventory MDP (Algorithm 6.4 flavor).

    On-policy TD control; epsilon-greedy behavior over the order-up-to
    actions. Time-limit truncation is bootstrapped (not treated as a
    terminal state).

    Returns:
        (q_table, env): q_table[(s, a)] for s, a in [0, s_max]; env is the
        training environment (its demand stream is exhausted).
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    env = InventoryEnv(
        s_max=s_max, demand_rate=demand_rate, c_order=c_order,
        h=h, p=p, periods=periods, seed=seed,
    )
    rng = np.random.default_rng(seed)
    q_table = {(s, a): 0.0 for s in range(s_max + 1) for a in range(s_max + 1)}

    def behavior_probs(state: int) -> np.ndarray:
        q = np.array([q_table[(state, a)] for a in range(s_max + 1)])
        n = len(q)
        best = np.flatnonzero(q == q.max())
        probs = np.full(n, epsilon / n, dtype=np.float64)
        probs[best] += (1.0 - epsilon) / len(best)
        return probs

    for _ in range(n_episodes):
        state = env.reset()
        action = int(rng.choice(s_max + 1, p=behavior_probs(state)))
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            if done:
                # Time-limit truncation: bootstrap, do not zero the target.
                next_q = max(q_table[(next_state, a)] for a in range(s_max + 1))
            else:
                next_action = int(rng.choice(s_max + 1, p=behavior_probs(next_state)))
                next_q = q_table[(next_state, next_action)]
            q_table[(state, action)] += alpha * (
                reward + gamma * next_q - q_table[(state, action)]
            )
            state = next_state
            if not done:
                action = next_action

    return q_table, env


def evaluate_inventory_policy(
    policy_fn: Callable[[int], int],
    episodes: int = 300,
    seed: int = 0,
    s_max: int = 10,
    demand_rate: float = 3.0,
    c_order: float = 1.0,
    h: float = 1.0,
    p: float = 5.0,
    periods: int = 12,
) -> float:
    """Average per-period cost of a policy (lower is better).

    Args:
        policy_fn: state -> order-up-to action.
        episodes: Evaluation episodes (seeded demand stream).
    """
    env = InventoryEnv(
        s_max=s_max, demand_rate=demand_rate, c_order=c_order,
        h=h, p=p, periods=periods, seed=seed,
    )
    total_cost = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            next_state, reward, done = env.step(policy_fn(state))
            total_cost += -reward
            state = next_state
    return total_cost / (episodes * periods)


def greedy_inventory_action(q_table: dict, state: int, s_max: int = 10) -> int:
    """Greedy order-up-to action for `state` under a learned Q table."""
    return int(max(range(s_max + 1), key=lambda a: q_table[(state, a)]))


# ============================================================================
# Algorithm 16.2: Single-server queue with a service-speed decision
# Serve-based queue MDP with waiting-time targets; Q-learning.
# Sutton & Barto Sections 6.5 (Q-learning) and 17.4 (reward design context)
# ============================================================================


class QueueEnv:
    """Single-server queue MDP with an expedited-service option.

    State: customers in the system at the start of the period, q in
    [0, q_max] (excess arrivals are lost). Actions:

        0 — normal service: completes one customer with probability p_normal;
            costs c_normal when used (q > 0)
        1 — expedited service: completes one customer with probability
            p_expedited; costs c_expedited (> c_normal)

    Waiting cost h per customer in the system per period (the waiting-time
    target). Reward = -(waiting + service) cost. The episode is a fixed
    horizon of `periods` periods; the time limit is a truncation, and
    bootstrapping applies (see bellman_target).

    Args:
        q_max: Queue capacity (state and cap on customers in system).
        arrival_rate: Poisson arrivals per period.
        p_normal: Completion probability of normal service.
        p_expedited: Completion probability of expedited service.
        c_normal: Service cost of normal service (per period serving).
        c_expedited: Service cost of expedited service.
        h: Waiting cost per customer per period (the waiting-time target).
        periods: Periods per episode.
        seed: RNG seed.
    """

    def __init__(
        self,
        q_max: int = 8,
        arrival_rate: float = 0.9,
        p_normal: float = 0.6,
        p_expedited: float = 0.95,
        c_normal: float = 1.0,
        c_expedited: float = 3.0,
        h: float = 2.0,
        periods: int = 40,
        seed: int | None = None,
    ) -> None:
        if q_max < 1:
            raise ValueError("q_max must be positive")
        if arrival_rate <= 0.0:
            raise ValueError("arrival_rate must be positive")
        if not 0.0 < p_normal <= 1.0 or not 0.0 < p_expedited <= 1.0:
            raise ValueError("service probabilities must be in (0, 1]")
        if periods < 1:
            raise ValueError("periods must be positive")
        self.q_max = int(q_max)
        self.arrival_rate = float(arrival_rate)
        self.p_normal = float(p_normal)
        self.p_expedited = float(p_expedited)
        self.c_normal = float(c_normal)
        self.c_expedited = float(c_expedited)
        self.h = float(h)
        self.periods = int(periods)
        self.rng = np.random.default_rng(seed)
        self.state: int | None = None
        self._t = 0

    def reset(self) -> int:
        """Start a new episode (empty system); return the initial state."""
        self.state = 0
        self._t = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Advance one period. Returns (next_state, reward, done).

        Args:
            action: 0 = normal service, 1 = expedited service.
        """
        if self.state is None:
            raise RuntimeError("call reset() first")
        if action not in (0, 1):
            raise ValueError("action must be 0 (normal) or 1 (expedited)")
        q = self.state
        service_cost = 0.0
        served = 0
        if q > 0:
            service_cost = self.c_normal if action == 0 else self.c_expedited
            p_done = self.p_normal if action == 0 else self.p_expedited
            if self.rng.random() < p_done:
                served = 1
        arrivals = int(self.rng.poisson(self.arrival_rate))
        next_q = min(self.q_max, max(0, q - served) + arrivals)
        cost = self.h * q + service_cost
        self.state = next_q
        self._t += 1
        done = self._t >= self.periods
        return next_q, -float(cost), done


def bellman_target(
    reward: float,
    next_q_values: np.ndarray,
    gamma: float,
    terminal: bool,
) -> float:
    """Q-learning target with explicit truncation handling.

    Args:
        reward: Observed reward r.
        next_q_values: Q(s', a) for all a in the next state.
        gamma: Discount factor.
        terminal: True for a true terminal state (target = r); False for a
            truncation such as a time limit, where the target bootstraps
            (r + gamma * max_a Q(s', a)) — the correct Bellman handling of
            truncated episodes.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    next_q_values = np.asarray(next_q_values, dtype=np.float64)
    if terminal:
        return float(reward)
    return float(reward + gamma * next_q_values.max())


def q_learning_queue(
    n_episodes: int = 600,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    seed: int | None = None,
    q_max: int = 8,
    arrival_rate: float = 0.9,
    p_normal: float = 0.6,
    p_expedited: float = 0.95,
    c_normal: float = 1.0,
    c_expedited: float = 3.0,
    h: float = 2.0,
    periods: int = 40,
) -> tuple[dict, QueueEnv]:
    """Train Q-learning on the queue MDP (Algorithm 6.2 flavor).

    Epsilon-greedy behavior over {normal, expedited}; time-limit
    truncation bootstraps (see bellman_target).

    Returns:
        (q_table, env): q_table[(q, action)] for q in [0, q_max].
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    env = QueueEnv(
        q_max=q_max,
        arrival_rate=arrival_rate,
        p_normal=p_normal,
        p_expedited=p_expedited,
        c_normal=c_normal,
        c_expedited=c_expedited,
        h=h,
        periods=periods,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    q_table = {(q, a): 0.0 for q in range(q_max + 1) for a in (0, 1)}

    def behavior_probs(state: int) -> np.ndarray:
        q_vals = np.array([q_table[(state, 0)], q_table[(state, 1)]])
        best = np.flatnonzero(q_vals == q_vals.max())
        probs = np.full(2, epsilon / 2.0, dtype=np.float64)
        probs[best] += (1.0 - epsilon) / len(best)
        return probs

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = int(rng.choice(2, p=behavior_probs(state)))
            next_state, reward, done = env.step(action)
            next_q_values = np.array(
                [q_table[(next_state, 0)], q_table[(next_state, 1)]],
                dtype=np.float64,
            )
            target = bellman_target(reward, next_q_values, gamma, terminal=False)
            q_table[(state, action)] += alpha * (target - q_table[(state, action)])
            state = next_state

    return q_table, env


def evaluate_queue_policy(
    policy_fn: Callable[[int], int],
    episodes: int = 300,
    seed: int = 0,
    q_max: int = 8,
    arrival_rate: float = 0.9,
    p_normal: float = 0.6,
    p_expedited: float = 0.95,
    c_normal: float = 1.0,
    c_expedited: float = 3.0,
    h: float = 2.0,
    periods: int = 40,
) -> float:
    """Average per-period cost of a queue-service policy (lower is better)."""
    env = QueueEnv(
        q_max=q_max,
        arrival_rate=arrival_rate,
        p_normal=p_normal,
        p_expedited=p_expedited,
        c_normal=c_normal,
        c_expedited=c_expedited,
        h=h,
        periods=periods,
        seed=seed,
    )
    total_cost = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            _, reward, done = env.step(policy_fn(state))
            total_cost += -reward
            state = env.state
    return total_cost / (episodes * periods)


def greedy_queue_action(q_table: dict, state: int) -> int:
    """Greedy service-speed action for `state` under a learned Q table."""
    return int(np.argmax([q_table[(state, 0)], q_table[(state, 1)]]))
