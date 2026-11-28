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

from rl_envs.pip_race import (
    N_FEATURES,
    WHITE,
    PipRaceEnv,
    TDGammonAgent,
    evaluate_match,
    linear_value,
    pip_race_features,
    semi_gradient_td_lambda_update,
)


def _epsilon_greedy_probs(
    q_table: dict, state: int, n_actions: int, epsilon: float,
) -> np.ndarray:
    """Return epsilon-greedy action probabilities, splitting greedy ties evenly.

    Every maximising action is treated as greedy, so ties keep the same uniform
    split as the original per-algorithm implementation.

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        q_table: See Algorithm 16.1 and the implementation for its role.
        state: See Algorithm 16.1 and the implementation for its role.
        n_actions: See Algorithm 16.1 and the implementation for its role.
        epsilon: See Algorithm 16.1 and the implementation for its role.

    Returns:
        epsilon greedy probs as defined in Algorithm 16.1."""
    # internal helper for the epsilon greedy probs step (book Chapter 16).
    # q values: action-value estimates for the state (book Chapter 16).
    q_values = np.array([q_table[(state, a)] for a in range(n_actions)])
    # best: best action or best value found so far (book Chapter 16).
    best = np.flatnonzero(q_values == q_values.max())
    # probs: action probabilities from the current policy (book Chapter 16).
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    # probs[best]: add value computed from its operands into the running value (book Chapter 16).
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


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

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16)."""

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
        """init (reference implementation).


        Args:
            s_max: See the relevant book section and function body.
            demand_rate: See the relevant book section and function body.
            c_order: See the relevant book section and function body.
            h: See the relevant book section and function body.
            p: See the relevant book section and function body.
            periods: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16)."""
        # initialize the parameters and state required by the algorithm (book Chapter 16).
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
        # self.state: int | None attribute/state used by the algorithm (book Chapter 16).
        self.state: int | None = None
        self._t = 0

    def reset(self, start_inventory: int | None = None) -> int:
        """Start a new episode; return the initial inventory level.

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

        Args:
            start_inventory: See Algorithm 16.1 and the implementation for its role.

        Returns:
            reset as defined in Algorithm 16.1."""
        # reset internal state for a new episode (book Chapter 16).
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

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

        Returns:
            step as defined in Algorithm 16.1."""
        # advance one step and return the observed transition (book Chapter 16).
        if self.state is None:
            raise RuntimeError("call reset() first")
        if not 0 <= order_up_to <= self.s_max:
            raise ValueError(f"order_up_to must be in [0, {self.s_max}]")
        # action: action selected for the current step (book Chapter 16).
        action = int(order_up_to)
        # ordered: maximum of the values (book Chapter 16).
        ordered = max(0, action - self.state)
        # position: value computed from its operands (book Chapter 16).
        position = self.state + ordered
        # demand: integer conversion (book Chapter 16).
        demand = int(self.rng.poisson(self.demand_rate))
        # leftover: maximum of the values (book Chapter 16).
        leftover = max(0, position - demand)
        # unmet: maximum of the values (book Chapter 16).
        unmet = max(0, demand - position)
        # cost: value computed from its operands (book Chapter 16).
        cost = self.c_order * ordered + self.h * leftover + self.p * unmet
        self.state = min(leftover, self.s_max)
        # self._t: add literal constant value into the running value (book Chapter 16).
        self._t += 1
        # done: termination flag for the current episode (book Chapter 16).
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

    Book: Sutton & Barto (2nd ed.), Algorithm 6.4 (Chapter 16).

    Args:
        n_episodes: See Algorithm 6.4 and the implementation for its role.
        alpha: See Algorithm 6.4 and the implementation for its role.
        epsilon: See Algorithm 6.4 and the implementation for its role.
        gamma: See Algorithm 6.4 and the implementation for its role.
        seed: See Algorithm 6.4 and the implementation for its role.
        s_max: See Algorithm 6.4 and the implementation for its role.
        demand_rate: See Algorithm 6.4 and the implementation for its role.
        c_order: See Algorithm 6.4 and the implementation for its role.
        h: See Algorithm 6.4 and the implementation for its role.
        p: See Algorithm 6.4 and the implementation for its role.
        periods: See Algorithm 6.4 and the implementation for its role."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    # env: environment instance being controlled (book Chapter 16).
    env = InventoryEnv(
        s_max=s_max, demand_rate=demand_rate, c_order=c_order,
        h=h, p=p, periods=periods, seed=seed,
    )
    # rng: seeded random generator so runs are reproducible (book Chapter 16).
    rng = np.random.default_rng(seed)
    # q table: action-value estimates Q(s, a) (book Chapter 16).
    q_table = {(s, a): 0.0 for s in range(s_max + 1) for a in range(s_max + 1)}

    for _ in range(n_episodes):
        # state: current state of the environment or MDP (book Chapter 16).
        state = env.reset()
        # probs: action probabilities from the current policy (book Chapter 16).
        probs = _epsilon_greedy_probs(q_table, state, s_max + 1, epsilon)
        # action: action selected for the current step (book Chapter 16).
        action = int(rng.choice(s_max + 1, p=probs))
        done = False
        while not done:
            # (next_state, reward, done): result of env.step(...) (book Chapter 16).
            next_state, reward, done = env.step(action)
            if done:
                # Time-limit truncation: bootstrap, do not zero the target.
                next_q = max(q_table[(next_state, a)] for a in range(s_max + 1))
            else:
                # next probs: result of _epsilon_greedy_probs(...) (book Chapter 16).
                next_probs = _epsilon_greedy_probs(q_table, next_state, s_max + 1, epsilon)
                # next action: integer conversion (book Chapter 16).
                next_action = int(rng.choice(s_max + 1, p=next_probs))
                # next q: action-value estimate for the next state (book Chapter 16).
                next_q = q_table[(next_state, next_action)]
            # q_table[state, action]: add value computed from its operands into the running ...
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

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Returns:
        evaluate inventory policy as defined in Algorithm 16.1."""
    # carry out the evaluate inventory policy step (book Chapter 16).
    # env: environment instance being controlled (book Chapter 16).
    env = InventoryEnv(
        s_max=s_max, demand_rate=demand_rate, c_order=c_order,
        h=h, p=p, periods=periods, seed=seed,
    )
    total_cost = 0.0
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(episodes):
        # state: current state of the environment or MDP (book Chapter 16).
        state = env.reset()
        done = False
        # Repeat while the condition holds: the algorithm's termination loop.
        while not done:
            # (next_state, reward, done): result of env.step(...) (book Chapter 16).
            next_state, reward, done = env.step(policy_fn(state))
            # total_cost: add value derived from a unary operation into the running value (book ...
            total_cost += -reward
            state = next_state
    return total_cost / (episodes * periods)


def greedy_inventory_action(q_table: dict, state: int, s_max: int = 10) -> int:
    """Greedy order-up-to action for `state` under a learned Q table.

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        q_table: See Algorithm 16.1 and the implementation for its role.
        state: See Algorithm 16.1 and the implementation for its role.
        s_max: See Algorithm 16.1 and the implementation for its role.

    Returns:
        greedy inventory action as defined in Algorithm 16.1."""
    # carry out the greedy inventory action step (book Chapter 16).
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

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16)."""

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
        """init (reference implementation).


        Args:
            q_max: See the relevant book section and function body.
            arrival_rate: See the relevant book section and function body.
            p_normal: See the relevant book section and function body.
            p_expedited: See the relevant book section and function body.
            c_normal: See the relevant book section and function body.
            c_expedited: See the relevant book section and function body.
            h: See the relevant book section and function body.
            periods: See the relevant book section and function body.
            seed: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16)."""
        # initialize the parameters and state required by the algorithm (book Chapter 16).
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
        # self.state: int | None attribute/state used by the algorithm (book Chapter 16).
        self.state: int | None = None
        self._t = 0

    def reset(self) -> int:
        """Start a new episode (empty system); return the initial state.


        Returns:
            reset as defined in Algorithm 16.1.

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16)."""
        # reset internal state for a new episode (book Chapter 16).
        self.state = 0
        self._t = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Advance one period. Returns (next_state, reward, done).

        Args:
            action: 0 = normal service, 1 = expedited service.

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

        Returns:
            step as defined in Algorithm 16.1."""
        # advance one step and return the observed transition (book Chapter 16).
        if self.state is None:
            raise RuntimeError("call reset() first")
        if action not in (0, 1):
            raise ValueError("action must be 0 (normal) or 1 (expedited)")
        q = self.state
        service_cost = 0.0
        served = 0
        if q > 0:
            # service cost: value chosen conditionally (book Chapter 16).
            service_cost = self.c_normal if action == 0 else self.c_expedited
            # p done: value chosen conditionally (book Chapter 16).
            p_done = self.p_normal if action == 0 else self.p_expedited
            if self.rng.random() < p_done:
                served = 1
        # arrivals: integer conversion (book Chapter 16).
        arrivals = int(self.rng.poisson(self.arrival_rate))
        # next q: action-value estimate for the next state (book Chapter 16).
        next_q = min(self.q_max, max(0, q - served) + arrivals)
        # cost: value computed from its operands (book Chapter 16).
        cost = self.h * q + service_cost
        self.state = next_q
        # self._t: add literal constant value into the running value (book Chapter 16).
        self._t += 1
        # done: termination flag for the current episode (book Chapter 16).
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

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Returns:
        bellman target as defined in Algorithm 16.1."""
    # carry out the bellman target step (book Chapter 16).
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    # next q values: result of np.asarray(...) (book Chapter 16).
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

    Book: Sutton & Barto (2nd ed.), Algorithm 6.2 (Chapter 16).

    Args:
        n_episodes: See Algorithm 6.2 and the implementation for its role.
        alpha: See Algorithm 6.2 and the implementation for its role.
        epsilon: See Algorithm 6.2 and the implementation for its role.
        gamma: See Algorithm 6.2 and the implementation for its role.
        seed: See Algorithm 6.2 and the implementation for its role.
        q_max: See Algorithm 6.2 and the implementation for its role.
        arrival_rate: See Algorithm 6.2 and the implementation for its role.
        p_normal: See Algorithm 6.2 and the implementation for its role.
        p_expedited: See Algorithm 6.2 and the implementation for its role.
        c_normal: See Algorithm 6.2 and the implementation for its role.
        c_expedited: See Algorithm 6.2 and the implementation for its role.
        h: See Algorithm 6.2 and the implementation for its role.
        periods: See Algorithm 6.2 and the implementation for its role."""
    # carry out the q learning queue step (book Chapter 16).
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    # env: environment instance being controlled (book Chapter 16).
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
    # rng: seeded random generator so runs are reproducible (book Chapter 16).
    rng = np.random.default_rng(seed)
    # q table: action-value estimates Q(s, a) (book Chapter 16).
    q_table = {(q, a): 0.0 for q in range(q_max + 1) for a in (0, 1)}

    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(n_episodes):
        # state: current state of the environment or MDP (book Chapter 16).
        state = env.reset()
        done = False
        # Repeat while the condition holds: the algorithm's termination loop.
        while not done:
            # probs: action probabilities from the current policy (book Chapter 16).
            probs = _epsilon_greedy_probs(q_table, state, 2, epsilon)
            # action: action selected for the current step (book Chapter 16).
            action = int(rng.choice(2, p=probs))
            # (next_state, reward, done): result of env.step(...) (book Chapter 16).
            next_state, reward, done = env.step(action)
            # next q values: array built from the supplied values (book Chapter 16).
            next_q_values = np.array(
                [q_table[(next_state, 0)], q_table[(next_state, 1)]],
                dtype=np.float64,
            )
            # target: target value for the current update (book Chapter 16).
            target = bellman_target(reward, next_q_values, gamma, terminal=False)
            # q_table[state, action]: add value computed from its operands into the running ...
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
    """Average per-period cost of a queue-service policy (lower is better).

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        policy_fn: See Algorithm 16.1 and the implementation for its role.
        episodes: See Algorithm 16.1 and the implementation for its role.
        seed: See Algorithm 16.1 and the implementation for its role.
        q_max: See Algorithm 16.1 and the implementation for its role.
        arrival_rate: See Algorithm 16.1 and the implementation for its role.
        p_normal: See Algorithm 16.1 and the implementation for its role.
        p_expedited: See Algorithm 16.1 and the implementation for its role.
        c_normal: See Algorithm 16.1 and the implementation for its role.
        c_expedited: See Algorithm 16.1 and the implementation for its role.
        h: See Algorithm 16.1 and the implementation for its role.
        periods: See Algorithm 16.1 and the implementation for its role.

    Returns:
        evaluate queue policy as defined in Algorithm 16.1."""
    # carry out the evaluate queue policy step (book Chapter 16).
    # env: environment instance being controlled (book Chapter 16).
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
    # Iterate over _: one pass per episode (book algorithm).
    for _ in range(episodes):
        # state: current state of the environment or MDP (book Chapter 16).
        state = env.reset()
        done = False
        # Repeat while the condition holds: the algorithm's termination loop.
        while not done:
            # (_, reward, done): result of env.step(...) (book Chapter 16).
            _, reward, done = env.step(policy_fn(state))
            # total_cost: add value derived from a unary operation into the running value (book ...
            total_cost += -reward
            state = env.state
    return total_cost / (episodes * periods)


def greedy_queue_action(q_table: dict, state: int) -> int:
    """Greedy service-speed action for `state` under a learned Q table.

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        q_table: See Algorithm 16.1 and the implementation for its role.
        state: See Algorithm 16.1 and the implementation for its role.

    Returns:
        greedy queue action as defined in Algorithm 16.1."""
    # carry out the greedy queue action step (book Chapter 16).
    return int(np.argmax([q_table[(state, 0)], q_table[(state, 1)]]))


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 16 / repo Ch. 16)
# ---------------------------------------------------------------------------

# TD-Gammon uses an undiscounted value update; the demo keeps that convention
# explicit instead of hiding a 1.0 inside the Bellman target and TD update.
_TD_GAMMON_GAMMA = 1.0
_WEIGHT_INIT_STD = 0.01


def _choose_gammon_play(
    plays: list[tuple],
    weights: np.ndarray,
    player: str,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple:
    """Pick a legal play, drawing the exploration test and index in that order.

    Keeping both draws on the shared generator preserves the self-play stream;
    greedy play maximises for WHITE and minimises for its opponent.

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        plays: See Algorithm 16.1 and the implementation for its role.
        weights: See Algorithm 16.1 and the implementation for its role.
        player: See Algorithm 16.1 and the implementation for its role.
        epsilon: See Algorithm 16.1 and the implementation for its role.
        rng: See Algorithm 16.1 and the implementation for its role.

    Returns:
        choose gammon play as defined in Algorithm 16.1."""
    # internal helper for the choose gammon play step (book Chapter 16).
    if rng.random() < epsilon:
        return plays[int(rng.integers(0, len(plays)))]
    # scored: collection built by comprehension (book Chapter 16).
    scored = [linear_value(weights, pip_race_features(p)) for p in plays]
    # index: value chosen conditionally (book Chapter 16).
    index = int(np.argmax(scored)) if player == WHITE else int(np.argmin(scored))
    return plays[index]


def _gammon_td_target(
    play: tuple, weights: np.ndarray, next_turns: int,
) -> tuple[float, float]:
    """Return (reward, v_next) for a play; a finished game has no successor.

    Bootstrapping from zero on a terminal play is how the sparse terminal
    reward propagates back through the self-play trace.

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        play: See Algorithm 16.1 and the implementation for its role.
        weights: See Algorithm 16.1 and the implementation for its role.
        next_turns: See Algorithm 16.1 and the implementation for its role.

    Returns:
        gammon td target as defined in Algorithm 16.1."""
    # internal helper for the gammon td target step (book Chapter 16).
    if PipRaceEnv.is_over(play, next_turns):
        return PipRaceEnv.outcome(play, next_turns), 0.0
    return 0.0, linear_value(weights, pip_race_features(play))


def plot_inventory_and_queue_demos(
    save_path: str | None = None,
    show: bool = False,
    seed: int = 0,
):
    """Inventory control and queue scheduling demos (Ch. 16 applications).

    Sutton & Barto (2nd ed.), Ch. 16 surveys RL applications/case studies; this
    chapter implements small tabular control applications in that spirit.
      - Left (inventory, Sec. 16 context / order-up-to control): compares average
        cost per period for a learned Sarsa policy vs simple baselines (zero-order
        and random), evaluated on identical seeded demand streams.
      - Right (single-server queue with expedited-service action): compares
        average cost per period for a learned Q-learning policy vs random and
        never-expediting baselines.
    Bars show mean cost (lower is better) over evaluation episodes.

    Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16).

    Args:
        save_path: See Algorithm 16.1 and the implementation for its role.
        show: See Algorithm 16.1 and the implementation for its role.
        seed: See Algorithm 16.1 and the implementation for its role."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Inventory
    q_inv, _ = sarsa_inventory(n_episodes=500, alpha=0.1, epsilon=0.1, gamma=0.95, seed=seed)
    # inv learned: result of evaluate_inventory_policy(...) (book Chapter 16).
    inv_learned = evaluate_inventory_policy(lambda s: greedy_inventory_action(q_inv, s), episodes=300, seed=seed + 1)
    inv_zero = evaluate_inventory_policy(lambda s: s, episodes=300, seed=seed + 1)  # order up to current -> no order
    rng = np.random.default_rng(seed + 2)
    # inv random: result of evaluate_inventory_policy(...) (book Chapter 16).
    inv_random = evaluate_inventory_policy(lambda s: int(rng.integers(0, 11)), episodes=300, seed=seed + 1)

    # Queue
    q_q, _ = q_learning_queue(n_episodes=800, alpha=0.1, epsilon=0.1, gamma=0.95, seed=seed)
    # q learned: result of evaluate_queue_policy(...) (book Chapter 16).
    q_learned = evaluate_queue_policy(lambda s: greedy_queue_action(q_q, s), episodes=300, seed=seed + 1)
    # q random: result of evaluate_queue_policy(...) (book Chapter 16).
    q_random = evaluate_queue_policy(lambda s: int(np.random.default_rng(seed + 3).integers(0, 2)), episodes=300, seed=seed + 1)
    q_never = evaluate_queue_policy(lambda s: 0, episodes=300, seed=seed + 1)  # normal service

    # (fig, (ax1, ax2)): result of plt.subplots(...) (book Chapter 16).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.bar(["Learned\n(Sarsa)", "Zero-order", "Random"], [inv_learned, inv_zero, inv_random],
            color=["#4c78a8", "#f58518", "#54a24b"])
    ax1.set_ylabel("Avg cost per period (lower better)")
    ax1.set_title("Inventory management demo")

    ax2.bar(["Learned\n(Q-learning)", "Random", "Never expedite"], [q_learned, q_random, q_never],
            color=["#4c78a8", "#f58518", "#54a24b"])
    ax2.set_ylabel("Avg cost per period (lower better)")
    ax2.set_title("Queue scheduling demo")

    fig.suptitle("S&B (2nd ed.), Ch. 16: Application-style control demos")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_td_gammon_learning(
    save_path: str | None = None,
    show: bool = False,
    n_games: int = 400,
    eval_every: int = 50,
    eval_games: int = 80,
    alpha: float = 0.05,
    lam: float = 0.7,
    epsilon: float = 0.05,
    seed: int = 0,
):
    """TD-Gammon-like self-play learning curve on PipRace (Sec. 16.1 demo).

    Sutton & Barto (2nd ed.), Sec. 16.1 (TD-Gammon): self-play temporal-
    difference learning with a function approximator learns strong play from
    sparse terminal rewards. This demo trains the linear PipRace value function
    (documented backgammon approximation in this repo) by semi-gradient TD(lambda)
    self-play and periodically evaluates greedy play vs a random opponent
    (`evaluate_match`). PipRace, features, and scale are simplified relative to
    TD-Gammon/backgammon; the plot illustrates learning progress, not book-level
    performance.

    Book: Sutton & Barto (2nd ed.), Section 16.1 (Chapter 16).

    Args:
        save_path: See Section 16.1 and the implementation for its role.
        show: See Section 16.1 and the implementation for its role.
        n_games: See Section 16.1 and the implementation for its role.
        eval_every: See Section 16.1 and the implementation for its role.
        eval_games: See Section 16.1 and the implementation for its role.
        alpha: See Section 16.1 and the implementation for its role.
        lam: See Section 16.1 and the implementation for its role.
        epsilon: See Section 16.1 and the implementation for its role.
        seed: See Section 16.1 and the implementation for its role."""
    # render the book figure with matplotlib (book Chapter 16).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # rng: seeded random generator so runs are reproducible (book Chapter 16).
    rng = np.random.default_rng(seed)
    # weights: sample from a normal distribution (book Chapter 16).
    weights = rng.normal(0.0, _WEIGHT_INIT_STD, N_FEATURES)
    # eval points: collection of the listed elements (book Chapter 16).
    eval_points = []
    # scores: collection of the listed elements (book Chapter 16).
    scores = []

    def maybe_eval(games_done: int):
        """maybe eval (reference implementation).


        Args:
            games_done: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Algorithm 16.1 (Chapter 16)."""
        # carry out the maybe eval step (book Chapter 16).
        if games_done % eval_every == 0 or games_done == n_games:
            # agent: result of TDGammonAgent(...) (book Chapter 16).
            agent = TDGammonAgent(weights.copy())
            # score: result of evaluate_match(...) (book Chapter 16).
            score = evaluate_match(agent, n_games=eval_games, seed=seed + games_done + 1)
            eval_points.append(games_done)
            scores.append(score)

    # Iterate over g: one pass per index (book algorithm).
    for g in range(1, n_games + 1):
        # env: environment instance being controlled (book Chapter 16).
        env = PipRaceEnv(seed=int(rng.integers(0, 2**31 - 1)))
        # state: current state of the environment or MDP (book Chapter 16).
        state = env.reset()
        # eligibility: zero-filled array (book Chapter 16).
        eligibility = np.zeros(N_FEATURES, dtype=np.float64)
        # Repeat while the condition holds: the algorithm's termination loop.
        while not PipRaceEnv.is_over(state, env.turn_count):
            # player: element selected from the indexed collection (book Chapter 16).
            player = state[2]
            # dice: result of env.roll_dice(...) (book Chapter 16).
            dice = env.roll_dice()
            # plays: result of PipRaceEnv.legal_dice_plays(...) (book Chapter 16).
            plays = PipRaceEnv.legal_dice_plays(state, player, dice)
            # play: result of _choose_gammon_play(...) (book Chapter 16).
            play = _choose_gammon_play(plays, weights, player, epsilon, rng)
            # features: result of pip_race_features(...) (book Chapter 16).
            features = pip_race_features(state)
            # v state: result of linear_value(...) (book Chapter 16).
            v_state = linear_value(weights, features)
            # next turns: value computed from its operands (book Chapter 16).
            next_turns = env.turn_count + 1
            # (reward, v_next): result of _gammon_td_target(...) (book Chapter 16).
            reward, v_next = _gammon_td_target(play, weights, next_turns)
            # delta: TD error or update difference for this step (book Chapter 16).
            delta = reward + _TD_GAMMON_GAMMA * v_next - v_state
            # (weights, eligibility): result of semi_gradient_td_lambda_update(...) (book ...
            weights, eligibility = semi_gradient_td_lambda_update(
                weights, eligibility, features, delta, alpha, _TD_GAMMON_GAMMA, lam
            )
            # state: current state of the environment or MDP (book Chapter 16).
            state = env.play_turn(dice, play)
        maybe_eval(g)

    # (fig, ax): result of plt.subplots(...) (book Chapter 16).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eval_points, scores, marker="o")
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Self-play training games")
    ax.set_ylabel("Score vs random (wins + 0.5*draws)")
    ax.set_ylim(0, 1)
    ax.set_title("S&B (2nd ed.), Sec. 16.1 (TD-Gammon-like): PipRace self-play")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
