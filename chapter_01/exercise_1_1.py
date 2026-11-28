"""
Exercise 1.1 — Gridworld Environment and Agent Simulation

Reference: Sutton & Barto, Section 1.3 "An Example Agent" — The Gridworld Example
           (Reinforcement Learning: An Introduction, 2nd Edition, Figure 1.2)

This module re-exports:
1. A GridWorld environment with the MDP structure from Figure 1.2
   (now in the shared ``rl_envs.gridworld`` package)
2. A simple agent that interacts with the environment
3. Visualization of state values and policies

Environment Structure (4x4 grid):
    (0,0) [A]  (0,1)     (0,2)     (0,3) [B]
    (1,0)      (1,1)     (1,2)     (1,3)
    (2,0)      (2,1)     (2,2)     (2,3)
    (3,0) [A+] (3,1)     (3,2)     (3,3) [D]

Special States:
    - A  (0,0): "up"/"left" -> A+ with +10; "down"/"right" -> B with +5
    - B  (0,3): "up"/"right" -> C (0,2) with +5; "down"/"left" -> D (3,3) with +5
    - A+ (3,0): Terminal (absorbing); +10 on entry from A, then no further reward
    - D  (3,3): Terminal (absorbing); +5 on entry from B, then no further reward

All other states: deterministic transitions in the chosen direction, reward -1.
"""

from __future__ import annotations

import numpy as np

from rl_envs.gridworld import (
    GreedyAgent,
    GridWorld,
    RandomAgent,
    run_episode,
    run_multiple_episodes,
)

__all__ = [
    # Re-exported from rl_envs.gridworld
    "GridWorld",
    "GreedyAgent",
    "RandomAgent",
    "run_episode",
    "run_multiple_episodes",
    # Chapter 1 figures and demo
    "main",
    "plot_gridworld_figure",
]

# GridWorld, the Sec. 1.3 agents and the episode runners live in the shared
# ``rl_envs`` package (``rl_envs.gridworld``); they are imported here so
# existing imports (``from exercise_1_1 import GridWorld, ...``) keep working.


def plot_gridworld_figure(save_path: str | None = None, show: bool = False):
    """Create a demonstration chart of this chapter's GridWorld example.

    Sutton & Barto (2nd ed.), Section 1.3 "An Example Agent", Figure 1.2
    (gridworld example): the book introduces the RL agent-environment interface
    with a gridworld MDP (states as grid cells, actions as moves, rewards on
    transitions, and terminal/special outcomes).

    This repository implements an adapted 4x4 gridworld demonstration (see
    ``GridWorld`` in this module and ``chapter_01/README.md``): special states
    A (0,0), B (0,3), C (0,2), terminals A+ (3,0, reward +10 on entry from A)
    and D (3,3), with deterministic moves and step reward -1 in ordinary cells.

    The figure visualizes the implemented environment layout and the special
    transition/reward structure so students can connect code to the book's
    gridworld concepts (states, actions, rewards, terminal states).

    Args:
        save_path: Optional path to save the figure (e.g. PNG). Parent
            directories should exist.
        show: If True, display the figure window (default False for headless use).

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    # env: environment instance being controlled (book Chapter 1).
    env = GridWorld(grid_width=4, grid_height=4, discount=1.0)
    # (fig, ax): result of plt.subplots(...) (book Chapter 1).
    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    # Draw grid cells (row 0 at top to match render/layout in this module).
    for r in range(env.grid_height):
        # Iterate over c: one pass per index (book algorithm).
        for c in range(env.grid_width):
            # state: current state of the environment or MDP (book Chapter 1).
            state = (r, c)
            if state == env.state_A_plus:
                face = "#d9f2d9"
            elif state == env.state_D:
                face = "#f9d6d5"
            elif state == env.state_A:
                face = "#dcecff"
            elif state == env.state_B:
                face = "#fff2cc"
            elif state == env.state_C:
                face = "#f3e8ff"
            else:
                face = "#ffffff"
            # rect: result of Rectangle(...) (book Chapter 1).
            rect = Rectangle((c, r), 1, 1, facecolor=face, edgecolor="#444444", linewidth=1.2)
            ax.add_patch(rect)
            # label: intermediate value used by the algorithm (book Chapter 1).
            label = f"({r},{c})"
            if state == env.state_A:
                label = "A (0,0)"
            elif state == env.state_B:
                label = "B (0,3)"
            elif state == env.state_A_plus:
                label = "A+ terminal\nr=+10"
            elif state == env.state_D:
                label = "D terminal"
            elif state == env.state_C:
                label = "C (0,2)"
            ax.text(c + 0.5, r + 0.5, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, text, color="#333333"):
        """arrow (reference implementation).

        Book: Sutton & Barto (2nd ed.), Chapter 1.

        Args:
            x1: See the relevant book section and function body.
            y1: See the relevant book section and function body.
            x2: See the relevant book section and function body.
            y2: See the relevant book section and function body.
            text: See the relevant book section and function body.
            color: See the relevant book section and function body.

        Returns:
            Value produced by the implementation (see the book section).

        Book: Sutton & Barto (2nd ed.), Section 1.1."""
        # carry out the arrow step (book Chapter 1).
        # patch: result of FancyArrowPatch(...) (book Chapter 1).
        patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12,
                                linewidth=1.4, color=color)
        ax.add_patch(patch)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 0.08, text, fontsize=7.5,
                ha="center", va="top", color=color,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})

    # Annotate special transitions (book concepts: actions -> next state, reward).
    # From A: UP/LEFT -> A+ (+10), DOWN/RIGHT -> B (+5)
    arrow(0.35, 0.65, 0.35, 2.6, "A: up/left\nr=+10", color="#2e7d32")
    arrow(0.75, 0.35, 2.6, 0.35, "A: down/right\nr=+5", color="#1565c0")
    # From B: UP/RIGHT -> C (+5), DOWN/LEFT -> D (+5)
    arrow(3.35, 0.65, 2.65, 0.65, "B: up/right\nr=+5", color="#6a1b9a")
    arrow(3.55, 0.75, 3.55, 2.6, "B: down/left\nr=+5", color="#6a1b9a")

    ax.set_xlim(0, env.grid_width)
    ax.set_ylim(env.grid_height, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.grid_width + 1))
    ax.set_yticks(range(env.grid_height + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_title(
        "GridWorld demonstration (adapted from Sutton & Barto, 2nd ed.,\n"
        "Sec. 1.3, Fig. 1.2): states, actions, rewards, terminals"
    )
    ax.text(0.0, 4.35,
            "Ordinary moves: deterministic step, r = -1 (unless off-grid: stay in place).",
            fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        matplotlib.use("TkAgg", force=True)
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def main() -> None:
    """Run a demo of the Gridworld environment.

    Book: Sutton & Barto (2nd ed.), Section 1.1 (Chapter 1)."""
    # carry out the main step (book Chapter 1).
    print("=" * 60)
    print("Gridworld Environment -- Sutton & Barto, Section 1.3")
    print("=" * 60)

    # env: environment instance being controlled (book Chapter 1).
    env = GridWorld(grid_width=4, grid_height=4, discount=1.0)
    # agent: result of RandomAgent(...) (book Chapter 1).
    agent = RandomAgent(n_actions=4)

    print("\n--- Initial Grid ---")
    print(env.render())

    print("\n--- Running 5 episodes with random agent ---")
    # stats: result of run_multiple_episodes(...) (book Chapter 1).
    stats = run_multiple_episodes(env, agent, n_episodes=5, epsilon=0.0)
    print(f"\nAverage total reward: {np.mean(stats['total_reward']):.2f}")
    print(f"Average steps per episode: {np.mean(stats['steps']):.1f}")

    print("\n--- Running 5 episodes with greedy agent ---")
    # greedy agent: result of GreedyAgent(...) (book Chapter 1).
    greedy_agent = GreedyAgent(n_actions=4)
    # stats2: result of run_multiple_episodes(...) (book Chapter 1).
    stats2 = run_multiple_episodes(env, greedy_agent, n_episodes=5, epsilon=0.0)
    print(f"\nAverage total reward: {np.mean(stats2['total_reward']):.2f}")
    print(f"Average steps per episode: {np.mean(stats2['steps']):.1f}")

    print("\n--- Special State Transitions ---")
    env.reset()
    print(f"State A: {env.state_A}")
    # Iterate over (name, idx): each item in [("UP", GridWorld.UP), ("LEFT", GridWorld.LEFT), ...
    for name, idx in [("UP", GridWorld.UP), ("LEFT", GridWorld.LEFT),
                      ("DOWN", GridWorld.DOWN), ("RIGHT", GridWorld.RIGHT)]:
        print(f"  {name:6s} -> {env.get_next_state(env.state_A, idx)} reward={env.get_reward(env.state_A, idx)}")
    print(f"\nState B: {env.state_B}")
    # Iterate over (name, idx): each item in [("UP", GridWorld.UP), ("LEFT", GridWorld.LEFT), ...
    for name, idx in [("UP", GridWorld.UP), ("LEFT", GridWorld.LEFT),
                      ("DOWN", GridWorld.DOWN), ("RIGHT", GridWorld.RIGHT)]:
        print(f"  {name:6s} -> {env.get_next_state(env.state_B, idx)} reward={env.get_reward(env.state_B, idx)}")


if __name__ == "__main__":
    main()
