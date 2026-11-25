import numpy as np
import chapter_08.solution as S

walls = ["1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"]
env = S.make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                       gamma=0.95, seed=1137, walls=walls)
q, m, stats = S.dyna_q(env, n_planning=5, n_episodes=300, alpha=0.1,
                       epsilon=0.1, gamma=0.95, seed=1137, max_steps=10000,
                       start_state="2,0")
s = "2,0"
deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
wall_set = set(walls)
for t in range(20):
    A = ("up", "down", "left", "right")
    vals = [q[(s, a)] for a in A]
    order = sorted(zip(A, vals), key=lambda kv: -kv[1])
    a = order[0][0]
    print(f"{t:2d} {s:5s} " + "  ".join(f"{aa}:{v:+.4f}" for aa, v in order))
    r, c = (int(x) for x in s.split(","))
    dr, dc = deltas[a]
    ns = f"{r + dr},{c + dc}"
    if 0 <= r + dr < 6 and 0 <= c + dc < 9 and ns not in wall_set:
        s = ns
    if s == "0,8":
        print("goal after", t + 1, "steps")
        break
