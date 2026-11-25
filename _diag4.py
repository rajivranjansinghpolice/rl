import time
import chapter_08.solution as S

walls = ["1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"]
A = ("up", "down", "left", "right")
deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
wall_set = set(walls)

def rollout(q):
    s = "2,0"
    path = [s]
    for t in range(40):
        if s == "0,8":
            return t
        a = A[max(range(4), key=lambda i: q[(s, A[i])])]
        r, c = (int(x) for x in s.split(","))
        dr, dc = deltas[a]
        ns = f"{r + dr},{c + dc}"
        if 0 <= r + dr < 6 and 0 <= c + dc < 9 and ns not in wall_set:
            s = ns
        path.append(s)
    return None

for rs in (1000, 1137, 1274):
    env = S.make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                           gamma=0.95, seed=rs, walls=walls)
    # check rollout after several training budgets
    for n_ep in (300, 700, 1200):
        env = S.make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                               gamma=0.95, seed=rs, walls=walls)
        q, m, stats = S.dyna_q(env, n_planning=5, n_episodes=n_ep, alpha=0.1,
                               epsilon=0.1, gamma=0.95, seed=rs, max_steps=10000,
                               start_state="2,0")
        print(f"seed {rs}, {n_ep:4d} episodes: rollout = {rollout(q)}")
