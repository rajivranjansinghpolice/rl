import time
import numpy as np
import chapter_08.solution as S

walls = ["1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"]
q_star = S._vi_qstar(6, 9, walls, "0,8", gamma=0.95)
print("q* pairs:", len(q_star))

for rs in (1137, 1274):
    t0 = time.time()
    env = S.make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                           gamma=0.95, seed=rs, walls=walls)
    visited = set()

    def cb(ep, steps, state, q_table, pops):
        visited.add(state)

    q, m, stats = S.dyna_q(env, n_planning=5, n_episodes=300, alpha=0.1,
                           epsilon=0.1, gamma=0.95, seed=rs, max_steps=10000,
                           start_state="2,0", step_callback=cb)
    gl = S._greedy_path_len(q, 6, 9, walls, "2,0", "0,8", 14)
    print(f"seed {rs}: greedy path len = {gl}, visited states = {len(visited)}, elapsed {time.time()-t0:.1f}s")
    # error over visited states/pairs only
    err_v = 0.0
    for s in visited:
        vs = max(q.get((s, a), 0.0) for a in ("up", "down", "left", "right"))
        vstar = max(q_star[(s, a)] for a in ("up", "down", "left", "right"))
        err_v = max(err_v, abs(vs - vstar))
    print(f"   max |V_greedy - V*| over visited: {err_v:.4f}")
    err_all = max(abs(q.get(p, 0.0) - t) for p, t in q_star.items())
    print(f"   max |Q - Q*| over ALL pairs:     {err_all:.4f}")
    nzero = sum(1 for p in q_star if q.get(p, 0.0) == 0.0)
    print(f"   untouched (q==0) pairs:          {nzero}/{len(q_star)}")
