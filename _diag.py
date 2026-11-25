import numpy as np
import chapter_08.solution as S

walls = ["1,2", "2,2", "3,2", "0,7", "1,7", "2,7", "4,5"]
print("bfs optimal k=1:", S._bfs_path_len(6, 9, walls, "2,0", "0,8"))

for rep, rs in enumerate([1000, 1137, 1274]):
    env = S.make_gridworld(rows=6, cols=9, start="2,0", goal="0,8",
                           gamma=0.95, seed=rs, walls=walls)
    q, m, stats = S.dyna_q(env, n_planning=5, n_episodes=120, alpha=0.1,
                           epsilon=0.1, gamma=0.95, seed=rs, max_steps=10000,
                           start_state="2,0")
    gl = S._greedy_path_len(q, 6, 9, walls, "2,0", "0,8", S._bfs_path_len(6, 9, walls, "2,0", "0,8"))
    print(f"repeat {rep} (seed {rs}): greedy_path_len={gl}")
