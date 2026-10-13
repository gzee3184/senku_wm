"""Generate OGBench Scene's official 5 evaluation tasks.

Each episode has a start state and a goal state, both stored as two consecutive rows.
Success is determined by the environment's all-objects goal check.

Usage: python scripts/data/make_scene_tasks.py --n 10 --seed 0 --out H5
"""
import argparse

import gymnasium
import h5py
import numpy as np
import ogbench.manipspace  # noqa: F401

KW = dict(width=224, height=224, mode="task", ob_type="pixels", visualize_info=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="episodes per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = gymnasium.make("visual-scene-v0", **KW).unwrapped
    chk = gymnasium.make("visual-scene-v0", **KW).unwrapped
    chk.reset(seed=0)
    px, qp, qv, b0, b1, task = [], [], [], [], [], []
    worst = 0.0
    for t in range(1, 6):
        for i in range(a.n):
            ob, info = env.reset(seed=a.seed * 100000 + t * 1000 + i, options={"task_id": t})
            q, v = env._data.qpos.copy(), env._data.qvel.copy()
            b = env._cur_button_states.copy()
            gi = env.cur_task_info["goal"]
            gq = q.copy()
            gq[14:17] = gi["block_xyzs"][0]
            gq[23], gq[24] = gi["drawer_pos"], gi["window_pos"]
            chk.set_state(q, v, b)
            worst = max(worst, float(np.abs(chk.render().astype(float) - ob.astype(float)).mean()))
            px += [ob, info["goal"]]
            qp += [q, gq]
            qv += [v, np.zeros_like(v)]
            b0 += [b[0], gi["button_states"][0]]
            b1 += [b[1], gi["button_states"][1]]
            task += [t, t]
    n = 5 * a.n
    print(f"{n} episodes; start re-render mean |diff| worst {worst:.3f} / 255", flush=True)
    assert worst < 1.0, "re-setting the start state does not reproduce OGBench's start frame"
    with h5py.File(a.out, "w") as f:
        f["pixels"] = np.stack(px).astype(np.uint8)
        f["privileged_qpos"] = np.stack(qp).astype(np.float32)
        f["privileged_qvel"] = np.stack(qv).astype(np.float32)
        f["privileged_button_0_state"] = np.asarray(b0, np.float32)
        f["privileged_button_1_state"] = np.asarray(b1, np.float32)
        f["action"] = np.zeros((2 * n, 5), np.float32)
        f["task_id"] = np.asarray(task, np.int32)
        f["ep_len"] = np.full(n, 2, np.int32)
        f["ep_offset"] = np.arange(0, 2 * n, 2, dtype=np.int64)


if __name__ == "__main__":
    main()
