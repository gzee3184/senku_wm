"""Short-horizon Visual AntMaze: reach the expert's position K steps ahead (LeWM-style goal = the dataset frame).

Starts come from H-JEPA's held-out validation episodes (visual_antmaze_medium_stitch_val.h5, never trained on). The env
is set to the recorded qpos/qvel at step t, the context is the recorded frames before t, and the goal is the recorded
frame at t + K. Only windows where the expert moves at least --min-disp are used (in most 25-step windows the ant moves
less than the 1.0 success radius). Success: within 1.0 of the expert's xy at t + K, budget 2 K steps.
--policy expert replays the recorded actions (harness check, should be ~100%), --policy still sends zero actions.

    python eval_antmaze_short.py --ckpt WM.pt --goal-offset 25 --seed 42 --out-json OUT.json
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

from eval_antmaze import VecAnt, make_goal_input, proprio_of  # noqa: E402
from senku.paths import ANTMAZE, data  # noqa: E402

VAL = "visual_antmaze_medium_stitch_val.h5"


def sample_windows(f, K, n, min_disp, seed, span):
    off, L, xy = f["ep_offset"][:], f["ep_len"][:], f["xy"][:]
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        e = int(rng.integers(len(L)))
        t = int(rng.integers(span, L[e] - 1 - K))
        r = int(off[e]) + t
        if np.linalg.norm(xy[r + K] - xy[r]) >= min_disp:
            out.append((e, t, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--policy", default="plan", choices=["plan", "expert", "still"])
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--budget", type=int, default=None, help="default 2 x goal offset")
    ap.add_argument("--min-disp", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=None, help="planning horizon in 5-step blocks (default K / 5)")
    ap.add_argument("--receding-horizon", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--cost-mode", choices=["g", "tokens", "both"], default="both")
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--goal-proprio", choices=["task", "current", "mean", "none"], default="task",
                    help="joint readings paired with the goal image: the task's (H-JEPA), the ant's current ones, the "
                         "training mean, or none (image only)")
    ap.add_argument("--solver", choices=["cem", "gd"], default="cem",
                    help="gd: the gradient planner with H-JEPA's AntMaze settings")
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401
    from sklearn import preprocessing

    from senku.envs.ogb_antmaze import AntMazeEnv
    from senku.planning import FRAME_SKIP, HIST_LEN
    K = args.goal_offset
    budget = args.budget or 2 * K
    span = (HIST_LEN - 1) * FRAME_SKIP
    f = h5py.File(data(args.val), "r")
    wins = sample_windows(f, K, args.episodes, args.min_disp, args.seed, span)
    N = len(wins)
    envs = [AntMazeEnv(maze_type="medium", dataset_type="stitch", width=64, height=64, camera_name="back")
            for _ in range(N)]
    obs, goals, hist, goal_xy, exp_act = [], [], [], [], []
    for i, (env, (e, t, r)) in enumerate(zip(envs, wins)):
        env.reset(seed=args.seed * 1000 + i)
        env.env.set_state(f["qpos"][r].astype(np.float64), f["qvel"][r].astype(np.float64))
        gxy = f["xy"][r + K].astype(np.float64)
        env.set_navigation_goal(goal_xy=gxy)
        obs.append(env.env.get_ob())
        goals.append(f["pixels"][r + K])
        hist.append(list(f["pixels"][r - span:r]))
        goal_xy.append(gxy)
        exp_act.append(f["action"][r:r + K])
    obs, goals, goal_xy = np.stack(obs), np.stack(goals), np.stack(goal_xy)
    rec = np.stack([f["pixels"][w[2]] for w in wins]).astype(np.float32)
    pix_err = float(np.abs(obs.astype(np.float32) - rec).mean())
    print(f"{N} windows, +{K}, budget {budget}, rendered vs recorded start frame: mean abs diff {pix_err:.2f}/255",
          flush=True)
    if args.policy == "plan":
        import stable_worldmodel as swm
        import torch  # noqa: F401

        from senku.model import load_model
        from senku.planning import CostModel, GDSolver, HeldCEM, HistoryPolicy, img_transform
        with h5py.File(data(ANTMAZE), "r") as g:
            A = g["action"][:]
        sc = preprocessing.StandardScaler().fit(A[~np.isnan(A).any(axis=1)])
        model = CostModel(load_model(args.ckpt, "cuda"), sc, args.cost_mode).to("cuda")
        tf = img_transform(224)
        H = args.horizon or max(1, K // FRAME_SKIP)
        config = swm.PlanConfig(horizon=H, receding_horizon=min(args.receding_horizon, H), action_block=FRAME_SKIP,
                                warm_start=args.solver == "cem")
        if args.solver == "gd":
            solver = GDSolver(model, budget=budget, seed=args.seed)
        else:
            solver = HeldCEM(model=model, batch_size=1, num_samples=args.num_samples, var_scale=1.0,
                             n_steps=args.n_steps, topk=30, device="cuda", seed=args.seed, hold=args.hold)
        policy = HistoryPolicy(solver=solver, config=config, process={"action": sc},
                               transform={"pixels": tf, "goal": tf, "pixels_hist": tf})
        policy.set_env(VecAnt(envs))
        use_prop = bool(model.model.proprio_dim)
        goal_input = make_goal_input(args.goal_proprio, model.model)
        if use_prop:   # recorded proprio before t; goal = the recorded proprio at t + K (H-JEPA's dataset_full goal)
            from senku.data import proprio_rows
            goal_prop = np.stack([proprio_rows(f, r + K, r + K + 1)[0] for _, _, r in wins])
            policy.seed_history(hist, [proprio_rows(f, r - span, r) for _, _, r in wins])
        else:
            policy.seed_history(hist)
    done = np.zeros(N, bool)
    succ_t = np.full(N, -1)

    def dist():
        return np.array([np.linalg.norm(e.env.get_xy() - g) for e, g in zip(envs, goal_xy)])

    d_start = dist()
    d_min = d_start.copy()
    t0 = time.time()
    for t in range(budget):
        if args.policy == "plan":
            info = {"pixels": obs[:, None], "goal": goals[:, None], "terminated": done.copy()}
            if use_prop:
                cur = np.stack([proprio_of(e) for e in envs])
                info.update(proprio=cur, goal_proprio=goal_input(goal_prop, cur))
            act = np.asarray(policy.get_action(info)).reshape(N, -1)
        elif args.policy == "expert":
            act = np.stack([a[t] if t < len(a) else np.zeros(8, np.float32) for a in exp_act])
        else:
            act = np.zeros((N, 8), np.float32)
        for i, e in enumerate(envs):
            if done[i]:
                continue
            o, _, term, _, inf = e.step(act[i].astype(np.float32))
            obs[i] = o
            if inf.get("success", 0.0) > 0 or term:
                done[i], succ_t[i] = True, t + 1
        d_min = np.minimum(d_min, dist())
        if done.all():
            break
    rate = 100.0 * done.mean()
    print(f"RESULT antmaze_short policy={args.policy} ckpt={args.ckpt} +{K} solver={args.solver} hold={args.hold} "
          f"seed={args.seed}: "
          f"success={rate:.1f}% ({done.sum()}/{N}) start {d_start.mean():.2f} closest {d_min.mean():.2f} "
          f"end {dist().mean():.2f} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        out = dict(vars(args), success_rate=rate, successes=done.tolist(), windows=[list(w) for w in wins],
                   steps_to_success=succ_t.tolist(), dist_start=d_start.tolist(), dist_min=d_min.tolist(),
                   dist_end=dist().tolist(), start_pix_err=pix_err)
        Path(args.out_json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
