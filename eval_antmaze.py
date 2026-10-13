"""Evaluate a world model on H-JEPA's 50 Visual AntMaze tasks (CEM or the gradient planner).

Each task starts from its recorded state; the goal is the task's goal image. The 64 x 64 frames are resized to
224 x 224. Success: the ant's xy within 1.0 of the goal xy. Budget: 350 env steps.

    python eval_antmaze.py --ckpt WM.pt --seed 42 --out-json OUT.json
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

from senku.paths import ANTMAZE, data  # noqa: E402

TASKS = "expert_grid_d3_n50.pt"


class VecAnt:
    """Wrapper for multiple environments to work with WorldModelPolicy."""

    def __init__(self, envs):
        from gymnasium.vector.utils import batch_space
        self.envs, self.num_envs = envs, len(envs)
        self.single_action_space = envs[0].action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)


def proprio_of(env):
    """qpos[2:] and qvel (H-JEPA's AntMaze proprioception, no xy)."""
    d = env.env.unwrapped.data
    return np.concatenate([d.qpos[2:], d.qvel]).astype(np.float32)


def make_goal_input(mode, wm):
    """Joint readings paired with the goal image (planner side): task = the task's goal joints (H-JEPA's eval),
    current = the ant's current ones, mean = the training mean, none = NaN (the model encodes the image only)."""
    def f(goal_prop, cur):
        if mode == "task":
            return goal_prop
        if mode == "current":
            return cur
        if mode == "mean":
            return np.repeat(wm.proprio_mean.cpu().numpy()[None], len(cur), 0).astype(np.float32)
        return np.full_like(cur, np.nan)
    return f


def setup_task(env, row, seed):
    """Reset to the task's recorded start state and use its stored goal image, as H-JEPA's eval does
    (callables set_state(qpos, qvel) + set_navigation_goal(goal_xy); goal = the file's neutral-pose goal render)."""
    env.reset(seed=seed)
    goal_xy = np.asarray(row["goal_xy"][0], np.float64)
    env.env.set_state(np.asarray(row["qpos"][0], np.float64), np.asarray(row["qvel"][0], np.float64))
    env.set_navigation_goal(goal_xy=goal_xy)
    return env.env.get_ob(), np.ascontiguousarray(np.asarray(row["goal"][0]).transpose(1, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=350)
    ap.add_argument("--goal-proprio", choices=["task", "current", "mean", "none"], default="task",
                    help="joint readings paired with the goal image: the task's (H-JEPA), the ant's current ones, the "
                         "training mean, or none (image only)")
    ap.add_argument("--goal-from", choices=["task", "expert_last"], default="task",
                    help="task: the task file's goal image and goal joints (H-JEPA). expert_last: the task expert's "
                         "last recorded frame and joints (a walking pose within 1.0 of the goal; diagnostic only). "
                         "Success is the goal xy either way")
    ap.add_argument("--solver", choices=["cem", "gd"], default="cem",
                    help="gd: the gradient planner with H-JEPA's AntMaze settings (AdamW lr 0.3, 100 candidates, "
                         "at most 90 steps, decreasing horizon)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="planning horizon in 5-step action blocks (default 16 for cem, 23 for gd as H-JEPA's LeWM)")
    ap.add_argument("--receding-horizon", type=int, default=2, help="blocks executed before replanning (H-JEPA: 2)")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--cost-mode", choices=["g", "tokens", "both"], default="both")
    ap.add_argument("--hold", action="store_true",
                    help="hold each action for its 5-step block (Cube's HeldCEM default). Off here: the ant needs a "
                         "different action every step; replaying the task experts' actions held per block reaches 0/50")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--tasks", default=TASKS, help="task file under $SENKU_HOME/data or a path")
    ap.add_argument("--save-traj", action="store_true", help="also store every env's xy at every step in the json")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    import torch
    from sklearn import preprocessing

    from senku.envs.ogb_antmaze import AntMazeEnv
    from senku.model import load_model
    from senku.planning import (FRAME_SKIP, HIST_LEN, CostModel, GDSolver, HeldCEM, HistoryPolicy, SolveTimer,
                                img_transform)
    dev = "cuda"
    tasks = torch.load(data(args.tasks), weights_only=False)["data"][:args.n_tasks]
    N = len(tasks)
    with h5py.File(data(ANTMAZE), "r") as f:
        A = f["action"][:]
    act_scaler = preprocessing.StandardScaler().fit(A[~np.isnan(A).any(axis=1)])
    model = CostModel(load_model(args.ckpt, dev), act_scaler, args.cost_mode).to(dev)
    tf = img_transform(224)
    args.horizon = args.horizon or (23 if args.solver == "gd" else 16)
    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon, action_block=FRAME_SKIP,
                            warm_start=args.solver == "cem")
    if args.solver == "gd":
        solver = GDSolver(model, budget=args.budget, seed=args.seed)
    else:
        solver = HeldCEM(model=model, batch_size=1, num_samples=args.num_samples, var_scale=1.0, n_steps=args.n_steps,
                         topk=30, device=dev, seed=args.seed, hold=args.hold)
    timer = SolveTimer(solver)
    policy = HistoryPolicy(solver=solver, config=config, process={"action": act_scaler},
                        transform={"pixels": tf, "goal": tf, "pixels_hist": tf})
    envs = [AntMazeEnv(maze_type="medium", dataset_type="stitch", width=64, height=64, camera_name="back")
            for _ in range(N)]
    obs, goals = zip(*[setup_task(e, t, args.seed * 1000 + i) for i, (e, t) in enumerate(zip(envs, tasks))])
    obs, goals = np.stack(obs), np.stack(goals)
    if args.goal_from == "expert_last":
        goals = np.stack([np.ascontiguousarray(np.asarray(t["pixels"][-1]).transpose(1, 2, 0)) for t in tasks])
    policy.set_env(VecAnt(envs))
    span = (HIST_LEN - 1) * FRAME_SKIP
    use_prop = bool(model.model.proprio_dim)
    goal_input = make_goal_input(args.goal_proprio, model.model)
    if use_prop:   # H-JEPA's eval also gives its models the current proprio and the task's goal_proprio
        goal_prop = np.stack([np.asarray(t["goal_proprio" if args.goal_from == "task" else "proprio"]
                                         [0 if args.goal_from == "task" else -1], np.float32) for t in tasks])
        policy.seed_history([[o] * span for o in obs], [[proprio_of(e)] * span for e in envs])
    else:
        policy.seed_history([[o] * span for o in obs])
    done = np.zeros(N, bool)
    succ_t = np.full(N, -1)
    goal_xy = np.stack([np.asarray(t["goal_xy"][0], np.float64) for t in tasks])

    def dist():
        return np.array([np.linalg.norm(e.env.get_xy() - g) for e, g in zip(envs, goal_xy)])

    d_start = dist()
    d_min = d_start.copy()
    traj = [[e.env.get_xy().round(3).tolist()] for e in envs]
    t0 = time.time()
    for t in range(args.budget):
        info = {"pixels": obs[:, None], "goal": goals[:, None], "terminated": done.copy()}
        if use_prop:
            cur = np.stack([proprio_of(e) for e in envs])
            info.update(proprio=cur, goal_proprio=goal_input(goal_prop, cur))
        act = np.asarray(policy.get_action(info)).reshape(N, -1)
        for i, e in enumerate(envs):
            if done[i]:
                continue
            o, _, term, _, inf = e.step(act[i].astype(np.float32))
            obs[i] = o
            if inf.get("success", 0.0) > 0 or term:
                done[i], succ_t[i] = True, t + 1
        d_min = np.minimum(d_min, dist())
        if args.save_traj:
            for i, e in enumerate(envs):
                if not done[i] or len(traj[i]) <= t + 1:
                    traj[i].append(e.env.get_xy().round(3).tolist())
        if done.all():
            break
        if (t + 1) % 50 == 0:
            print(f"step {t + 1}: {done.sum()}/{N} reached, mean dist {dist().mean():.2f} (start {d_start.mean():.2f}) "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    rate = 100.0 * done.mean()
    print(f"RESULT antmaze ckpt={args.ckpt} solver={args.solver} goal={args.goal_from}/{args.goal_proprio} "
          f"cost={args.cost_mode} H={args.horizon} seed={args.seed}: success={rate:.1f}% ({done.sum()}/{N}) "
          f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        out = dict(vars(args), success_rate=rate, successes=done.tolist(), steps_to_success=succ_t.tolist(),
                   dist_start=d_start.tolist(), dist_min=d_min.tolist(), dist_end=dist().tolist(),
                   goal_xy=goal_xy.tolist(), traj=traj if args.save_traj else None,
                   planner=timer.report(N, solver.S if args.solver == "gd" else args.num_samples,
                                        None if args.solver == "gd" else args.n_steps))
        Path(args.out_json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
