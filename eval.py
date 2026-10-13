"""Evaluate a world model with closed-loop planning on Cube, PushT, or OGBench Scene.

    python eval.py --ckpt CKPT --env cube --seed 42 --holdout-mod 10 --episodes 50
    python eval.py --ckpt CKPT --env cube --start-filter pick --seed 7 --holdout-mod 10
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402, F401
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from sklearn import preprocessing  # noqa: E402

from senku import tasks  # noqa: E402
from senku.model import load_model  # noqa: E402
from senku.paths import CKPT, CUBE, PUSHT, SCENE, data  # noqa: E402
from senku.planning import (  # noqa: E402
    FRAME_SKIP, HIST_LEN, CostModel, GDSolver, HeldCEM, HistoryPolicy, LeWMCostModel, LeWMGDCost, SolveTimer,
    img_transform)

CUBE_CALLABLES = [
    {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
    {"method": "set_target_pos", "args": {
        "cube_id": {"value": 0, "in_dataset": False},
        "target_pos": {"value": "goal_privileged_block_0_pos"},
        "target_quat": {"value": "goal_privileged_block_0_quat"}}},
]
ENVS = {
    "cube": dict(env_name="swm/OGBCube-v0", h5=CUBE, lewm="lewm-cube", callables=CUBE_CALLABLES, keys=["action"],
                 world_kw=dict(env_type="single", ob_type="states", multiview=False, width=224, height=224,
                               visualize_info=False, terminate_at_goal=True)),
    "pusht": dict(env_name="swm/PushT-v1", h5=PUSHT, lewm="lewm-pusht",
                  callables=[{"method": "_set_state", "args": {"state": {"value": "state"}}},
                             {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}}],
                  keys=["action", "proprio", "state"], world_kw={}),
    "scene": dict(env_name="visual-scene-v0", h5=SCENE, lewm=None, callables=None, keys=["action"],
                  world_kw=dict(width=224, height=224, terminate_at_goal=True, mode="task", ob_type="pixels",
                                visualize_info=False)),
}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Senku checkpoint; omit with --lewm")
    ap.add_argument("--lewm", action="store_true", help="evaluate LeWM ($SENKU_HOME/checkpoints/lewm-<env>/)")
    ap.add_argument("--env", choices=list(ENVS), default="cube")
    ap.add_argument("--cost-mode", choices=["g", "tokens", "both", "upper"], default="both",
                    help="planning cost: g, state tokens, or both z-scored and summed; upper = long-horizon levels")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--eval-budget", type=int, default=50)
    ap.add_argument("--holdout-mod", type=int, default=0, help="starts only from episodes with id %% N == R")
    ap.add_argument("--holdout-rem", type=int, default=0,
                    help="R for --holdout-mod; 0 = held-out test starts, 1 = validation starts (training episodes)")
    ap.add_argument("--start-filter", choices=["none", "hard", "hard_far", "pick", "place_away", "scene_changed"],
                    default="none")
    ap.add_argument("--task-file", default=None, help="H-JEPA eval_trajs task file (episodes_idx + start_steps)")
    ap.add_argument("--task-ep-offset", type=int, default=9800, help="absolute episode of the task file's episode 0")
    ap.add_argument("--task-h5", default=None, help="dataset of the task file's episodes (default: the training data)")
    ap.add_argument("--scene-tasks", default=None, help="OGBench Scene official 5-task file")
    # planner
    ap.add_argument("--solver", choices=["held", "cem", "gd"], default="held",
                    help="held = HeldCEM; cem = LeWM's CEM; gd = H-JEPA's gradient planner")
    ap.add_argument("--gd-samples", type=int, default=800, help="gd: candidates (H-JEPA's PushT and Cube: 800)")
    ap.add_argument("--gd-hold", action="store_true", help="gd: one action per 5-step block, held (as HeldCEM)")
    ap.add_argument("--no-hold", action="store_true", help="held: per-step actions instead of held blocks")
    ap.add_argument("--gd-lr", type=float, default=0.05, help="gd: AdamW lr (H-JEPA's PushT and Cube: 0.05)")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--receding-horizon", type=int, default=5)
    ap.add_argument("--keep-elite-frac", type=float, default=0.3)
    # long horizon
    ap.add_argument("--expert-subgoals", type=int, default=0,
                    help="use the episode's own dataset frames as subgoals every K steps")
    ap.add_argument("--subgoal-latents", choices=["real", "imagined"], default="real",
                    help="--expert-subgoals: encoded frames, or the model's latents along the expert actions")
    ap.add_argument("--subgoal-advance", choices=["time", "model"], default="time")
    ap.add_argument("--imagined-subgoals", type=int, default=0, help="subgoal spacing in env steps; 0 = off")
    ap.add_argument("--lh-search", choices=["cem", "rep", "upper"], default="cem")
    ap.add_argument("--lh-block", type=int, default=25)
    ap.add_argument("--lh-samples", type=int, default=300)
    ap.add_argument("--lh-iters", type=int, default=30)
    ap.add_argument("--lh-seg-budget", type=int, default=50)
    ap.add_argument("--lh-adv-frac", type=float, default=0.5)
    ap.add_argument("--lh-horizon", type=int, default=None, help="outer horizon in env steps (default --goal-offset)")
    ap.add_argument("--td-head", default=None, help="--lh-search rep: steps-apart head")
    ap.add_argument("--lh-rep", default=None, help="--lh-search rep: repertoire checkpoint")
    ap.add_argument("--lh-rep-beams", type=int, default=8)
    ap.add_argument("--lh-rep-top", type=int, default=8)
    ap.add_argument("--lh-rep-shuffle", action="store_true", help="give repertoire random states")
    ap.add_argument("--lh-rep-direct", action="store_true", help="execute repertoire open-loop")
    ap.add_argument("--lh-upper", default=None, help="--lh-search upper: upper-level checkpoint")
    ap.add_argument("--baseline", choices=["none", "still", "replay"], default="none",
                    help="no planner: zero actions, or the recorded actions of the start's episode")
    ap.add_argument("--video", default=None)
    ap.add_argument("--out-json", default=None)
    return ap


def dataset_pre_frames(h5, ep_idx, start_steps, n_pre):
    """Get frames from the dataset before each start step. Repeat the first frame if needed."""
    out = []
    with h5py.File(h5, "r") as f:
        off = f["ep_offset"][:]
        for e, s in zip(ep_idx, start_steps):
            out.append(np.stack([f["pixels"][int(off[e]) + max(int(s) - k, 0)] for k in range(n_pre, 0, -1)]))
    return out


class BaselinePolicy(swm.policy.BasePolicy):
    """No planner. still: zero actions (the agent does not move). replay: the episode's recorded actions from the
    start to the goal frame, then zero actions."""

    def __init__(self, h5, eps, starts, goal_offset, budget, mode):
        super().__init__()
        with h5py.File(h5, "r") as f:
            off, A = f["ep_offset"][:], f["action"]
            ad = A.shape[1]
            self.acts = np.zeros((len(eps), budget, ad), np.float32)
            if mode == "replay":
                for i, (e, s) in enumerate(zip(eps, starts)):
                    r = int(off[e]) + int(s)
                    self.acts[i, :goal_offset] = np.nan_to_num(A[r:r + goal_offset])
        self.t = 0

    def get_action(self, info_dict, **kwargs):
        a = self.acts[:, min(self.t, self.acts.shape[1] - 1)]
        self.t += 1
        return a


def main():
    args = build_parser().parse_args()
    assert args.lewm != bool(args.ckpt), "pass exactly one of --ckpt and --lewm"
    device = "cuda"
    E = ENVS[args.env]
    h5 = str(data(E["h5"]))
    if args.env == "scene":
        import ogbench.manipspace  # noqa: F401
        from senku.envs import scene
        scene.install()
        E = dict(E, callables=scene.CALLABLES)
    if args.scene_tasks:
        assert args.env == "scene" and args.goal_offset == 1
        with h5py.File(args.scene_tasks, "r") as f:
            args.episodes = len(f["ep_len"])
    world = swm.World(env_name=E["env_name"], num_envs=args.episodes, max_episode_steps=2 * args.eval_budget,
                      image_shape=(224, 224), **E["world_kw"])
    dataset = swm.data.HDF5Dataset(path=h5, keys_to_cache=E["keys"])
    if args.env == "scene":
        scene.prepare_dataset(dataset)
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    act = dataset.get_col_data("action")
    act_scaler = preprocessing.StandardScaler().fit(act[~np.isnan(act).any(axis=1)])
    process = {"action": act_scaler}
    for k in E["keys"][1:]:
        v = dataset.get_col_data(k)
        process[k] = process[f"goal_{k}"] = preprocessing.StandardScaler().fit(v[~np.isnan(v).any(axis=1)])
    if args.scene_tasks:
        h5 = args.scene_tasks
        dataset = swm.data.HDF5Dataset(path=h5, keys_to_cache=E["keys"])
        scene.prepare_dataset(dataset)
    elif args.task_h5:   # the action and joint scalers stay fit on the training data
        h5 = args.task_h5
        dataset = swm.data.HDF5Dataset(path=h5, keys_to_cache=E["keys"])
    ep_indices = np.unique(dataset.get_col_data(col))
    tf = img_transform(224)
    transform = {"pixels": tf, "goal": tf, "pixels_hist": tf}
    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon, action_block=FRAME_SKIP,
                            warm_start=args.solver != "gd")   # H-JEPA's planner: no warm start

    if args.lewm:
        lewm = swm.wm.utils.load_pretrained(str(CKPT / E["lewm"] / "weights.pt")).to(device).eval()
        lewm.requires_grad_(False)
        lewm.interpolate_pos_encoding = True
        model = LeWMCostModel(lewm, act_scaler).to(device) if args.solver in ("held", "gd") else lewm
        policy_cls = HistoryPolicy if args.solver in ("held", "gd") else swm.policy.WorldModelPolicy
    else:
        model = CostModel(load_model(args.ckpt, device), act_scaler, args.cost_mode).to(device)
        policy_cls = HistoryPolicy
    solver_kw = dict(model=model, batch_size=1, num_samples=args.num_samples, var_scale=1.0, n_steps=args.n_steps,
                     topk=args.topk, device=device, seed=args.seed)
    if args.solver == "gd":
        solver = GDSolver(model, num_samples=args.gd_samples, lr=args.gd_lr, budget=args.eval_budget, seed=args.seed,
                          cost=LeWMGDCost(lewm) if args.lewm else None, hold=args.gd_hold)
    elif args.solver == "held":
        solver = HeldCEM(keep_elite_frac=args.keep_elite_frac, hold=not args.no_hold, **solver_kw)
    else:
        solver = swm.solver.CEMSolver(**solver_kw)
    if policy_cls is swm.policy.WorldModelPolicy:
        transform.pop("pixels_hist")
    policy = policy_cls(solver=solver, config=config, process=process, transform=transform)

    ep_col, step_col = dataset.get_col_data(col), dataset.get_col_data("step_idx")
    ep_len = np.array([np.max(step_col[ep_col == e]) + 1 for e in ep_indices])
    max_start = dict(zip(ep_indices, ep_len - args.goal_offset - 1))
    valid = np.nonzero(step_col <= np.array([max_start[e] for e in ep_col]))[0]
    if args.start_filter == "scene_changed":
        valid = scene.changed_start_rows(dataset, valid, args.goal_offset)
    elif args.start_filter != "none":
        valid = tasks.FILTERS[args.start_filter](h5, valid, args.goal_offset)
    if args.holdout_mod:
        valid = valid[ep_col[valid] % args.holdout_mod == args.holdout_rem]
    if args.task_file:
        t = torch.load(args.task_file, weights_only=False)
        # H-JEPA's evaluation uses the frame goal_offset_steps - 1 after the start; both are allowed
        assert args.goal_offset in (t["goal_offset_steps"], t["goal_offset_steps"] - 1), (t["goal_offset_steps"],
                                                                                         args.goal_offset)
        idx = np.array([np.nonzero((ep_col == args.task_ep_offset + e) & (step_col == s))[0][0]
                        for e, s in zip(t["episodes_idx"], t["start_steps"])])[:args.episodes]   # first N tasks
    elif args.scene_tasks:
        idx = valid
    else:
        g = np.random.default_rng(args.seed)
        idx = np.sort(valid[g.choice(len(valid) - 1, size=args.episodes, replace=False)])
    u, inv = np.unique(idx, return_inverse=True)        # h5py needs increasing rows; the task file order is kept
    rows = dataset.get_row_data(u)
    eval_eps, eval_start = rows[col][inv], rows["step_idx"][inv]
    print(f"{len(valid)} valid starts; episodes {list(eval_eps[:8])}...", flush=True)

    if args.baseline != "none":
        policy = BaselinePolicy(h5, eval_eps, eval_start, args.goal_offset, args.eval_budget, args.baseline)
    world.set_policy(policy)
    timer = SolveTimer(solver)
    if policy_cls is HistoryPolicy and args.baseline == "none":
        policy.seed_history(dataset_pre_frames(h5, eval_eps, eval_start, (HIST_LEN - 1) * FRAME_SKIP))
    if args.expert_subgoals:
        n = args.goal_offset // args.expert_subgoals
        assert args.goal_offset % args.expert_subgoals == 0 and not args.lewm
        with h5py.File(h5, "r") as f:
            off, A = f["ep_offset"][:], f["action"]
            r0 = [int(off[e]) + int(s) for e, s in zip(eval_eps, eval_start)]
            frames = np.stack([
                np.stack([f["pixels"][r + args.expert_subgoals * (k + 1)] for k in range(n)])
                for r in r0
            ])
            raw = np.stack([A[r:r + args.goal_offset] for r in r0]).astype(np.float32)
        imagine = None
        if args.subgoal_latents == "imagined":
            from senku.long_horizon.subgoals import ImaginedPlanner
            imagine = ImaginedPlanner(model.model, act_scaler.mean_, act_scaler.scale_)
        assert args.subgoal_advance == "time" or imagine is not None
        policy.seed_subgoals(
            frames, args.eval_budget // n, imagine=imagine, actions=raw, every=args.expert_subgoals,
            advance=(args.lh_seg_budget, args.lh_adv_frac) if args.subgoal_advance == "model" else None
        )
    if args.lh_search == "upper":
        assert args.cost_mode == "upper" and args.lh_upper and not args.lewm
        from senku.long_horizon.upper_levels import UpperChain, load_upper
        model.upper = load_upper(args.lh_upper, device)
        policy.set_long_horizon(up_chain=UpperChain(model.model, model.upper, args.lh_horizon or args.goal_offset,
                                                    samples=args.lh_samples, iters=args.lh_iters, seed=args.seed))
    elif args.imagined_subgoals:
        assert not args.lewm and not args.expert_subgoals
        from senku.long_horizon.subgoals import ImaginedPlanner, SubgoalChain
        lh = ImaginedPlanner(model.model, act_scaler.mean_, act_scaler.scale_, block=args.lh_block,
                             samples=args.lh_samples, iters=args.lh_iters, seed=args.seed, search=args.lh_search)
        if args.lh_search == "rep":
            from senku.long_horizon.repertoire import load_rep
            from senku.long_horizon.td_head import load_td
            assert args.td_head and args.lh_rep and args.imagined_subgoals == 25
            lh.td_head, lh.rep = load_td(args.td_head, device), load_rep(args.lh_rep, device)
            lh.rep_beams, lh.rep_top, lh.rep_shuffle = args.lh_rep_beams, args.lh_rep_top, args.lh_rep_shuffle
        chain = SubgoalChain(lh, len(eval_eps), args.lh_horizon or args.goal_offset, sub_every=args.imagined_subgoals,
                             seg_budget=args.lh_seg_budget, adv_frac=args.lh_adv_frac)
        chain.direct = args.lh_rep_direct
        policy.set_long_horizon(lh_chain=chain)

    t0 = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=eval_start.tolist(), goal_offset=args.goal_offset,
                             eval_budget=args.eval_budget, episodes_idx=eval_eps.tolist(), callables=E["callables"],
                             video=args.video)
    minutes = (time.time() - t0) / 60
    samples, iters = (args.gd_samples, None) if args.solver == "gd" else (args.num_samples, args.n_steps)
    extra = {"planner": timer.report(len(eval_eps), samples, iters)}
    succ = np.asarray(metrics["episode_successes"]).astype(bool)
    if args.env == "scene":
        need = scene.start_needs(dataset, idx, args.goal_offset)
        extra["scene"] = {"objects": list(scene.OBJECTS), "needed": need.tolist(),
                          "final_at_goal": scene.final_object_success(world).tolist()}
        if args.scene_tasks:
            with h5py.File(args.scene_tasks, "r") as f:
                tid = f["task_id"][:][idx]
            extra["scene"]["task_id"] = tid.tolist()
            print("SCENE_TASKS " + " ".join(f"task{t} {int(succ[tid == t].sum())}/{int((tid == t).sum())}"
                                           for t in range(1, 6)), flush=True)
    if getattr(policy, "lh_chain", None) is not None:
        extra["subgoal_log"], extra["rep_log"] = policy.lh_chain.log, policy.lh_chain.p.rep_log
    if getattr(policy, "up_chain", None) is not None:
        extra["upper_log"] = policy.up_chain.log
    print(f"RESULT env={args.env} model={'lewm' if args.lewm else args.ckpt} filter={args.start_filter} "
          f"seed={args.seed}: success={metrics['success_rate']:.1f}% ({succ.sum()}/{len(succ)}) [{minutes:.1f} min]",
          flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"args": vars(args), "success_rate": metrics["success_rate"], "episode_successes": succ.tolist(),
                       "episodes": [int(e) for e in eval_eps], "start_steps": [int(s) for s in eval_start],
                       "minutes": minutes, **extra}, f, indent=1)


if __name__ == "__main__":
    main()
