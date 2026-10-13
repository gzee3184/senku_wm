"""Noisy-oracle Cube corpus from OGBench's CubePlanOracle.

Generates self-supervised training data using OGBench's plan oracle with noise injection and random actions.
Renders without goal markers. Stores simulator state for re-rendering. Includes wandering, approach-and-recover,
and missed grasps.

Usage: python scripts/data/gen_noisy_oracle_cube.py --out PATH --episodes 1200 --workers 10 [--seed 0]
"""
import argparse
import datetime
import multiprocessing as mp
import os
import subprocess
from pathlib import Path

from senku.paths import CUBE, data  # noqa: E402

import numpy as np

EXPERT = str(data(CUBE))
P_RANDOM, NOISE, SMOOTH, STEPS = 0.3, 0.1, 0.5, 300
_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from senku.envs.cube import make_cube_env as _make_env
        _ENV = _make_env()
        assert not _ENV._visualize_info, "goal marker must stay hidden (expert rendering)"
    return _ENV


def _new_target(env, info, rng):
    lo, hi = env._target_sampling_bounds
    info = dict(info)
    info["privileged/target_block"] = 0
    info["privileged/target_block_pos"] = np.array([*rng.uniform(lo, hi), 0.02])
    info["privileged/target_block_yaw"] = np.array([rng.uniform(0, 2 * np.pi)])
    return info


def _job(job):
    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
    env = _env()
    qpos, qvel, seed = job
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2 ** 32))  # oracle uses np.random for plan noise
    for gid in env._cube_target_geom_ids_list[0]:
        env._model.geom(gid).rgba[3] = 0.0
    env.set_state(qpos, qvel)
    oracle = CubePlanOracle(env=env, noise=NOISE, noise_smoothing=SMOOTH)
    tgt = _new_target(env, env.get_reset_info(), rng)
    oracle.reset(None, tgt)
    keep = {k: tgt[k] for k in (
        "privileged/target_block", "privileged/target_block_pos", "privileged/target_block_yaw"
    )}
    rec = {k: [] for k in ("pixels", "action", "qpos", "qvel", "cube", "quat", "ee", "opening", "contact")}
    info = env.get_reset_info()
    n_targets = 1
    for t in range(STEPS + 1):
        d = env.unwrapped.data
        rec["pixels"].append(env.render())
        rec["qpos"].append(d.qpos.copy())
        rec["qvel"].append(d.qvel.copy())
        rec["cube"].append(np.asarray(info["privileged/block_0_pos"]).copy())
        rec["quat"].append(np.asarray(info["privileged/block_0_quat"]).copy())
        rec["ee"].append(np.asarray(info["proprio/effector_pos"]).copy())
        rec["opening"].append(float(np.ravel(info["proprio/gripper_opening"])[0]))
        rec["contact"].append(float(np.ravel(info["proprio/gripper_contact"])[0]))
        if t == STEPS:
            break
        if rng.uniform() < P_RANDOM:
            a = rng.uniform(-1, 1, 5)
        else:
            a = np.asarray(oracle.select_action(None, {**info, **keep}))
        a = np.clip(a, -1, 1).astype(np.float32)
        rec["action"].append(a)
        _, _, _, _, info = env.step(a)
        if oracle.done:
            tgt = _new_target(env, info, rng)
            keep = {k: tgt[k] for k in keep}
            oracle.reset(None, tgt)
            n_targets += 1
    r = {k: np.stack(v) if k not in ("opening", "contact", "action") else np.asarray(v) for k, v in rec.items()}
    r["action"] = np.concatenate(
        [np.stack(rec["action"]), np.full((1, 5), np.nan, np.float32)]
    )  # terminal NaN, as expert
    r["n_targets"] = n_targets
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import h5py
    import hdf5plugin
    rng = np.random.default_rng(args.seed)
    with h5py.File(EXPERT, "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        train = np.nonzero(np.arange(len(ln)) % 10 != 0)[0]
        src = rng.choice(train, size=args.episodes, replace=False)
        jobs = [(f["qpos"][int(off[e])], f["qvel"][int(off[e])], int(rng.integers(2 ** 62))) for e in src]
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()
    comp = dict(**hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
    with h5py.File(args.out, "w") as h:
        h.attrs.update(dict(
            generator="scripts/data/gen_noisy_oracle_cube.py",
            git_commit=commit,
            created=datetime.datetime.now().isoformat(),
            env="swm/OGBCube-v0 via senku.envs.cube.make_cube_env",
            env_flags=(
                "env_type=single ob_type=states multiview=False width=224 height=224 visualize_info=False "
                "terminate_at_goal=False; target marker alpha forced to 0"
            ),
            goal_marker_visible=False,
            render_reference=EXPERT,
            holdout_rule="starts = first row of expert episodes with ep % 10 != 0 only",
            planner_data=False,
            seed=args.seed,
            policy="ogbench CubePlanOracle",
            plan_noise=NOISE,
            noise_smoothing=SMOOTH,
            p_random_action=P_RANDOM,
            steps=STEPS
        ))
        D, lens, row = {}, [], 0
        meta = {"src_ep": [], "n_targets": [], "max_rise": []}
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, ep in enumerate(pool.imap(_job, jobs, chunksize=1)):
                n = len(ep["pixels"])
                cols = {"pixels": ep["pixels"], "action": ep["action"], "qpos": ep["qpos"], "qvel": ep["qvel"],
                        "privileged_block_0_pos": ep["cube"], "privileged_block_0_quat": ep["quat"],
                        "proprio_effector_pos": ep["ee"], "proprio_gripper_opening": ep["opening"][:, None],
                        "proprio_gripper_contact": ep["contact"][:, None],
                        "ep_idx": np.full(n, len(lens), np.int32), "step_idx": np.arange(n, dtype=np.int64)}
                for c, v in cols.items():
                    if c not in D:
                        kw = comp if c == "pixels" else {}
                        D[c] = h.create_dataset(
                            c, shape=(0, *v.shape[1:]), maxshape=(None, *v.shape[1:]), dtype=v.dtype,
                            chunks=(25, *v.shape[1:]) if c == "pixels" else True, **kw
                        )
                    D[c].resize(row + n, 0)
                    D[c][row:row + n] = v
                row += n
                lens.append(n)
                meta["src_ep"].append(int(src[k]))
                meta["n_targets"].append(ep["n_targets"])
                meta["max_rise"].append(float(ep["cube"][:, 2].max() - ep["cube"][0, 2]))
                if k % 100 == 0:
                    print(f"{k + 1}/{len(jobs)} episodes, {row} rows", flush=True)
        lens = np.array(lens, np.int64)
        h.create_dataset("ep_len", data=lens)
        h.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(lens)[:-1]]))
        for c, v in meta.items():
            h.create_dataset(f"noisy_{c}", data=np.asarray(v))
    print(f"DONE {args.out}: {len(lens)} episodes, {row} rows", flush=True)


if __name__ == "__main__":
    main()
