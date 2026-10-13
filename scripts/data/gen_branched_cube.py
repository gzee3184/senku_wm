"""Counterfactual Cube corpus from branched expert episodes.

Generates self-supervised training data by re-simulating expert episodes with variant actions after a branch point.
Each start is split into (expert prefix + variant branch) trajectories. Pick variants close earlier/later or offset
the arm sideways. Carry variants release the cube at different times. Variants are rendered without goal markers.

Usage: python scripts/data/gen_branched_cube.py --out PATH --n-pick 600 --n-carry 600 --workers 12 [--seed 0]
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
GRIP, PREFIX, L = 4, 15, 40
CM = 0.45  # ~1 cm per step in normalized action units
_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from senku.envs.cube import make_cube_env as _make_env
        _ENV = _make_env()
        assert not _ENV._visualize_info, "goal marker must stay hidden (expert rendering)"
    return _ENV


def _onset(grip, close):
    hit = np.nonzero(grip > 0.2)[0] if close else np.nonzero(grip < -0.1)[0]
    return int(hit[0]) if len(hit) else -1


def variants(kind, exp, b, rng):
    """exp: expert raw actions from the branch point on [L, 5]; b = steps from branch point to the expert's switch."""
    out = [exp.copy()]
    if kind == 0:
        k = int(rng.integers(4, 16))
        v = exp.copy()
        v[:, GRIP] = exp[np.minimum(np.arange(L) + k, L - 1), GRIP]
        out.append(v)
        k = int(rng.integers(3, 11))
        v = exp.copy()
        v[:, GRIP] = exp[np.maximum(np.arange(L) - k, 0), GRIP]
        out.append(v)
        v = exp.copy()
        d = rng.standard_normal(2)
        d /= np.linalg.norm(d)
        lo = max(0, b - 10)
        v[lo:b, :2] += d * 0.45 * rng.uniform(1.5, 5.0) / max(1, b - lo)
        out.append(v)
    elif kind == 2:
        from scipy.ndimage import gaussian_filter1d
        d = rng.standard_normal(2)
        d /= np.linalg.norm(d)
        m = int(rng.integers(5, 16))
        v = exp.copy()
        v[:m, :2] += d * CM * rng.uniform(3.0, 8.0) / m
        out.append(v)
        m = int(rng.integers(5, 16))
        v = exp.copy()
        v[:m, 2] += rng.choice([-1, 1]) * CM * rng.uniform(3.0, 8.0) / m
        out.append(v)
        v = exp.copy()
        v[:, :4] += gaussian_filter1d(rng.standard_normal((L, 4)) * 0.3, sigma=2, axis=0)
        out.append(v)
    else:
        v = exp.copy()
        t0, m = int(rng.integers(0, 20)), int(rng.integers(2, 9))
        v[t0:t0 + m, GRIP] = -1.0
        out.append(v)
        k = int(rng.integers(8, 26))
        v = exp.copy()
        v[:, GRIP] = exp[np.minimum(np.arange(L) + k, L - 1), GRIP]
        out.append(v)
    return np.clip(np.stack(out), -1, 1)


def _job(job):
    """One start: re-simulate the prefix, then each branch; render every step."""
    from senku.envs.cube import read_env_state
    env = _env()
    qpos, qvel, prefix_act, bank = job
    eps = []
    for seq in bank:
        env.set_state(qpos, qvel)
        acts = np.concatenate([prefix_act, seq]).astype(np.float32)
        rec = {k: [] for k in ("pixels", "qpos", "qvel", "cube", "quat", "ee", "closure")}

        def snap():
            st = read_env_state(env)
            d = env.unwrapped.data
            rec["pixels"].append(env.render())
            rec["qpos"].append(d.qpos.copy())
            rec["qvel"].append(d.qvel.copy())
            rec["cube"].append(st["cube"])
            rec["quat"].append(d.joint("object_joint_0").qpos[3:7].copy())
            rec["ee"].append(st["ee"])
            rec["closure"].append(st["closure"])
        snap()
        for a in acts:
            env.step(a)
            snap()
        r = {k: np.stack(v) if k != "closure" else np.array(v) for k, v in rec.items()}
        r["action"] = np.concatenate(
            [acts, np.full((1, acts.shape[1]), np.nan, np.float32)]
        )  # terminal NaN, as expert
        eps.append(r)
    return eps


def pick_starts(f, n, kind, rng, heldout=False):
    ep_len, off = f["ep_len"][:], f["ep_offset"][:]
    cube = f["privileged_block_0_pos"][:]
    act = f["action"]
    train_eps = np.nonzero((np.arange(len(ep_len)) % 10 == 0) if heldout else (np.arange(len(ep_len)) % 10 != 0))[0]
    out = []
    tries = 0
    while len(out) < n:
        tries += 1
        assert tries < 200 * n, f"start sampling stuck (kind {kind}, {len(out)}/{n})"
        e = int(rng.choice(train_eps))
        o, ln = int(off[e]), int(ep_len[e])
        a = np.asarray(act[o:o + ln - 1])
        g = a[:, GRIP]
        closes = np.nonzero((g[1:] > 0.2) & (g[:-1] <= 0.2))[0] + 1                # every close command onset
        if g[0] > 0.2:
            closes = np.r_[0, closes]
        if kind == 0:
            closes = closes[closes >= PREFIX + 3]
            if not len(closes):
                continue
            on = int(rng.choice(closes))
            bp = on - int(rng.integers(3, min(20, on - PREFIX) + 1))
        elif kind == 2:
            closes = closes[closes >= PREFIX + 15]
            if not len(closes):
                continue
            on = int(rng.choice(closes))
            bp = on - int(rng.integers(15, min(40, on - PREFIX) + 1))
        else:
            if not len(closes):
                continue
            held = np.nonzero((np.arange(ln - 1) > closes[0] + 5) & (cube[o:o + ln - 1, 2] > 0.04) & (g > 0.2))[0]
            if len(held) < 5:
                continue
            bp = int(rng.choice(held[:max(1, len(held) - 5)]))
        if bp < PREFIX or bp + L >= ln - 1:
            continue
        exp = a[bp:bp + L]
        if np.isnan(exp).any():
            continue
        sw = _onset(exp[:, GRIP], kind != 1)
        b = sw if sw >= 0 else L // 2
        out.append((kind, e, o + bp - PREFIX, a[bp - PREFIX:bp], variants(kind, exp, b, rng)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pick", type=int, default=600)
    ap.add_argument("--n-carry", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-approach", type=int, default=0)
    ap.add_argument("--heldout-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import h5py
    import hdf5plugin
    rng = np.random.default_rng(args.seed)
    with h5py.File(EXPERT, "r") as f:
        ho = args.heldout_probe
        starts = (pick_starts(f, args.n_pick, 0, rng, ho) + pick_starts(f, args.n_carry, 1, rng, ho)
                  + pick_starts(f, args.n_approach, 2, rng, ho))
        jobs = [(f["qpos"][r], f["qvel"][r], pre, bank) for (_, _, r, pre, bank) in starts]
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()
    comp = dict(**hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
    with h5py.File(args.out, "w") as h:
        h.attrs.update(dict(
            generator="scripts/data/gen_branched_cube.py",
            git_commit=commit,
            created=datetime.datetime.now().isoformat(),
            env="swm/OGBCube-v0 via senku.envs.cube.make_cube_env",
            env_flags=(
                "env_type=single ob_type=states multiview=False width=224 height=224 visualize_info=False "
                "terminate_at_goal=False"
            ),
            goal_marker_visible=False,
            render_reference=EXPERT,
            holdout_rule=(
                "HELD-OUT PROBE SET: source episodes ep % 10 == 0 only, NEVER train on this file"
                if args.heldout_probe
                else "source expert episodes ep % 10 != 0 only (probes/evals use == 0)"
            ),
            planner_data=False,
            prefix=PREFIX,
            branch_len=L,
            seed=args.seed,
            variants=(
                "pick: 0 expert, 1 close earlier 4-15, 2 close later 3-10, 3 sideways 1.5-5 cm then close; "
                "carry: 0 expert, 1 open 2-8 steps then expert gripper, 2 open earlier 8-25; "
                "approach: 0 expert, 1 lateral 3-8 cm, 2 height 3-8 cm, 3 smoothed xyz/yaw noise"
            )
        ))
        D = {}
        meta = {k: [] for k in ("pair_id", "kind", "variant", "src_ep", "src_row", "max_rise", "final_cube")}
        lens, row = [], 0
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for pid, eps in enumerate(pool.imap(_job, jobs, chunksize=1)):
                kind, e, r, _, _ = starts[pid]
                for vi, ep in enumerate(eps):
                    n = len(ep["pixels"])
                    cols = {"pixels": ep["pixels"], "action": ep["action"], "qpos": ep["qpos"], "qvel": ep["qvel"],
                            "privileged_block_0_pos": ep["cube"], "privileged_block_0_quat": ep["quat"],
                            "proprio_effector_pos": ep["ee"], "gripper_closure": ep["closure"][:, None],
                            "ep_idx": np.full(n, len(lens), np.int32), "step_idx": np.arange(n, dtype=np.int64)}
                    for k, v in cols.items():
                        if k not in D:
                            kw = comp if k == "pixels" else {}
                            D[k] = h.create_dataset(
                                k, shape=(0, *v.shape[1:]), maxshape=(None, *v.shape[1:]),
                                dtype=v.dtype,
                                chunks=(25, *v.shape[1:]) if k == "pixels" else True, **kw
                            )
                        D[k].resize(row + n, 0)
                        D[k][row:row + n] = v
                    row += n
                    lens.append(n)
                    meta["pair_id"].append(pid)
                    meta["kind"].append(kind)
                    meta["variant"].append(vi)
                    meta["src_ep"].append(e)
                    meta["src_row"].append(r)
                    meta["max_rise"].append(float(ep["cube"][PREFIX:, 2].max() - ep["cube"][PREFIX, 2]))
                    meta["final_cube"].append(ep["cube"][-1])
                if pid % 50 == 0:
                    print(f"{pid + 1}/{len(jobs)} starts, {row} rows", flush=True)
        lens = np.array(lens, np.int64)
        h.create_dataset("ep_len", data=lens)
        h.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(lens)[:-1]]))
        for k, v in meta.items():
            h.create_dataset(f"branch_{k}", data=np.asarray(v))
    print(f"DONE {args.out}: {len(lens)} episodes, {row} rows", flush=True)


if __name__ == "__main__":
    main()
