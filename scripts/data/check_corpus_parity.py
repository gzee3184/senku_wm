"""Verify training corpus matches evaluation rendering and has no visual artifacts.

Checks metadata, simulator state, pixel parity with the evaluation environment, goal-marker visibility (Cube),
and generates a visual montage for inspection. Exit code 0 = PASS, 1 = FAIL.

Usage: python scripts/data/check_corpus_parity.py --corpus PATH --env cube [--reference PATH] [--out DIR]
"""
import argparse
import os
import sys
from pathlib import Path

from senku.paths import CUBE, PUSHT, SCENE, data  # noqa: E402

import numpy as np

REF = {"cube": str(data(CUBE)), "pusht": str(data(PUSHT)), "scene": str(data(SCENE))}
REQUIRED = ["generator", "git_commit", "env_flags", "holdout_rule", "planner_data"]


def marker_px(img):
    from scipy import ndimage
    x = img.astype(int)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    bright = (r > 150) & (g < 90) & (b < 90)
    dim = (r > g + 12) & (r > b + 2) & (r < 130) & (r > 55)
    return int((dim & ~ndimage.binary_dilation(bright, iterations=6)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--env", choices=["cube", "pusht", "scene"], required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default="corpus_checks")
    args = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401
    os.environ.setdefault("MUJOCO_GL", "egl")
    ref = args.reference or REF[args.env]
    rng = np.random.default_rng(0)
    fails, notes = [], []
    f = h5py.File(args.corpus, "r")
    # 1. metadata
    req = REQUIRED + (["goal_marker_visible"] if args.env == "cube" else [])
    miss = [k for k in req if k not in f.attrs]
    if miss:
        fails.append(f"metadata missing {miss}")
    if "planner_data" in f.attrs and bool(f.attrs["planner_data"]):
        fails.append("planner_data=True: the WM is SSL-only, never train on planner / on-policy episodes")
    if args.env == "cube" and "goal_marker_visible" in f.attrs and bool(f.attrs["goal_marker_visible"]):
        fails.append("goal_marker_visible=True: evaluation frames never show the marker")
    # 2. state
    skeys = {"cube": ["qpos", "qvel"], "pusht": ["state"], "scene": []}[args.env]
    has_state = all(k in f for k in skeys) and bool(skeys)
    if skeys and not has_state:
        fails.append(f"no stored state {skeys}: cannot re-render or audit")
    n = len(f["pixels"])
    rows = np.sort(rng.choice(n, size=min(32, n), replace=False))
    # 3. re-render parity
    if has_state:
        if args.env == "cube":
            from senku.envs.cube import make_cube_env as _make_env
            env = _make_env()
            diffs = []
            for r in rows:
                env.set_state(f["qpos"][r], f["qvel"][r])
                diffs.append(np.abs(env.render().astype(int) - f["pixels"][r].astype(int)).mean())
        else:
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            env = gym.make("swm/PushT-v1")
            env.reset(seed=0)
            diffs = []
            for r in rows:
                env.unwrapped._set_state(np.asarray(f["state"][r], np.float64))
                diffs.append(np.abs(env.render().astype(int) - f["pixels"][r].astype(int)).mean())
        diffs = np.array(diffs)
        notes.append(f"re-render |diff| mean {diffs.mean():.3f}, max row {diffs.max():.3f} (/255)")
        if diffs.max() >= 1.0:
            fails.append(
                f"re-render parity: max row diff {diffs.max():.2f}/255 >= 1.0 (rendering differs from evaluation)"
            )
    # 4. cube goal marker
    if args.env == "cube" and "ep_offset" in f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        eps = rng.choice(len(ln), size=min(40, len(ln)), replace=False)
        med = np.array([np.median([marker_px(f["pixels"][int(off[e]) + int(k)])
                                   for k in np.linspace(0, ln[e] - 1, 8).astype(int)]) for e in eps])
        share = float(np.mean(med > 100))
        notes.append(
            f"goal-marker detector: {share:.0%} of {len(eps)} episodes flagged "
            f"(expert <1%, marked corpora >90%)"
        )
        if share > 0.10:
            fails.append(f"goal marker visible in {share:.0%} of episodes")
    # 5. montage
    import imageio
    Path(args.out).mkdir(parents=True, exist_ok=True)
    g = h5py.File(ref, "r")
    rr = np.sort(rng.choice(len(g["pixels"]), size=6, replace=False))
    top = np.concatenate([f["pixels"][r] for r in rows[:6]], 1)
    bot = np.concatenate([g["pixels"][r] for r in rr], 1)
    png = Path(args.out) / f"{Path(args.corpus).stem}_vs_reference.png"
    imageio.imwrite(png, np.concatenate([top, bot], 0)[::2, ::2])
    notes.append(f"montage (top corpus, bottom reference) -> {png}: LOOK at it")
    print(f"corpus {args.corpus}\nreference {ref}")
    for s in notes:
        print("  note:", s)
    for s in fails:
        print("  FAIL:", s)
    print("VERDICT:", "FAIL" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
