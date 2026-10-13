"""Paired comparison of two methods on the same starts: exact McNemar test and a bootstrap 95% interval.

Each method is a list of result JSONs (one per seed) from eval.py, eval_antmaze.py or eval_antmaze_short.py. Files are
paired by seed (the file's seed field, or an `s<seed>.json` name); the starts of each pair must be the same.

    python scripts/paired_stats.py --a 'runs/senku/ho10_s*.json' --b 'runs/lewm/ho10_s*.json'
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.stats import binomtest


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"(?:^|_)s(\d+)\.json$", os.path.basename(f))
        d = json.load(open(f))
        succ = d.get("episode_successes", d.get("successes"))
        starts = list(zip(d.get("episodes", []), d.get("start_steps", []))) or d.get("windows") or None
        seed = d.get("seed", d.get("args", {}).get("seed"))
        out[seed if seed is not None else int(m.group(1)) if m else f] = (np.asarray(succ, bool), starts)
    return out


def compare(a, b, n_boot=10000, seed=0):
    seeds = sorted(set(a) & set(b), key=str)
    assert seeds, "no common seeds"
    sa, sb = [], []
    for s in seeds:
        (xa, ka), (xb, kb) = a[s], b[s]
        assert len(xa) == len(xb) and (ka is None or kb is None or ka == kb), f"seed {s}: starts differ"
        sa.append(xa)
        sb.append(xb)
    xa, xb = np.concatenate(sa), np.concatenate(sb)
    only_a, only_b = int((xa & ~xb).sum()), int((~xa & xb).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xa), (n_boot, len(xa)))
    diff = (xa[idx].mean(1) - xb[idx].mean(1)) * 100
    return dict(seeds=seeds, n=len(xa), a=100 * xa.mean(), b=100 * xb.mean(), only_a=only_a, only_b=only_b, p=p,
                ci=(float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob of method A's JSONs")
    ap.add_argument("--b", required=True, help="glob of method B's JSONs")
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    r = compare(load(args.a), load(args.b))
    print(f"{args.name}: n={r['n']} seeds={r['seeds']} A={r['a']:.1f} B={r['b']:.1f} diff={r['a'] - r['b']:+.1f} "
          f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] only A={r['only_a']} only B={r['only_b']} McNemar p={r['p']:.3g}")


if __name__ == "__main__":
    main()
