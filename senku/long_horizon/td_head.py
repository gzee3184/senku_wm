"""Steps-apart head: predicts env steps between two frames of the same episode.
Frozen world model latents (g + state tokens pooled to 2x2 grid) are input.
Self-supervised; no rewards or actions needed.
Train: python -m senku.long_horizon.td_head --ckpt WM.pt --out TD.pt
"""
import argparse
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from senku.data import parse_sources
from senku.model import load_model

EVERY, DMAX = 5, 1000


def grid_pool(s, k):
    """Adaptive-average-pool a square token grid to k x k cells. Input [N, T, D], output [N, k*k, D]."""
    n, t, d = s.shape
    side = int(round(t ** 0.5))
    return F.adaptive_avg_pool2d(s.transpose(1, 2).reshape(n, d, side, side), k).flatten(2).transpose(1, 2)


def td_feats(g, s):
    """Concatenate g and state tokens pooled to 2x2 grid. Input g [N, D], s [N, T, D]; output [N, 5D]."""
    return torch.cat([g.float(), grid_pool(s.float(), 2).flatten(1)], 1)


class TDHead(nn.Module):
    def __init__(self, d_in, hidden=1024):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.net = nn.Sequential(nn.Linear(3 * d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, y):
        """Normalized log-steps from x to y (0 = same frame, 1 = DMAX steps)."""
        x, y = (x - self.mu) / self.sd, (y - self.mu) / self.sd
        return self.net(torch.cat([x, y, y - x], -1)).squeeze(-1)


def load_td(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    h = TDHead(ck["d_in"]).to(device)
    h.load_state_dict(ck["head"])
    return h.eval().requires_grad_(False)


@torch.no_grad()
def encode_sources(wm, sources, holdout_mod=10, max_eps=None):
    from senku.utils import to_x
    feats, eps, steps, src, held = [], [], [], [], []
    for si, s in enumerate(sources):
        with h5py.File(s["path"], "r") as f:
            off, ln = f["ep_offset"][:], f["ep_len"][:]
            for e in range(len(ln) if max_eps is None else max_eps):
                rows = off[e] + np.arange(0, ln[e] - 1, EVERY)
                px = f["pixels"][rows[0]:rows[-1] + 1][::EVERY]
                for i in range(0, len(px), 256):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        g, p = wm.encode(to_x(px[i:i + 256]).float())
                        feats.append(td_feats(g, wm.state_tokens(p)).half().cpu())
                eps.append(np.full(len(rows), si * 100000 + e))
                steps.append(np.arange(len(rows)) * EVERY)
                src.append(np.full(len(rows), si))
                held.append(np.full(len(rows), bool(s.get("holdout")) and e % holdout_mod == 0))
                if e % 100 == 0:
                    print(f"encoded source {si} ep {e}/{len(ln)}", flush=True)
    return (torch.cat(feats), np.concatenate(eps), np.concatenate(steps), np.concatenate(src),
            np.concatenate(held))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="feature cache .pt (written if missing)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sources", default="cube_single_expert.h5:1.0:holdout",
                    help="the world model's training mix, as train.py --sources")
    ap.add_argument("--max-eps", type=int, default=None, help="smoke test: episodes per source")
    a = ap.parse_args()
    dev = "cuda"
    import os
    sources = parse_sources(a.sources)
    if a.cache and os.path.exists(a.cache):
        X, ep, st, src, held = torch.load(a.cache, weights_only=False)
    else:
        X, ep, st, src, held = encode_sources(load_model(a.ckpt, dev), sources, max_eps=a.max_eps)
        if a.cache:
            torch.save((X, ep, st, src, held), a.cache)
    X = X.to(dev)
    print(f"{len(X)} frames, {held.sum()} held out", flush=True)
    # per-episode index ranges (frames are stored contiguously per episode)
    starts = np.r_[0, np.nonzero(np.diff(ep))[0] + 1]
    ends = np.r_[starts[1:], len(ep)]
    ep_held = held[starts]
    w = np.array([sources[int(src[s0])]["weight"] for s0 in starts], float)
    n_src = np.bincount(src[starts], minlength=len(sources))
    w = w / n_src[src[starts]]
    tr, va = np.nonzero(~ep_held)[0], np.nonzero(ep_held)[0]
    rng = np.random.default_rng(0)

    def pairs(eps_pool, n, wts=None):
        e = rng.choice(eps_pool, n, p=None if wts is None else wts / wts.sum())
        L = ends[e] - starts[e]
        dmax = np.minimum(DMAX // EVERY, L - 1)
        k = np.where(rng.random(n) < 0.05, 0,
                     np.minimum(np.exp(rng.uniform(0, np.log(dmax + 1), n)).astype(int), dmax))
        i = (rng.random(n) * (L - k)).astype(int)
        return starts[e] + i, starts[e] + i + k, k * EVERY

    head = TDHead(X.shape[1]).to(dev)
    head.mu.copy_(X.float().mean(0))
    head.sd.copy_(X.float().std(0) + 1e-3)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    norm = np.log1p(DMAX)
    vi, vj, vd = pairs(va, 20000)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        i, j, d = pairs(tr, a.batch, w[tr])
        y = torch.as_tensor(np.log1p(d) / norm, dtype=torch.float32, device=dev)
        loss = F.smooth_l1_loss(head(X[i].float(), X[j].float()), y, beta=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps:
            with torch.no_grad():
                pv = torch.cat([head(X[vi[k:k + 4096]].float(), X[vj[k:k + 4096]].float())
                                for k in range(0, len(vi), 4096)]).cpu().numpy()
            from scipy.stats import spearmanr
            rho = spearmanr(pv, vd).correlation
            mae = np.abs(np.expm1(pv * norm) - vd).mean()
            print(f"[TD {step:6d}/{a.steps}] loss {loss.item():.4f} held-out spearman {rho:.3f} mae {mae:.1f} steps "
                  f"{step / (time.time() - t0):.1f} st/s", flush=True)
    torch.save({"head": head.state_dict(), "d_in": X.shape[1], "wm_ckpt": a.ckpt}, a.out)
    print("TD DONE", flush=True)


if __name__ == "__main__":
    main()
