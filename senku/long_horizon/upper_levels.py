"""Stagewise upper levels following H-JEPA (arXiv 2610.06805).

Stacked encoders with one-step residual dynamics on frozen world-model features.
Each level encodes the states and the actions of the level below (stride 2) and predicts one step ahead.
Loss combines teacher-forced prediction, open-loop prediction, and SIGReg regularization.
Trained on expert data only; planning uses CEM anchored to training macro-actions.
Train: python -m senku.long_horizon.upper_levels --cache FEATS.pt --out UP.pt [--levels 3]
"""
import argparse
import time

import numpy as np
import torch
from torch import nn


UDIM, ZD, AD, RAW = 960, 192, 4, 5
EVERY = 5  # env steps per cache frame


def mlp(d_in, d_out, hidden=384, ln=False):
    layers = [nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)]
    return nn.Sequential(*layers, nn.LayerNorm(d_out)) if ln else nn.Sequential(*layers)


class Level(nn.Module):
    """Single upper-level layer: state encoder, action decoder, one-step residual predictor."""
    def __init__(self, d_in, a_in, hidden=384, pred_hidden=1024):
        super().__init__()
        self.enc = mlp(d_in, ZD, hidden)
        self.act = mlp(a_in, AD, 128, ln=True)
        self.pred = nn.Sequential(nn.Linear(ZD + AD, pred_hidden), nn.GELU(), nn.Linear(pred_hidden, pred_hidden),
                                  nn.GELU(), nn.Linear(pred_hidden, ZD))

    def step(self, z, a):
        return z + self.pred(torch.cat([z, a], -1))


class UpperLevels(nn.Module):
    """Stack of upper levels above the frozen world model. Level i has stride 2^(i+1) * EVERY env steps."""

    def __init__(self, n_levels=2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(UDIM))
        self.register_buffer("sd", torch.ones(UDIM))
        self.register_buffer("amean", torch.zeros(RAW))
        self.register_buffer("astd", torch.ones(RAW))
        lv = [Level(UDIM, 2 * EVERY * RAW)]
        for _ in range(n_levels - 1):
            lv.append(Level(ZD, 2 * AD))
        self.levels = nn.ModuleList(lv)
        for i in range(n_levels):
            for k in ("a_mean", "a_std", "a_lo", "a_hi"):
                self.register_buffer(f"{k}{i}", torch.zeros(AD))

    @property
    def n(self):
        return len(self.levels)

    def stride(self, level):
        """env steps per step of upper level `level` (0 = level 2)."""
        return 2 ** (level + 1) * EVERY

    def lift(self, u, top=None):
        """level-1 features u [N, UDIM] -> list of upper states [z2, z3, ...] (up to level index `top`)."""
        z = (u.float() - self.mu) / self.sd
        out = []
        for lv in self.levels[:(self.n if top is None else top + 1)]:
            z = lv.enc(z)
            out.append(z)
        return out

    def macro(self, raw):
        """raw env actions [N, T, 5] (T = 10 * 2^k) -> list of macro-actions per level [N, T / stride_l, AD]."""
        N, T, _ = raw.shape
        x = ((raw - self.amean) / self.astd).reshape(N, T // (2 * EVERY), 2 * EVERY * RAW)
        out = [self.levels[0].act(x)]
        for lv in self.levels[1:]:
            a = out[-1]
            out.append(lv.act(a.reshape(N, a.shape[1] // 2, 2 * AD)))
        return out

    def anchor(self, level):
        return (getattr(self, f"a_mean{level}"), getattr(self, f"a_std{level}"), getattr(self, f"a_lo{level}"),
                getattr(self, f"a_hi{level}"))


def windows(ep, st, held, need):
    """Cache rows where need consecutive frames fit in one episode."""
    starts = np.r_[0, np.nonzero(np.diff(ep))[0] + 1]
    ends = np.r_[starts[1:], len(ep)]
    J = [np.arange(a, b - need + 1) for a, b in zip(starts, ends) if b - a >= need]
    J = np.concatenate(J) if J else np.zeros(0, int)
    return J, held[J]


def losses(up, U, A, sigreg, lam=0.09):
    """Per-level: teacher-forced one-step, open-loop two-step, and SIGReg loss."""
    B, Fr, _ = U.shape
    out, stats = 0.0, {}
    zs = up.lift(U.reshape(B * Fr, -1))
    acts = up.macro(A)
    for i in range(up.n):
        s = 2 ** (i + 1)
        z = zs[i].view(B, Fr, ZD)[:, ::s][:, :4]
        a = acts[i][:, :3]
        lv = up.levels[i]
        p1 = lv.step(z[:, :3].reshape(-1, ZD), a.reshape(-1, AD)).view(B, 3, ZD)
        tf = ((p1 - z[:, 1:]) ** 2).sum(-1).mean()
        p2 = lv.step(p1[:, :2].reshape(-1, ZD), a[:, 1:].reshape(-1, AD)).view(B, 2, ZD)
        ol = ((p2 - z[:, 2:]) ** 2).sum(-1).mean()
        sr = sigreg(z.transpose(0, 1))
        out = out + tf + ol + lam * sr
        stats.update({f"L{i + 2}_tf": tf.item(), f"L{i + 2}_ol": ol.item(), f"L{i + 2}_sig": sr.item()})
    return out, stats


def load_upper(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    up = UpperLevels(ck["n_levels"]).to(device)
    up.load_state_dict(ck["state"])
    return up.eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Feature cache (X, ep, st, src, held)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-sources", default="cube_single_expert.h5:1.0:holdout",
                    help="Sources of the feature cache, in order")
    ap.add_argument("--levels", type=int, default=2, help="upper levels above the frozen WM (2 = levels 2 and 3)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.09)
    ap.add_argument("--sources", default="expert", help="expert = source 0 (expert data) only")
    ap.add_argument("--max-eps", type=int, default=None)
    a = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401

    from senku.data import action_scaler, parse_sources
    from senku.losses import SIGReg
    SRC = parse_sources(a.data_sources)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, ep, st, src, held = torch.load(a.cache, weights_only=False)
    assert a.sources == "expert", "action rows below are read from source 0 (the expert h5) only"
    keep = src == 0
    if a.max_eps:
        keep &= (ep % 100000) < a.max_eps
    X, ep, st, src, held = X[keep], ep[keep], st[keep], src[keep], held[keep]
    with h5py.File(SRC[0]["path"], "r") as f:
        off, act = f["ep_offset"][:], f["action"][:].astype(np.float32)
    up = UpperLevels(a.levels).to(dev)
    mean, std = action_scaler(SRC[0]["path"])
    up.amean.copy_(torch.as_tensor(mean))
    up.astd.copy_(torch.as_tensor(std))
    up.mu.copy_(X[:200000].float().mean(0).to(dev))
    up.sd.copy_(X[:200000].float().std(0).to(dev) + 1e-3)
    need = 3 * 2 ** a.levels + 1                                   # frames for 4 states at the top stride
    J, Jh = windows(ep, st, held, need)
    tr, va = J[~Jh], J[Jh]
    print(f"{len(tr)} train / {len(va)} held-out windows of {need} frames ({(need - 1) * EVERY} env steps)", flush=True)
    Xd = X.to(dev)
    A_rows = torch.as_tensor(off[(ep % 100000)] + st, device=dev)    # action row of each cache frame
    actd = torch.as_tensor(act, device=dev)
    T = (need - 1) * EVERY

    def batch(idx):
        j = torch.as_tensor(idx, device=dev)
        U = Xd[j[:, None] + torch.arange(need, device=dev)].float()
        r = A_rows[j][:, None] + torch.arange(T, device=dev)
        return U, torch.nan_to_num(actd[r])

    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(up.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.02)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        loss, stats = losses(up, *batch(rng.choice(tr, a.batch)), sigreg, a.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(up.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps:
            with torch.no_grad():
                _, vs = losses(up, *batch(rng.choice(va, min(4096, len(va)))), sigreg, a.lam)
            print(f"[UP {step:6d}/{a.steps}] " + " ".join(f"{k} {v:.4f}" for k, v in vs.items())
                  + f" {step / (time.time() - t0):.1f} st/s", flush=True)
    with torch.no_grad():
        acts = [[] for _ in range(up.n)]
        for b in range(0, min(len(tr), 200000), 4096):
            _, A = batch(tr[b:b + 4096])
            for i, m in enumerate(up.macro(A)):
                acts[i].append(m.reshape(-1, AD))
        for i in range(up.n):
            m = torch.cat(acts[i])
            getattr(up, f"a_mean{i}").copy_(m.mean(0))
            getattr(up, f"a_std{i}").copy_(m.std(0))
            getattr(up, f"a_lo{i}").copy_(torch.quantile(m[:100000], 0.02, 0))
            getattr(up, f"a_hi{i}").copy_(torch.quantile(m[:100000], 0.98, 0))
    torch.save({"state": up.state_dict(), "n_levels": up.n, "cache": a.cache}, a.out)
    print("UP DONE", flush=True)


if __name__ == "__main__":
    main()


class UpperChain:
    """Top-down planning with the upper levels. CEM over top-level macro-actions to the goal, anchored to training
    macro-actions. Returns the first predicted top-level state as the level-1 planner's subgoal."""

    def __init__(self, m, up, horizon, samples=300, iters=30, topk=30, seed=0):
        self.m, self.up, self.H = m, up, horizon
        self.samples, self.iters, self.topk = samples, iters, topk
        self.top = up.n - 1
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.goal = {}
        self.log = []

    @torch.no_grad()
    def feats(self, x):
        from senku.long_horizon.td_head import td_feats
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g, p = self.m.encode(x.float())
            return td_feats(g, self.m.state_tokens(p)).float()

    @torch.no_grad()
    def plan(self, u_now, u_goal, remaining):
        L, up = self.top, self.up
        S = up.stride(L)
        H = max(1, -(-int(remaining) // S))
        mean, std, lo, hi = up.anchor(L)
        z0, zg = up.lift(u_now)[L], up.lift(u_goal)[L]
        mu = mean.expand(H, -1).clone()
        sd = std.expand(H, -1).clone()
        for _ in range(self.iters):
            x = torch.randn(self.samples, H, AD, generator=self.gen, device="cuda") * sd + mu
            x = torch.maximum(torch.minimum(x, hi), lo)
            x[0] = mu
            z = z0.expand(self.samples, -1)
            first = None
            for h in range(H):
                z = up.levels[L].step(z, x[:, h])
                first = z if first is None else first
            c = ((z - zg) ** 2).sum(-1)
            idx = torch.topk(c, self.topk, largest=False).indices
            mu, sd = x[idx].mean(0), x[idx].std(0) + 1e-4
        self.log.append(dict(H=H, best=float(c[idx[0]]), now=float(((z0 - zg) ** 2).sum())))
        return first[idx[0]][None]

    def step(self, i, x_hist, goal_x, t):
        if i not in self.goal:
            self.goal[i] = self.feats(goal_x[None])
        return self.plan(self.feats(x_hist[-1:]), self.goal[i], self.H - t)
