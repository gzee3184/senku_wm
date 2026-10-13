"""Goal-blind action repertoire for long-horizon planning.

Proposes 25-step action chunks from frozen world-model features of the current state,
using the patterns in the training corpus. Never sees the goal or scores proposals.
Two variants: CVAERep (continuous codes with learned prior) or MotifRep (VQ library
with prior classifier). Train: python -m senku.long_horizon.repertoire --variant cvae|motif
--cache FEATS.pt --out REP.pt
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from senku.long_horizon.td_head import EVERY, td_feats

CH, GDIM, ADIM = 25, 192, 5
ZDIM = 5 * GDIM + 2 * GDIM


def rep_input(g_hist, s):
    """[N, 7*D]: current-frame features, plus global features from 5 and 10 steps earlier."""
    return torch.cat([td_feats(g_hist[:, -1], s), g_hist[:, -2].float(), g_hist[:, -3].float()], 1)


def rep_input_cache(X, j, j1, j2):
    """Extract rep_input from cached features at rows j, j1 (t-5), j2 (t-10)."""
    return torch.cat([X[j].float(), X[j1, :GDIM].float(), X[j2, :GDIM].float()], 1)


def chunk_table(ep, st, src, held, actions_by_src, mean, std):
    """Cache rows with CH action steps that fit in one episode, and no NaN action."""
    starts = np.r_[0, np.nonzero(np.diff(ep))[0] + 1]
    ends = np.r_[starts[1:], len(ep)]
    J, J1, J2, C, H, S = [], [], [], [], [], []
    for a, b in zip(starts, ends):
        si, e = int(src[a]), int(ep[a] % 100000)
        off, ln, A = actions_by_src[si]
        o, L = int(off[e]), int(ln[e])
        for j in range(a, b):
            t = int(st[j])
            assert t == (j - a) * EVERY
            if t + CH > L - 1:
                break
            c = A[o + t:o + t + CH]
            if np.isnan(c).any():
                continue
            J.append(j)
            J1.append(max(j - 1, a))
            J2.append(max(j - 2, a))
            C.append((c - mean) / std)
            H.append(bool(held[j]))
            S.append(si)
    return dict(j=np.array(J), j1=np.array(J1), j2=np.array(J2), chunk=np.stack(C).astype(np.float16),
                held=np.array(H), src=np.array(S))


def mlp(d_in, d_out, hidden, depth=3):
    """Multi-layer perceptron: (depth - 1) hidden layers, then output."""
    layers, d = [], d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.zeros(ZDIM))
        self.register_buffer("sd", torch.ones(ZDIM))

    def set_norm(self, z):
        self.mu.copy_(z.float().mean(0))
        self.sd.copy_(z.float().std(0) + 1e-3)

    def nz(self, z):
        return (z.float() - self.mu) / self.sd


class CVAERep(_Base):
    """Continuous action codes with learned state-conditioned Gaussian prior."""
    kind = "cvae"

    def __init__(self, code=12, hidden=1024, beta=1.0):
        super().__init__()
        self.code, self.hidden, self.beta = code, hidden, beta
        self.enc = mlp(ZDIM + CH * ADIM, 2 * code, hidden)
        self.pri = mlp(ZDIM, 2 * code, hidden)
        self.dec = mlp(ZDIM + code, CH * ADIM, hidden)

    def prior(self, z):
        mu, ls = self.pri(self.nz(z)).chunk(2, -1)
        return mu, ls.clamp(-5, 2)

    def decode(self, z, code):
        return self.dec(torch.cat([self.nz(z), code.float()], -1)).view(-1, CH, ADIM)

    def loss(self, z, chunk):
        zn = self.nz(z)
        qm, qls = self.enc(torch.cat([zn, chunk.flatten(1).float()], -1)).chunk(2, -1)
        qls = qls.clamp(-5, 2)
        c = qm + qls.exp() * torch.randn_like(qm)
        rec = 0.5 * ((self.decode(z, c) - chunk.float()) ** 2).sum((1, 2))
        pm, pls = self.prior(z)
        kl = (pls - qls + (qls.exp() ** 2 + (qm - pm) ** 2) / (2 * pls.exp() ** 2) - 0.5).sum(-1)
        loss = (rec + self.beta * kl).mean()
        return loss, dict(rec=rec.mean().item() / (CH * ADIM), kl=kl.mean().item())

    def propose(self, z, n, gen=None):
        pm, pls = self.prior(z)
        eps = torch.randn(len(z), n, self.code, generator=gen, device="cpu").to(z.device).clamp(-2.5, 2.5)
        codes = pm[:, None] + pls.exp()[:, None] * eps
        ch = self.decode(z.repeat_interleave(n, 0), codes.flatten(0, 1)).view(len(z), n, CH, ADIM)
        return codes, ch

    def ckpt(self):
        return dict(kind=self.kind, kw=dict(code=self.code, hidden=self.hidden, beta=self.beta),
                    state=self.state_dict())


class MotifRep(_Base):
    """Vector-quantized motif library with prior classifier."""
    kind = "motif"

    def __init__(self, k=64, hidden=1024, edim=64, commit=0.25):
        super().__init__()
        self.k, self.hidden, self.edim, self.commit = k, hidden, edim, commit
        self.enc = mlp(ZDIM + CH * ADIM, edim, hidden)
        self.book = nn.Parameter(torch.randn(k, edim) * 0.1)
        self.pri = mlp(ZDIM, k, hidden)
        self.dec = mlp(ZDIM + edim, CH * ADIM, hidden)
        self.register_buffer("usage", torch.zeros(k))

    def prior_logits(self, z):
        return self.pri(self.nz(z))

    def _e(self, z, chunk):
        return self.enc(torch.cat([self.nz(z), chunk.flatten(1).float()], -1))

    def assign(self, z, chunk):
        return torch.cdist(self._e(z, chunk), self.book).argmin(-1)

    def decode(self, z, code):
        return self.dec(torch.cat([self.nz(z), self.book[code]], -1)).view(-1, CH, ADIM)

    def loss(self, z, chunk):
        e = self._e(z, chunk)
        k = torch.cdist(e, self.book).argmin(-1)
        q = self.book[k]
        qst = e + (q - e).detach()  # straight-through
        out = self.dec(torch.cat([self.nz(z), qst], -1)).view(-1, CH, ADIM)
        rec = 0.5 * ((out - chunk.float()) ** 2).sum((1, 2))
        vq = ((q - e.detach()) ** 2).sum(-1) + self.commit * ((e - q.detach()) ** 2).sum(-1)
        ce = F.cross_entropy(self.prior_logits(z), k)
        self.usage.mul_(0.99).add_(torch.bincount(k, minlength=self.k).float() * 0.01)
        loss = rec.mean() + vq.mean() + ce
        return loss, dict(rec=rec.mean().item() / (CH * ADIM), ce=ce.item(),
                          used=int((self.usage > 1e-3).sum()))

    @torch.no_grad()
    def reset_dead(self, z, chunk):
        dead = torch.nonzero(self.usage < 1e-3).flatten()
        if len(dead):
            e = self._e(z, chunk)
            self.book.data[dead] = e[torch.randint(0, len(e), (len(dead),))]
            self.usage[dead] = 1.0

    def propose(self, z, n, gen=None):
        codes = torch.topk(self.prior_logits(z), n, -1).indices                 # [N, n]
        ch = self.decode(z.repeat_interleave(n, 0), codes.flatten()).view(len(z), n, CH, ADIM)
        return codes, ch

    def ckpt(self):
        return dict(kind=self.kind, kw=dict(k=self.k, hidden=self.hidden, edim=self.edim, commit=self.commit),
                    state=self.state_dict())


def load_rep(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    rep = (CVAERep if ck["kind"] == "cvae" else MotifRep)(**ck["kw"]).to(device)
    rep.load_state_dict(ck["state"])
    rep.bank = ck.get("bank")
    if rep.bank is not None:
        rep.bank = rep.bank.to(device)
    return rep.eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cvae", "motif"], required=True)
    ap.add_argument("--cache", required=True, help="Feature cache (X, ep, st, src, held)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-sources", default="cube_single_expert.h5:1.0:holdout",
                    help="Sources of the feature cache, in order")
    ap.add_argument("--wm-ckpt", default=None)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--code", type=int, default=12)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-eps", type=int, default=None, help="Keep first N episodes per source")
    a = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401

    from senku.data import action_scaler, parse_sources
    SRC = parse_sources(a.data_sources)
    dev = "cuda"
    X, ep, st, src, held = torch.load(a.cache, weights_only=False)
    if a.max_eps:
        keep = (ep % 100000) < a.max_eps
        X, ep, st, src, held = X[keep], ep[keep], st[keep], src[keep], held[keep]
    mean, std = action_scaler(SRC[0]["path"])
    acts = []
    for s in SRC:
        with h5py.File(s["path"], "r") as f:
            acts.append((f["ep_offset"][:], f["ep_len"][:], f["action"][:].astype(np.float32)))
    tab = chunk_table(ep, st, src, held, acts, mean, std)
    X = X.to(dev)
    J, J1, J2 = (torch.as_tensor(tab[k], device=dev) for k in ("j", "j1", "j2"))
    CHK = torch.as_tensor(tab["chunk"], device=dev)
    tr, va = np.nonzero(~tab["held"])[0], np.nonzero(tab["held"])[0]
    w = np.array([SRC[s]["weight"] for s in tab["src"][tr]])
    w = w / np.bincount(tab["src"][tr], minlength=len(SRC))[tab["src"][tr]]
    w = w / w.sum()
    print(f"{len(tr)} train chunks, {len(va)} held-out; per-source {np.bincount(tab['src'], minlength=len(SRC))}",
          flush=True)
    z_of = lambda idx: rep_input_cache(X, J[idx], J1[idx], J2[idx])  # noqa: E731
    rep = (CVAERep(code=a.code, beta=a.beta) if a.variant == "cvae" else MotifRep(k=a.k)).to(dev)
    rng = np.random.default_rng(0)
    rep.set_norm(z_of(torch.as_tensor(rng.choice(tr, 50000), device=dev)))
    opt = torch.optim.AdamW(rep.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    vidx = torch.as_tensor(rng.choice(va, min(4096, len(va)), replace=False), device=dev)
    t0 = time.time()
    stats = {}
    for step in range(1, a.steps + 1):
        idx = torch.as_tensor(rng.choice(tr, a.batch, p=w), device=dev)
        loss, _ = rep.loss(z_of(idx), CHK[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if rep.kind == "motif" and step % 1000 == 0 and step < 0.8 * a.steps:
            rep.reset_dead(z_of(idx), CHK[idx])
        if step % 2000 == 0 or step == a.steps:
            stats = evaluate(rep, z_of(vidx), CHK[vidx])
            print(f"[REP {a.variant} {step:6d}/{a.steps}] loss {loss.item():.3f} held-out "
                  + " ".join(f"{k} {v:.4f}" for k, v in stats.items())
                  + f" {step / (time.time() - t0):.1f} st/s", flush=True)
    ck = rep.ckpt()
    ck.update(bank=z_of(vidx[:1024]).cpu(), wm_ckpt=a.wm_ckpt, val=stats, cache=a.cache)
    torch.save(ck, a.out)
    print("REP DONE", flush=True)


@torch.no_grad()
def evaluate(rep, z, chunk):
    """Held-out metrics: reconstruction, best-of-16 and mean proposal error, random-state error, code usage."""
    rep.eval()
    _, st = rep.loss(z, chunk)
    n = 16 if rep.kind == "cvae" else min(16, rep.k)
    _, prop = rep.propose(z, n, torch.Generator().manual_seed(0))
    err = ((prop - chunk[:, None].float()) ** 2).mean((2, 3))
    _, prop_r = rep.propose(z[torch.randperm(len(z), device=z.device)], n, torch.Generator().manual_seed(0))
    err_r = ((prop_r - chunk[:, None].float()) ** 2).mean((2, 3))
    out = dict(rec=st["rec"], best16=err.min(1).values.mean().item(), mean16=err.mean().item(),
               best16_randstate=err_r.min(1).values.mean().item())
    if rep.kind == "motif":
        out.update(ce=st["ce"], used=float(rep.assign(z, chunk).unique().numel()))
    else:
        out.update(kl=st["kl"])
    rep.train()
    return out


if __name__ == "__main__":
    main()
