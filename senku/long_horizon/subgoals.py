"""Imagined subgoals from the world model: plan in imagination and extract latent targets.

Two outer searches: CEM over coarse held actions, or a goal-blind repertoire whose proposals a steps-apart head
scores. The model's latents every 25 steps along the chosen plan are the subgoals. A subgoal is passed when the
current latent is close enough to it, or skipped after a time budget; both events start a new plan.
"""
import numpy as np
import torch

from senku.utils import to_x

FRAME_SKIP, HIST_LEN = 5, 3


REP_TIE = 1.0   # env steps: repertoire stage scores within this of the best count as tied


class ImaginedPlanner:
    """Plan a sequence of actions over T steps by optimizing them in imagination."""

    def __init__(self, m, act_mean, act_std, block=25, samples=300, iters=30, topk=30, keep=0.3, seed=0,
                 search="cem"):
        assert block % FRAME_SKIP == 0 and search in ("cem", "rep")
        self.m, self.block, self.samples, self.iters, self.topk, self.search = m, block, samples, iters, topk, search
        self.n_keep = int(keep * topk)
        self.td_head, self.rep = None, None
        self.rep_beams, self.rep_top, self.rep_stages, self.rep_shuffle = 8, 8, 8, False
        self.rep_log = []
        self.mean = torch.as_tensor(act_mean, dtype=torch.float32, device="cuda")
        self.std = torch.as_tensor(act_std, dtype=torch.float32, device="cuda")
        self.gen = torch.Generator(device="cuda").manual_seed(seed)

    @torch.no_grad()
    def encode_goal(self, px):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g, p = self.m.encode(to_x(px[None]).float())
            return g.float(), self.m.state_tokens(p).float()

    @torch.no_grad()
    def init_state(self, frames):
        """Encode frames [3, H, W, C] uint8 and return initial planner state."""
        return self.init_state_x(to_x(frames))

    @torch.no_grad()
    def init_state_x(self, x_hist):
        """Encode normalized frames [3, C, H, W] and return initial planner state."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g, p = self.m.encode(x_hist.float())
            return {"g_hist": g.float()[None], "s": self.m.state_tokens(p[-1:]).float()}

    def _roll(self, st, raw, every=None):
        """Imagine raw actions and return final latents and intermediate checkpoints every `every` steps."""
        N, T = raw.shape[:2]
        g_hist = st["g_hist"].expand(N, -1, -1).contiguous()
        s = st["s"].expand(N, -1, -1).contiguous()
        a = raw.reshape(N, T // FRAME_SKIP, FRAME_SKIP * raw.shape[-1])
        keep = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(a.shape[1]):
                g_next, s = self.m.step(g_hist, s, a[:, b])
                g_hist = torch.cat([g_hist[:, 1:], g_next.unsqueeze(1)], 1)
                if every and ((b + 1) * FRAME_SKIP) % every == 0:
                    keep.append((g_hist[:, -1].float(), s.float()))
        return g_hist[:, -1].float(), s.float(), keep

    @staticmethod
    def cost(g, s, g_goal, s_goal):
        """Planning cost: g and state-token distances, each z-scored over the candidates, summed."""
        def z(x):
            return (x - x.mean()) / (x.std() + 1e-6)
        return z(((g - g_goal) ** 2).sum(-1)) + z(((s - s_goal) ** 2).sum(-1).mean(-1))

    @torch.no_grad()
    def plan(self, st, g_goal, s_goal, T, init_raw=None):
        """Find best action sequence of length T by optimizing in imagination."""
        nb = -(-T // self.block)
        d = self.mean.numel()
        mu = torch.zeros(nb, d, device="cuda")
        if init_raw is not None:
            r = torch.as_tensor(init_raw[:nb * self.block], dtype=torch.float32, device="cuda")
            r = torch.cat([r, r[-1:].expand(nb * self.block - len(r), -1)]) if len(r) < nb * self.block else r
            mu = ((r - self.mean) / self.std).view(nb, self.block, d).mean(1)
        sd = torch.ones(nb, d, device="cuda")
        elite = None
        for _ in range(self.iters):
            x = torch.randn(self.samples, nb, d, generator=self.gen, device="cuda") * sd + mu
            x[0] = mu
            if elite is not None and self.n_keep:
                x[1:1 + self.n_keep] = elite[:self.n_keep]
            g, s, _ = self._roll(st, (x * self.std + self.mean).clamp(-1, 1).repeat_interleave(self.block, 1))
            idx = torch.topk(self.cost(g, s, g_goal, s_goal), self.topk, largest=False).indices
            elite = x[idx]
            mu, sd = elite.mean(0), elite.std(0)
        return (elite[0] * self.std + self.mean).clamp(-1, 1).repeat_interleave(self.block, 0)[:T].cpu().numpy()

    @torch.no_grad()
    def subgoals(self, st, raw, every):
        """Extract imagined latent checkpoints every `every` steps along the action sequence."""
        return self._roll(st, torch.as_tensor(raw, dtype=torch.float32, device="cuda")[None], every)[2]

    @torch.no_grad()
    def td_steps(self, g, s, g_goal, s_goal):
        """Predict env steps from current state (g, s) to goal state."""
        from senku.long_horizon.td_head import DMAX, td_feats
        x, y = td_feats(g, s), td_feats(g_goal, s_goal).expand(len(g), -1)
        return torch.expm1(self.td_head(x.float(), y.float()) * float(np.log1p(DMAX)))

    def rep_feats(self, g_hist, s):
        from senku.long_horizon.repertoire import rep_input
        z = rep_input(g_hist, s)
        if self.rep_shuffle:
            bank = self.rep.bank
            z = bank[torch.randint(0, len(bank), (len(z),), generator=self.gen, device="cuda")]
        return z

    def _rep_stage(self, g_hist, s, chunk_std):
        """Imagine one repertoire action chunk and update latent state."""
        from senku.long_horizon.repertoire import CH
        raw = (chunk_std * self.std + self.mean).clamp(-1, 1)
        a = raw.reshape(len(raw), CH // FRAME_SKIP, FRAME_SKIP * raw.shape[-1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(a.shape[1]):
                g_next, s = self.m.step(g_hist, s, a[:, b])
                g_hist = torch.cat([g_hist[:, 1:], g_next.unsqueeze(1)], 1)
        return g_hist.float(), s.float(), raw

    @torch.no_grad()
    def plan_rep(self, st, g_goal, s_goal, T):
        """Plan using repertoire proposals, scored by steps-to-goal."""
        from senku.long_horizon.repertoire import CH
        S = max(1, min(self.rep_stages, -(-T // CH)))
        td_now = float(self.td_steps(st["g_hist"][:, -1], st["s"], g_goal, s_goal)[0])
        fn = self._plan_rep_beam if self.rep.kind == "motif" else self._plan_rep_cem
        raw, keep, kstar, best, codes = fn(st, g_goal, s_goal, S, td_now)
        self.rep_log.append(dict(kstar=int(kstar), score=float(best), td_now=td_now, stages=S, codes=codes))
        return raw, keep, kstar

    def _plan_rep_beam(self, st, g_goal, s_goal, S, td_now):
        """Beam search over motif-library proposals, scoring by steps-to-goal."""
        from senku.long_horizon.repertoire import CH
        gh, s = st["g_hist"], st["s"]
        A = self.mean.numel()
        raws = torch.zeros(1, 0, A, device="cuda")
        codes = torch.zeros(1, 0, dtype=torch.long, device="cuda")
        kg = torch.zeros(1, 0, *gh.shape[2:], device="cuda")
        ks = torch.zeros(1, 0, *s.shape[1:], device="cuda")
        best, hist = (td_now, 0, None), []
        smin = td_now                                     # lowest score so far; near-ties go to the deeper stage
        for k in range(1, S + 1):
            c, ch = self.rep.propose(self.rep_feats(gh, s), min(self.rep_top, getattr(self.rep, "k", self.rep_top)),
                                     self.gen)
            m_ = c.shape[1]
            g2, s2, raw = self._rep_stage(gh.repeat_interleave(m_, 0), s.repeat_interleave(m_, 0), ch.flatten(0, 1))
            td = self.td_steps(g2[:, -1], s2, g_goal, s_goal)
            raws = torch.cat([raws.repeat_interleave(m_, 0), raw], 1)
            codes = torch.cat([codes.repeat_interleave(m_, 0), c.reshape(-1, 1)], 1)
            kg = torch.cat([kg.repeat_interleave(m_, 0), g2[:, -1:]], 1)
            ks = torch.cat([ks.repeat_interleave(m_, 0), s2[:, None]], 1)
            i0 = int(torch.argmin(td))
            sk = CH * k + float(td[i0])
            if sk <= smin + REP_TIE:                     # a perfect head scores every stage of a good path equally
                best, smin = (sk, k, i0), min(smin, sk)
            hist.append((raws, codes, kg, ks))
            order = torch.argsort(td)[:self.rep_beams]
            gh, s = g2[order], s2[order]
            raws, codes, kg, ks = raws[order], codes[order], kg[order], ks[order]
        kstar = best[1]
        if kstar == 0:
            r, c, g_, s_ = raws[0], codes[0], kg[0], ks[0]
        else:
            r, c, g_, s_ = (x[best[2]] for x in hist[kstar - 1])
            if kstar < S:
                fin = [j for j in range(len(codes)) if torch.equal(codes[j, :kstar], c)]
                if fin:
                    j = fin[0]
                    r, c, g_, s_ = raws[j], codes[j], kg[j], ks[j]
                else:
                    r = torch.cat([r, r[-CH:].repeat(S - kstar, 1)])
                    g_ = torch.cat([g_, g_[-1:].expand(S - kstar, -1)])
                    s_ = torch.cat([s_, s_[-1:].expand(S - kstar, -1, -1)])
        keep = [(g_[k][None], s_[k][None]) for k in range(len(g_))]
        return r.cpu().numpy(), keep, kstar, best[0], c.tolist()

    def _plan_rep_cem(self, st, g_goal, s_goal, S, td_now):
        """CEM over CVAE-learned action codes, scoring by steps-to-goal."""
        from senku.long_horizon.repertoire import CH
        dc = self.rep.code
        mu, sd = torch.zeros(S, dc, device="cuda"), torch.ones(S, dc, device="cuda")
        elite = None

        def roll(eps):
            n = len(eps)
            gh = st["g_hist"].expand(n, -1, -1).contiguous()
            s = st["s"].expand(n, -1, -1).contiguous()
            sc, raws, kg, ks = [], [], [], []
            for k in range(S):
                z = self.rep_feats(gh, s)
                pm, pls = self.rep.prior(z)
                gh, s, raw = self._rep_stage(gh, s, self.rep.decode(z, pm + pls.exp() * eps[:, k].clamp(-2.5, 2.5)))
                sc.append(CH * (k + 1) + self.td_steps(gh[:, -1], s, g_goal, s_goal))
                raws.append(raw)
                kg.append(gh[:, -1])
                ks.append(s)
            return torch.stack(sc, 1), torch.cat(raws, 1), torch.stack(kg, 1), torch.stack(ks, 1)

        for _ in range(self.iters):
            x = torch.randn(self.samples, S, dc, generator=self.gen, device="cuda") * sd + mu
            x[0] = mu
            if elite is not None and self.n_keep:
                x[1:1 + self.n_keep] = elite[:self.n_keep]
            sc = roll(x)[0]
            idx = torch.topk(sc.min(1).values.clamp(max=td_now), self.topk, largest=False).indices
            elite = x[idx]
            mu, sd = elite.mean(0), elite.std(0)
        sc, raws, kg, ks = roll(elite[:1])
        smin = min(float(sc[0].min()), td_now)
        ok = (sc[0] <= smin + REP_TIE).nonzero()
        kstar = int(ok.max()) + 1 if len(ok) else 0
        keep = [(kg[0, k][None], ks[0, k][None]) for k in range(S)]
        return raws[0].cpu().numpy(), keep, kstar, smin, elite[0].flatten().tolist()


class SubgoalChain:
    """Track and advance through a sequence of subgoal targets during hierarchical planning."""

    def __init__(self, planner, n_envs, horizon, sub_every=25, seg_budget=50, adv_frac=0.5):
        self.p, self.H, self.every, self.seg_budget, self.adv_frac = planner, horizon, sub_every, seg_budget, adv_frac
        self.goal = [None] * n_envs
        self.chain = [[] for _ in range(n_envs)]
        self.k = np.zeros(n_envs, int)
        self.seg_start = np.zeros(n_envs, int)
        self.d0 = np.ones(n_envs)
        self.raw = [None] * n_envs
        self.log = [dict(passed=0, abandoned=0, replans=0, events=[]) for _ in range(n_envs)]
        self.direct = False
        self._dbuf = [[] for _ in range(n_envs)]

    @staticmethod
    def dist(g, s, sg):
        """Distance to a subgoal in normalized (g, s) latent space."""
        g0, s0 = sg
        dg = ((g - g0) ** 2).sum() / ((g0 ** 2).sum() + 1e-6)
        ds = ((s - s0) ** 2).sum(-1).mean() / ((s0 ** 2).sum(-1).mean() + 1e-6)
        return float(dg + ds)

    def _ensure_goal(self, i, goal_x):
        if self.goal[i] is not None:
            return False
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g, p = self.p.m.encode(goal_x[None].float())
            self.goal[i] = (g.float(), self.p.m.state_tokens(p).float())
        return True

    def _new_chain(self, i, st, t):
        remaining = max(self.H - t, self.every)
        if self.p.search == "rep":
            raw, keep, kstar = self.p.plan_rep(st, *self.goal[i], remaining)
            self.chain[i] = keep[:kstar] + [self.goal[i]]
        else:
            raw = self.p.plan(st, *self.goal[i], remaining, init_raw=self.raw[i])
            sub = self.p.subgoals(st, raw, self.every)
            self.chain[i] = sub[:-1] + [self.goal[i]] if len(sub) else [self.goal[i]]
        self.raw[i] = raw
        self.k[i], self.seg_start[i] = 0, t
        self.log[i]["replans"] += 1

    def step(self, i, x_hist, goal_x, t):
        """Get current subgoal target, advancing to next if reached or timed out."""
        st = self.p.init_state_x(x_hist)
        if self._ensure_goal(i, goal_x):
            self._new_chain(i, st, t)
            self.d0[i] = self.dist(st["g_hist"][0, -1], st["s"][0], self.chain[i][0])
        g, s = st["g_hist"][0, -1], st["s"][0]
        if self.k[i] < len(self.chain[i]) - 1:
            reached = self.dist(g, s, self.chain[i][self.k[i]]) <= self.adv_frac * self.d0[i]
            if reached or t - self.seg_start[i] >= self.seg_budget:
                self.log[i]["passed" if reached else "abandoned"] += 1
                self.log[i]["events"].append((int(t), "pass" if reached else "abandon", int(self.k[i])))
                self._new_chain(i, st, t)
                self.d0[i] = self.dist(g, s, self.chain[i][self.k[i]])
        return self.chain[i][self.k[i]]

    def direct_action(self, i, x_hist, goal_x, t):
        """Execute plan open-loop: no inner replanning, just pop buffered actions."""
        if not self._dbuf[i]:
            self._ensure_goal(i, goal_x)
            self._new_chain(i, self.p.init_state_x(x_hist), t)
            self._dbuf[i] = list(self.raw[i][:self.every])
        return self._dbuf[i].pop(0)
