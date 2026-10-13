"""Planning: cost models, solvers (HeldCEM and the gradient planner) and the frame-history policy."""
import time
from collections import deque

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from torchvision.transforms import v2 as transforms

FRAME_SKIP, HIST_LEN = 5, 3


def img_transform(img_size=224):
    """ImageNet normalization and resize."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def _x(u8, tf):
    """Convert uint8 image to tensor and apply transform."""
    return tf(torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1))


class CostModel(nn.Module):
    """Cost [B, S] of CEM candidates: distance of the predicted end state to the goal.
    cost_mode: g (global latent), tokens (state tokens), both (each term z-scored over the candidates, then summed)
    or upper (distance in an upper level's latent, long-horizon only)."""

    def __init__(self, model, act_scaler, cost_mode="both", act_low=-1.0, act_high=1.0):
        super().__init__()
        self.model, self.cost_mode, self.amp = model, cost_mode, True
        self.register_buffer("act_mean", torch.tensor(act_scaler.mean_, dtype=torch.float32))
        self.register_buffer("act_std", torch.tensor(act_scaler.scale_, dtype=torch.float32))
        self.act_low, self.act_high = act_low, act_high
        self.upper = None

    @torch.no_grad()
    def get_action(self, info, horizon=1, prefix_actions=None):
        return torch.zeros(len(info["goal"]), horizon, self.act_mean.numel() * FRAME_SKIP)

    @torch.no_grad()
    def get_cost(self, info_dict, candidates):
        dev = self.act_mean.device
        m = self.model
        hist = info_dict["pixels_hist"][:, 0].to(dev)
        B, S, H, _ = candidates.shape
        prop_h = prop_goal = None
        if getattr(m, "proprio_dim", 0):
            prop_h = info_dict["proprio_hist"][:, 0].to(dev).float().flatten(0, 1)
            prop_goal = info_dict["goal_proprio"][:, 0].to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            g_h, p_h = m.encode(hist.flatten(0, 1).float(), proprio=prop_h)
            g_hist = g_h.view(B, HIST_LEN, -1)
            s = m.state_tokens(p_h.view(B, HIST_LEN, *p_h.shape[1:])[:, -1])
            if "sub_g" in info_dict:
                g_goal, s_goal = info_dict["sub_g"][:, 0].to(dev), info_dict["sub_s"][:, 0].to(dev)
            elif self.cost_mode != "upper":
                g_goal, p_goal = m.encode(info_dict["goal"][:, 0, -1].to(dev).float(), proprio=prop_goal)
                s_goal = m.state_tokens(p_goal)
            a = candidates.to(dev).float().view(B, S, H, FRAME_SKIP, -1)
            a = (a * self.act_std + self.act_mean).clamp(self.act_low, self.act_high).view(B * S, H, -1)
            g_hist, s = g_hist.repeat_interleave(S, 0), s.repeat_interleave(S, 0)
            for h in range(H):
                g_next, s = m.step(g_hist, s, a[:, h])
                g_hist = torch.cat([g_hist[:, 1:], g_next.unsqueeze(1)], 1)
        g_end, s = g_hist[:, -1].float(), s.float()
        if self.cost_mode == "upper":
            from senku.long_horizon.td_head import td_feats
            up = self.upper
            sub = info_dict["sub_up"][:, 0].to(dev).float().repeat_interleave(S, 0)
            zt = up.lift(td_feats(g_end, s), top=up.n - 1)[-1]
            return ((zt - sub) ** 2).sum(-1).view(B, S)
        cost_g = ((g_end - g_goal.float().repeat_interleave(S, 0)) ** 2).sum(-1).view(B, S)
        cost_p = ((s - s_goal.float().repeat_interleave(S, 0)) ** 2).sum(-1).mean(-1).view(B, S)
        if self.cost_mode == "g":
            return cost_g
        if self.cost_mode == "tokens":
            return cost_p
        return zscore(cost_g) + zscore(cost_p)


def zscore(x):
    """Standardize [B, S] costs over the S candidates of each env."""
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)


def ste_clamp(x, lo, hi):
    """Clamp forward, identity gradient (lets gradient planning push saturated actions back)."""
    return x + (x.clamp(lo, hi) - x).detach()


class GDCost:
    """Differentiable cost for GDSolver: the last `last_n` predicted (g, state tokens) against the goal.
    Returns the g and token terms separately. GDSolver divides each term by its spread over the initial candidates
    (fixed for the solve). This replaces the per-term z-score of the CEM cost, which has zero gradient when summed
    over the candidates."""

    def __init__(self, cm, last_n=2):
        self.cm, self.last_n = cm, last_n

    @torch.no_grad()
    def encode(self, info):
        """Context and goal latents for one env batch. info values: [B, ...] (no sample dim)."""
        cm, m = self.cm, self.cm.model
        dev = cm.act_mean.device
        hist = info["pixels_hist"].to(dev).float()
        B = hist.shape[0]
        prop_h = prop_goal = None
        if getattr(m, "proprio_dim", 0):
            prop_h = info["proprio_hist"].to(dev).float().flatten(0, 1)
            prop_goal = info["goal_proprio"].to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g_h, p_h = m.encode(hist.flatten(0, 1), proprio=prop_h)
            s = m.state_tokens(p_h.view(B, HIST_LEN, *p_h.shape[1:])[:, -1])
            g_goal, p_goal = m.encode(info["goal"][:, -1].to(dev).float(), proprio=prop_goal)
            s_goal = m.state_tokens(p_goal)
        return g_h.view(B, HIST_LEN, -1).float(), s.float(), g_goal.float(), s_goal.float()

    def __call__(self, ctx, cand):
        """ctx from encode (B envs); cand [B, S, H, A*FRAME_SKIP] normalized (or [S, H, .] with B=1) -> (g cost, token
        cost), each [B, S] ([S] for 3-D cand), with grad. Every env's S candidates are rolled out in one batch."""
        cm, m = self.cm, self.cm.model
        g_hist, s, g_goal, s_goal = ctx
        flat = cand.dim() == 3
        if flat:
            cand = cand.unsqueeze(0)
        B, S, H, _ = cand.shape
        a = cand.reshape(B * S, H, FRAME_SKIP, -1) * cm.act_std + cm.act_mean
        a = ste_clamp(a, cm.act_low, cm.act_high).view(B * S, H, -1)
        g_hist, s, g_goal, s_goal = (t.repeat_interleave(S, 0) for t in (g_hist, s, g_goal, s_goal))
        cg, cp = 0.0, 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for h in range(H):
                g_next, s = m.step(g_hist, s, a[:, h])
                g_hist = torch.cat([g_hist[:, 1:], g_next.unsqueeze(1)], 1)
                if h >= H - self.last_n:
                    cg = cg + ((g_next.float() - g_goal) ** 2).sum(-1)
                    cp = cp + ((s.float() - s_goal) ** 2).sum(-1).mean(-1)
        cg, cp = cg.view(B, S), cp.view(B, S)
        return (cg[0], cp[0]) if flat else (cg, cp)


class LeWMGDCost:
    """GDCost for a LeWM checkpoint (stable_worldmodel JEPA): LeWM's own rollout, cost = squared distance of the last
    `last_n` predicted embeddings to the goal embedding (H-JEPA's cost_last_n). Context and goal are encoded once per
    solve. Candidates are in LeWM's normalized action space, as LeWM is trained."""

    def __init__(self, lewm, last_n=2, device="cuda"):
        self.m, self.last_n, self.device = lewm, last_n, device

    @torch.no_grad()
    def encode(self, info):
        x = {k: v.to(self.device).unsqueeze(1) for k, v in info.items() if torch.is_tensor(v) and k != "pixels_hist"}
        x = {k: v.float() if v.is_floating_point() else v for k, v in x.items()}
        B, H = x["pixels"].shape[0], x["pixels"].shape[2]
        ad = next(m for m in self.m.action_encoder.modules() if isinstance(m, nn.Conv1d)).in_channels   # A x block
        x["action"] = torch.zeros(B, 1, H, ad, device=self.device)
        goal = {k: v[:, 0] for k, v in x.items()}   # the goal encoding of LeWM's get_cost
        goal["pixels"] = goal["goal"]
        for k in [k for k in x if k.startswith("goal_")]:
            goal[k[len("goal_"):]] = goal.pop(k)
        goal.pop("action")
        x["goal_emb"] = self.m.encode(goal)["emb"]
        self.m.rollout(x, torch.zeros(B, 1, H + 1, ad, device=self.device))   # caches emb
        return {k: x[k] for k in ("pixels", "goal", "emb", "goal_emb")}

    def __call__(self, ctx, cand):
        B, S = cand.shape[:2]
        info = {k: v.expand(B, S, *v.shape[2:]) for k, v in ctx.items()}
        pred = self.m.rollout(info, cand)["predicted_emb"][..., -self.last_n:, :]
        goal = ctx["goal_emb"][:, -1][:, None, None]   # goal_emb [B, T, D] -> last frame [B, 1, 1, D]
        c = ((pred - goal.expand_as(pred)) ** 2).sum((-1, -2))
        return c, torch.zeros_like(c)


def decreasing_horizon(call_idx, horizon, n_calls):
    """H-JEPA's per-call horizon schedule (stable_worldmodel solver.decreasing_horizon_schedule): horizons H..1 share
    the n_calls replans equally, extra calls go to the shortest horizons."""
    if horizon == 1 or n_calls <= 1:
        return horizon
    if n_calls < horizon:
        sched = [max(1, int(np.ceil(1 + (horizon - 1) * (n_calls - 1 - i) / (n_calls - 1)))) for i in range(n_calls)]
    else:
        per, extra = divmod(n_calls, horizon)
        sched = []
        for j, hv in enumerate(range(horizon, 0, -1)):
            sched += [hv] * (per + (1 if j >= horizon - extra else 0))
    return sched[min(call_idx, len(sched) - 1)]


class GDSolver:
    """H-JEPA's level-1 planner (stable_worldmodel GradientSolver as configured in H-JEPA's ant_flat.yaml): candidates
    start at N(0, 1) in normalized action space (candidate 0 = the mean), AdamW on the actions through the world model,
    early stop when the best cost stalls (patience 5, rel. delta 1e-2, after min_its), return the lowest-cost candidate.
    No warm start. With budget set, the horizon follows H-JEPA's decreasing schedule (the policy sets steps_taken)."""

    def __init__(self, cm, num_samples=100, n_steps=90, lr=0.3, min_its=30, patience=5, rel_delta=1e-2, last_n=2,
                 budget=None, seed=42, device="cuda", chunk=None, cost=None, hold=False):
        self.cost = cost or GDCost(cm, last_n)
        self.hold = hold   # optimize one action per 5-step block, held (as HeldCEM)
        self.chunk = chunk or max(1, 1000 // num_samples)   # ~1000 rollouts per batch (memory)
        self.S, self.n_steps, self.lr, self.min_its = num_samples, n_steps, lr, min_its
        self.patience, self.rel_delta, self.budget = patience, rel_delta, budget
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.device, self.steps_taken = device, 0
        self.env_iters = 0   # optimizer steps summed over envs (compute report)

    def configure(self, *, action_space, n_envs, config):
        self.cfg, self.ad = config, int(np.prod(action_space.shape[1:])) * config.action_block
        self.action_dim, self.n_envs, self.horizon = self.ad, n_envs, config.horizon   # swm Solver protocol

    def __call__(self, info, init_action=None):
        return self.solve(info)

    def _expand(self, v):
        """With hold: [..., d] block actions -> [..., d * FRAME_SKIP], each action repeated for its block."""
        if not self.hold:
            return v
        return v.unsqueeze(-2).expand(*v.shape[:-1], FRAME_SKIP, v.shape[-1]).reshape(*v.shape[:-1], self.ad)

    def cur_horizon(self):
        if self.budget is None:
            return self.cfg.horizon
        every = self.cfg.receding_horizon * self.cfg.action_block
        return decreasing_horizon(self.steps_taken // every, self.cfg.horizon, -(-self.budget // every))

    def solve(self, info, init_action=None):
        """Envs are solved `chunk` at a time in one batch; each env keeps its own scale, early stop and argmin, so the
        result matches solving them one by one (chunk=1), only faster."""
        H = self.cur_horizon()
        n = len(info["pixels_hist"])
        out = []
        for i0 in range(0, n, self.chunk):
            sl = slice(i0, min(n, i0 + self.chunk))
            ctx = self.cost.encode({k: v[sl] for k, v in info.items() if torch.is_tensor(v)})
            B = sl.stop - sl.start
            # one draw per env, in env order: the same random stream as solving the envs one by one
            d = self.ad // FRAME_SKIP if self.hold else self.ad
            x = torch.stack([torch.randn(self.S, H, d, generator=self.gen, device=self.device) for _ in range(B)])
            x[:, 0] = 0
            x.requires_grad_(True)
            opt = torch.optim.AdamW([x], lr=self.lr)
            best = torch.full((B,), float("inf"), device=self.device)
            stall = torch.zeros(B, dtype=torch.long, device=self.device)
            active = torch.ones(B, dtype=torch.bool, device=self.device)
            final, scale = x.detach().clone(), None
            for step in range(self.n_steps):
                self.env_iters += int(active.sum())
                cg, cp = self.cost(ctx, self._expand(x))
                if scale is None:
                    scale = (cg.std(1, keepdim=True).detach() + 1e-6, cp.std(1, keepdim=True).detach() + 1e-6)
                c = cg / scale[0] + cp / scale[1]
                c.sum().backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                score = c.min(1).values.detach()
                better = ~torch.isfinite(best) | (best - score > self.rel_delta * best.abs().clamp_min(1e-12))
                best = torch.where(better, score, best)
                stall = torch.where(better, torch.zeros_like(stall), stall + 1)
                stop = active & ~better & (stall >= self.patience)
                if step + 1 < self.min_its:
                    stop = torch.zeros_like(active)
                final[stop] = x.detach()[stop]
                active &= ~stop
                if not active.any():
                    break
            final[active] = x.detach()[active]
            with torch.no_grad():
                cg, cp = self.cost(ctx, self._expand(final))
                idx = (cg / scale[0] + cp / scale[1]).argmin(1)
                out.append(self._expand(final[torch.arange(B, device=self.device), idx]).cpu())
        a = torch.cat(out)
        keep = self.cfg.receding_horizon
        if a.shape[1] < keep:   # the policy executes `keep` blocks; H-JEPA's policy executes min(keep, H) and replans
            a = torch.cat([a, a.new_zeros(n, keep - a.shape[1], a.shape[2])], 1)
        return {"actions": a}


class SolveTimer:
    """Counts the planner calls and their GPU time (compute report). Wraps solver.solve in place."""

    def __init__(self, solver):
        self.solver, self.calls, self.seconds, solve = solver, 0, 0.0, solver.solve

        def timed(*a, **kw):
            torch.cuda.synchronize()
            t = time.time()
            out = solve(*a, **kw)
            torch.cuda.synchronize()
            self.calls, self.seconds = self.calls + 1, self.seconds + time.time() - t
            return out
        solver.solve = timed

    def report(self, n_env, samples, iters=None):
        """Predictor rollouts per episode: candidates x iterations per solve. iters=None: the gradient planner, whose
        iterations vary per env (each costs a forward and a backward pass)."""
        if iters is None:
            rollouts = self.solver.env_iters * samples / n_env
        else:
            rollouts = self.calls * samples * iters
        return {"solve_calls": self.calls, "solve_seconds": self.seconds, "envs": n_env,
                "rollouts_per_episode": rollouts}


class LeWMCostModel(nn.Module):
    """Wrap LeWM checkpoint to use HeldCEM solver."""

    def __init__(self, lewm, act_scaler):
        super().__init__()
        self.model = lewm
        self.register_buffer("act_mean", torch.tensor(act_scaler.mean_, dtype=torch.float32))

    @torch.no_grad()
    def get_action(self, info, horizon=1, prefix_actions=None):
        return torch.zeros(len(info["goal"]), horizon, self.act_mean.numel() * FRAME_SKIP)

    @torch.no_grad()
    def get_cost(self, info_dict, candidates):
        info = {k: v for k, v in info_dict.items() if k != "pixels_hist"}
        return self.model.get_cost(info, candidates.to(self.act_mean.device))


class HeldCEM(swm.solver.CEMSolver):
    """CEM with hold, keep_elite_frac, and return_best options."""

    def __init__(self, *args, hold=True, keep_elite_frac=0.3, return_best=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold, self.keep_elite_frac, self.return_best = hold, keep_elite_frac, return_best

    def _sample(self, bs, bm, bv):
        d = self.action_dim // FRAME_SKIP if self.hold else self.action_dim
        x = torch.randn(bs, self.num_samples, self.horizon, d, generator=self.torch_gen, device=self.device,
                        dtype=self.dtype)
        if self.hold:
            x = x.repeat(1, 1, 1, FRAME_SKIP)
        return x * bv.unsqueeze(1) + bm.unsqueeze(1)

    @torch.no_grad()
    def solve(self, info_dict, init_action=None):
        from stable_worldmodel.solver.utils import prepare_init_action
        total = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                          n_envs=total, action_dim=self.action_dim)
        mean, var = self.init_action_distrib(total, init_action)
        mean, var = mean.to(self.device), var.to(self.device)
        d = self.action_dim // FRAME_SKIP
        if self.hold:
            mean = mean.view(total, self.horizon, FRAME_SKIP, d).mean(2).repeat(1, 1, FRAME_SKIP)
        out = mean.clone()
        costs_out = []
        n_keep = int(self.keep_elite_frac * self.topk)
        for s0 in range(0, total, self.batch_size):
            s1 = min(s0 + self.batch_size, total)
            bs = s1 - s0
            bm, bv = mean[s0:s1], var[s0:s1]
            infos = {}
            for k, v in info_dict.items():
                v = v[s0:s1]
                if torch.is_tensor(v):
                    v = v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else None)
                    v = v.unsqueeze(1).expand(bs, self.num_samples, *v.shape[1:])
                elif isinstance(v, np.ndarray):
                    v = np.repeat(v[:, None, ...], self.num_samples, axis=1)
                infos[k] = v
            elite = None
            for _ in range(self.n_steps):
                cand = self._sample(bs, bm, bv)
                cand[:, 0] = bm
                if elite is not None and n_keep > 0:
                    cand[:, 1:1 + n_keep] = elite[:, :n_keep]
                vals, inds = torch.topk(self.model.get_cost(infos, cand), k=self.topk, dim=1, largest=False)
                elite = cand[torch.arange(bs, device=self.device)[:, None], inds]
                bm, bv = elite.mean(1), elite.std(1)
                if self.hold:
                    bv = bv.view(bs, self.horizon, FRAME_SKIP, d)[:, :, :1].repeat(1, 1, FRAME_SKIP, 1).view_as(bv)
            out[s0:s1] = elite[:, 0] if self.return_best else bm
            var[s0:s1] = bv
            costs_out.extend(vals.mean(1).cpu().tolist())
        return {"costs": costs_out, "actions": out.detach().cpu(), "mean": [out.detach().cpu()],
                "var": [var.detach().cpu()]}


class HistoryPolicy(swm.policy.WorldModelPolicy):
    """Gives the planner 3 frames 5 env steps apart (and their joint readings, if the model uses them).
    Optional long-horizon modes: expert or imagined subgoals (seed_subgoals) and the chains set by set_long_horizon."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subgoals = self.lh_chain = self.up_chain = None
        self._up_sub = self._lh_goal = None

    def set_long_horizon(self, lh_chain=None, up_chain=None):
        """lh_chain: a SubgoalChain (imagined subgoals). up_chain: an UpperChain (upper levels)."""
        self.lh_chain, self.up_chain = lh_chain, up_chain

    def seed_history(self, pre_frames, pre_proprio=None):
        """Initialize frame history buffers. pre_frames: [n_env, 10, H, W, C]; pre_proprio: [n_env, 10, P] for
        proprio models (the env loop then passes info["proprio"] [n_env, P] each step)."""
        span = (HIST_LEN - 1) * FRAME_SKIP + 1
        self._frames = [deque(list(f), maxlen=span) for f in pre_frames]
        self._props = None if pre_proprio is None else [deque(list(p), maxlen=span) for p in pre_proprio]
        self._t = 0

    def _x_hist(self, hist_i):
        return torch.stack([_x(h, self.transform["pixels_hist"]) for h in hist_i]).cuda()

    def _goal_x(self, goal_u8):
        return _x(goal_u8, self.transform["goal"]).cuda()

    def get_action(self, info_dict, **kwargs):
        pix = np.asarray(info_dict["pixels"])
        E = len(self._frames)
        hist = []
        for i, buf in enumerate(self._frames):
            buf.append(pix[i, -1])
            frames = list(buf)
            hist.append(np.stack([frames[-1 - FRAME_SKIP * k] for k in range(HIST_LEN - 1, -1, -1)]))
        info_dict = dict(info_dict)
        info_dict["pixels_hist"] = np.stack(hist)
        if self._props is not None:
            ph = []
            for i, buf in enumerate(self._props):
                buf.append(np.asarray(info_dict["proprio"])[i])
                ps = list(buf)
                ph.append(np.stack([ps[-1 - FRAME_SKIP * k] for k in range(HIST_LEN - 1, -1, -1)]))
            info_dict["proprio_hist"] = np.stack(ph).astype(np.float32)
            del info_dict["proprio"]
        t, self._t = self._t, self._t + 1
        if hasattr(self.solver, "steps_taken"):
            self.solver.steps_taken = t
        goal_u8 = np.asarray(info_dict["goal"]).reshape(E, -1, *pix.shape[-3:])[:, -1]
        if self._subgoals is not None:
            self._expert_subgoal(info_dict, t, hist, goal_u8)
        upc = self.up_chain
        if upc is not None:
            if self._up_sub is None:
                self._up_sub = [None] * E
            for i in range(E):
                if self._up_sub[i] is None or len(self._action_buffer[i]) == 0:
                    self._up_sub[i] = upc.step(i, self._x_hist(hist[i]), self._goal_x(goal_u8[i]), t)
            info_dict["sub_up"] = torch.cat(self._up_sub, 0).cpu()
        chain = self.lh_chain
        if chain is not None:
            if chain.direct:
                return np.stack([chain.direct_action(i, self._x_hist(hist[i]), self._goal_x(goal_u8[i]), t)
                                 for i in range(E)]).astype(np.float32)
            if self._lh_goal is None:
                self._lh_goal = [None] * E
            for i in range(E):
                if self._lh_goal[i] is None or len(self._action_buffer[i]) == 0:
                    self._lh_goal[i] = chain.step(i, self._x_hist(hist[i]), self._goal_x(goal_u8[i]), t)
            info_dict["sub_g"] = torch.cat([g for g, _ in self._lh_goal], 0).cpu()
            info_dict["sub_s"] = torch.cat([s for _, s in self._lh_goal], 0).cpu()
        return super().get_action(info_dict, **kwargs)

    def seed_subgoals(self, frames, seg_len, imagine=None, actions=None, every=None, advance=None):
        """Set expert frames as subgoals. Frames [E, n, H, W, C] with stride every. Transitions every seg_len steps."""
        self._subgoals, self._seg_len = frames, seg_len
        self._imag, self._imag_raw, self._imag_every = imagine, actions, every
        self._adv, self._k = advance, None

    @torch.no_grad()
    def _imagine(self, i, hist_i, goal_u8):
        p = self._imag
        keep = p.subgoals(p.init_state_x(self._x_hist(hist_i)), self._imag_raw[i], self._imag_every)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gg, pp = p.m.encode(self._goal_x(goal_u8)[None].float())
        return keep[:-1] + [(gg.float(), p.m.state_tokens(pp).float())]

    def _expert_subgoal(self, info_dict, t, hist, goal_u8):
        sg = self._subgoals
        E, n = sg.shape[:2]
        if self._imag is not None and t == 0:
            self._lat = [self._imagine(i, hist[i], sg[i, -1]) for i in range(E)]
        if self._adv is not None:
            ks = self._advance_model(t, hist)
        else:
            ks = np.full(E, min(t // self._seg_len, n - 1))
        if self._imag is None:
            g = np.asarray(info_dict["goal"])
            info_dict["goal"] = sg[np.arange(E), ks].reshape(g.shape)
        else:
            info_dict["sub_g"] = torch.cat([self._lat[i][k][0] for i, k in enumerate(ks)], 0).cpu()
            info_dict["sub_s"] = torch.cat([self._lat[i][k][1] for i, k in enumerate(ks)], 0).cpu()

    @torch.no_grad()
    def _advance_model(self, t, hist):
        from senku.long_horizon.subgoals import SubgoalChain
        budget, frac = self._adv
        E = len(hist)
        if self._k is None:
            self._k, self._seg0, self._d0 = np.zeros(E, int), np.zeros(E, int), np.ones(E)
        for i in range(E):
            if self._k[i] >= len(self._lat[i]) - 1 or (t > 0 and len(self._action_buffer[i]) > 0):
                continue
            st = self._imag.init_state_x(self._x_hist(hist[i]))
            d = SubgoalChain.dist(st["g_hist"][0, -1], st["s"][0], self._lat[i][self._k[i]])
            if t == 0:
                self._d0[i] = d
                continue
            if d <= frac * self._d0[i] or t - self._seg0[i] >= budget:
                self._k[i] += 1
                self._seg0[i] = t
                self._d0[i] = SubgoalChain.dist(st["g_hist"][0, -1], st["s"][0], self._lat[i][self._k[i]])
        return self._k
