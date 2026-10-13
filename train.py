"""Train a Senku world model from video frames and actions.

    python train.py --checkpoint-dir runs/cube --sources cube_single_expert.h5:1.0:holdout9800
"""
import argparse
import json
import os
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from senku.data import WindowDataset, parse_sources, proprio_stats
from senku.losses import SIGReg, diversity_loss, dyn_loss, effective_rank, grad_share, patch_residual, rdmreg
from senku.model import WorldModel, strip_compile_prefix
from senku.paths import CUBE
from senku.utils import normalize_frames


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=f"{CUBE}:1.0:holdout9800",
                   help="file.h5:weight[:holdout|holdoutN],... (files under $SENKU_HOME/data or full paths)")
    p.add_argument("--action-dim", type=int, default=None, help="default: read from the first source")
    p.add_argument("--checkpoint-dir", required=True)
    # architecture
    p.add_argument("--encoder-mode", choices=["joint", "token_aux"], default="token_aux")
    p.add_argument("--dyn-arch", choices=["adaln", "readwrite"], default="readwrite")
    p.add_argument("--no-g-proj", dest="g_proj", action="store_false")
    p.add_argument("--no-g-sparse", dest="g_sparse", action="store_false")
    p.add_argument("--pred-tokens", type=int, choices=[49, 196], default=49)
    p.add_argument("--encoder-depth", type=int, default=5)
    p.add_argument("--pred-depth", type=int, default=6)
    p.add_argument("--pred-heads", type=int, default=3)
    # losses
    p.add_argument("--n-targets", type=int, default=2)
    p.add_argument("--hop-decay", type=float, default=0.5)
    p.add_argument("--g-weight", type=float, default=0.5)
    p.add_argument("--no-motion", action="store_true", help="uniform token weights in the prediction loss")
    p.add_argument("--main-grad-share", type=float, default=0.3,
                   help="token_aux: share of the main predictor's gradient that reaches the encoder")
    p.add_argument("--w-sigreg", type=float, default=0.08)
    p.add_argument("--w-div", type=float, default=0.05)
    p.add_argument("--w-sigreg-roll", type=float, default=0.04)
    p.add_argument("--w-div-roll", type=float, default=0.025)
    p.add_argument("--w-sigreg-res", type=float, default=0.03)
    p.add_argument("--w-rdmreg-g", type=float, default=10.0)
    p.add_argument("--w-idm", type=float, default=0.0,
                   help="inverse dynamics loss weight: an MLP head predicts the action block from (g_t, g_t+1)")
    p.add_argument("--idm-hidden", type=int, default=512)
    # optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr-mult", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--lr-floor", type=float, default=0.1, help="cosine decays to lr * this")
    p.add_argument("--total-steps", type=int, default=100000)
    p.add_argument("--stop-at", type=int, default=0, help="stop early at this step")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=min(24, os.cpu_count() or 1))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--save-interval", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--proprio", action="store_true",
                   help="AntMaze: add qpos[2:] and qvel (H-JEPA's proprioception) to the encoder's global token")
    p.add_argument("--init", default=None,
                   help="start from these weights (missing keys, e.g. the proprio embedding, keep their init); "
                        "new optimizer and schedule")
    p.add_argument("--resume", default=None)
    p.add_argument("--reschedule", action="store_true",
                   help="with --resume: keep weights and AdamW state, start a new warmup + cosine at the resume step")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def rollout(step_fn, g_all, s_all, a, hl, args, sigreg, sparse):
    """Predict multiple targets recursively. Returns (dyn_loss, sigreg_loss, diversity_loss)."""
    nt = g_all.shape[1] - hl
    g_hist, s_cur = g_all[:, :hl], s_all[:, hl - 1]
    l_dyn, l_sr, l_dr, wsum = 0.0, 0.0, 0.0, 0.0
    for h in range(nt):
        g_next, s_next = step_fn(g_hist, s_cur, a[:, h])
        w = args.hop_decay ** h
        l_dyn = l_dyn + w * dyn_loss(g_next, s_next, g_all[:, hl + h], s_all[:, hl + h], s_all[:, hl + h - 1],
                                     g_weight=args.g_weight, motion=not args.no_motion)
        wsum += w
        ent = torch.cat([g_next.unsqueeze(1), s_next], 1)
        l_sr = l_sr + sigreg((ent[:, 1:] if sparse else ent).transpose(0, 1))
        l_dr = l_dr + diversity_loss(ent)
        g_hist = torch.cat([g_hist[:, 1:], g_next.unsqueeze(1)], 1)
        s_cur = s_next
    return l_dyn / wsum, l_sr / nt, l_dr / nt


def compute_losses(model, sigreg, batch, args, dev):
    ctx, tgt = batch["ctx_pixels"], batch["tgt_pixels"]
    B, hl, nt = ctx.shape[0], ctx.shape[1], tgt.shape[1]
    a = batch["action_blocks"].to(dev).float().clamp(-1, 1)
    prop = batch["proprio"].to(dev).float().flatten(0, 1) if "proprio" in batch else None
    x = normalize_frames(torch.cat([ctx, tgt], 1).flatten(0, 1), dev)
    g_all, p_all, _ = model.encode(x, return_raw=True, proprio=prop)
    D = g_all.shape[-1]
    g_all, p_all = g_all.view(B, hl + nt, D), p_all.view(B, hl + nt, -1, D)
    s_all = model.state_tokens(p_all.flatten(0, 1)).view(B, hl + nt, -1, D)
    sparse, zero = model.g_sparse, torch.zeros((), device=dev)

    aux = model.encoder_mode == "token_aux"
    lam = args.main_grad_share
    g_m, s_m = (grad_share(g_all, lam), grad_share(s_all, lam)) if aux else (g_all, s_all)
    l_dyn, l_sr, l_dr = rollout(model.step, g_m, s_m, a, hl, args, sigreg, sparse)
    l_aux = zero
    if aux:
        def aux_step(g_hist, s, a_block):
            g_next, s_next = model.aux_predictor(g_hist, s, a_block)
            return model.rectify(g_next), s_next
        la, lsa, lda = rollout(aux_step, g_all, s_all, a, hl, args, sigreg, sparse)
        l_aux = la + args.w_sigreg_roll * lsa + args.w_div_roll * lda

    entities0 = torch.cat([g_all[:, hl - 1:hl], p_all[:, hl - 1]], 1)
    l_sig = sigreg((entities0[:, 1:] if sparse else entities0).transpose(0, 1))
    l_div = diversity_loss(entities0)
    l_rdm = rdmreg(g_all) if args.w_rdmreg_g > 0 else zero
    l_res = sigreg(patch_residual(p_all[:, hl - 1]).transpose(0, 1)) if args.w_sigreg_res > 0 else zero
    l_idm = zero
    if args.w_idm > 0:   # blocks between consecutive frames: the context blocks + the first target block
        blocks = torch.cat([batch["ctx_action_blocks"].to(dev).float().clamp(-1, 1), a[:, :1]], 1)
        pred = model.idm(g_m[:, :hl].flatten(0, 1), g_m[:, 1:hl + 1].flatten(0, 1))
        l_idm = torch.nn.functional.mse_loss(pred, blocks.flatten(0, 1))
    total = (l_dyn + l_aux + args.w_sigreg * l_sig + args.w_div * l_div + args.w_rdmreg_g * l_rdm
             + args.w_sigreg_res * l_res + args.w_sigreg_roll * l_sr + args.w_div_roll * l_dr + args.w_idm * l_idm)
    return {"total": total, "dyn": l_dyn, "aux": l_aux, "sigreg": l_sig, "div": l_div, "rdmreg_g": l_rdm,
            "sigreg_res": l_res, "sigreg_roll": l_sr, "div_roll": l_dr, "idm": l_idm, "entities0": entities0.detach()}


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "training_log.json")
    sources = parse_sources(args.sources)
    if args.action_dim is None:
        with h5py.File(sources[0]["path"], "r") as f:
            args.action_dim = int(f["action"].shape[1])
    model = WorldModel(action_dim=args.action_dim, pred_tokens=args.pred_tokens, encoder_depth=args.encoder_depth,
                       pred_depth=args.pred_depth, pred_heads=args.pred_heads, encoder_mode=args.encoder_mode,
                       dyn_arch=args.dyn_arch, g_proj=args.g_proj, g_sparse=args.g_sparse,
                       idm_hidden=args.idm_hidden if args.w_idm > 0 else 0,
                       proprio_dim=27 if args.proprio else 0).to(dev)
    if args.proprio:
        mu, sd = proprio_stats(sources[0]["path"])
        model.proprio_mean.copy_(torch.from_numpy(mu))
        model.proprio_std.copy_(torch.from_numpy(sd))
    if args.init:
        sd0 = strip_compile_prefix(torch.load(args.init, map_location=dev, weights_only=False)["world_model"])
        missing, unexpected = model.load_state_dict(sd0, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("proprio_") for k in missing), missing
        print(f"init from {args.init}, new: {missing}", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    sigreg = SIGReg().to(dev)
    params = list(model.parameters())
    enc_ids = {id(p) for p in model.encoder.parameters()}
    if model.g_projector is not None:
        enc_ids |= {id(p) for p in model.g_projector.parameters()}
    groups = [{"params": [p for p in params if id(p) not in enc_ids], "lr": args.lr},
              {"params": [p for p in params if id(p) in enc_ids], "lr": args.lr * args.encoder_lr_mult}]
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    end = args.stop_at or args.total_steps
    s0 = [0]

    def lr_lambda(s):
        s = s - s0[0]
        if s < args.warmup:
            return (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.total_steps - s0[0] - args.warmup)
        return args.lr_floor + (1 - args.lr_floor) * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step, log = 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(strip_compile_prefix(ck["world_model"]))
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        if args.reschedule:
            for grp, base in zip(opt.param_groups, base_lrs):
                grp["initial_lr"] = base
            s0[0] = step
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=step - 1)
            sched.base_lrs = list(base_lrs)
            for grp, base in zip(opt.param_groups, base_lrs):
                grp["lr"] = base * lr_lambda(step)
        else:
            sched.load_state_dict(ck["scheduler"])
        if os.path.exists(log_path):
            log = [e for e in json.load(open(log_path)) if e["step"] <= step]
        print(f"resumed {args.resume} at step {step}", flush=True)
    if args.compile:
        model.encoder = torch.compile(model.encoder)
        model.predictor = torch.compile(model.predictor)
    ds = WindowDataset(sources, n_targets=args.n_targets, proprio=args.proprio)
    sampler = WeightedRandomSampler(torch.as_tensor(ds.weights), num_samples=len(ds), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed + step))
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"config={model.config} windows={len(ds)} sources={args.sources}", flush=True)
    model.train()
    it = iter(dl)
    t_last, s_last, skipped = time.time(), step, 0
    while step < end:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compute_losses(model, sigreg, batch, args, dev)
        ok = torch.isfinite(out["total"])
        if ok:
            out["total"].backward()
            ok = torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if ok:
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            skipped += 1
        sched.step()
        step += 1
        if step % args.log_interval == 0 or step == end:
            now = time.time()
            rec = {k: float(v.detach() if torch.is_tensor(v) else v) for k, v in out.items() if k != "entities0"}
            rec.update(step=step, erank=effective_rank(out["entities0"]), lr=sched.get_last_lr()[0],
                       st_per_s=(step - s_last) / max(1e-6, now - t_last),
                       mem_gb=torch.cuda.max_memory_allocated() / 1e9, skipped=skipped)
            t_last, s_last, skipped = now, step, 0
            log.append(rec)
            with open(log_path, "w") as f:
                json.dump(log, f)
            print(f"[{step:07d}/{end:07d}] " + " ".join(f"{k} {v:.4g}" for k, v in rec.items() if k != "step"),
                  flush=True)
        if step % args.save_interval == 0 or step == end:
            path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            torch.save({"step": step, "config": model.config, "world_model": strip_compile_prefix(model.state_dict()),
                        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "args": vars(args)}, path)
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
