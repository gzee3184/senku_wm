"""World model with a global latent g and state tokens. Predicts one step from history and actions."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from senku.encoder import Encoder, SDPABlock


def rep_relu(x):
    """ReLU forward (exact zeros), GELU gradient backward (no dead units)."""
    gl = F.gelu(x)
    return torch.relu(x).detach() + gl - gl.detach()


def causal_history_mask(hist_len, n_extra):
    """Boolean mask, True = may attend. The history g's are causal; everything else attends freely."""
    S = hist_len + n_extra
    mask = torch.ones(S, S, dtype=torch.bool)
    for i in range(hist_len):
        mask[i, i + 1:hist_len] = False
    return mask


def pool_tokens(p, n_tokens):
    """[B, 196, D] -> [B, n_tokens, D] by square average pooling."""
    B, N, D = p.shape
    if n_tokens == N:
        return p
    s, t = math.isqrt(N), math.isqrt(n_tokens)
    assert s * s == N and t * t == n_tokens and s % t == 0, (N, n_tokens)
    x = p.transpose(1, 2).reshape(B, D, s, s)
    return F.avg_pool2d(x, s // t).flatten(2).transpose(1, 2)


class AdaLNBlock(nn.Module):
    """AdaLN-zero block (DiT): the condition c sets shift, scale and gate of both sub-layers."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, c, mask):
        s1, b1, g1, s2, b2, g2 = self.mod(c).unsqueeze(1).chunk(6, dim=-1)
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x) * (1 + s1) + b1).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + g1 * self.out_proj(o.transpose(1, 2).reshape(B, N, D))
        return x + g2 * self.mlp(self.ln2(x) * (1 + s2) + b2)


class TokenPredictor(nn.Module):
    """Self-attention over [g history, state tokens, action token]. With adaln=True the action also
    modulates every block; with adaln=False the action is one input token only.
    The output is a zero-init residual on (g_t, state tokens)."""

    def __init__(self, dim=192, action_dim=5, frame_skip=5, hist_len=3, depth=6, heads=3, n_tokens=49, adaln=True):
        super().__init__()
        self.hist_len, self.adaln = hist_len, adaln
        self.act_emb = nn.Sequential(nn.Linear(action_dim * frame_skip, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.hist_pos = nn.Parameter(torch.randn(1, hist_len, dim) * 0.02)
        self.tok_pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.act_pos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.register_buffer("mask", causal_history_mask(hist_len, n_tokens + 1), persistent=False)

    def forward(self, g_hist, s, a_block):
        a = self.act_emb(a_block)
        x = torch.cat([g_hist + self.hist_pos, s + self.tok_pos, a.unsqueeze(1) + self.act_pos], 1)
        c = a if self.adaln else torch.zeros_like(a)
        for blk in self.blocks:
            x = blk(x, c, self.mask)
        hl, n = self.hist_len, s.shape[1]
        nxt = torch.cat([g_hist[:, -1:], s], 1) + self.delta(self.out_norm(x[:, hl - 1:hl + n]))
        return nxt[:, 0], nxt[:, 1:]


class CrossAttn(nn.Module):
    """Queries x attend to keys and values m (the caller normalizes both)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, m):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(m).view(B, m.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(o.transpose(1, 2).reshape(B, N, D))


class ReadBlock(nn.Module):
    """g stream: AdaLN(action) self-attention, cross-attention READ from the state tokens, MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln_x = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_m = nn.LayerNorm(dim)
        self.read = CrossAttn(dim, heads)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, mem, c, mask):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.mod(c).unsqueeze(1).chunk(9, dim=-1)
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x) * (1 + s1) + b1).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + g1 * self.out_proj(o.transpose(1, 2).reshape(B, N, D))
        x = x + g2 * self.read(self.ln_x(x) * (1 + s2) + b2, self.ln_m(mem))
        return x + g3 * self.mlp(self.ln2(x) * (1 + s3) + b3)


class WriteBlock(nn.Module):
    """State tokens: self-attention, cross-attention WRITE from (g_t, g_t+1), MLP. No action input."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.self_blk = SDPABlock(dim, heads, mlp_ratio)
        self.ln_x = nn.LayerNorm(dim)
        self.ln_k = nn.LayerNorm(dim)
        self.write = CrossAttn(dim, heads)

    def forward(self, s, keys):
        s = self.self_blk(s)
        return s + self.write(self.ln_x(s), self.ln_k(keys))


class ReadWritePredictor(nn.Module):
    """READ: [g history, action] attends to the state tokens and predicts g_t+1 = g_t + delta.
    WRITE: the state tokens attend to each other and to (g_t, g_t+1) and predict s_t+1 = s_t + delta."""

    def __init__(self, dim=192, action_dim=5, frame_skip=5, hist_len=3, depth=6, heads=3, n_tokens=49, write_depth=2):
        super().__init__()
        self.hist_len = hist_len
        self.act_emb = nn.Sequential(nn.Linear(action_dim * frame_skip, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.hist_pos = nn.Parameter(torch.randn(1, hist_len, dim) * 0.02)
        self.act_pos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.tok_pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.key_pos = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.read_blocks = nn.ModuleList([ReadBlock(dim, heads) for _ in range(depth)])
        self.write_blocks = nn.ModuleList([WriteBlock(dim, heads) for _ in range(write_depth)])
        self.g_norm, self.s_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.g_delta, self.s_delta = nn.Linear(dim, dim), nn.Linear(dim, dim)
        for lin in (self.g_delta, self.s_delta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("mask", causal_history_mask(hist_len, 1), persistent=False)

    def write(self, s, g_last, g_next):
        keys = torch.stack([g_last, g_next], 1) + self.key_pos
        y = s + self.tok_pos
        for blk in self.write_blocks:
            y = blk(y, keys)
        return s + self.s_delta(self.s_norm(y))

    def forward(self, g_hist, s, a_block):
        a = self.act_emb(a_block)
        x = torch.cat([g_hist + self.hist_pos, a.unsqueeze(1) + self.act_pos], 1)
        mem = s + self.tok_pos
        for blk in self.read_blocks:
            x = blk(x, mem, a, self.mask)
        g_last = g_hist[:, -1]
        g_next = g_last + self.g_delta(self.g_norm(x[:, self.hist_len - 1]))
        return g_next, self.write(s, g_last, g_next)


def lewm_projector(dim, hidden=2048):
    """LeWM's projector: Linear-BatchNorm-GELU-Linear."""
    return nn.Sequential(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, dim))


class IDMHead(nn.Module):
    """Inverse dynamics head (training only): (g_t, g_t+1) -> the 5-step action block between them."""

    def __init__(self, dim=192, hidden=512, out=25):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(2 * dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out)

    def forward(self, z_t, z_next):
        h = F.gelu(self.fc1(torch.cat([self.norm(z_t), self.norm(z_next)], -1)))
        return self.out(F.gelu(self.fc2(h)))


class WorldModel(nn.Module):
    """Encodes frames and predicts the next global latent and state tokens from history and actions.
    encoder_mode: 'joint' (main predictor only) or 'token_aux' (auxiliary action predictor).
    dyn_arch: 'readwrite' (ReadWritePredictor) or 'adaln' (TokenPredictor).
    proprio_dim > 0: an MLP embeds the standardized joint readings (AntMaze: qpos[2:] and qvel, as in H-JEPA) and adds
    them to the encoder's global token. Its last layer starts at zero, so the model starts as the image-only model."""

    def __init__(self, embed_dim=192, action_dim=5, frame_skip=5, hist_len=3, encoder_depth=5, encoder_heads=3,
                 pred_depth=6, pred_heads=3, pred_tokens=49, encoder_mode="token_aux", dyn_arch="readwrite",
                 g_proj=True, g_sparse=True, idm_hidden=0, proprio_dim=0):
        super().__init__()
        assert encoder_mode in ("joint", "token_aux") and dyn_arch in ("adaln", "readwrite")
        self.config = dict(embed_dim=embed_dim, action_dim=action_dim, frame_skip=frame_skip, hist_len=hist_len,
                           encoder_depth=encoder_depth, encoder_heads=encoder_heads, pred_depth=pred_depth,
                           pred_heads=pred_heads, pred_tokens=pred_tokens, encoder_mode=encoder_mode,
                           dyn_arch=dyn_arch, g_proj=g_proj, g_sparse=g_sparse)
        if proprio_dim:
            self.config["proprio_dim"] = proprio_dim
        self.encoder_mode, self.g_sparse, self.pred_tokens = encoder_mode, g_sparse, pred_tokens
        self.encoder = Encoder(embed_dim, encoder_depth, encoder_heads)
        args = (embed_dim, action_dim, frame_skip, hist_len, pred_depth, pred_heads, pred_tokens)
        self.predictor = ReadWritePredictor(*args) if dyn_arch == "readwrite" else TokenPredictor(*args, adaln=True)
        self.g_projector = lewm_projector(embed_dim) if g_proj else None
        self.aux_predictor = TokenPredictor(*args, adaln=False) if encoder_mode == "token_aux" else None
        # training-only head (train.py --w-idm); not in self.config, and load_model drops its weights
        self.idm = IDMHead(embed_dim, idm_hidden, action_dim * frame_skip) if idm_hidden else None
        self.proprio_dim = proprio_dim
        if proprio_dim:
            self.proprio_emb = nn.Sequential(nn.Linear(proprio_dim, embed_dim), nn.SiLU(),
                                             nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.proprio_emb[2].weight)
            nn.init.zeros_(self.proprio_emb[2].bias)
            self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
            self.register_buffer("proprio_std", torch.ones(proprio_dim))

    def encode(self, x, return_raw=False, proprio=None):
        """proprio: raw [B, proprio_dim] (required when the model has proprio_dim)."""
        extra = None
        if self.proprio_dim:
            assert proprio is not None, "this model needs proprio (NaN rows = missing: image only)"
            miss = torch.isnan(proprio).any(-1, keepdim=True)
            pz = torch.where(miss, self.proprio_mean, proprio.float())
            extra = self.proprio_emb(((pz - self.proprio_mean) / self.proprio_std).to(x.dtype))
            extra = torch.where(miss, torch.zeros_like(extra), extra)
        g_raw, p = self.encoder(x, extra)
        g = self.g_projector(g_raw) if self.g_projector is not None else g_raw
        g = self.rectify(g)
        return (g, p, g_raw) if return_raw else (g, p)

    def state_tokens(self, p):
        return pool_tokens(p, self.pred_tokens)

    def step(self, g_hist, s, a_block):
        g_next, s_next = self.predictor(g_hist, s, a_block)
        return self.rectify(g_next), s_next

    def rectify(self, g):
        return rep_relu(g) if self.g_sparse else g


def strip_compile_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


_LEGACY_KEYS = ("attn", "sinkhorn_iters", "action_cond", "idm_hidden", "grad_ckpt")
_LEGACY_DEFAULTS = dict(encoder_mode="joint", dyn_arch="adaln", g_proj=False, g_sparse=False)
_UNSUPPORTED = dict(attn="sdpa", action_cond="both", dyn_state="tokens", g_head=False, read_topk=0, write_action=False)


def config_from_checkpoint(cfg):
    """Accepts this repo's config and the config of checkpoints trained before the release (extra keys, some
    defaults left out)."""
    for k, v in _UNSUPPORTED.items():
        assert cfg.get(k, v) == v, f"checkpoint uses {k}={cfg[k]}, which this release does not support"
    out = dict(_LEGACY_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k not in _LEGACY_KEYS and k not in _UNSUPPORTED})
    return out


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = WorldModel(**config_from_checkpoint(ck["config"])).to(device)
    sd = {k: v for k, v in strip_compile_prefix(ck["world_model"]).items() if not k.startswith("idm.")}
    m.load_state_dict(sd)
    return m.eval().requires_grad_(False)
