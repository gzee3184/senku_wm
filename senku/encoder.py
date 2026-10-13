"""Image encoder: convolutional stem, positional encoding, and ViT. Returns global token and patch tokens."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class SDPABlock(nn.Module):
    """Pre-norm ViT block with softmax attention (F.scaled_dot_product_attention)."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out_proj(o.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.ln2(x))


class PositionalEncoding2D(nn.Module):
    """2D positional encoding with sinusoidal signals over spatial grid."""

    def __init__(self, d_model=192, grid_h=14, grid_w=14):
        super().__init__()
        assert d_model % 4 == 0
        d_half = d_model // 2
        y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, grid_h), torch.linspace(-1.0, 1.0, grid_w), indexing="ij")
        div = torch.exp(torch.arange(0, d_half, 2, dtype=torch.float32) * -(math.log(10000.0) / d_half))
        px, py = x.unsqueeze(-1) * div, y.unsqueeze(-1) * div
        pe = torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], -1).view(1, grid_h * grid_w, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe.to(dtype=x.dtype, device=x.device)


class Transformer(nn.Module):
    """Stack of ViT blocks with final layer norm."""

    def __init__(self, dim, depth, heads):
        super().__init__()
        self.blocks = nn.ModuleList([SDPABlock(dim, heads) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.final_ln(x)


class Encoder(nn.Module):
    def __init__(self, embed_dim=192, depth=5, heads=3, stem_channels=(96, 192, 384)):
        super().__init__()
        chans = (3, *stem_channels, embed_dim)
        layers = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.GELU()]
        self.stem = nn.Sequential(*layers)
        self.pos_2d = PositionalEncoding2D(embed_dim, 14, 14)
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.transformer = Transformer(embed_dim, depth, heads)

    def forward(self, x, extra=None):
        """x: [B, 3, 224, 224] -> (g [B, D], patch tokens [B, 196, D]). extra [B, D] is added to the global token."""
        tokens = self.pos_2d(self.stem(x).flatten(2).transpose(1, 2))
        glob = self.global_token.expand(x.shape[0], -1, -1)
        if extra is not None:
            glob = glob + extra.unsqueeze(1).to(glob.dtype)
        out = self.transformer(torch.cat([glob, tokens], 1))
        return out[:, 0], out[:, 1:]
