"""Training losses."""
import torch
import torch.nn.functional as F
from torch import nn


def token_err(pred, tgt):
    """MSE plus cosine distance between predicted and target tokens."""
    return (pred - tgt).pow(2).mean(-1) + (1 - F.cosine_similarity(pred, tgt, dim=-1))


def motion_weights(s_prev, s_tgt, floor=0.1):
    """Per-token weights that sum to 1 per frame, proportional to how much each token changed (plus a floor).
    A fully static frame gets uniform weights."""
    m = (s_tgt - s_prev).norm(dim=-1).detach()
    mean = m.mean(-1, keepdim=True)
    w = floor + torch.where(mean > 1e-8, m / mean.clamp_min(1e-8), torch.ones_like(m))
    return w / w.sum(-1, keepdim=True)


def dyn_loss(g_pred, s_pred, g_tgt, s_tgt, s_prev, g_weight=0.5, motion=True):
    """g gets a fixed share g_weight of the loss; the state tokens share the rest."""
    lg = token_err(g_pred, g_tgt).mean()
    e = token_err(s_pred, s_tgt)
    lp = (motion_weights(s_prev, s_tgt) * e).sum(-1).mean() if motion else e.mean()
    return g_weight * lg + (1 - g_weight) * lp


class SIGReg(nn.Module):
    """LeWM's sketched isotropic Gaussian regularizer (Epps-Pulley statistic over random projections).
    Input (T, B, D): the test is over axis B."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True) + 1e-8)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def diversity_loss(entities):
    """Mean squared pairwise cosine between the M tokens of each frame. entities: [B, M, D]."""
    B, M, _ = entities.shape
    e = F.normalize(entities, dim=-1)
    sim = torch.einsum("bid,bjd->bij", e, e)
    eye = torch.eye(M, device=entities.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_fill(eye, 0.0).square().sum() / (B * M * (M - 1))


def rdmreg(z, n_proj=1024, scale=0.5 ** 0.5):
    """Sliced 2-Wasserstein distance between z [..., D] and a rectified Laplace target ReLU(Laplace(0, scale))^D
    (RDMReg of the sparse latent-particle WM, arXiv 2608.22764)."""
    z = z.reshape(-1, z.shape[-1]).float()
    n, d = z.shape
    c = torch.randn(d, n_proj, device=z.device)
    c = c / c.norm(dim=0, keepdim=True)
    u = torch.rand(n, d, device=z.device).clamp(1e-6, 1 - 1e-6) - 0.5
    y = torch.relu(-scale * torch.sign(u) * torch.log1p(-2 * u.abs()))
    zp, _ = torch.sort(z @ c, dim=0)
    yp, _ = torch.sort(y @ c, dim=0)
    return (zp - yp).pow(2).mean()


def patch_residual(p):
    """[B, N, D] -> each token minus the mean token of its frame."""
    return p - p.mean(1, keepdim=True)


def grad_share(x, lam):
    """Value of x, gradient scaled by lam (0 = detach, 1 = identity)."""
    if lam == 0:
        return x.detach()
    return x.detach() + lam * (x - x.detach())


@torch.no_grad()
def effective_rank(entities):
    """exp(entropy of the normalized singular values) of the [B * M, D] token matrix."""
    flat = entities.reshape(-1, entities.shape[-1]).float()
    S = torch.linalg.svdvals(flat - flat.mean(0, keepdim=True))
    p = S / (S.sum() + 1e-8)
    p = p[p > 1e-8]
    return float(torch.exp(-(p * p.log()).sum()))
