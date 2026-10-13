import numpy as np
import torch

from senku.losses import SIGReg
from senku.long_horizon.upper_levels import AD, UDIM, ZD, UpperLevels, losses, windows


def test_windows_stay_inside_episodes():
    ep = np.r_[np.zeros(10, int), np.ones(4, int), np.full(13, 2)]
    st = np.r_[np.arange(10), np.arange(4), np.arange(13)] * 5
    held = ep == 2
    J, Jh = windows(ep, st, held, 7)
    assert J.tolist() == [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 20]
    assert Jh.tolist() == [False] * 4 + [True] * 7


def test_shapes_and_macro_strides():
    up = UpperLevels(2)
    u = torch.randn(5, UDIM)
    z2, z3 = up.lift(u)
    assert z2.shape == (5, ZD) and z3.shape == (5, ZD)
    a2, a3 = up.macro(torch.randn(5, 40, 5))
    assert a2.shape == (5, 4, AD) and a3.shape == (5, 2, AD)
    assert up.stride(0) == 10 and up.stride(1) == 20


def test_losses_decrease_on_a_learnable_toy():
    """Level-1 features drift by the mean action: the upper dynamics must learn it."""
    torch.manual_seed(0)
    up = UpperLevels(2)
    sig = SIGReg(knots=17, num_proj=64)
    opt = torch.optim.Adam(up.parameters(), 1e-3)
    W = torch.randn(5, UDIM) * 0.5

    def batch(n=64):
        A = torch.randn(n, 60, 5)
        base = torch.randn(n, 1, UDIM)
        U = base + torch.cumsum(A.view(n, 12, 5, 5).mean(2) @ W, 1)
        U = torch.cat([base, U], 1)                                   # 13 frames
        return U, A
    first = None
    for i in range(300):
        loss, _ = losses(up, *batch(), sig)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.7 * first
