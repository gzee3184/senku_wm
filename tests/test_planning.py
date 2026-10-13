from types import SimpleNamespace

import numpy as np
import pytest
import torch

from senku.planning import FRAME_SKIP, GDSolver, decreasing_horizon, zscore

# H-JEPA's stable_worldmodel.solver.decreasing_horizon_schedule(num_calls, configured_horizon)
HJEPA_SCHEDULES = {
    (35, 23): [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4,
               4, 3, 3, 2, 2, 1, 1],
    (5, 5): [5, 4, 3, 2, 1],
    (10, 23): [23, 21, 19, 16, 14, 11, 9, 6, 4, 1],
    (25, 5): [5] * 5 + [4] * 5 + [3] * 5 + [2] * 5 + [1] * 5,
    (7, 3): [3, 3, 2, 2, 1, 1, 1],
    (3, 23): [23, 12, 1],
}


@pytest.mark.parametrize("key", list(HJEPA_SCHEDULES))
def test_decreasing_horizon_matches_hjepa(key):
    n_calls, horizon = key
    assert [decreasing_horizon(i, horizon, n_calls) for i in range(n_calls)] == HJEPA_SCHEDULES[key]


class QuadCost:
    """Toy differentiable cost: distance of the summed actions to a per-env target. Two terms, as GDCost."""

    def encode(self, info):
        return info["target"]

    def __call__(self, target, cand):
        d = cand.sum(2) - target[:, None]
        return (d ** 2).sum(-1), d.abs().sum(-1)


def make_solver(chunk, hold=False, seed=0):
    s = GDSolver(None, num_samples=8, n_steps=40, lr=0.1, min_its=5, seed=seed, device="cpu", chunk=chunk,
                 cost=QuadCost(), hold=hold)
    cfg = SimpleNamespace(horizon=3, receding_horizon=2, action_block=FRAME_SKIP)
    s.configure(action_space=SimpleNamespace(shape=(4, 2)), n_envs=4, config=cfg)
    return s


def info(n=4):
    return {"pixels_hist": torch.zeros(n, 1), "target": torch.linspace(-1, 1, n)[:, None].repeat(1, 2 * FRAME_SKIP)}


def test_gd_batched_equals_one_by_one():
    a1 = make_solver(chunk=1).solve(info())["actions"]
    a4 = make_solver(chunk=4).solve(info())["actions"]
    assert a1.shape == (4, 3, 2 * FRAME_SKIP)
    torch.testing.assert_close(a1, a4)


def test_gd_hold_repeats_each_action_over_its_block():
    a = make_solver(chunk=2, hold=True).solve(info())["actions"].view(4, 3, FRAME_SKIP, 2)
    assert torch.equal(a, a[:, :, :1].expand_as(a))


def test_zscore_per_env():
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 30.0, 50.0]])
    z = zscore(x)
    np.testing.assert_allclose(z.mean(1).numpy(), 0, atol=1e-6)
    np.testing.assert_allclose(z.numpy()[0], z.numpy()[1], atol=1e-5)
