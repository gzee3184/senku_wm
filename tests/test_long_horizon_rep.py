import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="planner is CUDA-only")
D, T_TOK = 192, 49


class StubWM:
    def step(self, g_hist, s, a):                       # a [N, 25] = 5 steps x 5 dims
        g = g_hist[:, -1].clone()
        g[:, 0] += a.view(len(a), 5, 5)[:, :, 0].float().mean(1) * 0.2   # 1 per 25 steps at action +1
        return g, s


class StubHead:
    def __call__(self, x, y):                           # td_feats: column 0 = g0
        from senku.long_horizon.td_head import DMAX
        steps = 25 * (x[:, 0] - y[:, 0]).abs()
        return torch.log1p(steps) / float(np.log1p(DMAX))


class StubMotif(torch.nn.Module):
    kind = "motif"

    def prior_logits(self, z):
        return torch.tensor([[1.0, 0.5]], device=z.device).expand(len(z), -1)

    def decode(self, z, code):
        ch = torch.zeros(len(code), 25, 5, device=z.device)
        ch[:, :, 0] = torch.where(code == 0, 1.0, -1.0)[:, None]
        return ch

    def propose(self, z, n, gen=None):
        codes = torch.topk(self.prior_logits(z), n, -1).indices
        return codes, self.decode(z.repeat_interleave(n, 0), codes.flatten()).view(len(z), n, 25, 5)


def _planner():
    from senku.long_horizon.subgoals import ImaginedPlanner
    p = ImaginedPlanner(StubWM(), np.zeros(5, np.float32), np.ones(5, np.float32), search="rep")
    p.td_head, p.rep, p.rep_beams, p.rep_top = StubHead(), StubMotif(), 4, 2
    return p


def _st(g0):
    g = torch.zeros(1, 3, D, device="cuda")
    g[:, :, 0] = g0
    return {"g_hist": g, "s": torch.zeros(1, T_TOK, D, device="cuda")}


def _goal(g0):
    g = torch.zeros(1, D, device="cuda")
    g[0, 0] = g0
    return g, torch.zeros(1, T_TOK, D, device="cuda")


def test_beam_finds_the_less_likely_motif_when_it_reaches_the_goal():
    p = _planner()
    raw, keep, kstar = p.plan_rep(_st(0.0), *_goal(-3.0), 200)
    assert raw.shape == (200, 5) and len(keep) == 8
    assert kstar == 3                                    # three -1 chunks reach g0 = -3
    assert np.allclose(raw[:75, 0], -1.0)
    assert abs(float(keep[2][0][0, 0]) + 3.0) < 1e-3
    assert p.rep_log[-1]["kstar"] == 3


def test_stages_shrink_with_remaining_and_never_zero():
    p = _planner()
    raw, keep, _ = p.plan_rep(_st(0.0), *_goal(2.0), 30)
    assert len(keep) == 2 and raw.shape == (50, 5)
    raw, keep, _ = p.plan_rep(_st(0.0), *_goal(2.0), 5)
    assert len(keep) == 1


def test_kstar_zero_gives_goal_only_chain():
    from senku.long_horizon.subgoals import SubgoalChain
    p = _planner()
    _, _, kstar = p.plan_rep(_st(1.0), *_goal(1.0), 200)
    assert kstar == 0
    ch = SubgoalChain(p, 1, 200, sub_every=25)
    ch.goal[0] = _goal(1.0)
    ch._new_chain(0, _st(1.0), 0)
    assert len(ch.chain[0]) == 1 and ch.chain[0][0] is ch.goal[0]


def test_chain_is_imagined_latents_then_goal():
    from senku.long_horizon.subgoals import SubgoalChain
    p = _planner()
    ch = SubgoalChain(p, 1, 200, sub_every=25)
    ch.goal[0] = _goal(2.0)
    ch._new_chain(0, _st(0.0), 0)
    assert len(ch.chain[0]) == 3
    assert abs(float(ch.chain[0][0][0][0, 0]) - 1.0) < 1e-3 and ch.chain[0][-1] is ch.goal[0]


def test_rep_feats_equals_rep_input():
    from senku.long_horizon.repertoire import rep_input
    p = _planner()
    st = _st(0.5)
    st["g_hist"] = torch.randn(1, 3, D, device="cuda")
    torch.testing.assert_close(p.rep_feats(st["g_hist"], st["s"]), rep_input(st["g_hist"], st["s"]))
