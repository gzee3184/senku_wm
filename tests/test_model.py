import pytest
import torch

from senku.losses import SIGReg
from senku.model import WorldModel, config_from_checkpoint, load_model


@pytest.mark.parametrize("dyn_arch", ["readwrite", "adaln"])
@pytest.mark.parametrize("encoder_mode", ["token_aux", "joint"])
def test_step_shapes_and_identity_at_init(dyn_arch, encoder_mode):
    m = WorldModel(dyn_arch=dyn_arch, encoder_mode=encoder_mode).eval()
    g, p = m.encode(torch.randn(2, 3, 224, 224))
    s = m.state_tokens(p)
    assert g.shape == (2, 192) and p.shape == (2, 196, 192) and s.shape == (2, 49, 192)
    g_hist = torch.stack([g, g, g], 1)
    g_next, s_next = m.step(g_hist, s, torch.randn(2, 25))
    assert torch.allclose(g_next, g) and torch.allclose(s_next, s)   # zero-init residual outputs
    assert (m.aux_predictor is not None) == (encoder_mode == "token_aux")


def test_save_load_roundtrip(tmp_path):
    m = WorldModel()
    path = tmp_path / "m.pt"
    torch.save({"config": m.config, "world_model": m.state_dict()}, path)
    m2 = load_model(path, "cpu")
    assert m2.config == m.config


def test_legacy_config():
    cfg = dict(embed_dim=192, action_dim=5, frame_skip=5, hist_len=3, encoder_depth=5, encoder_heads=3, attn="sdpa",
               sinkhorn_iters=24, pred_depth=6, pred_heads=3, pred_tokens=49, action_cond="both", idm_hidden=512,
               grad_ckpt=False, encoder_mode="token_aux", g_proj=True, dyn_arch="readwrite", g_sparse=True)
    out = config_from_checkpoint(cfg)
    assert out["dyn_arch"] == "readwrite" and "attn" not in out
    with pytest.raises(AssertionError):
        config_from_checkpoint(dict(cfg, attn="sinkhorn"))


def test_sigreg_prefers_gaussian():
    torch.manual_seed(0)
    sig = SIGReg()
    assert sig(torch.randn(1, 512, 16)) < sig(torch.rand(1, 512, 16) * 0.1)
