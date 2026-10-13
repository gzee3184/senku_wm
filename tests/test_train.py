import torch

import train
from senku.losses import SIGReg
from senku.model import WorldModel


def test_losses_finite_and_backward():
    torch.manual_seed(0)
    args = train.get_args(["--checkpoint-dir", "unused"])
    m = WorldModel().train()
    batch = {"ctx_pixels": torch.randint(0, 255, (2, 3, 64, 64, 3), dtype=torch.uint8),
             "tgt_pixels": torch.randint(0, 255, (2, 2, 64, 64, 3), dtype=torch.uint8),
             "ctx_action_blocks": torch.rand(2, 2, 25) * 2 - 1, "action_blocks": torch.rand(2, 2, 25) * 2 - 1}
    out = train.compute_losses(m, SIGReg(num_proj=64), batch, args, "cpu")
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert m.encoder.global_token.grad is not None
