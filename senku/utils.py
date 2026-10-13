"""Image preprocessing for the model: uint8 frames to ImageNet-normalized 224 x 224 floats."""
import numpy as np
import torch
import torch.nn.functional as F

IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_frames(u8, dev):
    """uint8 tensor [N, H, W, C] -> normalized float [N, 3, 224, 224] on dev (bilinear resize if needed)."""
    x = u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - IM_MEAN.to(dev)) / IM_STD.to(dev)


def to_x(u8, dev="cuda"):
    """uint8 array [..., H, W, C] -> normalized float [..., 3, 224, 224] on dev."""
    t = torch.from_numpy(np.asarray(u8))
    lead = t.shape[:-3]
    x = normalize_frames(t.reshape(-1, *t.shape[-3:]), dev)
    return x.view(*lead, *x.shape[1:])
