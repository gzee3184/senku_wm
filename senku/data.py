"""Training windows: hist_len context frames and n_targets target frames, frame_skip env steps apart.

Frame k is at row t0 + k * frame_skip. Action block k holds the frame_skip actions that take frame k to frame k + 1,
time-major (25 numbers for Cube). action_blocks[h] starts at the last context frame.
"""
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from torch.utils.data import Dataset

from senku.paths import data

HOLDOUT_MOD = 10


def parse_sources(spec):
    """Parse source spec into dicts. Format: file.h5:weight[:holdout|holdoutN][:whitelist.npy],...
    holdout drops episodes with id % 10 == 0; holdoutN also drops ids >= N."""
    out = []
    for item in spec.split(","):
        parts = item.split(":")
        path, weight = parts[0], float(parts[1])
        holdout = len(parts) > 2 and parts[2].startswith("holdout")
        holdout_from = int(parts[2][7:]) if holdout and len(parts[2]) > 7 else None
        whitelist = parts[3] if len(parts) > 3 and parts[3] else None
        out.append({"path": str(data(path)), "weight": weight, "holdout": holdout, "holdout_from": holdout_from,
                    "whitelist": whitelist})
    return out


class WindowDataset(Dataset):
    def __init__(self, sources, hist_len=3, n_targets=2, frame_skip=5, stride=3, proprio=False):
        self.sources, self.hl, self.nt, self.fs, self.proprio = sources, hist_len, n_targets, frame_skip, proprio
        self.span = (hist_len - 1 + n_targets) * frame_skip
        index, weights = [], []
        for si, s in enumerate(sources):
            with h5py.File(s["path"], "r") as f:
                off, lens = f["ep_offset"][:], f["ep_len"][:]
            eps = np.arange(len(lens))
            if s.get("holdout"):
                eps = eps[eps % HOLDOUT_MOD != 0]
            if s.get("holdout_from") is not None:
                eps = eps[eps < s["holdout_from"]]
            if s.get("whitelist"):
                eps = np.intersect1d(eps, np.load(s["whitelist"]))
            rows = []
            for e in eps:
                last_start = int(lens[e]) - self.span - 2      # keeps the terminal NaN action out
                if last_start >= 0:
                    rows.extend(int(off[e]) + t for t in range(0, last_start + 1, stride))
            rows = np.asarray(rows, np.int64)
            index.append(np.stack([np.full(len(rows), si, np.int64), rows], 1))
            weights.append(np.full(len(rows), s["weight"] / max(1, len(rows))))
        self.index = np.concatenate(index)
        self.weights = np.concatenate(weights)
        self._files = {}

    def __len__(self):
        return len(self.index)

    def _file(self, si):
        if si not in self._files:
            self._files[si] = h5py.File(self.sources[si]["path"], "r")
        return self._files[si]

    def __getitem__(self, i):
        si, t0 = (int(x) for x in self.index[i])
        f = self._file(si)
        frames = f["pixels"][t0:t0 + self.span + 1][::self.fs]   # contiguous read, then subsample (faster in h5py)
        acts = f["action"][t0:t0 + self.span].astype(np.float32)
        if np.isnan(acts).any():
            raise ValueError(f"NaN action in window source={si} row={t0}")
        blocks = acts.reshape(self.hl - 1 + self.nt, self.fs * acts.shape[1])
        out = {"ctx_pixels": frames[:self.hl], "tgt_pixels": frames[self.hl:],
               "ctx_action_blocks": blocks[:self.hl - 1], "action_blocks": blocks[self.hl - 1:]}
        if self.proprio:
            out["proprio"] = proprio_rows(f, t0, t0 + self.span + 1)[::self.fs]
        return out


def proprio_rows(f, a, b):
    """AntMaze proprioception as H-JEPA uses it: qpos[2:] (no xy) and qvel, rows a:b, float32."""
    return np.concatenate([f["qpos"][a:b, 2:], f["qvel"][a:b]], 1).astype(np.float32)


def proprio_stats(path, n=2_000_000):
    """Mean and std of the proprioception over the first n rows."""
    with h5py.File(data(path), "r") as f:
        x = proprio_rows(f, 0, min(n, len(f["qpos"])))
    return x.mean(0), x.std(0) + 1e-6


def action_scaler(path):
    """Return mean and std of actions in dataset, with NaN rows dropped."""
    with h5py.File(data(path), "r") as f:
        a = f["action"][:]
    a = a[~np.isnan(a).any(axis=1)]
    return a.mean(0).astype(np.float32), a.std(0).astype(np.float32)
