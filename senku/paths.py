"""Data and checkpoint paths. SENKU_HOME: ~/.senku by default; datasets under $SENKU_HOME/data."""
import os
from pathlib import Path

HOME = Path(os.environ.get("SENKU_HOME", Path.home() / ".senku")).expanduser()
DATA = HOME / "data"
CKPT = HOME / "checkpoints"

CUBE = "cube_single_expert.h5"
PUSHT = "pusht_expert_train.h5"
SCENE = "scene_play_224px_official_policy.h5"
ANTMAZE = "visual_antmaze_medium_explore_stitch_train.h5"


def data(name):
    """Resolve a dataset name or path to its full path."""
    p = Path(name).expanduser()
    return p if p.is_absolute() or p.exists() else DATA / name
