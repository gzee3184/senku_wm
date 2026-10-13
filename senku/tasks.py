"""Cube episode filters. Return valid row indices where start and goal states meet criteria. All distances in metres."""
import h5py
import numpy as np


def _state(h5):
    with h5py.File(h5, "r") as f:
        return f["privileged_block_0_pos"][:], f["proprio_effector_pos"][:], f["proprio_gripper_contact"][:, 0]


def hard(h5, valid, goal_offset, min_xy=0.0):
    """Cube on table, no contact, goal >= 5 cm away, gripper >= 2 cm from cube."""
    cube, ee, contact = _state(h5)
    g = valid + goal_offset
    keep = ((cube[valid, 2] < 0.03) & (contact[valid] < 0.1)
            & (np.linalg.norm(cube[g] - cube[valid], axis=1) >= 0.05)
            & (np.linalg.norm(ee[valid] - cube[valid], axis=1) >= 0.02)
            & (np.linalg.norm(ee[valid, :2] - cube[valid, :2], axis=1) >= min_xy))
    return valid[keep]


def hard_far(h5, valid, goal_offset):
    """hard filter plus gripper >= 3 cm from cube."""
    return hard(h5, valid, goal_offset, 0.03)


def pick(h5, valid, goal_offset, lift=0.035):
    """hard_far start; goal: cube in contact and lifted."""
    v = hard_far(h5, valid, goal_offset)
    cube, _, contact = _state(h5)
    g = v + goal_offset
    return v[(cube[g, 2] >= lift) & (contact[g] >= 0.1)]


def place_away(h5, valid, goal_offset, away=0.10):
    """Start mid-carry (cube >= 1 cm up); goal: cube resting, >= 5 cm from start, gripper >= away."""
    cube, ee, contact = _state(h5)
    g = valid + goal_offset
    keep = ((cube[valid, 2] - 0.02 >= 0.01)
            & (np.abs(cube[g, 2] - 0.02) < 0.005) & (contact[g] < 0.1)
            & (np.linalg.norm(cube[g, :2] - cube[valid, :2], axis=1) >= 0.05)
            & (np.linalg.norm(ee[g, :2] - cube[g, :2], axis=1) >= away))
    return valid[keep]


FILTERS = {"hard": hard, "hard_far": hard_far, "pick": pick, "place_away": place_away}
