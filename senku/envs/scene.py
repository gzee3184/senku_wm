"""OGBench Scene environment helpers for the training protocol.

Stores the scene layout (privileged qpos, button states) and goal configuration in the dataset.
The environment's task-mode success check determines episode success.
"""
import numpy as np

# privileged_qpos layout (nq = 25): 0-5 arm, 6-13 gripper, 14-16 cube xyz, 17-20 cube quat, 21-22 button joints,
# 23 drawer slide, 24 window slide
CUBE, DRAWER, WINDOW = slice(14, 17), 23, 24
TOL = 0.04
OBJECTS = ("cube", "button_0", "button_1", "drawer", "window")
CALLABLES = [
    {"method": "set_state", "args": {"qpos": {"value": "privileged_qpos"}, "qvel": {"value": "privileged_qvel"},
                                     "button_states": {"value": "button_states"}}},
    {"method": "set_scene_goal", "args": {"goal_qpos": {"value": "goal_privileged_qpos"},
                                          "goal_button_states": {"value": "goal_button_states"}}},
]


def set_scene_goal(self, goal_qpos, goal_button_states):
    """Targets of the env's task-mode success from a dataset goal state (no rendering change: visualize_info off)."""
    goal_qpos = np.asarray(goal_qpos, np.float64)
    mid = self._cube_target_mocap_ids[0]
    self._data.mocap_pos[mid] = goal_qpos[CUBE]
    self._target_button_states = np.asarray(goal_button_states).astype(int).copy()
    self._target_drawer_pos = float(goal_qpos[DRAWER])
    self._target_window_pos = float(goal_qpos[WINDOW])
    self._success = False


def install():
    from ogbench.manipspace.envs.scene_env import SceneEnv
    SceneEnv.set_scene_goal = set_scene_goal


def prepare_dataset(dataset):
    """The Scene h5 has no ep_idx / step_idx / button_states columns: add them to the dataset's cache."""
    lens, offs = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    ep = np.repeat(np.arange(len(lens)), lens)
    step = np.concatenate([np.arange(n) for n in lens])
    b = np.stack([dataset.get_col_data(f"privileged_button_{i}_state") for i in range(2)], 1).astype(np.int64)
    for k, v in (("ep_idx", ep), ("step_idx", step), ("button_states", b)):
        dataset._cache[k] = v
        if k not in dataset._keys:
            dataset._keys.append(k)
    assert offs[-1] + lens[-1] == len(ep)
    h5_rows = dataset.get_row_data

    def get_row_data(row_idx):           # the stock one reads every key from the file; the added ones are cache-only
        dataset._open()
        return {c: (dataset._cache[c][row_idx] if c in dataset._cache else dataset.h5_file[c][row_idx])
                for c in dataset._keys}
    dataset.get_row_data = get_row_data
    del h5_rows


def needs(q0, q1, b0, b1):
    """Which objects are NOT at the goal at the start (the env's own tolerances), [N, 5] bool."""
    return np.stack([np.linalg.norm(q1[:, CUBE] - q0[:, CUBE], axis=1) > TOL,
                     b0[:, 0] != b1[:, 0], b0[:, 1] != b1[:, 1],
                     np.abs(q1[:, DRAWER] - q0[:, DRAWER]) > TOL,
                     np.abs(q1[:, WINDOW] - q0[:, WINDOW]) > TOL], 1)


def changed_start_rows(dataset, valid, goal_offset):
    q = dataset.get_col_data("privileged_qpos")
    b = dataset._cache["button_states"]
    n = needs(q[valid], q[valid + goal_offset], b[valid], b[valid + goal_offset])
    return valid[n.any(1)]


def start_needs(dataset, rows, goal_offset):
    q = dataset.get_col_data("privileged_qpos")
    b = dataset._cache["button_states"]
    return needs(q[rows], q[rows + goal_offset], b[rows], b[rows + goal_offset])


def final_object_success(world):
    """Per env, per object: at its goal at the end of the episode (or at termination), [N, 5] bool."""
    out = []
    for e in world.envs.envs:
        c, bt, d, w = e.unwrapped._compute_successes()
        out.append([bool(c[0]), bool(bt[0]), bool(bt[1]), bool(d), bool(w)])
    return np.asarray(out)
