"""OGBench Cube (single) helpers: a raw env for data generation and a ground-truth state readout."""
import numpy as np


def make_cube_env():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    env = gym.make("swm/OGBCube-v0", env_type="single", ob_type="states", multiview=False,
                   width=224, height=224, visualize_info=False, terminate_at_goal=False)
    env.reset(seed=0)
    return env.unwrapped


def read_env_state(env):
    """Gripper pinch site, cube position, gripper closure (~0 open .. ~0.8 closed), right-pad contact, goal cube."""
    u = env.unwrapped
    return dict(ee=u.data.site_xpos[u.model.site("ur5e/robotiq/pinch").id].copy(),
                cube=u.data.joint("object_joint_0").qpos[:3].copy(),
                closure=float(u.data.qpos[6]),
                contact=float(np.clip(np.linalg.norm(u.data.body("ur5e/robotiq/right_pad").cfrc_ext) / 50, 0, 1)),
                goal=u.data.mocap_pos[u._cube_target_mocap_ids[0]].copy())
