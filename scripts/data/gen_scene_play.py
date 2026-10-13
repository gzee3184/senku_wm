"""Generate Scene corpus using OGBench's task-chaining play policy.

Replicates OGBench's generate_manipspace.py at arbitrary resolution. Writes one episode
at a time to HDF5 with minimal RAM. Stores pixels, actions, and privileged simulator state.

Usage: python scripts/data/gen_scene_play.py --num-episodes 1000 --width 224 --height 224 --save-path PATH.h5
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium
import h5py
import numpy as np
from tqdm import trange

import ogbench.manipspace  # noqa: F401 (registers envs)
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle


def collect_episode(env, agents, p_stack, p_random_action):
    """Run one episode (with scene-specific health-check retry), return per-step arrays."""
    while True:
        ob, info = env.reset()
        agent = agents[info["privileged/target_task"]]
        agent.reset(ob, info)

        obs_l, act_l, qpos_l, qvel_l, btn_l = [], [], [], [], []
        done = False
        while not done:
            if np.random.rand() < p_random_action:
                action = env.action_space.sample()
            else:
                action = np.array(agent.select_action(ob, info))
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if agent.done:
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                agent = agents[agent_info["privileged/target_task"]]
                agent.reset(agent_ob, agent_info)

            obs_l.append(ob)
            act_l.append(action)
            qpos_l.append(info["prev_qpos"])
            qvel_l.append(info["prev_qvel"])
            btn_l.append(info["prev_button_states"])
            ob = next_ob

        qpos_arr = np.asarray(qpos_l, dtype=np.float32)
        block_xyzs = qpos_arr[:, 14:17]
        is_healthy = not (
            (block_xyzs[:, 1] >= 0.29).any()
            or ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any()
        )
        if is_healthy:
            return {
                "pixels": np.stack(obs_l).astype(np.uint8),
                "action": np.asarray(act_l, dtype=np.float32),
                "qpos": qpos_arr,
                "qvel": np.asarray(qvel_l, dtype=np.float32),
                "button_states": np.asarray(btn_l, dtype=np.int64),
            }
        print("Unhealthy episode, retrying...", flush=True)


def _append(ds, arr, cur):
    n = arr.shape[0]
    ds.resize(cur + n, axis=0)
    ds[cur:cur + n] = arr
    return n


def main(args):
    env = gymnasium.make(
        args.env_name, width=args.width, height=args.height,
        terminate_at_goal=False, mode="data_collection",
        max_episode_steps=args.max_episode_steps,
    )
    agents = {
        "cube": CubePlanOracle(env=env, noise=args.noise, noise_smoothing=args.noise_smoothing),
        "button": ButtonPlanOracle(env=env, noise=args.noise, noise_smoothing=args.noise_smoothing),
        "drawer": DrawerPlanOracle(env=env, noise=args.noise, noise_smoothing=args.noise_smoothing),
        "window": WindowPlanOracle(env=env, noise=args.noise, noise_smoothing=args.noise_smoothing),
    }

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    H, W = args.height, args.width
    with h5py.File(args.save_path, "w") as f:
        pixels_ds = f.create_dataset("pixels", shape=(0, H, W, 3), maxshape=(None, H, W, 3),
                                      dtype=np.uint8, chunks=(100, H, W, 3), compression="lzf")
        action_ds = f.create_dataset("action", shape=(0, 5), maxshape=(None, 5),
                                      dtype=np.float32, chunks=(1000, 5))
        btn0_ds = f.create_dataset("privileged_button_0_state", shape=(0,), maxshape=(None,),
                                    dtype=np.float32, chunks=(1000,))
        btn1_ds = f.create_dataset("privileged_button_1_state", shape=(0,), maxshape=(None,),
                                    dtype=np.float32, chunks=(1000,))
        qpos_ds = f.create_dataset("privileged_qpos", shape=(0, 25), maxshape=(None, 25),
                                    dtype=np.float32, chunks=(1000, 25))
        qvel_ds = f.create_dataset("privileged_qvel", shape=(0, 24), maxshape=(None, 24),
                                    dtype=np.float32, chunks=(1000, 24))

        ep_offsets, ep_lens = [], []
        cur = 0
        for ep_idx in trange(args.num_episodes):
            ep = collect_episode(env, agents, p_stack=0.5, p_random_action=args.p_random_action)
            n = ep["pixels"].shape[0]
            _append(pixels_ds, ep["pixels"], cur)
            _append(action_ds, ep["action"], cur)
            _append(qpos_ds, ep["qpos"], cur)
            _append(qvel_ds, ep["qvel"], cur)
            _append(btn0_ds, ep["button_states"][:, 0].astype(np.float32), cur)
            _append(btn1_ds, ep["button_states"][:, 1].astype(np.float32), cur)
            ep_offsets.append(cur)
            ep_lens.append(n)
            cur += n
            f.flush()

        f.create_dataset("ep_offset", data=np.asarray(ep_offsets, dtype=np.int64))
        f.create_dataset("ep_len", data=np.asarray(ep_lens, dtype=np.int32))
        f.attrs.update(noise=args.noise, p_random_action=args.p_random_action, seed=args.seed)

    print(f"Wrote {args.save_path}: {cur} steps across {args.num_episodes} episodes "
          f"(resolution {H}x{W})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-name", default="visual-scene-v0")
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--noise-smoothing", type=float, default=0.5)
    ap.add_argument("--p-random-action", type=float, default=0.0)
    ap.add_argument("--num-episodes", type=int, default=1000)
    ap.add_argument("--max-episode-steps", type=int, default=1001)
    ap.add_argument("--save-path", required=True)
    args = ap.parse_args()
    np.random.seed(args.seed)
    main(args)
