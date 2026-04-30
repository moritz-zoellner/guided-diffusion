"""
Convert native CALVIN npz episodes to the robomimic HDF5 layout used by DynaGuide.

The base diffusion policy does not use behavior labels, but the output schema matches
DynaGuide's CALVIN HDF5 files:

    data/demo_i/actions
    data/demo_i/obs/eye_in_hand
    data/demo_i/obs/third_person
    data/demo_i/obs/proprio
    data/demo_i/obs/states

Actions are CALVIN rel_actions clipped to [-1, 1], matching DynaGuide.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


def _episode_index(path):
    return int(path.stem.split("_")[-1])


def _load_env_meta(calvin_split_dir):
    hydra_config = Path(calvin_split_dir) / ".hydra" / "merged_config.yaml"
    if not hydra_config.exists():
        raise FileNotFoundError(f"missing CALVIN hydra config: {hydra_config}")

    config = OmegaConf.load(hydra_config)
    env_kwargs = OmegaConf.to_container(config.env, resolve=True)
    env_kwargs.pop("_target_", None)
    env_kwargs.pop("_recursive_", None)

    return {
        "env_name": "PlayTableSimEnv",
        "type": 2,
        "env_kwargs": env_kwargs,
    }


def _load_episode_windows(calvin_split_dir, chunk_length=None):
    calvin_split_dir = Path(calvin_split_dir)
    episode_files = sorted(calvin_split_dir.glob("episode_*.npz"), key=_episode_index)
    if not episode_files:
        raise FileNotFoundError(f"no episode_*.npz files found under {calvin_split_dir}")

    episode_ids = {_episode_index(path) for path in episode_files}
    windows_path = calvin_split_dir / "ep_start_end_ids.npy"
    if windows_path.exists():
        windows = np.load(windows_path)
    else:
        ids = sorted(episode_ids)
        windows = np.array([[ids[0], ids[-1]]], dtype=np.int64)

    demo_windows = []
    for start, end in windows:
        start = int(start)
        end = int(end)
        current = start
        while current <= end:
            chunk_end = end if chunk_length is None else min(end, current + int(chunk_length) - 1)
            ids = [idx for idx in range(current, chunk_end + 1) if idx in episode_ids]
            if ids:
                demo_windows.append(ids)
            current = chunk_end + 1

    if not demo_windows:
        raise RuntimeError(f"no valid demo windows found under {calvin_split_dir}")
    return demo_windows


def _read_step(calvin_split_dir, episode_id):
    return np.load(Path(calvin_split_dir) / f"episode_{episode_id:07d}.npz")


def convert_calvin_to_hdf5(calvin_split_dir, output, chunk_length=None):
    calvin_split_dir = Path(calvin_split_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env_meta = _load_env_meta(calvin_split_dir)
    demo_windows = _load_episode_windows(calvin_split_dir, chunk_length=chunk_length)

    total_samples = 0
    with h5py.File(output, "w") as h5:
        data_group = h5.create_group("data")
        data_group.attrs["env_args"] = json.dumps(env_meta)

        for demo_idx, episode_ids in enumerate(tqdm(demo_windows, desc="writing demos")):
            actions = []
            eye_in_hand = []
            third_person = []
            proprio = []
            states = []

            for episode_id in episode_ids:
                step = _read_step(calvin_split_dir, episode_id)
                actions.append(np.clip(step["rel_actions"], -1.0, 1.0).astype(np.float32))
                eye_in_hand.append(step["rgb_gripper"])
                third_person.append(step["rgb_static"])
                proprio.append(step["robot_obs"].astype(np.float32))
                states.append(step["scene_obs"].astype(np.float32))

            demo = data_group.create_group(f"demo_{demo_idx}")
            demo.attrs["num_samples"] = len(actions)
            demo.attrs["behavior"] = "calvin_episode"
            demo.create_dataset("actions", data=np.stack(actions, axis=0), compression="gzip")
            demo.create_dataset("obs/eye_in_hand", data=np.stack(eye_in_hand, axis=0), compression="gzip")
            demo.create_dataset("obs/third_person", data=np.stack(third_person, axis=0), compression="gzip")
            demo.create_dataset("obs/proprio", data=np.stack(proprio, axis=0), compression="gzip")
            demo.create_dataset("obs/states", data=np.stack(states, axis=0), compression="gzip")
            demo.create_dataset("rewards", data=np.zeros(len(actions), dtype=np.float32))
            dones = np.zeros(len(actions), dtype=np.float32)
            dones[-1] = 1.0
            demo.create_dataset("dones", data=dones)
            total_samples += len(actions)

        data_group.attrs["total"] = total_samples

    print(f"wrote {len(demo_windows)} demos / {total_samples} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CALVIN split directory containing episode_*.npz")
    parser.add_argument("--output", required=True, help="output robomimic HDF5 path")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="optional max timesteps per demo; useful for splitting the single debug trajectory",
    )
    args = parser.parse_args()

    convert_calvin_to_hdf5(args.input, args.output, chunk_length=args.chunk_length)
