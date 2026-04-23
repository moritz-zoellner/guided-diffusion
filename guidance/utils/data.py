import h5py
import numpy as np
import os
import random
import torch
import torch.nn as nn
import imageio
import json
import re
from functools import reduce

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


###############################################################################
#                                demo selection                               #
###############################################################################

def get_demo_keys(path):
    with h5py.File(path, 'r') as f:
        keys = list(f['data'].keys())
    # Numerical sort: 'demo_9' comes before 'demo_10'
    return sorted(keys, key=lambda x: int(re.search(r'\d+', x).group()))


def collect_trajectories(path, machine_percent=0.5, min_machine_percent=0.05):
    """
    path1, path2: human / expert datasets
    path3: machine-generated / mixed dataset

    If machine_percent is very small (< min_machine_percent) or <= 0,
    no machine demos are included.

    Returns:
        list of (path, demo_key)
    """
    human_demos = [(path1, k) for k in get_demo_keys(path1)]
    human_demos += [(path2, k) for k in get_demo_keys(path2)]
    num_human = len(human_demos)

    if machine_percent <= 0 or machine_percent < min_machine_percent:
        mg_count_needed = 0
    else:
        total_needed = int(num_human / (1 - machine_percent))
        mg_count_needed = max(0, total_needed - num_human)

        mg_all_keys = get_demo_keys(path3)
        mg_keys_selected = mg_all_keys[-mg_count_needed:] if mg_count_needed > 0 else []
        mg_demos = [(path3, k) for k in mg_keys_selected]

        all_demos = human_demos + mg_demos

    all_demos = human_demos

    print("Dataset Summary:")
    print(f" - Human demos: {num_human}")
    print(f" - Machine demos selected: {mg_count_needed}")
    print(f" - Total trajectories: {len(all_demos)}")
    return all_demos


###############################################################################
#                loading demos and building trajectory dataset                #
###############################################################################

def build_state_traj(obs_group, state_components=[]):
    """
    Builds verified 29D state trajectory from robomimic obs group.

    State layout:
      [ p_can_to_eef(3),
        q_can_to_eef_6d(6),
        p_can(3),
        q_can_6d(6),
        p_eef(3),
        q_eef_6d(6),
        g_pos(2?) ]

    For CanLift this should sum to 29.
    """
    # Relative can->eef
    # p_can_to_eef = obs_group["object"][:, 0:3]
    # q_can_to_eef = quat_to_6d(xyzw_to_wxyz_batch(obs_group["object"][:, 3:7]))

    # # Absolute can
    # p_can = obs_group["object"][:, 7:10]
    # q_can = quat_to_6d(xyzw_to_wxyz_batch(obs_group["object"][:, 10:14]))

    # # Absolute eef
    # p_eef = obs_group["robot0_eef_pos"][:]
    # q_eef = quat_to_6d(xyzw_to_wxyz_batch(obs_group["robot0_eef_quat"][:]))

    # # Gripper
    # g_pos = obs_group["robot0_gripper_qpos"][:]

    # s_traj = np.concatenate(
    #     [p_can_to_eef, q_can_to_eef, p_can, q_can, p_eef, q_eef, g_pos],
    #     axis=-1,
    # )

    components = []
    for f in state_components:
        components.append(f(obs_group))

    s_traj = np.concatenate(components, axis=-1)

    return s_traj.astype(np.float32)


def build_trajectory_dataset(selected_demos, state_components=[]):
    """
    Main dataset structure for training stages.

    Returns:
        trajectories: list of dicts
            each dict has:
              - path
              - demo_id
              - states:      (T, state_dim)
              - actions:     (T, action_dim)
              - next_states: (T, state_dim)
              - deltas:      (T, state_dim), where deltas[t] = next_states[t] - states[t]
    """
    trajectories = []

    for path, d_id in selected_demos:
        with h5py.File(path, "r") as f:
            obs = f[f"data/{d_id}/obs"]
            next_obs = f[f"data/{d_id}/next_obs"]
            acts = f[f"data/{d_id}/actions"][:].astype(np.float32)

            s_traj = build_state_traj(obs, state_components=state_components)   # shape (T, state_dim)
            s_next_traj = build_state_traj(next_obs, state_components=state_components)
            delta_traj = (s_next_traj - s_traj).astype(np.float32)

            # In robomimic low-dim data, obs, next_obs, actions are aligned by index t.
            T_states = s_traj.shape[0]
            T_next_states = s_next_traj.shape[0]
            T_actions = acts.shape[0]

            if not (T_actions == T_states == T_next_states):
                raise ValueError(
                    f"{path}::{d_id}: expected actions = states = next_states, "
                    f"got states={T_states}, actions={T_actions}, next_states={T_next_states}"
                )

            trajectories.append({
                "path": path,
                "demo_id": d_id,
                "states": s_traj,
                "actions": acts,
                "next_states": s_next_traj,
                "deltas": delta_traj,
            })

    # Summary
    n_traj = len(trajectories)
    n_trans = sum(tr["actions"].shape[0] for tr in trajectories)
    state_dim = trajectories[0]["states"].shape[1]
    action_dim = trajectories[0]["actions"].shape[1]

    print("Trajectory Dataset Summary:")
    print(f" - Num trajectories: {n_traj}")
    print(f" - Num transitions:  {n_trans}")
    print(f" - State dim:        {state_dim}")
    print(f" - Action dim:       {action_dim}")

    return trajectories


def split_trajectories(trajectories, val_ratio=0.2, seed=42):
    """
    Splits full trajectories into train / val sets.
    """
    n = len(trajectories)
    if n < 2:
        raise ValueError("Need at least 2 trajectories to create a train/val split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n - 1, n_val)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_traj = [trajectories[i] for i in train_idx]
    val_traj = [trajectories[i] for i in val_idx]

    print("Split Summary:")
    print(f" - Total trajectories: {n}")
    print(f" - Train trajectories: {len(train_traj)}")
    print(f" - Val trajectories:   {len(val_traj)}")

    return train_traj, val_traj


def flatten_trajectory_dataset(trajectories, keys=[]):
    """
    Useful for normalization stats and one-step supervised training.

    Returns concatenated arrays over trajectory dimension.
    """
    flat_selected_data = {
        k: np.concatenate([tr[k] for tr in trajectories], axis=0)
        for k in keys
    }

    # print("Flat Dataset Summary:")
    # print(f" - S:      {S.shape}")
    # print(f" - A:      {A.shape}")
    # print(f" - S_next: {S_next.shape}")
    # print(f" - D:      {D.shape}")

    # return (
    #     S.astype(np.float32),
    #     A.astype(np.float32),
    #     S_next.astype(np.float32),
    #     D.astype(np.float32),
    # )

    return flat_selected_data


###############################################################################
#                                normalization                                #
###############################################################################

def compute_normalization_stats(trajectories, keys=[]):
    """
    Computes global mean/std over transitions for a given trajectory split.

    For Option B training, delta stats are the target normalization stats.
    """
    flat_selected_data = flatten_trajectory_dataset(trajectories, keys=keys)

    stats_list = [{
        f"{k}_mean": v.mean(axis=0),
        f"{k}_std": v.std(axis=0) + 1e-8
    } for k, v in flat_selected_data.items()]

    stats = reduce(lambda x, y: x | y, stats_list)

    return stats


class DynamicsTransitionDataset(Dataset):
    """
    One-step dynamics dataset for Option B.

    __getitem__ returns normalized (s_t, a_t, delta_t).
    """
    def __init__(self, trajectories, stats, keys=[], normalization_blacklist=[]):
        self.keys = keys
        self.normalization_blacklist = normalization_blacklist
        self.flat_data = flatten_trajectory_dataset(trajectories, keys=keys)
        self.stats = stats

    def __len__(self):
        return self.flat_data["states"].shape[0]

    def __getitem__(self, idx):
        unnorm_data = {
            k: self.flat_data[k][idx] for k in self.keys
        }

        def normalize(k, x):
            if k not in self.normalization_blacklist:
                return (x - self.stats[f"{k}_mean"]) / self.stats[f"{k}_std"]
            else:
                return x

        norm_data = {
            k: torch.from_numpy(normalize(k, v)).float() for k, v in unnorm_data.items()
        }

        return norm_data
