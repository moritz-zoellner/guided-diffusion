import h5py
import numpy as np
import os
import random
import torch
import torch.nn as nn
import imageio
import json
import re

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch._higher_order_ops import scan
from tqdm import tqdm

from copy import deepcopy
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import corallab_stl.torch as stl
from corallab_stl.automata import get_spot_formula_and_aps

from .utils.data import (
    get_demo_keys,
    collect_trajectories,
    build_trajectory_dataset,
    split_trajectories,
    compute_normalization_stats,
    DynamicsTransitionDataset,
)
from .world_model_utils import (
    DynamicsMLP,
    quat_to_6d,
    xyzw_to_wxyz_batch,
    build_state_from_obs_dict,
    deconstruct_state,
    load_model_for_eval,
    plot_rzz_world_model_branches,
    predict_next_state_from_raw,
    reconstruct_rotation_matrix_from_6d,
    rollout_policy_for_rzz_analysis,
    rzz_from_state_29d,
)




def add_labels(trajectories, predicates):

    def labeling_func(s):
        return torch.stack([p({"state": s}) >= 0.0 for p in predicates], axis=-1)

    def overwrite_with_next_label(c, x):
        previous, carry = c

        def write_carry():
            return (x.clone(), carry.clone()), carry.clone()

        def update_carry():
            return (x.clone(), previous.clone()), previous.clone()

        # if x == previous
        return torch.cond((x == previous).all(),
                          # overwrite x with next label (carry)
                          write_carry,
                          # update carry to be previous
                          # overwrite x with next label (carry)
                          update_carry)

    for traj in trajectories:
        states = torch.tensor(traj["states"])
        next_states = torch.tensor(traj["next_states"])

        labels = labeling_func(states)
        immediate_next_labels = labeling_func(next_states)
        _, next_labels = scan(overwrite_with_next_label, (immediate_next_labels[-2], immediate_next_labels[-1]), labels, reverse=True)

        traj["labels"] = labels.float()
        traj["next_labels"] = next_labels.float()

    return trajectories
    

def main():
    dataset_path = "/home/shared/data/toy_squares/train/data.hdf5"
    demos = [(dataset_path, k) for k in get_demo_keys(dataset_path)]
    trajectories = build_trajectory_dataset(demos, state_components=[
        lambda s: s["agent_pos"][:],
        lambda s: s["states"][:], # blue, red, green, yellow
    ])

    # labels ##################################################################

    state_var = stl.Var("state", dim=10,
                        agent_pos=(0, 2),
                        blue_pos=(2, 4),
                        red_pos=(4, 6),
                        green_pos=(6, 8),
                        yellow_pos=(8, 10))

    cube_radius = 0.2
    at_blue = stl.Predicate(state_var, lambda state, _: cube_radius - (state[0:2] - state[2:4]).norm(), 0.0)
    at_red = stl.Predicate(state_var, lambda state, _: cube_radius - (state[0:2] - state[4:6]).norm(), 0.0)
    at_green = stl.Predicate(state_var, lambda state, _: cube_radius - (state[0:2] - state[6:8]).norm(), 0.0)
    at_yellow = stl.Predicate(state_var, lambda state, _: cube_radius - (state[0:2] - state[8:10]).norm(), 0.0)

    trajectories = add_labels(trajectories, [at_green, at_blue, at_red, at_yellow])

    train_trajectories, val_trajectories = split_trajectories(
        trajectories,
        val_ratio=0.2,
        seed=42,
    )

    keys = ["states", "actions", "next_states", "deltas", "labels", "next_labels"]

    # IMPORTANT: normalize using train split only
    stats = compute_normalization_stats(train_trajectories, keys=keys)
    train_dataset = DynamicsTransitionDataset(train_trajectories, stats, keys=keys, normalization_blacklist=["labels", "next_labels"])
    val_dataset = DynamicsTransitionDataset(val_trajectories, stats, keys=keys, normalization_blacklist=["labels", "next_labels"])



if __name__ == "__main__":
    main()
