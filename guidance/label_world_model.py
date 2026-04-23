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

import mlflow

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
from .utils.mlflow import (
    run_artifact_path
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


###############################################################################
#                                     data                                    #
###############################################################################

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


###############################################################################
#                                    model                                    #
###############################################################################



###############################################################################
#                                     main                                    #
###############################################################################

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

    split_seed = 42
    train_trajectories, val_trajectories = split_trajectories(
        trajectories,
        val_ratio=0.2,
        seed=split_seed,
    )

    keys = ["states", "actions", "next_states", "deltas", "labels", "next_labels"]

    # IMPORTANT: normalize using train split only
    stats = compute_normalization_stats(train_trajectories, keys=keys)

    train_dataset = DynamicsTransitionDataset(train_trajectories, stats, keys=keys, normalization_blacklist=["labels", "next_labels"])
    val_dataset = DynamicsTransitionDataset(val_trajectories, stats, keys=keys, normalization_blacklist=["labels", "next_labels"])

    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader setup for deterministic shuffling
    batch_size = 256
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Instantiate model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DynamicsMLP(state_dim=10, action_dim=2, hidden_dim=256)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Training configuration
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 50  # early stopping
    epochs_since_improvement = 0


    with mlflow.start_run() as run:

        norm_stats_path = None
        provenance_path = None

        # Persist normalization stats needed for inference-time denormalization
        with run_artifact_path(run, "normalization_stats.npz") as path:
            norm_stats_path = path
            np.savez_compressed(norm_stats_path, **stats)

        # Persist dataset provenance for run reproducibility
        with run_artifact_path(run, "data_provenance.json") as path:
            selected_demo_refs = [{"path": p, "demo_id": d} for p, d in demos]
            train_demo_refs = [{"path": tr["path"], "demo_id": tr["demo_id"]} for tr in train_trajectories]
            val_demo_refs = [{"path": tr["path"], "demo_id": tr["demo_id"]} for tr in val_trajectories]

            provenance = {
                "seed": seed,
                "source_paths": [
                    dataset_path,
                ],
                "selection": {
                    "machine_percent": 0,
                    "selected_demos": selected_demo_refs,
                },
                "split": {
                    "val_ratio": 0.2,
                    "split_seed": split_seed,
                    "train_demos": train_demo_refs,
                    "val_demos": val_demo_refs,
                },
            }

            provenance_path = path
            with open(provenance_path, "w") as f:
                json.dump(provenance, f, indent=2)

        # Save config
        training_config = {
            # "run_name_requested": run_name,
            # "run_dir": run_dir,
            "seed": seed,
            "state_dim": 29,
            "action_dim": 7,
            "hidden_dim": 256,
            "activation": "SiLU",
            "num_epochs": num_epochs,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "loss_fn": "MSELoss",
            "patience": patience,
            "num_train_transitions": len(train_dataset),
            "num_val_transitions": len(val_dataset),
            # "checkpoint_path": checkpoint_path,
            "normalization_stats_path": norm_stats_path,
            "data_provenance_path": provenance_path,
        }
        mlflow.log_params(training_config)

        print(f"Normalization stats saved to {norm_stats_path}")
        print(f"Data provenance saved to {provenance_path}")

        # Training loop

        train_losses = []
        val_losses = []
        best_epoch = -1

        for epoch in range(num_epochs):

            # ===== Train epoch =====
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for batch in train_loader:
                s_t = batch["states"].to(device)
                a_t = batch["actions"].to(device)
                delta_t = batch["deltas"].to(device)

                optimizer.zero_grad()
                delta_t_pred = model(s_t, a_t)
                loss = loss_fn(delta_t_pred, delta_t)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            train_loss = train_loss_sum / max(1, train_batches)
            train_losses.append(train_loss)

            # ===== Val epoch =====
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    s_t = batch["states"].to(device)
                    a_t = batch["actions"].to(device)
                    delta_t = batch["deltas"].to(device)

                    delta_t_pred = model(s_t, a_t)
                    loss = loss_fn(delta_t_pred, delta_t)

                    val_loss_sum += loss.item()
                    val_batches += 1

            val_loss = val_loss_sum / max(1, val_batches)
            val_losses.append(val_loss)

            # ===== Checkpoint best model =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_since_improvement = 0

                with run_artifact_path(run, "best_model.pt") as path:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "model_config": {
                                "state_dim": 29,
                                "action_dim": 7,
                                "hidden_dim": 256,
                                "activation": "SiLU",
                            },
                            "normalization_stats": stats,
                        },
                        path,
                    )
                print(f"Epoch {epoch+1:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f} | BEST")
            else:
                epochs_since_improvement += 1
                print(f"Epoch {epoch+1:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

            mlflow.log_metrics({
                "train loss": train_loss,
                "val loss": val_loss
            }, epoch)
                
            # Early stopping
            if epochs_since_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:////home/tassos/.local/share/mlflow/runs.db")
    if os.getenv("MLFLOW_EXPERIMENT_NAME") is None:
        mlflow.set_experiment("guided-diffusion")

    main()
