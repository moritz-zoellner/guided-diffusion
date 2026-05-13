"""Train a low-level real-world dynamics model.

The model predicts one-step normalized state deltas from
    state_t = concat(configured obs keys)
    action_t = actions[t]

Rotations should enter state as rot6d. If a requested key ends in ``_rot6d`` and
the dataset only contains the matching ``_quat`` key, it is converted on read.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from real_world_data import (
        DEFAULT_ACTION_KEY,
        DEFAULT_STATE_KEYS,
        build_state_from_demo,
        get_demo_keys,
        parse_key_list,
        split_trajectories,
    )
except ModuleNotFoundError:
    from real_world_experiments.real_world_data import (
        DEFAULT_ACTION_KEY,
        DEFAULT_STATE_KEYS,
        build_state_from_demo,
        get_demo_keys,
        parse_key_list,
        split_trajectories,
    )

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def build_dynamics_trajectories(hdf5_path, demo_keys, state_keys, action_key):
    trajectories = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            states_all = build_state_from_demo(demo, state_keys)
            actions_all = demo[action_key][:].astype(np.float32)
            if len(states_all) != len(actions_all):
                raise ValueError(f"{demo_key}: states/actions length mismatch")
            if len(actions_all) < 2:
                continue
            states = states_all[:-1]
            next_states = states_all[1:]
            trajectories.append(
                {
                    "path": str(hdf5_path),
                    "demo_id": demo_key,
                    "states": states,
                    "actions": actions_all[:-1],
                    "next_states": next_states,
                    "deltas": (next_states - states).astype(np.float32),
                }
            )
    if not trajectories:
        raise ValueError("No valid trajectories found.")
    return trajectories


def flatten_trajectories(trajectories):
    keys = ["states", "actions", "next_states", "deltas"]
    return {key: np.concatenate([traj[key] for traj in trajectories], axis=0).astype(np.float32) for key in keys}


def normalization_stats(flat_train):
    return {
        "state_mean": flat_train["states"].mean(axis=0),
        "state_std": flat_train["states"].std(axis=0) + 1e-8,
        "action_mean": flat_train["actions"].mean(axis=0),
        "action_std": flat_train["actions"].std(axis=0) + 1e-8,
        "delta_mean": flat_train["deltas"].mean(axis=0),
        "delta_std": flat_train["deltas"].std(axis=0) + 1e-8,
    }


class DynamicsDataset(Dataset):
    def __init__(self, flat_data, stats):
        self.states = ((flat_data["states"] - stats["state_mean"]) / stats["state_std"]).astype(np.float32)
        self.actions = ((flat_data["actions"] - stats["action_mean"]) / stats["action_std"]).astype(np.float32)
        self.deltas = ((flat_data["deltas"] - stats["delta_mean"]) / stats["delta_std"]).astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.from_numpy(self.states[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "deltas": torch.from_numpy(self.deltas[idx]),
        }


class DynamicsMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, depth=3, dropout=0.0):
        super().__init__()
        layers = []
        current_dim = state_dim + action_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, loss_fn, stats, device):
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    sq_error_sum = 0.0
    zero_sq_error_sum = 0.0
    n_values = 0
    delta_std = torch.as_tensor(stats["delta_std"], device=device, dtype=torch.float32)
    delta_mean = torch.as_tensor(stats["delta_mean"], device=device, dtype=torch.float32)
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            deltas = batch["deltas"].to(device)
            pred = model(states, actions)
            loss = loss_fn(pred, deltas)
            loss_sum += loss.item() * states.shape[0]
            n_samples += states.shape[0]
            raw_error = (pred - deltas) * delta_std
            zero_error = -1.0 * (deltas * delta_std + delta_mean)
            sq_error_sum += torch.square(raw_error).sum().item()
            zero_sq_error_sum += torch.square(zero_error).sum().item()
            n_values += raw_error.numel()
    mse = sq_error_sum / max(1, n_values)
    zero_mse = zero_sq_error_sum / max(1, n_values)
    return {
        "loss": loss_sum / max(1, n_samples),
        "raw_delta_rmse": float(np.sqrt(mse)),
        "zero_delta_rmse": float(np.sqrt(zero_mse)),
    }


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"hd{args.hidden_dim}_depth{args.depth}_lr{args.lr:g}_epochs{args.epochs}_{timestamp}"
    run_dir = args.output_root / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = args.output_root / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_loss_plot(train_losses, val_losses, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", alpha=0.85)
    ax.plot(val_losses, label="Val Loss", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Delta MSE")
    ax.set_title("Real-World Dynamics World Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_dynamics_model_for_eval(model_or_run_path, device="cpu", checkpoint_name="best_model.pt"):
    checkpoint_path = Path(model_or_run_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        loaded = np.load(checkpoint_path.parent / "normalization_stats.npz")
        stats = {key: loaded[key] for key in loaded.files}
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
    model = DynamicsMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stats, checkpoint


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/real_world/cheezit_pouring.hdf5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/real_world/dynamics_world_model"))
    parser.add_argument("--state-keys", default=",".join(DEFAULT_STATE_KEYS))
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument("--split", choices=["mask", "random"], default="mask")
    parser.add_argument("--train-mask", default="train")
    parser.add_argument("--val-mask", default="valid")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    state_keys = parse_key_list(args.state_keys)
    all_keys = get_demo_keys(args.dataset)
    if args.max_demos is not None:
        all_keys = all_keys[: args.max_demos]

    if args.split == "mask":
        train_keys = get_demo_keys(args.dataset, mask=args.train_mask)
        val_keys = get_demo_keys(args.dataset, mask=args.val_mask)
        if args.max_demos is not None:
            allowed = set(all_keys)
            train_keys = [key for key in train_keys if key in allowed]
            val_keys = [key for key in val_keys if key in allowed]
        train_trajectories = build_dynamics_trajectories(args.dataset, train_keys, state_keys, args.action_key)
        val_trajectories = build_dynamics_trajectories(args.dataset, val_keys, state_keys, args.action_key)
    else:
        trajectories = build_dynamics_trajectories(args.dataset, all_keys, state_keys, args.action_key)
        train_trajectories, val_trajectories = split_trajectories(trajectories, args.val_ratio, args.split_seed)

    flat_train = flatten_trajectories(train_trajectories)
    flat_val = flatten_trajectories(val_trajectories)
    stats = normalization_stats(flat_train)
    train_dataset = DynamicsDataset(flat_train, stats)
    val_dataset = DynamicsDataset(flat_val, stats)

    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {
        "state_dim": int(flat_train["states"].shape[-1]),
        "action_dim": int(flat_train["actions"].shape[-1]),
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
    }
    model = DynamicsMLP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    run_dir = make_run_dir(args)
    np.savez_compressed(run_dir / "normalization_stats.npz", **stats)
    train_config = vars(args) | {
        "dataset": str(args.dataset),
        "output_root": str(args.output_root),
        "run_dir": str(run_dir),
        "state_keys": list(state_keys),
        "action_key": args.action_key,
        "rotation_representation": "rot6d",
        "target": "next_state_delta",
        "model_config": model_config,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "device": device,
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str))
    provenance = {
        "train_demos": [{"path": str(traj["path"]), "demo_id": traj["demo_id"]} for traj in train_trajectories],
        "val_demos": [{"path": str(traj["path"]), "demo_id": traj["demo_id"]} for traj in val_trajectories],
    }
    (run_dir / "data_provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"Run directory: {run_dir}")
    print(f"State keys: {state_keys}")
    print(f"State dim: {model_config['state_dim']} | Action dim: {model_config['action_dim']} | Device: {device}")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    log_lines = []
    last_epoch = -1

    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["deltas"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(states, actions)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * states.shape[0]
            train_samples += states.shape[0]

        train_loss = train_loss_sum / max(1, train_samples)
        val_metrics = evaluate(model, val_loader, loss_fn, stats, device)
        val_loss = val_metrics["loss"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        is_best = val_loss < best_val_loss - 1e-6
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "model_config": model_config,
                    "training_config": train_config,
                    "normalization_stats": stats,
                },
                run_dir / "best_model.pt",
            )
        else:
            epochs_since_improvement += 1
        line = (
            f"epoch {epoch + 1:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"raw_rmse {val_metrics['raw_delta_rmse']:.6f} | zero_raw_rmse {val_metrics['zero_delta_rmse']:.6f}"
            f"{' | best' if is_best else ''}"
        )
        print(line, flush=True)
        log_lines.append(line)
        (run_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n")
        if epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    final_metrics = evaluate(model, val_loader, loss_fn, stats, device)
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1] if train_losses else None,
            "val_loss": val_losses[-1] if val_losses else None,
            "val_metrics": final_metrics,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "model_config": model_config,
            "training_config": train_config,
            "normalization_stats": stats,
        },
        run_dir / "final_model.pt",
    )
    (run_dir / "training_stats.json").write_text(
        json.dumps(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            },
            indent=2,
        )
    )
    save_loss_plot(train_losses, val_losses, run_dir / "training_loss_curves.png")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

