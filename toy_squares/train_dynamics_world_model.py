"""Train a low-level Toy Squares dynamics model.

This mirrors the CALVIN dynamics trainer, but the state is intentionally small:

    state_t = concat(obs/agent_pos, obs/states)
    action_t = actions[t]
    target = state_{t+1} - state_t

The model never uses images. It predicts one-step normalized low-level state
deltas from low-level state and raw environment action.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt


STATE_COMPONENTS = ("agent_pos", "states")
ACTION_KEY = "actions"


def demo_sort_key(name: str) -> int:
    match = re.search(r"\d+", name)
    return int(match.group()) if match else 0


def get_demo_keys(hdf5_path: Path, mask: str | None = None) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        if mask is None:
            keys = list(f["data"].keys())
        else:
            keys = [key.decode("utf-8") for key in np.asarray(f[f"mask/{mask}"])]
    return sorted(keys, key=demo_sort_key)


def build_low_level_state(obs_group) -> np.ndarray:
    components = [obs_group[key][:].astype(np.float32) for key in STATE_COMPONENTS]
    return np.concatenate(components, axis=-1).astype(np.float32)


def build_toy_dynamics_trajectories(hdf5_path: Path, demo_keys: list[str]) -> list[dict]:
    trajectories = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            states = build_low_level_state(demo["obs"])
            next_states = build_low_level_state(demo["next_obs"])
            actions = demo[ACTION_KEY][:].astype(np.float32)

            if not (len(states) == len(next_states) == len(actions)):
                raise ValueError(
                    f"{demo_key}: expected states, next_states, actions to align; "
                    f"got states={len(states)}, next_states={len(next_states)}, actions={len(actions)}"
                )
            if len(actions) < 2:
                continue

            trajectories.append(
                {
                    "path": str(hdf5_path),
                    "demo_id": demo_key,
                    "states": states,
                    "actions": actions,
                    "next_states": next_states,
                    "deltas": (next_states - states).astype(np.float32),
                }
            )

    if not trajectories:
        raise ValueError("No valid trajectories found.")
    return trajectories


def split_trajectories(trajectories: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if len(trajectories) < 2:
        raise ValueError("Need at least 2 trajectories for a train/val split.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(trajectories))
    n_val = max(1, int(round(len(trajectories) * val_ratio)))
    n_val = min(len(trajectories) - 1, n_val)
    val_indices = set(int(idx) for idx in permutation[:n_val])
    train = [traj for idx, traj in enumerate(trajectories) if idx not in val_indices]
    val = [traj for idx, traj in enumerate(trajectories) if idx in val_indices]
    return train, val


def flatten_trajectories(trajectories: list[dict]) -> dict[str, np.ndarray]:
    keys = ["states", "actions", "next_states", "deltas"]
    return {key: np.concatenate([traj[key] for traj in trajectories], axis=0).astype(np.float32) for key in keys}


def normalization_stats(flat_train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "state_mean": flat_train["states"].mean(axis=0),
        "state_std": flat_train["states"].std(axis=0) + 1e-8,
        "action_mean": flat_train["actions"].mean(axis=0),
        "action_std": flat_train["actions"].std(axis=0) + 1e-8,
        "delta_mean": flat_train["deltas"].mean(axis=0),
        "delta_std": flat_train["deltas"].std(axis=0) + 1e-8,
    }


class ToyDynamicsDataset(Dataset):
    def __init__(self, flat_data: dict[str, np.ndarray], stats: dict[str, np.ndarray]):
        self.states = ((flat_data["states"] - stats["state_mean"]) / stats["state_std"]).astype(np.float32)
        self.actions = ((flat_data["actions"] - stats["action_mean"]) / stats["action_std"]).astype(np.float32)
        self.deltas = ((flat_data["deltas"] - stats["delta_mean"]) / stats["delta_std"]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.from_numpy(self.states[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "deltas": torch.from_numpy(self.deltas[idx]),
        }


class DynamicsMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        current_dim = state_dim + action_dim
        for _ in range(int(depth)):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, stats: dict[str, np.ndarray], device: str) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    sq_error_sum = 0.0
    abs_error_sum = 0.0
    zero_sq_error_sum = 0.0
    zero_abs_error_sum = 0.0
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
            zero_error = -(deltas * delta_std + delta_mean)
            sq_error_sum += torch.square(raw_error).sum().item()
            abs_error_sum += torch.abs(raw_error).sum().item()
            zero_sq_error_sum += torch.square(zero_error).sum().item()
            zero_abs_error_sum += torch.abs(zero_error).sum().item()
            n_values += raw_error.numel()

    mse = sq_error_sum / max(1, n_values)
    zero_mse = zero_sq_error_sum / max(1, n_values)
    return {
        "loss": loss_sum / max(1, n_samples),
        "raw_delta_mse": mse,
        "raw_delta_rmse": float(np.sqrt(mse)),
        "raw_delta_mae": abs_error_sum / max(1, n_values),
        "zero_delta_mse": zero_mse,
        "zero_delta_rmse": float(np.sqrt(zero_mse)),
        "zero_delta_mae": zero_abs_error_sum / max(1, n_values),
    }


def make_run_dir(args) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"hd{args.hidden_dim}_depth{args.depth}_drop{args.dropout:g}_"
        f"lr{args.lr:g}_epochs{args.epochs}_{timestamp}"
    )
    run_dir = args.output_root / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = args.output_root / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_loss_plot(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="train", alpha=0.85)
    ax.plot(val_losses, label="val", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized delta MSE")
    ax.set_title("Toy Squares Dynamics World Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def resolve_checkpoint_path(model_or_run_path, checkpoint_name: str = "best_model.pt") -> Path:
    model_or_run_path = Path(model_or_run_path)
    if model_or_run_path.is_dir():
        return model_or_run_path / checkpoint_name
    return model_or_run_path


def load_dynamics_model_for_eval(model_or_run_path, device: str = "cpu", checkpoint_name: str = "best_model.pt"):
    checkpoint_path = resolve_checkpoint_path(model_or_run_path, checkpoint_name=checkpoint_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        stats_path = checkpoint_path.parent / "normalization_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
        loaded = np.load(stats_path)
        stats = {key: loaded[key] for key in loaded.files}
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}

    model = DynamicsMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    meta = {"checkpoint_path": str(checkpoint_path), "run_dir": str(checkpoint_path.parent)}
    return model, stats, checkpoint, meta


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/toy_squares_data/two_block_train/data.hdf5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/toy_squares/dynamics_world_model"))
    parser.add_argument("--split", choices=["mask", "random"], default="mask")
    parser.add_argument("--train-mask", default="train")
    parser.add_argument("--val-mask", default="valid")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

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
        train_trajectories = build_toy_dynamics_trajectories(args.dataset, train_keys)
        val_trajectories = build_toy_dynamics_trajectories(args.dataset, val_keys)
    else:
        trajectories = build_toy_dynamics_trajectories(args.dataset, all_keys)
        train_trajectories, val_trajectories = split_trajectories(trajectories, args.val_ratio, args.split_seed)

    flat_train = flatten_trajectories(train_trajectories)
    flat_val = flatten_trajectories(val_trajectories)
    stats = normalization_stats(flat_train)

    train_dataset = ToyDynamicsDataset(flat_train, stats)
    val_dataset = ToyDynamicsDataset(flat_val, stats)
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )

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
        "state_components": list(STATE_COMPONENTS),
        "action_key": ACTION_KEY,
        "target": "next_low_level_state_delta",
        "model_config": model_config,
        "optimizer": "Adam",
        "loss_fn": "MSELoss",
        "num_parameters": int(sum(param.numel() for param in model.parameters())),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "device": device,
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str))
    (run_dir / "data_provenance.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "split_seed": args.split_seed,
                "split": args.split,
                "dataset": str(args.dataset),
                "train_demos": [{"path": traj["path"], "demo_id": traj["demo_id"]} for traj in train_trajectories],
                "val_demos": [{"path": traj["path"], "demo_id": traj["demo_id"]} for traj in val_trajectories],
            },
            indent=2,
        )
    )

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Train trajectories: {len(train_trajectories):,} | Val trajectories: {len(val_trajectories):,}", flush=True)
    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}", flush=True)
    print(f"State dim: {model_config['state_dim']} | Action dim: {model_config['action_dim']} | Device: {device}", flush=True)

    train_losses = []
    val_losses = []
    val_raw_delta_rmses = []
    val_zero_delta_rmses = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    last_epoch = -1
    log_lines = []

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
        val_raw_delta_rmses.append(val_metrics["raw_delta_rmse"])
        val_zero_delta_rmses.append(val_metrics["zero_delta_rmse"])

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
            f"raw_rmse {val_metrics['raw_delta_rmse']:.6f} | "
            f"zero_raw_rmse {val_metrics['zero_delta_rmse']:.6f}"
            f"{' | best' if is_best else ''}"
        )
        print(line, flush=True)
        log_lines.append(line)
        (run_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n")

        if epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}.", flush=True)
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
    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "num_epochs_trained": len(train_losses),
        "best_raw_delta_rmse": val_raw_delta_rmses[best_epoch] if best_epoch >= 0 else None,
        "best_zero_delta_rmse": val_zero_delta_rmses[best_epoch] if best_epoch >= 0 else None,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir),
    }
    (run_dir / "training_stats.json").write_text(json.dumps(summary, indent=2))
    save_loss_plot(train_losses, val_losses, run_dir / "training_loss_curves.png")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}", flush=True)
    print(f"Training stats: {run_dir / 'training_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
