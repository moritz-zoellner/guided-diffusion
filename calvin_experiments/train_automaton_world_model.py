import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from label_calvin_world_model import LABEL_NAMES, label_scene_states, next_changed_labels


matplotlib.use("Agg")
from matplotlib import pyplot as plt


def get_demo_keys(hdf5_path, mask=None):
    with h5py.File(hdf5_path, "r") as f:
        if mask is None:
            keys = list(f["data"].keys())
        else:
            keys = [key.decode("utf-8") for key in np.asarray(f[f"mask/{mask}"])]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def build_calvin_trajectories(hdf5_path, demo_keys):
    trajectories = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            robot_obs = demo["obs/proprio"][:].astype(np.float32)
            scene_obs = demo["obs/states"][:].astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)

            if not (len(robot_obs) == len(scene_obs) == len(actions)):
                raise ValueError(
                    f"{demo_key}: expected robot_obs, scene_obs, and actions to align, "
                    f"got {len(robot_obs)}, {len(scene_obs)}, {len(actions)}"
                )
            if len(actions) < 2:
                continue

            labels_all = label_scene_states(scene_obs)
            states_all = np.concatenate([robot_obs, scene_obs], axis=-1)

            labels = labels_all[:-1]
            immediate_next_labels = labels_all[1:]
            next_labels = next_changed_labels(labels, immediate_next_labels)
            states = states_all[:-1]
            actions = actions[:-1]

            trajectories.append(
                {
                    "path": str(hdf5_path),
                    "demo_id": demo_key,
                    "states": states,
                    "actions": actions,
                    "labels": labels,
                    "next_labels": next_labels,
                }
            )

    return trajectories


def split_trajectories(trajectories, val_ratio, seed):
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


def flatten_trajectories(trajectories, action_horizon):
    flat = {key: [] for key in ["states", "actions", "labels", "next_labels"]}

    for traj in trajectories:
        n_steps = len(traj["actions"])
        n_chunks = n_steps - action_horizon
        if n_chunks <= 0:
            continue

        action_chunks = np.stack(
            [traj["actions"][idx : idx + action_horizon] for idx in range(n_chunks)],
            axis=0,
        ).reshape(n_chunks, -1)

        flat["states"].append(traj["states"][:-action_horizon])
        flat["actions"].append(action_chunks)
        flat["labels"].append(traj["labels"][:-action_horizon])
        flat["next_labels"].append(traj["next_labels"][action_horizon:])

    if not flat["states"]:
        raise ValueError(f"No trajectories were long enough for action_horizon={action_horizon}.")

    return {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in flat.items()}


def normalization_stats(flat_train):
    return {
        "states_mean": flat_train["states"].mean(axis=0),
        "states_std": flat_train["states"].std(axis=0) + 1e-8,
        "actions_mean": flat_train["actions"].mean(axis=0),
        "actions_std": flat_train["actions"].std(axis=0) + 1e-8,
    }


class AutomatonDataset(Dataset):
    def __init__(self, flat_data, stats):
        self.states = ((flat_data["states"] - stats["states_mean"]) / stats["states_std"]).astype(np.float32)
        self.actions = ((flat_data["actions"] - stats["actions_mean"]) / stats["actions_std"]).astype(np.float32)
        self.labels = flat_data["labels"].astype(np.float32)
        self.next_labels = flat_data["next_labels"].astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.from_numpy(self.states[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "next_labels": torch.from_numpy(self.next_labels[idx]),
        }


def make_mlp(in_dim, hidden_dim, depth, dropout):
    layers = []
    current_dim = in_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class AutomatonMLP(nn.Module):
    def __init__(
        self,
        state_dim,
        label_dim,
        action_chunk_dim,
        state_hidden,
        label_hidden,
        action_hidden,
        head_hidden,
        state_depth=2,
        action_depth=2,
        label_depth=1,
        head_depth=2,
        dropout=0.05,
    ):
        super().__init__()
        self.state_enc = make_mlp(state_dim, state_hidden, state_depth, dropout)
        self.label_enc = make_mlp(label_dim, label_hidden, label_depth, dropout)
        self.action_enc = make_mlp(action_chunk_dim, action_hidden, action_depth, dropout)

        head_layers = []
        current_dim = state_hidden + label_hidden + action_hidden
        for _ in range(head_depth):
            head_layers.extend([nn.Linear(current_dim, head_hidden), nn.SiLU()])
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            current_dim = head_hidden
        head_layers.append(nn.Linear(current_dim, label_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, states, actions, labels):
        z = torch.cat([self.state_enc(states), self.action_enc(actions), self.label_enc(labels)], dim=-1)
        return self.head(z)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_counts(array):
    counts = Counter(tuple(int(x) for x in row) for row in array.astype(int))
    return {str(label): int(count) for label, count in counts.items()}


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    bit_correct = 0
    exact_correct = 0
    bit_count = 0

    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            labels = batch["labels"].to(device)
            targets = batch["next_labels"].to(device)

            logits = model(states, actions, labels)
            loss = loss_fn(logits, targets)
            batch_size = states.shape[0]
            loss_sum += loss.item() * batch_size
            n_samples += batch_size

            preds = (torch.sigmoid(logits) >= 0.5).float()
            bit_correct += (preds == targets).sum().item()
            bit_count += targets.numel()
            exact_correct += (preds == targets).all(dim=-1).sum().item()

    return {
        "loss": loss_sum / max(1, n_samples),
        "bit_acc": bit_correct / max(1, bit_count),
        "exact_acc": exact_correct / max(1, n_samples),
    }


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"h{args.action_horizon}_sh{args.state_hidden}_ah{args.action_hidden}_"
        f"lh{args.label_hidden}_hh{args.head_hidden}_lr{args.lr:g}_"
        f"epochs{args.epochs}_{timestamp}"
    )
    run_dir = args.output_root / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = args.output_root / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_loss_plot(train_losses, val_losses, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", alpha=0.8)
    ax.plot(val_losses, label="Val Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogits Loss")
    ax.set_title("CALVIN Automaton Model Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("calvin/dataset/calvin_D_training.hdf5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/calvin/automaton_world_model"))
    parser.add_argument("--split", choices=["random", "mask"], default="random")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-mask", default="train")
    parser.add_argument("--val-mask", default="valid")
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--state-hidden", type=int, default=64)
    parser.add_argument("--action-hidden", type=int, default=96)
    parser.add_argument("--label-hidden", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
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
        train_trajectories = build_calvin_trajectories(args.dataset, train_keys)
        val_trajectories = build_calvin_trajectories(args.dataset, val_keys)
    else:
        trajectories = build_calvin_trajectories(args.dataset, all_keys)
        train_trajectories, val_trajectories = split_trajectories(trajectories, args.val_ratio, args.split_seed)

    flat_train = flatten_trajectories(train_trajectories, args.action_horizon)
    flat_val = flatten_trajectories(val_trajectories, args.action_horizon)
    stats = normalization_stats(flat_train)

    train_dataset = AutomatonDataset(flat_train, stats)
    val_dataset = AutomatonDataset(flat_val, stats)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {
        "state_dim": int(flat_train["states"].shape[-1]),
        "label_dim": len(LABEL_NAMES),
        "action_chunk_dim": int(flat_train["actions"].shape[-1]),
        "state_hidden": args.state_hidden,
        "label_hidden": args.label_hidden,
        "action_hidden": args.action_hidden,
        "head_hidden": args.head_hidden,
        "dropout": args.dropout,
    }
    model = AutomatonMLP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    run_dir = make_run_dir(args)
    checkpoint_path = run_dir / "best_model.pt"
    final_checkpoint_path = run_dir / "final_model.pt"
    config_path = run_dir / "train_config.json"
    norm_stats_path = run_dir / "normalization_stats.npz"
    provenance_path = run_dir / "data_provenance.json"
    training_stats_path = run_dir / "training_stats.json"
    label_counts_path = run_dir / "label_counts.json"
    loss_plot_path = run_dir / "training_loss_curves.png"
    log_path = run_dir / "training_log.txt"

    np.savez_compressed(norm_stats_path, **stats)

    train_config = vars(args) | {
        "dataset": str(args.dataset),
        "output_root": str(args.output_root),
        "run_dir": str(run_dir),
        "label_names": LABEL_NAMES,
        "model_config": model_config,
        "optimizer": "Adam",
        "loss_fn": "BCEWithLogitsLoss",
        "num_parameters": int(sum(param.numel() for param in model.parameters())),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "device": device,
    }
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2, default=str)

    provenance = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split": args.split,
        "val_ratio": args.val_ratio,
        "dataset": str(args.dataset),
        "action_horizon": args.action_horizon,
        "state_input": ["obs/proprio", "obs/states"],
        "action_input": "actions",
        "label_source": "obs/states",
        "train_demos": [{"path": str(traj["path"]), "demo_id": traj["demo_id"]} for traj in train_trajectories],
        "val_demos": [{"path": str(traj["path"]), "demo_id": traj["demo_id"]} for traj in val_trajectories],
    }
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)

    counts = {
        "train_labels": label_counts(flat_train["labels"]),
        "train_next_labels": label_counts(flat_train["next_labels"]),
        "val_labels": label_counts(flat_val["labels"]),
        "val_next_labels": label_counts(flat_val["next_labels"]),
    }
    with open(label_counts_path, "w") as f:
        json.dump(counts, f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Parameters: {train_config['num_parameters']:,}")
    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")
    print(f"Model config: {model_config}")

    train_losses = []
    val_losses = []
    val_bit_accs = []
    val_exact_accs = []
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
            labels = batch["labels"].to(device)
            targets = batch["next_labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(states, actions, labels)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = states.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / max(1, train_samples)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_bit_accs.append(val_metrics["bit_acc"])
        val_exact_accs.append(val_metrics["exact_acc"])

        is_best = val_loss < best_val_loss - 1e-5
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
                    "val_bit_acc": val_metrics["bit_acc"],
                    "val_exact_acc": val_metrics["exact_acc"],
                    "model_config": model_config,
                    "training_config": train_config,
                    "normalization_stats": stats,
                },
                checkpoint_path,
            )
        else:
            epochs_since_improvement += 1

        line = (
            f"epoch {epoch + 1:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"bit_acc {val_metrics['bit_acc']:.4f} | exact_acc {val_metrics['exact_acc']:.4f}"
            f"{' | best' if is_best else ''}"
        )
        print(line, flush=True)
        log_lines.append(line)
        log_path.write_text("\n".join(log_lines) + "\n")

        if epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}: no val improvement for {args.patience} epochs.")
            break

    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1] if train_losses else None,
            "val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "model_config": model_config,
            "training_config": train_config,
            "normalization_stats": stats,
        },
        final_checkpoint_path,
    )

    training_stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_bit_accs": val_bit_accs,
        "val_exact_accs": val_exact_accs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "num_epochs_trained": len(train_losses),
        "checkpoint_path": str(checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "normalization_stats_path": str(norm_stats_path),
        "data_provenance_path": str(provenance_path),
        "loss_plot_path": str(loss_plot_path),
    }
    with open(training_stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    save_loss_plot(train_losses, val_losses, loss_plot_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f} at epoch {best_epoch + 1}.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"Loss plot: {loss_plot_path}")


if __name__ == "__main__":
    main()
