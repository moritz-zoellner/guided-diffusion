"""Train the Toy Squares automaton model on one- or two-event futures.

The original notebook model predicts one future block label from the current
state, the current label, and an 8-action chunk. This script keeps that setup
but can also train an 8-logit model:

    logits[:, 0:4] -> next nonzero label
    logits[:, 4:8] -> next nonzero label after that

The second head is useful for checking whether the world model can see the
ordered block sequence in two-block scripted demonstrations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

matplotlib.use("Agg")
from matplotlib import pyplot as plt


STATE_BLOCK_NAMES = ["blue", "red", "green", "yellow"]
LABEL_NAMES = ["at_green", "at_blue", "at_red", "at_yellow"]
LABEL_TO_STATE_BLOCK_IDX = [2, 0, 1, 3]


def demo_sort_key(name: str) -> int:
    return int(re.search(r"\d+", name).group())


def get_demo_keys(hdf5_path: Path, mask: str | None = None) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        if mask is None:
            keys = list(f["data"].keys())
        else:
            keys = [key.decode("utf-8") for key in np.asarray(f[f"mask/{mask}"])]
    return sorted(keys, key=demo_sort_key)


def label_states(states: np.ndarray, radius: float) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    agent = states[:, 0:2]
    blocks = states[:, 2:10].reshape(len(states), 4, 2)
    labels = []
    for block_idx in LABEL_TO_STATE_BLOCK_IDX:
        dist = np.linalg.norm(agent - blocks[:, block_idx, :], axis=-1)
        labels.append(dist <= radius)
    return np.stack(labels, axis=-1).astype(np.float32)


def future_nonzero_event_labels(labels: np.ndarray, immediate_next_labels: np.ndarray, max_events: int = 2) -> np.ndarray:
    """Return the next `max_events` nonzero label plateaus after each step.

    All-zero "not touching a block" plateaus are skipped. This makes a
    two-block demo produce, near the start, next=first block and
    next_next=second block instead of next_next=the zero gap after contact.
    """

    labels = np.asarray(labels, dtype=np.float32)
    immediate_next_labels = np.asarray(immediate_next_labels, dtype=np.float32)
    if len(labels) != len(immediate_next_labels):
        raise ValueError("labels and immediate_next_labels must have the same length")

    timeline = np.concatenate([labels[:1], immediate_next_labels], axis=0)
    out = np.zeros((len(labels), max_events, labels.shape[-1]), dtype=np.float32)

    for t in range(len(labels)):
        previous = timeline[t]
        events = []
        for future_label in timeline[t + 1 :]:
            if np.array_equal(future_label, previous):
                continue
            previous = future_label
            if np.any(future_label > 0.5):
                events.append(future_label)
                if len(events) >= max_events:
                    break
        for event_idx, event in enumerate(events):
            out[t, event_idx] = event

    return out


def build_automaton_trajectories(hdf5_path: Path, demo_keys: list[str], radius: float) -> list[dict]:
    trajectories = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            obs = demo["obs"]
            next_obs = demo["next_obs"]
            states = np.concatenate([obs["agent_pos"][:], obs["states"][:]], axis=-1).astype(np.float32)
            next_states = np.concatenate([next_obs["agent_pos"][:], next_obs["states"][:]], axis=-1).astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)
            if not (len(states) == len(next_states) == len(actions)):
                raise ValueError(
                    f"{demo_key}: expected states, next_states, actions to align; "
                    f"got {len(states)}, {len(next_states)}, {len(actions)}"
                )
            if len(actions) < 2:
                continue

            labels = label_states(states, radius=radius)
            immediate_next_labels = label_states(next_states, radius=radius)
            future_events = future_nonzero_event_labels(labels, immediate_next_labels, max_events=2)

            trajectories.append(
                {
                    "path": str(hdf5_path),
                    "demo_id": demo_key,
                    "states": states,
                    "actions": actions,
                    "labels": labels,
                    "next_labels": future_events[:, 0],
                    "next_next_labels": future_events[:, 1],
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


def horizon_targets(values: np.ndarray, action_horizon: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n_chunks = len(values) - int(action_horizon)
    if n_chunks <= 0:
        return np.empty((0, values.shape[-1]), dtype=np.float32)
    return values[action_horizon:].astype(np.float32)


def flatten_trajectories(trajectories: list[dict], action_horizon: int, predict_next_next: bool) -> dict[str, np.ndarray]:
    keys = ["states", "actions", "labels", "next_labels"]
    if predict_next_next:
        keys.append("next_next_labels")
    flat = {key: [] for key in keys}

    for traj in trajectories:
        n_steps = len(traj["actions"])
        n_chunks = n_steps - int(action_horizon)
        if n_chunks <= 0:
            continue

        action_chunks = np.stack(
            [traj["actions"][idx : idx + action_horizon] for idx in range(n_chunks)],
            axis=0,
        ).reshape(n_chunks, -1)

        flat["states"].append(traj["states"][:-action_horizon])
        flat["actions"].append(action_chunks)
        flat["labels"].append(traj["labels"][:-action_horizon])
        flat["next_labels"].append(horizon_targets(traj["next_labels"], action_horizon))
        if predict_next_next:
            flat["next_next_labels"].append(horizon_targets(traj["next_next_labels"], action_horizon))

    if not flat["states"]:
        raise ValueError(f"No trajectories were long enough for action_horizon={action_horizon}.")
    return {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in flat.items()}


def normalization_stats(flat_train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    stats = {
        "states_mean": flat_train["states"].mean(axis=0),
        "states_std": flat_train["states"].std(axis=0) + 1e-8,
        "actions_mean": flat_train["actions"].mean(axis=0),
        "actions_std": flat_train["actions"].std(axis=0) + 1e-8,
        "labels_mean": flat_train["labels"].mean(axis=0),
        "labels_std": flat_train["labels"].std(axis=0) + 1e-8,
        "next_labels_mean": flat_train["next_labels"].mean(axis=0),
        "next_labels_std": flat_train["next_labels"].std(axis=0) + 1e-8,
    }
    if "next_next_labels" in flat_train:
        stats["next_next_labels_mean"] = flat_train["next_next_labels"].mean(axis=0)
        stats["next_next_labels_std"] = flat_train["next_next_labels"].std(axis=0) + 1e-8
    return stats


class AutomatonDataset(Dataset):
    def __init__(self, flat_data: dict[str, np.ndarray], stats: dict[str, np.ndarray], predict_next_next: bool):
        self.states = ((flat_data["states"] - stats["states_mean"]) / stats["states_std"]).astype(np.float32)
        self.actions = ((flat_data["actions"] - stats["actions_mean"]) / stats["actions_std"]).astype(np.float32)
        self.labels = flat_data["labels"].astype(np.float32)
        self.next_labels = flat_data["next_labels"].astype(np.float32)
        self.predict_next_next = predict_next_next
        self.next_next_labels = flat_data.get("next_next_labels")
        if self.next_next_labels is not None:
            self.next_next_labels = self.next_next_labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "states": torch.from_numpy(self.states[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "next_labels": torch.from_numpy(self.next_labels[idx]),
        }
        if self.predict_next_next:
            item["next_next_labels"] = torch.from_numpy(self.next_next_labels[idx])
        return item


class AutomatonMLP(nn.Module):
    def __init__(
        self,
        state_dim=10,
        label_dim=4,
        action_chunk_dim=16,
        state_hidden=64,
        label_hidden=16,
        action_hidden=64,
        hidden_dim=128,
        output_dim=8,
    ):
        super().__init__()
        self.state_enc = nn.Sequential(nn.Linear(state_dim, state_hidden), nn.SiLU(), nn.Linear(state_hidden, state_hidden), nn.SiLU())
        self.label_enc = nn.Sequential(nn.Linear(label_dim, label_hidden), nn.SiLU(), nn.Linear(label_hidden, label_hidden), nn.SiLU())
        self.action_enc = nn.Sequential(nn.Linear(action_chunk_dim, action_hidden), nn.SiLU(), nn.Linear(action_hidden, action_hidden), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(state_hidden + label_hidden + action_hidden, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, states, actions, labels):
        z = torch.cat([self.state_enc(states), self.action_enc(actions), self.label_enc(labels)], dim=-1)
        return self.head(z)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_counts(array: np.ndarray) -> dict[str, int]:
    counts = Counter(tuple(int(x) for x in row) for row in np.asarray(array).astype(int))
    return {str(label): int(count) for label, count in counts.items()}


def make_target_matrix(flat_data: dict[str, np.ndarray], predict_next_next: bool) -> np.ndarray:
    if predict_next_next:
        return np.concatenate([flat_data["next_labels"], flat_data["next_next_labels"]], axis=-1).astype(np.float32)
    return flat_data["next_labels"].astype(np.float32)


def positive_class_weights(targets: np.ndarray, max_pos_weight: float) -> np.ndarray:
    """Build BCE positive weights so rare future-contact labels still matter."""

    targets = np.asarray(targets, dtype=np.float32)
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, max_pos_weight)
    return weights.astype(np.float32)


def predictability_diagnostics(flat_data: dict[str, np.ndarray], action_horizon: int) -> dict[str, object]:
    """Cheap data-side checks for what the 8-action chunk can plausibly reveal."""

    states = flat_data["states"]
    action_chunks = flat_data["actions"].reshape(len(states), int(action_horizon), -1)
    next_labels = flat_data["next_labels"]
    next_next_labels = flat_data.get("next_next_labels")

    agent = states[:, 0:2]
    blocks = states[:, 2:10].reshape(len(states), 4, 2)
    label_order_blocks = blocks[:, LABEL_TO_STATE_BLOCK_IDX, :]
    mean_action = action_chunks.mean(axis=1)
    action_norm = mean_action / (np.linalg.norm(mean_action, axis=-1, keepdims=True) + 1e-8)
    block_vec = label_order_blocks - agent[:, None, :]
    block_vec_norm = block_vec / (np.linalg.norm(block_vec, axis=-1, keepdims=True) + 1e-8)
    direction_scores = (action_norm[:, None, :] * block_vec_norm).sum(axis=-1)
    direction_choice = direction_scores.argmax(axis=-1)

    next_nonzero = next_labels.sum(axis=-1) > 0.5
    if next_nonzero.any():
        next_truth = next_labels[next_nonzero].argmax(axis=-1)
        direction_next_top1 = float((direction_choice[next_nonzero] == next_truth).mean())
    else:
        direction_next_top1 = float("nan")

    diagnostics = {
        "num_samples": int(len(states)),
        "action_horizon": int(action_horizon),
        "next_positive_rate": next_labels.mean(axis=0).round(6).tolist(),
        "mean_action_direction_next_top1": direction_next_top1,
    }
    if next_next_labels is not None:
        next_next_nonzero = next_next_labels.sum(axis=-1) > 0.5
        diagnostics["next_next_positive_rate"] = next_next_labels.mean(axis=0).round(6).tolist()
        diagnostics["next_next_nonzero_samples"] = int(next_next_nonzero.sum())
        conditional_counts = {}
        majority_correct = 0
        for next_idx in range(next_labels.shape[-1]):
            mask = next_next_nonzero & (next_labels.sum(axis=-1) > 0.5) & (next_labels.argmax(axis=-1) == next_idx)
            counts = np.bincount(next_next_labels[mask].argmax(axis=-1), minlength=next_next_labels.shape[-1])
            total = int(counts.sum())
            majority_correct += int(counts.max()) if total else 0
            conditional_counts[LABEL_NAMES[next_idx]] = {
                "num_samples": total,
                "next_next_counts": {LABEL_NAMES[i]: int(counts[i]) for i in range(len(LABEL_NAMES))},
                "majority_accuracy": float(counts.max() / total) if total else float("nan"),
            }
        diagnostics["next_next_counts_conditioned_on_next"] = conditional_counts
        diagnostics["next_next_majority_ceiling_given_next"] = float(
            majority_correct / max(1, int(next_next_nonzero.sum()))
        )
    return diagnostics


def head_metrics(logits: torch.Tensor, targets: torch.Tensor, prefix: str) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    bit_acc = (preds == targets).float().mean().item()
    exact_acc = (preds == targets).all(dim=-1).float().mean().item()
    nonzero_mask = targets.sum(dim=-1) > 0.5
    if nonzero_mask.any():
        nonzero_targets = targets[nonzero_mask]
        nonzero_preds = preds[nonzero_mask]
        nonzero_probs = probs[nonzero_mask]
        true_idx = torch.argmax(nonzero_targets, dim=-1)
        top1_idx = torch.argmax(nonzero_probs, dim=-1)
        true_prob = nonzero_probs.gather(1, true_idx[:, None]).mean().item()
        nonzero_exact = (nonzero_preds == nonzero_targets).all(dim=-1).float().mean().item()
        top1_acc = (top1_idx == true_idx).float().mean().item()
    else:
        true_prob = float("nan")
        nonzero_exact = float("nan")
        top1_acc = float("nan")
    return {
        f"{prefix}_bit_acc": bit_acc,
        f"{prefix}_exact_acc": exact_acc,
        f"{prefix}_nonzero_exact_acc": nonzero_exact,
        f"{prefix}_nonzero_top1_acc": top1_acc,
        f"{prefix}_mean_true_label_prob": true_prob,
        f"{prefix}_positive_rate": targets.mean().item(),
    }


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: str, predict_next_next: bool) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            labels = batch["labels"].to(device)
            next_labels = batch["next_labels"].to(device)
            if predict_next_next:
                targets = torch.cat([next_labels, batch["next_next_labels"].to(device)], dim=-1)
            else:
                targets = next_labels
            logits = model(states, actions, labels)
            loss = loss_fn(logits, targets)
            batch_size = states.shape[0]
            loss_sum += loss.item() * batch_size
            n_samples += batch_size
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = {"loss": loss_sum / max(1, n_samples)}
    metrics |= head_metrics(logits[:, :4], targets[:, :4], "next")
    if predict_next_next:
        metrics |= head_metrics(logits[:, 4:8], targets[:, 4:8], "next_next")
    return metrics


def make_run_dir(args) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode = "two_event" if args.predict_next_next else "one_event"
    run_name = (
        f"{mode}_h{args.action_horizon}_sh{args.state_hidden}_ah{args.action_hidden}_"
        f"lh{args.label_hidden}_hh{args.hidden_dim}_lr{args.lr:g}_{timestamp}"
    )
    run_dir = args.output_root / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = args.output_root / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_loss_plot(history: list[dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([row["train_loss"] for row in history], label="train", alpha=0.85)
    ax.plot([row["val_loss"] for row in history], label="val", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogits loss")
    ax.set_title("Toy Squares Automaton World Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/toy_squares_data/two_block_train/data.hdf5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/automaton_world_model_two_step"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--predict-next-next", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-positive-labels", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--state-hidden", type=int, default=64)
    parser.add_argument("--label-hidden", type=int, default=16)
    parser.add_argument("--action-hidden", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    demo_keys = get_demo_keys(args.dataset)
    if args.max_demos is not None:
        demo_keys = demo_keys[: args.max_demos]

    trajectories = build_automaton_trajectories(args.dataset, demo_keys, radius=args.radius)
    train_trajectories, val_trajectories = split_trajectories(trajectories, args.val_ratio, args.split_seed)
    flat_train = flatten_trajectories(train_trajectories, args.action_horizon, args.predict_next_next)
    flat_val = flatten_trajectories(val_trajectories, args.action_horizon, args.predict_next_next)
    stats = normalization_stats(flat_train)

    train_dataset = AutomatonDataset(flat_train, stats, args.predict_next_next)
    val_dataset = AutomatonDataset(flat_val, stats, args.predict_next_next)
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
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dim = 8 if args.predict_next_next else 4
    model_config = {
        "state_dim": int(flat_train["states"].shape[-1]),
        "label_dim": len(LABEL_NAMES),
        "action_chunk_dim": int(flat_train["actions"].shape[-1]),
        "state_hidden": args.state_hidden,
        "label_hidden": args.label_hidden,
        "action_hidden": args.action_hidden,
        "hidden_dim": args.hidden_dim,
        "output_dim": output_dim,
    }
    model = AutomatonMLP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target_mean = make_target_matrix(flat_train, args.predict_next_next).mean(axis=0)
    pos_weight = None
    if args.balance_positive_labels:
        pos_weight = positive_class_weights(make_target_matrix(flat_train, args.predict_next_next), args.max_pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    run_dir = make_run_dir(args)
    np.savez_compressed(run_dir / "normalization_stats.npz", **stats)
    train_config = vars(args) | {
        "dataset": str(args.dataset),
        "output_root": str(args.output_root),
        "run_dir": str(run_dir),
        "label_names": LABEL_NAMES,
        "state_block_names": STATE_BLOCK_NAMES,
        "label_to_state_block_idx": LABEL_TO_STATE_BLOCK_IDX,
        "model_config": model_config,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "device": device,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "target_mean": target_mean.tolist(),
        "pos_weight": pos_weight.tolist() if pos_weight is not None else None,
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str))
    (run_dir / "label_counts.json").write_text(
        json.dumps(
            {
                "train_labels": label_counts(flat_train["labels"]),
                "train_next_labels": label_counts(flat_train["next_labels"]),
                "val_labels": label_counts(flat_val["labels"]),
                "val_next_labels": label_counts(flat_val["next_labels"]),
                "train_next_next_labels": label_counts(flat_train["next_next_labels"]) if args.predict_next_next else {},
                "val_next_next_labels": label_counts(flat_val["next_next_labels"]) if args.predict_next_next else {},
            },
            indent=2,
        )
    )
    (run_dir / "predictability_diagnostics.json").write_text(
        json.dumps(
            {
                "train": predictability_diagnostics(flat_train, args.action_horizon),
                "val": predictability_diagnostics(flat_val, args.action_horizon),
            },
            indent=2,
        )
    )
    (run_dir / "data_provenance.json").write_text(
        json.dumps(
            {
                "train_demos": [{"path": traj["path"], "demo_id": traj["demo_id"]} for traj in train_trajectories],
                "val_demos": [{"path": traj["path"], "demo_id": traj["demo_id"]} for traj in val_trajectories],
            },
            indent=2,
        )
    )

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}", flush=True)
    print(f"Model parameters: {train_config['num_parameters']:,} | Device: {device}", flush=True)
    if pos_weight is not None:
        print(f"Positive BCE weights: {np.round(pos_weight, 3).tolist()}", flush=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    last_epoch = -1
    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            labels = batch["labels"].to(device)
            next_labels = batch["next_labels"].to(device)
            if args.predict_next_next:
                targets = torch.cat([next_labels, batch["next_next_labels"].to(device)], dim=-1)
            else:
                targets = next_labels
            optimizer.zero_grad(set_to_none=True)
            logits = model(states, actions, labels)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * states.shape[0]
            train_samples += states.shape[0]

        train_loss = train_loss_sum / max(1, train_samples)
        val_metrics = evaluate(model, val_loader, loss_fn, device, args.predict_next_next)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"]} | val_metrics
        history.append(row)
        is_best = val_metrics["loss"] < best_val_loss - 1e-5
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_metrics": val_metrics,
                    "model_config": model_config,
                    "training_config": train_config,
                    "normalization_stats": stats,
                    "label_mean": target_mean.astype(np.float32),
                    "pos_weight": pos_weight.astype(np.float32) if pos_weight is not None else None,
                },
                run_dir / "best_model.pt",
            )
        else:
            epochs_since_improvement += 1

        print(
            f"epoch {epoch + 1:03d} | train {train_loss:.5f} | val {val_metrics['loss']:.5f} | "
            f"next_top1 {val_metrics['next_nonzero_top1_acc']:.3f} | "
            f"next2_top1 {val_metrics.get('next_next_nonzero_top1_acc', float('nan')):.3f}"
            f"{' | best' if is_best else ''}",
            flush=True,
        )
        (run_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        if epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}.", flush=True)
            break

    final_metrics = evaluate(model, val_loader, loss_fn, device, args.predict_next_next)
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": history[-1]["train_loss"] if history else None,
            "val_loss": final_metrics["loss"],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "val_metrics": final_metrics,
            "model_config": model_config,
            "training_config": train_config,
            "normalization_stats": stats,
            "label_mean": target_mean.astype(np.float32),
            "pos_weight": pos_weight.astype(np.float32) if pos_weight is not None else None,
        },
        run_dir / "final_model.pt",
    )
    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "num_epochs_trained": len(history),
        "best_metrics": history[best_epoch] if best_epoch >= 0 else {},
        "final_metrics": final_metrics,
        "run_dir": str(run_dir),
    }
    (run_dir / "training_stats.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "prediction_strength.json").write_text(json.dumps(summary["best_metrics"], indent=2))
    save_loss_plot(history, run_dir / "training_loss_curves.png")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}", flush=True)
    print(f"Prediction strength: {run_dir / 'prediction_strength.json'}", flush=True)


if __name__ == "__main__":
    main()
