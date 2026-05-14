"""Evaluate the real-world automaton world model on held-out demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover - sklearn is available in the pixi env.
    average_precision_score = None
    roc_auc_score = None

try:
    from label_real_world import LABEL_NAMES, TARGET_RULE_DESCRIPTIONS, LabelConfig, horizon_targets
    from train_automaton_world_model import AutomatonMLP, build_automaton_trajectories, flatten_trajectories
except ModuleNotFoundError:
    from real_world_experiments.label_real_world import LABEL_NAMES, TARGET_RULE_DESCRIPTIONS, LabelConfig, horizon_targets
    from real_world_experiments.train_automaton_world_model import (
        AutomatonMLP,
        build_automaton_trajectories,
        flatten_trajectories,
    )


def _safe_metric(fn, y_true, y_score):
    if fn is None or len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_score))


def _prediction_batches(model, flat, stats, batch_size, device):
    states = ((flat["states"] - stats["states_mean"]) / stats["states_std"]).astype(np.float32)
    actions = ((flat["actions"] - stats["actions_mean"]) / stats["actions_std"]).astype(np.float32)
    labels = flat["labels"].astype(np.float32)

    probs = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            logits = model(
                torch.from_numpy(states[start:end]).to(device),
                torch.from_numpy(actions[start:end]).to(device),
                torch.from_numpy(labels[start:end]).to(device),
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _event_metadata(trajectories, action_horizon, target_rule):
    current_labels = []
    targets = []
    rise_in_horizon = []

    for traj in trajectories:
        labels = traj["labels"].astype(np.int32)
        next_labels = traj["next_labels"].astype(np.int32)
        n_chunks = len(traj["actions"]) - int(action_horizon)
        if n_chunks <= 0:
            continue

        chunk_targets = horizon_targets(labels, next_labels, action_horizon, target_rule).astype(np.int32)
        for idx in range(n_chunks):
            window = labels[idx + 1 : idx + action_horizon + 1]
            previous = np.concatenate([labels[idx : idx + 1], labels[idx + 1 : idx + action_horizon]], axis=0)
            current_labels.append(labels[idx])
            targets.append(chunk_targets[idx])
            rise_in_horizon.append(((previous == 0) & (window == 1)).any(axis=0))

    return {
        "current_labels": np.asarray(current_labels, dtype=bool),
        "targets": np.asarray(targets, dtype=bool),
        "rise_in_horizon": np.asarray(rise_in_horizon, dtype=bool),
    }


def _load_config(run_dir):
    config = json.loads((run_dir / "train_config.json").read_text())
    label_config = LabelConfig(**config.get("label_config", {}))
    label_keys = config.get("label_keys") or {"eef_quat_key": "eef_quat_wxyz", "gripper_width_key": "gripper_width"}
    return config, label_config, label_keys


def evaluate_run(run_dir, split, checkpoint_name, batch_size, device):
    run_dir = Path(run_dir)
    provenance = json.loads((run_dir / "data_provenance.json").read_text())
    config, label_config, label_keys = _load_config(run_dir)

    if split == "train":
        demos = provenance["train_demos"]
    elif split == "val":
        demos = provenance["val_demos"]
    else:
        demos = provenance["train_demos"] + provenance["val_demos"]

    dataset = Path(config["dataset"])
    demo_keys = [demo["demo_id"] for demo in demos]
    state_keys = config["state_keys"]
    action_key = config["action_key"]
    action_horizon = int(config["action_horizon"])
    target_rule = config.get("target_rule", "max_next")

    trajectories = build_automaton_trajectories(dataset, demo_keys, state_keys, action_key, label_config, label_keys)
    flat = flatten_trajectories(trajectories, action_horizon, target_rule)
    metadata = _event_metadata(trajectories, action_horizon, target_rule)

    stats_npz = np.load(run_dir / "normalization_stats.npz")
    stats = {key: stats_npz[key] for key in stats_npz.files}
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device, weights_only=False)

    model = AutomatonMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = _prediction_batches(model, flat, stats, batch_size, device)
    targets = flat["next_labels"].astype(np.float32)
    preds = (probs >= 0.5).astype(np.float32)

    if len(probs) != len(targets) or len(probs) != len(metadata["current_labels"]):
        raise ValueError("prediction, target, and metadata lengths do not align")

    majority = np.asarray([targets[:, idx].mean() >= 0.5 for idx in range(targets.shape[-1])], dtype=np.float32)
    majority_preds = np.repeat(majority[None, :], len(targets), axis=0)

    label_results = {}
    label_names = config.get("label_names", LABEL_NAMES)
    for idx, name in enumerate(label_names):
        y_true = targets[:, idx]
        y_score = probs[:, idx]
        y_pred = preds[:, idx]
        bce = F.binary_cross_entropy(
            torch.from_numpy(y_score).clamp(1e-6, 1 - 1e-6),
            torch.from_numpy(y_true),
        ).item()

        current = metadata["current_labels"][:, idx]
        rises = metadata["rise_in_horizon"][:, idx]
        masks = {
            "current_false_rise_in_horizon": (~current) & rises,
            "current_false_target_positive_no_rise": (~current) & (y_true > 0.5) & (~rises),
            "current_false_target_negative": (~current) & (y_true <= 0.5),
            "current_true": current,
        }
        groups = {}
        for group_name, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                groups[group_name] = {"n": 0}
                continue
            group_scores = y_score[mask]
            groups[group_name] = {
                "n": n,
                "target_mean": float(y_true[mask].mean()),
                "pred_mean": float(group_scores.mean()),
                "pred_p10": float(np.quantile(group_scores, 0.1)),
                "pred_p50": float(np.quantile(group_scores, 0.5)),
                "pred_p90": float(np.quantile(group_scores, 0.9)),
            }

        event_mask = masks["current_false_rise_in_horizon"]
        negative_mask = masks["current_false_target_negative"]
        event_vs_negative_auc = None
        if event_mask.any() and negative_mask.any():
            pair_true = np.concatenate([np.ones(event_mask.sum()), np.zeros(negative_mask.sum())])
            pair_score = np.concatenate([y_score[event_mask], y_score[negative_mask]])
            event_vs_negative_auc = _safe_metric(roc_auc_score, pair_true, pair_score)

        label_results[name] = {
            "target_positive_rate": float(y_true.mean()),
            "pred_mean": float(y_score.mean()),
            "bce": float(bce),
            "accuracy_at_0.5": float((y_pred == y_true).mean()),
            "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
            "average_precision": _safe_metric(average_precision_score, y_true, y_score),
            "event_vs_negative_auc": event_vs_negative_auc,
            "groups": groups,
        }

    return {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", np.nan)),
        "split": split,
        "target_rule": target_rule,
        "target_rule_description": TARGET_RULE_DESCRIPTIONS[target_rule],
        "num_samples": int(len(probs)),
        "label_names": label_names,
        "bit_accuracy_at_0.5": float((preds == targets).mean()),
        "exact_accuracy_at_0.5": float((preds == targets).all(axis=-1).mean()),
        "majority_bit_accuracy": float((majority_preds == targets).mean()),
        "majority_exact_accuracy": float((majority_preds == targets).all(axis=-1).mean()),
        "label_results": label_results,
    }


def print_report(report):
    print(f"Run: {report['run_dir']}")
    print(
        f"Checkpoint: {report['checkpoint']} | epoch={report['checkpoint_epoch']} "
        f"| val_loss={report['checkpoint_val_loss']:.4f}"
    )
    print(f"Split: {report['split']} | samples={report['num_samples']:,}")
    print(f"Target rule: {report['target_rule']} ({report['target_rule_description']})")
    print(
        f"Exact acc: {report['exact_accuracy_at_0.5']:.3f} "
        f"(majority {report['majority_exact_accuracy']:.3f}) | "
        f"bit acc: {report['bit_accuracy_at_0.5']:.3f} "
        f"(majority {report['majority_bit_accuracy']:.3f})"
    )
    print("\nPer-label validation diagnostics")
    for name, result in report["label_results"].items():
        auc = "nan" if result["roc_auc"] is None else f"{result['roc_auc']:.3f}"
        ap = "nan" if result["average_precision"] is None else f"{result['average_precision']:.3f}"
        ev_auc = "nan" if result["event_vs_negative_auc"] is None else f"{result['event_vs_negative_auc']:.3f}"
        event_group = result["groups"]["current_false_rise_in_horizon"]
        neg_group = result["groups"]["current_false_target_negative"]
        event_mean = event_group.get("pred_mean")
        neg_mean = neg_group.get("pred_mean")
        event_mean_str = "nan" if event_mean is None else f"{event_mean:.3f}"
        neg_mean_str = "nan" if neg_mean is None else f"{neg_mean:.3f}"
        print(
            f"  {name:<15} pos={result['target_positive_rate']:.3f} "
            f"pred={result['pred_mean']:.3f} bce={result['bce']:.3f} "
            f"auc={auc} ap={ap} event_auc={ev_auc} "
            f"event_pred={event_mean_str} neg_pred={neg_mean_str}"
        )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--checkpoint", choices=["best_model.pt", "final_model.pt"], default="best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = evaluate_run(args.run_dir, args.split, args.checkpoint, args.batch_size, device)
    print_report(report)

    if args.output is None:
        stem = args.checkpoint.replace(".pt", "")
        args.output = args.run_dir / f"{stem}_{args.split}_event_diagnostics.json"
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved diagnostics: {args.output}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
