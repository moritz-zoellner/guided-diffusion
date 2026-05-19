#!/usr/bin/env python3
"""
Offline diagnostics for the Toy Squares LTLDoG paper-test rollouts.

The paper-faithful LTLDoG baseline samples one full state-action trajectory and
then executes it once, either via generated actions or sampled-state waypoints.
This script compares those two views:

1. Does the sampled trajectory itself satisfy the requested formula?
2. If it does, does the real environment execution follow that trajectory?
3. Are static block positions being hallucinated by the trajectory model?

The output lives next to the paper-test rollouts so we can keep the comparison
auditable without rerunning expensive sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTION_DIM = 2
OBS_DIM = 10
LABEL_TO_BLOCK_SLICE = {
    "blue": slice(2, 4),
    "red": slice(4, 6),
    "green": slice(6, 8),
    "yellow": slice(8, 10),
}


@dataclass
class RolloutDiagnostics:
    horizon_name: str
    method: str
    rollout: str
    run_dir: str
    chain: Tuple[str, ...]
    complete: bool
    stages: int
    steps: int
    predicted_sequence_value_generated_blocks: float
    predicted_sequence_sat_generated_blocks: bool
    predicted_sequence_value_fixed_blocks: float
    predicted_sequence_sat_fixed_blocks: bool
    predicted_first_value_generated_blocks: float
    predicted_first_sat_generated_blocks: bool
    predicted_first_value_fixed_blocks: float
    predicted_first_sat_fixed_blocks: bool
    actual_sequence_value: float
    actual_sequence_sat: bool
    actual_first_value: float
    actual_first_sat: bool
    agent_rmse_same_index: float
    agent_rmse_after_action: float
    block_drift_max: float
    block_drift_mean: float
    block_drift_at_generated_best_first: float
    action_norm_mean: float
    action_clamp_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/toy_squares_rollouts/baseline_ltldog/rollouts/paper_test"),
        help="Root paper-test rollout directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write diagnostics. Defaults to ROOT/diagnostics.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Method folders to include. Defaults to all ltldog* folders.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="Horizon folders to include, e.g. horizon_01_blue.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_vector(item: np.ndarray) -> np.ndarray:
    arr = np.asarray(item, dtype=np.float32)
    while arr.ndim > 1:
        arr = arr[..., -1]
    return arr.reshape(-1)


def actual_observations(run_dir: Path) -> np.ndarray:
    with np.load(run_dir / "low_level_obs.npz", allow_pickle=True) as data:
        agent_raw = np.asarray(data["agent_pos"])
        states_raw = np.asarray(data["states"])

    if agent_raw.dtype == object:
        agent = np.stack([latest_vector(item)[:2] for item in agent_raw], axis=0)
    else:
        agent = np.stack([latest_vector(item)[:2] for item in agent_raw], axis=0)

    if states_raw.dtype == object:
        states = np.stack([latest_vector(item)[:8] for item in states_raw], axis=0)
    else:
        states = np.stack([latest_vector(item)[:8] for item in states_raw], axis=0)

    return np.concatenate([agent, states], axis=-1).astype(np.float32)


def predicted_trajectory(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(run_dir / "trace.npz") as data:
        actions = np.asarray(data["actions"], dtype=np.float32)
        predicted = np.asarray(data["predicted"], dtype=np.float32)
    if predicted.size == 0:
        return actions, np.zeros((0, OBS_DIM), dtype=np.float32)
    return actions, predicted[:, ACTION_DIM : ACTION_DIM + OBS_DIM]


def fixed_block_observations(pred_obs: np.ndarray, actual_obs: np.ndarray) -> np.ndarray:
    fixed = pred_obs.copy()
    if len(fixed) == 0 or len(actual_obs) == 0:
        return fixed
    fixed[:, 2:OBS_DIM] = actual_obs[0, 2:OBS_DIM][None]
    return fixed


def label_robustness(obs: np.ndarray, label: str, radius: float = 0.2) -> np.ndarray:
    agent = obs[:, 0:2]
    block = obs[:, LABEL_TO_BLOCK_SLICE[label]]
    return radius - np.linalg.norm(agent - block, axis=-1)


def sequence_value(obs: np.ndarray, chain: Sequence[str], radius: float = 0.2) -> Tuple[float, bool]:
    if len(obs) == 0:
        return float("-inf"), False
    score = label_robustness(obs, chain[0], radius=radius)
    for label in chain[1:]:
        prefix = np.maximum.accumulate(score)
        shifted_prefix = np.full_like(prefix, -1.0e6)
        shifted_prefix[1:] = prefix[:-1]
        score = np.minimum(shifted_prefix, label_robustness(obs, label, radius=radius))
    value = float(np.max(score))
    return value, bool(value > 0.0)


def first_target_value(obs: np.ndarray, chain: Sequence[str], radius: float = 0.2) -> Tuple[float, bool, int]:
    if len(obs) == 0:
        return float("-inf"), False, -1
    r = label_robustness(obs, chain[0], radius=radius)
    idx = int(np.argmax(r))
    value = float(r[idx])
    return value, bool(value > 0.0), idx


def finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a[:n] - b[:n]) ** 2, axis=-1))))


def diagnose_rollout(horizon_name: str, method: str, run_dir: Path) -> RolloutDiagnostics:
    summary = load_json(run_dir / "rollout_summary.json")
    chain = tuple(summary["chain"])
    actions, pred_obs = predicted_trajectory(run_dir)
    act_obs = actual_observations(run_dir)
    pred_fixed = fixed_block_observations(pred_obs, act_obs)

    pred_seq_gen_value, pred_seq_gen_sat = sequence_value(pred_obs, chain)
    pred_seq_fixed_value, pred_seq_fixed_sat = sequence_value(pred_fixed, chain)
    pred_first_gen_value, pred_first_gen_sat, pred_best_idx = first_target_value(pred_obs, chain)
    pred_first_fixed_value, pred_first_fixed_sat, _ = first_target_value(pred_fixed, chain)
    act_seq_value, act_seq_sat = sequence_value(act_obs, chain)
    act_first_value, act_first_sat, _ = first_target_value(act_obs, chain)

    block_drift = np.zeros((0,), dtype=np.float32)
    block_drift_at_best = float("nan")
    if len(pred_obs) and len(act_obs):
        initial_blocks = act_obs[0, 2:OBS_DIM][None]
        per_step_block_drift = np.linalg.norm(
            pred_obs[:, 2:OBS_DIM].reshape(len(pred_obs), 4, 2)
            - initial_blocks.reshape(1, 4, 2),
            axis=-1,
        )
        block_drift = per_step_block_drift.max(axis=-1)
        if 0 <= pred_best_idx < len(block_drift):
            block_drift_at_best = float(block_drift[pred_best_idx])

    clamp_fraction = float(np.mean(np.abs(actions) >= 0.999)) if actions.size else float("nan")
    action_norm_mean = float(np.mean(np.linalg.norm(actions, axis=-1))) if len(actions) else float("nan")

    return RolloutDiagnostics(
        horizon_name=horizon_name,
        method=method,
        rollout=run_dir.name,
        run_dir=str(run_dir),
        chain=chain,
        complete=bool(summary.get("complete", False)),
        stages=int(summary.get("stages", 0)),
        steps=int(summary.get("steps", 0)),
        predicted_sequence_value_generated_blocks=pred_seq_gen_value,
        predicted_sequence_sat_generated_blocks=pred_seq_gen_sat,
        predicted_sequence_value_fixed_blocks=pred_seq_fixed_value,
        predicted_sequence_sat_fixed_blocks=pred_seq_fixed_sat,
        predicted_first_value_generated_blocks=pred_first_gen_value,
        predicted_first_sat_generated_blocks=pred_first_gen_sat,
        predicted_first_value_fixed_blocks=pred_first_fixed_value,
        predicted_first_sat_fixed_blocks=pred_first_fixed_sat,
        actual_sequence_value=act_seq_value,
        actual_sequence_sat=act_seq_sat,
        actual_first_value=act_first_value,
        actual_first_sat=act_first_sat,
        agent_rmse_same_index=finite_rmse(pred_obs[:, 0:2], act_obs[:, 0:2]),
        agent_rmse_after_action=finite_rmse(pred_obs[:, 0:2], act_obs[1:, 0:2]),
        block_drift_max=float(np.max(block_drift)) if len(block_drift) else float("nan"),
        block_drift_mean=float(np.mean(block_drift)) if len(block_drift) else float("nan"),
        block_drift_at_generated_best_first=block_drift_at_best,
        action_norm_mean=action_norm_mean,
        action_clamp_fraction=clamp_fraction,
    )


def iter_rollout_dirs(root: Path, methods: Sequence[str] | None, horizons: Sequence[str] | None) -> Iterable[Tuple[str, str, Path]]:
    horizon_dirs = sorted(root.glob("horizon_*"))
    if horizons:
        wanted_horizons = set(horizons)
        horizon_dirs = [p for p in horizon_dirs if p.name in wanted_horizons]

    for horizon_dir in horizon_dirs:
        method_dirs = sorted(p for p in horizon_dir.iterdir() if p.is_dir())
        if methods:
            wanted_methods = set(methods)
            method_dirs = [p for p in method_dirs if p.name in wanted_methods]
        else:
            method_dirs = [p for p in method_dirs if p.name.startswith("ltldog")]
        for method_dir in method_dirs:
            for rollout_dir in sorted(method_dir.glob("rollout_*")):
                if (rollout_dir / "trace.npz").exists() and (rollout_dir / "rollout_summary.json").exists():
                    yield horizon_dir.name, method_dir.name, rollout_dir


def summarize(rows: Sequence[RolloutDiagnostics]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[RolloutDiagnostics]] = {}
    for row in rows:
        grouped.setdefault((row.horizon_name, row.method), []).append(row)

    summaries = []
    for (horizon_name, method), vals in sorted(grouped.items()):
        summaries.append(
            {
                "horizon": horizon_name,
                "method": method,
                "n": len(vals),
                "chain": list(vals[0].chain),
                "actual_complete_rate": float(np.mean([v.complete for v in vals])),
                "actual_sequence_sat_rate": float(np.mean([v.actual_sequence_sat for v in vals])),
                "actual_first_target_sat_rate": float(np.mean([v.actual_first_sat for v in vals])),
                "predicted_sequence_sat_rate_generated_blocks": float(
                    np.mean([v.predicted_sequence_sat_generated_blocks for v in vals])
                ),
                "predicted_sequence_sat_rate_fixed_blocks": float(
                    np.mean([v.predicted_sequence_sat_fixed_blocks for v in vals])
                ),
                "predicted_first_target_sat_rate_generated_blocks": float(
                    np.mean([v.predicted_first_sat_generated_blocks for v in vals])
                ),
                "predicted_first_target_sat_rate_fixed_blocks": float(
                    np.mean([v.predicted_first_sat_fixed_blocks for v in vals])
                ),
                "mean_stages": float(np.mean([v.stages for v in vals])),
                "mean_steps": float(np.mean([v.steps for v in vals])),
                "mean_predicted_sequence_value_generated_blocks": float(
                    np.mean([v.predicted_sequence_value_generated_blocks for v in vals])
                ),
                "mean_predicted_sequence_value_fixed_blocks": float(
                    np.mean([v.predicted_sequence_value_fixed_blocks for v in vals])
                ),
                "mean_actual_sequence_value": float(np.mean([v.actual_sequence_value for v in vals])),
                "mean_agent_rmse_same_index": float(np.mean([v.agent_rmse_same_index for v in vals])),
                "mean_agent_rmse_after_action": float(np.mean([v.agent_rmse_after_action for v in vals])),
                "mean_block_drift_max": float(np.mean([v.block_drift_max for v in vals])),
                "mean_block_drift_at_generated_best_first": float(
                    np.mean([v.block_drift_at_generated_best_first for v in vals])
                ),
                "mean_action_norm": float(np.mean([v.action_norm_mean for v in vals])),
                "mean_action_clamp_fraction": float(np.mean([v.action_clamp_fraction for v in vals])),
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, summaries: Sequence[Dict]) -> None:
    if not summaries:
        return

    labels = [f"{row['horizon'].replace('horizon_', 'H')} | {row['method']}" for row in summaries]
    metrics = [
        ("predicted_first_target_sat_rate_generated_blocks", "predicted first target"),
        ("predicted_sequence_sat_rate_generated_blocks", "predicted full formula"),
        ("actual_sequence_sat_rate", "actual full formula"),
    ]
    x = np.arange(len(summaries))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10, len(summaries) * 0.85), 5.5))
    for offset, (key, label) in zip([-width, 0.0, width], metrics):
        ax.bar(x + offset, [row[key] for row in summaries], width=width, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("LTLDoG Plan-vs-Execution Diagnostics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "plan_vs_execution_rates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(summaries) * 0.85), 5.5))
    ax.bar(x - 0.18, [row["mean_agent_rmse_after_action"] for row in summaries], width=0.36, label="agent RMSE")
    ax.bar(x + 0.18, [row["mean_block_drift_max"] for row in summaries], width=0.36, label="max block drift")
    ax.set_ylabel("normalized coordinate distance")
    ax.set_title("Generated Trajectory Drift")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_drift.png", dpi=180)
    plt.close(fig)


def plot_example_rollouts(output_dir: Path, rows: Sequence[RolloutDiagnostics]) -> None:
    grouped: Dict[Tuple[str, str], RolloutDiagnostics] = {}
    for row in rows:
        grouped.setdefault((row.horizon_name, row.method), row)
    if not grouped:
        return

    refs = [grouped[key] for key in sorted(grouped)]
    ncols = 2
    nrows = int(np.ceil(len(refs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4.5, 4.2 * nrows)), squeeze=False)
    colors = {
        "blue": "#1e88ff",
        "red": "#ff4136",
        "green": "#34c759",
        "yellow": "#ffd60a",
    }

    for ax, row in zip(axes.ravel(), refs):
        actions, pred_obs = predicted_trajectory(Path(row.run_dir))
        del actions
        act_obs = actual_observations(Path(row.run_dir))
        initial_blocks = act_obs[0, 2:OBS_DIM].reshape(4, 2)
        labels = ["blue", "red", "green", "yellow"]

        for label, xy in zip(labels, initial_blocks):
            ax.scatter(xy[0], xy[1], marker="s", s=260, color=colors[label], edgecolor="none", alpha=0.85, zorder=1)
        if len(pred_obs):
            ax.plot(pred_obs[:, 0], pred_obs[:, 1], color="#2f80ed", linestyle="--", linewidth=1.4, alpha=0.72, label="sampled state")
            ax.scatter(pred_obs[0, 0], pred_obs[0, 1], color="#2f80ed", s=22, zorder=3)
        if len(act_obs):
            ax.plot(act_obs[:, 0], act_obs[:, 1], color="#111111", linewidth=1.55, alpha=0.78, label="executed state")
            ax.scatter(act_obs[-1, 0], act_obs[-1, 1], color="#111111", s=22, zorder=3)
        ax.set_title(
            f"{row.horizon_name.replace('horizon_', 'H')} | {row.method}\n"
            f"pred seq={int(row.predicted_sequence_sat_generated_blocks)} actual={int(row.actual_sequence_sat)}",
            fontsize=9,
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)

    for ax in axes.ravel()[len(refs) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "plan_vs_execution_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.root / "diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        diagnose_rollout(horizon_name, method, rollout_dir)
        for horizon_name, method, rollout_dir in iter_rollout_dirs(args.root, args.methods, args.horizons)
    ]
    summaries = summarize(rows)

    rollout_rows = []
    for row in rows:
        item = dict(row.__dict__)
        item["chain"] = list(row.chain)
        rollout_rows.append(item)
    with open(output_dir / "diagnostics_rollouts.json", "w", encoding="utf-8") as f:
        json.dump(rollout_rows, f, indent=2)
    with open(output_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    write_csv(output_dir / "diagnostics_summary.csv", summaries)
    plot_summary(output_dir, summaries)
    plot_example_rollouts(output_dir, rows)

    print(f"Wrote {len(rows)} rollout diagnostics to {output_dir}")
    for row in summaries:
        print(
            f"{row['horizon']} {row['method']}: "
            f"pred_first={row['predicted_first_target_sat_rate_generated_blocks']:.2f}, "
            f"pred_seq={row['predicted_sequence_sat_rate_generated_blocks']:.2f}, "
            f"actual_seq={row['actual_sequence_sat_rate']:.2f}, "
            f"agent_rmse={row['mean_agent_rmse_after_action']:.3f}, "
            f"block_drift={row['mean_block_drift_max']:.3f}"
        )


if __name__ == "__main__":
    main()
