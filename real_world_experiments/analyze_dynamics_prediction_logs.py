#!/usr/bin/env python3
"""Analyze logged real-world dynamics model predictions against executed rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _as_state(obs: dict) -> np.ndarray:
    return np.concatenate([
        np.asarray(obs["eef_pos"], dtype=np.float64),
        np.asarray(obs["eef_rot6d"], dtype=np.float64),
        np.asarray(obs["gripper_binary"], dtype=np.float64),
        np.asarray(obs["cheezit_pos"], dtype=np.float64),
        np.asarray(obs["cheezit_rot6d"], dtype=np.float64),
    ])


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / np.maximum(np.linalg.norm(a2_orth, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _rot_angle_deg(pred_rot6d: np.ndarray, true_rot6d: np.ndarray) -> np.ndarray:
    pred_r = _rot6d_to_matrix(pred_rot6d)
    true_r = _rot6d_to_matrix(true_rot6d)
    rel = np.swapaxes(pred_r, -1, -2) @ true_r
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


def _percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _load_events(path: Path) -> list[dict]:
    events = []
    with path.open() as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def collect_rows(run_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    skipped = defaultdict(int)
    rollout_dirs = sorted((run_dir / "rollouts").glob("rollout_*"))
    for rollout_dir in rollout_dirs:
        events_path = rollout_dir / "events.jsonl"
        if not events_path.exists():
            continue
        events = _load_events(events_path)
        actual_by_decision = {
            int(event["decision_idx"]): _as_state(event["reached_obs"])
            for event in events
            if event.get("type") == "target_reached" and "reached_obs" in event
        }
        for event in events:
            if event.get("type") != "chunk_sample":
                continue
            dyn = event.get("dynamics_prediction")
            if not dyn:
                skipped["missing_dynamics_prediction"] += 1
                continue
            pred_states = np.asarray(dyn.get("selected", {}).get("state", []), dtype=np.float64)
            if pred_states.ndim != 2 or pred_states.shape[1] != 19:
                skipped["bad_prediction_shape"] += 1
                continue
            base_decision = int(event["decision_idx_before"])
            for horizon_idx, pred_state in enumerate(pred_states, start=1):
                actual = actual_by_decision.get(base_decision + horizon_idx)
                if actual is None:
                    skipped["missing_actual_reached_state"] += 1
                    continue
                state_err = pred_state - actual
                eef_pos_err = state_err[0:3]
                obj_pos_err = state_err[10:13]
                eef_rot_err_deg = float(_rot_angle_deg(pred_state[3:9][None, :], actual[3:9][None, :])[0])
                obj_rot_err_deg = float(_rot_angle_deg(pred_state[13:19][None, :], actual[13:19][None, :])[0])
                rows.append({
                    "run_dir": str(run_dir),
                    "rollout": rollout_dir.name,
                    "chunk_idx": int(event["chunk_idx"]),
                    "selected_candidate": int(event.get("selected_candidate", -1)),
                    "decision_idx_before": base_decision,
                    "decision_idx_actual": base_decision + horizon_idx,
                    "horizon": horizon_idx,
                    "eef_pos_err_mm": float(np.linalg.norm(eef_pos_err) * 1000.0),
                    "eef_x_abs_err_mm": float(abs(eef_pos_err[0]) * 1000.0),
                    "eef_y_abs_err_mm": float(abs(eef_pos_err[1]) * 1000.0),
                    "eef_z_abs_err_mm": float(abs(eef_pos_err[2]) * 1000.0),
                    "cheezit_pos_err_mm": float(np.linalg.norm(obj_pos_err) * 1000.0),
                    "cheezit_x_abs_err_mm": float(abs(obj_pos_err[0]) * 1000.0),
                    "cheezit_y_abs_err_mm": float(abs(obj_pos_err[1]) * 1000.0),
                    "cheezit_z_abs_err_mm": float(abs(obj_pos_err[2]) * 1000.0),
                    "eef_rot_err_deg": eef_rot_err_deg,
                    "cheezit_rot_err_deg": obj_rot_err_deg,
                    "eef_rot6d_l2": float(np.linalg.norm(state_err[3:9])),
                    "cheezit_rot6d_l2": float(np.linalg.norm(state_err[13:19])),
                    "gripper_abs_err": float(abs(state_err[9])),
                    "gripper_sign_match": bool(np.sign(pred_state[9]) == np.sign(actual[9])),
                    "state_l2": float(np.linalg.norm(state_err)),
                })
    return rows, dict(skipped)


def summarize(rows: list[dict], skipped: dict) -> dict:
    metrics = [
        "eef_pos_err_mm",
        "cheezit_pos_err_mm",
        "eef_rot_err_deg",
        "cheezit_rot_err_deg",
        "eef_rot6d_l2",
        "cheezit_rot6d_l2",
        "gripper_abs_err",
        "state_l2",
    ]
    summary = {
        "n_aligned_prediction_steps": len(rows),
        "skipped": skipped,
        "metrics": {metric: _percentiles(np.asarray([row[metric] for row in rows])) for metric in metrics},
        "by_horizon": {},
    }
    for horizon in sorted({row["horizon"] for row in rows}):
        horizon_rows = [row for row in rows if row["horizon"] == horizon]
        summary["by_horizon"][str(horizon)] = {
            metric: _percentiles(np.asarray([row[metric] for row in horizon_rows]))
            for metric in metrics
        }
    if rows:
        summary["gripper_sign_accuracy"] = float(np.mean([row["gripper_sign_match"] for row in rows]))
        summary["n_rollouts"] = len(set(row["rollout"] for row in rows))
        summary["n_chunks"] = len(set((row["rollout"], row["chunk_idx"]) for row in rows))
    return summary


def write_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_cdf(ax, values: np.ndarray, label: str) -> None:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        return
    y = np.arange(1, values.size + 1) / values.size
    ax.plot(values, y, linewidth=2.0, label=label)


def write_cdf_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _plot_cdf(axes[0], [row["eef_pos_err_mm"] for row in rows], "EEF pos")
    _plot_cdf(axes[0], [row["cheezit_pos_err_mm"] for row in rows], "object pos")
    axes[0].set_xlabel("position error [mm]")
    axes[0].set_ylabel("CDF")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    _plot_cdf(axes[1], [row["eef_rot_err_deg"] for row in rows], "EEF rot")
    _plot_cdf(axes[1], [row["cheezit_rot_err_deg"] for row in rows], "object rot")
    axes[1].set_xlabel("rotation error [deg]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    _plot_cdf(axes[2], [row["state_l2"] for row in rows], "full 19D state")
    _plot_cdf(axes[2], [row["gripper_abs_err"] for row in rows], "gripper abs")
    axes[2].set_xlabel("raw state-space error")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle("Dynamics prediction error CDF, selected policy chunks")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_horizon_plot(rows: list[dict], path: Path) -> None:
    horizons = sorted({row["horizon"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, label in [
        ("eef_pos_err_mm", "EEF pos"),
        ("cheezit_pos_err_mm", "object pos"),
    ]:
        med = [np.percentile([row[metric] for row in rows if row["horizon"] == h], 50) for h in horizons]
        p90 = [np.percentile([row[metric] for row in rows if row["horizon"] == h], 90) for h in horizons]
        axes[0].plot(horizons, med, marker="o", label=f"{label} median")
        axes[0].plot(horizons, p90, marker=".", linestyle="--", label=f"{label} p90")
    axes[0].set_xlabel("prediction horizon [executed actions]")
    axes[0].set_ylabel("position error [mm]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    for metric, label in [
        ("eef_rot_err_deg", "EEF rot"),
        ("cheezit_rot_err_deg", "object rot"),
    ]:
        med = [np.percentile([row[metric] for row in rows if row["horizon"] == h], 50) for h in horizons]
        p90 = [np.percentile([row[metric] for row in rows if row["horizon"] == h], 90) for h in horizons]
        axes[1].plot(horizons, med, marker="o", label=f"{label} median")
        axes[1].plot(horizons, p90, marker=".", linestyle="--", label=f"{label} p90")
    axes[1].set_xlabel("prediction horizon [executed actions]")
    axes[1].set_ylabel("rotation error [deg]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Dynamics prediction error by horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs/real_world/paper_plots/dynamics_prediction_diagnostics") / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped = collect_rows(run_dir)
    summary = summarize(rows, skipped)
    summary["run_dir"] = str(run_dir)
    summary["output_dir"] = str(output_dir.resolve())

    write_rows_csv(rows, output_dir / "per_step_errors.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if rows:
        write_cdf_plot(rows, output_dir / "component_error_cdf.pdf")
        write_cdf_plot(rows, output_dir / "component_error_cdf.png")
        write_horizon_plot(rows, output_dir / "error_by_horizon.pdf")
        write_horizon_plot(rows, output_dir / "error_by_horizon.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
