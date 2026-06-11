#!/usr/bin/env python3
"""Plot relative Cheez-It Euler components with dynamics-model branches."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from real_world_experiments.real_world_data import rot6d_to_matrix
except ModuleNotFoundError:
    from real_world_data import rot6d_to_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_release_regrasp_right_left_cycle3_epoch160/rollouts/rollout_000/events.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"
AXIS_NAMES = ("x/roll", "y/pitch", "z/yaw")
COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": [
                "Computer Modern Typewriter",
                "CMU Typewriter Text",
                "DejaVu Sans Mono",
            ],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "rollout_start":
        return event.get("obs")
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") in {"chunk_sample", "decision"}:
        return event.get("obs")
    return None


def selected_prediction_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    pred = event.get("dynamics_prediction")
    if not isinstance(pred, dict):
        return None
    selected = pred.get("selected")
    if not isinstance(selected, dict):
        return None
    if "cheezit_rot6d" not in selected:
        return None
    return selected


def relative_euler(rot6d: np.ndarray, reference_matrix: np.ndarray) -> np.ndarray:
    matrices = rot6d_to_matrix(np.asarray(rot6d, dtype=np.float32))
    rel = np.einsum("ij,...jk->...ik", reference_matrix.T, matrices)
    euler = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=True)
    euler = euler.reshape(rel.shape[:-2] + (3,))
    return euler


def unwrap_euler_series(euler: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(euler), axis=0))


def first_event_decision(events: list[dict[str, Any]], label_name: str, to_value: int) -> int | None:
    for event in events:
        if event.get("type") != "target_reached":
            continue
        for label_event in event.get("label_events", []) or []:
            if label_event.get("label_name") == label_name and int(label_event.get("to", 0)) == to_value:
                return int(label_event.get("decision_idx", event.get("decision_idx", 0)))
    return None


def true_series(events: list[dict[str, Any]], pickup_decision: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    decisions = []
    rot6d = []
    labels = []
    label_events = []
    ref_matrix = None
    for event in events:
        obs = obs_from_event(event)
        if obs is None or event.get("type") not in {"rollout_start", "target_reached"}:
            continue
        decision = int(event.get("decision_idx", 0))
        if decision == pickup_decision:
            ref_matrix = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float32)[None])[0]
        decisions.append(decision)
        rot6d.append(obs["cheezit_rot6d"])
        labels.append(event.get("label") or event.get("current_label") or [0, 0, 0])
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
    if ref_matrix is None:
        raise ValueError(f"Could not find pickup reference at decision {pickup_decision}")
    euler = unwrap_euler_series(relative_euler(np.asarray(rot6d, dtype=np.float32), ref_matrix))
    return np.asarray(decisions, dtype=np.int32), euler, np.asarray(labels, dtype=np.int32), label_events, ref_matrix


def collect_branches(
    events: list[dict[str, Any]],
    pickup_decision: int,
    reference_matrix: np.ndarray,
    *,
    window_steps: int,
) -> list[dict[str, Any]]:
    branches = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        selected = selected_prediction_from_event(event)
        if selected is None:
            continue
        start = int(event.get("decision_idx_before", event.get("decision_idx", 0)))
        rel_start = start - pickup_decision
        if rel_start < 0 or rel_start > window_steps:
            continue
        pred_rot6d = np.asarray(selected["cheezit_rot6d"], dtype=np.float32)
        pred_euler = relative_euler(pred_rot6d, reference_matrix)
        branches.append(
            {
                "chunk_idx": int(event.get("chunk_idx", len(branches))),
                "start_decision": start,
                "rel_start": rel_start,
                "x": rel_start + np.arange(1, len(pred_euler) + 1, dtype=np.int32),
                "euler": pred_euler,
            }
        )
    return branches


def label_intervals(decisions: np.ndarray, labels: np.ndarray, pickup_decision: int, label_idx: int) -> list[tuple[int, int]]:
    intervals = []
    active_start = None
    for idx, decision in enumerate(decisions):
        rel_decision = int(decision - pickup_decision)
        active = bool(labels[idx, label_idx] > 0)
        if active and active_start is None:
            active_start = rel_decision
        if not active and active_start is not None:
            intervals.append((active_start, int(decisions[max(0, idx - 1)] - pickup_decision)))
            active_start = None
    if active_start is not None:
        intervals.append((active_start, int(decisions[-1] - pickup_decision)))
    return intervals


def align_branch_euler_to_start(pred_euler: np.ndarray, true_start: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(pred_euler, dtype=np.float64)
    for axis in range(3):
        seq = np.concatenate([[float(true_start[axis])], np.asarray(pred_euler[:, axis], dtype=np.float64)])
        aligned[:, axis] = np.degrees(np.unwrap(np.radians(seq)))[1:]
    return aligned


def branch_error_rows(rel_decisions: np.ndarray, true_euler: np.ndarray, branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_by_step = {int(step): true_euler[idx] for idx, step in enumerate(rel_decisions)}
    rows = []
    for branch in branches:
        true_start = np.asarray([np.interp(branch["rel_start"], rel_decisions, true_euler[:, axis]) for axis in range(3)])
        aligned_pred = align_branch_euler_to_start(branch["euler"], true_start)
        for horizon, (step, pred) in enumerate(zip(branch["x"], aligned_pred), start=1):
            true = true_by_step.get(int(step))
            if true is None:
                continue
            for axis, name in enumerate(AXIS_NAMES):
                rows.append(
                    {
                        "chunk_idx": int(branch["chunk_idx"]),
                        "rel_start": int(branch["rel_start"]),
                        "horizon": int(horizon),
                        "rel_decision": int(step),
                        "axis": name,
                        "pred_deg": float(pred[axis]),
                        "true_deg": float(true[axis]),
                        "abs_err_deg": float(abs(pred[axis] - true[axis])),
                    }
                )
    return rows


def summarize_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for axis in AXIS_NAMES:
        vals = np.asarray([row["abs_err_deg"] for row in rows if row["axis"] == axis], dtype=np.float64)
        if len(vals) == 0:
            continue
        summary[axis] = {
            "n": int(len(vals)),
            "mean_abs_err_deg": float(np.mean(vals)),
            "median_abs_err_deg": float(np.median(vals)),
            "p90_abs_err_deg": float(np.percentile(vals, 90)),
            "max_abs_err_deg": float(np.max(vals)),
        }
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: Path,
    rel_decisions: np.ndarray,
    true_euler: np.ndarray,
    labels: np.ndarray,
    label_events: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    pickup_decision: int,
    *,
    window_steps: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.2), sharex=True)
    right_intervals = label_intervals(rel_decisions + pickup_decision, labels, pickup_decision, 1)
    left_intervals = label_intervals(rel_decisions + pickup_decision, labels, pickup_decision, 2)
    for axis, ax in enumerate(axes):
        for idx, (start, end) in enumerate(right_intervals):
            ax.axvspan(start, end, color="#ffcc00", alpha=0.18, label="pouring_right" if idx == 0 else None)
        for idx, (start, end) in enumerate(left_intervals):
            ax.axvspan(start, end, color="#8ecae6", alpha=0.18, label="pouring_left" if idx == 0 else None)
        wrote_branch_label = False
        for branch in branches:
            true_start_vec = np.asarray([np.interp(branch["rel_start"], rel_decisions, true_euler[:, j]) for j in range(3)])
            aligned_pred = align_branch_euler_to_start(branch["euler"], true_start_vec)
            true_start = float(true_start_vec[axis])
            x = np.concatenate([[branch["rel_start"]], branch["x"]])
            y = np.concatenate([[true_start], aligned_pred[:, axis]])
            ax.plot(
                x,
                y,
                color="#d000ff",
                alpha=0.38,
                linewidth=1.0,
                label="WM selected chunk" if not wrote_branch_label else None,
            )
            wrote_branch_label = True
        ax.plot(rel_decisions, true_euler[:, axis], color=COLORS[axis], linewidth=1.7, label=f"true {AXIS_NAMES[axis]}")
        for event in label_events:
            rel = int(event.get("decision_idx", 0)) - pickup_decision
            if rel < 0 or rel > window_steps:
                continue
            value = float(np.interp(rel, rel_decisions, true_euler[:, axis]))
            ax.scatter(rel, value, s=42, marker="*", color="#111111", zorder=5)
            ax.text(rel, value, f" {event.get('label_name')} {event.get('from')}->{event.get('to')}", fontsize=7, va="center")
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.8)
        ax.set_ylabel("deg")
        ax.set_title(AXIS_NAMES[axis])
        ax.set_xlim(0, window_steps)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("decisions after first grasp")
    fig.suptitle("Cheez-It relative XYZ Euler: true trace and pure world-model selected-chunk branches", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-steps", type=int, default=90)
    parser.add_argument("--name", type=str, default="release_regrasp_rollout000_first_right_tilt_euler_wm_branches")
    args = parser.parse_args()

    configure_matplotlib()
    events = load_events(args.events)
    pickup_decision = first_event_decision(events, "can_grabbed", 1)
    if pickup_decision is None:
        raise ValueError("Could not find can_grabbed 0->1 event")
    decisions, true_euler, labels, label_events, reference_matrix = true_series(events, pickup_decision)
    rel_decisions = decisions - pickup_decision
    mask = (rel_decisions >= 0) & (rel_decisions <= args.window_steps)
    branches = collect_branches(events, pickup_decision, reference_matrix, window_steps=args.window_steps)
    rows = branch_error_rows(rel_decisions[mask], true_euler[mask], branches)
    out_png = args.output_dir / f"{args.name}.png"
    write_plot(
        out_png,
        rel_decisions[mask],
        true_euler[mask],
        labels[mask],
        label_events,
        branches,
        pickup_decision,
        window_steps=args.window_steps,
    )
    csv_path = args.output_dir / f"{args.name}_errors.csv"
    write_rows_csv(csv_path, rows)
    summary = {
        "events": str(args.events),
        "pickup_decision": int(pickup_decision),
        "window_steps": int(args.window_steps),
        "n_true_points": int(np.sum(mask)),
        "n_branches": int(len(branches)),
        "axis_error_summary": summarize_errors(rows),
        "plot_png": str(out_png),
        "plot_pdf": str(out_png.with_suffix(".pdf")),
        "errors_csv": str(csv_path),
        "euler_convention": "relative to first pickup, scipy Rotation.as_euler('xyz')",
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
