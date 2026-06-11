#!/usr/bin/env python3
"""Plot true Cheez-It angle with dynamics-world-model branch predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "rollout_start":
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


def angle_from_rot6d(rot6d: np.ndarray, mode: str, reference_rot6d: np.ndarray | None = None) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    R = rot6d_to_matrix(rot6d)
    if mode == "tilt":
        z_axis = R[..., :, 2]
        denom = np.maximum(np.linalg.norm(z_axis, axis=-1), 1e-8)
        return np.degrees(np.arccos(np.clip(z_axis[..., 2] / denom, -1.0, 1.0)))
    if mode == "relative":
        if reference_rot6d is None:
            raise ValueError("relative angle requires a reference_rot6d")
        R0 = rot6d_to_matrix(np.asarray(reference_rot6d, dtype=np.float32))
        rel = np.einsum("ij,...jk->...ik", R0.T, R)
        trace = np.trace(rel, axis1=-2, axis2=-1)
        return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    if mode == "yaw":
        yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
        return np.degrees(np.unwrap(yaw))
    raise ValueError(f"Unknown angle mode {mode!r}")


def collect_true_series(events: list[dict[str, Any]], angle_mode: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    decisions = []
    rot6d = []
    label_events = []
    for event in events:
        obs = obs_from_event(event)
        if obs is None:
            continue
        decisions.append(int(event.get("decision_idx", 0)))
        rot6d.append(obs["cheezit_rot6d"])
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
    if not rot6d:
        raise ValueError("No Cheez-It observations found")
    rot6d_arr = np.asarray(rot6d, dtype=np.float32)
    angles = angle_from_rot6d(rot6d_arr, angle_mode, reference_rot6d=rot6d_arr[0])
    return np.asarray(decisions, dtype=np.int32), angles, label_events, rot6d_arr[0]


def collect_branches(
    events: list[dict[str, Any]],
    angle_mode: str,
    reference_rot6d: np.ndarray,
) -> list[dict[str, Any]]:
    branches = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        selected = selected_prediction_from_event(event)
        if selected is None:
            continue
        start = int(event.get("decision_idx_before", event.get("decision_idx", 0)))
        pred_rot6d = np.asarray(selected["cheezit_rot6d"], dtype=np.float32)
        pred_angle = angle_from_rot6d(pred_rot6d, angle_mode, reference_rot6d=reference_rot6d)
        branches.append(
            {
                "chunk_idx": int(event.get("chunk_idx", len(branches))),
                "start_decision": start,
                "x": start + np.arange(1, len(pred_angle) + 1, dtype=np.int32),
                "angle": pred_angle,
            }
        )
    return branches


def downsample_branches(branches: list[dict[str, Any]], max_branches: int | None) -> list[dict[str, Any]]:
    if max_branches is None or max_branches <= 0 or len(branches) <= max_branches:
        return branches
    keep = np.linspace(0, len(branches) - 1, max_branches, dtype=int)
    return [branches[idx] for idx in keep]


def branch_error_rows(true_decisions: np.ndarray, true_angles: np.ndarray, branches: list[dict[str, Any]]) -> list[dict[str, float]]:
    true_by_decision = {int(d): float(a) for d, a in zip(true_decisions, true_angles)}
    rows = []
    for branch in branches:
        for h_idx, (decision, pred_angle) in enumerate(zip(branch["x"], branch["angle"]), start=1):
            true_angle = true_by_decision.get(int(decision))
            if true_angle is None:
                continue
            rows.append(
                {
                    "chunk_idx": int(branch["chunk_idx"]),
                    "start_decision": int(branch["start_decision"]),
                    "horizon": int(h_idx),
                    "decision": int(decision),
                    "pred_angle_deg": float(pred_angle),
                    "true_angle_deg": true_angle,
                    "abs_err_deg": abs(float(pred_angle) - true_angle),
                }
            )
    return rows


def write_error_csv(path: Path, rows: list[dict[str, float]]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_errors(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    err = np.asarray([row["abs_err_deg"] for row in rows], dtype=np.float64)
    by_horizon = {}
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        vals = np.asarray([row["abs_err_deg"] for row in rows if int(row["horizon"]) == horizon], dtype=np.float64)
        by_horizon[str(horizon)] = {
            "n": int(len(vals)),
            "mean_abs_err_deg": float(np.mean(vals)),
            "median_abs_err_deg": float(np.median(vals)),
            "p90_abs_err_deg": float(np.percentile(vals, 90)),
        }
    return {
        "n_matched_predictions": int(len(err)),
        "mean_abs_err_deg": float(np.mean(err)),
        "median_abs_err_deg": float(np.median(err)),
        "p90_abs_err_deg": float(np.percentile(err, 90)),
        "max_abs_err_deg": float(np.max(err)),
        "by_horizon": by_horizon,
    }


def write_plot(
    path: Path,
    true_decisions: np.ndarray,
    true_angles: np.ndarray,
    branches: list[dict[str, Any]],
    label_events: list[dict[str, Any]],
    error_rows: list[dict[str, float]],
    *,
    angle_mode: str,
    title: str,
    xlim: tuple[float, float] | None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), sharex=False, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    branch_label_written = False
    for branch in branches:
        x = np.concatenate([[branch["start_decision"]], branch["x"]])
        start_angle = np.interp(branch["start_decision"], true_decisions, true_angles)
        y = np.concatenate([[start_angle], branch["angle"]])
        ax.plot(
            x,
            y,
            color="#d000ff",
            alpha=0.45,
            linewidth=1.15,
            label="WM selected chunk" if not branch_label_written else None,
        )
        branch_label_written = True
    ax.plot(true_decisions, true_angles, color="#111111", linewidth=1.5, label="true measured object angle")
    for event in label_events:
        decision = int(event.get("decision_idx", 0))
        if decision < true_decisions[0] or decision > true_decisions[-1]:
            continue
        angle = float(np.interp(decision, true_decisions, true_angles))
        ax.scatter(decision, angle, s=42, marker="*", zorder=5)
        ax.text(
            decision,
            angle,
            f" {event.get('label_name')} {event.get('from')}->{event.get('to')}",
            fontsize=7,
            va="center",
        )
    ylabel = {
        "tilt": "Cheez-It tilt from world vertical [deg]",
        "relative": "Cheez-It rotation from rollout start [deg]",
        "yaw": "Cheez-It yaw [deg]",
    }[angle_mode]
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    err_ax = axes[1]
    if error_rows:
        horizons = sorted({int(row["horizon"]) for row in error_rows})
        means = []
        p90s = []
        for horizon in horizons:
            vals = np.asarray([row["abs_err_deg"] for row in error_rows if int(row["horizon"]) == horizon], dtype=np.float64)
            means.append(float(np.mean(vals)))
            p90s.append(float(np.percentile(vals, 90)))
        err_ax.plot(horizons, means, "-o", color="#1f77b4", label="mean abs error")
        err_ax.plot(horizons, p90s, "-o", color="#d62728", label="p90 abs error")
        err_ax.set_xticks(horizons)
        err_ax.legend(loc="upper left")
    err_ax.set_xlabel("world-model horizon step")
    err_ax.set_ylabel("angle error [deg]")
    err_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--angle-mode", choices=["tilt", "relative", "yaw"], default="tilt")
    parser.add_argument("--max-branches", type=int, default=160, help="Downsample plotted branches only; error stats still use all branches.")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    configure_matplotlib()
    events = load_events(args.events)
    true_decisions, true_angles, label_events, reference_rot6d = collect_true_series(events, args.angle_mode)
    branches = collect_branches(events, args.angle_mode, reference_rot6d)
    plot_branches = downsample_branches(branches, args.max_branches)
    error_rows = branch_error_rows(true_decisions, true_angles, branches)
    error_summary = summarize_errors(error_rows)

    stem = args.name or f"{args.events.parent.parent.parent.name}_{args.events.parent.name}_{args.angle_mode}_wm_branches"
    out_png = args.output_dir / f"{stem}.png"
    title = f"{args.events.parent.parent.parent.name}/{args.events.parent.name}: object {args.angle_mode} with pure dynamics-model branches"
    xlim = tuple(args.xlim) if args.xlim is not None else None
    write_plot(
        out_png,
        true_decisions,
        true_angles,
        plot_branches,
        label_events,
        error_rows,
        angle_mode=args.angle_mode,
        title=title,
        xlim=xlim,
    )
    write_error_csv(args.output_dir / f"{stem}_errors.csv", error_rows)
    summary = {
        "events": str(args.events),
        "angle_mode": args.angle_mode,
        "n_true_points": int(len(true_angles)),
        "n_branches": int(len(branches)),
        "true_angle_deg": {
            "min": float(np.min(true_angles)),
            "median": float(np.median(true_angles)),
            "max": float(np.max(true_angles)),
            "first": float(true_angles[0]),
            "last": float(true_angles[-1]),
        },
        "error_summary": error_summary,
        "plot_png": str(out_png),
        "plot_pdf": str(out_png.with_suffix(".pdf")),
        "errors_csv": str(args.output_dir / f"{stem}_errors.csv"),
        "xlim": list(xlim) if xlim is not None else None,
    }
    write_path = args.output_dir / f"{stem}_summary.json"
    write_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
