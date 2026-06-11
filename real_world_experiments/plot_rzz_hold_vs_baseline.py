#!/usr/bin/env python3
"""Compare an object-rzz-hold rollout against a baseline left-bowl rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float64)
    a1 = arr[..., 0:3]
    a2 = arr[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def obs_from_event(event: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
    if event.get("type") == "rollout_start":
        return event.get("obs"), int(event.get("decision_idx", 0))
    if event.get("type") == "target_reached":
        return event.get("reached_obs"), int(event.get("decision_idx", 0))
    return None, None


def collect(path: Path, name: str) -> dict[str, Any]:
    events = load_events(path)
    rows = []
    label_events = []
    rollout_end = None
    guidance = []
    for event in events:
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        if event.get("type") == "rollout_end":
            rollout_end = event
        if event.get("type") == "chunk_sample":
            log = ((event.get("selection") or {}).get("object_rzz_hold_guidance") or {})
            if log:
                guidance.append({"decision_before": int(event.get("decision_idx_before", 0)), **log})
        obs, decision = obs_from_event(event)
        if obs is None or "cheezit_rot6d" not in obs:
            continue
        mat = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float64)[None])[0]
        rows.append(
            {
                "decision": int(decision),
                "label": np.asarray(event.get("current_label") or event.get("label") or [0, 0, 0], dtype=int),
                "eef": np.asarray(obs["eef_pos"], dtype=np.float64),
                "obj": np.asarray(obs["cheezit_pos"], dtype=np.float64),
                "rzz": float(mat[2, 2]),
            }
        )
    if not rows:
        raise ValueError(f"No trajectory observations found in {path}")
    grab_dec = None
    pour_dec = None
    for event in label_events:
        if event.get("label_name") == "can_grabbed" and int(event.get("to", -1)) == 1:
            grab_dec = int(event["decision_idx"])
        if event.get("label_name") == "pouring_left" and int(event.get("to", -1)) == 1:
            pour_dec = int(event["decision_idx"])
    if grab_dec is None:
        grabbed = [row for row in rows if row["label"][0] > 0]
        grab_dec = grabbed[0]["decision"] if grabbed else rows[0]["decision"]
    grab_idx = next((i for i, row in enumerate(rows) if row["decision"] >= grab_dec), 0)
    rzz0 = rows[grab_idx]["rzz"]
    decisions = np.asarray([row["decision"] for row in rows], dtype=np.int32)
    rel_decisions = decisions - grab_dec
    eef = np.asarray([row["eef"] for row in rows], dtype=np.float64)
    obj = np.asarray([row["obj"] for row in rows], dtype=np.float64)
    rzz = np.asarray([row["rzz"] for row in rows], dtype=np.float64)
    tilt = np.degrees(np.arccos(np.clip(rzz, -1.0, 1.0)))
    tilt0 = float(tilt[grab_idx])
    rzz_err = np.abs(rzz - rzz0)
    first_dev = {}
    for threshold in (0.01, 0.02, 0.03, 0.05):
        idx = next((i for i in range(grab_idx, len(rows)) if rzz_err[i] >= threshold), None)
        first_dev[str(threshold)] = None if idx is None else {
            "decision": int(decisions[idx]),
            "relative_decision": int(rel_decisions[idx]),
            "eef_xy": eef[idx, :2].astype(float).tolist(),
            "object_xy": obj[idx, :2].astype(float).tolist(),
            "rzz_error": float(rzz_err[idx]),
        }
    return {
        "name": name,
        "path": str(path),
        "events": events,
        "label_events": label_events,
        "rollout_end": rollout_end,
        "decisions": decisions,
        "relative_decisions": rel_decisions,
        "eef": eef,
        "obj": obj,
        "rzz": rzz,
        "rzz0": float(rzz0),
        "rzz_error": rzz_err,
        "tilt": tilt,
        "tilt0": tilt0,
        "grab_decision": int(grab_dec),
        "pour_decision": None if pour_dec is None else int(pour_dec),
        "guidance": guidance,
        "first_rzz_deviation": first_dev,
        "success": None if rollout_end is None else bool(rollout_end.get("success")),
    }


def plot_pair(baseline: dict[str, Any], rzz_hold: dict[str, Any], out_path: Path, threshold_x: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_xy, ax_err, ax_eefx, ax_guidance = axes.flat
    colors = {baseline["name"]: "#7f7f7f", rzz_hold["name"]: "#1f77b4"}

    for item in (baseline, rzz_hold):
        color = colors[item["name"]]
        ax_xy.plot(item["eef"][:, 0], item["eef"][:, 1], color=color, linewidth=1.5, label=f"{item['name']} EEF")
        ax_xy.plot(item["obj"][:, 0], item["obj"][:, 1], color=color, linestyle="--", linewidth=1.3, label=f"{item['name']} object")
        ax_xy.scatter(item["obj"][0, 0], item["obj"][0, 1], color=color, marker="s", s=35)
        ax_xy.scatter(item["obj"][-1, 0], item["obj"][-1, 1], color=color, marker="x", s=45)
        for event in item["label_events"]:
            decision = int(event["decision_idx"])
            idx = int(np.argmin(np.abs(item["decisions"] - decision)))
            marker = "*" if event["label_name"] == "pouring_left" else "o"
            ax_xy.scatter(item["eef"][idx, 0], item["eef"][idx, 1], color=color, marker=marker, s=70, zorder=4)
    ax_xy.axvline(threshold_x, color="#d62728", linestyle=":", linewidth=1.1, label=f"rzz gate x={threshold_x:.2f}")
    ax_xy.set_title("Top-down trajectory")
    ax_xy.set_xlabel("world x [m]")
    ax_xy.set_ylabel("world y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(fontsize=7, ncol=2)

    for item in (baseline, rzz_hold):
        color = colors[item["name"]]
        ax_err.plot(item["relative_decisions"], item["rzz_error"], color=color, linewidth=1.6, label=item["name"])
        if item["pour_decision"] is not None:
            ax_err.axvline(item["pour_decision"] - item["grab_decision"], color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_err.axhline(0.02, color="#d62728", linestyle=":", linewidth=1.1, label="|drzz|=0.02")
    ax_err.set_title("|rzz - rzz_at_grasp| after grasp")
    ax_err.set_xlabel("decisions after grasp")
    ax_err.set_ylabel("absolute rzz error")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend()

    for item in (baseline, rzz_hold):
        color = colors[item["name"]]
        ax_eefx.plot(item["relative_decisions"], item["eef"][:, 0], color=color, linewidth=1.6, label=item["name"])
        if item["pour_decision"] is not None:
            ax_eefx.axvline(item["pour_decision"] - item["grab_decision"], color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_eefx.axhline(threshold_x, color="#d62728", linestyle=":", linewidth=1.1, label="hold gate")
    ax_eefx.set_title("EEF x after grasp")
    ax_eefx.set_xlabel("decisions after grasp")
    ax_eefx.set_ylabel("world x [m]")
    ax_eefx.grid(True, alpha=0.25)
    ax_eefx.legend()

    applied = [g for g in rzz_hold["guidance"] if g.get("applied")]
    skipped = [g for g in rzz_hold["guidance"] if not g.get("applied")]
    if applied:
        xs = np.asarray([g["decision_before"] - rzz_hold["grab_decision"] for g in applied], dtype=float)
        ax_guidance.plot(xs, [g.get("pre_mean_abs_rzz_error", np.nan) for g in applied], marker="o", color="#d62728", linestyle="--", label="pre")
        ax_guidance.plot(xs, [g.get("post_mean_abs_rzz_error", np.nan) for g in applied], marker="o", color="#2ca02c", label="post")
        ax2 = ax_guidance.twinx()
        ax2.plot(xs, [g.get("mean_rotation_action_delta_l2", np.nan) for g in applied], marker="x", color="#6a4c93", label="rot action delta")
        ax2.set_ylabel("rotation action delta L2")
        lines, labels = ax_guidance.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_guidance.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    ax_guidance.set_title(f"rzz-hold optimizer logs ({len(applied)} applied, {len(skipped)} skipped)")
    ax_guidance.set_xlabel("decisions after grasp")
    ax_guidance.set_ylabel("mean abs rzz error")
    ax_guidance.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def compact_summary(item: dict[str, Any]) -> dict[str, Any]:
    pour_rel = None if item["pour_decision"] is None else item["pour_decision"] - item["grab_decision"]
    before_gate = item["eef"][:, 0] > -0.60
    after_grasp = item["decisions"] >= item["grab_decision"]
    mask = before_gate & after_grasp
    return {
        "events": item["path"],
        "success": item["success"],
        "grab_decision": item["grab_decision"],
        "pour_decision": item["pour_decision"],
        "pour_relative_decision": None if pour_rel is None else int(pour_rel),
        "rzz_at_grasp": item["rzz0"],
        "max_abs_rzz_error_before_x_gate": None if not np.any(mask) else float(np.max(item["rzz_error"][mask])),
        "mean_abs_rzz_error_before_x_gate": None if not np.any(mask) else float(np.mean(item["rzz_error"][mask])),
        "first_rzz_deviation": item["first_rzz_deviation"],
        "final_label_events": item["label_events"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--rzz-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default="rzz_hold_vs_baseline_left_rollout_comparison")
    parser.add_argument("--threshold-x", type=float, default=-0.60)
    args = parser.parse_args()

    baseline = collect(args.baseline_events, "baseline left")
    rzz_hold = collect(args.rzz_events, "rzz hold")
    out_path = args.output_dir / f"{args.name}.png"
    plot_pair(baseline, rzz_hold, out_path, args.threshold_x)
    summary = {
        "baseline": compact_summary(baseline),
        "rzz_hold": compact_summary(rzz_hold),
        "threshold_x": args.threshold_x,
        "plot_png": str(out_path),
        "plot_pdf": str(out_path.with_suffix(".pdf")),
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
