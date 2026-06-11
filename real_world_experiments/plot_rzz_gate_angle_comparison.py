#!/usr/bin/env python3
"""Compare can upright angle before an EEF-x gate for baseline vs rzz-hold runs."""

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def discover_events(dirs: list[Path]) -> list[Path]:
    events: list[Path] = []
    for root in dirs:
        events.extend(sorted(root.glob("rollouts/rollout_*/events.jsonl")))
    return events


def collect(path: Path, group: str) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    label_events: list[dict[str, Any]] = []
    rollout_end: dict[str, Any] | None = None

    for event in load_jsonl(path):
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        if event.get("type") == "rollout_end":
            rollout_end = event

        obs, decision = obs_from_event(event)
        if obs is None or decision is None:
            continue
        if "eef_pos" not in obs or "cheezit_rot6d" not in obs:
            continue

        matrix = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float64)[None])[0]
        rzz = float(matrix[2, 2])
        rows.append(
            {
                "decision": int(decision),
                "label": np.asarray(event.get("current_label") or event.get("label") or [0, 0, 0], dtype=np.int32),
                "eef": np.asarray(obs["eef_pos"], dtype=np.float64),
                "rzz": rzz,
                "angle_deg": math.degrees(math.acos(float(np.clip(rzz, -1.0, 1.0)))),
            }
        )

    if not rows:
        return None

    grab_decision = None
    pour_left_decision = None
    pour_right_decision = None
    for event in label_events:
        if event.get("label_name") == "can_grabbed" and int(event.get("to", -1)) == 1:
            grab_decision = int(event["decision_idx"])
        if event.get("label_name") == "pouring_left" and int(event.get("to", -1)) == 1:
            pour_left_decision = int(event["decision_idx"])
        if event.get("label_name") == "pouring_right" and int(event.get("to", -1)) == 1:
            pour_right_decision = int(event["decision_idx"])

    if grab_decision is None:
        grabbed = [row for row in rows if row["label"].shape[0] > 0 and row["label"][0] > 0]
        if not grabbed:
            return None
        grab_decision = int(grabbed[0]["decision"])

    decisions = np.asarray([row["decision"] for row in rows], dtype=np.int32)
    eef = np.asarray([row["eef"] for row in rows], dtype=np.float64)
    angle_deg = np.asarray([row["angle_deg"] for row in rows], dtype=np.float64)
    rzz = np.asarray([row["rzz"] for row in rows], dtype=np.float64)

    return {
        "path": str(path),
        "run": path.parent.parent.parent.name,
        "rollout": path.parent.name,
        "name": f"{path.parent.parent.parent.name}/{path.parent.name}",
        "group": group,
        "success": None if rollout_end is None else bool(rollout_end.get("success")),
        "termination_reason": None if rollout_end is None else rollout_end.get("reason"),
        "grab_decision": int(grab_decision),
        "pour_left_decision": pour_left_decision,
        "pour_right_decision": pour_right_decision,
        "decisions": decisions,
        "relative_decision": decisions - int(grab_decision),
        "eef_x": eef[:, 0],
        "eef_y": eef[:, 1],
        "angle_deg": angle_deg,
        "rzz": rzz,
    }


def gate_index(item: dict[str, Any], gate_x: float) -> int | None:
    rel = item["relative_decision"]
    for idx in np.where(rel >= 0)[0]:
        if float(item["eef_x"][idx]) <= gate_x:
            return int(idx)
    return None


def truncated(item: dict[str, Any], gate_x: float) -> tuple[np.ndarray, np.ndarray, int | None]:
    rel = item["relative_decision"]
    angle = item["angle_deg"]
    idx0_candidates = np.where(rel >= 0)[0]
    if len(idx0_candidates) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), None
    start = int(idx0_candidates[0])
    gate = gate_index(item, gate_x)
    end = len(rel) - 1 if gate is None else gate
    return rel[start : end + 1], angle[start : end + 1], gate


def group_stats(items: list[dict[str, Any]], gate_x: float, angle_success_threshold_deg: float) -> dict[str, Any]:
    gate_rows = []
    no_gate = []
    for item in items:
        idx = gate_index(item, gate_x)
        if idx is None:
            no_gate.append(item["name"])
            continue
        rel, angle, _ = truncated(item, gate_x)
        max_angle = float(np.max(angle)) if angle.size else float(item["angle_deg"][idx])
        gate_rows.append(
            {
                "name": item["name"],
                "path": item["path"],
                "success": item["success"],
                "termination_reason": item["termination_reason"],
                "gate_relative_decision": int(item["relative_decision"][idx]),
                "gate_decision": int(item["decisions"][idx]),
                "gate_eef_x": float(item["eef_x"][idx]),
                "gate_eef_y": float(item["eef_y"][idx]),
                "gate_angle_deg": float(item["angle_deg"][idx]),
                "gate_rzz": float(item["rzz"][idx]),
                "max_angle_deg_until_gate": max_angle,
                "angle_success": bool(max_angle <= angle_success_threshold_deg),
                "pour_left_relative_decision": (
                    None
                    if item["pour_left_decision"] is None
                    else int(item["pour_left_decision"] - item["grab_decision"])
                ),
                "pour_right_relative_decision": (
                    None
                    if item["pour_right_decision"] is None
                    else int(item["pour_right_decision"] - item["grab_decision"])
                ),
            }
        )

    angles = np.asarray([row["gate_angle_deg"] for row in gate_rows], dtype=np.float64)
    max_angles = np.asarray([row["max_angle_deg_until_gate"] for row in gate_rows], dtype=np.float64)
    rels = np.asarray([row["gate_relative_decision"] for row in gate_rows], dtype=np.float64)
    return {
        "n": len(items),
        "n_gate_reached": len(gate_rows),
        "n_gate_missing": len(no_gate),
        "n_label_success": int(sum(bool(item["success"]) for item in items)),
        "angle_success_threshold_deg": angle_success_threshold_deg,
        "n_angle_success": int(sum(bool(row["angle_success"]) for row in gate_rows)),
        "gate_missing": no_gate,
        "gate_rows": gate_rows,
        "gate_angle_deg_mean": None if angles.size == 0 else float(np.mean(angles)),
        "gate_angle_deg_median": None if angles.size == 0 else float(np.median(angles)),
        "gate_angle_deg_min": None if angles.size == 0 else float(np.min(angles)),
        "gate_angle_deg_max": None if angles.size == 0 else float(np.max(angles)),
        "gate_angle_deg_p10_p50_p90": None if angles.size == 0 else np.percentile(angles, [10, 50, 90]).tolist(),
        "max_angle_deg_until_gate_mean": None if max_angles.size == 0 else float(np.mean(max_angles)),
        "max_angle_deg_until_gate_median": None if max_angles.size == 0 else float(np.median(max_angles)),
        "max_angle_deg_until_gate_min": None if max_angles.size == 0 else float(np.min(max_angles)),
        "max_angle_deg_until_gate_max": None if max_angles.size == 0 else float(np.max(max_angles)),
        "max_angle_deg_until_gate_p10_p50_p90": (
            None if max_angles.size == 0 else np.percentile(max_angles, [10, 50, 90]).tolist()
        ),
        "gate_relative_decision_p10_p50_p90": None if rels.size == 0 else np.percentile(rels, [10, 50, 90]).tolist(),
    }


def threshold_sweep(
    baseline_rows: list[dict[str, Any]],
    guided_rows: list[dict[str, Any]],
    max_angle: float,
    step: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    thresholds = np.arange(0.0, max_angle + 0.5 * step, step)
    rows = []
    for threshold in thresholds:
        b_pass = [row for row in baseline_rows if row["max_angle_deg_until_gate"] <= threshold]
        g_pass = [row for row in guided_rows if row["max_angle_deg_until_gate"] <= threshold]
        rows.append(
            {
                "angle_threshold_deg": float(threshold),
                "baseline_pass": len(b_pass),
                "baseline_total": len(baseline_rows),
                "guided_pass": len(g_pass),
                "guided_total": len(guided_rows),
            }
        )

    candidates = [row for row in rows if row["baseline_pass"] >= 1]
    if not candidates:
        return rows, None
    best = max(candidates, key=lambda row: (row["guided_pass"], -row["baseline_pass"], -row["angle_threshold_deg"]))
    return rows, best


def filter_by_required_label_event(items: list[dict[str, Any]], label_name: str | None) -> list[dict[str, Any]]:
    if label_name is None:
        return items
    if label_name == "pouring_left":
        return [item for item in items if item.get("pour_left_decision") is not None]
    if label_name == "pouring_right":
        return [item for item in items if item.get("pour_right_decision") is not None]
    if label_name == "can_grabbed":
        return [item for item in items if item.get("grab_decision") is not None]
    raise ValueError(f"Unsupported label filter: {label_name}")


def plot(
    baseline: list[dict[str, Any]],
    guided: list[dict[str, Any]],
    summary: dict[str, Any],
    out_path: Path,
    gate_x: float,
    angle_ylim: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax_traces, ax_dist, ax_sweep = axes

    for item in baseline:
        rel, angle, gate = truncated(item, gate_x)
        if rel.size == 0:
            continue
        ax_traces.plot(rel, angle, color="#9a9a9a", alpha=0.45, linewidth=1.0)
        if gate is not None:
            ax_traces.scatter(rel[-1], angle[-1], s=18, color="#555555", alpha=0.7)

    for item in guided:
        rel, angle, gate = truncated(item, gate_x)
        if rel.size == 0:
            continue
        ax_traces.plot(rel, angle, color="#1f77b4", alpha=0.75, linewidth=1.4)
        if gate is not None:
            ax_traces.scatter(rel[-1], angle[-1], s=24, color="#1f77b4", alpha=0.9)

    ax_traces.set_title(f"can angle until EEF x <= {gate_x:.2f}")
    ax_traces.set_xlabel("decisions after grasp")
    ax_traces.set_ylabel("can upright angle arccos(rzz) [deg]")
    ax_traces.set_ylim(*angle_ylim)
    ax_traces.grid(True, alpha=0.25)
    ax_traces.plot([], [], color="#777777", linewidth=2, label="baseline")
    ax_traces.plot([], [], color="#1f77b4", linewidth=2, label="rzz hold")
    ax_traces.legend(loc="best")

    baseline_angles = [row["max_angle_deg_until_gate"] for row in summary["baseline"]["gate_rows"]]
    guided_angles = [row["max_angle_deg_until_gate"] for row in summary["guided"]["gate_rows"]]
    rng = np.random.default_rng(7)
    ax_dist.scatter(
        rng.normal(0, 0.035, size=len(baseline_angles)),
        baseline_angles,
        color="#777777",
        alpha=0.75,
        label="baseline",
    )
    ax_dist.scatter(
        rng.normal(1, 0.035, size=len(guided_angles)),
        guided_angles,
        color="#1f77b4",
        alpha=0.85,
        label="rzz hold",
    )
    if baseline_angles:
        ax_dist.boxplot([baseline_angles, guided_angles], positions=[0, 1], widths=0.25, showfliers=False)
    best = summary.get("best_gate_angle_threshold")
    if best is not None:
        ax_dist.axhline(best["angle_threshold_deg"], color="#d62728", linestyle=":", linewidth=1.2)
    ax_dist.set_title("worst angle before gate")
    ax_dist.set_xticks([0, 1], ["baseline", "rzz hold"])
    ax_dist.set_ylabel("max angle until gate [deg]")
    ax_dist.set_ylim(*angle_ylim)
    ax_dist.grid(True, axis="y", alpha=0.25)

    sweep = summary["angle_threshold_sweep"]
    xs = [row["angle_threshold_deg"] for row in sweep]
    ax_sweep.plot(xs, [row["baseline_pass"] for row in sweep], color="#777777", label="baseline")
    ax_sweep.plot(xs, [row["guided_pass"] for row in sweep], color="#1f77b4", label="rzz hold")
    if best is not None:
        ax_sweep.axvline(best["angle_threshold_deg"], color="#d62728", linestyle=":", linewidth=1.2)
    ax_sweep.set_title("classifier: max pre-gate angle <= threshold")
    ax_sweep.set_xlabel("angle threshold [deg]")
    ax_sweep.set_ylabel("runs classified success")
    ax_sweep.set_xlim(*angle_ylim)
    ax_sweep.set_ylim(bottom=-0.2)
    ax_sweep.grid(True, alpha=0.25)
    ax_sweep.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", action="append", type=Path, required=True)
    parser.add_argument("--guided-dir", action="append", type=Path, required=True)
    parser.add_argument("--gate-x", type=float, default=-0.61)
    parser.add_argument("--max-angle", type=float, default=30.0)
    parser.add_argument("--threshold-step", type=float, default=0.5)
    parser.add_argument("--angle-success-threshold-deg", type=float, default=7.5)
    parser.add_argument("--require-label-event", choices=["can_grabbed", "pouring_right", "pouring_left"], default=None)
    parser.add_argument("--angle-ylim", type=float, nargs=2, default=(0.0, 35.0))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="rzz_hold_gate_angle_comparison")
    args = parser.parse_args()

    baseline = [item for path in discover_events(args.baseline_dir) if (item := collect(path, "baseline")) is not None]
    guided = [item for path in discover_events(args.guided_dir) if (item := collect(path, "guided")) is not None]
    baseline = filter_by_required_label_event(baseline, args.require_label_event)
    guided = filter_by_required_label_event(guided, args.require_label_event)
    if not baseline:
        raise RuntimeError("No baseline traces found.")
    if not guided:
        raise RuntimeError("No guided traces found.")

    baseline_stats = group_stats(baseline, args.gate_x, args.angle_success_threshold_deg)
    guided_stats = group_stats(guided, args.gate_x, args.angle_success_threshold_deg)
    sweep, best = threshold_sweep(
        baseline_stats["gate_rows"],
        guided_stats["gate_rows"],
        max_angle=args.max_angle,
        step=args.threshold_step,
    )

    summary = {
        "gate_x": args.gate_x,
        "baseline_dirs": [str(path) for path in args.baseline_dir],
        "guided_dirs": [str(path) for path in args.guided_dir],
        "require_label_event": args.require_label_event,
        "success_metric": {
            "name": "max_can_angle_until_eef_x_gate",
            "gate_x": args.gate_x,
            "threshold_deg": args.angle_success_threshold_deg,
            "definition": "success iff max arccos(rzz) from can_grabbed through first EEF x <= gate_x is <= threshold_deg",
        },
        "baseline": baseline_stats,
        "guided": guided_stats,
        "angle_threshold_sweep": sweep,
        "best_gate_angle_threshold": best,
        "notes": [
            "Can angle is arccos(rzz), where rzz is world z dot object z from cheezit_rot6d.",
            "Traces are plotted from can_grabbed until the first post-grasp EEF x <= gate_x crossing.",
            "The classifier uses the worst can angle over that whole pre-gate segment; runs that never cross the gate are not counted as pass.",
        ],
    }

    out_png = args.output_dir / f"{args.name}.png"
    out_json = args.output_dir / f"{args.name}_summary.json"
    plot(baseline, guided, summary, out_png, gate_x=args.gate_x, angle_ylim=tuple(args.angle_ylim))
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
