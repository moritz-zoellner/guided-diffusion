#!/usr/bin/env python3
"""Plot spatial gate variables over decisions-after-grasp for real-world runs."""

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
DEFAULT_BASELINE_DIRS = [
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_1",
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_2",
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3",
]
DEFAULT_HIGHLIGHT = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_left_rzz_hold_xminus060_epoch160_n1/rollouts/rollout_000/events.jsonl"
)


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
    events = []
    for root in dirs:
        events.extend(sorted(root.glob("rollouts/rollout_*/events.jsonl")))
    return events


def collect(path: Path, group: str) -> dict[str, Any] | None:
    rows = []
    label_events = []
    rollout_end = None
    for event in load_jsonl(path):
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        if event.get("type") == "rollout_end":
            rollout_end = event
        obs, decision = obs_from_event(event)
        if obs is None or "cheezit_rot6d" not in obs:
            continue
        matrix = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float64)[None])[0]
        rzz = float(matrix[2, 2])
        rows.append(
            {
                "decision": int(decision),
                "label": np.asarray(event.get("current_label") or event.get("label") or [0, 0, 0], dtype=np.int32),
                "eef": np.asarray(obs["eef_pos"], dtype=np.float64),
                "obj": np.asarray(obs["cheezit_pos"], dtype=np.float64),
                "rzz": rzz,
            }
        )
    if not rows:
        return None
    if rollout_end is not None and not bool(rollout_end.get("success")):
        return None

    grab_decision = None
    pour_decision = None
    for event in label_events:
        if event.get("label_name") == "can_grabbed" and int(event.get("to", -1)) == 1:
            grab_decision = int(event["decision_idx"])
        if event.get("label_name") == "pouring_left" and int(event.get("to", -1)) == 1:
            pour_decision = int(event["decision_idx"])
    if grab_decision is None:
        grabbed = [row for row in rows if row["label"][0] > 0]
        if not grabbed:
            return None
        grab_decision = int(grabbed[0]["decision"])

    decisions = np.asarray([row["decision"] for row in rows], dtype=np.int32)
    grab_idx = next((idx for idx, decision in enumerate(decisions) if decision >= grab_decision), 0)
    eef = np.asarray([row["eef"] for row in rows], dtype=np.float64)
    obj = np.asarray([row["obj"] for row in rows], dtype=np.float64)
    rzz = np.asarray([row["rzz"] for row in rows], dtype=np.float64)
    return {
        "path": str(path),
        "group": group,
        "run": path.parent.parent.parent.name,
        "rollout": path.parent.name,
        "label_events": label_events,
        "grab_decision": int(grab_decision),
        "pour_decision": None if pour_decision is None else int(pour_decision),
        "relative_decision": decisions - int(grab_decision),
        "decisions": decisions,
        "eef": eef,
        "obj": obj,
        "eef_x": eef[:, 0],
        "eef_y": eef[:, 1],
        "eef_xy_sq": np.sum(eef[:, :2] ** 2, axis=-1),
        "obj_x": obj[:, 0],
        "obj_y": obj[:, 1],
        "obj_xy_sq": np.sum(obj[:, :2] ** 2, axis=-1),
        "rzz": rzz,
        "rzz_at_grasp": float(rzz[grab_idx]),
        "rzz_error": np.abs(rzz - float(rzz[grab_idx])),
    }


def percentile_band(traces: list[dict[str, Any]], x_grid: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for item in traces:
        x = item["relative_decision"]
        y = item[key]
        order = np.argsort(x)
        xu = x[order]
        yu = y[order]
        arr = np.full_like(x_grid, np.nan, dtype=np.float64)
        keep = (x_grid >= xu[0]) & (x_grid <= xu[-1])
        arr[keep] = np.interp(x_grid[keep], xu, yu)
        values.append(arr)
    stack = np.asarray(values, dtype=np.float64)
    return (
        np.nanpercentile(stack, 10, axis=0),
        np.nanpercentile(stack, 50, axis=0),
        np.nanpercentile(stack, 90, axis=0),
    )


def first_crossing(item: dict[str, Any], key: str, value: float, direction: str) -> dict[str, Any] | None:
    for idx in np.where(item["relative_decision"] >= 0)[0]:
        current = float(item[key][idx])
        if (direction == "le" and current <= value) or (direction == "ge" and current >= value):
            return {
                "decision": int(item["decisions"][idx]),
                "relative_decision": int(item["relative_decision"][idx]),
                "value": current,
                "eef_xy": item["eef"][idx, :2].astype(float).tolist(),
                "object_xy": item["obj"][idx, :2].astype(float).tolist(),
                "rzz_error": float(item["rzz_error"][idx]),
            }
    return None


def gate_stats(traces: list[dict[str, Any]], key: str, values: list[float], direction: str) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        crossings = [first_crossing(item, key, value, direction) for item in traces]
        crossings = [item for item in crossings if item is not None]
        if not crossings:
            rows.append({"key": key, "threshold": value, "direction": direction, "n": 0})
            continue
        rows.append(
            {
                "key": key,
                "threshold": value,
                "direction": direction,
                "n": len(crossings),
                "relative_decision_p10_p50_p90": np.percentile(
                    [item["relative_decision"] for item in crossings], [10, 50, 90]
                ).astype(float).tolist(),
                "rzz_error_p10_p50_p90": np.percentile(
                    [item["rzz_error"] for item in crossings], [10, 50, 90]
                ).astype(float).tolist(),
                "eef_x_p10_p50_p90": np.percentile(
                    [item["eef_xy"][0] for item in crossings], [10, 50, 90]
                ).astype(float).tolist(),
                "eef_y_p10_p50_p90": np.percentile(
                    [item["eef_xy"][1] for item in crossings], [10, 50, 90]
                ).astype(float).tolist(),
            }
        )
    return rows


def plot(
    traces: list[dict[str, Any]],
    highlight: dict[str, Any] | None,
    out_path: Path,
    x_gates: list[float],
    xy_sq_gates: list[float],
    time_xlim: tuple[float, float],
) -> dict[str, Any]:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex="col")
    panels = [
        ("eef_x", "EEF world x [m]", axes[0, 0]),
        ("eef_xy_sq", r"EEF $x^2+y^2$ [$m^2$]", axes[1, 0]),
        ("obj_xy_sq", r"object $x^2+y^2$ [$m^2$]", axes[2, 0]),
        ("rzz_error", "|rzz - rzz_at_grasp|", axes[0, 1]),
        ("eef_y", "EEF world y [m]", axes[1, 1]),
        ("rzz", "rzz", axes[2, 1]),
    ]

    min_rel = min(int(np.nanmin(item["relative_decision"])) for item in traces)
    max_rel = max(int(np.nanmax(item["relative_decision"])) for item in traces + ([highlight] if highlight else []))
    grid = np.arange(min(-5, min_rel), max_rel + 1)

    for key, ylabel, ax in panels:
        for item in traces:
            ax.plot(item["relative_decision"], item[key], color="#9a9a9a", alpha=0.27, linewidth=1.0)
            if item["pour_decision"] is not None:
                ax.axvline(item["pour_decision"] - item["grab_decision"], color="#bdbdbd", alpha=0.12, linewidth=0.8)
        lo, med, hi = percentile_band(traces, grid, key)
        ax.fill_between(grid, lo, hi, color="#7f7f7f", alpha=0.18, label="baseline p10-p90")
        ax.plot(grid, med, color="#111111", linewidth=2.0, label="baseline median")
        if highlight is not None:
            ax.plot(highlight["relative_decision"], highlight[key], color="#1f77b4", linewidth=2.0, label="rzz hold")
            if highlight["pour_decision"] is not None:
                ax.axvline(
                    highlight["pour_decision"] - highlight["grab_decision"],
                    color="#1f77b4",
                    linestyle="--",
                    alpha=0.75,
                    linewidth=1.0,
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(*time_xlim)

    for gate in x_gates:
        axes[0, 0].axhline(gate, color="#d62728", linestyle=":", linewidth=1.0)
    for gate in xy_sq_gates:
        axes[1, 0].axhline(gate, color="#d62728", linestyle=":", linewidth=1.0)
    for err in (0.01, 0.02, 0.03, 0.05):
        axes[0, 1].axhline(err, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.45)

    axes[0, 0].set_title("candidate x gate over decisions after grasp")
    axes[1, 0].set_title("candidate radial gate over decisions after grasp")
    axes[0, 1].set_title("rzz error for context")
    axes[2, 0].set_xlabel("decisions after grasp")
    axes[2, 1].set_xlabel("decisions after grasp")
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

    pour_rows = []
    for item in traces:
        if item["pour_decision"] is None:
            continue
        idx = int(np.argmin(np.abs(item["decisions"] - item["pour_decision"])))
        pour_rows.append(
            {
                "relative_decision": int(item["pour_decision"] - item["grab_decision"]),
                "eef_x": float(item["eef_x"][idx]),
                "eef_y": float(item["eef_y"][idx]),
                "eef_xy_sq": float(item["eef_xy_sq"][idx]),
                "obj_xy_sq": float(item["obj_xy_sq"][idx]),
                "rzz_error": float(item["rzz_error"][idx]),
            }
        )

    def pct(rows: list[dict[str, Any]], key: str) -> list[float] | None:
        if not rows:
            return None
        return np.percentile([item[key] for item in rows], [10, 50, 90]).astype(float).tolist()

    summary = {
        "plot_png": str(out_path),
        "plot_pdf": str(out_path.with_suffix(".pdf")),
        "n_baseline_rollouts": len(traces),
        "highlight": None if highlight is None else highlight["path"],
        "x_gate_stats": gate_stats(traces, "eef_x", x_gates, "le"),
        "eef_xy_sq_gate_stats": gate_stats(traces, "eef_xy_sq", xy_sq_gates, "ge"),
        "pour_stats": {
            "relative_decision_p10_p50_p90": pct(pour_rows, "relative_decision"),
            "eef_x_p10_p50_p90": pct(pour_rows, "eef_x"),
            "eef_y_p10_p50_p90": pct(pour_rows, "eef_y"),
            "eef_xy_sq_p10_p50_p90": pct(pour_rows, "eef_xy_sq"),
            "obj_xy_sq_p10_p50_p90": pct(pour_rows, "obj_xy_sq"),
            "rzz_error_p10_p50_p90": pct(pour_rows, "rzz_error"),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, action="append", default=None)
    parser.add_argument("--highlight-events", type=Path, default=DEFAULT_HIGHLIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default="left_baselines_spatial_gate_threshold_tuning")
    parser.add_argument("--x-gate", type=float, action="append", default=None)
    parser.add_argument("--xy-sq-gate", type=float, action="append", default=None)
    parser.add_argument("--time-xlim", type=float, nargs=2, default=(0.0, 70.0), metavar=("MIN", "MAX"))
    args = parser.parse_args()

    baseline_dirs = args.baseline_dir if args.baseline_dir else DEFAULT_BASELINE_DIRS
    traces = []
    for path in discover_events(baseline_dirs):
        item = collect(path, "baseline")
        if item is not None:
            traces.append(item)
    if not traces:
        raise ValueError("No successful baseline traces found")
    highlight = collect(args.highlight_events, "highlight") if args.highlight_events and args.highlight_events.exists() else None
    x_gates = args.x_gate if args.x_gate is not None else [-0.60, -0.63, -0.64]
    xy_sq_gates = args.xy_sq_gate if args.xy_sq_gate is not None else [0.36, 0.39, 0.41]
    out_path = args.output_dir / f"{args.name}.png"
    summary = plot(traces, highlight, out_path, x_gates, xy_sq_gates, tuple(args.time_xlim))
    summary["baseline_dirs"] = [str(path) for path in baseline_dirs]
    summary["x_gates"] = x_gates
    summary["xy_sq_gates"] = xy_sq_gates
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
