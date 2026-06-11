#!/usr/bin/env python3
"""Plot Cheez-It roll/pitch/yaw diagnostics from real-world event logs."""

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
    return None


def collect_series(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    decisions = []
    rot6d = []
    labels = []
    label_events = []
    for event in events:
        obs = obs_from_event(event)
        if obs is None:
            continue
        decisions.append(int(event.get("decision_idx", 0)))
        rot6d.append(obs["cheezit_rot6d"])
        labels.append(event.get("label") or event.get("current_label") or [0, 0, 0])
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
    if not rot6d:
        raise ValueError("No Cheez-It observations found")
    return (
        np.asarray(decisions, dtype=np.int32),
        np.asarray(rot6d, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        label_events,
    )


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values), axis=0))


def euler_xyz_deg(matrices: np.ndarray) -> np.ndarray:
    # SciPy lowercase xyz is extrinsic/world-axis XYZ. We use it here as a
    # readable roll/pitch/yaw diagnostic rather than as the eventual control API.
    return unwrap_degrees(Rotation.from_matrix(matrices).as_euler("xyz", degrees=True))


def relative_euler_by_grasp(
    decisions: np.ndarray,
    matrices: np.ndarray,
    labels: np.ndarray,
    label_events: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    rel = np.full((len(decisions), 3), np.nan, dtype=np.float64)
    segments = []
    current_ref = None
    current_start = None
    event_by_decision = {}
    for event in label_events:
        if event.get("label_name") == "can_grabbed":
            event_by_decision[int(event["decision_idx"])] = int(event["to"])

    for idx, decision in enumerate(decisions):
        if decision in event_by_decision and event_by_decision[decision] == 1:
            current_ref = matrices[idx].copy()
            current_start = idx
        grabbed = bool(labels[idx, 0] > 0)
        if grabbed and current_ref is None:
            current_ref = matrices[idx].copy()
            current_start = idx
        if grabbed and current_ref is not None:
            rel_matrix = current_ref.T @ matrices[idx]
            rel[idx] = Rotation.from_matrix(rel_matrix).as_euler("xyz", degrees=True)
        if decision in event_by_decision and event_by_decision[decision] == 0 and current_ref is not None:
            segments.append({"start_idx": int(current_start), "end_idx": int(idx)})
            current_ref = None
            current_start = None
    if current_ref is not None and current_start is not None:
        segments.append({"start_idx": int(current_start), "end_idx": int(len(decisions) - 1)})

    for axis in range(3):
        valid = np.isfinite(rel[:, axis])
        if np.any(valid):
            rel[valid, axis] = np.degrees(np.unwrap(np.radians(rel[valid, axis])))
    return rel, segments


def label_intervals(decisions: np.ndarray, labels: np.ndarray, label_idx: int) -> list[tuple[int, int]]:
    intervals = []
    active_start = None
    for idx, decision in enumerate(decisions):
        active = bool(labels[idx, label_idx] > 0)
        if active and active_start is None:
            active_start = int(decision)
        if not active and active_start is not None:
            intervals.append((active_start, int(decisions[max(0, idx - 1)])))
            active_start = None
    if active_start is not None:
        intervals.append((active_start, int(decisions[-1])))
    return intervals


def summarize_pour_axis_spikes(
    decisions: np.ndarray,
    relative_euler: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for label_idx, label_name in ((1, "pouring_right"), (2, "pouring_left")):
        intervals = label_intervals(decisions, labels, label_idx)
        rows = []
        for start, end in intervals:
            mask = (decisions >= start) & (decisions <= end) & np.all(np.isfinite(relative_euler), axis=1)
            if not np.any(mask):
                continue
            vals = relative_euler[mask]
            start_vals = vals[0:1]
            delta = vals - start_vals
            ranges = np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0)
            max_abs_delta = np.nanmax(np.abs(delta), axis=0)
            dominant_range = int(np.nanargmax(ranges))
            dominant_delta = int(np.nanargmax(max_abs_delta))
            rows.append(
                {
                    "start_decision": int(start),
                    "end_decision": int(end),
                    "duration_decisions": int(end - start),
                    "range_deg": {AXIS_NAMES[i]: float(ranges[i]) for i in range(3)},
                    "max_abs_delta_from_interval_start_deg": {
                        AXIS_NAMES[i]: float(max_abs_delta[i]) for i in range(3)
                    },
                    "dominant_axis_by_range": AXIS_NAMES[dominant_range],
                    "dominant_axis_by_delta": AXIS_NAMES[dominant_delta],
                }
            )
        summaries[label_name] = {
            "intervals": rows,
            "n_intervals": len(rows),
        }
    return summaries


def shade_pours(ax, intervals: dict[str, list[tuple[int, int]]]) -> None:
    for label, spans in intervals.items():
        color = "#ffcc00" if label == "pouring_right" else "#8ecae6"
        for idx, (start, end) in enumerate(spans):
            ax.axvspan(start, end, color=color, alpha=0.18, label=label if idx == 0 else None)


def plot_with_axis_projections(
    path: Path,
    decisions: np.ndarray,
    abs_euler: np.ndarray,
    rel_euler: np.ndarray,
    matrices: np.ndarray,
    labels: np.ndarray,
    label_events: list[dict[str, Any]],
    *,
    xlim: tuple[float, float] | None,
) -> None:
    intervals = {
        "pouring_right": label_intervals(decisions, labels, 1),
        "pouring_left": label_intervals(decisions, labels, 2),
    }
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), sharex=True)
    panels = [
        ("relative to current grasp: extrinsic XYZ Euler", rel_euler),
        ("absolute object orientation: extrinsic XYZ Euler", abs_euler),
    ]
    for ax, (title, values) in zip(axes[:2], panels):
        shade_pours(ax, intervals)
        for axis, name in enumerate(AXIS_NAMES):
            ax.plot(decisions, values[:, axis], color=COLORS[axis], linewidth=1.2, label=name)
        for event in label_events:
            decision = int(event.get("decision_idx", 0))
            if event.get("label_name") == "can_grabbed":
                ax.axvline(decision, color="#555555", linestyle=":", linewidth=0.8, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("angle [deg]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=5)

    ax = axes[2]
    shade_pours(ax, intervals)
    axis_z = matrices[:, 2, :]  # row z equals world-z dot local x/y/z for column-stored axes.
    for axis, name in enumerate(("r_zx", "r_zy", "r_zz")):
        ax.plot(decisions, axis_z[:, axis], color=COLORS[axis], linewidth=1.2, label=name)
    for event in label_events:
        decision = int(event.get("decision_idx", 0))
        if event.get("label_name") == "can_grabbed":
            ax.axvline(decision, color="#555555", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_title("object local axes projected onto world z")
    ax.set_ylabel("world z dot object axis")
    ax.set_xlabel("decision")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=5)
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--name", type=str, default="release_regrasp_rollout000_cheezit_euler_axes")
    args = parser.parse_args()

    configure_matplotlib()
    events = load_events(args.events)
    decisions, rot6d, labels, label_events = collect_series(events)
    matrices = rot6d_to_matrix(rot6d)
    abs_euler = euler_xyz_deg(matrices)
    rel_euler, grasp_segments = relative_euler_by_grasp(decisions, matrices, labels, label_events)
    xlim = tuple(args.xlim) if args.xlim is not None else None

    out_png = args.output_dir / f"{args.name}.png"
    plot_with_axis_projections(
        out_png,
        decisions,
        abs_euler,
        rel_euler,
        matrices,
        labels,
        label_events,
        xlim=xlim,
    )
    summary = {
        "events": str(args.events),
        "euler_convention": "scipy Rotation.as_euler('xyz'), extrinsic/world-axis XYZ diagnostic",
        "relative_reference": "reset at each can_grabbed 0->1 event",
        "label_names": ["can_grabbed", "pouring_right", "pouring_left"],
        "grasp_segments": [
            {
                "start_decision": int(decisions[item["start_idx"]]),
                "end_decision": int(decisions[item["end_idx"]]),
            }
            for item in grasp_segments
        ],
        "pour_axis_summary": summarize_pour_axis_spikes(decisions, rel_euler, labels),
        "plot_png": str(out_png),
        "plot_pdf": str(out_png.with_suffix(".pdf")),
        "xlim": list(xlim) if xlim is not None else None,
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
