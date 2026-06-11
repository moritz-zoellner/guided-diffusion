#!/usr/bin/env python3
"""Plot real-world top-down trajectory panels for cyclic and safety runs."""

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
from matplotlib.patches import Patch, Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CYCLE_RUN = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    "automaton_release_regrasp_right_left_cycle3_epoch160"
)
DEFAULT_SAFETY_RUN = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    "automaton_left_safety_box0_epoch160_n10_3"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/xy_trace_panels"

FIG_DPI = 300
OUR_BLUE = "#275fca"
DARK_GRAY = "#5f6368"
LIGHT_GRAY = "#c3c7cd"
UNSAFE_RED = "#b85f5a"
AXIS_GRAY = "#8a8a8a"
PANEL_FRAME_LW = 0.9


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": [
                "Computer Modern Typewriter",
                "CMU Typewriter Text",
                "DejaVu Sans Mono",
            ],
            "mathtext.fontset": "cm",
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_xy_axis(ax: plt.Axes, title: str) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(PANEL_FRAME_LW)
    ax.grid(True, alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", color=AXIS_GRAY, width=0.45, length=2.0, pad=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(title, pad=3.0)


def read_cycle_trace(run_dir: Path) -> dict[str, np.ndarray]:
    csv_path = run_dir / "rollouts/rollout_000/recovered_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing recovered trajectory: {csv_path}")
    eef_xy = []
    object_xy = []
    decision_idx = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            eef_xy.append([float(row["eef_x"]), float(row["eef_y"])])
            object_xy.append([float(row["object_x"]), float(row["object_y"])])
            decision_idx.append(int(row["decision_idx"]))
    return {
        "eef_xy": np.asarray(eef_xy, dtype=np.float64),
        "object_xy": np.asarray(object_xy, dtype=np.float64),
        "decision_idx": np.asarray(decision_idx, dtype=np.int32),
    }


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "rollout_end":
        return event.get("final_obs")
    if event.get("type") in {"rollout_start", "decision", "chunk_sample"}:
        return event.get("obs")
    return None


def read_event_trace(rollout_dir: Path) -> dict[str, np.ndarray]:
    eef_xy = []
    object_xy = []
    with (rollout_dir / "events.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            obs = obs_from_event(event)
            if not obs:
                continue
            if "eef_pos" in obs:
                eef_xy.append(obs["eef_pos"][:2])
            if "cheezit_pos" in obs:
                object_xy.append(obs["cheezit_pos"][:2])
    return {
        "eef_xy": np.asarray(eef_xy, dtype=np.float64),
        "object_xy": np.asarray(object_xy, dtype=np.float64),
    }


def read_safety_traces(run_dir: Path) -> list[dict[str, np.ndarray]]:
    traces = []
    for rollout_dir in sorted((run_dir / "rollouts").glob("rollout_*")):
        events_path = rollout_dir / "events.jsonl"
        if not events_path.exists():
            continue
        trace = read_event_trace(rollout_dir)
        if len(trace["eef_xy"]):
            traces.append(trace)
    if not traces:
        raise ValueError(f"No safety rollout traces found under {run_dir / 'rollouts'}")
    return traces


def read_shrunk_safety_box(run_dir: Path, shrink_m: float) -> dict[str, float]:
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    box = config["safety_box"]
    x_min = float(box["x_min"]) + shrink_m
    x_max = float(box["x_max"]) - shrink_m
    y_min = float(box["y_min"]) + shrink_m
    y_max = float(box["y_max"]) - shrink_m
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Safety-box shrink {shrink_m} is too large for {box}")
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "shrink_m": shrink_m,
        "raw": box,
    }


def padded_limits(points: np.ndarray, pad_frac: float = 0.08, min_pad: float = 0.01) -> tuple[float, float, float, float]:
    x_pad = max(min_pad, pad_frac * float(np.ptp(points[:, 0])))
    y_pad = max(min_pad, pad_frac * float(np.ptp(points[:, 1])))
    return (
        float(points[:, 0].min() - x_pad),
        float(points[:, 0].max() + x_pad),
        float(points[:, 1].min() - y_pad),
        float(points[:, 1].max() + y_pad),
    )


def plot_cycle_panel(ax: plt.Axes, trace: dict[str, np.ndarray]) -> None:
    xy = trace["eef_xy"]
    obj = trace["object_xy"]
    ax.plot(xy[:, 0], xy[:, 1], color=OUR_BLUE, linewidth=1.1, alpha=0.88)
    ax.scatter(xy[:, 0], xy[:, 1], color=OUR_BLUE, s=3.0, alpha=0.35, linewidths=0)
    ax.scatter(obj[:, 0], obj[:, 1], color=LIGHT_GRAY, s=3.0, alpha=0.22, linewidths=0)
    ax.scatter(xy[0, 0], xy[0, 1], color="white", edgecolor="black", s=20, linewidth=0.55, zorder=5)
    ax.scatter(xy[-1, 0], xy[-1, 1], color=OUR_BLUE, marker="x", s=24, linewidth=0.75, zorder=5)
    xmin, xmax, ymin, ymax = padded_limits(np.vstack([xy, obj]))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    style_xy_axis(ax, "Cyclic Instruction")


def plot_safety_panel(ax: plt.Axes, traces: list[dict[str, np.ndarray]], safety_box: dict[str, float]) -> None:
    all_points = []
    for trace in traces:
        xy = trace["eef_xy"]
        obj = trace["object_xy"]
        all_points.append(xy)
        ax.plot(xy[:, 0], xy[:, 1], color=DARK_GRAY, linewidth=0.9, alpha=0.34)
        ax.scatter(xy[:, 0], xy[:, 1], color=DARK_GRAY, s=3.0, alpha=0.18, linewidths=0)
        if len(obj):
            all_points.append(obj)
            ax.scatter(obj[:, 0], obj[:, 1], color=LIGHT_GRAY, s=3.0, alpha=0.18, linewidths=0)
        ax.scatter(xy[0, 0], xy[0, 1], color="white", edgecolor="black", s=16, linewidth=0.5, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=DARK_GRAY, marker="x", s=20, linewidth=0.7, zorder=5)

    ax.add_patch(
        Rectangle(
            (safety_box["x_min"], safety_box["y_min"]),
            safety_box["x_max"] - safety_box["x_min"],
            safety_box["y_max"] - safety_box["y_min"],
            facecolor=UNSAFE_RED,
            edgecolor=UNSAFE_RED,
            linewidth=1.0,
            alpha=0.18,
            zorder=2,
        )
    )
    points = np.vstack(all_points)
    corners = np.asarray(
        [
            [safety_box["x_min"], safety_box["y_min"]],
            [safety_box["x_max"], safety_box["y_max"]],
        ],
        dtype=np.float64,
    )
    xmin, xmax, ymin, ymax = padded_limits(np.vstack([points, corners]))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    style_xy_axis(ax, "Safety Constraint")


def plot_panels(args: argparse.Namespace) -> dict[str, Any]:
    configure_matplotlib()
    cycle_trace = read_cycle_trace(args.cycle_run)
    safety_traces = read_safety_traces(args.safety_run)
    safety_box = read_shrunk_safety_box(args.safety_run, args.safety_box_shrink_m)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), gridspec_kw={"wspace": 0.26})
    plot_cycle_panel(axes[0], cycle_trace)
    plot_safety_panel(axes[1], safety_traces, safety_box)

    legend_handles = [
        Patch(facecolor=OUR_BLUE, edgecolor="black", linewidth=0.35, label="cyclic EEF"),
        Patch(facecolor=DARK_GRAY, edgecolor="black", linewidth=0.35, label="safety EEF"),
        Patch(facecolor=LIGHT_GRAY, edgecolor="black", linewidth=0.35, label="object poses"),
        Patch(facecolor=UNSAFE_RED, edgecolor=UNSAFE_RED, linewidth=0.35, alpha=0.35, label="forbidden square"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        handlelength=1.25,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.23)

    output_stem = args.output_dir / args.stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "cycle_run": str(args.cycle_run),
        "safety_run": str(args.safety_run),
        "cycle_points": int(len(cycle_trace["eef_xy"])),
        "safety_rollouts": len(safety_traces),
        "safety_points": int(sum(len(trace["eef_xy"]) for trace in safety_traces)),
        "safety_box_plotted": {key: float(value) if isinstance(value, (int, float)) else value for key, value in safety_box.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot real-world cyclic and safety top-down xy panels.")
    parser.add_argument("--cycle-run", type=Path, default=DEFAULT_CYCLE_RUN)
    parser.add_argument("--safety-run", type=Path, default=DEFAULT_SAFETY_RUN)
    parser.add_argument("--safety-box-shrink-m", type=float, default=0.0025)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="real_world_cyclic_safety_xy_panels")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = plot_panels(args)
    summary_path = (args.output_dir / args.stem).with_name(args.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {(args.output_dir / args.stem).with_suffix('.png')}")
    print(f"wrote {(args.output_dir / args.stem).with_suffix('.pdf')}")
    print(f"wrote {summary_path}")
    print(f"plotted safety box: {summary['safety_box_plotted']}")


if __name__ == "__main__":
    main()
