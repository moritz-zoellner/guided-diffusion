#!/usr/bin/env python3
"""Plot top-down real-world EEF xy coverage from rollout event logs."""

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
from matplotlib.patches import Patch, Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/base_dp_eval/base_dp_epoch160_n5_8"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/xy_coverage"

FIG_DPI = 220
COLORS = {
    "pouring_left": "#275fca",
    "pouring_right": "#5f6368",
    "no_pour": "#b85f5a",
    "object": "#c3c7cd",
}


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
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rollout_outcome(summary: dict[str, Any]) -> str:
    final_label = summary.get("final_label") or []
    event_names = {event.get("label_name") for event in summary.get("label_events", [])}
    if "pouring_left" in event_names or (len(final_label) > 2 and int(final_label[2]) == 1):
        return "pouring_left"
    if "pouring_right" in event_names or (len(final_label) > 1 and int(final_label[1]) == 1):
        return "pouring_right"
    return "no_pour"


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "rollout_end":
        return event.get("final_obs")
    if event.get("type") in {"rollout_start", "decision", "chunk_sample"}:
        return event.get("obs")
    return None


def read_rollout_trace(rollout_dir: Path) -> dict[str, Any]:
    summary_path = rollout_dir / "rollout_summary.json"
    events_path = rollout_dir / "events.jsonl"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    eef_xy = []
    object_xy = []
    with events_path.open("r", encoding="utf-8") as f:
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
        "rollout_dir": str(rollout_dir),
        "rollout_idx": int(summary.get("rollout_idx", rollout_dir.name.split("_")[-1])),
        "outcome": rollout_outcome(summary),
        "eef_xy": np.asarray(eef_xy, dtype=np.float64),
        "object_xy": np.asarray(object_xy, dtype=np.float64),
    }


def collect_traces(run_dir: Path) -> list[dict[str, Any]]:
    traces = []
    for rollout_dir in sorted((run_dir / "rollouts").glob("rollout_*")):
        if (rollout_dir / "events.jsonl").exists() and (rollout_dir / "rollout_summary.json").exists():
            trace = read_rollout_trace(rollout_dir)
            if len(trace["eef_xy"]):
                traces.append(trace)
    if not traces:
        raise ValueError(f"No rollout event traces found under {run_dir / 'rollouts'}")
    return traces


def parse_box(raw: list[float] | None) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if len(raw) != 4:
        raise ValueError("--box expects four values: x_min x_max y_min y_max")
    x_min, x_max, y_min, y_max = map(float, raw)
    return min(x_min, x_max), max(x_min, x_max), min(y_min, y_max), max(y_min, y_max)


def summarize_points(points: np.ndarray, traces: list[dict[str, Any]]) -> dict[str, Any]:
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "n_rollouts": len(traces),
        "n_eef_points": int(len(points)),
        "x": {str(p): float(np.percentile(points[:, 0], p)) for p in percentiles},
        "y": {str(p): float(np.percentile(points[:, 1], p)) for p in percentiles},
        "outcome_counts": {
            outcome: sum(1 for trace in traces if trace["outcome"] == outcome)
            for outcome in ["pouring_left", "pouring_right", "no_pour"]
        },
    }


def plot_traces(
    traces: list[dict[str, Any]],
    output_stem: Path,
    *,
    title: str,
    box: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    configure_matplotlib()
    all_eef = np.concatenate([trace["eef_xy"] for trace in traces], axis=0)
    all_obj = np.concatenate(
        [trace["object_xy"] for trace in traces if len(trace["object_xy"])],
        axis=0,
    )
    summary = summarize_points(all_eef, traces)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for trace in traces:
        xy = trace["eef_xy"]
        color = COLORS[trace["outcome"]]
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.32, linewidth=1.0)
        ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.22, s=5, linewidths=0)
        ax.scatter(xy[0, 0], xy[0, 1], color=color, marker="o", s=18, edgecolor="black", linewidth=0.25)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="x", s=25, linewidth=0.7)

    if len(all_obj):
        ax.scatter(
            all_obj[:, 0],
            all_obj[:, 1],
            color=COLORS["object"],
            alpha=0.20,
            s=6,
            linewidths=0,
            label="object poses",
        )

    if box is not None:
        x_min, x_max, y_min, y_max = box
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="#d62728",
                edgecolor="#d62728",
                linewidth=1.2,
                alpha=0.14,
                label="candidate safety square",
            )
        )

    x_pad = max(0.01, 0.08 * float(np.ptp(all_eef[:, 0])))
    y_pad = max(0.01, 0.08 * float(np.ptp(all_eef[:, 1])))
    ax.set_xlim(float(all_eef[:, 0].min() - x_pad), float(all_eef[:, 0].max() + x_pad))
    ax.set_ylim(float(all_eef[:, 1].min() - y_pad), float(all_eef[:, 1].max() + y_pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(title)

    legend_handles = [
        Patch(facecolor=COLORS["pouring_left"], edgecolor="black", linewidth=0.35, label="pour left"),
        Patch(facecolor=COLORS["pouring_right"], edgecolor="black", linewidth=0.35, label="pour right"),
        Patch(facecolor=COLORS["no_pour"], edgecolor="black", linewidth=0.35, label="no pour"),
        Patch(facecolor=COLORS["object"], edgecolor="black", linewidth=0.35, label="object poses"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot top-down EEF xy coverage from real-world rollouts.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="base_dp_epoch160_n5_8_xy_coverage")
    parser.add_argument("--title", default="Base Policy EEF XY Coverage")
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=None,
        help="Optional safety square/rectangle to overlay.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    traces = collect_traces(args.run_dir)
    output_stem = args.output_dir / args.stem
    summary = plot_traces(
        traces,
        output_stem,
        title=args.title,
        box=parse_box(args.box),
    )
    summary_path = output_stem.with_name(output_stem.name + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {summary_path}")
    print(
        "EEF xy range: "
        f"x=[{summary['x']['0']:.4f}, {summary['x']['100']:.4f}], "
        f"y=[{summary['y']['0']:.4f}, {summary['y']['100']:.4f}]"
    )
    print(
        "EEF xy 5-95%: "
        f"x=[{summary['x']['5']:.4f}, {summary['x']['95']:.4f}], "
        f"y=[{summary['y']['5']:.4f}, {summary['y']['95']:.4f}]"
    )


if __name__ == "__main__":
    main()
