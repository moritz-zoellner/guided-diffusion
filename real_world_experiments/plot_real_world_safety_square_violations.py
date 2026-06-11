#!/usr/bin/env python3
"""Plot real-world rollout XY traces and safety-square violation rates."""

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

from plot_real_world_xy_coverage import COLORS, collect_traces, configure_matplotlib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_1"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/safety_guidance_tuning"


def shifted_boxes(
    base_box: tuple[float, float, float, float],
    n_shifts: int,
    dy_step: float,
) -> list[tuple[float, float, float, float]]:
    x_min, x_max, y_min, y_max = base_box
    return [(x_min, x_max, y_min + i * dy_step, y_max + i * dy_step) for i in range(n_shifts)]


def signed_distance_xy(xy: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = box
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.maximum(np.maximum(x_min - x, x - x_max), 0.0)
    dy = np.maximum(np.maximum(y_min - y, y - y_max), 0.0)
    outside = np.hypot(dx, dy)
    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    inside_margin = np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y])
    dist = outside
    dist[inside] = -inside_margin[inside]
    return dist


def summarize_violations(
    traces: list[dict[str, Any]],
    boxes: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    rows = []
    per_rollout = []
    total_points = int(sum(len(trace["eef_xy"]) for trace in traces))

    for box_idx, box in enumerate(boxes):
        rollout_hits = []
        point_hits = 0
        closest = {"rollout_idx": None, "min_signed_distance_m": float("inf")}
        for trace in traces:
            xy = trace["eef_xy"]
            dist = signed_distance_xy(xy, box)
            inside_count = int(np.sum(dist <= 0.0))
            min_dist = float(np.min(dist))
            violated = inside_count > 0
            rollout_hits.append(violated)
            point_hits += inside_count
            if min_dist < closest["min_signed_distance_m"]:
                closest = {
                    "rollout_idx": int(trace["rollout_idx"]),
                    "min_signed_distance_m": min_dist,
                }
            per_rollout.append(
                {
                    "box_idx": int(box_idx),
                    "rollout_idx": int(trace["rollout_idx"]),
                    "violated": bool(violated),
                    "inside_point_count": inside_count,
                    "n_points": int(len(xy)),
                    "min_signed_distance_m": min_dist,
                }
            )

        x_min, x_max, y_min, y_max = box
        rows.append(
            {
                "box_idx": int(box_idx),
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "violating_rollouts": int(sum(rollout_hits)),
                "n_rollouts": int(len(traces)),
                "rollout_violation_rate": float(np.mean(rollout_hits)),
                "inside_point_count": int(point_hits),
                "n_eef_points": total_points,
                "point_violation_rate": float(point_hits / max(1, total_points)),
                "closest_rollout_idx": closest["rollout_idx"],
                "min_signed_distance_m": float(closest["min_signed_distance_m"]),
            }
        )

    return {"squares": rows, "per_rollout": per_rollout}


def draw_boxes(ax, boxes: list[tuple[float, float, float, float]]) -> None:
    cmap = plt.get_cmap("Reds")
    for idx, box in enumerate(boxes):
        x_min, x_max, y_min, y_max = box
        color = cmap(0.35 + 0.5 * idx / max(1, len(boxes) - 1))
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=color,
                edgecolor="#8b0000",
                linewidth=0.8,
                alpha=0.14,
                zorder=1,
            )
        )
        ax.text(
            x_max + 0.002,
            0.5 * (y_min + y_max),
            str(idx),
            fontsize=6,
            color="#7f1d1d",
            va="center",
        )


def plot(
    traces: list[dict[str, Any]],
    boxes: list[tuple[float, float, float, float]],
    summary: dict[str, Any],
    output_stem: Path,
    title: str,
) -> None:
    configure_matplotlib()
    all_eef = np.concatenate([trace["eef_xy"] for trace in traces], axis=0)
    all_obj = np.concatenate([trace["object_xy"] for trace in traces if len(trace["object_xy"])], axis=0)

    fig, (ax_xy, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.8),
        gridspec_kw={"width_ratios": [1.25, 0.8]},
    )

    draw_boxes(ax_xy, boxes)
    for trace in traces:
        xy = trace["eef_xy"]
        color = COLORS[trace["outcome"]]
        ax_xy.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.38, linewidth=1.0, zorder=3)
        ax_xy.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.20, s=5, linewidths=0, zorder=3)
        ax_xy.scatter(xy[0, 0], xy[0, 1], color=color, marker="o", s=16, edgecolor="black", linewidth=0.25, zorder=4)
        ax_xy.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="x", s=28, linewidth=0.8, zorder=4)

    if len(all_obj):
        ax_xy.scatter(all_obj[:, 0], all_obj[:, 1], color=COLORS["object"], alpha=0.18, s=6, linewidths=0, zorder=2)

    x_pad = max(0.01, 0.08 * float(np.ptp(all_eef[:, 0])))
    y_pad = max(0.01, 0.08 * float(np.ptp(all_eef[:, 1])))
    ax_xy.set_xlim(float(all_eef[:, 0].min() - x_pad), float(all_eef[:, 0].max() + x_pad))
    y_min = min(float(all_eef[:, 1].min()), min(box[2] for box in boxes))
    y_max = max(float(all_eef[:, 1].max()), max(box[3] for box in boxes))
    ax_xy.set_ylim(float(y_min - y_pad), float(y_max + y_pad))
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.25, linewidth=0.5)
    ax_xy.set_xlabel("world x [m]")
    ax_xy.set_ylabel("world y [m]")
    ax_xy.set_title("Executed EEF XY")

    legend_handles = [
        Patch(facecolor=COLORS["pouring_left"], edgecolor="black", linewidth=0.35, label="pour left"),
        Patch(facecolor=COLORS["pouring_right"], edgecolor="black", linewidth=0.35, label="pour right"),
        Patch(facecolor=COLORS["no_pour"], edgecolor="black", linewidth=0.35, label="no pour"),
        Patch(facecolor="#d62728", edgecolor="#8b0000", linewidth=0.35, alpha=0.18, label="test squares"),
    ]
    ax_xy.legend(handles=legend_handles, loc="best", frameon=True)

    square_rows = summary["squares"]
    idxs = [row["box_idx"] for row in square_rows]
    rates = [100.0 * row["rollout_violation_rate"] for row in square_rows]
    bars = ax_bar.bar(idxs, rates, color="#d95f02", alpha=0.78)
    ax_bar.set_ylim(0.0, 105.0)
    ax_bar.set_xlabel("square index")
    ax_bar.set_ylabel("violating rollouts [%]")
    ax_bar.set_title("Violation Rate")
    ax_bar.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_bar.set_xticks(idxs)
    for bar, row in zip(bars, square_rows):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.0,
            f"{row['violating_rollouts']}/{row['n_rollouts']}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.suptitle(title)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_tables(summary: dict[str, Any], output_stem: Path) -> None:
    summary_path = output_stem.with_name(output_stem.name + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = output_stem.with_name(output_stem.name + "_summary.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary["squares"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["squares"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--x-min", type=float, default=-0.60)
    parser.add_argument("--x-max", type=float, default=-0.55)
    parser.add_argument("--y-min", type=float, default=-0.10)
    parser.add_argument("--y-max", type=float, default=-0.05)
    parser.add_argument("--n-shifts", type=int, default=10)
    parser.add_argument("--dy-step", type=float, default=-0.01)
    parser.add_argument("--title", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir.expanduser()
    traces = collect_traces(run_dir)
    box = (
        min(args.x_min, args.x_max),
        max(args.x_min, args.x_max),
        min(args.y_min, args.y_max),
        max(args.y_min, args.y_max),
    )
    boxes = shifted_boxes(box, max(1, int(args.n_shifts)), float(args.dy_step))
    summary = summarize_violations(traces, boxes)
    summary.update(
        {
            "run_dir": str(run_dir),
            "n_rollouts": len(traces),
            "base_box": {"x_min": box[0], "x_max": box[1], "y_min": box[2], "y_max": box[3]},
            "n_shifts": len(boxes),
            "dy_step": float(args.dy_step),
        }
    )

    stem = args.stem or f"{run_dir.name}_y_shift_square_violations"
    output_stem = args.output_dir / stem
    title = args.title or f"{run_dir.name}: EEF XY and y-shifted square violations"
    plot(traces, boxes, summary, output_stem, title)
    write_tables(summary, output_stem)

    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_name(output_stem.name + '_summary.json')}")
    print(f"wrote {output_stem.with_name(output_stem.name + '_summary.csv')}")
    for row in summary["squares"]:
        print(
            f"box {row['box_idx']:02d} y=[{row['y_min']:.3f},{row['y_max']:.3f}]: "
            f"{row['violating_rollouts']}/{row['n_rollouts']} rollouts "
            f"({100.0 * row['rollout_violation_rate']:.1f}%), "
            f"min_signed={1000.0 * row['min_signed_distance_m']:.1f}mm"
        )


if __name__ == "__main__":
    main()
