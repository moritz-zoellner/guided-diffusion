#!/usr/bin/env python3
"""Grid probe: shift safety square in y and sweep gradient steps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

from plot_offline_safety_guidance_branches import (
    DEFAULT_BOX,
    DEFAULT_OUTPUT_DIR,
    chunk_events,
    configure_matplotlib,
    infer_dynamics_ckpt,
    load_dynamics,
    obs_to_state,
    predict_branch,
    read_rollout,
    refine_for_safety,
    signed_distance_xy,
)


DEFAULT_ROLLOUT = Path(
    "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    "automaton_left_epoch160_n10_3/rollouts/rollout_006"
)
DEFAULT_GRADIENT_STEPS = (0, 5, 10, 20, 30)
STEP_COLORS = {
    0: "#ff8c00",
    5: "#bcbd22",
    10: "#2ca02c",
    20: "#1f77b4",
    30: "#d62728",
}


def shifted_box(base_box: tuple[float, float, float, float], dy: float) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = base_box
    return x_min, x_max, y_min + dy, y_max + dy


def draw_box(ax, box: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = box
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="#d62728",
            edgecolor="#d62728",
            linewidth=1.1,
            alpha=0.16,
            zorder=1,
        )
    )


def step_color(gradient_steps: int, idx: int) -> str:
    if gradient_steps in STEP_COLORS:
        return STEP_COLORS[gradient_steps]
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % 10]


def build_predictions(
    rollout_dir: Path,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    boxes: list[tuple[float, float, float, float]],
    gradient_steps_values: list[int],
    *,
    guidance_scale: float,
    step_size: float,
    action_reg: float,
    tau: float,
    margin: float,
    rollout_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    events, eef_xy, obj_xy = read_rollout(rollout_dir)
    chunks = chunk_events(events)
    if not chunks:
        raise ValueError(f"No chunk_sample events found in {rollout_dir}")

    results = []
    for box_idx, box in enumerate(boxes):
        box_rows = []
        for chunk in chunks:
            obs = chunk["obs"]
            state_np = obs_to_state(obs)
            selected_chunk = np.asarray(chunk["selected_chunk"], dtype=np.float32)
            start_xy = np.asarray(obs["eef_pos"][:2], dtype=np.float64)
            branch = {
                "chunk_idx": int(chunk["chunk_idx"]),
                "decision_idx_before": int(chunk["decision_idx_before"]),
                "start_xy": start_xy.astype(float).tolist(),
                "predictions": {},
            }
            for gradient_steps in gradient_steps_values:
                refined = refine_for_safety(
                    state_np,
                    selected_chunk,
                    model,
                    stats,
                    device,
                    box,
                    guidance_scale=guidance_scale,
                    gradient_steps=gradient_steps,
                    step_size=step_size,
                    action_reg=action_reg,
                    tau=tau,
                    margin=margin,
                    rollout_scale=rollout_scale,
                )
                pred_state = predict_branch(
                    state_np,
                    refined,
                    model,
                    stats,
                    device,
                    rollout_scale=rollout_scale,
                )
                pred_xy = pred_state[:, :2]
                dist = signed_distance_xy(pred_xy, box)
                branch["predictions"][str(gradient_steps)] = {
                    "eef_xy": pred_xy.astype(float).tolist(),
                    "inside_count": int(np.sum(dist < 0.0)),
                    "min_signed_distance_m": float(np.min(dist)),
                    "mean_action_delta_l2": float(np.mean(np.linalg.norm(refined - selected_chunk, axis=-1))),
                }
            box_rows.append(branch)
        results.append({"box_idx": box_idx, "box": box, "branches": box_rows})

    summary = {}
    for row in results:
        box = row["box"]
        true_dist = signed_distance_xy(eef_xy, box)
        aggregate = {
            "true_inside_count": int(np.sum(true_dist < 0.0)),
            "true_min_signed_distance_m": float(np.min(true_dist)),
            "by_gradient_steps": {},
        }
        for gradient_steps in gradient_steps_values:
            key = str(gradient_steps)
            inside = np.asarray(
                [branch["predictions"][key]["inside_count"] for branch in row["branches"]],
                dtype=np.int64,
            )
            mins = np.asarray(
                [branch["predictions"][key]["min_signed_distance_m"] for branch in row["branches"]],
                dtype=np.float64,
            )
            deltas = np.asarray(
                [branch["predictions"][key]["mean_action_delta_l2"] for branch in row["branches"]],
                dtype=np.float64,
            )
            aggregate["by_gradient_steps"][key] = {
                "chunks_with_predicted_violation": int(np.sum(inside > 0)),
                "total_predicted_inside_points": int(np.sum(inside)),
                "min_signed_distance_m": float(np.min(mins)),
                "median_min_signed_distance_m": float(np.median(mins)),
                "mean_action_delta_l2": float(np.mean(deltas)),
            }
        summary[str(row["box_idx"])] = aggregate

    return eef_xy, obj_xy, results, summary


def plot_grid(
    rollout_dir: Path,
    eef_xy: np.ndarray,
    obj_xy: np.ndarray,
    results: list[dict],
    gradient_steps_values: list[int],
    output_stem: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), squeeze=False, constrained_layout=True)

    for ax, result in zip(axes.ravel(), results):
        box = result["box"]
        draw_box(ax, box)
        if len(obj_xy):
            ax.scatter(obj_xy[:, 0], obj_xy[:, 1], color="#b8bec8", alpha=0.14, s=7, linewidths=0)
        ax.plot(eef_xy[:, 0], eef_xy[:, 1], color="#111111", linewidth=1.55, zorder=4)
        ax.scatter(eef_xy[0, 0], eef_xy[0, 1], marker="o", s=20, color="#111111", zorder=5)
        ax.scatter(eef_xy[-1, 0], eef_xy[-1, 1], marker="x", s=34, color="#111111", zorder=5)

        for branch in result["branches"]:
            start_xy = np.asarray(branch["start_xy"], dtype=np.float64)
            for idx, gradient_steps in enumerate(gradient_steps_values):
                pred_xy = np.asarray(branch["predictions"][str(gradient_steps)]["eef_xy"], dtype=np.float64)
                path_xy = np.vstack([start_xy[None, :], pred_xy])
                ax.plot(
                    path_xy[:, 0],
                    path_xy[:, 1],
                    color=step_color(gradient_steps, idx),
                    alpha=0.24 if gradient_steps else 0.18,
                    linewidth=0.9,
                    zorder=3,
                )
                ax.scatter(
                    path_xy[-1, 0],
                    path_xy[-1, 1],
                    color=step_color(gradient_steps, idx),
                    alpha=0.42,
                    s=7,
                    linewidths=0,
                    zorder=4,
                )

        true_dist = signed_distance_xy(eef_xy, box)
        dy = box[2] - DEFAULT_BOX[2]
        ax.set_title(
            f"dy={dy:+.2f}m  true min={1000*np.min(true_dist):+.1f}mm",
            fontsize=9,
        )
        x_min, x_max, y_min, y_max = box
        ax.set_xlim(min(np.min(eef_xy[:, 0]) - 0.01, x_min - 0.055), max(np.max(eef_xy[:, 0]) + 0.01, x_max + 0.055))
        ax.set_ylim(min(np.min(eef_xy[:, 1]) - 0.01, y_min - 0.05), max(np.max(eef_xy[:, 1]) + 0.01, y_max + 0.05))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22, linewidth=0.45)
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")

    handles = [
        Line2D([0], [0], color="#111111", linewidth=1.7, label="true executed EEF"),
        Line2D([0], [0], color="#d62728", linewidth=5.5, alpha=0.22, label="shifted forbidden square"),
    ]
    for idx, gradient_steps in enumerate(gradient_steps_values):
        handles.append(
            Line2D([0], [0], color=step_color(gradient_steps, idx), linewidth=1.8, label=f"grad steps={gradient_steps}")
        )
    axes[0, 0].legend(handles=handles, loc="best", frameon=True)
    fig.suptitle(f"{rollout_dir.name}: y-shifted square, gradient-step sweep")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--dynamics-ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="rollout_006_y_shift_gradient_grid")
    parser.add_argument("--box", nargs=4, type=float, default=DEFAULT_BOX, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--n-shifts", type=int, default=10)
    parser.add_argument("--shift-step-y", type=float, default=-0.01)
    parser.add_argument("--gradient-steps-values", default=",".join(str(v) for v in DEFAULT_GRADIENT_STEPS))
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.0003)
    parser.add_argument("--action-reg", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_box = (
        min(args.box[0], args.box[1]),
        max(args.box[0], args.box[1]),
        min(args.box[2], args.box[3]),
        max(args.box[2], args.box[3]),
    )
    boxes = [shifted_box(base_box, idx * args.shift_step_y) for idx in range(args.n_shifts)]
    gradient_steps_values = [int(value) for value in args.gradient_steps_values.split(",") if value.strip()]

    events, _, _ = read_rollout(args.rollout_dir)
    dynamics_ckpt = infer_dynamics_ckpt(chunk_events(events), args.dynamics_ckpt)
    device = torch.device(args.device)
    model, stats = load_dynamics(dynamics_ckpt, device)

    eef_xy, obj_xy, results, aggregate = build_predictions(
        args.rollout_dir,
        model,
        stats,
        device,
        boxes,
        gradient_steps_values,
        guidance_scale=args.guidance_scale,
        step_size=args.step_size,
        action_reg=args.action_reg,
        tau=args.tau,
        margin=args.margin,
        rollout_scale=args.rollout_scale,
    )

    output_stem = args.output_dir / args.stem
    plot_grid(args.rollout_dir, eef_xy, obj_xy, results, gradient_steps_values, output_stem)

    summary = {
        "rollout_dir": str(args.rollout_dir),
        "dynamics_ckpt": str(dynamics_ckpt),
        "base_box": {"x_min": base_box[0], "x_max": base_box[1], "y_min": base_box[2], "y_max": base_box[3]},
        "shift_step_y": args.shift_step_y,
        "n_shifts": args.n_shifts,
        "guidance": {
            "gradient_steps_values": gradient_steps_values,
            "guidance_scale": args.guidance_scale,
            "step_size": args.step_size,
            "action_reg": args.action_reg,
            "tau": args.tau,
            "margin": args.margin,
            "rollout_scale": args.rollout_scale,
        },
        "aggregate_by_shift": aggregate,
        "plots": {"png": str(output_stem.with_suffix(".png")), "pdf": str(output_stem.with_suffix(".pdf"))},
    }
    summary_path = output_stem.with_name(output_stem.name + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"dynamics: {dynamics_ckpt}")
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {summary_path}")
    for idx, box in enumerate(boxes):
        agg = aggregate[str(idx)]
        print(
            f"shift {idx:02d} dy={idx * args.shift_step_y:+.2f} "
            f"box_y=[{box[2]:+.3f},{box[3]:+.3f}] "
            f"true_inside={agg['true_inside_count']} "
            f"true_min={1000.0 * agg['true_min_signed_distance_m']:+.1f}mm"
        )
        for steps in gradient_steps_values:
            row = agg["by_gradient_steps"][str(steps)]
            print(
                f"  steps={steps:>2}: chunks_violate={row['chunks_with_predicted_violation']}, "
                f"inside_pred={row['total_predicted_inside_points']}, "
                f"min={1000.0 * row['min_signed_distance_m']:+.1f}mm, "
                f"delta={row['mean_action_delta_l2']:.5f}"
            )


if __name__ == "__main__":
    main()
