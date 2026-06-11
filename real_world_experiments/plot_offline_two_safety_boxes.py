#!/usr/bin/env python3
"""Offline safety-guidance plot for two forbidden XY boxes."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

from plot_offline_safety_guidance_branches import (
    DEFAULT_OUTPUT_DIR,
    chunk_events,
    configure_matplotlib,
    dynamics_rollout,
    infer_dynamics_ckpt,
    load_dynamics,
    obs_to_state,
    predict_branch,
    read_rollout,
    signed_distance_xy,
)


DEFAULT_ROLLOUT = Path(
    "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    "automaton_left_epoch160_n10_3/rollouts/rollout_006"
)
DEFAULT_BOXES = [
    (-0.60, -0.55, -0.10, -0.05),
    (-0.60, -0.55, -0.18, -0.13),
]
STEP_COLORS = {
    0: "#ff8c00",
    10: "#2ca02c",
    20: "#1f77b4",
    30: "#d62728",
}


def normalize_box(raw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = raw
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def torch_signed_distances_to_boxes(
    pred_xy: torch.Tensor,
    boxes: list[tuple[float, float, float, float]],
) -> torch.Tensor:
    distances = []
    for x_min, x_max, y_min, y_max in boxes:
        center = torch.tensor(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0],
            device=pred_xy.device,
            dtype=pred_xy.dtype,
        )
        half = torch.tensor(
            [(x_max - x_min) / 2.0, (y_max - y_min) / 2.0],
            device=pred_xy.device,
            dtype=pred_xy.dtype,
        )
        q = torch.abs(pred_xy - center) - half
        distances.append(
            torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
            + torch.minimum(
                torch.maximum(q[..., 0], q[..., 1]),
                torch.zeros((), device=pred_xy.device, dtype=pred_xy.dtype),
            )
        )
    return torch.stack(distances, dim=-1)


def multi_box_robustness(
    state0: torch.Tensor,
    actions: torch.Tensor,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    boxes: list[tuple[float, float, float, float]],
    *,
    margin: float,
    tau: float,
    rollout_scale: float,
) -> torch.Tensor:
    pred_xy = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)[..., :2]
    distances = torch_signed_distances_to_boxes(pred_xy, boxes)
    capped = torch.clamp(distances.reshape(distances.shape[0], -1), max=margin)
    return -tau * torch.logsumexp(-capped / tau, dim=-1)


def refine_for_multi_box_safety(
    state_np: np.ndarray,
    actions_np: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    boxes: list[tuple[float, float, float, float]],
    *,
    guidance_scale: float,
    gradient_steps: int,
    step_size: float,
    action_reg: float,
    tau: float,
    margin: float,
    rollout_scale: float,
) -> np.ndarray:
    if guidance_scale == 0.0 or gradient_steps <= 0 or step_size == 0.0:
        return np.asarray(actions_np, dtype=np.float32)
    state0 = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
    original = torch.as_tensor(actions_np[None], device=device, dtype=torch.float32)
    actions = original.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([actions], lr=step_size)
    for _ in range(gradient_steps):
        opt.zero_grad(set_to_none=True)
        robustness = multi_box_robustness(
            state0,
            actions,
            model,
            stats,
            boxes,
            margin=margin,
            tau=tau,
            rollout_scale=rollout_scale,
        )
        regularizer = torch.mean((actions - original) ** 2, dim=(1, 2))
        objective = guidance_scale * robustness - action_reg * regularizer
        (-objective.mean()).backward()
        opt.step()
        with torch.no_grad():
            actions.clamp_(-1.0, 1.0)
    return actions[0].detach().cpu().numpy().astype(np.float32)


def multi_box_stats(xy: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    per_box = []
    all_dist = []
    for box_idx, box in enumerate(boxes):
        dist = signed_distance_xy(xy, box)
        all_dist.append(dist)
        per_box.append(
            {
                "box_idx": box_idx,
                "box": list(map(float, box)),
                "inside_count": int(np.sum(dist < 0.0)),
                "min_signed_distance_m": float(np.min(dist)),
            }
        )
    stacked = np.stack(all_dist, axis=-1)
    union_dist = np.min(stacked, axis=-1)
    return {
        "inside_count": int(np.sum(union_dist < 0.0)),
        "min_signed_distance_m": float(np.min(union_dist)),
        "per_box": per_box,
    }


def draw_boxes(ax, boxes: list[tuple[float, float, float, float]]) -> None:
    colors = ["#d62728", "#9467bd"]
    for idx, (x_min, x_max, y_min, y_max) in enumerate(boxes):
        color = colors[idx % len(colors)]
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=color,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.16,
                zorder=1,
            )
        )
        ax.text(x_max + 0.002, 0.5 * (y_min + y_max), f"box {idx}", fontsize=7, color=color, va="center")


def step_color(steps: int, idx: int) -> str:
    if steps in STEP_COLORS:
        return STEP_COLORS[steps]
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % 10]


def build_branches(
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
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    events, eef_xy, obj_xy = read_rollout(rollout_dir)
    chunks = chunk_events(events)
    if not chunks:
        raise ValueError(f"No chunk_sample events found in {rollout_dir}")

    branches = []
    for chunk in chunks:
        state_np = obs_to_state(chunk["obs"])
        selected_chunk = np.asarray(chunk["selected_chunk"], dtype=np.float32)
        branch = {
            "chunk_idx": int(chunk["chunk_idx"]),
            "decision_idx_before": int(chunk["decision_idx_before"]),
            "start_xy": np.asarray(chunk["obs"]["eef_pos"][:2], dtype=np.float64).tolist(),
            "predictions": {},
        }
        for steps in gradient_steps_values:
            refined = refine_for_multi_box_safety(
                state_np,
                selected_chunk,
                model,
                stats,
                device,
                boxes,
                guidance_scale=guidance_scale,
                gradient_steps=steps,
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
            pred_stats = multi_box_stats(pred_xy, boxes)
            branch["predictions"][str(steps)] = {
                "eef_xy": pred_xy.astype(float).tolist(),
                **pred_stats,
                "mean_action_delta_l2": float(np.mean(np.linalg.norm(refined - selected_chunk, axis=-1))),
            }
        branches.append(branch)

    true_stats = multi_box_stats(eef_xy, boxes)
    summary = {
        "rollout_dir": str(rollout_dir),
        "boxes": [list(map(float, box)) for box in boxes],
        "true_path": true_stats,
        "gradient_steps": {},
    }
    for steps in gradient_steps_values:
        key = str(steps)
        inside = np.asarray([branch["predictions"][key]["inside_count"] for branch in branches], dtype=np.int64)
        mins = np.asarray(
            [branch["predictions"][key]["min_signed_distance_m"] for branch in branches],
            dtype=np.float64,
        )
        deltas = np.asarray(
            [branch["predictions"][key]["mean_action_delta_l2"] for branch in branches],
            dtype=np.float64,
        )
        summary["gradient_steps"][key] = {
            "chunks_with_predicted_violation": int(np.sum(inside > 0)),
            "total_predicted_inside_points": int(np.sum(inside)),
            "min_signed_distance_m_min": float(np.min(mins)),
            "min_signed_distance_m_mean": float(np.mean(mins)),
            "mean_action_delta_l2_mean": float(np.mean(deltas)),
            "mean_action_delta_l2_max": float(np.max(deltas)),
        }
    return eef_xy, obj_xy, branches, summary


def plot(
    rollout_dir: Path,
    eef_xy: np.ndarray,
    obj_xy: np.ndarray,
    branches: list[dict[str, Any]],
    boxes: list[tuple[float, float, float, float]],
    gradient_steps_values: list[int],
    output_stem: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    titles = ["full rollout", "zoom around double safety region"]

    for ax, title in zip(axes, titles):
        draw_boxes(ax, boxes)
        if len(obj_xy):
            ax.scatter(obj_xy[:, 0], obj_xy[:, 1], color="#b8bec8", alpha=0.18, s=8, linewidths=0)
        ax.plot(eef_xy[:, 0], eef_xy[:, 1], color="#111111", linewidth=1.8, label="true executed EEF", zorder=4)
        ax.scatter(eef_xy[0, 0], eef_xy[0, 1], marker="o", s=24, color="#111111", zorder=5)
        ax.scatter(eef_xy[-1, 0], eef_xy[-1, 1], marker="x", s=42, color="#111111", zorder=5)

        for branch in branches:
            start_xy = np.asarray(branch["start_xy"], dtype=np.float64)
            ax.scatter(start_xy[0], start_xy[1], color="#111111", s=9, alpha=0.35, zorder=5)
            for value_idx, steps in enumerate(gradient_steps_values):
                pred_xy = np.asarray(branch["predictions"][str(steps)]["eef_xy"], dtype=np.float64)
                path_xy = np.vstack([start_xy[None, :], pred_xy])
                ax.plot(
                    path_xy[:, 0],
                    path_xy[:, 1],
                    color=step_color(steps, value_idx),
                    alpha=0.38 if steps else 0.24,
                    linewidth=1.05,
                    zorder=3,
                )
                ax.scatter(
                    path_xy[-1, 0],
                    path_xy[-1, 1],
                    color=step_color(steps, value_idx),
                    alpha=0.55,
                    s=9,
                    linewidths=0,
                    zorder=4,
                )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.set_title(title)

    all_xy = [eef_xy]
    for branch in branches:
        for steps in gradient_steps_values:
            all_xy.append(np.asarray(branch["predictions"][str(steps)]["eef_xy"], dtype=np.float64))
    stacked = np.concatenate([xy for xy in all_xy if len(xy)], axis=0)
    x_pad = max(0.01, 0.08 * float(np.ptp(stacked[:, 0])))
    y_pad = max(0.01, 0.08 * float(np.ptp(stacked[:, 1])))
    axes[0].set_xlim(float(stacked[:, 0].min() - x_pad), float(stacked[:, 0].max() + x_pad))
    axes[0].set_ylim(float(stacked[:, 1].min() - y_pad), float(stacked[:, 1].max() + y_pad))

    x_min = min(box[0] for box in boxes)
    x_max = max(box[1] for box in boxes)
    y_min = min(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    axes[1].set_xlim(x_min - 0.06, x_max + 0.06)
    axes[1].set_ylim(y_min - 0.035, y_max + 0.035)

    handles = [
        Line2D([0], [0], color="#111111", linewidth=1.8, label="true executed EEF"),
        Line2D([0], [0], color="#d62728", linewidth=6.0, alpha=0.22, label="upper forbidden box"),
        Line2D([0], [0], color="#9467bd", linewidth=6.0, alpha=0.22, label="lower forbidden box"),
    ]
    for idx, steps in enumerate(gradient_steps_values):
        handles.append(Line2D([0], [0], color=step_color(steps, idx), linewidth=1.8, label=f"steps={steps}"))
    axes[0].legend(handles=handles, loc="best", frameon=True)
    fig.suptitle(f"{rollout_dir.name}: two-box offline safety optimization")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_boxes(values: list[float]) -> list[tuple[float, float, float, float]]:
    if len(values) % 4 != 0:
        raise ValueError("--boxes expects groups of four numbers: x_min x_max y_min y_max")
    return [normalize_box(tuple(values[idx : idx + 4])) for idx in range(0, len(values), 4)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--dynamics-ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="rollout_006_two_safety_boxes")
    parser.add_argument("--boxes", nargs="+", type=float, default=[v for box in DEFAULT_BOXES for v in box])
    parser.add_argument("--gradient-steps-values", nargs="+", type=int, default=[0, 10, 20, 30])
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
    boxes = parse_boxes(args.boxes)
    events, _, _ = read_rollout(args.rollout_dir)
    chunks = chunk_events(events)
    ckpt = infer_dynamics_ckpt(chunks, args.dynamics_ckpt)
    device = torch.device(args.device)
    model, stats = load_dynamics(ckpt, device)
    eef_xy, obj_xy, branches, summary = build_branches(
        args.rollout_dir,
        model,
        stats,
        device,
        boxes,
        args.gradient_steps_values,
        guidance_scale=args.guidance_scale,
        step_size=args.step_size,
        action_reg=args.action_reg,
        tau=args.tau,
        margin=args.margin,
        rollout_scale=args.rollout_scale,
    )
    summary.update(
        {
            "dynamics_ckpt": str(ckpt),
            "params": {
                "guidance_scale": args.guidance_scale,
                "step_size": args.step_size,
                "action_reg": args.action_reg,
                "tau": args.tau,
                "margin": args.margin,
                "rollout_scale": args.rollout_scale,
            },
        }
    )
    output_stem = args.output_dir / args.stem
    plot(args.rollout_dir, eef_xy, obj_xy, branches, boxes, args.gradient_steps_values, output_stem)
    summary_path = output_stem.with_name(output_stem.name + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {summary_path}")
    print(
        "true path: "
        f"inside={summary['true_path']['inside_count']}, "
        f"min={summary['true_path']['min_signed_distance_m'] * 1000.0:+.1f}mm"
    )
    for steps in args.gradient_steps_values:
        row = summary["gradient_steps"][str(steps)]
        print(
            f"steps={steps:>2}: chunks_violate="
            f"{row['chunks_with_predicted_violation']}/{len(branches)}, "
            f"min={row['min_signed_distance_m_min'] * 1000.0:+.1f}mm, "
            f"delta={row['mean_action_delta_l2_mean']:.5f}"
        )


if __name__ == "__main__":
    main()
