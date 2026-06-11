#!/usr/bin/env python3
"""Paired offline safety-guidance action-regularization sweep plot."""

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
import numpy as np
import torch

from plot_offline_safety_guidance_branches import (
    DEFAULT_BOX,
    DEFAULT_OUTPUT_DIR,
    chunk_events,
    configure_matplotlib,
    draw_box,
    infer_dynamics_ckpt,
    load_dynamics,
    obs_to_state,
    predict_branch,
    read_rollout,
    refine_for_safety,
    scale_color,
    signed_distance_xy,
)


DEFAULT_ROLLOUTS = (
    Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3/rollouts/rollout_004"),
    Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3/rollouts/rollout_006"),
)
DEFAULT_ACTION_REGS = (0.0, 0.0005, 0.005, 0.05, 0.5, 5.0)


def build_branch_payload(
    rollout_dir: Path,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    action_regs: list[float],
    box: tuple[float, float, float, float],
    guidance_scale: float,
    gradient_steps: int,
    step_size: float,
    tau: float,
    margin: float,
    rollout_scale: float,
) -> dict:
    events, eef_xy, obj_xy = read_rollout(rollout_dir)
    chunks = chunk_events(events)
    if not chunks:
        raise ValueError(f"No chunk_sample events found in {rollout_dir}")

    branches = []
    for chunk in chunks:
        obs = chunk["obs"]
        state_np = obs_to_state(obs)
        selected_chunk = np.asarray(chunk["selected_chunk"], dtype=np.float32)
        start_xy = np.asarray(obs["eef_pos"][:2], dtype=np.float64)
        predictions = {}
        for action_reg in action_regs:
            refined_actions = refine_for_safety(
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
                refined_actions,
                model,
                stats,
                device,
                rollout_scale=rollout_scale,
            )
            pred_xy = pred_state[:, :2]
            dist = signed_distance_xy(pred_xy, box)
            predictions[str(action_reg)] = {
                "eef_xy": pred_xy.astype(float).tolist(),
                "min_signed_distance_m": float(np.min(dist)),
                "inside_count": int(np.sum(dist < 0.0)),
                "mean_action_delta_l2": float(np.mean(np.linalg.norm(refined_actions - selected_chunk, axis=-1))),
            }
        branches.append(
            {
                "chunk_idx": int(chunk["chunk_idx"]),
                "decision_idx_before": int(chunk["decision_idx_before"]),
                "start_xy": start_xy.astype(float).tolist(),
                "selected_candidate": int(chunk.get("selected_candidate", -1)),
                "predictions": predictions,
            }
        )

    true_dist = signed_distance_xy(eef_xy, box)
    return {
        "rollout_dir": str(rollout_dir),
        "rollout_name": rollout_dir.name,
        "eef_xy": eef_xy,
        "obj_xy": obj_xy,
        "branches": branches,
        "true_path": {
            "n_points": int(len(eef_xy)),
            "inside_count": int(np.sum(true_dist < 0.0)),
            "min_signed_distance_m": float(np.min(true_dist)),
        },
    }


def aggregate(payload: dict, action_regs: list[float]) -> dict:
    rows = {}
    for action_reg in action_regs:
        key = str(action_reg)
        inside = np.asarray(
            [branch["predictions"][key]["inside_count"] for branch in payload["branches"]],
            dtype=np.int64,
        )
        mins = np.asarray(
            [branch["predictions"][key]["min_signed_distance_m"] for branch in payload["branches"]],
            dtype=np.float64,
        )
        deltas = np.asarray(
            [branch["predictions"][key]["mean_action_delta_l2"] for branch in payload["branches"]],
            dtype=np.float64,
        )
        rows[key] = {
            "chunks_with_predicted_violation": int(np.sum(inside > 0)),
            "total_predicted_inside_points": int(np.sum(inside)),
            "min_signed_distance_m": float(np.min(mins)),
            "median_min_signed_distance_m": float(np.median(mins)),
            "mean_action_delta_l2": float(np.mean(deltas)),
        }
    return rows


def plot_pair(
    payloads: list[dict],
    action_regs: list[float],
    box: tuple[float, float, float, float],
    output_stem: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(
        len(payloads),
        2,
        figsize=(13.6, 5.25 * len(payloads)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, payload in enumerate(payloads):
        eef_xy = payload["eef_xy"]
        obj_xy = payload["obj_xy"]
        branches = payload["branches"]
        for col_idx, ax in enumerate(axes[row_idx]):
            draw_box(ax, box)
            if len(obj_xy):
                ax.scatter(obj_xy[:, 0], obj_xy[:, 1], color="#b8bec8", alpha=0.18, s=8, linewidths=0)
            ax.plot(eef_xy[:, 0], eef_xy[:, 1], color="#111111", linewidth=1.8, zorder=4)
            ax.scatter(eef_xy[0, 0], eef_xy[0, 1], marker="o", s=24, color="#111111", zorder=5)
            ax.scatter(eef_xy[-1, 0], eef_xy[-1, 1], marker="x", s=42, color="#111111", zorder=5)

            for branch in branches:
                start_xy = np.asarray(branch["start_xy"], dtype=np.float64)
                ax.scatter(start_xy[0], start_xy[1], color="#111111", s=9, alpha=0.35, zorder=5)
                for value_idx, action_reg in enumerate(action_regs):
                    pred_xy = np.asarray(branch["predictions"][str(action_reg)]["eef_xy"], dtype=np.float64)
                    path_xy = np.vstack([start_xy[None, :], pred_xy])
                    ax.plot(
                        path_xy[:, 0],
                        path_xy[:, 1],
                        color=scale_color(action_reg, value_idx),
                        alpha=0.35 if action_reg != 0.0 else 0.25,
                        linewidth=1.0,
                        zorder=3,
                    )
                    ax.scatter(
                        path_xy[-1, 0],
                        path_xy[-1, 1],
                        color=scale_color(action_reg, value_idx),
                        alpha=0.48,
                        s=8,
                        linewidths=0,
                        zorder=4,
                    )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_xlabel("world x [m]")
            ax.set_ylabel("world y [m]")
            ax.set_title(f"{payload['rollout_name']} {'full' if col_idx == 0 else 'zoom'}")

        all_xy = [eef_xy]
        for branch in branches:
            for action_reg in action_regs:
                all_xy.append(np.asarray(branch["predictions"][str(action_reg)]["eef_xy"], dtype=np.float64))
        stacked = np.concatenate([xy for xy in all_xy if len(xy)], axis=0)
        x_pad = max(0.01, 0.08 * float(np.ptp(stacked[:, 0])))
        y_pad = max(0.01, 0.08 * float(np.ptp(stacked[:, 1])))
        axes[row_idx, 0].set_xlim(float(stacked[:, 0].min() - x_pad), float(stacked[:, 0].max() + x_pad))
        axes[row_idx, 0].set_ylim(float(stacked[:, 1].min() - y_pad), float(stacked[:, 1].max() + y_pad))

        x_min, x_max, y_min, y_max = box
        axes[row_idx, 1].set_xlim(x_min - 0.055, x_max + 0.055)
        axes[row_idx, 1].set_ylim(y_min - 0.06, y_max + 0.06)

    handles = [
        Line2D([0], [0], color="#111111", linewidth=1.8, label="true executed EEF"),
        Line2D([0], [0], color="#b8bec8", marker="o", linestyle="", markersize=5, label="object poses"),
        Line2D([0], [0], color="#d62728", linewidth=6.0, alpha=0.22, label="forbidden square"),
    ]
    for idx, action_reg in enumerate(action_regs):
        handles.append(
            Line2D([0], [0], color=scale_color(action_reg, idx), linewidth=1.8, label=f"action_reg={action_reg:g}")
        )
    axes[0, 0].legend(handles=handles, loc="best", frameon=True)
    fig.suptitle("Offline safety guidance: action regularization sweep")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dirs", nargs="+", type=Path, default=list(DEFAULT_ROLLOUTS))
    parser.add_argument("--dynamics-ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="automaton_left_epoch160_n10_3_action_reg_sweep_pair")
    parser.add_argument("--action-regs", default=",".join(str(v) for v in DEFAULT_ACTION_REGS))
    parser.add_argument("--box", nargs=4, type=float, default=DEFAULT_BOX, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--gradient-steps", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.0003)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    action_regs = [float(value) for value in args.action_regs.split(",") if value.strip()]
    box = (
        min(args.box[0], args.box[1]),
        max(args.box[0], args.box[1]),
        min(args.box[2], args.box[3]),
        max(args.box[2], args.box[3]),
    )

    first_events, _, _ = read_rollout(args.rollout_dirs[0])
    dynamics_ckpt = infer_dynamics_ckpt(chunk_events(first_events), args.dynamics_ckpt)
    device = torch.device(args.device)
    model, stats = load_dynamics(dynamics_ckpt, device)

    payloads = [
        build_branch_payload(
            rollout_dir,
            model,
            stats,
            device,
            action_regs=action_regs,
            box=box,
            guidance_scale=args.guidance_scale,
            gradient_steps=args.gradient_steps,
            step_size=args.step_size,
            tau=args.tau,
            margin=args.margin,
            rollout_scale=args.rollout_scale,
        )
        for rollout_dir in args.rollout_dirs
    ]

    output_stem = args.output_dir / args.stem
    plot_pair(payloads, action_regs, box, output_stem)

    summary = {
        "rollout_dirs": [str(path) for path in args.rollout_dirs],
        "dynamics_ckpt": str(dynamics_ckpt),
        "box": {"x_min": box[0], "x_max": box[1], "y_min": box[2], "y_max": box[3], "margin": args.margin},
        "guidance": {
            "sweep_param": "action_reg",
            "action_regs": action_regs,
            "guidance_scale": args.guidance_scale,
            "gradient_steps": args.gradient_steps,
            "step_size": args.step_size,
            "tau": args.tau,
            "rollout_scale": args.rollout_scale,
        },
        "rollouts": [],
        "plots": {"png": str(output_stem.with_suffix(".png")), "pdf": str(output_stem.with_suffix(".pdf"))},
    }

    print(f"dynamics: {dynamics_ckpt}")
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    for payload in payloads:
        agg = aggregate(payload, action_regs)
        compact_payload = {
            "rollout_dir": payload["rollout_dir"],
            "rollout_name": payload["rollout_name"],
            "true_path": payload["true_path"],
            "aggregate_by_action_reg": agg,
        }
        summary["rollouts"].append(compact_payload)
        print(
            f"{payload['rollout_name']}: true inside={payload['true_path']['inside_count']}, "
            f"min_dist={1000.0 * payload['true_path']['min_signed_distance_m']:.1f}mm"
        )
        for action_reg in action_regs:
            row = agg[str(action_reg)]
            print(
                f"  action_reg={action_reg:g}: "
                f"chunks_with_violation={row['chunks_with_predicted_violation']}/{len(payload['branches'])}, "
                f"inside_pred_points={row['total_predicted_inside_points']}, "
                f"min_dist={1000.0 * row['min_signed_distance_m']:.1f}mm, "
                f"mean_action_delta_l2={row['mean_action_delta_l2']:.5f}"
            )

    summary_path = output_stem.with_name(output_stem.name + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
