#!/usr/bin/env python3
"""Offline safety-guidance branch plots from real-world rollout event logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world_experiments.train_dynamics_world_model import load_dynamics_model_for_eval


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/safety_guidance_tuning"
DEFAULT_BOX = (-0.60, -0.55, -0.10, -0.05)
DEFAULT_SCALES = (0.0, 1.0, 5.0)
COLORS = {
    0.0: "#ff8c00",
    0.0005: "#bcbd22",
    0.005: "#2ca02c",
    0.01: "#17becf",
    0.02: "#9467bd",
    0.05: "#d62728",
    0.1: "#8c564b",
    0.5: "#9467bd",
    1.0: "#1f77b4",
    5.0: "#17becf",
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
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "rollout_end":
        return event.get("final_obs")
    if event.get("type") in {"rollout_start", "decision", "chunk_sample"}:
        return event.get("obs")
    return None


def obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["eef_pos"], dtype=np.float32),
            np.asarray(obs["eef_rot6d"], dtype=np.float32),
            np.asarray(obs["gripper_binary"], dtype=np.float32).reshape(-1),
            np.asarray(obs["cheezit_pos"], dtype=np.float32),
            np.asarray(obs["cheezit_rot6d"], dtype=np.float32),
        ]
    ).astype(np.float32)


def project_rw_dyn_state(state: torch.Tensor) -> torch.Tensor:
    def project_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
        first = rot6d[..., :3] / (rot6d[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
        second = rot6d[..., 3:] - (first * rot6d[..., 3:]).sum(-1, keepdim=True) * first
        second = second / (second.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.cat([first, second], dim=-1)

    return torch.cat(
        [
            state[..., :3],
            project_rot6d(state[..., 3:9]),
            state[..., 9:13],
            project_rot6d(state[..., 13:19]),
        ],
        dim=-1,
    )


def dynamics_rollout(
    state0: torch.Tensor,
    actions: torch.Tensor,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    *,
    rollout_scale: float,
) -> torch.Tensor:
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    state = state0.expand(actions.shape[0], -1)
    states = []
    for idx in range(actions.shape[1]):
        delta_norm = model(
            (state - stats["state_mean"]) / stats["state_std"],
            ((rollout_scale * actions[:, idx]) - stats["action_mean"]) / stats["action_std"],
        )
        state = project_rw_dyn_state(state + delta_norm * stats["delta_std"] + stats["delta_mean"])
        states.append(state)
    return torch.stack(states, dim=1)


def signed_distance_xy(xy: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = box
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float64)
    half = np.array([(x_max - x_min) / 2.0, (y_max - y_min) / 2.0], dtype=np.float64)
    q = np.abs(np.asarray(xy, dtype=np.float64) - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside


def safety_robustness(
    state0: torch.Tensor,
    actions: torch.Tensor,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    box: tuple[float, float, float, float],
    *,
    margin: float,
    tau: float,
    rollout_scale: float,
) -> torch.Tensor:
    x_min, x_max, y_min, y_max = box
    pred_xy = dynamics_rollout(
        state0,
        actions,
        model,
        stats,
        rollout_scale=rollout_scale,
    )[..., :2]
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
    dist = (
        torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        + torch.minimum(torch.maximum(q[..., 0], q[..., 1]), torch.zeros((), device=pred_xy.device))
    )
    return -tau * torch.logsumexp(-torch.clamp(dist, max=margin) / tau, dim=-1)


def refine_for_safety(
    state_np: np.ndarray,
    actions_np: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    box: tuple[float, float, float, float],
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
        robust = safety_robustness(
            state0,
            actions,
            model,
            stats,
            box,
            margin=margin,
            tau=tau,
            rollout_scale=rollout_scale,
        )
        regularizer = torch.mean((actions - original) ** 2, dim=(1, 2))
        objective = guidance_scale * robust - action_reg * regularizer
        (-objective.mean()).backward()
        opt.step()
        with torch.no_grad():
            actions.clamp_(-1.0, 1.0)
    return actions[0].detach().cpu().numpy().astype(np.float32)


def read_rollout(rollout_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    events = []
    eef_xy = []
    obj_xy = []
    with (rollout_dir / "events.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            events.append(event)
            obs = obs_from_event(event)
            if obs and "eef_pos" in obs:
                eef_xy.append(obs["eef_pos"][:2])
            if obs and "cheezit_pos" in obs:
                obj_xy.append(obs["cheezit_pos"][:2])
            if event.get("type") == "rollout_end":
                break
    return (
        events,
        np.asarray(eef_xy, dtype=np.float64),
        np.asarray(obj_xy, dtype=np.float64),
    )


def chunk_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == "chunk_sample"]


def pick_rollout(run_dir: Path, box: tuple[float, float, float, float]) -> Path:
    best = None
    best_key = None
    for rollout_dir in sorted((run_dir / "rollouts").glob("rollout_*")):
        summary_path = rollout_dir / "rollout_summary.json"
        events_path = rollout_dir / "events.jsonl"
        if not summary_path.exists() or not events_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if not summary.get("success", False):
            continue
        event_names = [event.get("label_name") for event in summary.get("label_events", [])]
        if "pouring_left" not in event_names:
            continue
        _, eef_xy, _ = read_rollout(rollout_dir)
        if len(eef_xy) == 0:
            continue
        dist = signed_distance_xy(eef_xy, box)
        key = (int(np.sum(dist < 0.0)), -float(np.min(dist)), len(eef_xy))
        if best_key is None or key > best_key:
            best_key = key
            best = rollout_dir
    if best is None:
        raise ValueError(f"No successful pour-left rollout found under {run_dir}")
    return best


def infer_dynamics_ckpt(chunks: list[dict[str, Any]], fallback: Path | None) -> Path:
    for chunk in chunks:
        dyn = chunk.get("dynamics_prediction") or {}
        ckpt = dyn.get("model_ckpt_path")
        if ckpt:
            return Path(ckpt)
    if fallback is not None:
        return fallback
    raise ValueError("No dynamics model path found in chunk_sample logs; pass --dynamics-ckpt")


def load_dynamics(ckpt_path: Path, device: torch.device):
    model, stats_np, _checkpoint = load_dynamics_model_for_eval(ckpt_path, device=device)
    stats = {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in stats_np.items()
    }
    model.eval()
    return model, stats


def predict_branch(
    state_np: np.ndarray,
    actions_np: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    rollout_scale: float,
) -> np.ndarray:
    state0 = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
    actions = torch.as_tensor(actions_np[None], device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)[0]
    return pred.detach().cpu().numpy()


def scale_color(scale: float, idx: int) -> str:
    if scale in COLORS:
        return COLORS[scale]
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return fallback[idx % len(fallback)]


def draw_box(ax, box: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = box
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="#d62728",
            edgecolor="#d62728",
            linewidth=1.3,
            alpha=0.16,
            zorder=1,
        )
    )


def plot_branches(
    rollout_dir: Path,
    eef_xy: np.ndarray,
    obj_xy: np.ndarray,
    branches: list[dict[str, Any]],
    sweep_values: list[float],
    sweep_param: str,
    box: tuple[float, float, float, float],
    output_stem: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    titles = ["full rollout", "zoom near safety square"]

    for ax, title in zip(axes, titles):
        draw_box(ax, box)
        if len(obj_xy):
            ax.scatter(obj_xy[:, 0], obj_xy[:, 1], color="#b8bec8", alpha=0.18, s=8, linewidths=0)
        ax.plot(eef_xy[:, 0], eef_xy[:, 1], color="#111111", linewidth=1.8, label="true executed EEF", zorder=4)
        ax.scatter(eef_xy[0, 0], eef_xy[0, 1], marker="o", s=24, color="#111111", zorder=5)
        ax.scatter(eef_xy[-1, 0], eef_xy[-1, 1], marker="x", s=42, color="#111111", zorder=5)

        for branch in branches:
            start_xy = np.asarray(branch["start_xy"], dtype=np.float64)
            ax.scatter(start_xy[0], start_xy[1], color="#111111", s=9, alpha=0.35, zorder=5)
            for value_idx, value in enumerate(sweep_values):
                pred_xy = np.asarray(branch["predictions"][str(value)]["eef_xy"], dtype=np.float64)
                path_xy = np.vstack([start_xy[None, :], pred_xy])
                ax.plot(
                    path_xy[:, 0],
                    path_xy[:, 1],
                    color=scale_color(value, value_idx),
                    alpha=0.38 if value != 0.0 else 0.28,
                    linewidth=1.05,
                    zorder=3,
                )
                ax.scatter(
                    path_xy[-1, 0],
                    path_xy[-1, 1],
                    color=scale_color(value, value_idx),
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
        for value in sweep_values:
            all_xy.append(np.asarray(branch["predictions"][str(value)]["eef_xy"], dtype=np.float64))
    stacked = np.concatenate([xy for xy in all_xy if len(xy)], axis=0)
    x_pad = max(0.01, 0.08 * float(np.ptp(stacked[:, 0])))
    y_pad = max(0.01, 0.08 * float(np.ptp(stacked[:, 1])))
    axes[0].set_xlim(float(stacked[:, 0].min() - x_pad), float(stacked[:, 0].max() + x_pad))
    axes[0].set_ylim(float(stacked[:, 1].min() - y_pad), float(stacked[:, 1].max() + y_pad))

    x_min, x_max, y_min, y_max = box
    axes[1].set_xlim(x_min - 0.055, x_max + 0.055)
    axes[1].set_ylim(y_min - 0.06, y_max + 0.06)

    handles = [
        Line2D([0], [0], color="#111111", linewidth=1.8, label="true executed EEF"),
        Line2D([0], [0], color="#b8bec8", marker="o", linestyle="", markersize=5, label="object poses"),
        Line2D([0], [0], color="#d62728", linewidth=6.0, alpha=0.22, label="forbidden square"),
    ]
    label = "scale" if sweep_param == "guidance_scale" else sweep_param
    for idx, value in enumerate(sweep_values):
        handles.append(Line2D([0], [0], color=scale_color(value, idx), linewidth=1.8, label=f"{label}={value:g}"))
    axes[0].legend(handles=handles, loc="best", frameon=True)
    fig.suptitle(f"{rollout_dir.name}: offline safety-guidance branch probes ({label} sweep)")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--rollout-dir", type=Path, default=None)
    parser.add_argument("--dynamics-ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sweep-param", choices=("guidance_scale", "step_size"), default="guidance_scale")
    parser.add_argument("--values", default=None, help="Comma-separated values for --sweep-param.")
    parser.add_argument("--scales", default=",".join(str(v) for v in DEFAULT_SCALES))
    parser.add_argument("--box", nargs=4, type=float, default=DEFAULT_BOX, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--gradient-steps", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.0003)
    parser.add_argument("--action-reg", type=float, default=0.05)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means plot all chunk resamples.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    box = (
        min(args.box[0], args.box[1]),
        max(args.box[0], args.box[1]),
        min(args.box[2], args.box[3]),
        max(args.box[2], args.box[3]),
    )
    raw_values = args.values if args.values is not None else args.scales
    sweep_values = [float(value) for value in raw_values.split(",") if value.strip()]
    rollout_dir = args.rollout_dir or pick_rollout(args.run_dir, box)
    events, eef_xy, obj_xy = read_rollout(rollout_dir)
    chunks = chunk_events(events)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        raise ValueError(f"No chunk_sample events found in {rollout_dir}")

    device = torch.device(args.device)
    dynamics_ckpt = infer_dynamics_ckpt(chunks, args.dynamics_ckpt)
    model, stats = load_dynamics(dynamics_ckpt, device)

    branches = []
    for chunk in chunks:
        obs = chunk["obs"]
        state_np = obs_to_state(obs)
        selected_chunk = np.asarray(chunk["selected_chunk"], dtype=np.float32)
        start_xy = np.asarray(obs["eef_pos"][:2], dtype=np.float64)
        predictions = {}
        for value in sweep_values:
            guidance_scale = args.guidance_scale
            step_size = args.step_size
            if args.sweep_param == "guidance_scale":
                guidance_scale = value
            elif args.sweep_param == "step_size":
                step_size = value
            refined_actions = refine_for_safety(
                state_np,
                selected_chunk,
                model,
                stats,
                device,
                box,
                guidance_scale=guidance_scale,
                gradient_steps=args.gradient_steps,
                step_size=step_size,
                action_reg=args.action_reg,
                tau=args.tau,
                margin=args.margin,
                rollout_scale=args.rollout_scale,
            )
            pred_state = predict_branch(
                state_np,
                refined_actions,
                model,
                stats,
                device,
                rollout_scale=args.rollout_scale,
            )
            pred_xy = pred_state[:, :2]
            dist = signed_distance_xy(pred_xy, box)
            predictions[str(value)] = {
                "eef_xy": pred_xy.astype(float).tolist(),
                "min_signed_distance_m": float(np.min(dist)),
                "inside_count": int(np.sum(dist < 0.0)),
                "mean_action_delta_l2": float(np.mean(np.linalg.norm(refined_actions - selected_chunk, axis=-1))),
                "guidance_scale": float(guidance_scale),
                "step_size": float(step_size),
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

    stem = (
        args.output_dir
        / f"{rollout_dir.parent.parent.name}_{rollout_dir.name}_safety_branches_{args.sweep_param}"
    )
    plot_branches(rollout_dir, eef_xy, obj_xy, branches, sweep_values, args.sweep_param, box, stem)

    true_dist = signed_distance_xy(eef_xy, box)
    summary = {
        "run_dir": str(args.run_dir),
        "rollout_dir": str(rollout_dir),
        "dynamics_ckpt": str(dynamics_ckpt),
        "box": {"x_min": box[0], "x_max": box[1], "y_min": box[2], "y_max": box[3], "margin": args.margin},
        "guidance": {
            "sweep_param": args.sweep_param,
            "sweep_values": sweep_values,
            "guidance_scale": args.guidance_scale,
            "gradient_steps": args.gradient_steps,
            "step_size": args.step_size,
            "action_reg": args.action_reg,
            "tau": args.tau,
            "rollout_scale": args.rollout_scale,
        },
        "true_path": {
            "n_points": int(len(eef_xy)),
            "inside_count": int(np.sum(true_dist < 0.0)),
            "min_signed_distance_m": float(np.min(true_dist)),
        },
        "branches": branches,
        "aggregate_by_scale": {},
        "plots": {"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf"))},
    }
    for value in sweep_values:
        key = str(value)
        mins = np.asarray([branch["predictions"][key]["min_signed_distance_m"] for branch in branches], dtype=np.float64)
        inside_counts = np.asarray([branch["predictions"][key]["inside_count"] for branch in branches], dtype=np.int64)
        deltas = np.asarray([branch["predictions"][key]["mean_action_delta_l2"] for branch in branches], dtype=np.float64)
        summary["aggregate_by_scale"][key] = {
            "chunks_with_predicted_violation": int(np.sum(inside_counts > 0)),
            "total_predicted_inside_points": int(np.sum(inside_counts)),
            "min_signed_distance_m": float(np.min(mins)),
            "median_min_signed_distance_m": float(np.median(mins)),
            "mean_action_delta_l2": float(np.mean(deltas)),
        }
    summary_path = stem.with_name(stem.name + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"rollout: {rollout_dir}")
    print(f"dynamics: {dynamics_ckpt}")
    print(f"wrote {stem.with_suffix('.png')}")
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {summary_path}")
    print(
        "true path: "
        f"inside={summary['true_path']['inside_count']}, "
        f"min_dist={1000.0 * summary['true_path']['min_signed_distance_m']:.1f}mm"
    )
    label = "scale" if args.sweep_param == "guidance_scale" else args.sweep_param
    for value in sweep_values:
        row = summary["aggregate_by_scale"][str(value)]
        print(
            f"{label}={value:g}: chunks_with_violation={row['chunks_with_predicted_violation']}/{len(branches)}, "
            f"inside_pred_points={row['total_predicted_inside_points']}, "
            f"min_dist={1000.0 * row['min_signed_distance_m']:.1f}mm, "
            f"mean_action_delta_l2={row['mean_action_delta_l2']:.5f}"
        )


if __name__ == "__main__":
    main()
