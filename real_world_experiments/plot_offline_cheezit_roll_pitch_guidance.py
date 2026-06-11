#!/usr/bin/env python3
"""Offline guidance proof-of-concept for suppressing object roll/pitch.

The allowed coordinate is relative z/yaw. The optimizer penalizes relative
x/roll and y/pitch of the Cheez-It object, measured from the pickup pose, over
the dynamics-model rollout of each selected action chunk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world_experiments.real_world_data import rot6d_to_matrix
from real_world_experiments.train_dynamics_world_model import load_dynamics_model_for_eval


DEFAULT_EVENTS = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_release_regrasp_right_left_cycle3_epoch160/rollouts/rollout_000/events.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"
DEFAULT_DYNAMICS = (
    REPO_ROOT
    / "outputs/real_world/dynamics_world_model/"
    / "hd128_depth2_lr0.001_epochs120_2026-05-13_17-53-04/best_model.pt"
)
DEFAULT_STEP_SIZES = (0.0, 0.0003, 0.001, 0.003, 0.005, 0.01)
AXIS_NAMES = ("x/roll", "y/pitch", "z/yaw")
TRUE_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
STEP_COLORS = {
    0.0: "#ff8c00",
    0.0003: "#9467bd",
    0.001: "#1f77b4",
    0.003: "#2ca02c",
    0.005: "#d62728",
    0.01: "#17becf",
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
            "legend.fontsize": 7,
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
    if event.get("type") in {"chunk_sample", "decision"}:
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


def selected_chunk(event: dict[str, Any]) -> np.ndarray | None:
    if event.get("selected_chunk") is not None:
        return np.asarray(event["selected_chunk"], dtype=np.float32)
    candidates = event.get("candidate_action_chunks")
    selected = event.get("selected_candidate")
    if candidates is None or selected is None:
        return None
    return np.asarray(candidates[int(selected)], dtype=np.float32)


def first_event_decision(events: list[dict[str, Any]], label_name: str, to_value: int) -> int | None:
    for event in events:
        if event.get("type") != "target_reached":
            continue
        for label_event in event.get("label_events", []) or []:
            if label_event.get("label_name") == label_name and int(label_event.get("to", 0)) == to_value:
                return int(label_event.get("decision_idx", event.get("decision_idx", 0)))
    return None


def relative_euler_np(rot6d: np.ndarray, reference_matrix: np.ndarray) -> np.ndarray:
    matrices = rot6d_to_matrix(np.asarray(rot6d, dtype=np.float32))
    rel = np.einsum("ij,...jk->...ik", reference_matrix.T, matrices)
    euler = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=True)
    return euler.reshape(rel.shape[:-2] + (3,))


def unwrap_euler_series(euler: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(euler), axis=0))


def align_branch_euler_to_start(pred_euler: np.ndarray, true_start: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(pred_euler, dtype=np.float64)
    for axis in range(3):
        seq = np.concatenate([[float(true_start[axis])], np.asarray(pred_euler[:, axis], dtype=np.float64)])
        aligned[:, axis] = np.degrees(np.unwrap(np.radians(seq)))[1:]
    return aligned


def true_series(events: list[dict[str, Any]], pickup_decision: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    decisions = []
    rot6d = []
    labels = []
    label_events = []
    reference_matrix = None
    for event in events:
        if event.get("type") not in {"rollout_start", "target_reached"}:
            continue
        obs = obs_from_event(event)
        if obs is None:
            continue
        decision = int(event.get("decision_idx", 0))
        if decision == pickup_decision:
            reference_matrix = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float32)[None])[0]
        decisions.append(decision)
        rot6d.append(obs["cheezit_rot6d"])
        labels.append(event.get("label") or event.get("current_label") or [0, 0, 0])
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
    if reference_matrix is None:
        raise ValueError(f"Could not find pickup reference at decision {pickup_decision}")
    euler = unwrap_euler_series(relative_euler_np(np.asarray(rot6d, dtype=np.float32), reference_matrix))
    return np.asarray(decisions, dtype=np.int32), euler, np.asarray(labels, dtype=np.int32), label_events, reference_matrix


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


def torch_rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first = rot6d[..., :3] / (rot6d[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
    second = rot6d[..., 3:] - (first * rot6d[..., 3:]).sum(-1, keepdim=True) * first
    second = second / (second.norm(dim=-1, keepdim=True) + 1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def torch_relative_roll_pitch(states: torch.Tensor, reference_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    obj_rot = torch_rot6d_to_matrix(states[..., 13:19])
    rel = torch.einsum("ij,...jk->...ik", reference_matrix.T, obj_rot)
    roll_x = torch.atan2(rel[..., 2, 1], rel[..., 2, 2])
    pitch_y = torch.asin(torch.clamp(-rel[..., 2, 0], -0.9999, 0.9999))
    return roll_x, pitch_y


def optimize_chunk(
    state_np: np.ndarray,
    actions_np: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    reference_matrix: torch.Tensor,
    *,
    step_size: float,
    gradient_steps: int,
    action_reg: float,
    rollout_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    state0 = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
    original = torch.as_tensor(actions_np[None], device=device, dtype=torch.float32)
    if step_size <= 0.0 or gradient_steps <= 0:
        with torch.no_grad():
            states = dynamics_rollout(state0, original, model, stats, rollout_scale=rollout_scale)
            roll, pitch = torch_relative_roll_pitch(states, reference_matrix)
            loss = torch.mean(roll**2 + pitch**2)
        return np.asarray(actions_np, dtype=np.float32), {
            "roll_pitch_loss_rad2": float(loss.detach().cpu()),
            "mean_action_delta": 0.0,
            "max_action_delta": 0.0,
        }

    actions = original.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([actions], lr=float(step_size))
    for _ in range(int(gradient_steps)):
        opt.zero_grad(set_to_none=True)
        states = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)
        roll, pitch = torch_relative_roll_pitch(states, reference_matrix)
        target_loss = torch.mean(roll**2 + pitch**2)
        reg_loss = torch.mean((actions - original) ** 2)
        loss = target_loss + float(action_reg) * reg_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            actions.clamp_(-1.0, 1.0)

    with torch.no_grad():
        states = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)
        roll, pitch = torch_relative_roll_pitch(states, reference_matrix)
        loss = torch.mean(roll**2 + pitch**2)
        delta = torch.abs(actions - original)
    return actions[0].detach().cpu().numpy().astype(np.float32), {
        "roll_pitch_loss_rad2": float(loss.detach().cpu()),
        "mean_action_delta": float(delta.mean().detach().cpu()),
        "max_action_delta": float(delta.max().detach().cpu()),
    }


def collect_guided_branches(
    events: list[dict[str, Any]],
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    reference_matrix_np: np.ndarray,
    *,
    pickup_decision: int,
    window_steps: int,
    step_sizes: list[float],
    gradient_steps: int,
    action_reg: float,
    rollout_scale: float,
) -> tuple[dict[float, list[dict[str, Any]]], list[dict[str, Any]]]:
    reference_matrix = torch.as_tensor(reference_matrix_np, device=device, dtype=torch.float32)
    by_step = {float(step): [] for step in step_sizes}
    rows = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        start = int(event.get("decision_idx_before", event.get("decision_idx", 0)))
        rel_start = start - pickup_decision
        if rel_start < 0 or rel_start > window_steps:
            continue
        obs = event.get("obs")
        actions_np = selected_chunk(event)
        if obs is None or actions_np is None:
            continue
        state_np = obs_to_state(obs)
        for step_size in step_sizes:
            guided_actions, opt_stats = optimize_chunk(
                state_np,
                actions_np,
                model,
                stats,
                device,
                reference_matrix,
                step_size=float(step_size),
                gradient_steps=gradient_steps,
                action_reg=action_reg,
                rollout_scale=rollout_scale,
            )
            with torch.no_grad():
                states = dynamics_rollout(
                    torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0),
                    torch.as_tensor(guided_actions[None], device=device, dtype=torch.float32),
                    model,
                    stats,
                    rollout_scale=rollout_scale,
                )
            pred_rot6d = states[0, :, 13:19].detach().cpu().numpy()
            pred_euler = relative_euler_np(pred_rot6d, reference_matrix_np)
            branch = {
                "chunk_idx": int(event.get("chunk_idx", len(rows))),
                "rel_start": rel_start,
                "x": rel_start + np.arange(1, len(pred_euler) + 1, dtype=np.int32),
                "euler": pred_euler,
            }
            by_step[float(step_size)].append(branch)
            rows.append(
                {
                    "chunk_idx": int(event.get("chunk_idx", len(rows))),
                    "rel_start": int(rel_start),
                    "step_size": float(step_size),
                    "mean_abs_roll_deg": float(np.mean(np.abs(pred_euler[:, 0]))),
                    "mean_abs_pitch_deg": float(np.mean(np.abs(pred_euler[:, 1]))),
                    "mean_abs_yaw_deg": float(np.mean(np.abs(pred_euler[:, 2]))),
                    **opt_stats,
                }
            )
    return by_step, rows


def label_intervals(rel_decisions: np.ndarray, labels: np.ndarray, label_idx: int) -> list[tuple[int, int]]:
    intervals = []
    active_start = None
    for idx, rel_decision in enumerate(rel_decisions):
        active = bool(labels[idx, label_idx] > 0)
        if active and active_start is None:
            active_start = int(rel_decision)
        if not active and active_start is not None:
            intervals.append((active_start, int(rel_decisions[max(0, idx - 1)])))
            active_start = None
    if active_start is not None:
        intervals.append((active_start, int(rel_decisions[-1])))
    return intervals


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for step_size in sorted({float(row["step_size"]) for row in rows}):
        vals = [row for row in rows if float(row["step_size"]) == step_size]
        summary[str(step_size)] = {
            "n_chunks": int(len(vals)),
            "mean_abs_roll_deg": float(np.mean([v["mean_abs_roll_deg"] for v in vals])),
            "mean_abs_pitch_deg": float(np.mean([v["mean_abs_pitch_deg"] for v in vals])),
            "mean_abs_yaw_deg": float(np.mean([v["mean_abs_yaw_deg"] for v in vals])),
            "mean_action_delta": float(np.mean([v["mean_action_delta"] for v in vals])),
            "max_action_delta": float(np.max([v["max_action_delta"] for v in vals])),
            "mean_roll_pitch_loss_rad2": float(np.mean([v["roll_pitch_loss_rad2"] for v in vals])),
        }
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: Path,
    rel_decisions: np.ndarray,
    true_euler: np.ndarray,
    labels: np.ndarray,
    by_step: dict[float, list[dict[str, Any]]],
    *,
    window_steps: int,
) -> None:
    step_sizes = sorted(by_step)
    n_rows = 3
    n_cols = len(step_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.45 * n_cols, 8.8), sharex=True, sharey="row")
    axes = np.asarray(axes)
    right_intervals = label_intervals(rel_decisions, labels, 1)
    left_intervals = label_intervals(rel_decisions, labels, 2)
    for col, step_size in enumerate(step_sizes):
        for axis in range(3):
            ax = axes[axis, col]
            for idx, (start, end) in enumerate(right_intervals):
                ax.axvspan(start, end, color="#ffcc00", alpha=0.16, label="right" if idx == 0 and axis == 0 else None)
            for idx, (start, end) in enumerate(left_intervals):
                ax.axvspan(start, end, color="#8ecae6", alpha=0.16, label="left" if idx == 0 and axis == 0 else None)
            wrote_branch = False
            for branch in by_step[step_size]:
                true_start = np.asarray([np.interp(branch["rel_start"], rel_decisions, true_euler[:, j]) for j in range(3)])
                pred = align_branch_euler_to_start(branch["euler"], true_start)
                x = np.concatenate([[branch["rel_start"]], branch["x"]])
                y = np.concatenate([[true_start[axis]], pred[:, axis]])
                ax.plot(
                    x,
                    y,
                    color=STEP_COLORS.get(step_size, "#d000ff"),
                    alpha=0.42,
                    linewidth=0.95,
                    label="WM optimized chunk" if not wrote_branch and axis == 0 else None,
                )
                wrote_branch = True
            ax.plot(rel_decisions, true_euler[:, axis], color=TRUE_COLORS[axis], linewidth=1.55, label="true" if axis == 0 else None)
            ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.8)
            ax.set_xlim(0, window_steps)
            ax.grid(True, alpha=0.22)
            if col == 0:
                ax.set_ylabel(f"{AXIS_NAMES[axis]} [deg]")
            if axis == 0:
                ax.set_title("unguided" if step_size == 0.0 else f"adam lr={step_size:g}")
            if axis == n_rows - 1:
                ax.set_xlabel("decisions after grasp")
            if col == n_cols - 1 and axis == 0:
                ax.legend(loc="upper right")
    fig.suptitle("Offline guidance: penalize relative x/roll and y/pitch; allow z/yaw", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--dynamics-ckpt", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-steps", type=int, default=90)
    parser.add_argument("--step-sizes", type=float, nargs="+", default=list(DEFAULT_STEP_SIZES))
    parser.add_argument("--gradient-steps", type=int, default=20)
    parser.add_argument("--action-reg", type=float, default=0.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="release_regrasp_rollout000_roll_pitch_zero_guidance_sweep")
    args = parser.parse_args()

    configure_matplotlib()
    events = load_events(args.events)
    pickup_decision = first_event_decision(events, "can_grabbed", 1)
    if pickup_decision is None:
        raise ValueError("Could not find can_grabbed 0->1 event")
    decisions, true_euler, labels, _, reference_matrix = true_series(events, pickup_decision)
    rel_decisions = decisions - pickup_decision
    mask = (rel_decisions >= 0) & (rel_decisions <= args.window_steps)

    device = torch.device(args.device)
    model, stats_np, _ = load_dynamics_model_for_eval(args.dynamics_ckpt, device=device)
    stats = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in stats_np.items()}

    by_step, rows = collect_guided_branches(
        events,
        model,
        stats,
        device,
        reference_matrix,
        pickup_decision=pickup_decision,
        window_steps=args.window_steps,
        step_sizes=[float(v) for v in args.step_sizes],
        gradient_steps=args.gradient_steps,
        action_reg=args.action_reg,
        rollout_scale=args.rollout_scale,
    )

    out_png = args.output_dir / f"{args.name}.png"
    write_plot(
        out_png,
        rel_decisions[mask],
        true_euler[mask],
        labels[mask],
        by_step,
        window_steps=args.window_steps,
    )
    csv_path = args.output_dir / f"{args.name}_chunks.csv"
    write_rows_csv(csv_path, rows)
    summary = {
        "events": str(args.events),
        "dynamics_ckpt": str(args.dynamics_ckpt),
        "pickup_decision": int(pickup_decision),
        "window_steps": int(args.window_steps),
        "step_sizes": [float(v) for v in args.step_sizes],
        "gradient_steps": int(args.gradient_steps),
        "action_reg": float(args.action_reg),
        "rollout_scale": float(args.rollout_scale),
        "n_chunks": int(len(next(iter(by_step.values()), []))),
        "objective": "mean(relative_roll_x_rad^2 + relative_pitch_y_rad^2) over dynamics rollout",
        "by_step_size": summarize_rows(rows),
        "plot_png": str(out_png),
        "plot_pdf": str(out_png.with_suffix(".pdf")),
        "chunk_csv": str(csv_path),
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
