#!/usr/bin/env python3
"""Offline proof-of-concept: guide Cheez-It object tilt via dynamics-model rzz.

The objective is written on rzz = world_z dot object_local_z = cos(tilt), while
the plots show the corresponding human-readable tilt angle in degrees.
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
DEFAULT_STEP_SIZES = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)


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
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "rollout_start":
        return event.get("obs")
    if event.get("type") in {"decision", "chunk_sample"}:
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


def torch_object_rzz_from_state(states: torch.Tensor) -> torch.Tensor:
    rot6d = states[..., 13:19]
    first = rot6d[..., :3] / (rot6d[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
    second = rot6d[..., 3:] - (first * rot6d[..., 3:]).sum(-1, keepdim=True) * first
    second = second / (second.norm(dim=-1, keepdim=True) + 1e-8)
    third = torch.cross(first, second, dim=-1)
    return third[..., 2]


def np_object_rzz_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    return rot6d_to_matrix(np.asarray(rot6d, dtype=np.float32))[..., 2, 2]


def angle_deg_from_rzz(rzz: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(rzz, torch.Tensor):
        values = rzz.detach().cpu().numpy()
    else:
        values = np.asarray(rzz)
    return np.degrees(np.arccos(np.clip(values, -1.0, 1.0)))


def find_first_grasp_decision(events: list[dict[str, Any]]) -> int:
    for event in events:
        if event.get("type") != "target_reached":
            continue
        for label_event in event.get("label_events", []) or []:
            if label_event.get("label_name") == "can_grabbed" and int(label_event.get("to", 0)) == 1:
                return int(label_event.get("decision_idx", event.get("decision_idx", 0)))
    return 0


def true_series(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    decisions = []
    rzz = []
    label_events = []
    for event in events:
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        if event.get("type") not in {"rollout_start", "target_reached"}:
            continue
        obs = obs_from_event(event)
        if obs is None:
            continue
        decisions.append(int(event.get("decision_idx", 0)))
        rzz.append(float(np_object_rzz_from_rot6d(np.asarray(obs["cheezit_rot6d"], dtype=np.float32))))
    return np.asarray(decisions, dtype=np.int32), np.asarray(rzz, dtype=np.float32), label_events


def selected_chunk(event: dict[str, Any]) -> np.ndarray | None:
    if event.get("selected_chunk") is not None:
        return np.asarray(event["selected_chunk"], dtype=np.float32)
    candidates = event.get("candidate_action_chunks")
    selected = event.get("selected_candidate")
    if candidates is None or selected is None:
        return None
    return np.asarray(candidates[int(selected)], dtype=np.float32)


def optimize_chunk_for_rzz(
    state_np: np.ndarray,
    actions_np: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    target_rzz: float,
    step_size: float,
    gradient_steps: int,
    action_reg: float,
    rollout_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    original = torch.as_tensor(actions_np[None], device=device, dtype=torch.float32)
    if step_size <= 0.0 or gradient_steps <= 0:
        with torch.no_grad():
            states = dynamics_rollout(
                torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0),
                original,
                model,
                stats,
                rollout_scale=rollout_scale,
            )
            rzz = torch_object_rzz_from_state(states)
            loss = torch.mean((rzz - target_rzz) ** 2)
        return np.asarray(actions_np, dtype=np.float32), {
            "loss": float(loss.detach().cpu()),
            "mean_rzz": float(rzz.mean().detach().cpu()),
            "mean_action_delta": 0.0,
            "max_action_delta": 0.0,
        }

    state0 = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
    actions = original.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([actions], lr=step_size)
    for _ in range(int(gradient_steps)):
        opt.zero_grad(set_to_none=True)
        states = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)
        rzz = torch_object_rzz_from_state(states)
        target_loss = torch.mean((rzz - target_rzz) ** 2)
        reg_loss = torch.mean((actions - original) ** 2)
        loss = target_loss + float(action_reg) * reg_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            actions.clamp_(-1.0, 1.0)

    with torch.no_grad():
        states = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)
        rzz = torch_object_rzz_from_state(states)
        target_loss = torch.mean((rzz - target_rzz) ** 2)
        delta = torch.abs(actions - original)
    return actions[0].detach().cpu().numpy().astype(np.float32), {
        "loss": float(target_loss.detach().cpu()),
        "mean_rzz": float(rzz.mean().detach().cpu()),
        "mean_action_delta": float(delta.mean().detach().cpu()),
        "max_action_delta": float(delta.max().detach().cpu()),
    }


def collect_guided_branches(
    events: list[dict[str, Any]],
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    pickup_decision: int,
    window_steps: int,
    target_rzz: float,
    step_sizes: list[float],
    gradient_steps: int,
    action_reg: float,
    rollout_scale: float,
) -> tuple[dict[float, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_step = {float(step): [] for step in step_sizes}
    rows = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        start_decision = int(event.get("decision_idx_before", event.get("decision_idx", 0)))
        rel_start = start_decision - pickup_decision
        if rel_start < 0 or rel_start > window_steps:
            continue
        obs = event.get("obs")
        actions_np = selected_chunk(event)
        if obs is None or actions_np is None:
            continue
        state_np = obs_to_state(obs)
        state0 = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
        for step_size in step_sizes:
            guided_actions, opt_stats = optimize_chunk_for_rzz(
                state_np,
                actions_np,
                model,
                stats,
                device,
                target_rzz=target_rzz,
                step_size=float(step_size),
                gradient_steps=gradient_steps,
                action_reg=action_reg,
                rollout_scale=rollout_scale,
            )
            with torch.no_grad():
                states = dynamics_rollout(
                    state0,
                    torch.as_tensor(guided_actions[None], device=device, dtype=torch.float32),
                    model,
                    stats,
                    rollout_scale=rollout_scale,
                )
                rzz = torch_object_rzz_from_state(states)[0].detach().cpu().numpy()
            angle = angle_deg_from_rzz(rzz)
            branch = {
                "start_decision": start_decision,
                "rel_start": rel_start,
                "x": rel_start + np.arange(1, len(angle) + 1, dtype=np.int32),
                "angle_deg": angle,
                "rzz": rzz,
            }
            by_step[float(step_size)].append(branch)
            rows.append(
                {
                    "chunk_idx": int(event.get("chunk_idx", len(rows))),
                    "start_decision": start_decision,
                    "rel_start": rel_start,
                    "step_size": float(step_size),
                    "mean_abs_angle_err_to_target_deg": float(np.mean(np.abs(angle - angle_deg_from_rzz(np.array([target_rzz]))[0]))),
                    "max_abs_angle_err_to_target_deg": float(np.max(np.abs(angle - angle_deg_from_rzz(np.array([target_rzz]))[0]))),
                    "mean_rzz": float(np.mean(rzz)),
                    "mean_abs_rzz_err": float(np.mean(np.abs(rzz - target_rzz))),
                    **opt_stats,
                }
            )
    return by_step, rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for step_size in sorted({float(row["step_size"]) for row in rows}):
        vals = [row for row in rows if float(row["step_size"]) == step_size]
        summary[str(step_size)] = {
            "n_chunks": int(len(vals)),
            "mean_abs_angle_err_to_target_deg": float(np.mean([v["mean_abs_angle_err_to_target_deg"] for v in vals])),
            "mean_abs_rzz_err": float(np.mean([v["mean_abs_rzz_err"] for v in vals])),
            "mean_action_delta": float(np.mean([v["mean_action_delta"] for v in vals])),
            "max_action_delta": float(np.max([v["max_action_delta"] for v in vals])),
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
    true_angle: np.ndarray,
    label_events: list[dict[str, Any]],
    by_step: dict[float, list[dict[str, Any]]],
    pickup_decision: int,
    *,
    target_angle_deg: float,
    window_steps: int,
) -> None:
    step_sizes = sorted(by_step)
    n_cols = 2
    n_rows = int(np.ceil(len(step_sizes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    colors = {
        0.0: "#ff8c00",
        1e-5: "#9467bd",
        3e-5: "#1f77b4",
        1e-4: "#2ca02c",
        3e-4: "#d62728",
        1e-3: "#17becf",
    }
    for ax, step_size in zip(axes_arr, step_sizes):
        branches = by_step[step_size]
        for idx, branch in enumerate(branches):
            start_angle = float(np.interp(branch["rel_start"], rel_decisions, true_angle))
            x = np.concatenate([[branch["rel_start"]], branch["x"]])
            y = np.concatenate([[start_angle], branch["angle_deg"]])
            ax.plot(
                x,
                y,
                color=colors.get(step_size, "#d000ff"),
                alpha=0.45,
                linewidth=1.15,
                label="WM branch" if idx == 0 else None,
            )
        ax.plot(rel_decisions, true_angle, color="#111111", linewidth=1.6, label="true")
        ax.axhline(target_angle_deg, color="#d62728", linestyle="--", linewidth=1.0, label="target 20deg")
        for event in label_events:
            rel = int(event.get("decision_idx", 0)) - pickup_decision
            if rel < 0 or rel > window_steps:
                continue
            angle = float(np.interp(rel, rel_decisions, true_angle))
            ax.scatter(rel, angle, s=45, marker="*", color="#111111", zorder=5)
            ax.text(rel, angle, f" {event.get('label_name')} {event.get('from')}->{event.get('to')}", fontsize=7, va="center")
        title = "unguided" if step_size == 0.0 else f"adam step_size={step_size:g}"
        ax.set_title(title)
        ax.set_xlim(0, window_steps)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    for ax in axes_arr[len(step_sizes) :]:
        ax.axis("off")
    for ax in axes_arr[::n_cols]:
        ax.set_ylabel("Cheez-It tilt [deg]")
    for ax in axes_arr[-n_cols:]:
        ax.set_xlabel("decisions after first grasp")
    fig.suptitle("Offline rzz guidance toward object tilt = 20deg, first 50 decisions after pickup", y=0.995)
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
    parser.add_argument("--target-angle-deg", type=float, default=20.0)
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument("--step-sizes", type=float, nargs="+", default=list(DEFAULT_STEP_SIZES))
    parser.add_argument("--gradient-steps", type=int, default=20)
    parser.add_argument("--action-reg", type=float, default=0.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="release_regrasp_rollout000_rzz_angle20_step_size_sweep")
    args = parser.parse_args()

    configure_matplotlib()
    events = load_events(args.events)
    pickup_decision = find_first_grasp_decision(events)
    decisions, true_rzz, label_events = true_series(events)
    rel_decisions = decisions - pickup_decision
    mask = (rel_decisions >= 0) & (rel_decisions <= args.window_steps)
    if not np.any(mask):
        raise ValueError("No true observations in requested post-grasp window")
    true_angle = angle_deg_from_rzz(true_rzz)

    device = torch.device(args.device)
    model, stats_np, _ = load_dynamics_model_for_eval(args.dynamics_ckpt, device=device)
    stats = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in stats_np.items()}
    target_rzz = float(np.cos(np.deg2rad(args.target_angle_deg)))

    by_step, rows = collect_guided_branches(
        events,
        model,
        stats,
        device,
        pickup_decision=pickup_decision,
        window_steps=args.window_steps,
        target_rzz=target_rzz,
        step_sizes=[float(v) for v in args.step_sizes],
        gradient_steps=args.gradient_steps,
        action_reg=args.action_reg,
        rollout_scale=args.rollout_scale,
    )
    out_png = args.output_dir / f"{args.name}.png"
    write_plot(
        out_png,
        rel_decisions[mask],
        true_angle[mask],
        label_events,
        by_step,
        pickup_decision,
        target_angle_deg=args.target_angle_deg,
        window_steps=args.window_steps,
    )
    csv_path = args.output_dir / f"{args.name}_chunks.csv"
    write_rows_csv(csv_path, rows)
    summary = {
        "events": str(args.events),
        "dynamics_ckpt": str(args.dynamics_ckpt),
        "pickup_decision": int(pickup_decision),
        "window_steps": int(args.window_steps),
        "target_angle_deg": float(args.target_angle_deg),
        "target_rzz": target_rzz,
        "step_sizes": [float(v) for v in args.step_sizes],
        "gradient_steps": int(args.gradient_steps),
        "action_reg": float(args.action_reg),
        "rollout_scale": float(args.rollout_scale),
        "n_chunks_in_window": int(len(next(iter(by_step.values()), []))),
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
