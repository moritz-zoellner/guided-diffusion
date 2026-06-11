#!/usr/bin/env python3
"""Offline sweep for real-world rzz-hold Adam step size.

This mirrors the runtime object_rzz_hold_guidance path: optimize selected
chunks with the dynamics model, keep xyz/gripper actions unchanged, and apply
the rzz objective only for predicted steps before the EEF-x gate.
"""

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
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world_experiments.train_dynamics_world_model import load_dynamics_model_for_eval


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"
DEFAULT_DYNAMICS = (
    REPO_ROOT
    / "outputs/real_world/dynamics_world_model/"
    / "hd128_depth2_lr0.001_epochs120_2026-05-13_17-53-04/best_model.pt"
)
DEFAULT_EVENTS = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_left_rzz1_hold_xminus063_epoch160_n1/rollouts/rollout_000/events.jsonl"
)


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def torch_rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first = rot6d[..., :3] / (rot6d[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
    second = rot6d[..., 3:] - (first * rot6d[..., 3:]).sum(-1, keepdim=True) * first
    second = second / (second.norm(dim=-1, keepdim=True) + 1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def rzz_from_states(states: torch.Tensor) -> torch.Tensor:
    return torch_rot6d_to_matrix(states[..., 13:19])[..., 2, 2]


def angle_deg_from_rzz(rzz: np.ndarray | torch.Tensor) -> np.ndarray:
    values = rzz.detach().cpu().numpy() if torch.is_tensor(rzz) else np.asarray(rzz)
    return np.degrees(np.arccos(np.clip(values, -1.0, 1.0)))


def selected_chunk(event: dict[str, Any]) -> np.ndarray | None:
    chunk = event.get("selected_chunk")
    if chunk is not None:
        return np.asarray(chunk, dtype=np.float32)
    candidates = event.get("candidate_action_chunks")
    selected = event.get("selected_candidate")
    if candidates is None or selected is None:
        return None
    return np.asarray(candidates[int(selected)], dtype=np.float32)


def first_grasp_decision(events: list[dict[str, Any]]) -> int:
    for event in events:
        if event.get("type") != "target_reached":
            continue
        for label_event in event.get("label_events", []) or []:
            if label_event.get("label_name") == "can_grabbed" and int(label_event.get("to", 0)) == 1:
                return int(label_event.get("decision_idx", event.get("decision_idx", 0)))
    return 0


def pour_decision(events: list[dict[str, Any]]) -> int | None:
    for event in events:
        if event.get("type") != "target_reached":
            continue
        for label_event in event.get("label_events", []) or []:
            if label_event.get("label_name") == "pouring_left" and int(label_event.get("to", 0)) == 1:
                return int(label_event.get("decision_idx", event.get("decision_idx", 0)))
    return None


def true_trajectory(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    rows = []
    for event in events:
        if event.get("type") == "rollout_start":
            obs = event.get("obs")
        elif event.get("type") == "target_reached":
            obs = event.get("reached_obs")
        else:
            continue
        if obs is None:
            continue
        state = obs_to_state(obs)
        rows.append(
            {
                "decision": int(event.get("decision_idx", 0)),
                "eef_x": state[0],
                "eef_y": state[1],
                "rzz": float(torch_rot6d_to_matrix(torch.as_tensor(state[13:19][None]))[0, 2, 2]),
            }
        )
    return {
        "decision": np.asarray([row["decision"] for row in rows], dtype=np.int32),
        "eef_x": np.asarray([row["eef_x"] for row in rows], dtype=np.float32),
        "eef_y": np.asarray([row["eef_y"] for row in rows], dtype=np.float32),
        "rzz": np.asarray([row["rzz"] for row in rows], dtype=np.float32),
    }


def active_rzz_rollout(
    state0: torch.Tensor,
    actions: torch.Tensor,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    *,
    gate_x: float,
    rollout_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = dynamics_rollout(state0, actions, model, stats, rollout_scale=rollout_scale)
    rzz = rzz_from_states(states)
    eef_x = states[..., 0]
    mask = eef_x > float(gate_x)
    return rzz, eef_x, mask


def optimize_chunk(
    state_np: np.ndarray,
    actions_np: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    step_size: float,
    gradient_steps: int,
    target_rzz: float,
    gate_x: float,
    action_reg: float,
    rollout_scale: float,
    rotation_only: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    state0 = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
    original = torch.as_tensor(actions_np[None], device=device, dtype=torch.float32)
    target = torch.as_tensor(float(target_rzz), device=device, dtype=torch.float32)

    with torch.no_grad():
        pre_rzz, pre_eef_x, pre_mask = active_rzz_rollout(
            state0, original, model, stats, gate_x=gate_x, rollout_scale=rollout_scale
        )
    active_steps = int(pre_mask.sum().detach().cpu())
    if step_size <= 0.0 or gradient_steps <= 0 or active_steps <= 0:
        return actions_np.astype(np.float32), summarize_prediction(
            pre_rzz, pre_eef_x, pre_mask, target, original, original, active_steps
        )

    actions = original.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([actions], lr=float(step_size))
    for _ in range(int(gradient_steps)):
        opt.zero_grad(set_to_none=True)
        rzz, _, mask = active_rzz_rollout(
            state0, actions, model, stats, gate_x=gate_x, rollout_scale=rollout_scale
        )
        mask_f = mask.to(dtype=actions.dtype)
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        target_loss = torch.sum(((rzz - target) ** 2) * mask_f, dim=-1) / denom
        reg_loss = torch.mean((actions - original) ** 2, dim=(1, 2))
        loss = target_loss + float(action_reg) * reg_loss
        loss.mean().backward()
        opt.step()
        with torch.no_grad():
            actions.clamp_(-1.0, 1.0)
            if rotation_only:
                actions[..., :3] = original[..., :3]
                if actions.shape[-1] > 6:
                    actions[..., 6:] = original[..., 6:]

    with torch.no_grad():
        post_rzz, post_eef_x, post_mask = active_rzz_rollout(
            state0, actions, model, stats, gate_x=gate_x, rollout_scale=rollout_scale
        )
    stats_row = summarize_prediction(post_rzz, post_eef_x, post_mask, target, original, actions, active_steps)
    return actions[0].detach().cpu().numpy().astype(np.float32), stats_row


def summarize_prediction(
    rzz: torch.Tensor,
    eef_x: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    original: torch.Tensor,
    actions: torch.Tensor,
    active_steps: int,
) -> dict[str, float]:
    mask_f = mask.to(dtype=rzz.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    rzz_err = torch.abs(rzz - target)
    action_delta = torch.abs(actions - original)
    rot_delta = torch.linalg.norm(actions[..., 3:6] - original[..., 3:6], dim=-1)
    return {
        "active_steps": float(active_steps),
        "mean_abs_target_rzz_error": float((rzz_err * mask_f).sum().detach().cpu() / denom.detach().cpu()),
        "max_abs_target_rzz_error": (
            float(torch.max(torch.where(mask, rzz_err, torch.zeros_like(rzz_err))).detach().cpu())
            if int(mask.sum().detach().cpu()) > 0
            else float("nan")
        ),
        "mean_angle_deg": float(np.nanmean(angle_deg_from_rzz(torch.where(mask, rzz, torch.full_like(rzz, np.nan))))),
        "max_angle_deg": float(np.nanmax(angle_deg_from_rzz(torch.where(mask, rzz, torch.full_like(rzz, np.nan))))),
        "mean_rzz": float(((rzz * mask_f).sum() / denom).detach().cpu()),
        "min_pred_eef_x": float(torch.min(eef_x).detach().cpu()),
        "max_pred_eef_x": float(torch.max(eef_x).detach().cpu()),
        "mean_action_delta": float(action_delta.mean().detach().cpu()),
        "max_action_delta": float(action_delta.max().detach().cpu()),
        "mean_rotation_action_delta_l2": float(rot_delta.mean().detach().cpu()),
        "max_rotation_action_delta_l2": float(rot_delta.max().detach().cpu()),
    }


def collect_chunks(
    events: list[dict[str, Any]],
    *,
    grab_decision: int,
    window_steps: int,
    gate_x: float,
) -> list[dict[str, Any]]:
    chunks = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        decision = int(event.get("decision_idx_before", event.get("decision_idx", 0)))
        rel = decision - int(grab_decision)
        if rel < 0 or rel > int(window_steps):
            continue
        obs = event.get("obs")
        chunk = selected_chunk(event)
        if obs is None or chunk is None:
            continue
        eef_x = float(np.asarray(obs["eef_pos"], dtype=np.float32)[0])
        if eef_x <= float(gate_x):
            continue
        chunks.append(
            {
                "decision": decision,
                "relative_decision": rel,
                "state": obs_to_state(obs),
                "actions": chunk,
            }
        )
    return chunks


def run_sweep(
    chunks: list[dict[str, Any]],
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    step_sizes: list[float],
    gradient_steps: int,
    target_rzz: float,
    gate_x: float,
    action_reg: float,
    rollout_scale: float,
    rotation_only: bool,
) -> tuple[list[dict[str, Any]], dict[float, list[dict[str, Any]]]]:
    rows = []
    branches: dict[float, list[dict[str, Any]]] = {float(v): [] for v in step_sizes}
    for chunk_idx, chunk in enumerate(chunks):
        for step_size in step_sizes:
            actions, stat = optimize_chunk(
                chunk["state"],
                chunk["actions"],
                model,
                stats,
                device,
                step_size=float(step_size),
                gradient_steps=gradient_steps,
                target_rzz=target_rzz,
                gate_x=gate_x,
                action_reg=action_reg,
                rollout_scale=rollout_scale,
                rotation_only=rotation_only,
            )
            state0 = torch.as_tensor(chunk["state"], device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_states = dynamics_rollout(
                    state0,
                    torch.as_tensor(actions[None], device=device, dtype=torch.float32),
                    model,
                    stats,
                    rollout_scale=rollout_scale,
                )
                rzz = rzz_from_states(pred_states)[0].detach().cpu().numpy()
                eef_x = pred_states[0, :, 0].detach().cpu().numpy()
            angle = angle_deg_from_rzz(rzz)
            branches[float(step_size)].append(
                {
                    "chunk_idx": chunk_idx,
                    "relative_decision": int(chunk["relative_decision"]),
                    "x": int(chunk["relative_decision"]) + np.arange(1, len(angle) + 1),
                    "angle_deg": angle,
                    "eef_x": eef_x,
                }
            )
            rows.append(
                {
                    "chunk_idx": chunk_idx,
                    "decision": int(chunk["decision"]),
                    "relative_decision": int(chunk["relative_decision"]),
                    "step_size": float(step_size),
                    **stat,
                }
            )
    return rows, branches


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for step_size in sorted({float(row["step_size"]) for row in rows}):
        vals = [row for row in rows if float(row["step_size"]) == step_size]
        summary[str(step_size)] = {
            "n_chunks": int(len(vals)),
            "mean_abs_target_rzz_error": float(np.nanmean([v["mean_abs_target_rzz_error"] for v in vals])),
            "max_abs_target_rzz_error": float(np.nanmax([v["max_abs_target_rzz_error"] for v in vals])),
            "mean_angle_deg": float(np.nanmean([v["mean_angle_deg"] for v in vals])),
            "max_angle_deg": float(np.nanmax([v["max_angle_deg"] for v in vals])),
            "mean_rotation_action_delta_l2": float(np.nanmean([v["mean_rotation_action_delta_l2"] for v in vals])),
            "max_rotation_action_delta_l2": float(np.nanmax([v["max_rotation_action_delta_l2"] for v in vals])),
            "max_action_delta": float(np.nanmax([v["max_action_delta"] for v in vals])),
        }
    return summary


def plot_sweep(
    out_path: Path,
    traj: dict[str, np.ndarray],
    branches: dict[float, list[dict[str, Any]]],
    *,
    grab_decision: int,
    pour_decision_value: int | None,
    gate_x: float,
    ylimit: tuple[float, float],
) -> None:
    rel_true = traj["decision"] - int(grab_decision)
    true_angle = angle_deg_from_rzz(traj["rzz"])
    step_sizes = sorted(branches)
    n_cols = 3
    n_rows = int(np.ceil(len(step_sizes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.15 * n_rows), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    colors = {
        0.0: "#8c8c8c",
        0.0005: "#7b61ff",
        0.001: "#1f77b4",
        0.0015: "#2ca02c",
        0.002: "#ff7f0e",
        0.003: "#d62728",
        0.005: "#8c1d40",
    }
    for ax, step_size in zip(axes_arr, step_sizes):
        color = colors.get(float(step_size), "#9467bd")
        for branch in branches[step_size]:
            ax.plot(branch["x"], branch["angle_deg"], color=color, alpha=0.36, linewidth=1.05)
            gate_hits = np.where(branch["eef_x"] <= float(gate_x))[0]
            if gate_hits.size:
                idx = int(gate_hits[0])
                ax.scatter(branch["x"][idx], branch["angle_deg"][idx], s=18, color=color, zorder=4)
        ax.plot(rel_true, true_angle, color="#111111", linewidth=1.7, label="executed")
        if pour_decision_value is not None:
            ax.axvline(int(pour_decision_value) - int(grab_decision), color="#111111", linestyle="--", linewidth=1.0)
        ax.set_title("unguided" if step_size == 0 else f"step_size={step_size:g}")
        ax.set_ylim(*ylimit)
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[len(step_sizes) :]:
        ax.axis("off")
    for ax in axes_arr[::n_cols]:
        ax.set_ylabel("predicted upright angle [deg]")
    for ax in axes_arr[-n_cols:]:
        ax.set_xlabel("decisions after grasp")
    fig.suptitle(f"Offline rzz=1 hold sweep; dots mark predicted x <= {gate_x:.2f}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--events", action="append", type=Path, default=None)
    parser.add_argument("--events-glob", action="append", default=None)
    parser.add_argument("--dynamics-ckpt", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="rzz1_hold_xminus063_step_size_sweep")
    parser.add_argument("--step-sizes", nargs="+", type=float, default=[0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005])
    parser.add_argument("--gradient-steps", type=int, default=20)
    parser.add_argument("--target-rzz", type=float, default=1.0)
    parser.add_argument("--gate-x", type=float, default=-0.63)
    parser.add_argument("--window-steps", type=int, default=35)
    parser.add_argument("--action-reg", type=float, default=0.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--rotation-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--angle-ylim", nargs=2, type=float, default=[0.0, 25.0])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    event_paths = list(args.events or [])
    for pattern in args.events_glob or []:
        event_paths.extend(sorted(Path(".").glob(pattern)))
    if not event_paths:
        event_paths = [DEFAULT_EVENTS]
    device = torch.device(args.device)
    model, stats_np, _ = load_dynamics_model_for_eval(args.dynamics_ckpt, device=device)
    stats = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in stats_np.items()}

    all_rows = []
    per_run = []
    first_plot_data = None
    for events_path in event_paths:
        events = load_events(events_path)
        grab = first_grasp_decision(events)
        pour = pour_decision(events)
        chunks = collect_chunks(events, grab_decision=grab, window_steps=args.window_steps, gate_x=args.gate_x)
        rows, branches = run_sweep(
            chunks,
            model,
            stats,
            device,
            step_sizes=[float(v) for v in args.step_sizes],
            gradient_steps=args.gradient_steps,
            target_rzz=args.target_rzz,
            gate_x=args.gate_x,
            action_reg=args.action_reg,
            rollout_scale=args.rollout_scale,
            rotation_only=args.rotation_only,
        )
        for row in rows:
            row["events"] = str(events_path)
        all_rows.extend(rows)
        per_run.append(
            {
                "events": str(events_path),
                "grab_decision": int(grab),
                "pour_decision": None if pour is None else int(pour),
                "n_chunks_used": int(len(chunks)),
                "by_step_size": summarize_rows(rows),
            }
        )
        if first_plot_data is None:
            first_plot_data = (true_trajectory(events), branches, grab, pour)

    out_png = args.output_dir / f"{args.name}.png"
    if first_plot_data is not None:
        plot_sweep(
            out_png,
            first_plot_data[0],
            first_plot_data[1],
            grab_decision=first_plot_data[2],
            pour_decision_value=first_plot_data[3],
            gate_x=args.gate_x,
            ylimit=tuple(args.angle_ylim),
        )

    summary = {
        "events": [str(path) for path in event_paths],
        "dynamics_ckpt": str(args.dynamics_ckpt),
        "target_rzz": float(args.target_rzz),
        "gate_x": float(args.gate_x),
        "step_sizes": [float(v) for v in args.step_sizes],
        "gradient_steps": int(args.gradient_steps),
        "action_reg": float(args.action_reg),
        "rotation_only": bool(args.rotation_only),
        "window_steps": int(args.window_steps),
        "n_rows": int(len(all_rows)),
        "aggregate_by_step_size": summarize_rows(all_rows),
        "per_run": per_run,
        "plot_png": str(out_png),
        "plot_pdf": str(out_png.with_suffix(".pdf")),
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
