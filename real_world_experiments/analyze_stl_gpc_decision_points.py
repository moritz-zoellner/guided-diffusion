#!/usr/bin/env python3
"""Offline decision-point test for the real-world STL-GPC baseline.

For each rollout, this script takes the first chunk where the target chain is
trying to choose a pour direction, scores the logged DP candidates with the
same STL-GPC wrist-twist proxy used by the ROS node, and compares that selection
against the candidate pool's automaton pour probabilities.
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
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world_experiments.plot_offline_safety_guidance_branches import (
    dynamics_rollout,
    obs_to_state,
)
from real_world_experiments.real_world_data import rot6d_to_matrix
from real_world_experiments.train_dynamics_world_model import load_dynamics_model_for_eval
from calvin_experiments.calvin_rollout_utils import automaton_model_for_eval


DEFAULT_RUN_DIRS = (
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_1",
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_2",
)
DEFAULT_DYNAMICS = (
    REPO_ROOT
    / "outputs/real_world/dynamics_world_model/hd128_depth2_lr0.001_epochs120_2026-05-13_17-53-04/best_model.pt"
)
DEFAULT_AUTOMATON = (
    REPO_ROOT
    / "outputs/real_world/automaton_world_model/"
    / "h8_max_next_lr0.0001_epochs120_2026-05-25_16-27-36/best_model.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/stl_gpc_sequence_eval/decision_point_analysis"
)
LABEL_NAMES = ("can_grabbed", "pouring_right", "pouring_left")
RIGHT_IDX = 1
LEFT_IDX = 2


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def local_z_twist_from_eef_rot6d(current_rot6d: np.ndarray, pred_rot6d: np.ndarray) -> np.ndarray:
    current_r = rot6d_to_matrix(np.asarray(current_rot6d, dtype=np.float32)[None])[0]
    pred_r = rot6d_to_matrix(np.asarray(pred_rot6d, dtype=np.float32))
    rel_r = np.einsum("ij,...jk->...ik", current_r.T, pred_r)
    rotvec = Rotation.from_matrix(rel_r.reshape(-1, 3, 3)).as_rotvec().reshape(rel_r.shape[:-2] + (3,))
    return rotvec[..., 2]


def stl_scores_for_event(
    event: dict[str, Any],
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    target_label: str,
    threshold_deg: float,
    rollout_scale: float,
    score_rule: str,
) -> dict[str, np.ndarray]:
    actions = np.asarray(event["candidate_action_chunks"], dtype=np.float32)
    obs = event["obs"]
    state0 = torch.as_tensor(obs_to_state(obs), device=device, dtype=torch.float32).unsqueeze(0)
    action_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_states = dynamics_rollout(state0, action_t, model, stats, rollout_scale=rollout_scale)
    pred_np = pred_states.detach().cpu().numpy()

    current_wrist = event.get("wrist_joint_delta_rad")
    if current_wrist is None and event.get("wrist_joint_delta_deg") is not None:
        current_wrist = float(np.deg2rad(event["wrist_joint_delta_deg"]))
    current_wrist = 0.0 if current_wrist is None else float(current_wrist)
    twist_delta = local_z_twist_from_eef_rot6d(
        np.asarray(obs["eef_rot6d"], dtype=np.float32),
        pred_np[..., 3:9],
    )
    pred_wrist = current_wrist + twist_delta
    signed = pred_wrist if target_label == "pouring_right" else -pred_wrist
    robustness_by_step = signed - np.deg2rad(float(threshold_deg))
    if score_rule == "final":
        score = robustness_by_step[:, -1]
    elif score_rule == "mean":
        score = robustness_by_step.mean(axis=1)
    else:
        score = robustness_by_step.max(axis=1)
    return {
        "score": score.astype(np.float64),
        "signed_wrist_deg": np.rad2deg(signed).astype(np.float64),
        "max_signed_wrist_deg": np.rad2deg(signed.max(axis=1)).astype(np.float64),
    }


def automaton_probs_for_event(
    event: dict[str, Any],
    automaton_model: torch.nn.Module | None,
    automaton_stats: dict[str, np.ndarray] | None,
    device: torch.device,
) -> np.ndarray:
    logged_probs = event.get("label_probs")
    if logged_probs is not None:
        return np.asarray(logged_probs, dtype=np.float64)
    if automaton_model is None or automaton_stats is None:
        raise ValueError("event has no label_probs and no automaton checkpoint was provided")

    action_chunks = np.asarray(event["candidate_action_chunks"], dtype=np.float32)
    n_candidates, _, action_dim = action_chunks.shape
    action_chunk_dim = len(automaton_stats["actions_mean"])
    automaton_horizon = action_chunk_dim // action_dim
    flat_chunks = action_chunks[:, :automaton_horizon, :].reshape(n_candidates, -1)
    state = obs_to_state(event["obs"])
    states = np.repeat(state[None, :], n_candidates, axis=0)
    labels = np.repeat(np.asarray(event["current_label"], dtype=np.float32)[None, :], n_candidates, axis=0)

    states_t = torch.as_tensor(
        (states - automaton_stats["states_mean"]) / automaton_stats["states_std"],
        device=device,
        dtype=torch.float32,
    )
    actions_t = torch.as_tensor(
        (flat_chunks - automaton_stats["actions_mean"]) / automaton_stats["actions_std"],
        device=device,
        dtype=torch.float32,
    )
    labels_t = torch.as_tensor(labels, device=device, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(automaton_model(states_t, actions_t, labels_t)).detach().cpu().numpy()
    return probs.astype(np.float64)


def first_pour_chunk(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        selection = event.get("selection") or {}
        if int(selection.get("chain_pos", -1)) == 1:
            return event
    return None


def summarize_values(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": int(len(arr)),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def rank_desc(values: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(values))
    return int(np.where(order == int(index))[0][0] + 1)


def analyze_run(
    run_dir: Path,
    dynamics_model: torch.nn.Module,
    dynamics_stats: dict[str, torch.Tensor],
    automaton_model: torch.nn.Module | None,
    automaton_stats: dict[str, np.ndarray] | None,
    device: torch.device,
    *,
    threshold_deg: float,
    rollout_scale: float,
    score_rule: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_config = json.loads((run_dir / "run_config.json").read_text())
    run_summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    rollout_summary_by_idx = {int(row["rollout_idx"]): row for row in run_summary.get("rollouts", [])}
    target_chain = run_config.get("target_chain_parsed", [])
    if len(target_chain) < 2:
        raise ValueError(f"{run_dir} target_chain_parsed has no pour target")
    target_idx = int(target_chain[1]["label_idx"])
    target_label = LABEL_NAMES[target_idx]
    opposite_idx = LEFT_IDX if target_idx == RIGHT_IDX else RIGHT_IDX
    opposite_label = LABEL_NAMES[opposite_idx]

    rollout_rows = []
    candidate_rows = []
    for events_path in sorted((run_dir / "rollouts").glob("rollout_*/events.jsonl")):
        rollout_idx = int(events_path.parent.name.split("_")[-1])
        event = first_pour_chunk(load_jsonl(events_path))
        if event is None:
            continue
        label_probs = automaton_probs_for_event(event, automaton_model, automaton_stats, device)
        stl = stl_scores_for_event(
            event,
            dynamics_model,
            dynamics_stats,
            device,
            target_label=target_label,
            threshold_deg=threshold_deg,
            rollout_scale=rollout_scale,
            score_rule=score_rule,
        )
        stl_scores = stl["score"]
        target_probs = label_probs[:, target_idx]
        opposite_probs = label_probs[:, opposite_idx]
        margins = target_probs - opposite_probs
        desired_wins = margins > 0.0

        stl_idx = int(np.argmax(stl_scores))
        logged_selected_idx = int(event["selected_candidate"])
        automaton_idx = int(np.argmax(target_probs))
        base_idx = 0
        random_success_frac = float(np.mean(desired_wins))
        stl_success = bool(desired_wins[stl_idx])
        base_success = bool(desired_wins[base_idx])
        auto_success = bool(desired_wins[automaton_idx])
        logged_success = bool(desired_wins[logged_selected_idx])
        spear = spearmanr(stl_scores, margins).correlation
        rollout_summary = rollout_summary_by_idx.get(rollout_idx, {})

        rollout_rows.append(
            {
                "run_name": run_dir.name,
                "rollout_idx": rollout_idx,
                "target_label": target_label,
                "opposite_label": opposite_label,
                "chunk_idx": int(event["chunk_idx"]),
                "decision_idx_before": int(event["decision_idx_before"]),
                "n_candidates": int(len(stl_scores)),
                "base_idx": int(base_idx),
                "stl_selected_idx": int(stl_idx),
                "logged_selected_idx": int(logged_selected_idx),
                "automaton_selected_idx": int(automaton_idx),
                "rollout_success": rollout_summary.get("success"),
                "rollout_termination_reason": rollout_summary.get("termination_reason"),
                "rollout_label_sequence": [
                    row.get("label_name") for row in rollout_summary.get("label_events", [])
                ],
                "random_candidate_desired_win_frac": random_success_frac,
                "base_first_candidate_desired_win": base_success,
                "stl_selected_desired_win": stl_success,
                "automaton_selected_desired_win": auto_success,
                "logged_selected_desired_win": logged_success,
                "base_margin": float(margins[base_idx]),
                "stl_margin": float(margins[stl_idx]),
                "automaton_margin": float(margins[automaton_idx]),
                "logged_selected_margin": float(margins[logged_selected_idx]),
                "stl_selected_target_prob": float(target_probs[stl_idx]),
                "stl_selected_opposite_prob": float(opposite_probs[stl_idx]),
                "base_target_prob": float(target_probs[base_idx]),
                "base_opposite_prob": float(opposite_probs[base_idx]),
                "automaton_target_prob": float(target_probs[automaton_idx]),
                "automaton_opposite_prob": float(opposite_probs[automaton_idx]),
                "logged_selected_target_prob": float(target_probs[logged_selected_idx]),
                "logged_selected_opposite_prob": float(opposite_probs[logged_selected_idx]),
                "stl_score_min": float(stl_scores.min()),
                "stl_score_median": float(np.median(stl_scores)),
                "stl_score_max": float(stl_scores.max()),
                "stl_max_signed_wrist_deg_min": float(stl["max_signed_wrist_deg"].min()),
                "stl_max_signed_wrist_deg_median": float(np.median(stl["max_signed_wrist_deg"])),
                "stl_max_signed_wrist_deg_max": float(stl["max_signed_wrist_deg"].max()),
                "stl_selected_max_signed_wrist_deg": float(stl["max_signed_wrist_deg"][stl_idx]),
                "base_max_signed_wrist_deg": float(stl["max_signed_wrist_deg"][base_idx]),
                "automaton_max_signed_wrist_deg": float(stl["max_signed_wrist_deg"][automaton_idx]),
                "logged_selected_max_signed_wrist_deg": float(stl["max_signed_wrist_deg"][logged_selected_idx]),
                "stl_rank_of_automaton_selected": rank_desc(stl_scores, automaton_idx),
                "automaton_margin_rank_of_stl_selected": rank_desc(margins, stl_idx),
                "spearman_stl_score_vs_automaton_margin": float(spear) if spear == spear else None,
            }
        )
        for cand_idx in range(len(stl_scores)):
            candidate_rows.append(
                {
                    "run_name": run_dir.name,
                    "rollout_idx": rollout_idx,
                    "target_label": target_label,
                    "candidate_idx": int(cand_idx),
                    "stl_score": float(stl_scores[cand_idx]),
                    "stl_max_signed_wrist_deg": float(stl["max_signed_wrist_deg"][cand_idx]),
                    "automaton_target_prob": float(target_probs[cand_idx]),
                    "automaton_opposite_prob": float(opposite_probs[cand_idx]),
                    "automaton_margin": float(margins[cand_idx]),
                    "desired_win": bool(desired_wins[cand_idx]),
                    "is_stl_selected": bool(cand_idx == stl_idx),
                    "is_base_first": bool(cand_idx == base_idx),
                    "is_automaton_selected": bool(cand_idx == automaton_idx),
                    "is_logged_selected": bool(cand_idx == logged_selected_idx),
                }
            )

    summary = {
        "run_dir": str(run_dir),
        "target_label": target_label,
        "opposite_label": opposite_label,
        "num_decision_points": int(len(rollout_rows)),
        "random_candidate_desired_win_frac": summarize_values(
            [row["random_candidate_desired_win_frac"] for row in rollout_rows]
        ),
        "base_first_candidate_success_rate": float(np.mean([row["base_first_candidate_desired_win"] for row in rollout_rows]))
        if rollout_rows
        else None,
        "stl_selected_success_rate": float(np.mean([row["stl_selected_desired_win"] for row in rollout_rows]))
        if rollout_rows
        else None,
        "automaton_selected_success_rate": float(np.mean([row["automaton_selected_desired_win"] for row in rollout_rows]))
        if rollout_rows
        else None,
        "logged_selected_success_rate": float(np.mean([row["logged_selected_desired_win"] for row in rollout_rows]))
        if rollout_rows
        else None,
        "actual_rollout_success_rate": float(
            np.mean([bool(row["rollout_success"]) for row in rollout_rows if row["rollout_success"] is not None])
        )
        if any(row["rollout_success"] is not None for row in rollout_rows)
        else None,
        "base_margin": summarize_values([row["base_margin"] for row in rollout_rows]),
        "stl_margin": summarize_values([row["stl_margin"] for row in rollout_rows]),
        "automaton_margin": summarize_values([row["automaton_margin"] for row in rollout_rows]),
        "logged_selected_margin": summarize_values([row["logged_selected_margin"] for row in rollout_rows]),
        "spearman_stl_score_vs_automaton_margin": summarize_values(
            [
                row["spearman_stl_score_vs_automaton_margin"]
                for row in rollout_rows
                if row["spearman_stl_score_vs_automaton_margin"] is not None
            ]
        ),
        "stl_max_signed_wrist_deg_max": summarize_values(
            [row["stl_max_signed_wrist_deg_max"] for row in rollout_rows]
        ),
        "stl_max_signed_wrist_deg_median": summarize_values(
            [row["stl_max_signed_wrist_deg_median"] for row in rollout_rows]
        ),
    }
    return summary, rollout_rows, candidate_rows


def write_plot(output_path: Path, rollout_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for target_label, color in [("pouring_left", "#1f77b4"), ("pouring_right", "#ff7f0e")]:
        sub = [row for row in candidate_rows if row["target_label"] == target_label]
        if not sub:
            continue
        ax.scatter(
            [row["stl_max_signed_wrist_deg"] for row in sub],
            [row["automaton_margin"] for row in sub],
            s=18,
            alpha=0.35,
            color=color,
            label=f"{target_label} candidates",
        )
    selected = [row for row in candidate_rows if row["is_stl_selected"]]
    ax.scatter(
        [row["stl_max_signed_wrist_deg"] for row in selected],
        [row["automaton_margin"] for row in selected],
        marker="*",
        s=145,
        color="#d62728",
        edgecolor="#111111",
        linewidth=0.7,
        label="STL selected",
    )
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    ax.axvline(90.0, color="0.45", linestyle=":", linewidth=1)
    ax.set_xlabel("STL proxy max signed wrist twist [deg]")
    ax.set_ylabel("automaton margin p(desired) - p(opposite)")
    ax.set_title("first post-grip pour decision candidates")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    xs = np.arange(len(rollout_rows))
    colors = ["#1f77b4" if row["target_label"] == "pouring_left" else "#ff7f0e" for row in rollout_rows]
    ax.scatter(xs - 0.18, [row["base_margin"] for row in rollout_rows], marker="o", color="0.45", label="base first")
    ax.scatter(xs, [row["stl_margin"] for row in rollout_rows], marker="*", s=110, color=colors, label="STL selected")
    ax.scatter(xs + 0.18, [row["automaton_margin"] for row in rollout_rows], marker="x", color="#2ca02c", label="automaton selected")
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("decision point")
    ax.set_ylabel("automaton margin p(desired) - p(opposite)")
    ax.set_title("selection comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="*", default=list(DEFAULT_RUN_DIRS))
    parser.add_argument("--dynamics-ckpt", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--automaton-ckpt", type=Path, default=DEFAULT_AUTOMATON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-deg", type=float, default=90.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--score-rule", choices=["eventually", "final", "mean"], default="eventually")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dynamics_model, stats_np, _ = load_dynamics_model_for_eval(args.dynamics_ckpt, device=device)
    dynamics_stats = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in stats_np.items()}
    automaton_model = None
    automaton_stats = None
    if args.automaton_ckpt:
        automaton_model, automaton_stats, _automaton_meta = automaton_model_for_eval(args.automaton_ckpt, device)

    summaries = []
    rollout_rows = []
    candidate_rows = []
    for run_dir in args.run_dir:
        summary, rows, candidates = analyze_run(
            run_dir,
            dynamics_model,
            dynamics_stats,
            automaton_model,
            automaton_stats,
            device,
            threshold_deg=args.threshold_deg,
            rollout_scale=args.rollout_scale,
            score_rule=args.score_rule,
        )
        summaries.append(summary)
        rollout_rows.extend(rows)
        candidate_rows.extend(candidates)

    overall = {
        "dynamics_ckpt": str(args.dynamics_ckpt),
        "automaton_ckpt": str(args.automaton_ckpt) if args.automaton_ckpt else None,
        "threshold_deg": float(args.threshold_deg),
        "rollout_scale": float(args.rollout_scale),
        "score_rule": args.score_rule,
        "runs": summaries,
        "overall": {
            "num_decision_points": int(len(rollout_rows)),
            "base_first_candidate_success_rate": float(np.mean([r["base_first_candidate_desired_win"] for r in rollout_rows]))
            if rollout_rows
            else None,
            "stl_selected_success_rate": float(np.mean([r["stl_selected_desired_win"] for r in rollout_rows]))
            if rollout_rows
            else None,
            "automaton_selected_success_rate": float(np.mean([r["automaton_selected_desired_win"] for r in rollout_rows]))
            if rollout_rows
            else None,
            "logged_selected_success_rate": float(np.mean([r["logged_selected_desired_win"] for r in rollout_rows]))
            if rollout_rows
            else None,
            "actual_rollout_success_rate": float(
                np.mean([bool(r["rollout_success"]) for r in rollout_rows if r["rollout_success"] is not None])
            )
            if any(r["rollout_success"] is not None for r in rollout_rows)
            else None,
            "random_candidate_desired_win_frac": summarize_values(
                [r["random_candidate_desired_win_frac"] for r in rollout_rows]
            ),
            "base_margin": summarize_values([r["base_margin"] for r in rollout_rows]),
            "stl_margin": summarize_values([r["stl_margin"] for r in rollout_rows]),
            "automaton_margin": summarize_values([r["automaton_margin"] for r in rollout_rows]),
            "logged_selected_margin": summarize_values([r["logged_selected_margin"] for r in rollout_rows]),
            "spearman_stl_score_vs_automaton_margin": summarize_values(
                [
                    r["spearman_stl_score_vs_automaton_margin"]
                    for r in rollout_rows
                    if r["spearman_stl_score_vs_automaton_margin"] is not None
                ]
            ),
        },
        "decision_points": rollout_rows,
    }
    summary_path = args.output_dir / "stl_gpc_decision_point_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2, sort_keys=True))
    write_plot(args.output_dir / "stl_gpc_decision_point_comparison.png", rollout_rows, candidate_rows)
    print(json.dumps(overall["overall"], indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
