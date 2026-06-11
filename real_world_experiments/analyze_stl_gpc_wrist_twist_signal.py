#!/usr/bin/env python3
"""Sanity-check an STL-GPC wrist-twist robustness signal on logged DP candidates.

The dynamics world model does not predict wrist_3_joint directly. This script
therefore uses the predicted EEF rotation to estimate the local-z twist over the
candidate horizon, then adds that increment to the measured wrist delta at the
chunk start. That is the closest STL-GPC proxy available from the current world
model state.
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


DEFAULT_ROLLOUT = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_left_epoch160_n10_2/rollouts/rollout_000"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/stl_gpc_sanity"
DEFAULT_DYNAMICS = (
    REPO_ROOT
    / "outputs/real_world/dynamics_world_model/"
    / "hd128_depth2_lr0.001_epochs120_2026-05-13_17-53-04/best_model.pt"
)


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


def load_events(rollout_dir: Path) -> list[dict[str, Any]]:
    with (rollout_dir / "events.jsonl").open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def selected_chunk(event: dict[str, Any]) -> np.ndarray | None:
    if event.get("selected_chunk") is not None:
        return np.asarray(event["selected_chunk"], dtype=np.float32)
    candidates = event.get("candidate_action_chunks")
    selected = event.get("selected_candidate")
    if candidates is None or selected is None:
        return None
    return np.asarray(candidates[int(selected)], dtype=np.float32)


def candidate_chunks(event: dict[str, Any]) -> np.ndarray | None:
    chunks = event.get("candidate_action_chunks")
    if chunks is None:
        return None
    return np.asarray(chunks, dtype=np.float32)


def local_z_twist_from_eef_rot6d(current_rot6d: np.ndarray, pred_rot6d: np.ndarray) -> np.ndarray:
    current_R = rot6d_to_matrix(np.asarray(current_rot6d, dtype=np.float32)[None])[0]
    pred_R = rot6d_to_matrix(np.asarray(pred_rot6d, dtype=np.float32))
    rel_R = np.einsum("ij,...jk->...ik", current_R.T, pred_R)
    rotvec = Rotation.from_matrix(rel_R.reshape(-1, 3, 3)).as_rotvec().reshape(rel_R.shape[:-2] + (3,))
    return rotvec[..., 2]


def rollout_candidate_states(
    event: dict[str, Any],
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    rollout_scale: float,
) -> np.ndarray | None:
    obs = event.get("obs")
    actions = candidate_chunks(event)
    if obs is None or actions is None:
        return None
    state0 = torch.as_tensor(obs_to_state(obs), device=device, dtype=torch.float32).unsqueeze(0)
    action_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    with torch.no_grad():
        states = dynamics_rollout(state0, action_t, model, stats, rollout_scale=rollout_scale)
    return states.detach().cpu().numpy()


def stl_twist_scores_for_event(
    event: dict[str, Any],
    pred_states: np.ndarray,
    *,
    target_label: str,
    threshold_deg: float,
) -> dict[str, np.ndarray]:
    obs = event["obs"]
    current_wrist = float(event.get("wrist_joint_delta_rad", 0.0))
    current_eef_rot = np.asarray(obs["eef_rot6d"], dtype=np.float32)
    pred_eef_rot = pred_states[..., 3:9]
    twist_delta = local_z_twist_from_eef_rot6d(current_eef_rot, pred_eef_rot)
    pred_wrist = current_wrist + twist_delta
    if target_label == "pouring_left":
        signed = -pred_wrist
    else:
        signed = pred_wrist
    threshold = np.deg2rad(float(threshold_deg))
    robustness = np.max(signed - threshold, axis=1)
    max_signed_deg = np.degrees(np.max(signed, axis=1))
    return {
        "robustness_rad": robustness,
        "max_signed_twist_deg": max_signed_deg,
        "pred_wrist_delta_rad": pred_wrist,
    }


def rank_desc(values: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(values))
    return int(np.where(order == int(index))[0][0] + 1)


def analyze(
    rollout_dir: Path,
    dynamics_ckpt: Path,
    output_dir: Path,
    *,
    target_label: str | None,
    threshold_deg: float,
    rollout_scale: float,
    device_name: str,
) -> dict[str, Any]:
    events = load_events(rollout_dir)
    device = torch.device(device_name)
    model, stats_np, _ = load_dynamics_model_for_eval(dynamics_ckpt, device=device)
    stats = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in stats_np.items()}

    chunk_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        selection = event.get("selection", {})
        active_target = selection.get("target_label_name")
        if active_target not in {"pouring_right", "pouring_left"}:
            continue
        if target_label is not None and active_target != target_label:
            continue
        pred_states = rollout_candidate_states(event, model, stats, device, rollout_scale=rollout_scale)
        if pred_states is None:
            continue
        scores = stl_twist_scores_for_event(
            event,
            pred_states,
            target_label=active_target,
            threshold_deg=threshold_deg,
        )
        robustness = scores["robustness_rad"]
        max_twist_deg = scores["max_signed_twist_deg"]
        label_probs = np.asarray(event.get("label_probs"), dtype=np.float64)
        label_idx = int(selection.get("target_label_idx", 1 if active_target == "pouring_right" else 2))
        auto_scores = label_probs[:, label_idx]
        selected = int(event.get("selected_candidate", np.argmax(auto_scores)))
        stl_best = int(np.argmax(robustness))
        auto_best = int(np.argmax(auto_scores))
        spear = spearmanr(robustness, auto_scores).correlation
        pear = np.corrcoef(robustness, auto_scores)[0, 1] if len(robustness) > 1 else np.nan
        row = {
            "chunk_idx": int(event.get("chunk_idx", len(chunk_rows))),
            "decision_idx_before": int(event.get("decision_idx_before", 0)),
            "target_label": active_target,
            "selected_candidate": selected,
            "automaton_best_candidate": auto_best,
            "stl_best_candidate": stl_best,
            "selected_stl_rank": rank_desc(robustness, selected),
            "selected_automaton_rank": rank_desc(auto_scores, selected),
            "stl_best_automaton_rank": rank_desc(auto_scores, stl_best),
            "automaton_best_stl_rank": rank_desc(robustness, auto_best),
            "selected_stl_max_twist_deg": float(max_twist_deg[selected]),
            "stl_best_max_twist_deg": float(max_twist_deg[stl_best]),
            "automaton_best_max_twist_deg": float(max_twist_deg[auto_best]),
            "stl_twist_deg_min": float(np.min(max_twist_deg)),
            "stl_twist_deg_median": float(np.median(max_twist_deg)),
            "stl_twist_deg_max": float(np.max(max_twist_deg)),
            "stl_robustness_rad_min": float(np.min(robustness)),
            "stl_robustness_rad_median": float(np.median(robustness)),
            "stl_robustness_rad_max": float(np.max(robustness)),
            "automaton_score_min": float(np.min(auto_scores)),
            "automaton_score_median": float(np.median(auto_scores)),
            "automaton_score_max": float(np.max(auto_scores)),
            "spearman_stl_vs_automaton": float(spear) if spear == spear else None,
            "pearson_stl_vs_automaton": float(pear) if pear == pear else None,
        }
        chunk_rows.append(row)
        for cand_idx in range(len(robustness)):
            candidate_rows.append(
                {
                    "chunk_idx": row["chunk_idx"],
                    "decision_idx_before": row["decision_idx_before"],
                    "candidate_idx": int(cand_idx),
                    "target_label": active_target,
                    "stl_robustness_rad": float(robustness[cand_idx]),
                    "stl_max_signed_twist_deg": float(max_twist_deg[cand_idx]),
                    "automaton_score": float(auto_scores[cand_idx]),
                    "is_selected": bool(cand_idx == selected),
                    "is_stl_best": bool(cand_idx == stl_best),
                    "is_automaton_best": bool(cand_idx == auto_best),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{rollout_dir.parent.parent.name}_{rollout_dir.name}_stl_gpc_twist_signal"
    write_csv(output_dir / f"{stem}_chunks.csv", chunk_rows)
    write_csv(output_dir / f"{stem}_candidates.csv", candidate_rows)
    plot_path = output_dir / f"{stem}.png"
    write_plot(plot_path, chunk_rows, candidate_rows, threshold_deg=threshold_deg)
    summary = {
        "rollout_dir": str(rollout_dir),
        "dynamics_ckpt": str(dynamics_ckpt),
        "target_label_filter": target_label,
        "threshold_deg": float(threshold_deg),
        "rollout_scale": float(rollout_scale),
        "n_chunks_analyzed": int(len(chunk_rows)),
        "n_candidates_analyzed": int(len(candidate_rows)),
        "mean_spearman_stl_vs_automaton": float(np.nanmean([r["spearman_stl_vs_automaton"] for r in chunk_rows])) if chunk_rows else None,
        "median_selected_stl_rank": float(np.median([r["selected_stl_rank"] for r in chunk_rows])) if chunk_rows else None,
        "median_stl_best_automaton_rank": float(np.median([r["stl_best_automaton_rank"] for r in chunk_rows])) if chunk_rows else None,
        "chunks": chunk_rows,
        "plot_png": str(plot_path),
        "plot_pdf": str(plot_path.with_suffix(".pdf")),
        "chunks_csv": str(output_dir / f"{stem}_chunks.csv"),
        "candidates_csv": str(output_dir / f"{stem}_candidates.csv"),
        "world_model_state_note": "No wrist_joint_delta in dynamics state; STL score uses measured current wrist plus predicted EEF local-z twist from eef_rot6d.",
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, chunk_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], *, threshold_deg: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    if candidate_rows:
        x = np.asarray([r["stl_max_signed_twist_deg"] for r in candidate_rows], dtype=np.float64)
        y = np.asarray([r["automaton_score"] for r in candidate_rows], dtype=np.float64)
        chunks = np.asarray([r["chunk_idx"] for r in candidate_rows], dtype=np.int32)
        sc = ax.scatter(x, y, c=chunks, cmap="viridis", s=18, alpha=0.55, label="candidate")
        fig.colorbar(sc, ax=ax, label="chunk")
        selected = [r for r in candidate_rows if r["is_selected"]]
        if selected:
            ax.scatter(
                [r["stl_max_signed_twist_deg"] for r in selected],
                [r["automaton_score"] for r in selected],
                marker="*",
                s=160,
                edgecolor="#111111",
                facecolor="#ffcc00",
                linewidth=0.8,
                label="automaton selected",
            )
        stl_best = [r for r in candidate_rows if r["is_stl_best"]]
        if stl_best:
            ax.scatter(
                [r["stl_max_signed_twist_deg"] for r in stl_best],
                [r["automaton_score"] for r in stl_best],
                marker="x",
                s=90,
                color="#d62728",
                linewidth=1.5,
                label="STL-GPC best",
            )
    ax.axvline(threshold_deg, color="#d62728", linestyle="--", linewidth=1.0, label="90 deg threshold")
    ax.set_xlabel("STL proxy max signed wrist twist [deg]")
    ax.set_ylabel("automaton p(target label)")
    ax.set_title("candidate signal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    if chunk_rows:
        decisions = [r["decision_idx_before"] for r in chunk_rows]
        ax.fill_between(
            decisions,
            [r["stl_twist_deg_min"] for r in chunk_rows],
            [r["stl_twist_deg_max"] for r in chunk_rows],
            color="#1f77b4",
            alpha=0.18,
            label="candidate min/max",
        )
        ax.plot(decisions, [r["stl_twist_deg_median"] for r in chunk_rows], "-o", color="#1f77b4", label="candidate median")
        ax.plot(decisions, [r["selected_stl_max_twist_deg"] for r in chunk_rows], "-*", color="#ff7f0e", label="automaton selected")
        ax.plot(decisions, [r["stl_best_max_twist_deg"] for r in chunk_rows], "-x", color="#d62728", label="STL-GPC best")
    ax.axhline(threshold_deg, color="#d62728", linestyle="--", linewidth=1.0, label="90 deg threshold")
    ax.set_xlabel("decision before chunk")
    ax.set_ylabel("max signed twist over WM horizon [deg]")
    ax.set_title("score over rollout")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--dynamics-ckpt", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-label", choices=["pouring_right", "pouring_left"], default=None)
    parser.add_argument("--threshold-deg", type=float, default=90.0)
    parser.add_argument("--rollout-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    configure_matplotlib()
    summary = analyze(
        args.rollout_dir,
        args.dynamics_ckpt,
        args.output_dir,
        target_label=args.target_label,
        threshold_deg=args.threshold_deg,
        rollout_scale=args.rollout_scale,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
