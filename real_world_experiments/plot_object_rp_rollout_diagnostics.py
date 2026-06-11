#!/usr/bin/env python3
"""Diagnose object roll/pitch/yaw guidance runs from real-world event logs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"
AXES = ("x/roll", "y/pitch", "z/yaw")
COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float64)
    a1 = arr[..., 0:3]
    a2 = arr[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "rollout_start":
        return event.get("obs")
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "decision":
        return event.get("obs")
    return None


def unwrap_deg(euler: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(euler), axis=0))


def rel_euler_deg(rot6d: np.ndarray, ref_rot6d: np.ndarray) -> np.ndarray:
    matrices = rot6d_to_matrix(rot6d)
    ref = rot6d_to_matrix(np.asarray(ref_rot6d, dtype=np.float64)[None])[0]
    rel = np.einsum("ij,...jk->...ik", ref.T, matrices)
    return unwrap_deg(Rotation.from_matrix(rel).as_euler("xyz", degrees=True))


def selected_pred_rot6d(event: dict[str, Any]) -> np.ndarray | None:
    pred = event.get("dynamics_prediction") or {}
    selected = pred.get("selected") or {}
    if "cheezit_rot6d" in selected:
        return np.asarray(selected["cheezit_rot6d"], dtype=np.float64)
    state = selected.get("state")
    if state is not None:
        arr = np.asarray(state, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 19:
            return arr[:, 13:19]
    return None


def collect(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    label_events = []
    ref_event = None
    chunk_rows = []

    for event in events:
        etype = event.get("type")
        if etype == "object_roll_pitch_reference_recorded":
            ref_event = event
        if etype == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        obs = obs_from_event(event)
        if obs is not None and "cheezit_rot6d" in obs:
            rows.append(
                {
                    "event_type": etype,
                    "decision": int(event.get("decision_idx", event.get("decision_idx_before", 0))),
                    "label": event.get("current_label") or event.get("label") or [0, 0, 0],
                    "eef_pos": obs.get("eef_pos"),
                    "cheezit_pos": obs.get("cheezit_pos"),
                    "cheezit_rot6d": obs["cheezit_rot6d"],
                    "wrist_delta_deg": event.get("wrist_joint_delta_deg"),
                }
            )
        if etype == "chunk_sample":
            sel = event.get("selection") or {}
            rp = sel.get("object_roll_pitch_guidance") or {}
            pred_rot6d = selected_pred_rot6d(event)
            chunk_rows.append(
                {
                    "decision_before": int(event.get("decision_idx_before", 0)),
                    "chunk_idx": int(event.get("chunk_idx", -1)),
                    "target": f"{sel.get('target_mode')}({sel.get('target_label_name')})",
                    "selected_candidate": event.get("selected_candidate"),
                    "selected_label_probs": sel.get("selected_label_probs"),
                    "object_rp": rp,
                    "pred_rot6d": pred_rot6d,
                    "obs": event.get("obs"),
                }
            )

    if not rows:
        raise ValueError("No Cheez-It observations found in events")

    if ref_event is not None:
        ref_rot6d = np.asarray(ref_event["cheezit_rot6d"], dtype=np.float64)
        ref_decision = int(ref_event.get("decision_idx", 0))
        ref_source = "object_roll_pitch_reference_recorded"
    else:
        grabbed_rows = [r for r in rows if r["label"] and int(r["label"][0]) == 1]
        ref_row = grabbed_rows[0] if grabbed_rows else rows[0]
        ref_rot6d = np.asarray(ref_row["cheezit_rot6d"], dtype=np.float64)
        ref_decision = int(ref_row["decision"])
        ref_source = "first grabbed obs" if grabbed_rows else "first obs"

    decisions = np.asarray([r["decision"] for r in rows], dtype=np.int32)
    labels = np.asarray([r["label"] for r in rows], dtype=np.int32)
    rot6d = np.asarray([r["cheezit_rot6d"] for r in rows], dtype=np.float64)
    obj_pos = np.asarray([r["cheezit_pos"] for r in rows], dtype=np.float64)
    eef_pos = np.asarray([r["eef_pos"] for r in rows], dtype=np.float64)
    wrist = np.asarray(
        [np.nan if r["wrist_delta_deg"] is None else float(r["wrist_delta_deg"]) for r in rows],
        dtype=np.float64,
    )
    rel_euler = rel_euler_deg(rot6d, ref_rot6d)
    abs_euler = unwrap_deg(Rotation.from_matrix(rot6d_to_matrix(rot6d)).as_euler("xyz", degrees=True))
    return {
        "rows": rows,
        "decisions": decisions,
        "labels": labels,
        "rot6d": rot6d,
        "obj_pos": obj_pos,
        "eef_pos": eef_pos,
        "wrist_delta_deg": wrist,
        "rel_euler_deg": rel_euler,
        "abs_euler_deg": abs_euler,
        "ref_rot6d": ref_rot6d,
        "ref_decision": ref_decision,
        "ref_source": ref_source,
        "label_events": label_events,
        "chunk_rows": chunk_rows,
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(data: dict[str, Any], path: Path) -> None:
    decisions = data["decisions"]
    labels = data["labels"]
    rel_euler = data["rel_euler_deg"]
    abs_euler = data["abs_euler_deg"]
    obj_pos = data["obj_pos"]
    eef_pos = data["eef_pos"]
    wrist = data["wrist_delta_deg"]
    ref_decision = data["ref_decision"]
    ref_rot6d = data["ref_rot6d"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=False)

    for axis, name in enumerate(AXES):
        axes[0].plot(decisions, rel_euler[:, axis], color=COLORS[axis], linewidth=1.4, label=f"true {name}")
    for chunk in data["chunk_rows"]:
        pred_rot6d = chunk["pred_rot6d"]
        if pred_rot6d is None:
            continue
        x0 = chunk["decision_before"]
        xs = np.arange(x0 + 1, x0 + 1 + len(pred_rot6d))
        pred_euler = rel_euler_deg(pred_rot6d, ref_rot6d)
        for axis in range(3):
            axes[0].plot(xs, pred_euler[:, axis], color=COLORS[axis], alpha=0.22, linewidth=1.0)
    axes[0].axvline(ref_decision, color="#111111", linestyle=":", linewidth=1.0, label="guidance ref")
    axes[0].set_title("Cheez-It orientation relative to object-roll/pitch guidance reference")
    axes[0].set_ylabel("angle [deg]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=4, loc="upper left")

    for axis, name in enumerate(AXES):
        axes[1].plot(decisions, abs_euler[:, axis], color=COLORS[axis], linewidth=1.1, label=name)
    axes[1].set_title("Absolute object orientation")
    axes[1].set_ylabel("angle [deg]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3, loc="upper left")

    axes[2].plot(decisions, wrist, color="#333333", linewidth=1.2, label="wrist delta")
    axes[2].plot(decisions, labels[:, 0] * 30.0, color="#6a4c93", linewidth=1.0, label="grabbed x30")
    axes[2].plot(decisions, labels[:, 1] * 35.0, color="#ffb000", linewidth=1.0, label="right x35")
    axes[2].plot(decisions, labels[:, 2] * 40.0, color="#00a6d6", linewidth=1.0, label="left x40")
    axes[2].set_title("Wrist and online labels")
    axes[2].set_ylabel("deg / label scale")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(ncol=4, loc="upper left")

    axes[3].plot(eef_pos[:, 0], eef_pos[:, 1], color="#111111", linewidth=1.2, label="EEF")
    axes[3].plot(obj_pos[:, 0], obj_pos[:, 1], color="#d95f02", linewidth=1.2, label="Cheez-It")
    axes[3].scatter(obj_pos[0, 0], obj_pos[0, 1], color="#2ca02c", s=25, label="start obj", zorder=4)
    axes[3].scatter(obj_pos[-1, 0], obj_pos[-1, 1], color="#d62728", s=25, label="last obj", zorder=4)
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].set_title("Top-down XY trace")
    axes[3].set_xlabel("world x [m]")
    axes[3].set_ylabel("world y [m]")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(ncol=4, loc="best")

    rp_chunks = [c for c in data["chunk_rows"] if (c["object_rp"] or {}).get("applied")]
    if rp_chunks:
        xs = np.asarray([c["decision_before"] for c in rp_chunks], dtype=float)
        pre_roll = [c["object_rp"].get("pre_mean_abs_roll_deg", np.nan) for c in rp_chunks]
        post_roll = [c["object_rp"].get("post_mean_abs_roll_deg", np.nan) for c in rp_chunks]
        pre_pitch = [c["object_rp"].get("pre_mean_abs_pitch_deg", np.nan) for c in rp_chunks]
        post_pitch = [c["object_rp"].get("post_mean_abs_pitch_deg", np.nan) for c in rp_chunks]
        delta = [c["object_rp"].get("mean_action_delta_l2", np.nan) for c in rp_chunks]
        axes[4].plot(xs, pre_roll, color=COLORS[0], linestyle="--", marker="o", label="pre roll")
        axes[4].plot(xs, post_roll, color=COLORS[0], linestyle="-", marker="o", label="post roll")
        axes[4].plot(xs, pre_pitch, color=COLORS[1], linestyle="--", marker="o", label="pre pitch")
        axes[4].plot(xs, post_pitch, color=COLORS[1], linestyle="-", marker="o", label="post pitch")
        ax2 = axes[4].twinx()
        ax2.plot(xs, delta, color="#6a4c93", marker="x", linewidth=1.1, label="mean action delta")
        ax2.set_ylabel("mean action delta L2")
        lines, labels_l = axes[4].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[4].legend(lines + lines2, labels_l + labels2, ncol=5, loc="upper left")
    axes[4].set_title("What the object roll/pitch optimizer thought it did per sampled chunk")
    axes[4].set_xlabel("decision")
    axes[4].set_ylabel("mean abs angle [deg]")
    axes[4].grid(True, alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def summarize(data: dict[str, Any], events: list[dict[str, Any]], plot_path: Path) -> dict[str, Any]:
    decisions = data["decisions"]
    labels = data["labels"]
    rel = data["rel_euler_deg"]
    obj_pos = data["obj_pos"]
    eef_pos = data["eef_pos"]
    chunk_rows = data["chunk_rows"]
    rp_chunks = [c for c in chunk_rows if (c["object_rp"] or {}).get("applied")]
    label_events = []
    for event in events:
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])

    last = events[-1] if events else {}
    selected_probs = []
    for chunk in chunk_rows:
        probs = chunk.get("selected_label_probs")
        if probs:
            selected_probs.append(
                {
                    "decision_before": chunk["decision_before"],
                    "target": chunk["target"],
                    "p_can": float(probs[0]),
                    "p_right": float(probs[1]),
                    "p_left": float(probs[2]),
                }
            )
    return {
        "n_events": len(events),
        "last_event_type": last.get("type"),
        "last_decision": int(last.get("decision_idx", last.get("decision_idx_before", -1))),
        "has_rollout_end": any(e.get("type") == "rollout_end" for e in events),
        "reference_decision": int(data["ref_decision"]),
        "reference_source": data["ref_source"],
        "final_label": labels[-1].astype(int).tolist(),
        "label_events": label_events,
        "object_xy_start_m": obj_pos[0, :2].astype(float).tolist(),
        "object_xy_final_m": obj_pos[-1, :2].astype(float).tolist(),
        "object_xy_displacement_m": (obj_pos[-1, :2] - obj_pos[0, :2]).astype(float).tolist(),
        "eef_xy_displacement_m": (eef_pos[-1, :2] - eef_pos[0, :2]).astype(float).tolist(),
        "relative_euler_final_deg": {AXES[i]: float(rel[-1, i]) for i in range(3)},
        "relative_euler_max_abs_deg": {AXES[i]: float(np.nanmax(np.abs(rel[:, i]))) for i in range(3)},
        "n_chunk_samples": len(chunk_rows),
        "n_object_rp_guided_chunks": len(rp_chunks),
        "object_rp_guided_chunks": [
            {
                "decision_before": int(c["decision_before"]),
                "target": c["target"],
                "pre_mean_abs_roll_deg": c["object_rp"].get("pre_mean_abs_roll_deg"),
                "post_mean_abs_roll_deg": c["object_rp"].get("post_mean_abs_roll_deg"),
                "pre_mean_abs_pitch_deg": c["object_rp"].get("pre_mean_abs_pitch_deg"),
                "post_mean_abs_pitch_deg": c["object_rp"].get("post_mean_abs_pitch_deg"),
                "mean_action_delta_l2": c["object_rp"].get("mean_action_delta_l2"),
            }
            for c in rp_chunks
        ],
        "selected_label_probs_by_chunk": selected_probs,
        "plot_png": str(plot_path),
        "plot_pdf": str(plot_path.with_suffix(".pdf")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    configure_matplotlib()
    events = load_events(args.events)
    data = collect(events)
    name = args.name or args.events.parent.parent.parent.name + "_" + args.events.parent.name + "_object_rp_diagnostics"
    plot_path = args.output_dir / f"{name}.png"
    plot(data, plot_path)
    summary = summarize(data, events, plot_path)
    summary_path = args.output_dir / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
