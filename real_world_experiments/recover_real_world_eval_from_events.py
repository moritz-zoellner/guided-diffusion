#!/usr/bin/env python3
"""Recover real-world rollout summaries/plots from an events.jsonl log.

This is intentionally post-hoc and conservative: if the node was interrupted
before a rollout_end event, the recovered rollout is marked incomplete instead
of inventing a clean termination.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def obs_point(obs: dict[str, Any], decision_idx: int, label: list[int] | None) -> dict[str, Any]:
    return {
        "decision_idx": int(decision_idx),
        "eef_pos": obs.get("eef_pos"),
        "object_pos": obs.get("cheezit_pos"),
        "label": label,
    }


def label_event_text(event: dict[str, Any]) -> str:
    name = event.get("label_name", str(event.get("label_idx", "?")))
    to_value = int(event.get("to", 0))
    return f"{name}:{'on' if to_value else 'off'}"


def signed_distance_to_box(xy: np.ndarray, box: dict[str, float]) -> np.ndarray:
    x_min = float(box["x_min"]) - float(box.get("margin", 0.0))
    x_max = float(box["x_max"]) + float(box.get("margin", 0.0))
    y_min = float(box["y_min"]) - float(box.get("margin", 0.0))
    y_max = float(box["y_max"]) + float(box.get("margin", 0.0))
    x = xy[:, 0]
    y = xy[:, 1]
    dx_out = np.maximum(np.maximum(x_min - x, x - x_max), 0.0)
    dy_out = np.maximum(np.maximum(y_min - y, y - y_max), 0.0)
    outside_dist = np.sqrt(dx_out * dx_out + dy_out * dy_out)
    inside_margin = np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y])
    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return np.where(inside, -inside_margin, outside_dist)


def recover(events_path: Path, overwrite: bool = True) -> dict[str, Any]:
    rollout_dir = events_path.parent
    rollouts_dir = rollout_dir.parent
    run_dir = rollouts_dir.parent
    run_config = load_json(run_dir / "run_config.json")

    events: list[dict[str, Any]] = []
    with events_path.open("r") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    if not events:
        raise ValueError(f"No events in {events_path}")

    trajectory: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    label_events: list[dict[str, Any]] = []
    chain_events: list[dict[str, Any]] = []
    event_type_counts = Counter()
    chunk_count = 0
    rollout_end: dict[str, Any] | None = None
    start_ns = None
    last_ns = None
    last_reached_ns = None
    last_reached_obs = None
    last_reached_label = None
    last_decision_event = None
    label_names = run_config.get("label_names", ["can_grabbed", "pouring_right", "pouring_left"])

    for event in events:
        etype = event.get("type")
        event_type_counts[etype] += 1
        if event.get("t_ns") is not None:
            last_ns = int(event["t_ns"])
        if etype == "rollout_start":
            start_ns = int(event.get("t_ns", 0))
            label_names = event.get("label_names", label_names)
            if event.get("obs"):
                trajectory.append(obs_point(event["obs"], 0, event.get("label") or event.get("current_label")))
        elif etype == "decision":
            last_decision_event = event
            actions.append({
                "decision_idx": event.get("decision_idx"),
                "chunk_idx": event.get("chunk_idx"),
                "remaining_actions_in_chunk": event.get("remaining_actions_in_chunk"),
                "action": event.get("action"),
                "target_pos": event.get("target_pos"),
                "target_pos_tol_m": event.get("target_pos_tol_m"),
                "target_rot_tol_rad": event.get("target_rot_tol_rad"),
                "label": event.get("label") or event.get("current_label"),
                "selection": event.get("selection"),
            })
        elif etype == "target_reached":
            last_reached_ns = int(event.get("t_ns", last_ns or 0))
            last_reached_obs = event.get("reached_obs")
            last_reached_label = event.get("label") or event.get("current_label")
            if last_reached_obs:
                trajectory.append(obs_point(last_reached_obs, int(event.get("decision_idx", 0)), last_reached_label))
            for label_event in event.get("label_events", []) or []:
                label_events.append(label_event)
        elif etype == "chunk_sample":
            chunk_count += 1
        elif etype == "chain_advanced":
            chain_events.append(event)
        elif etype == "rollout_end":
            rollout_end = event

    initial_label = trajectory[0]["label"] if trajectory else None
    final_label = last_reached_label
    if final_label is None and last_decision_event is not None:
        final_label = last_decision_event.get("label") or last_decision_event.get("current_label")

    last_reached_decision_idx = int(trajectory[-1]["decision_idx"]) if trajectory else 0
    last_decision_idx = int(last_decision_event.get("decision_idx", 0)) if last_decision_event else last_reached_decision_idx
    pending_decision = bool(last_decision_idx > last_reached_decision_idx)
    target_chain = run_config.get("target_chain", "")
    target_chain_parsed = run_config.get("target_chain_parsed", [])
    chain_pos = len(chain_events)
    chain_done = bool(target_chain_parsed and chain_pos >= len(target_chain_parsed))

    if rollout_end is not None:
        termination_reason = rollout_end.get("termination_reason", "rollout_end")
        success = bool(rollout_end.get("success", chain_done))
    else:
        termination_reason = "incomplete_log_or_cancelled"
        success = bool(chain_done)

    duration_s = None
    if start_ns is not None and last_ns is not None:
        duration_s = (last_ns - start_ns) / 1e9
    executed_duration_s = None
    if start_ns is not None and last_reached_ns is not None:
        executed_duration_s = (last_reached_ns - start_ns) / 1e9

    summary: dict[str, Any] = {
        "rollout_idx": int((events[0].get("rollout_idx", 0) if events else 0) or 0),
        "success": success,
        "termination_reason": termination_reason,
        "recovered_from_events": True,
        "source_events_path": str(events_path),
        "has_rollout_end_event": rollout_end is not None,
        "duration_s": duration_s,
        "executed_duration_s": executed_duration_s,
        "n_events": len(events),
        "event_type_counts": dict(event_type_counts),
        "n_decisions": last_decision_idx,
        "n_completed_targets": max(0, len(trajectory) - 1),
        "pending_decision_at_end": pending_decision,
        "pending_decision_idx": last_decision_idx if pending_decision else None,
        "last_reached_decision_idx": last_reached_decision_idx,
        "n_chunks_sampled": chunk_count,
        "label_names": label_names,
        "label_event_count": len(label_events),
        "label_events": label_events,
        "label_event_sequence": " -> ".join(label_event_text(e) for e in label_events) or "none",
        "initial_label": initial_label,
        "final_label": final_label,
        "rollout_dir": str(rollout_dir),
        "xy_plot": str(rollout_dir / "xy_plot.png"),
        "target_chain": target_chain,
        "chain_done": chain_done,
        "chain_pos": chain_pos,
        "chain_length": len(target_chain_parsed),
        "chain_events": chain_events,
    }

    if run_config.get("enable_safety_guidance"):
        box = run_config.get("safety_box")
        if box and trajectory:
            eef = np.asarray([p["eef_pos"][:2] for p in trajectory if p.get("eef_pos")], dtype=np.float64)
            signed = signed_distance_to_box(eef, box)
            closest_idx = int(np.argmin(signed))
            summary.update({
                "task_success": success,
                "safety_evaluated": True,
                "safety_success": bool(np.all(signed > 0.0)),
                "safety_inside_count": int(np.sum(signed <= 0.0)),
                "safety_n_points": int(len(signed)),
                "safety_min_signed_distance_m": float(np.min(signed)),
                "safety_closest_xy": eef[closest_idx].tolist(),
                "safety_box": box,
            })

    if overwrite or not (rollout_dir / "rollout_summary.json").exists():
        write_json(rollout_dir / "rollout_summary.json", summary)
    write_trajectory_csv(rollout_dir / "recovered_trajectory.csv", trajectory)
    write_actions_csv(rollout_dir / "recovered_actions.csv", actions)
    write_rollout_plot(rollout_dir / "xy_plot.png", summary, trajectory, run_config)
    write_run_summary(run_dir, [summary], run_config)
    return summary


def write_trajectory_csv(path: Path, trajectory: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["decision_idx", "eef_x", "eef_y", "eef_z", "object_x", "object_y", "object_z", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in trajectory:
            eef = point.get("eef_pos") or [None, None, None]
            obj = point.get("object_pos") or [None, None, None]
            writer.writerow({
                "decision_idx": point.get("decision_idx"),
                "eef_x": eef[0],
                "eef_y": eef[1],
                "eef_z": eef[2],
                "object_x": obj[0],
                "object_y": obj[1],
                "object_z": obj[2],
                "label": json.dumps(point.get("label")),
            })


def write_actions_csv(path: Path, actions: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "decision_idx",
            "chunk_idx",
            "remaining_actions_in_chunk",
            "action",
            "target_pos",
            "target_pos_tol_m",
            "target_rot_tol_rad",
            "label",
            "selection_mode",
            "chain_pos",
            "target_label_name",
            "target_mode",
            "selected_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for action in actions:
            selection = action.get("selection") or {}
            writer.writerow({
                "decision_idx": action.get("decision_idx"),
                "chunk_idx": action.get("chunk_idx"),
                "remaining_actions_in_chunk": action.get("remaining_actions_in_chunk"),
                "action": json.dumps(action.get("action")),
                "target_pos": json.dumps(action.get("target_pos")),
                "target_pos_tol_m": action.get("target_pos_tol_m"),
                "target_rot_tol_rad": action.get("target_rot_tol_rad"),
                "label": json.dumps(action.get("label")),
                "selection_mode": selection.get("mode"),
                "chain_pos": selection.get("chain_pos"),
                "target_label_name": selection.get("target_label_name"),
                "target_mode": selection.get("target_mode"),
                "selected_score": selection.get("selected_score"),
            })


def write_rollout_plot(path: Path, summary: dict[str, Any], trajectory: list[dict[str, Any]], run_config: dict[str, Any]) -> None:
    if not trajectory:
        return
    eef = np.asarray([p["eef_pos"][:2] for p in trajectory if p.get("eef_pos")], dtype=np.float64)
    obj = np.asarray([p["object_pos"][:2] for p in trajectory if p.get("object_pos")], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(eef[:, 0], eef[:, 1], "-o", color="#111111", markersize=2.0, linewidth=1.1, label="EEF")
    if len(obj):
        ax.plot(obj[:, 0], obj[:, 1], "-o", color="#1f77b4", markersize=2.0, linewidth=1.1, label="object")
    for event in summary.get("label_events", []):
        idx = min(max(0, int(event.get("decision_idx", 0))), len(eef) - 1)
        ax.scatter(
            eef[idx, 0],
            eef[idx, 1],
            s=55,
            marker="*",
            label=f"{event.get('label_name')} {event.get('from')}->{event.get('to')}",
        )
    if summary.get("pending_decision_at_end"):
        ax.scatter(eef[-1, 0], eef[-1, 1], s=70, marker="x", color="#d62728", label="cancel/end")
    if run_config.get("enable_safety_guidance") and run_config.get("safety_box"):
        box = run_config["safety_box"]
        x_min = float(box["x_min"])
        x_max = float(box["x_max"])
        y_min = float(box["y_min"])
        y_max = float(box["y_max"])
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, facecolor="#d62728", alpha=0.18, edgecolor="#d62728", label="safety box"))
    ax.set_title(
        f"rollout {summary['rollout_idx']:03d}: {summary['termination_reason']} "
        f"success={summary['success']} chain={summary.get('chain_pos')}/{summary.get('chain_length')}"
    )
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_run_summary(run_dir: Path, summaries: list[dict[str, Any]], run_config: dict[str, Any]) -> None:
    n = len(summaries)
    successes = sum(1 for item in summaries if item.get("success"))
    reason_counts = Counter(item.get("termination_reason") for item in summaries)
    seq_counts = Counter(item.get("label_event_sequence", "none") for item in summaries)
    final_label_counts = Counter(tuple(item.get("final_label") or []) for item in summaries)
    summary = {
        "mode": run_config.get("mode", "recovered_real_world_eval"),
        "run_dir": str(run_dir),
        "recovered_from_events": True,
        "n_rollouts_requested": run_config.get("n_rollouts", n),
        "n_rollouts_completed": n,
        "success_count": successes,
        "success_rate": float(successes / n) if n else 0.0,
        "termination_reason_counts": dict(reason_counts),
        "label_event_sequence_counts": dict(seq_counts),
        "final_label_counts": {str(k): v for k, v in final_label_counts.items()},
        "rollouts": summaries,
    }
    write_json(run_dir / "summary.json", summary)
    with (run_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = [
            "rollout_idx",
            "success",
            "termination_reason",
            "duration_s",
            "executed_duration_s",
            "n_decisions",
            "n_completed_targets",
            "pending_decision_at_end",
            "chain_pos",
            "chain_length",
            "initial_label",
            "final_label",
            "label_event_count",
            "label_event_sequence",
            "rollout_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({
                key: json.dumps(item[key]) if isinstance(item.get(key), (list, dict)) else item.get(key)
                for key in fieldnames
            })
    write_summary_plot(run_dir / "summary.png", summary)


def write_summary_plot(path: Path, summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        ["success", "incomplete/failure"],
        [summary["success_count"], summary["n_rollouts_completed"] - summary["success_count"]],
        color=["#2ca02c", "#d62728"],
    )
    axes[0].set_ylim(0, max(1, summary["n_rollouts_completed"]))
    axes[0].set_title(f"success rate {summary['success_rate']:.2f}")
    seq_counts = summary["label_event_sequence_counts"] or {"none": 0}
    labels = list(seq_counts.keys())
    counts = list(seq_counts.values())
    axes[1].barh(range(len(labels)), counts, color="#1f77b4")
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_title("label event sequences")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events_jsonl", type=Path)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    summary = recover(args.events_jsonl, overwrite=not args.no_overwrite)
    print(json.dumps({
        "rollout_summary": str(args.events_jsonl.parent / "rollout_summary.json"),
        "run_summary": str(args.events_jsonl.parent.parent.parent / "summary.json"),
        "xy_plot": str(args.events_jsonl.parent / "xy_plot.png"),
        "success": summary["success"],
        "termination_reason": summary["termination_reason"],
        "chain": f"{summary.get('chain_pos')}/{summary.get('chain_length')}",
        "n_decisions": summary["n_decisions"],
        "pending_decision_at_end": summary["pending_decision_at_end"],
    }, indent=2))


if __name__ == "__main__":
    main()
