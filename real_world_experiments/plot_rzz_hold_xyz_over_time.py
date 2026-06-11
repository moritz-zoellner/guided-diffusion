#!/usr/bin/env python3
"""Plot rzz, EEF x, and EEF y over decisions-after-grasp for selected rollouts."""

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
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def obs_from_event(event: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
    if event.get("type") == "rollout_start":
        return event.get("obs"), int(event.get("decision_idx", 0))
    if event.get("type") == "target_reached":
        return event.get("reached_obs"), int(event.get("decision_idx", 0))
    return None, None


def collect(path: Path, name: str) -> dict[str, Any]:
    rows = []
    label_events = []
    rollout_end = None
    guidance_logs = []
    gate_x = None
    target_rzz = None

    for event in load_jsonl(path):
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        if event.get("type") == "rollout_end":
            rollout_end = event
        if event.get("type") == "chunk_sample":
            log = ((event.get("selection") or {}).get("object_rzz_hold_guidance") or {})
            if log:
                guidance_logs.append({"decision_before": int(event.get("decision_idx_before", 0)), **log})
                if gate_x is None and log.get("until_eef_x") is not None:
                    gate_x = float(log["until_eef_x"])
                if target_rzz is None and log.get("target_rzz") is not None:
                    target_rzz = float(log["target_rzz"])

        obs, decision = obs_from_event(event)
        if obs is None or "eef_pos" not in obs or "cheezit_rot6d" not in obs:
            continue
        matrix = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float64)[None])[0]
        rows.append(
            {
                "decision": int(decision),
                "eef": np.asarray(obs["eef_pos"], dtype=np.float64),
                "rzz": float(matrix[2, 2]),
            }
        )

    if not rows:
        raise ValueError(f"No usable observations in {path}")

    grab_decision = None
    pour_decision = None
    for event in label_events:
        if event.get("label_name") == "can_grabbed" and int(event.get("to", -1)) == 1:
            grab_decision = int(event["decision_idx"])
        if event.get("label_name") == "pouring_left" and int(event.get("to", -1)) == 1:
            pour_decision = int(event["decision_idx"])
    if grab_decision is None:
        grab_decision = rows[0]["decision"]

    decisions = np.asarray([row["decision"] for row in rows], dtype=np.int32)
    eef = np.asarray([row["eef"] for row in rows], dtype=np.float64)
    rzz = np.asarray([row["rzz"] for row in rows], dtype=np.float64)
    rel = decisions - int(grab_decision)
    grab_idx = int(np.argmin(np.abs(decisions - int(grab_decision))))
    rzz_ref = float(rzz[grab_idx])

    return {
        "name": name,
        "path": str(path),
        "success": None if rollout_end is None else bool(rollout_end.get("success")),
        "termination_reason": None if rollout_end is None else rollout_end.get("reason"),
        "decisions": decisions,
        "relative_decision": rel,
        "eef": eef,
        "eef_x": eef[:, 0],
        "eef_y": eef[:, 1],
        "rzz": rzz,
        "upright_angle_deg": np.degrees(np.arccos(np.clip(rzz, -1.0, 1.0))),
        "rzz_ref": rzz_ref,
        "rzz_error": np.abs(rzz - rzz_ref),
        "target_rzz": target_rzz,
        "target_rzz_error": None if target_rzz is None else np.abs(rzz - float(target_rzz)),
        "grab_decision": int(grab_decision),
        "pour_decision": None if pour_decision is None else int(pour_decision),
        "gate_x": gate_x,
        "guidance_logs": guidance_logs,
    }


def first_crossing(item: dict[str, Any], key: str, threshold: float, direction: str) -> dict[str, Any] | None:
    vals = np.asarray(item[key], dtype=np.float64)
    for idx in np.where(item["relative_decision"] >= 0)[0]:
        hit = vals[idx] <= threshold if direction == "le" else vals[idx] >= threshold
        if hit:
            return {
                "decision": int(item["decisions"][idx]),
                "relative_decision": int(item["relative_decision"][idx]),
                "eef_x": float(item["eef_x"][idx]),
                "eef_y": float(item["eef_y"][idx]),
                "rzz": float(item["rzz"][idx]),
                "rzz_error": float(item["rzz_error"][idx]),
                "target_rzz_error": (
                    None
                    if item.get("target_rzz_error") is None
                    else float(item["target_rzz_error"][idx])
                ),
            }
    return None


def summarize(item: dict[str, Any], common_gate_x: float | None) -> dict[str, Any]:
    rel = item["relative_decision"]
    after_grasp = rel >= 0
    pour_rel = None if item["pour_decision"] is None else int(item["pour_decision"] - item["grab_decision"])
    gate = None if item["gate_x"] is None else first_crossing(item, "eef_x", float(item["gate_x"]), "le")
    common_gate = None if common_gate_x is None else first_crossing(item, "eef_x", float(common_gate_x), "le")
    active = after_grasp.copy()
    if item["gate_x"] is not None:
        active &= item["eef_x"] > float(item["gate_x"])
    elif common_gate_x is not None:
        active &= item["eef_x"] > float(common_gate_x)
    target_err = item.get("target_rzz_error")
    applied = [log for log in item["guidance_logs"] if log.get("applied")]
    return {
        "name": item["name"],
        "events": item["path"],
        "success": item["success"],
        "grab_decision": item["grab_decision"],
        "pour_decision": item["pour_decision"],
        "pour_relative_decision": pour_rel,
        "gate_x": item["gate_x"],
        "gate_crossing": gate,
        "common_gate_x": common_gate_x,
        "common_gate_crossing": common_gate,
        "rzz_at_grasp": item["rzz_ref"],
        "target_rzz": item.get("target_rzz"),
        "max_abs_rzz_error_before_gate": None if not np.any(active) else float(np.max(item["rzz_error"][active])),
        "mean_abs_rzz_error_before_gate": None if not np.any(active) else float(np.mean(item["rzz_error"][active])),
        "max_abs_target_rzz_error_before_gate": (
            None if target_err is None or not np.any(active) else float(np.max(target_err[active]))
        ),
        "mean_abs_target_rzz_error_before_gate": (
            None if target_err is None or not np.any(active) else float(np.mean(target_err[active]))
        ),
        "final_eef_x": float(item["eef_x"][-1]),
        "final_eef_y": float(item["eef_y"][-1]),
        "final_rzz": float(item["rzz"][-1]),
        "final_upright_angle_deg": float(item["upright_angle_deg"][-1]),
        "guidance_applied_chunks": len(applied),
        "guidance_applied_decisions_before": [int(log["decision_before"]) for log in applied],
    }


def plot(
    items: list[dict[str, Any]],
    out_path: Path,
    xlim: tuple[float, float] | None,
    common_gate_x: float | None,
    angle_ylim: tuple[float, float] | None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = ["#222222", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    panels = [
        ("upright_angle_deg", "object upright angle [deg]"),
        ("eef_x", "EEF world x [m]"),
        ("eef_y", "EEF world y [m]"),
    ]
    for ax, (key, ylabel) in zip(axes, panels):
        for idx, item in enumerate(items):
            color = colors[idx % len(colors)]
            ax.plot(item["relative_decision"], item[key], color=color, linewidth=1.8, label=item["name"])
            gate_x = common_gate_x if common_gate_x is not None else item["gate_x"]
            gate = None if gate_x is None else first_crossing(item, "eef_x", float(gate_x), "le")
            if gate is not None:
                ax.axvline(gate["relative_decision"], color=color, linestyle=":", linewidth=1.4, alpha=0.8)
                gate_idx = int(np.argmin(np.abs(item["relative_decision"] - gate["relative_decision"])))
                ax.scatter(
                    item["relative_decision"][gate_idx],
                    item[key][gate_idx],
                    color=color,
                    marker="o",
                    s=34,
                    zorder=4,
                )
            if item["pour_decision"] is not None:
                ax.axvline(
                    item["pour_decision"] - item["grab_decision"],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                )
            if key == "eef_x" and item["gate_x"] is not None:
                ax.axhline(float(item["gate_x"]), color=color, linestyle=":", linewidth=1.0, alpha=0.8)
            if key == "eef_x" and common_gate_x is not None:
                ax.axhline(float(common_gate_x), color="#d62728", linestyle=":", linewidth=1.1, alpha=0.7)
            if key == "upright_angle_deg":
                ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axvline(0, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    if angle_ylim is not None:
        axes[0].set_ylim(*angle_ylim)
    title = "upright-angle comparison aligned at grasp; dashed = pour, dotted/circle = x gate"
    if common_gate_x is not None:
        title += f" ({common_gate_x:.2f} m)"
    axes[0].set_title(title)
    axes[-1].set_xlabel("decisions after grasp")
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_trace_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.parent.parent.parent.name + "/" + path.parent.name, path
    name, path = raw.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trace", action="append", required=True, help="name=/path/to/events.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default="rzz_hold_xyz_over_time")
    parser.add_argument("--xlim", type=float, nargs=2, default=None)
    parser.add_argument("--common-gate-x", type=float, default=None)
    parser.add_argument("--angle-ylim", type=float, nargs=2, default=None)
    args = parser.parse_args()

    items = [collect(path, name) for name, path in map(parse_trace_arg, args.trace)]
    out_path = args.output_dir / f"{args.name}.png"
    plot(
        items,
        out_path,
        None if args.xlim is None else tuple(args.xlim),
        args.common_gate_x,
        None if args.angle_ylim is None else tuple(args.angle_ylim),
    )
    summary = {
        "plot_png": str(out_path),
        "plot_pdf": str(out_path.with_suffix(".pdf")),
        "common_gate_x": args.common_gate_x,
        "traces": [summarize(item, args.common_gate_x) for item in items],
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
