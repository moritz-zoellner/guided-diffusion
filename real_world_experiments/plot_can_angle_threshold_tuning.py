#!/usr/bin/env python3
"""Plot can tilt/rzz traces for choosing an EEF-x guidance gate."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/cheezit_angle_guidance_tuning"
DEFAULT_BASELINE_DIRS = [
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_1",
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_2",
    REPO_ROOT / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3",
]
DEFAULT_HIGHLIGHT = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_left_rzz_hold_xminus060_epoch160_n1/rollouts/rollout_000/events.jsonl"
)


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


def collect_events(path: Path, group: str) -> dict[str, Any] | None:
    events = load_jsonl(path)
    rows = []
    label_events = []
    rollout_end = None
    for event in events:
        if event.get("type") == "target_reached":
            label_events.extend(event.get("label_events", []) or [])
        if event.get("type") == "rollout_end":
            rollout_end = event
        obs, decision = obs_from_event(event)
        if obs is None or "cheezit_rot6d" not in obs:
            continue
        matrix = rot6d_to_matrix(np.asarray(obs["cheezit_rot6d"], dtype=np.float64)[None])[0]
        rzz = float(matrix[2, 2])
        rows.append(
            {
                "decision": int(decision),
                "label": np.asarray(event.get("current_label") or event.get("label") or [0, 0, 0], dtype=int),
                "eef": np.asarray(obs["eef_pos"], dtype=np.float64),
                "obj": np.asarray(obs["cheezit_pos"], dtype=np.float64),
                "rzz": rzz,
                "tilt_deg": math.degrees(math.acos(np.clip(rzz, -1.0, 1.0))),
            }
        )
    if not rows:
        return None
    success = None if rollout_end is None else bool(rollout_end.get("success"))
    if success is False:
        return None
    grab_decision = None
    pour_decision = None
    for event in label_events:
        if event.get("label_name") == "can_grabbed" and int(event.get("to", -1)) == 1:
            grab_decision = int(event["decision_idx"])
        if event.get("label_name") == "pouring_left" and int(event.get("to", -1)) == 1:
            pour_decision = int(event["decision_idx"])
    if grab_decision is None:
        grabbed = [row for row in rows if row["label"][0] > 0]
        if not grabbed:
            return None
        grab_decision = int(grabbed[0]["decision"])
    grab_idx = next((i for i, row in enumerate(rows) if row["decision"] >= grab_decision), 0)
    decisions = np.asarray([row["decision"] for row in rows], dtype=np.int32)
    eef = np.asarray([row["eef"] for row in rows], dtype=np.float64)
    obj = np.asarray([row["obj"] for row in rows], dtype=np.float64)
    rzz = np.asarray([row["rzz"] for row in rows], dtype=np.float64)
    tilt_deg = np.asarray([row["tilt_deg"] for row in rows], dtype=np.float64)
    return {
        "path": str(path),
        "group": group,
        "name": f"{path.parent.parent.name}/{path.parent.name}",
        "success": success,
        "label_events": label_events,
        "grab_decision": int(grab_decision),
        "pour_decision": None if pour_decision is None else int(pour_decision),
        "relative_decision": decisions - int(grab_decision),
        "decisions": decisions,
        "eef": eef,
        "obj": obj,
        "rzz": rzz,
        "rzz_at_grasp": float(rzz[grab_idx]),
        "rzz_error": np.abs(rzz - float(rzz[grab_idx])),
        "tilt_deg": tilt_deg,
        "tilt_at_grasp_deg": float(tilt_deg[grab_idx]),
    }


def discover_events(dirs: list[Path]) -> list[Path]:
    out = []
    for root in dirs:
        out.extend(sorted(root.glob("rollouts/rollout_*/events.jsonl")))
    return out


def percentile_band(traces: list[dict[str, Any]], x_grid: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = []
    for item in traces:
        x = item["relative_decision"]
        y = item[key]
        order = np.argsort(x)
        xu = x[order]
        yu = y[order]
        keep = (x_grid >= xu[0]) & (x_grid <= xu[-1])
        arr = np.full_like(x_grid, np.nan, dtype=np.float64)
        arr[keep] = np.interp(x_grid[keep], xu, yu)
        vals.append(arr)
    stack = np.asarray(vals, dtype=np.float64)
    return (
        np.nanpercentile(stack, 10, axis=0),
        np.nanpercentile(stack, 50, axis=0),
        np.nanpercentile(stack, 90, axis=0),
    )


def plot(
    traces: list[dict[str, Any]],
    highlight: dict[str, Any] | None,
    out_path: Path,
    gate_xs: list[float],
    *,
    time_xlim: tuple[float, float] | None = None,
    world_x_xlim: tuple[float, float] | None = None,
    rzz_error_ylim: tuple[float, float] | None = None,
) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_tilt, ax_rzz_time, ax_x_rzz, ax_x_err = axes.flat
    baseline = traces
    max_rel = max(int(np.nanmax(t["relative_decision"])) for t in baseline + ([highlight] if highlight else []))
    x_grid = np.arange(min(-5, min(int(np.nanmin(t["relative_decision"])) for t in baseline)), max_rel + 1)

    for item in baseline:
        ax_tilt.plot(item["relative_decision"], item["tilt_deg"], color="#9a9a9a", alpha=0.35, linewidth=1.0)
        ax_rzz_time.plot(item["relative_decision"], item["rzz_error"], color="#9a9a9a", alpha=0.35, linewidth=1.0)
        ax_x_rzz.plot(item["eef"][:, 0], item["rzz"], color="#9a9a9a", alpha=0.35, linewidth=1.0)
        ax_x_err.plot(item["eef"][:, 0], item["rzz_error"], color="#9a9a9a", alpha=0.35, linewidth=1.0)
        if item["pour_decision"] is not None:
            rel_pour = item["pour_decision"] - item["grab_decision"]
            ax_tilt.axvline(rel_pour, color="#bdbdbd", alpha=0.15, linewidth=0.8)
            ax_rzz_time.axvline(rel_pour, color="#bdbdbd", alpha=0.15, linewidth=0.8)

    for key, ax in (("tilt_deg", ax_tilt), ("rzz_error", ax_rzz_time)):
        lo, med, hi = percentile_band(baseline, x_grid, key)
        ax.fill_between(x_grid, lo, hi, color="#7f7f7f", alpha=0.18, label="baseline p10-p90")
        ax.plot(x_grid, med, color="#111111", linewidth=2.0, label="baseline median")

    if highlight is not None:
        ax_tilt.plot(highlight["relative_decision"], highlight["tilt_deg"], color="#1f77b4", linewidth=2.0, label="rzz hold")
        ax_rzz_time.plot(highlight["relative_decision"], highlight["rzz_error"], color="#1f77b4", linewidth=2.0, label="rzz hold")
        ax_x_rzz.plot(highlight["eef"][:, 0], highlight["rzz"], color="#1f77b4", linewidth=2.0, label="rzz hold")
        ax_x_err.plot(highlight["eef"][:, 0], highlight["rzz_error"], color="#1f77b4", linewidth=2.0, label="rzz hold")
        if highlight["pour_decision"] is not None:
            rel_pour = highlight["pour_decision"] - highlight["grab_decision"]
            ax_tilt.axvline(rel_pour, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.8)
            ax_rzz_time.axvline(rel_pour, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.8)

    for gate_x in gate_xs:
        ax_x_rzz.axvline(gate_x, color="#d62728", linestyle=":", linewidth=1.1)
        ax_x_err.axvline(gate_x, color="#d62728", linestyle=":", linewidth=1.1)
    for err in (0.01, 0.02, 0.03, 0.05):
        ax_rzz_time.axhline(err, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.45)
        ax_x_err.axhline(err, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.45)

    ax_tilt.set_title("can tilt angle over time")
    ax_tilt.set_xlabel("decisions after grasp")
    ax_tilt.set_ylabel("tilt angle arccos(rzz) [deg]")
    if time_xlim is not None:
        ax_tilt.set_xlim(*time_xlim)
    ax_tilt.grid(True, alpha=0.25)
    ax_tilt.legend()

    ax_rzz_time.set_title("|rzz - rzz_at_grasp| over time")
    ax_rzz_time.set_xlabel("decisions after grasp")
    ax_rzz_time.set_ylabel("absolute rzz error")
    if time_xlim is not None:
        ax_rzz_time.set_xlim(*time_xlim)
    if rzz_error_ylim is not None:
        ax_rzz_time.set_ylim(*rzz_error_ylim)
    ax_rzz_time.grid(True, alpha=0.25)
    ax_rzz_time.legend()

    ax_x_rzz.set_title("EEF world x vs rzz")
    ax_x_rzz.set_xlabel("EEF world x [m]")
    ax_x_rzz.set_ylabel("rzz = world z dot object z")
    if world_x_xlim is not None:
        ax_x_rzz.set_xlim(*world_x_xlim)
    ax_x_rzz.grid(True, alpha=0.25)
    ax_x_rzz.legend(["baseline rollouts", "rzz hold"] if highlight is not None else ["baseline rollouts"], loc="best")

    ax_x_err.set_title("EEF world x vs |rzz - rzz_at_grasp|")
    ax_x_err.set_xlabel("EEF world x [m]")
    ax_x_err.set_ylabel("absolute rzz error")
    if world_x_xlim is not None:
        ax_x_err.set_xlim(*world_x_xlim)
    if rzz_error_ylim is not None:
        ax_x_err.set_ylim(*rzz_error_ylim)
    ax_x_err.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

    rows = []
    for threshold in (0.01, 0.02, 0.03, 0.05):
        xs = []
        ys = []
        rels = []
        for item in baseline:
            after = np.where((item["relative_decision"] >= 0) & (item["rzz_error"] >= threshold))[0]
            if len(after) == 0:
                continue
            idx = int(after[0])
            xs.append(float(item["eef"][idx, 0]))
            ys.append(float(item["eef"][idx, 1]))
            rels.append(int(item["relative_decision"][idx]))
        if xs:
            rows.append(
                {
                    "rzz_error_threshold": threshold,
                    "n": len(xs),
                    "eef_x_p10_p50_p90": np.percentile(xs, [10, 50, 90]).astype(float).tolist(),
                    "eef_y_p10_p50_p90": np.percentile(ys, [10, 50, 90]).astype(float).tolist(),
                    "relative_decision_p10_p50_p90": np.percentile(rels, [10, 50, 90]).astype(float).tolist(),
                }
            )

    pour_xs = []
    pour_rels = []
    for item in baseline:
        if item["pour_decision"] is None:
            continue
        idx = int(np.argmin(np.abs(item["decisions"] - item["pour_decision"])))
        pour_xs.append(float(item["eef"][idx, 0]))
        pour_rels.append(int(item["pour_decision"] - item["grab_decision"]))
    return {
        "plot_png": str(out_path),
        "plot_pdf": str(out_path.with_suffix(".pdf")),
        "n_baseline_rollouts": len(baseline),
        "highlight": None if highlight is None else highlight["path"],
        "gate_xs": gate_xs,
        "first_rzz_error_crossing_stats": rows,
        "pour_eef_x_p10_p50_p90": np.percentile(pour_xs, [10, 50, 90]).astype(float).tolist() if pour_xs else None,
        "pour_relative_decision_p10_p50_p90": np.percentile(pour_rels, [10, 50, 90]).astype(float).tolist() if pour_rels else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, action="append", default=None)
    parser.add_argument("--highlight-events", type=Path, default=DEFAULT_HIGHLIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default="left_baselines_can_angle_rzz_threshold_tuning")
    parser.add_argument("--gate-x", type=float, action="append", default=None)
    parser.add_argument("--time-xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--world-xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--rzz-error-ylim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    baseline_dirs = args.baseline_dir if args.baseline_dir else DEFAULT_BASELINE_DIRS
    traces = []
    for path in discover_events(baseline_dirs):
        item = collect_events(path, "baseline")
        if item is not None:
            traces.append(item)
    if not traces:
        raise ValueError("No successful baseline traces found")
    highlight = collect_events(args.highlight_events, "highlight") if args.highlight_events and args.highlight_events.exists() else None
    out_path = args.output_dir / f"{args.name}.png"
    gate_x = args.gate_x if args.gate_x is not None else [-0.60, -0.63]
    summary = plot(
        traces,
        highlight,
        out_path,
        gate_x,
        time_xlim=tuple(args.time_xlim) if args.time_xlim is not None else None,
        world_x_xlim=tuple(args.world_xlim) if args.world_xlim is not None else None,
        rzz_error_ylim=tuple(args.rzz_error_ylim) if args.rzz_error_ylim is not None else None,
    )
    summary["baseline_dirs"] = [str(path) for path in baseline_dirs]
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
