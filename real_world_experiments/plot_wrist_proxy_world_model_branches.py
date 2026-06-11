#!/usr/bin/env python3
"""Compare true wrist angle, EEF-derived wrist proxy, and WM branches."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from real_world_experiments.real_world_data import rot6d_to_matrix
except ModuleNotFoundError:
    from real_world_data import rot6d_to_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    / "automaton_left_epoch160_n10_2/rollouts/rollout_000"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/stl_gpc_sanity"


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


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "rollout_start":
        return event.get("obs")
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") in {"decision", "chunk_sample"}:
        return event.get("obs")
    return None


def selected_prediction_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    pred = event.get("dynamics_prediction")
    if not isinstance(pred, dict):
        return None
    selected = pred.get("selected")
    if not isinstance(selected, dict):
        return None
    if "eef_rot6d" not in selected:
        return None
    return selected


def local_z_twist_deg(current_eef_rot6d: np.ndarray, future_eef_rot6d: np.ndarray) -> np.ndarray:
    current_R = rot6d_to_matrix(np.asarray(current_eef_rot6d, dtype=np.float32)[None])[0]
    future_R = rot6d_to_matrix(np.asarray(future_eef_rot6d, dtype=np.float32))
    rel_R = np.einsum("ij,...jk->...ik", current_R.T, future_R)
    rotvec = Rotation.from_matrix(rel_R.reshape(-1, 3, 3)).as_rotvec().reshape(rel_R.shape[:-2] + (3,))
    return np.degrees(rotvec[..., 2])


def collect_measured_series(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    decisions = []
    wrist_deg = []
    eef_rot6d = []
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
        wrist_deg.append(float(event.get("wrist_joint_delta_deg", np.nan)))
        eef_rot6d.append(obs["eef_rot6d"])
    if not decisions:
        raise ValueError("No measured observations in events")
    return (
        np.asarray(decisions, dtype=np.int32),
        np.asarray(wrist_deg, dtype=np.float64),
        np.asarray(eef_rot6d, dtype=np.float32),
        label_events,
    )


def cumulative_eef_proxy(wrist_deg: np.ndarray, eef_rot6d: np.ndarray) -> np.ndarray:
    proxy = np.empty_like(wrist_deg, dtype=np.float64)
    proxy[0] = wrist_deg[0]
    for idx in range(1, len(proxy)):
        proxy[idx] = proxy[idx - 1] + float(local_z_twist_deg(eef_rot6d[idx - 1], eef_rot6d[idx]))
    return proxy


def one_step_eef_proxy(wrist_deg: np.ndarray, eef_rot6d: np.ndarray) -> np.ndarray:
    proxy = np.empty_like(wrist_deg, dtype=np.float64)
    proxy[0] = wrist_deg[0]
    for idx in range(1, len(proxy)):
        proxy[idx] = wrist_deg[idx - 1] + float(local_z_twist_deg(eef_rot6d[idx - 1], eef_rot6d[idx]))
    return proxy


def absolute_eef_proxy_from_start(wrist_deg: np.ndarray, eef_rot6d: np.ndarray) -> np.ndarray:
    return wrist_deg[0] + local_z_twist_deg(eef_rot6d[0], eef_rot6d)


def collect_world_model_branches(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    branches = []
    for event in events:
        if event.get("type") != "chunk_sample":
            continue
        obs = event.get("obs")
        selected = selected_prediction_from_event(event)
        if obs is None or selected is None:
            continue
        start_decision = int(event.get("decision_idx_before", 0))
        start_wrist = float(event.get("wrist_joint_delta_deg", np.nan))
        pred_eef_rot6d = np.asarray(selected["eef_rot6d"], dtype=np.float32)
        pred_wrist = start_wrist + local_z_twist_deg(obs["eef_rot6d"], pred_eef_rot6d)
        branches.append(
            {
                "chunk_idx": int(event.get("chunk_idx", len(branches))),
                "start_decision": start_decision,
                "start_wrist_deg": start_wrist,
                "target_label_name": (event.get("selection") or {}).get("target_label_name"),
                "x": start_decision + np.arange(1, len(pred_wrist) + 1, dtype=np.int32),
                "pred_wrist_deg": pred_wrist,
            }
        )
    return branches


def branch_error_rows(decisions: np.ndarray, wrist_deg: np.ndarray, branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_by_decision = {int(d): float(w) for d, w in zip(decisions, wrist_deg)}
    rows = []
    for branch in branches:
        for horizon, (decision, pred) in enumerate(zip(branch["x"], branch["pred_wrist_deg"]), start=1):
            true = true_by_decision.get(int(decision))
            if true is None:
                continue
            rows.append(
                {
                    "chunk_idx": int(branch["chunk_idx"]),
                    "start_decision": int(branch["start_decision"]),
                    "target_label_name": branch.get("target_label_name"),
                    "horizon": int(horizon),
                    "decision": int(decision),
                    "pred_wrist_deg": float(pred),
                    "true_wrist_deg": true,
                    "abs_err_deg": abs(float(pred) - true),
                }
            )
    return rows


def summarize_errors(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {
        "n": int(len(values)),
        "mean_abs_err_deg": float(np.mean(np.abs(values))),
        "median_abs_err_deg": float(np.median(np.abs(values))),
        "p90_abs_err_deg": float(np.percentile(np.abs(values), 90)),
        "max_abs_err_deg": float(np.max(np.abs(values))),
    }


def summarize_branch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    err = np.asarray([row["abs_err_deg"] for row in rows], dtype=np.float64)
    by_horizon = {}
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        vals = np.asarray([row["abs_err_deg"] for row in rows if int(row["horizon"]) == horizon], dtype=np.float64)
        by_horizon[str(horizon)] = summarize_errors(vals)
    return {
        "overall": summarize_errors(err),
        "by_horizon": by_horizon,
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: Path,
    decisions: np.ndarray,
    wrist_deg: np.ndarray,
    cumulative_proxy: np.ndarray,
    one_step_proxy: np.ndarray,
    abs_proxy: np.ndarray,
    branches: list[dict[str, Any]],
    label_events: list[dict[str, Any]],
    *,
    xlim: tuple[float, float] | None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.0]})
    ax = axes[0]
    branch_label = False
    for branch in branches:
        x = np.concatenate([[branch["start_decision"]], branch["x"]])
        y = np.concatenate([[branch["start_wrist_deg"]], branch["pred_wrist_deg"]])
        ax.plot(
            x,
            y,
            color="#d000ff",
            alpha=0.32,
            linewidth=1.0,
            label="WM selected chunk proxy" if not branch_label else None,
        )
        branch_label = True
    ax.plot(decisions, wrist_deg, color="#111111", linewidth=1.7, label="true wrist_3 delta")
    ax.plot(decisions, cumulative_proxy, color="#1f77b4", linewidth=1.2, linestyle="--", label="EEF proxy cumulative")
    ax.plot(decisions, one_step_proxy, color="#2ca02c", linewidth=1.0, linestyle=":", label="EEF proxy one-step anchored")
    ax.plot(decisions, abs_proxy, color="#9467bd", linewidth=0.9, alpha=0.7, linestyle="-.", label="EEF proxy from rollout start")
    ax.axhline(90.0, color="#d62728", linestyle="--", linewidth=0.9, label="+90 deg")
    ax.axhline(-90.0, color="#d62728", linestyle="--", linewidth=0.9, label="-90 deg")
    for event in label_events:
        decision = int(event.get("decision_idx", 0))
        if decision < decisions[0] or decision > decisions[-1]:
            continue
        true = float(np.interp(decision, decisions, wrist_deg))
        ax.scatter(decision, true, s=50, marker="*", color="#ffcc00", edgecolor="#111111", linewidth=0.5, zorder=5)
        ax.text(decision, true, f" {event.get('label_name')} {event.get('from')}->{event.get('to')}", fontsize=7, va="center")
    ax.set_ylabel("wrist delta / proxy [deg]")
    ax.set_title("true wrist angle vs EEF-derived proxy and world-model 8-step branches")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)

    err_ax = axes[1]
    err_ax.plot(decisions, cumulative_proxy - wrist_deg, color="#1f77b4", linewidth=1.1, label="cumulative proxy error")
    err_ax.plot(decisions, one_step_proxy - wrist_deg, color="#2ca02c", linewidth=1.0, linestyle=":", label="one-step proxy error")
    err_ax.plot(decisions, abs_proxy - wrist_deg, color="#9467bd", linewidth=0.9, alpha=0.75, linestyle="-.", label="start proxy error")
    err_ax.axhline(0.0, color="#111111", linewidth=0.8)
    err_ax.set_xlabel("decision")
    err_ax.set_ylabel("proxy - true [deg]")
    err_ax.grid(True, alpha=0.25)
    err_ax.legend(loc="best", ncol=3)
    if xlim is not None:
        err_ax.set_xlim(*xlim)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(rollout_dir: Path, output_dir: Path, *, xlim: tuple[float, float] | None) -> dict[str, Any]:
    events = load_events(rollout_dir)
    decisions, wrist_deg, eef_rot6d, label_events = collect_measured_series(events)
    cumulative_proxy = cumulative_eef_proxy(wrist_deg, eef_rot6d)
    one_step_proxy = one_step_eef_proxy(wrist_deg, eef_rot6d)
    abs_proxy = absolute_eef_proxy_from_start(wrist_deg, eef_rot6d)
    branches = collect_world_model_branches(events)
    branch_rows = branch_error_rows(decisions, wrist_deg, branches)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{rollout_dir.parent.parent.name}_{rollout_dir.name}_wrist_proxy_wm_branches"
    if xlim is not None:
        stem += f"_x{int(xlim[0])}_{int(xlim[1])}"
    plot_path = output_dir / f"{stem}.png"
    write_plot(
        plot_path,
        decisions,
        wrist_deg,
        cumulative_proxy,
        one_step_proxy,
        abs_proxy,
        branches,
        label_events,
        xlim=xlim,
    )
    write_rows_csv(output_dir / f"{stem}_branch_errors.csv", branch_rows)
    measured_rows = [
        {
            "decision": int(decision),
            "true_wrist_deg": float(true),
            "eef_proxy_cumulative_deg": float(cum),
            "eef_proxy_one_step_anchored_deg": float(one),
            "eef_proxy_from_start_deg": float(abs_v),
        }
        for decision, true, cum, one, abs_v in zip(decisions, wrist_deg, cumulative_proxy, one_step_proxy, abs_proxy)
    ]
    write_rows_csv(output_dir / f"{stem}_measured_series.csv", measured_rows)
    summary = {
        "rollout_dir": str(rollout_dir),
        "n_measured_points": int(len(decisions)),
        "n_world_model_branches": int(len(branches)),
        "xlim": list(xlim) if xlim is not None else None,
        "measured_proxy_error": {
            "cumulative_eef_proxy_minus_true": summarize_errors(cumulative_proxy - wrist_deg),
            "one_step_eef_proxy_minus_true": summarize_errors(one_step_proxy - wrist_deg),
            "absolute_from_start_eef_proxy_minus_true": summarize_errors(abs_proxy - wrist_deg),
        },
        "world_model_branch_error": summarize_branch_rows(branch_rows),
        "plot_png": str(plot_path),
        "plot_pdf": str(plot_path.with_suffix(".pdf")),
        "branch_errors_csv": str(output_dir / f"{stem}_branch_errors.csv"),
        "measured_series_csv": str(output_dir / f"{stem}_measured_series.csv"),
        "method_note": "WM branches use start true wrist_joint_delta_deg plus local-z EEF twist predicted by dynamics eef_rot6d.",
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    args = parser.parse_args()
    configure_matplotlib()
    xlim = tuple(args.xlim) if args.xlim is not None else None
    summary = analyze(args.rollout_dir, args.output_dir, xlim=xlim)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
