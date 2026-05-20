#!/usr/bin/env python
"""Diagnose Toy Squares STL coverage for a TeLoGraF-style baseline."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from telograf_toy_squares.toy_specs import (  # noqa: E402
    DEFAULT_CHAIN_BASE,
    DEFAULT_RADIUS,
    LABEL_NAMES,
    deoverlap_windows,
    ensure_output_dir,
    hdf5_demo_splits,
    iter_spec_windows,
    label_edges,
    load_demo_arrays,
    sorted_demo_keys,
    toy_paper_specs,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/telograf/toy_squares/diagnostics"))
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--horizon-candidates", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    parser.add_argument("--pre-event-steps", type=int, default=16)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--chain-base", type=str, default=",".join(DEFAULT_CHAIN_BASE))
    parser.add_argument("--max-chain-horizon", type=int, default=5)
    parser.add_argument("--trainable-threshold", type=int, default=1000)
    parser.add_argument("--pilot-threshold", type=int, default=200)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--no-padding", action="store_true", help="Require every mined window to have full real length.")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def parse_chain_base(value: str) -> List[str]:
    labels = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    unknown = [label for label in labels if label not in LABEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown chain labels: {unknown}. Valid labels: {list(LABEL_NAMES)}")
    return labels


def summarize_lengths(lengths: np.ndarray, horizon_candidates: List[int]) -> Dict:
    if len(lengths) == 0:
        return {"count": 0}
    out = {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "median": float(np.percentile(lengths, 50)),
        "p75": float(np.percentile(lengths, 75)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(np.max(lengths)),
    }
    for horizon in horizon_candidates:
        out[f"count_ge_h{horizon}"] = int(np.sum(lengths >= int(horizon)))
        out[f"fraction_ge_h{horizon}"] = float(np.mean(lengths >= int(horizon)))
    return out


def plot_lengths(path: Path, lengths: np.ndarray, horizon_candidates: List[int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=min(40, max(5, len(np.unique(lengths)))), color="#2563eb", alpha=0.85)
    for horizon in horizon_candidates:
        ax.axvline(horizon, color="#ef4444", linewidth=1.0, alpha=0.55)
        ax.text(horizon, ax.get_ylim()[1] * 0.92, f"H={horizon}", rotation=90, va="top", fontsize=8)
    ax.set_xlabel("demo transitions")
    ax.set_ylabel("count")
    ax.set_title("Toy Squares successful demo lengths")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_xy(path: Path, traces: List[np.ndarray], max_items: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for states in traces[:max_items]:
        ax.plot(states[:, 0], states[:, 1], color="#2563eb", alpha=0.18, linewidth=0.9)
        blocks = states[0, 2:10].reshape(4, 2)
        ax.scatter(blocks[:, 0], blocks[:, 1], s=18, color=["#1e88ff", "#ff4136", "#34c759", "#ffd60a"], edgecolor="black", linewidth=0.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_title("Toy Squares agent trajectories and block starts")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def status_for(train_windows: int, trainable_threshold: int, pilot_threshold: int) -> str:
    if train_windows >= trainable_threshold:
        return "trainable"
    if train_windows >= pilot_threshold:
        return "pilot"
    return "sparse"


def main() -> None:
    args = parse_args()
    run_dir = ensure_output_dir(args.output_root / time.strftime("%Y%m%d_%H%M%S"))
    plot_dir = ensure_output_dir(run_dir / "plots")
    chain_base = parse_chain_base(args.chain_base)
    specs = toy_paper_specs(chain_base=chain_base, max_chain_horizon=args.max_chain_horizon, radius=args.radius)

    label_true_counts = Counter()
    label_edge_counts = Counter()
    label_edge_counts_by_split = defaultdict(Counter)
    split_sample_counts = Counter()
    target_counts = Counter()
    lengths = []
    traces = []
    all_windows_by_horizon: Dict[int, Dict[str, List[Dict]]] = {
        int(h): {spec["id"]: [] for spec in specs} for h in args.horizon_candidates
    }

    with h5py.File(args.dataset, "r") as h5:
        splits = hdf5_demo_splits(h5)
        keys = sorted_demo_keys(h5, max_demos=args.max_demos)
        for demo_i, key in enumerate(keys):
            demo = load_demo_arrays(h5, key, splits.get(key, "unknown"), radius=args.radius)
            lengths.append(demo.length)
            split_sample_counts[demo.split] += demo.length
            if demo.target_label is not None:
                target_counts[demo.target_label] += 1
            edges = label_edges(demo.labels)
            for idx, label in enumerate(LABEL_NAMES):
                label_true_counts[label] += int(np.sum(demo.labels[:, idx]))
                label_edge_counts[label] += int(len(edges[label]))
                label_edge_counts_by_split[demo.split][label] += int(len(edges[label]))
            if len(traces) < 200:
                traces.append(demo.state_seq)

            for horizon in args.horizon_candidates:
                for spec in specs:
                    raw = list(
                        iter_spec_windows(
                            spec,
                            demo,
                            int(horizon),
                            args.pre_event_steps,
                            allow_padding=not args.no_padding,
                        )
                    )
                    all_windows_by_horizon[int(horizon)][spec["id"]].extend(raw)

            print(
                f"[{demo_i + 1:04d}/{len(keys):04d}] {key} split={demo.split} "
                f"T={demo.length} target={demo.target_label}"
            )

    lengths_np = np.asarray(lengths, dtype=np.int64)
    length_summary = summarize_lengths(lengths_np, [int(h) for h in args.horizon_candidates])

    label_rows = []
    total_label_steps = max(1, int(sum(length + 1 for length in lengths)))
    for label in LABEL_NAMES:
        row = {
            "label": label,
            "true_fraction": float(label_true_counts[label] / total_label_steps),
            "rising_edges_total": int(label_edge_counts[label]),
            "target_demos": int(target_counts[label]),
        }
        for split in sorted(split_sample_counts):
            row[f"rising_edges_{split}"] = int(label_edge_counts_by_split[split][label])
        label_rows.append(row)

    coverage_rows = []
    recommendation_by_horizon = {}
    for horizon in args.horizon_candidates:
        selected_specs = []
        for spec in specs:
            raw = all_windows_by_horizon[int(horizon)][spec["id"]]
            kept = deoverlap_windows(raw, int(horizon))
            train_windows = int(sum(item["split"] == "train" for item in kept))
            valid_windows = int(sum(item["split"] == "valid" for item in kept))
            padded = [int(item.get("padded_steps", 0)) for item in kept]
            row = {
                "horizon": int(horizon),
                "spec_id": spec["id"],
                "type": spec["type"],
                "formula": spec["formula"],
                "raw_windows_total": int(len(raw)),
                "deoverlap_windows_total": int(len(kept)),
                "train_windows": train_windows,
                "valid_windows": valid_windows,
                "mean_padded_steps": float(np.mean(padded)) if padded else 0.0,
                "max_padded_steps": int(max(padded)) if padded else 0,
                "source_demo_count": int(len({item["demo_key"] for item in kept})),
                "status": status_for(train_windows, args.trainable_threshold, args.pilot_threshold),
            }
            coverage_rows.append(row)
            if row["status"] in {"trainable", "pilot"} or spec["id"].startswith("paper_") or spec["type"] == "eventual":
                spec_out = dict(spec)
                spec_out["status"] = row["status"]
                spec_out["train_windows"] = train_windows
                spec_out["valid_windows"] = valid_windows
                spec_out["mean_padded_steps"] = row["mean_padded_steps"]
                selected_specs.append(spec_out)
        recommendation_by_horizon[str(horizon)] = selected_specs

    coverage_rows.sort(key=lambda row: (int(row["horizon"]), -int(row["train_windows"]), row["spec_id"]))
    write_csv(run_dir / "atomic_label_counts.csv", label_rows)
    write_csv(run_dir / "spec_coverage.csv", coverage_rows)

    selected_for_requested = recommendation_by_horizon.get(str(args.horizon), [])
    recommendation = {
        "dataset": str(args.dataset),
        "horizon": int(args.horizon),
        "horizon_candidates": [int(h) for h in args.horizon_candidates],
        "pre_event_steps": int(args.pre_event_steps),
        "radius": float(args.radius),
        "chain_base": chain_base,
        "max_chain_horizon": int(args.max_chain_horizon),
        "allow_padding": not args.no_padding,
        "trainable_threshold": int(args.trainable_threshold),
        "pilot_threshold": int(args.pilot_threshold),
        "demo_count": int(len(lengths)),
        "split_sample_counts": {k: int(v) for k, v in split_sample_counts.items()},
        "target_demo_counts": {k: int(v) for k, v in target_counts.items()},
        "length_summary": length_summary,
        "selected_specs": selected_for_requested,
        "selected_specs_by_horizon": recommendation_by_horizon,
    }
    write_json(run_dir / "recommendation.json", recommendation)

    if not args.skip_plots:
        plot_lengths(plot_dir / "demo_lengths.png", lengths_np, [int(h) for h in args.horizon_candidates])
        plot_xy(plot_dir / "xy_traces.png", traces)

    print(f"\nwrote diagnostics to {run_dir}")
    print("length summary:", length_summary)
    print(f"coverage for requested H={args.horizon}:")
    for row in [r for r in coverage_rows if int(r["horizon"]) == int(args.horizon)]:
        print(
            f"  {row['spec_id']}: status={row['status']} train={row['train_windows']} "
            f"valid={row['valid_windows']} mean_pad={row['mean_padded_steps']:.1f} {row['formula']}"
        )


if __name__ == "__main__":
    main()
