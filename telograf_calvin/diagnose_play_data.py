#!/usr/bin/env python
"""Diagnose paper-STL coverage in CALVIN play data."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from telograf_calvin.paper_specs import (
    FIXED_SAFETY_BOX,
    LABEL_NAMES,
    core_paper_specs,
    deoverlap_windows,
    diagnostic_specs,
    ensure_output_dir,
    hdf5_demo_splits,
    iter_spec_windows,
    load_demo_arrays,
    sorted_demo_keys,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/calvin.hdf5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/telograf/diagnostics"))
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--pre-event-steps", type=int, default=16)
    parser.add_argument("--trainable-threshold", type=int, default=1000)
    parser.add_argument("--pilot-threshold", type=int, default=200)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--no-triples", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def plot_event_counts(path: Path, rows: List[Dict]) -> None:
    labels = [row["label"] for row in rows]
    counts = [row["rising_edges_total"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, counts, color="#2563eb")
    ax.set_ylabel("rising edges")
    ax.set_title("CALVIN paper-label event counts")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_xy_heatmap(path: Path, xy: np.ndarray) -> None:
    box = FIXED_SAFETY_BOX.normalized()
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(xy):
        ax.hist2d(xy[:, 0], xy[:, 1], bins=120, cmap="viridis")
    ax.add_patch(
        Rectangle(
            (box.x_min, box.y_min),
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            facecolor="#ef4444",
            edgecolor="#7f1d1d",
            alpha=0.35,
            linewidth=2,
        )
    )
    ax.set_xlabel("EEF x")
    ax.set_ylabel("EEF y")
    ax.set_title("EEF XY density with fixed paper safety box")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = ensure_output_dir(args.output_root / time.strftime("%Y%m%d_%H%M%S"))
    plot_dir = ensure_output_dir(run_dir / "plots")

    specs = diagnostic_specs(include_triples=not args.no_triples, include_gripper=True)
    core_ids = {spec["id"] for spec in core_paper_specs(include_gripper=True)}

    label_true_counts = Counter()
    label_edge_counts = Counter()
    label_edge_counts_by_split = defaultdict(Counter)
    split_sample_counts = Counter()
    total_samples = 0
    xy_chunks = []

    coverage = {
        spec["id"]: {
            "spec_id": spec["id"],
            "type": spec["type"],
            "formula": spec["formula"],
            "is_core_paper_spec": spec["id"] in core_ids,
            "raw_windows_total": 0,
            "deoverlap_windows_total": 0,
            "train_windows": 0,
            "valid_windows": 0,
            "unknown_windows": 0,
            "source_demos": set(),
        }
        for spec in specs
    }

    with h5py.File(args.dataset, "r") as h5:
        splits = hdf5_demo_splits(h5)
        keys = sorted_demo_keys(h5, max_demos=args.max_demos)
        for demo_i, key in enumerate(keys):
            demo = load_demo_arrays(h5, key, splits.get(key, "unknown"))
            total_samples += demo.length
            split_sample_counts[demo.split] += demo.length
            for idx, label in enumerate(LABEL_NAMES):
                label_true_counts[label] += int(np.sum(demo.labels[:, idx]))
                n_edges = int(len(demo.edges[label]))
                label_edge_counts[label] += n_edges
                label_edge_counts_by_split[demo.split][label] += n_edges
            xy_chunks.append(demo.eef_xy[:: max(1, demo.length // 5000)])

            for spec in specs:
                raw = list(iter_spec_windows(spec, demo, args.horizon, args.pre_event_steps))
                kept = deoverlap_windows(raw, args.horizon)
                row = coverage[spec["id"]]
                row["raw_windows_total"] += len(raw)
                row["deoverlap_windows_total"] += len(kept)
                row[f"{demo.split}_windows"] = row.get(f"{demo.split}_windows", 0) + len(kept)
                if kept:
                    row["source_demos"].add(key)

            print(
                f"[{demo_i + 1:03d}/{len(keys):03d}] {key} split={demo.split} "
                f"T={demo.length} edges={sum(len(v) for v in demo.edges.values())}"
            )

    label_rows = []
    for label in LABEL_NAMES:
        row = {
            "label": label,
            "true_fraction": float(label_true_counts[label] / max(1, total_samples)),
            "rising_edges_total": int(label_edge_counts[label]),
        }
        for split in sorted(split_sample_counts):
            row[f"rising_edges_{split}"] = int(label_edge_counts_by_split[split][label])
        label_rows.append(row)

    coverage_rows = []
    selected_specs = []
    for spec in specs:
        row = coverage[spec["id"]]
        row["source_demo_count"] = len(row.pop("source_demos"))
        train_windows = int(row.get("train_windows", 0))
        if train_windows >= args.trainable_threshold:
            status = "trainable"
        elif train_windows >= args.pilot_threshold:
            status = "pilot"
        else:
            status = "sparse"
        row["status"] = status
        coverage_rows.append(row)
        if status in {"trainable", "pilot"} or row["is_core_paper_spec"]:
            spec_with_status = dict(spec)
            spec_with_status["status"] = status
            spec_with_status["train_windows"] = train_windows
            spec_with_status["valid_windows"] = int(row.get("valid_windows", 0))
            selected_specs.append(spec_with_status)

    coverage_rows.sort(
        key=lambda r: (
            0 if r["is_core_paper_spec"] else 1,
            -int(r["deoverlap_windows_total"]),
            r["spec_id"],
        )
    )

    write_csv(run_dir / "atomic_label_counts.csv", label_rows)
    write_csv(run_dir / "spec_coverage.csv", coverage_rows)

    recommendation = {
        "dataset": str(args.dataset),
        "horizon": int(args.horizon),
        "pre_event_steps": int(args.pre_event_steps),
        "trainable_threshold": int(args.trainable_threshold),
        "pilot_threshold": int(args.pilot_threshold),
        "total_samples": int(total_samples),
        "split_sample_counts": {k: int(v) for k, v in split_sample_counts.items()},
        "fixed_safety_box": asdict(FIXED_SAFETY_BOX),
        "selected_specs": selected_specs,
        "core_paper_spec_ids": sorted(core_ids),
    }
    write_json(run_dir / "recommendation.json", recommendation)

    if not args.skip_plots:
        plot_event_counts(plot_dir / "atomic_event_counts.png", label_rows)
        xy = np.concatenate(xy_chunks, axis=0) if xy_chunks else np.zeros((0, 2), dtype=np.float32)
        plot_xy_heatmap(plot_dir / "eef_xy_heatmap.png", xy)

    print(f"\nwrote diagnostics to {run_dir}")
    print("top paper/spec coverage:")
    for row in coverage_rows[:20]:
        print(
            f"  {row['spec_id']}: status={row['status']} "
            f"train={row.get('train_windows', 0)} valid={row.get('valid_windows', 0)} "
            f"formula={row['formula']}"
        )


if __name__ == "__main__":
    main()
