#!/usr/bin/env python
"""Mine TeLoGraF-CALVIN training records from CALVIN play data."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import h5py
import numpy as np

from telograf_calvin.paper_specs import (
    ACTION_DIM,
    STATE_DIM,
    core_paper_specs,
    deoverlap_windows,
    diagnostic_specs,
    ensure_output_dir,
    hdf5_demo_splits,
    iter_spec_windows,
    load_demo_arrays,
    sorted_demo_keys,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/calvin.hdf5"))
    parser.add_argument("--recommendation", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/telograf/datasets"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--pre-event-steps", type=int, default=None)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--max-per-spec", type=int, default=5000)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--include-sparse-core", action="store_true")
    parser.add_argument("--pilot-only", action="store_true", help="Drop trainable extras; keep pilot/core specs only.")
    parser.add_argument("--eventual-only", action="store_true", help="Mine only atomic F label specs.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_specs(args: argparse.Namespace) -> tuple[List[Dict], int, int, Dict]:
    meta: Dict = {}
    if args.recommendation is None:
        horizon = 128 if args.horizon is None else int(args.horizon)
        pre_event_steps = 16 if args.pre_event_steps is None else int(args.pre_event_steps)
        if args.eventual_only:
            specs = [spec for spec in diagnostic_specs(include_triples=False, include_gripper=False) if spec["type"] == "eventual"]
            meta["spec_source"] = "diagnostic_specs_eventual_only"
        else:
            specs = core_paper_specs(include_gripper=True)
            meta["spec_source"] = "core_paper_specs"
        return specs, horizon, pre_event_steps, meta

    with args.recommendation.open("r") as f:
        recommendation = json.load(f)
    horizon = int(args.horizon if args.horizon is not None else recommendation.get("horizon", 128))
    pre_event_steps = int(
        args.pre_event_steps if args.pre_event_steps is not None else recommendation.get("pre_event_steps", 16)
    )
    specs = []
    for spec in recommendation.get("selected_specs", []):
        status = spec.get("status", "unknown")
        is_core = spec.get("id") in set(recommendation.get("core_paper_spec_ids", []))
        if status in {"trainable", "pilot"} or is_core:
            specs.append(spec)
        elif args.include_sparse_core and is_core:
            specs.append(spec)
    if args.pilot_only:
        specs = [spec for spec in specs if spec.get("status") in {"pilot", "sparse"} or spec.get("id", "").startswith("paper_")]
    if args.eventual_only:
        specs = [spec for spec in specs if spec.get("type") == "eventual"]
    meta["spec_source"] = str(args.recommendation)
    meta["diagnostics"] = recommendation
    return specs, horizon, pre_event_steps, meta


def make_record(demo, window: Mapping, horizon: int, spec_index: int) -> Dict:
    start = int(window["start"])
    end = start + horizon
    traj = demo.states[start : end + 1].astype(np.float32)
    actions = demo.actions[start:end].astype(np.float32)
    if traj.shape != (horizon + 1, STATE_DIM):
        raise ValueError(f"Bad trajectory shape {traj.shape}; expected {(horizon + 1, STATE_DIM)}")
    if actions.shape != (horizon, ACTION_DIM):
        raise ValueError(f"Bad action shape {actions.shape}; expected {(horizon, ACTION_DIM)}")

    spec = dict(window["spec"])
    return {
        "env": "calvin",
        "spec_id": str(window["spec_id"]),
        "stl_seed": int(spec_index),
        "stl_type_i": int(spec_index),
        "formula": str(spec.get("formula", "")),
        "spec": spec,
        "state": traj[0].astype(np.float32),
        "trajs": traj,
        "us": actions,
        "actions": actions,
        "obs": traj,
        "score": np.asarray([float(window["score"])], dtype=np.float32),
        "split": str(window["split"]),
        "demo_key": str(window["demo_key"]),
        "start": int(window["start"]),
        "end": int(window["end"]),
        "event_times": dict(window.get("event_times", {})),
        "extra": {k: v for k, v in window.items() if k not in {"spec", "event_times"}},
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    specs, horizon, pre_event_steps, meta = load_specs(args)
    if not specs:
        raise RuntimeError("No specs selected. Run diagnostics or pass --include-sparse-core for core specs.")

    name = args.name or f"calvin_play_h{horizon}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_output_dir(args.output_root / name)

    records = []
    per_spec_counts = {}
    with h5py.File(args.dataset, "r") as h5:
        splits = hdf5_demo_splits(h5)
        keys = sorted_demo_keys(h5, max_demos=args.max_demos)
        demos_by_key = {}
        all_windows_by_spec: Dict[str, List[Dict]] = {}

        for demo_i, key in enumerate(keys):
            demo = load_demo_arrays(h5, key, splits.get(key, "unknown"))
            demos_by_key[key] = demo
            for spec in specs:
                raw = list(iter_spec_windows(spec, demo, horizon, pre_event_steps))
                all_windows_by_spec.setdefault(spec["id"], []).extend(raw)
            print(f"[{demo_i + 1:03d}/{len(keys):03d}] mined candidates from {key} split={demo.split}")

        spec_index_by_id = {spec["id"]: idx for idx, spec in enumerate(specs)}
        remaining_global = args.max_windows
        for spec in specs:
            spec_id = spec["id"]
            kept = deoverlap_windows(all_windows_by_spec.get(spec_id, []), horizon)
            random.shuffle(kept)
            cap = len(kept) if args.max_per_spec is None else min(len(kept), int(args.max_per_spec))
            if remaining_global is not None:
                cap = min(cap, int(remaining_global))
            selected = kept[:cap]
            per_spec_counts[spec_id] = {
                "available": int(len(kept)),
                "selected": int(len(selected)),
                "train": int(sum(item["split"] == "train" for item in selected)),
                "valid": int(sum(item["split"] == "valid" for item in selected)),
                "formula": spec.get("formula", ""),
            }
            for window in selected:
                demo = demos_by_key[window["demo_key"]]
                records.append(make_record(demo, window, horizon, spec_index_by_id[spec_id]))
            if remaining_global is not None:
                remaining_global -= len(selected)
                if remaining_global <= 0:
                    break

    if not records:
        raise RuntimeError("No records mined for selected specs.")

    data = np.asarray(records, dtype=object)
    meta.update(
        {
            "dataset": str(args.dataset),
            "horizon": int(horizon),
            "pre_event_steps": int(pre_event_steps),
            "training_scope": "atomic_eventual_only" if args.eventual_only else "selected_specs",
            "composition_training": False if args.eventual_only else None,
            "composition_note": (
                "Only atomic F(label) behavior chunks are mined for TeLoGraF training. "
                "Composite paper STLs are intentionally held out for evaluation."
                if args.eventual_only
                else "Specs were selected by the provided arguments/recommendation."
            ),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "num_records": int(len(records)),
            "specs": specs,
            "per_spec_counts": per_spec_counts,
        }
    )
    np.savez(out_dir / "data.npz", data=data, meta=np.asarray(json.dumps(meta), dtype=object))
    write_json(out_dir / "metadata.json", meta)
    print(f"\nwrote {len(records)} records to {out_dir / 'data.npz'}")
    for spec_id, count in per_spec_counts.items():
        print(f"  {spec_id}: selected={count['selected']} available={count['available']} {count['formula']}")


if __name__ == "__main__":
    main()
