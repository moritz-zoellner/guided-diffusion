#!/usr/bin/env python
"""Mine Toy Squares HDF5 rollouts into TeLoGraF-compatible STL records."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping

import h5py
import numpy as np

from telograf_toy_squares.toy_specs import (
    ACTION_DIM,
    DEFAULT_CHAIN_BASE,
    DEFAULT_RADIUS,
    STATE_DIM,
    deoverlap_windows,
    ensure_output_dir,
    evaluate_spec_sequence,
    hdf5_demo_splits,
    iter_spec_windows,
    load_demo_arrays,
    padded_action_window,
    padded_state_window,
    sorted_demo_keys,
    toy_paper_specs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--recommendation", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/telograf/toy_squares/datasets"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--pre-event-steps", type=int, default=None)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--chain-base", type=str, default=",".join(DEFAULT_CHAIN_BASE))
    parser.add_argument("--max-chain-horizon", type=int, default=5)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--max-per-spec", type=int, default=5000)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--eventual-only", action="store_true")
    parser.add_argument("--no-padding", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_chain_base(value: str) -> List[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def load_specs(args: argparse.Namespace) -> tuple[List[Dict], int, int, Dict]:
    meta: Dict = {}
    if args.recommendation is None:
        horizon = 128 if args.horizon is None else int(args.horizon)
        pre_event_steps = 16 if args.pre_event_steps is None else int(args.pre_event_steps)
        specs = toy_paper_specs(
            chain_base=parse_chain_base(args.chain_base),
            max_chain_horizon=args.max_chain_horizon,
            radius=args.radius,
        )
        if args.eventual_only:
            specs = [spec for spec in specs if spec["type"] == "eventual"]
        meta["spec_source"] = "toy_paper_specs"
        return specs, horizon, pre_event_steps, meta

    with args.recommendation.open("r", encoding="utf-8") as f:
        recommendation = json.load(f)
    horizon = int(args.horizon if args.horizon is not None else recommendation.get("horizon", 128))
    pre_event_steps = int(
        args.pre_event_steps if args.pre_event_steps is not None else recommendation.get("pre_event_steps", 16)
    )
    specs = recommendation.get("selected_specs_by_horizon", {}).get(str(horizon))
    if specs is None:
        specs = recommendation.get("selected_specs", [])
    if args.eventual_only:
        specs = [spec for spec in specs if spec["type"] == "eventual"]
    meta["spec_source"] = str(args.recommendation)
    meta["diagnostics"] = recommendation
    return list(specs), horizon, pre_event_steps, meta


def make_record(demo, window: Mapping, horizon: int, spec_index: int) -> Dict:
    start = int(window["start"])
    traj = padded_state_window(demo.state_seq, start, horizon)
    actions = padded_action_window(demo.actions, start, horizon)
    if traj.shape != (horizon + 1, STATE_DIM):
        raise ValueError(f"Bad trajectory shape {traj.shape}; expected {(horizon + 1, STATE_DIM)}")
    if actions.shape != (horizon, ACTION_DIM):
        raise ValueError(f"Bad action shape {actions.shape}; expected {(horizon, ACTION_DIM)}")
    ok, score = evaluate_spec_sequence(window["spec"], traj)
    if not ok:
        raise ValueError(f"Mined window no longer satisfies {window['spec_id']} after padding.")

    spec = dict(window["spec"])
    return {
        "env": "toy_squares",
        "spec_id": str(window["spec_id"]),
        "stl_seed": int(spec_index),
        "stl_type_i": int(spec_index),
        "formula": str(spec.get("formula", "")),
        "spec": spec,
        "state": traj[0].astype(np.float32),
        "trajs": traj.astype(np.float32),
        "us": actions.astype(np.float32),
        "actions": actions.astype(np.float32),
        "obs": traj.astype(np.float32),
        "score": np.asarray([float(score)], dtype=np.float32),
        "split": str(window["split"]),
        "demo_key": str(window["demo_key"]),
        "target_label": window.get("target_label"),
        "start": int(window["start"]),
        "end": int(window["end"]),
        "demo_length": int(window.get("demo_length", demo.length)),
        "padded_steps": int(window.get("padded_steps", 0)),
        "event_times": dict(window.get("event_times", {})),
        "extra": {k: v for k, v in window.items() if k not in {"spec", "event_times"}},
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    specs, horizon, pre_event_steps, meta = load_specs(args)
    if not specs:
        raise RuntimeError("No specs selected. Run diagnostics first or drop --eventual-only.")

    name = args.name or f"toy_squares_h{horizon}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_output_dir(args.output_root / name)

    records = []
    per_spec_counts = {}
    allow_padding = not args.no_padding
    with h5py.File(args.dataset, "r") as h5:
        splits = hdf5_demo_splits(h5)
        keys = sorted_demo_keys(h5, max_demos=args.max_demos)
        demos_by_key = {}
        all_windows_by_spec: Dict[str, List[Dict]] = {}

        for demo_i, key in enumerate(keys):
            demo = load_demo_arrays(h5, key, splits.get(key, "unknown"), radius=args.radius)
            demos_by_key[key] = demo
            for spec in specs:
                raw = list(
                    iter_spec_windows(
                        spec,
                        demo,
                        horizon,
                        pre_event_steps,
                        allow_padding=allow_padding,
                    )
                )
                all_windows_by_spec.setdefault(spec["id"], []).extend(raw)
            print(f"[{demo_i + 1:04d}/{len(keys):04d}] mined candidates from {key} split={demo.split}")

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
            padded = [int(item.get("padded_steps", 0)) for item in selected]
            per_spec_counts[spec_id] = {
                "available": int(len(kept)),
                "selected": int(len(selected)),
                "train": int(sum(item["split"] == "train" for item in selected)),
                "valid": int(sum(item["split"] == "valid" for item in selected)),
                "mean_padded_steps": float(np.mean(padded)) if padded else 0.0,
                "max_padded_steps": int(max(padded)) if padded else 0,
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
            "radius": float(args.radius),
            "allow_padding": bool(allow_padding),
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
        print(
            f"  {spec_id}: selected={count['selected']} available={count['available']} "
            f"mean_pad={count['mean_padded_steps']:.1f} {count['formula']}"
        )


if __name__ == "__main__":
    main()
