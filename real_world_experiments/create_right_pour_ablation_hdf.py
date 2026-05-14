"""Create a one-way right-pour ablation HDF5 from the real-world dataset.

The ablation keeps demos whose clean pour-event sequence is either:

    right
    right -> left

and truncates each trajectory shortly after the final selected pour label turns
off. This removes the ambiguous return-to-pickup/drop tail from diffusion policy
training without touching the original HDF5 or source pickle episodes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

try:
    from label_real_world import LABEL_NAMES, label_demo, load_label_config
    from real_world_data import demo_sort_key, get_demo_keys, read_obs_array
except ModuleNotFoundError:
    from real_world_experiments.label_real_world import LABEL_NAMES, label_demo, load_label_config
    from real_world_experiments.real_world_data import demo_sort_key, get_demo_keys, read_obs_array

matplotlib.use("Agg")
from matplotlib import pyplot as plt


RIGHT_LABEL = "pouring_right"
LEFT_LABEL = "pouring_left"


def pour_sequence(labels: np.ndarray) -> list[str]:
    right = labels[:, LABEL_NAMES.index(RIGHT_LABEL)] > 0.5
    left = labels[:, LABEL_NAMES.index(LEFT_LABEL)] > 0.5
    seq = []
    prev_right = False
    prev_left = False
    for is_right, is_left in zip(right, left):
        if is_right and not prev_right:
            seq.append("right")
        if is_left and not prev_left:
            seq.append("left")
        prev_right = bool(is_right)
        prev_left = bool(is_left)
    return seq


def cutoff_after_final_event(labels: np.ndarray, sequence: list[str], post_label_steps: int) -> int:
    final_event = sequence[-1]
    label_idx = LABEL_NAMES.index(RIGHT_LABEL if final_event == "right" else LEFT_LABEL)
    active = np.flatnonzero(labels[:, label_idx] > 0.5)
    if len(active) == 0:
        raise ValueError("Cannot determine cutoff for a sequence with no active final label")
    cutoff = int(active[-1] + 1 + post_label_steps)
    return min(cutoff, len(labels))


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_dataset_sliced(src_dataset, dst_group, name, original_length, new_length):
    value = src_dataset[()]
    if value.ndim > 0 and value.shape[0] == original_length:
        value = value[:new_length]
    if name == "dones" and value.ndim == 1 and len(value) == new_length:
        value = np.zeros(new_length, dtype=bool)
        value[-1] = True
    if name == "actions" and value.ndim == 2 and len(value) == new_length:
        value = value.copy()
        value[-1, :6] = 0.0
    compression = "gzip" if np.asarray(value).size > 0 and np.asarray(value).ndim > 0 else None
    dst_group.create_dataset(name, data=value, compression=compression)


def copy_group_sliced(src_group, dst_group, original_length, new_length):
    copy_attrs(src_group, dst_group)
    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            copy_dataset_sliced(item, dst_group, name, original_length, new_length)
        elif isinstance(item, h5py.Group):
            child = dst_group.create_group(name)
            copy_group_sliced(item, child, original_length, new_length)
        else:
            raise TypeError(f"Unsupported HDF5 item type at {item.name}: {type(item)}")


def selected_records(input_path, config, post_label_steps):
    records = []
    rejected = []
    with h5py.File(input_path, "r") as src:
        for demo_key in get_demo_keys(input_path):
            demo = src[f"data/{demo_key}"]
            labels = label_demo(demo, config=config)
            seq = pour_sequence(labels)
            if seq not in (["right"], ["right", "left"]):
                rejected.append({"demo_key": demo_key, "sequence": seq})
                continue
            cutoff = cutoff_after_final_event(labels, seq, post_label_steps)
            records.append(
                {
                    "source_demo_key": demo_key,
                    "sequence": seq,
                    "original_length": int(len(labels)),
                    "truncated_length": int(cutoff),
                    "cutoff_index_exclusive": int(cutoff),
                    "post_label_steps": int(post_label_steps),
                }
            )
    return records, rejected


def create_hdf(input_path, output_path, records, rejected, overwrite):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_source_keys = [record["source_demo_key"] for record in records]
    source_to_new = {source_key: f"demo_{idx}" for idx, source_key in enumerate(selected_source_keys)}
    total = 0
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        data_group = dst.create_group("data")
        copy_attrs(src["data"], data_group)

        for record in records:
            source_key = record["source_demo_key"]
            new_key = source_to_new[source_key]
            src_demo = src[f"data/{source_key}"]
            dst_demo = data_group.create_group(new_key)
            original_length = int(src_demo.attrs["num_samples"])
            new_length = int(record["truncated_length"])
            copy_group_sliced(src_demo, dst_demo, original_length, new_length)
            dst_demo.attrs["num_samples"] = new_length
            dst_demo.attrs["source_demo_key"] = source_key
            dst_demo.attrs["ablation_sequence"] = ",".join(record["sequence"])
            dst_demo.attrs["ablation_cutoff_index_exclusive"] = int(record["cutoff_index_exclusive"])
            record["new_demo_key"] = new_key
            total += new_length

        data_group.attrs["total"] = int(total)

        if "mask" in src:
            mask_group = dst.create_group("mask")
            for mask_name, mask_dataset in src["mask"].items():
                source_keys = [key.decode("utf-8") for key in mask_dataset[()]]
                new_keys = [source_to_new[key] for key in source_keys if key in source_to_new]
                mask_group.create_dataset(mask_name, data=np.asarray([key.encode("utf-8") for key in new_keys]))

        metadata = {
            "input": str(input_path),
            "output": str(output_path),
            "selection": "clean right-only and clean right-then-left pour-event sequences",
            "cutoff": "truncate post_label_steps after the final selected pour label turns off",
            "num_demos": len(records),
            "total_samples": int(total),
            "label_names": LABEL_NAMES,
            "records": records,
            "rejected": rejected,
            "notes": [
                "Original source HDF5 and ROS pickle episodes are read-only and untouched.",
                "The final action translational/rotational delta is zeroed after truncation.",
                "dones is reset so only the truncated final sample is terminal.",
            ],
        }
        metadata_group = dst.create_group("metadata")
        metadata_group.create_dataset("json", data=json.dumps(metadata, indent=2).encode("utf-8"))

    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def plot_xy(output_path, plot_path):
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=demo_sort_key)
        fig, ax = plt.subplots(figsize=(8, 8))
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            eef_pos = read_obs_array(demo, "eef_pos")
            cheezit_pos = read_obs_array(demo, "cheezit_pos")
            seq = demo.attrs.get("ablation_sequence", "")
            color = "#d62728" if seq == "right" else "#7f3c8d"
            ax.plot(eef_pos[:, 0], eef_pos[:, 1], color=color, alpha=0.28, linewidth=0.8)
            ax.plot(cheezit_pos[:, 0], cheezit_pos[:, 1], color=color, alpha=0.12, linewidth=0.8, linestyle="--")
            ax.scatter(eef_pos[0, 0], eef_pos[0, 1], marker="s", color="#2ca02c", s=10, alpha=0.55)
            ax.scatter(eef_pos[-1, 0], eef_pos[-1, 1], marker="x", color="#111111", s=12, alpha=0.55)
        ax.plot([], [], color="#d62728", label="right only EEF")
        ax.plot([], [], color="#7f3c8d", label="right then left EEF")
        ax.plot([], [], color="#555555", linestyle="--", label="Cheez-It pose trace")
        ax.scatter([], [], marker="s", color="#2ca02c", label="trajectory start")
        ax.scatter([], [], marker="x", color="#111111", label="truncated end")
        ax.set_title("Right-pour ablation: truncated one-way trajectories")
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/real_world/cheezit_pouring.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("data/real_world/cheezit_pouring_right_ablation.hdf5"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/real_world/right_ablation/eef_cheezit_xy.png"))
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--post-label-steps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_label_config(args.label_config)
    records, rejected = selected_records(args.input, config, args.post_label_steps)
    metadata = create_hdf(args.input, args.output, records, rejected, overwrite=args.overwrite)
    plot_path = plot_xy(args.output, args.plot)

    sequence_counts = {}
    for record in records:
        key = "->".join(record["sequence"])
        sequence_counts[key] = sequence_counts.get(key, 0) + 1
    print(f"Wrote {metadata['num_demos']} demos / {metadata['total_samples']} samples to {args.output}")
    print(f"Sequence counts: {sequence_counts}")
    print(f"Rejected demos: {len(rejected)}")
    print(f"Wrote XY plot: {plot_path}")


if __name__ == "__main__":
    main()
