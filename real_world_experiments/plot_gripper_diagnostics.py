"""Plot raw and binarized gripper signals used by the real-world DP."""

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
    from real_world_data import demo_sort_key, get_demo_keys
except ModuleNotFoundError:
    from real_world_experiments.real_world_data import demo_sort_key, get_demo_keys

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def _demo_arrays(demo):
    raw = demo["obs/gripper_width"][:].reshape(-1)
    obs_binary = demo["obs/gripper_binary"][:].reshape(-1)
    action_gripper = demo["actions"][:, 6].reshape(-1)
    return raw, obs_binary, action_gripper


def transition_summary(hdf5_path):
    summary = {
        "num_demos": 0,
        "raw_min": None,
        "raw_max": None,
        "raw_median": None,
        "obs_binary_counts": {},
        "action_gripper_counts": {},
        "num_switches_hist": {},
        "open_to_closed_hist": {},
        "closed_to_open_hist": {},
        "per_demo": [],
    }
    all_raw = []
    all_obs = []
    all_actions = []
    switches = []
    open_to_closed = []
    closed_to_open = []

    with h5py.File(hdf5_path, "r") as f:
        for demo_key in get_demo_keys(hdf5_path):
            raw, obs_binary, action_gripper = _demo_arrays(f[f"data/{demo_key}"])
            diff = np.diff(obs_binary)
            num_switches = int((diff != 0).sum())
            num_open_to_closed = int(((obs_binary[:-1] < 0) & (obs_binary[1:] > 0)).sum())
            num_closed_to_open = int(((obs_binary[:-1] > 0) & (obs_binary[1:] < 0)).sum())
            switch_indices = np.flatnonzero(diff != 0).astype(int) + 1

            all_raw.append(raw)
            all_obs.append(obs_binary)
            all_actions.append(action_gripper)
            switches.append(num_switches)
            open_to_closed.append(num_open_to_closed)
            closed_to_open.append(num_closed_to_open)
            summary["per_demo"].append(
                {
                    "demo_key": demo_key,
                    "num_steps": int(len(raw)),
                    "raw_min": float(raw.min()),
                    "raw_max": float(raw.max()),
                    "num_switches": num_switches,
                    "open_to_closed": num_open_to_closed,
                    "closed_to_open": num_closed_to_open,
                    "switch_indices": switch_indices.tolist(),
                }
            )

    raw = np.concatenate(all_raw)
    obs = np.concatenate(all_obs)
    actions = np.concatenate(all_actions)
    summary["num_demos"] = len(all_raw)
    summary["raw_min"] = float(raw.min())
    summary["raw_median"] = float(np.median(raw))
    summary["raw_max"] = float(raw.max())
    summary["obs_binary_counts"] = {str(float(v)): int((obs == v).sum()) for v in np.unique(obs)}
    summary["action_gripper_counts"] = {str(float(v)): int((actions == v).sum()) for v in np.unique(actions)}
    summary["num_switches_hist"] = {str(v): int(switches.count(v)) for v in sorted(set(switches))}
    summary["open_to_closed_hist"] = {str(v): int(open_to_closed.count(v)) for v in sorted(set(open_to_closed))}
    summary["closed_to_open_hist"] = {str(v): int(closed_to_open.count(v)) for v in sorted(set(closed_to_open))}
    return summary


def plot_grid(hdf5_path, output_path, demo_keys, threshold):
    n_cols = 5
    n_rows = int(np.ceil(len(demo_keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    with h5py.File(hdf5_path, "r") as f:
        for ax, demo_key in zip(axes, demo_keys):
            raw, obs_binary, action_gripper = _demo_arrays(f[f"data/{demo_key}"])
            t = np.arange(len(raw))
            ax2 = ax.twinx()
            ax.plot(t, raw, color="#222222", linewidth=1.0, label="raw width")
            ax.axhline(threshold, color="#777777", linestyle=":", linewidth=1.0, label="threshold")
            ax2.step(t, obs_binary, where="post", color="#1f77b4", linewidth=1.1, alpha=0.9, label="obs gripper_binary")
            ax2.step(t, action_gripper, where="post", color="#d62728", linewidth=0.8, alpha=0.7, label="action[:, 6]")
            ax.set_title(demo_key, fontsize=9)
            ax.set_xlabel("timestep")
            ax.set_ylim(-10, 265)
            ax2.set_ylim(-1.25, 1.25)
            ax.grid(True, alpha=0.2)
            if ax in axes[::n_cols]:
                ax.set_ylabel("raw gripper_width")
            ax2.set_yticks([-1, 1])
    for ax in axes[len(demo_keys) :]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], color="#222222", label="raw width"),
        Line2D([0], [0], color="#777777", linestyle=":", label="threshold"),
        Line2D([0], [0], color="#1f77b4", label="obs gripper_binary"),
        Line2D([0], [0], color="#d62728", label="action[:, 6]"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("-1 corresponds to low raw width/open; +1 corresponds to high raw width/closed", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_aligned(hdf5_path, output_path, threshold):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    raw_segments = []
    obs_segments = []
    action_segments = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in sorted(f["data"].keys(), key=demo_sort_key):
            raw, obs_binary, action_gripper = _demo_arrays(f[f"data/{demo_key}"])
            switch = np.flatnonzero((obs_binary[:-1] < 0) & (obs_binary[1:] > 0))
            if len(switch) == 0:
                continue
            center = int(switch[0] + 1)
            start = max(0, center - 60)
            end = min(len(raw), center + 160)
            x = np.arange(start, end) - center
            raw_segments.append((x, raw[start:end]))
            obs_segments.append((x, obs_binary[start:end]))
            action_segments.append((x, action_gripper[start:end]))

    for x, raw in raw_segments:
        axes[0].plot(x, raw, color="#222222", alpha=0.18, linewidth=0.8)
    for x, obs in obs_segments:
        axes[1].step(x, obs, where="post", color="#1f77b4", alpha=0.22, linewidth=0.8)
    for x, action in action_segments:
        axes[1].step(x, action, where="post", color="#d62728", alpha=0.18, linewidth=0.8)

    axes[0].axhline(threshold, color="#777777", linestyle=":", linewidth=1.0)
    for ax in axes:
        ax.axvline(0, color="#111111", linestyle="--", linewidth=0.9)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("raw gripper_width")
    axes[1].set_ylabel("binary / action")
    axes[1].set_xlabel("timesteps relative to first open->closed transition")
    axes[1].set_yticks([-1, 1])
    axes[0].set_title("All demos aligned at first gripper close transition")
    axes[1].plot([], [], color="#1f77b4", label="obs gripper_binary")
    axes[1].plot([], [], color="#d62728", label="action[:, 6]")
    axes[1].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/real_world/cheezit_pouring_right_ablation.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_world/gripper_diagnostics"))
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=128.5)
    args = parser.parse_args()

    demo_keys = get_demo_keys(args.dataset)[args.demo_start : args.demo_start + args.num_demos]
    stem = args.dataset.stem
    grid_path = args.output_dir / f"{stem}_gripper_grid_{len(demo_keys)}.png"
    aligned_path = args.output_dir / f"{stem}_gripper_aligned_close.png"
    summary_path = args.output_dir / f"{stem}_gripper_summary.json"

    summary = transition_summary(args.dataset)
    plot_grid(args.dataset, grid_path, demo_keys, args.threshold)
    plot_aligned(args.dataset, aligned_path, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Wrote gripper diagnostics:")
    print(grid_path)
    print(aligned_path)
    print(summary_path)
    print("Summary:", summary["num_switches_hist"], summary["open_to_closed_hist"], summary["closed_to_open_hist"])


if __name__ == "__main__":
    main()
