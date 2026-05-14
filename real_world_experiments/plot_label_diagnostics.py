"""Plot real-world label diagnostics for manual validation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

try:
    from label_real_world import LABEL_NAMES, load_label_config, label_demo
    from real_world_data import (
        get_demo_keys,
        quat_wxyz_conjugate,
        quat_wxyz_multiply,
        quat_wxyz_to_rotvec,
        read_obs_array,
    )
except ModuleNotFoundError:
    from real_world_experiments.label_real_world import LABEL_NAMES, load_label_config, label_demo
    from real_world_experiments.real_world_data import (
        get_demo_keys,
        quat_wxyz_conjugate,
        quat_wxyz_multiply,
        quat_wxyz_to_rotvec,
        read_obs_array,
    )

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


LABEL_COLORS = {
    "can_grabbed": "#1f77b4",
    "pouring_right": "#d62728",
    "pouring_left": "#7f3c8d",
}


def local_twist_deg(eef_quat_wxyz: np.ndarray) -> np.ndarray:
    eef_quat_wxyz = np.asarray(eef_quat_wxyz, dtype=np.float32)
    q_start = np.repeat(eef_quat_wxyz[:1], len(eef_quat_wxyz), axis=0)
    q_delta_local = quat_wxyz_multiply(quat_wxyz_conjugate(q_start), eef_quat_wxyz)
    return np.rad2deg(quat_wxyz_to_rotvec(q_delta_local)[:, 2])


def add_time_colored_line(ax, xy, cmap="viridis", alpha=0.9, linewidth=1.2):
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) < 2:
        return None
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    collection = LineCollection(segments, cmap=cmap, alpha=alpha, linewidth=linewidth)
    collection.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(collection)
    ax.update_datalim(xy)
    ax.autoscale_view()
    return collection


def plot_special_demo(dataset, output_dir, demo_key, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset, "r") as f:
        demo = f[f"data/{demo_key}"]
        gripper = read_obs_array(demo, "gripper_width").reshape(-1)
        cheezit_pos = read_obs_array(demo, "cheezit_pos")
        labels = label_demo(demo, config=config)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.arange(len(gripper))
    axes[0].plot(t, gripper, color="#222222", linewidth=1.2)
    axes[0].axhline(config.gripper_closed_threshold, color=LABEL_COLORS["can_grabbed"], linestyle=":", linewidth=1.5)
    axes[0].set_title(f"{demo_key}: raw gripper width")
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("gripper_width")
    axes[0].grid(True, alpha=0.25)

    add_time_colored_line(axes[1], cheezit_pos[:, :2], linewidth=1.5)
    axes[1].scatter(cheezit_pos[0, 0], cheezit_pos[0, 1], marker="s", color="#2ca02c", s=36, label="start", zorder=5)
    axes[1].scatter(cheezit_pos[-1, 0], cheezit_pos[-1, 1], marker="x", color="#111111", s=44, label="end", zorder=5)
    for idx, name in enumerate(LABEL_NAMES):
        active = labels[:, idx] > 0.5
        if active.any():
            axes[1].scatter(
                cheezit_pos[active, 0],
                cheezit_pos[active, 1],
                s=8,
                color=LABEL_COLORS[name],
                label=name,
                alpha=0.75,
                zorder=4,
            )
    axes[1].set_title(f"{demo_key}: Cheez-It XY")
    axes[1].set_xlabel("world x [m]")
    axes[1].set_ylabel("world y [m]")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / f"{demo_key}_gripper_and_cheezit_xy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_twist_grid(dataset, output_dir, demo_keys, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cols = 5
    n_rows = int(np.ceil(len(demo_keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.8 * n_rows), sharey=True)
    axes = np.asarray(axes).reshape(-1)

    on = float(config.pour_twist_on_deg)
    off = float(config.pour_twist_off_deg)
    with h5py.File(dataset, "r") as f:
        for ax, demo_key in zip(axes, demo_keys):
            demo = f[f"data/{demo_key}"]
            twist = local_twist_deg(read_obs_array(demo, "eef_quat_wxyz"))
            labels = label_demo(demo, config=config)
            t = np.arange(len(twist))
            ax.plot(t, twist, color="#222222", linewidth=1.0)
            ax.axhline(on, color=LABEL_COLORS["pouring_right"], linestyle=":", linewidth=1.3)
            ax.axhline(-on, color=LABEL_COLORS["pouring_left"], linestyle=":", linewidth=1.3)
            ax.axhline(off, color=LABEL_COLORS["pouring_right"], linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axhline(-off, color=LABEL_COLORS["pouring_left"], linestyle="--", linewidth=0.8, alpha=0.5)
            for idx, name in enumerate(LABEL_NAMES[1:], start=1):
                active = labels[:, idx] > 0.5
                if active.any():
                    ax.scatter(t[active], twist[active], s=5, color=LABEL_COLORS[name], alpha=0.8)
            ax.set_title(demo_key, fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("t")
    for ax in axes[len(demo_keys) :]:
        ax.axis("off")
    for ax in axes[::n_cols]:
        ax.set_ylabel("local twist z [deg]")
    fig.suptitle("EEF local wrist twist over time; dotted lines are +/- tilt-on thresholds", y=0.995)
    fig.tight_layout()
    path = output_dir / f"twist_over_time_{len(demo_keys)}_episodes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_xy_grid(dataset, output_dir, demo_keys, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cols = 5
    n_rows = int(np.ceil(len(demo_keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    with h5py.File(dataset, "r") as f:
        for ax, demo_key in zip(axes, demo_keys):
            demo = f[f"data/{demo_key}"]
            eef_pos = read_obs_array(demo, "eef_pos")
            cheezit_pos = read_obs_array(demo, "cheezit_pos")
            labels = label_demo(demo, config=config)

            ax.plot(eef_pos[:, 0], eef_pos[:, 1], color="#111111", linewidth=0.9, alpha=0.75, label="EEF")
            ax.plot(cheezit_pos[:, 0], cheezit_pos[:, 1], color="#bdbdbd", linewidth=1.0, alpha=0.75, label="Cheez-It")
            ax.scatter(eef_pos[0, 0], eef_pos[0, 1], marker="s", color="#2ca02c", s=18, zorder=5)
            ax.scatter(eef_pos[-1, 0], eef_pos[-1, 1], marker="x", color="#111111", s=24, zorder=5)
            for idx, name in enumerate(LABEL_NAMES):
                active = labels[:, idx] > 0.5
                if active.any():
                    ax.scatter(
                        cheezit_pos[active, 0],
                        cheezit_pos[active, 1],
                        s=5 if name == "can_grabbed" else 9,
                        color=LABEL_COLORS[name],
                        alpha=0.35 if name == "can_grabbed" else 0.85,
                        label=name,
                        zorder=4,
                    )
            ax.set_title(demo_key, fontsize=9)
            ax.set_xlabel("world x [m]")
            ax.set_ylabel("world y [m]")
            ax.axis("equal")
            ax.grid(True, alpha=0.2)
    for ax in axes[len(demo_keys) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=5, fontsize=9)
    fig.suptitle("EEF and Cheez-It XY; colored Cheez-It points show active labels", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = output_dir / f"xy_label_overlay_{len(demo_keys)}_episodes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def find_no_grab_demo(dataset, config):
    with h5py.File(dataset, "r") as f:
        for demo_key in get_demo_keys(dataset):
            labels = label_demo(f[f"data/{demo_key}"], config=config)
            if labels[:, 0].sum() == 0:
                return demo_key
    return None


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/real_world/cheezit_pouring.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_world/label_diagnostics"))
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-keys", default=None, help="Comma-separated demo ids. Overrides --demo-start/--num-demos.")
    parser.add_argument("--special-demo", default=None)
    parser.add_argument("--skip-special-demo", action="store_true")
    args = parser.parse_args()

    config = load_label_config(args.label_config)
    if args.demo_keys:
        demo_keys = [key.strip() for key in args.demo_keys.split(",") if key.strip()]
    else:
        demo_keys = get_demo_keys(args.dataset)[args.demo_start : args.demo_start + args.num_demos]
    special_demo = None if args.skip_special_demo else (args.special_demo or find_no_grab_demo(args.dataset, config))

    paths = []
    if special_demo is not None:
        paths.append(plot_special_demo(args.dataset, args.output_dir, special_demo, config))
    paths.append(plot_twist_grid(args.dataset, args.output_dir, demo_keys, config))
    paths.append(plot_xy_grid(args.dataset, args.output_dir, demo_keys, config))

    print("Wrote label diagnostic plots:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
