"""Plot real-robot dynamics guidance rollouts in top-down XY.

The current rollout logger records EEF position only. If future logs include
label fields, this script will color by active label. Otherwise it colors by
safety status relative to the forbidden XY box used by the guidance code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


SAFETY_X_MIN, SAFETY_X_MAX = -0.55, -0.51
SAFETY_Y_MIN, SAFETY_Y_MAX = -0.10, -0.07
SAFETY_MARGIN = 0.005

LABEL_COLORS = {
    "can_grabbed": "#1f77b4",
    "pouring_right": "#d62728",
    "pouring_left": "#7f3c8d",
    "safe": "#2ca02c",
    "margin": "#ff9f1c",
    "inside_forbidden_box": "#d62728",
    "unlabeled": "#777777",
}


def default_safety_box() -> dict[str, float]:
    return {
        "x_min": SAFETY_X_MIN,
        "x_max": SAFETY_X_MAX,
        "y_min": SAFETY_Y_MIN,
        "y_max": SAFETY_Y_MAX,
        "margin": SAFETY_MARGIN,
    }


def load_rollout(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safety_box_from_rows(rows: list[dict]) -> dict[str, float]:
    """Read safety params from rollout rows when present, else use node constants."""
    box = default_safety_box()
    aliases = {
        "x_min": ("safety_x_min", "SAFETY_X_MIN", "x_min"),
        "x_max": ("safety_x_max", "SAFETY_X_MAX", "x_max"),
        "y_min": ("safety_y_min", "SAFETY_Y_MIN", "y_min"),
        "y_max": ("safety_y_max", "SAFETY_Y_MAX", "y_max"),
        "margin": ("safety_margin", "SAFETY_MARGIN", "margin"),
    }
    container_keys = ("safety_box", "safety_bbox", "bbox", "safety_params")

    for row in rows:
        sources = [row]
        sources.extend(row[key] for key in container_keys if isinstance(row.get(key), dict))
        for source in sources:
            for canonical, names in aliases.items():
                for name in names:
                    if name in source:
                        box[canonical] = float(source[name])
        if any(
            name in row
            for names in aliases.values()
            for name in names
        ) or any(isinstance(row.get(key), dict) for key in container_keys):
            break
    return box


def load_training_eef_trajectories(dataset: Path) -> list[np.ndarray]:
    if dataset is None or not dataset.exists():
        return []
    trajectories = []
    with h5py.File(dataset, "r") as f:
        for demo_key in sorted(f["data"].keys()):
            demo = f[f"data/{demo_key}"]
            if "obs/eef_pos" in demo:
                trajectories.append(np.asarray(demo["obs/eef_pos"][:, :2], dtype=np.float32))
    return trajectories


def infer_label(row: dict) -> str:
    for key in ("active_label", "label_name", "label"):
        value = row.get(key)
        if isinstance(value, str):
            return value

    labels = row.get("labels") or row.get("current_label")
    names = row.get("label_names") or ["can_grabbed", "pouring_right", "pouring_left"]
    if isinstance(labels, list) and labels:
        active = [names[i] for i, v in enumerate(labels) if i < len(names) and float(v) > 0.5]
        if active:
            return "+".join(active)

    return ""


def signed_distance_to_safety_box(xy: np.ndarray, box: dict[str, float]) -> np.ndarray:
    center = np.array([(box["x_min"] + box["x_max"]) / 2.0, (box["y_min"] + box["y_max"]) / 2.0])
    half = np.array([(box["x_max"] - box["x_min"]) / 2.0, (box["y_max"] - box["y_min"]) / 2.0])
    q = np.abs(xy - center) - half
    outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside + inside


def fallback_safety_labels(xy: np.ndarray, box: dict[str, float]) -> list[str]:
    dist = signed_distance_to_safety_box(xy, box)
    labels = []
    for d in dist:
        if d < 0.0:
            labels.append("inside_forbidden_box")
        elif d < box["margin"]:
            labels.append("margin")
        else:
            labels.append("safe")
    return labels


def add_colored_segments(ax, xy: np.ndarray, labels: list[str]) -> None:
    if len(xy) < 2:
        return
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    colors = [LABEL_COLORS.get(labels[i], LABEL_COLORS["unlabeled"]) for i in range(len(segments))]
    lc = LineCollection(segments, colors=colors, linewidths=1.8, alpha=0.9)
    ax.add_collection(lc)


def draw_training_distribution(ax, trajectories: list[np.ndarray]) -> None:
    for xy in trajectories:
        if len(xy) >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=0.55, alpha=0.045, zorder=0)


def draw_safety_box(ax, box: dict[str, float]) -> None:
    margin_rect = Rectangle(
        (box["x_min"] - box["margin"], box["y_min"] - box["margin"]),
        (box["x_max"] - box["x_min"]) + 2 * box["margin"],
        (box["y_max"] - box["y_min"]) + 2 * box["margin"],
        fill=False,
        edgecolor=LABEL_COLORS["margin"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    box_rect = Rectangle(
        (box["x_min"], box["y_min"]),
        box["x_max"] - box["x_min"],
        box["y_max"] - box["y_min"],
        fill=True,
        facecolor=LABEL_COLORS["inside_forbidden_box"],
        edgecolor=LABEL_COLORS["inside_forbidden_box"],
        alpha=0.12,
        linewidth=1.5,
    )
    ax.add_patch(margin_rect)
    ax.add_patch(box_rect)


def plot_rollouts(paths: list[Path], output: Path, training_trajectories: list[np.ndarray]) -> None:
    n = len(paths)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 6.0 * rows), squeeze=False)

    used_labels = set()
    for ax, path in zip(axes.ravel(), paths):
        rows_data = load_rollout(path)
        safety_box = safety_box_from_rows(rows_data)
        trajectory_rows = [row for row in rows_data if "eef_xyz" in row]
        xyz = np.asarray([row["eef_xyz"] for row in trajectory_rows], dtype=np.float32)
        if len(xyz) == 0:
            ax.set_title(f"{path.name}: no eef_xyz")
            ax.axis("off")
            continue

        xy = xyz[:, :2]
        logged_labels = [infer_label(row) for row in trajectory_rows]
        if any(logged_labels):
            labels = [label if label else "unlabeled" for label in logged_labels]
            label_mode = "logged labels"
        else:
            labels = fallback_safety_labels(xy, safety_box)
            label_mode = "safety status"

        used_labels.update(labels)
        draw_training_distribution(ax, training_trajectories)
        draw_safety_box(ax, safety_box)
        add_colored_segments(ax, xy, labels)
        ax.scatter(xy[0, 0], xy[0, 1], marker="s", s=42, color="#111111", label="start", zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="x", s=54, color="#111111", label="end", zorder=5)

        timed_rows = [row for row in trajectory_rows if "t_ns" in row]
        duration_s = (timed_rows[-1]["t_ns"] - timed_rows[0]["t_ns"]) / 1e9 if len(timed_rows) > 1 else 0.0
        ax.set_title(f"{path.name}\n{len(xy)} samples, {duration_s:.1f}s", fontsize=10)
        ax.text(
            0.01,
            0.99,
            f"{label_mode}\n"
            f"box x=[{safety_box['x_min']:.3f},{safety_box['x_max']:.3f}], "
            f"y=[{safety_box['y_min']:.3f},{safety_box['y_max']:.3f}], "
            f"m={safety_box['margin']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 2.0},
        )
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    for ax in axes.ravel()[len(paths):]:
        ax.axis("off")

    legend_labels = [
        name for name in (
            "safe",
            "margin",
            "inside_forbidden_box",
            "can_grabbed",
            "pouring_right",
            "pouring_left",
            "unlabeled",
        )
        if name in used_labels
    ]
    handles = [
        Line2D([0], [0], color=LABEL_COLORS[name], lw=2.5, label=name)
        for name in legend_labels
    ]
    handles.extend([
        Line2D([0], [0], color="#1f77b4", lw=1.5, alpha=0.45, label="training EEF demos"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#111111", label="start"),
        Line2D([0], [0], marker="x", color="#111111", linestyle="none", label="end"),
        Line2D([0], [0], color=LABEL_COLORS["margin"], linestyle="--", label="safety margin"),
        Line2D([0], [0], color=LABEL_COLORS["inside_forbidden_box"], lw=5, alpha=0.25, label="forbidden box"),
    ])
    fig.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)), fontsize=9)
    fig.suptitle("Real-world dynamics guidance rollouts: top-down EEF XY", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, default=Path("outputs/real_world/dynamics_guidance"))
    parser.add_argument("--dataset", type=Path, default=Path("data/real_world/cheezit_pouring.hdf5"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.rollout_dir.glob("rollout_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No rollout_*.jsonl files found under {args.rollout_dir}")

    output = args.output or args.rollout_dir / "rollout_xy_safety_overlay.png"
    training_trajectories = load_training_eef_trajectories(args.dataset)
    plot_rollouts(paths, output, training_trajectories)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
