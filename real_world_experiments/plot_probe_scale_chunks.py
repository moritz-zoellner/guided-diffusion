"""Plot executed EEF path with world-model predicted chunks for guidance probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


DEFAULT_VALUES = None


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safety_box_from_rows(rows: list[dict]) -> dict[str, float]:
    box = {"x_min": -0.55, "x_max": -0.51, "y_min": -0.10, "y_max": -0.07, "margin": 0.005}
    for row in rows:
        source = row.get("safety_box")
        if isinstance(source, dict):
            for key in box:
                if key in source:
                    box[key] = float(source[key])
            break
    return box


def parse_probe_chunks(rows: list[dict], values: tuple[float, ...] | None) -> tuple[np.ndarray, list[dict], str]:
    executed = []
    chunks = []
    latest_eef = None
    probe_event_idx = 0
    probe_label = "guidance scale"

    for row in rows:
        if "eef_xyz" in row:
            latest_eef = np.asarray(row["eef_xyz"][:2], dtype=np.float32)
            executed.append(latest_eef)
            continue

        if latest_eef is None:
            continue

        if row.get("type") == "probe_scales":
            entries = [
                (float(entry["guidance_scale"]), entry)
                for entry in row.get("scales", [])
            ]
            probe_label = "guidance scale"
        elif row.get("type") == "probe_param":
            entries = [
                (float(entry["probe_value"]), entry)
                for entry in row.get("entries", [])
            ]
            params = [entry.get("probe_param") for entry in row.get("entries", []) if entry.get("probe_param")]
            if params:
                probe_label = params[0]
        else:
            continue

        keep_values = set(values) if values is not None else None
        for value, entry in entries:
            if keep_values is not None and value not in keep_values:
                continue
            predicted_states = np.asarray(entry["predicted_states"], dtype=np.float32)
            if predicted_states.ndim != 2 or predicted_states.shape[1] < 2:
                continue
            pred_xy = predicted_states[:, :2]
            xy = np.vstack([latest_eef[None], pred_xy])
            chunks.append({"chunk_idx": probe_event_idx, "value": value, "xy": xy})
        probe_event_idx += 1

    return np.asarray(executed, dtype=np.float32), chunks, probe_label


def draw_safety_box(ax, box: dict[str, float]) -> None:
    margin = box["margin"]
    ax.add_patch(Rectangle(
        (box["x_min"] - margin, box["y_min"] - margin),
        (box["x_max"] - box["x_min"]) + 2 * margin,
        (box["y_max"] - box["y_min"]) + 2 * margin,
        fill=False,
        edgecolor="#ff9f1c",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    ))
    ax.add_patch(Rectangle(
        (box["x_min"], box["y_min"]),
        box["x_max"] - box["x_min"],
        box["y_max"] - box["y_min"],
        fill=True,
        facecolor="#d62728",
        edgecolor="#d62728",
        alpha=0.12,
        linewidth=1.6,
        zorder=1,
    ))


def value_colors(values: tuple[float, ...]) -> dict[float, tuple[float, float, float, float]]:
    ordered = sorted(set(float(value) for value in values))
    palette = [
        "#000000",  # 0 / baseline
        "#d7191c",  # red
        "#2c7bb6",  # blue
        "#00a651",  # green
        "#7b3294",  # purple
        "#ff7f00",  # orange
        "#00c5ff",  # cyan
        "#f781bf",  # pink
        "#8c510a",  # brown
        "#ffd92f",  # yellow
        "#4d4d4d",  # dark gray
        "#1b9e77",  # teal
    ]
    return {
        value: palette[i % len(palette)]
        for i, value in enumerate(ordered)
    }


def plot(
    path: Path,
    output: Path,
    values: tuple[float, ...] | None,
    fit_predictions: bool,
) -> None:
    rows = load_rows(path)
    safety_box = safety_box_from_rows(rows)
    executed, chunks, probe_label = parse_probe_chunks(rows, values)
    if len(executed) == 0:
        raise ValueError(f"{path} contains no eef_xyz rows")
    if not chunks:
        raise ValueError(f"{path} contains no probe chunks for values {values}")

    unique_chunk_ids = sorted({chunk["chunk_idx"] for chunk in chunks})
    colors_by_value = value_colors(tuple(sorted({chunk["value"] for chunk in chunks})))

    fig, ax = plt.subplots(figsize=(15, 11))
    draw_safety_box(ax, safety_box)

    ax.plot(
        executed[:, 0],
        executed[:, 1],
        color="#111111",
        linewidth=1.15,
        alpha=0.45,
        label="executed EEF",
        zorder=3,
    )
    ax.scatter(executed[0, 0], executed[0, 1], marker="s", s=46, color="#111111", alpha=0.7, zorder=9)
    ax.scatter(executed[-1, 0], executed[-1, 1], marker="x", s=62, color="#111111", alpha=0.7, zorder=9)

    all_xy = [executed]
    for chunk in chunks:
        xy = chunk["xy"]
        all_xy.append(xy)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=colors_by_value[chunk["value"]],
            linestyle="-",
            marker="o",
            markersize=3.4,
            markeredgewidth=0.45,
            markeredgecolor="white",
            linewidth=1.75,
            alpha=0.78,
            zorder=5,
        )
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            color=colors_by_value[chunk["value"]],
            s=13,
            alpha=0.6,
            zorder=6,
        )

    box_xy = np.array([
        [safety_box["x_min"] - safety_box["margin"], safety_box["y_min"] - safety_box["margin"]],
        [safety_box["x_max"] + safety_box["margin"], safety_box["y_max"] + safety_box["margin"]],
    ])
    if fit_predictions:
        all_xy.append(box_xy)
        stacked = np.vstack(all_xy)
    else:
        stacked = np.vstack([executed, box_xy])
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 0.06)
    pad = np.maximum(0.035, 0.18 * span)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(
        f"{path.name}: executed EEF and world-model predicted chunks\n"
        f"color = {probe_label}, dots = predicted world-model steps ({len(unique_chunk_ids)} chunks)"
    )

    scale_handles = [
        Line2D(
            [0],
            [0],
            color=colors_by_value[value],
            linestyle="-",
            marker="o",
            markersize=4.0,
            lw=2.2,
            label=f"{probe_label} {value:g}",
        )
        for value in sorted(colors_by_value)
    ]
    fixed_handles = [
        Line2D([0], [0], color="#111111", lw=1.3, alpha=0.55, label="executed EEF"),
        Line2D([0], [0], color="#ff9f1c", linestyle="--", lw=1.8, label="safety margin"),
        Line2D([0], [0], color="#d62728", lw=5.0, alpha=0.25, label="forbidden box"),
    ]
    ax.legend(handles=fixed_handles + scale_handles, loc="upper right", framealpha=0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, default=None)
    parser.add_argument("--rollout-dir", type=Path, default=Path("outputs/real_world/dynamics_guidance"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--scales",
        "--values",
        dest="values",
        type=float,
        nargs="+",
        default=DEFAULT_VALUES,
        help="Probe values to plot. Defaults to all values found in the rollout.",
    )
    parser.add_argument(
        "--fit-predictions",
        action="store_true",
        help="Fit axes to all predicted chunks. By default the view is zoomed around executed trajectory and safety box.",
    )
    args = parser.parse_args()

    rollout = args.rollout
    if rollout is None:
        candidates = sorted(args.rollout_dir.glob("rollout_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No rollout_*.jsonl files found under {args.rollout_dir}")
        rollout = candidates[-1]

    output = args.output or rollout.with_name(f"{rollout.stem}_probe_scale_chunks_xy.png")
    values = None if args.values is None else tuple(args.values)
    plot(rollout, output, values, args.fit_predictions)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
