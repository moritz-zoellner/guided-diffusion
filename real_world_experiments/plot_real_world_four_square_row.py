#!/usr/bin/env python3
"""Create the four-square real-world summary row for the paper."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world_experiments.plot_real_world_paper_results import (  # noqa: E402
    DEFAULT_BASE_POLICY_SUMMARY,
    DEFAULT_HINT2_LEFT_SUMMARY,
    DEFAULT_HINT2_RIGHT_SUMMARY,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    collect_values,
    load_observed_chain_prefix_rate,
    load_values_csv,
)
from real_world_experiments.plot_real_world_xy_trace_panels import (  # noqa: E402
    DEFAULT_SAFETY_RUN,
)
from real_world_experiments.plot_rzz_gate_angle_comparison import (  # noqa: E402
    collect as collect_rzz_gate_angle,
    discover_events as discover_rzz_events,
    filter_by_required_label_event as filter_rzz_by_required_label_event,
    group_stats as rzz_group_stats,
)


LATEX_TEXTWIDTH_PT = 397.48499
PT_PER_IN = 72.27
TEXTWIDTH_IN = LATEX_TEXTWIDTH_PT / PT_PER_IN
FOUR_SQUARE_ROW_HEIGHT_IN = 1.50
FOUR_PANEL_LEFT_MARGIN_IN = 0.29
FOUR_PANEL_RIGHT_MARGIN_IN = 0.02
FOUR_PANEL_FRAME_GAP_IN = 0.15
FOUR_PANEL_AXIS_BOTTOM_IN = 0.36
FOUR_PANEL_TOP_MARGIN_IN = 0.16
FOUR_PANEL_BAR_SIDE_IN = FOUR_SQUARE_ROW_HEIGHT_IN - FOUR_PANEL_AXIS_BOTTOM_IN - FOUR_PANEL_TOP_MARGIN_IN
FOUR_PANEL_IMAGE_HEIGHT_IN = FOUR_PANEL_BAR_SIDE_IN
FOUR_PANEL_IMAGE_WIDTH_IN = (
    TEXTWIDTH_IN
    - FOUR_PANEL_LEFT_MARGIN_IN
    - FOUR_PANEL_RIGHT_MARGIN_IN
    - 3 * FOUR_PANEL_FRAME_GAP_IN
    - 2 * FOUR_PANEL_BAR_SIDE_IN
) / 2
FOUR_PANEL_CROP_BOTTOM_IN = 0.0
FOUR_PANEL_CROP_TOP_IN = FOUR_SQUARE_ROW_HEIGHT_IN
FINAL_VERTICAL_CROP_PAD_IN = 0.02
FOUR_PANEL_LEGEND_BOTTOM_IN = 0.05
IMAGE_VERTICAL_CROP_FROM_BOTTOM_FRAC = 0.4
FIG_DPI = 300

DARK_GRAY = "#5f6368"
AXIS_GRAY = "#8a8a8a"
UNSAFE_RED = "#b85f5a"
PANEL_FRAME_LW = 0.9
ZERO_BAR_VISUAL_HEIGHT = 0.012
ZERO_BAR_X_Y = 0.038
ZERO_BAR_X_MARKER_SIZE = 7.0
ZERO_BAR_X_LINEWIDTH = 0.55
BOTTOM_GROUP_LABEL_Y = -0.035
BAR_PANEL_TITLE_SIZE = 6.0
# Distance from each panel title to its frame, in points.
PANEL_TITLE_PAD_PT = 2.5
# Moves all four plot/image panels vertically without changing their size.
# Positive moves the panels up; negative moves them down.
PANEL_ROW_VERTICAL_SHIFT_IN = 0.0

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/four_square_row"
DEFAULT_REGION_BASELINE_RUN = DEFAULT_HINT2_LEFT_SUMMARY.parent
DEFAULT_STL_GPC_LEFT_SUMMARY = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/stl_gpc_sequence_eval/"
    "stl_gpc_left_epoch160_n10/summary.json"
)
DEFAULT_STL_GPC_RIGHT_SUMMARY = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/stl_gpc_sequence_eval/"
    "stl_gpc_right_epoch160_n10/summary.json"
)
DEFAULT_STL_GPC_REGION_RUN = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/stl_gpc_sequence_eval/"
    "stl_gpc_left_safety_box0_epoch160_n10_graspfix"
)
DEFAULT_BASE_POLICY_IMAGE = DEFAULT_OUTPUT_DIR / "mpv-shot0004 (1).png"
DEFAULT_COMPLEX_TASKS_IMAGE = DEFAULT_OUTPUT_DIR / "complex_tasks (1)-1.jpg"
DEFAULT_ANGLE_BASELINE_RUN = (
    REPO_ROOT / "outputs/real_world/paper_rollouts/base_dp_eval/base_dp_epoch160_n5_8"
)
DEFAULT_HINT2_ANGLE_RUN = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    "automaton_left_rzz1_hold_xminus063_step002_epoch160_n10_3"
)
DEFAULT_STL_GPC_ANGLE_RUN = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/stl_gpc_sequence_eval/"
    "stl_gpc_left_rzz1_hold_xminus060_epoch160_n10"
)
DEFAULT_ANGLE_GATE_X = -0.60
DEFAULT_ANGLE_SUCCESS_THRESHOLD_DEG = 6.0


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": [
                "Computer Modern Typewriter",
                "CMU Typewriter Text",
                "DejaVu Sans Mono",
            ],
            "mathtext.fontset": "cm",
            "axes.labelsize": 6.0,
            "axes.titlesize": 7.0,
            "axes.titleweight": "normal",
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.2,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_axes_in_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    return fig.add_axes([left / fig_w, bottom / fig_h, width / fig_w, height / fig_h])


def add_mixed_four_panel_row(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    bar_side = FOUR_PANEL_BAR_SIDE_IN
    image_width = FOUR_PANEL_IMAGE_WIDTH_IN
    image_height = FOUR_PANEL_IMAGE_HEIGHT_IN
    gap = FOUR_PANEL_FRAME_GAP_IN
    left = FOUR_PANEL_LEFT_MARGIN_IN
    bottom = FOUR_PANEL_AXIS_BOTTOM_IN + PANEL_ROW_VERTICAL_SHIFT_IN

    behavior_left = left
    safety_left = behavior_left + bar_side + gap
    base_left = safety_left + bar_side + gap
    complex_left = base_left + image_width + gap

    return (
        add_axes_in_inches(fig, behavior_left, bottom, bar_side, bar_side),
        add_axes_in_inches(fig, safety_left, bottom, bar_side, bar_side),
        add_axes_in_inches(fig, base_left, bottom, image_width, image_height),
        add_axes_in_inches(fig, complex_left, bottom, image_width, image_height),
    )


def style_square_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(PANEL_FRAME_LW)
    ax.tick_params(axis="both", color=AXIS_GRAY, width=0.45, length=1.8, pad=1.0)


def plot_placeholder(ax: plt.Axes, title: str, detail: str) -> None:
    style_square_frame(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f6f7f9")
    ax.set_title(title, pad=PANEL_TITLE_PAD_PT)
    ax.text(
        0.5,
        0.50,
        detail,
        ha="center",
        va="center",
        fontsize=5.2,
        color=DARK_GRAY,
        linespacing=1.15,
    )


def square_crop_image(image):
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = max(0, (height - side) // 2)
    x0 = max(0, (width - side) // 2)
    return image[y0 : y0 + side, x0 : x0 + side]


def crop_image_to_aspect(
    image,
    target_aspect: float,
    *,
    vertical_crop_from_bottom_frac: float = IMAGE_VERTICAL_CROP_FROM_BOTTOM_FRAC,
):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or target_aspect <= 0:
        return image

    current_aspect = width / height
    if np.isclose(current_aspect, target_aspect, rtol=1e-3, atol=1e-3):
        return image

    if current_aspect < target_aspect:
        desired_height = max(1, min(height, int(round(width / target_aspect))))
        removed_height = height - desired_height
        crop_from_bottom = int(round(removed_height * np.clip(vertical_crop_from_bottom_frac, 0.0, 1.0)))
        y0 = removed_height - crop_from_bottom
        return image[y0 : y0 + desired_height, :]

    desired_width = max(1, min(width, int(round(height * target_aspect))))
    x0 = max(0, (width - desired_width) // 2)
    return image[:, x0 : x0 + desired_width]


def add_bottom_group_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        BOTTOM_GROUP_LABEL_Y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=plt.rcParams["xtick.labelsize"],
        color="black",
        clip_on=False,
    )


def plot_image_or_placeholder(
    ax: plt.Axes,
    title: str,
    image_path: Path | None,
    detail: str,
    *,
    bottom_label: str | None = None,
    target_aspect: float | None = None,
) -> None:
    if image_path is None or not image_path.exists():
        plot_placeholder(ax, title, detail)
        if bottom_label is not None:
            add_bottom_group_label(ax, bottom_label)
        return
    style_square_frame(ax)
    with Image.open(image_path) as pil_image:
        image = np.asarray(pil_image.convert("RGB"))
    if target_aspect is None:
        image = square_crop_image(image)
    else:
        image = crop_image_to_aspect(image, target_aspect)
    ax.imshow(image, aspect="auto")
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=PANEL_TITLE_PAD_PT)
    if bottom_label is not None:
        add_bottom_group_label(ax, bottom_label)


def style_bar_axis(ax: plt.Axes, *, show_ylabel: bool) -> None:
    style_square_frame(ax)
    ax.grid(axis="y", alpha=0.22, linewidth=0.32)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.04)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.axhline(1.0, color=AXIS_GRAY, linewidth=0.42, linestyle=(0, (2.0, 2.0)), zorder=2)
    if show_ylabel:
        ax.set_ylabel("success", labelpad=2.0)
    else:
        ax.tick_params(axis="y", labelleft=False)


def plot_bar_panel(
    ax: plt.Axes,
    values: dict[str, dict[str, float]],
    *,
    title: str,
    groups: list[tuple[str, str]],
    show_ylabel: bool,
) -> None:
    style_bar_axis(ax, show_ylabel=show_ylabel)
    bar_width = 0.105
    drawn_bar_width = bar_width * 0.84
    group_gap = 0.20 if len(groups) > 1 else 0.0
    cluster_width = len(METHOD_ORDER) * bar_width
    centers = []
    bar_positions = []
    x = 0.0
    for group_key, group_label in groups:
        group_start = x
        for method_idx, method in enumerate(METHOD_ORDER):
            pos = group_start + method_idx * bar_width
            bar_positions.append(pos)
            value = values[group_key][method]
            ax.bar(
                pos,
                visual_bar_height(value),
                width=drawn_bar_width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.32,
                zorder=3,
            )
            mark_zero_bar(ax, pos, value)
        centers.append(group_start + (len(METHOD_ORDER) - 1) * bar_width / 2.0)
        x = group_start + cluster_width + group_gap

    if len(groups) == 1:
        center = centers[0]
        ax.set_xlim(center - 0.25, center + 0.25)
    else:
        left_edge = min(bar_positions) - drawn_bar_width / 2.0
        right_edge = max(bar_positions) + drawn_bar_width / 2.0
        ax.set_xlim(left_edge - 0.11, right_edge + 0.11)
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in groups])
    ax.tick_params(axis="x", bottom=False, labelbottom=True, pad=1.5)
    ax.set_title(title, pad=PANEL_TITLE_PAD_PT, fontsize=BAR_PANEL_TITLE_SIZE)


def ensure_metric_groups(values: dict[str, dict[str, float]]) -> None:
    for group in ("region", "angle"):
        values.setdefault(group, {method: 0.0 for method in METHOD_ORDER})


def visual_bar_height(value: float) -> float:
    if np.isfinite(value) and np.isclose(value, 0.0):
        return ZERO_BAR_VISUAL_HEIGHT
    return value


def mark_zero_bar(ax: plt.Axes, pos: float, value: float) -> None:
    if not (np.isfinite(value) and np.isclose(value, 0.0)):
        return
    ax.scatter(
        [pos],
        [ZERO_BAR_X_Y],
        marker="x",
        s=ZERO_BAR_X_MARKER_SIZE,
        color=UNSAFE_RED,
        linewidths=ZERO_BAR_X_LINEWIDTH,
        zorder=5,
    )


def load_shrunk_region_box(config_path: Path, shrink_m: float) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    raw_box = config["safety_box"]
    x_min = float(raw_box["x_min"]) + shrink_m
    x_max = float(raw_box["x_max"]) - shrink_m
    y_min = float(raw_box["y_min"]) + shrink_m
    y_max = float(raw_box["y_max"]) - shrink_m
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Safety region shrink {shrink_m} is too large for {raw_box}")
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "shrink_m": float(shrink_m),
        "raw": raw_box,
    }


def obs_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "target_reached":
        return event.get("reached_obs")
    if event.get("type") == "rollout_end":
        return event.get("final_obs")
    if event.get("type") in {"rollout_start", "decision"}:
        return event.get("obs")
    return None


def iter_rollout_eef_xy(rollout_dir: Path) -> list[tuple[float, float]]:
    events_path = rollout_dir / "events.jsonl"
    points: list[tuple[float, float]] = []
    if not events_path.exists():
        return points
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            obs = obs_from_event(event)
            if not obs or "eef_pos" not in obs:
                continue
            eef_pos = obs["eef_pos"]
            points.append((float(eef_pos[0]), float(eef_pos[1])))
    return points


def point_inside_box(x: float, y: float, box: dict[str, Any]) -> bool:
    return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]


def signed_clearance_to_box(x: float, y: float, box: dict[str, Any]) -> float:
    if point_inside_box(x, y, box):
        return -min(x - box["x_min"], box["x_max"] - x, y - box["y_min"], box["y_max"] - y)
    dx = max(box["x_min"] - x, 0.0, x - box["x_max"])
    dy = max(box["y_min"] - y, 0.0, y - box["y_max"])
    return math.sqrt(dx * dx + dy * dy)


def rollout_matches_prefix(rollout: dict[str, Any], desired_prefix: list[str]) -> bool:
    observed = [
        event.get("label_name")
        for event in rollout.get("label_events", [])
        if int(event.get("to", 0)) == 1
    ]
    return observed[: len(desired_prefix)] == desired_prefix


def load_task_success_by_rollout(run_dir: Path, desired_prefix: list[str]) -> tuple[dict[str, bool], float]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}, 0.0
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    task_success_by_rollout: dict[str, bool] = {}
    for rollout in summary.get("rollouts", []):
        rollout_dir = Path(rollout.get("rollout_dir", f"rollout_{int(rollout.get('rollout_idx', 0)):03d}"))
        task_success_by_rollout[rollout_dir.name] = rollout_matches_prefix(rollout, desired_prefix)

    n = len(task_success_by_rollout)
    task_rate = sum(task_success_by_rollout.values()) / n if n else 0.0
    return task_success_by_rollout, task_rate


def compute_region_safety_success(
    run_dir: Path,
    box: dict[str, Any],
    *,
    task_success_by_rollout: dict[str, bool] | None = None,
) -> tuple[float, dict[str, Any]]:
    rollout_summaries: list[dict[str, Any]] = []
    safe_count = 0
    joint_success_count = 0
    for rollout_dir in sorted((run_dir / "rollouts").glob("rollout_*")):
        points = iter_rollout_eef_xy(rollout_dir)
        inside_points = [(x, y) for x, y in points if point_inside_box(x, y, box)]
        clearances = [signed_clearance_to_box(x, y, box) for x, y in points]
        min_clearance = min(clearances) if clearances else None
        safety_success = bool(points) and not inside_points
        task_success = (
            task_success_by_rollout.get(rollout_dir.name, False)
            if task_success_by_rollout is not None
            else None
        )
        joint_success = safety_success if task_success is None else safety_success and task_success
        safe_count += int(safety_success)
        joint_success_count += int(joint_success)
        rollout_summaries.append(
            {
                "rollout": rollout_dir.name,
                "n_points": len(points),
                "inside_count": len(inside_points),
                "safety_success": safety_success,
                "task_success": task_success,
                "joint_success": joint_success,
                "min_signed_clearance_m": min_clearance,
                "first_inside_xy": list(inside_points[0]) if inside_points else None,
            }
        )

    n = len(rollout_summaries)
    rate = safe_count / n if n else 0.0
    return rate, {
        "run_dir": str(run_dir),
        "n_rollouts": n,
        "success_rate": rate,
        "safety_success_rate": rate,
        "joint_success_rate": joint_success_count / n if n else 0.0,
        "safe_count": safe_count,
        "joint_success_count": joint_success_count,
        "box": box,
        "rollouts": rollout_summaries,
    }


def collect_angle_items(run_dir: Path, group: str) -> list[dict[str, Any]]:
    return [
        item
        for path in discover_rzz_events([run_dir])
        if (item := collect_rzz_gate_angle(path, group)) is not None
    ]


def angle_rate_from_stats(stats: dict[str, Any], denominator: int) -> float:
    return float(stats["n_angle_success"] / denominator) if denominator else 0.0


def compute_angle_safety_success(
    baseline_run: Path,
    guided_run: Path,
    stl_gpc_run: Path | None,
    *,
    gate_x: float,
    threshold_deg: float,
    baseline_required_label: str = "pouring_left",
) -> tuple[float, float, float, dict[str, Any]]:
    baseline_all = collect_angle_items(baseline_run, "baseline")
    guided = collect_angle_items(guided_run, "hint2")
    stl_gpc = [] if stl_gpc_run is None else collect_angle_items(stl_gpc_run, "stl_gpc")
    baseline = filter_rzz_by_required_label_event(baseline_all, baseline_required_label)

    baseline_stats = rzz_group_stats(baseline, gate_x, threshold_deg)
    guided_stats = rzz_group_stats(guided, gate_x, threshold_deg)
    stl_gpc_stats = rzz_group_stats(stl_gpc, gate_x, threshold_deg)

    baseline_denominator = len(baseline)
    guided_denominator = len(guided)
    stl_gpc_denominator = len(stl_gpc)
    baseline_rate = angle_rate_from_stats(baseline_stats, baseline_denominator)
    guided_rate = angle_rate_from_stats(guided_stats, guided_denominator)
    stl_gpc_rate = angle_rate_from_stats(stl_gpc_stats, stl_gpc_denominator)
    baseline_stats["rate_denominator"] = baseline_denominator
    baseline_stats["success_rate"] = baseline_rate
    baseline_stats["required_label_event"] = baseline_required_label
    baseline_stats["n_all_rollouts"] = len(baseline_all)
    guided_stats["rate_denominator"] = guided_denominator
    guided_stats["success_rate"] = guided_rate
    stl_gpc_stats["rate_denominator"] = stl_gpc_denominator
    stl_gpc_stats["success_rate"] = stl_gpc_rate

    return baseline_rate, guided_rate, stl_gpc_rate, {
        "gate_x": float(gate_x),
        "threshold_deg": float(threshold_deg),
        "baseline_run": str(baseline_run),
        "guided_run": str(guided_run),
        "stl_gpc_run": None if stl_gpc_run is None else str(stl_gpc_run),
        "metric": "success iff max arccos(rzz) from first can_grabbed through first EEF x <= gate_x is <= threshold_deg",
        "baseline": baseline_stats,
        "hint2": guided_stats,
        "stl_gpc": stl_gpc_stats,
    }


def save_fig_vertical_crop(
    fig: plt.Figure,
    output_stem: Path,
    bottom: float,
    top: float,
    *,
    pad: float = FINAL_VERTICAL_CROP_PAD_IN,
) -> dict[str, float]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_w, _ = fig.get_size_inches()
    crop_bottom = max(float(bottom), float(tight.y0) - float(pad))
    crop_top = min(float(top), float(tight.y1) + float(pad))
    if crop_top <= crop_bottom:
        crop_bottom = float(bottom)
        crop_top = float(top)
    crop = Bbox.from_extents(0.0, crop_bottom, float(fig_w), crop_top)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches=crop, pad_inches=0.0)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=crop, pad_inches=0.0)
    return {
        "crop_bottom_in": float(crop_bottom),
        "crop_top_in": float(crop_top),
        "output_width_in": float(fig_w),
        "output_height_in": float(crop_top - crop_bottom),
        "pad_in": float(pad),
    }


def plot_four_square_row(args: argparse.Namespace) -> dict[str, Any]:
    configure_matplotlib()
    values = collect_values(args)
    ensure_metric_groups(values)
    values["pour_left"]["stl_gpc"] = load_observed_chain_prefix_rate(
        args.stl_gpc_left_summary,
        ["can_grabbed", "pouring_left"],
    )
    values["pour_right"]["stl_gpc"] = load_observed_chain_prefix_rate(
        args.stl_gpc_right_summary,
        ["can_grabbed", "pouring_right"],
    )
    load_values_csv(args.values_csv, values)
    region_box = load_shrunk_region_box(args.region_box_config, args.region_box_shrink_m)
    baseline_region_rate, baseline_region_summary = compute_region_safety_success(
        args.region_baseline_run,
        region_box,
    )
    stl_gpc_region_task, stl_gpc_region_task_rate = load_task_success_by_rollout(
        args.stl_gpc_region_run,
        ["can_grabbed", "pouring_left"],
    )
    stl_gpc_region_rate, stl_gpc_region_summary = compute_region_safety_success(
        args.stl_gpc_region_run,
        region_box,
        task_success_by_rollout=stl_gpc_region_task,
    )
    hint2_region_task, hint2_region_task_rate = load_task_success_by_rollout(
        args.hint2_region_run,
        ["can_grabbed", "pouring_left"],
    )
    hint2_region_rate, hint2_region_summary = compute_region_safety_success(
        args.hint2_region_run,
        region_box,
        task_success_by_rollout=hint2_region_task,
    )
    base_pour_left_rate = values["pour_left"]["base_policy"]
    values["region"]["base_policy"] = baseline_region_rate * base_pour_left_rate
    values["region"]["stl_gpc"] = stl_gpc_region_summary["joint_success_rate"]
    values["region"]["hint2"] = hint2_region_summary["joint_success_rate"]
    baseline_region_summary["base_pour_left_rate"] = base_pour_left_rate
    baseline_region_summary["base_joint_region_success_rate"] = values["region"]["base_policy"]
    stl_gpc_region_summary["task_success_rate"] = stl_gpc_region_task_rate
    hint2_region_summary["task_success_rate"] = hint2_region_task_rate
    baseline_angle_rate, hint2_angle_rate, stl_gpc_angle_rate, angle_summary = compute_angle_safety_success(
        args.angle_baseline_run,
        args.hint2_angle_run,
        args.stl_gpc_angle_run,
        gate_x=args.angle_gate_x,
        threshold_deg=args.angle_success_threshold_deg,
    )
    values["angle"]["base_policy"] = baseline_angle_rate
    values["angle"]["hint2"] = hint2_angle_rate
    values["angle"]["stl_gpc"] = stl_gpc_angle_rate

    fig = plt.figure(figsize=(TEXTWIDTH_IN, FOUR_SQUARE_ROW_HEIGHT_IN), constrained_layout=False)
    ax_behavior, ax_safety, ax_base, ax_complex = add_mixed_four_panel_row(fig)
    image_panel_aspect = FOUR_PANEL_IMAGE_WIDTH_IN / FOUR_PANEL_IMAGE_HEIGHT_IN

    plot_bar_panel(
        ax_behavior,
        values,
        title="Behavior Selection",
        groups=[("pour_left", "pour_left"), ("pour_right", "pour_right")],
        show_ylabel=True,
    )
    plot_bar_panel(
        ax_safety,
        values,
        title="Safety Constraints",
        groups=[("region", "region"), ("angle", "angle")],
        show_ylabel=False,
    )
    plot_image_or_placeholder(
        ax_base,
        "Base Policy",
        args.base_policy_image,
        "insert image\nboth behaviors",
        target_aspect=image_panel_aspect,
    )
    plot_image_or_placeholder(
        ax_complex,
        "Multi-step Tasks",
        args.complex_tasks_image,
        "insert image",
        bottom_label="cyclic",
        target_aspect=image_panel_aspect,
    )

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, FOUR_PANEL_LEGEND_BOTTOM_IN / FOUR_SQUARE_ROW_HEIGHT_IN),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#d4d4d4",
        facecolor="white",
        handlelength=1.2,
        columnspacing=0.95,
        borderpad=0.22,
    )
    legend.get_frame().set_linewidth(0.45)

    output_stem = args.output_dir / args.stem
    export_crop = save_fig_vertical_crop(
        fig,
        output_stem,
        FOUR_PANEL_CROP_BOTTOM_IN,
        FOUR_PANEL_CROP_TOP_IN,
    )
    plt.close(fig)

    return {
        "output_png": str(output_stem.with_suffix(".png")),
        "output_pdf": str(output_stem.with_suffix(".pdf")),
        "values": values,
        "stl_gpc_left_summary": str(args.stl_gpc_left_summary),
        "stl_gpc_right_summary": str(args.stl_gpc_right_summary),
        "stl_gpc_region_run": str(args.stl_gpc_region_run),
        "base_policy_image": str(args.base_policy_image) if args.base_policy_image else None,
        "base_policy_image_found": bool(args.base_policy_image and args.base_policy_image.exists()),
        "complex_tasks_image": str(args.complex_tasks_image) if args.complex_tasks_image else None,
        "complex_tasks_image_found": bool(args.complex_tasks_image and args.complex_tasks_image.exists()),
        "figure_size_in": [TEXTWIDTH_IN, FOUR_SQUARE_ROW_HEIGHT_IN],
        "export_size_in": [export_crop["output_width_in"], export_crop["output_height_in"]],
        "export_crop": export_crop,
        "bar_panel_size_in": [FOUR_PANEL_BAR_SIDE_IN, FOUR_PANEL_BAR_SIDE_IN],
        "image_panel_size_in": [FOUR_PANEL_IMAGE_WIDTH_IN, FOUR_PANEL_IMAGE_HEIGHT_IN],
        "region_safety_box": region_box,
        "region_baseline_summary": baseline_region_summary,
        "stl_gpc_region_summary": stl_gpc_region_summary,
        "hint2_region_summary": hint2_region_summary,
        "angle_summary": angle_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the real-world four-square paper row.")
    parser.add_argument("--base-policy-summary", type=Path, default=DEFAULT_BASE_POLICY_SUMMARY)
    parser.add_argument("--hint2-left-summary", type=Path, default=DEFAULT_HINT2_LEFT_SUMMARY)
    parser.add_argument("--hint2-right-summary", type=Path, default=DEFAULT_HINT2_RIGHT_SUMMARY)
    parser.add_argument("--stl-gpc-left-summary", type=Path, default=DEFAULT_STL_GPC_LEFT_SUMMARY)
    parser.add_argument("--stl-gpc-right-summary", type=Path, default=DEFAULT_STL_GPC_RIGHT_SUMMARY)
    parser.add_argument("--stl-gpc-region-run", type=Path, default=DEFAULT_STL_GPC_REGION_RUN)
    parser.add_argument(
        "--base-policy-image",
        type=Path,
        default=DEFAULT_BASE_POLICY_IMAGE,
        help="Optional image rendered inside the Base Policy square.",
    )
    parser.add_argument(
        "--complex-tasks-image",
        type=Path,
        default=DEFAULT_COMPLEX_TASKS_IMAGE,
        help="Optional image rendered inside the Complex Tasks square.",
    )
    parser.add_argument(
        "--region-baseline-run",
        type=Path,
        default=DEFAULT_REGION_BASELINE_RUN,
        help="Pour-left guided run without region safety, scored as the baseline Region bar.",
    )
    parser.add_argument(
        "--hint2-region-run",
        type=Path,
        default=DEFAULT_SAFETY_RUN,
        help="Region-safety guided run scored as the hint^2 Region bar.",
    )
    parser.add_argument("--region-box-config", type=Path, default=DEFAULT_SAFETY_RUN / "run_config.json")
    parser.add_argument("--region-box-shrink-m", type=float, default=0.0025)
    parser.add_argument("--angle-baseline-run", type=Path, default=DEFAULT_ANGLE_BASELINE_RUN)
    parser.add_argument("--hint2-angle-run", type=Path, default=DEFAULT_HINT2_ANGLE_RUN)
    parser.add_argument("--stl-gpc-angle-run", type=Path, default=DEFAULT_STL_GPC_ANGLE_RUN)
    parser.add_argument("--angle-gate-x", type=float, default=DEFAULT_ANGLE_GATE_X)
    parser.add_argument("--angle-success-threshold-deg", type=float, default=DEFAULT_ANGLE_SUCCESS_THRESHOLD_DEG)
    parser.add_argument("--values-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="real_world_four_square_row")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = plot_four_square_row(args)
    summary_path = (args.output_dir / args.stem).with_name(args.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {summary['output_png']}")
    print(f"wrote {summary['output_pdf']}")
    print(f"wrote {summary_path}")
    print(f"final height: {summary['export_size_in'][1]:.4f} in")


if __name__ == "__main__":
    main()
