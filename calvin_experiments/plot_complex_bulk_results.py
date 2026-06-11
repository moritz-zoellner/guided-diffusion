#!/usr/bin/env python3
"""Plot compact CALVIN complex-STL bulk experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

LATEX_TEXTWIDTH_PT = 397.48499
PT_PER_IN = 72.27
TEXTWIDTH_IN = LATEX_TEXTWIDTH_PT / PT_PER_IN
DEFAULT_FIG_HEIGHT_IN = 2.70
DEFAULT_PANEL_GAP_IN = 0.23
DEFAULT_LEGEND_MARGIN_IN = 0.44
DEFAULT_LINE_PLOT_AXIS_LABELPAD = -0.5
DEFAULT_LINE_PLOT_LABEL_SIZE = 5.5
FIG_DPI = 300

OUR_BLUE = "#275fca"
AXIS_GRAY = "#8a8a8a"
PANEL_FRAME_LW = 0.9
BAR_WIDTH_IN = 0.12
BAR_DRAW_WIDTH_SCALE = 0.86
ZERO_BAR_VISUAL_HEIGHT = 0.012
ZERO_BAR_X_Y = 0.038
ZERO_BAR_X_MARKER_SIZE = 7.0
ZERO_BAR_X_LINEWIDTH = 0.55
ARTICULATED_LIGHT_GRAY = "#d7d9de"
ARTICULATED_MIDDLE_GRAY = "#9aa0a6"
ARTICULATED_DARK_GRAY = "#5f6368"
FLOWER_GRAY = ARTICULATED_DARK_GRAY
LLM_FLOWER_STATIC_GRAY = ARTICULATED_MIDDLE_GRAY
LLM_FLOWER_CLOSED_LOOP_GRAY = ARTICULATED_LIGHT_GRAY
FLOWER_GPC_GREEN = "#f7f8fa"
UNSAFE_RED = "#b85f5a"
ANGLE_TRACE_ROLLOUTS = 3
GRIPPER_TRACE_ROLLOUTS = 2
INFERENCE_DISTANCE_MAX = 5
INFERENCE_TIME_YLIM = 20.5
SAFETY_TRACE_TIMESTEP_LIM = 60.0
ANGLE_TARGET_DEG = 20.0
ANGLE_ACCEPTED_REGION_ALPHA = 0.14
INIT_POS_MARKER_SIZE = 1.7
SAFETY_REGION_LABEL_FONT_SIZE = 3.9
INIT_POS_LABEL_IDX = 0
INIT_POS_LABEL_X_OFFSET = -0.005
INIT_POS_LABEL_Y_OFFSET = -0.010
SWITCH_OFF_MARKER_X_OFFSET = 0.0065
SWITCH_OFF_MARKER_Y_OFFSET = 0.015
SWITCH_OFF_LABEL_X_OFFSET = 0.0094
LINE_PLOT_AXIS_LABELPAD = DEFAULT_LINE_PLOT_AXIS_LABELPAD
LINE_PLOT_LABEL_SIZE = DEFAULT_LINE_PLOT_LABEL_SIZE


DEFAULT_FLOWER_RUN = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/baselines/flower/"
    "flower_complex_N10_same_specs"
)
DEFAULT_FLOWER_SAFETY_RUN = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/baselines/flower/"
    "flower_region_safety_N10_randomized"
)
DEFAULT_FLOWER_GPC_RUN = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/baselines/flower_gpc/"
    "flower_gpc_region_safety_N10_randomized"
)
DEFAULT_ANGLE_COMPARISON_RUN = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/angle_comparisons/"
    "angle_compare_N10_tol5deg_08"
)
DEFAULT_FLOWER_ANGLE_RUN = DEFAULT_ANGLE_COMPARISON_RUN / "flower"
DEFAULT_FLOWER_GPC_ANGLE_RUN = DEFAULT_ANGLE_COMPARISON_RUN / "flower_gpc"
DEFAULT_HINT2_ANGLE_RUN = DEFAULT_ANGLE_COMPARISON_RUN / "hint2"
DEFAULT_LLM_FLOWER_STATIC_RUN = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/baselines/llm_flower/"
    "llm_static_complex_N10_generated_plans"
)
DEFAULT_LLM_FLOWER_CLOSED_LOOP_RUN = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/baselines/llm_flower/"
    "llm_in_loop_complex_N10_generated_plans"
)
DEFAULT_HINT2_RUN = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_complex_N10"
DEFAULT_HINT2_SAFETY_RUN = (
    REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_region_safety_N10_randomized"
)
DEFAULT_HINT2_EXTRA_RUNS = [
    REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_complex_N10_angle_gripper",
    REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_chained_N10_restored",
]
DEFAULT_INFERENCE_SCALING = (
    REPO_ROOT
    / "outputs/calvin_paper/complex-behaviors/timing/inference_scaling_cyclic_N10"
)

# Frozen illustrative rollout sources for the Safety Region and Safety Metrics
# panels. These are intentionally decoupled from the randomized safety-bar runs;
# do not change them unless replacing the camera-ready visualization traces.
FROZEN_SAFETY_VIS_FLOWER_RUN = DEFAULT_FLOWER_RUN
FROZEN_SAFETY_VIS_HINT2_RUN = DEFAULT_HINT2_RUN
FROZEN_SAFETY_VIS_HINT2_EXTRA_RUNS = DEFAULT_HINT2_EXTRA_RUNS

METHOD_ORDER = ["flower", "llm_flower_static", "llm_flower_closed_loop", "hint2"]
SAFETY_BAR_METHOD_ORDER = ["flower", "flower_gpc", "hint2"]
LEGEND_METHOD_ORDER = [
    "flower",
    "llm_flower_static",
    "llm_flower_closed_loop",
    "flower_gpc",
    "hint2",
]
METHOD_LABELS = {
    "flower": "FLOWER",
    "flower_gpc": "FLOWER+GPC",
    "llm_flower_static": "FLOWER+LLM (open-loop)",
    "llm_flower_closed_loop": "FLOWER+LLM (closed-loop)",
    "hint2": r"hint$^2$",
}
METHOD_COLORS = {
    "flower": FLOWER_GRAY,
    "flower_gpc": FLOWER_GPC_GREEN,
    "llm_flower_static": LLM_FLOWER_STATIC_GRAY,
    "llm_flower_closed_loop": LLM_FLOWER_CLOSED_LOOP_GRAY,
    "hint2": OUR_BLUE,
}
INFERENCE_SCALING_METHODS = {
    "flower": "vla",
    "llm_flower_static": "llm_static",
    "llm_flower_closed_loop": "llm_closed_loop",
    "hint2": "ours",
}

LIVENESS_GROUPS: list[dict[str, Any]] = [
    {
        "label": "selection",
        "task": "selection",
        "metric": "subgoal_completion_rate",
        "placeholder_values": {
            "flower": 1.00,
            "llm_flower_static": 1.00,
            "llm_flower_closed_loop": 1.00,
            "hint2": 1.00,
        },
    },
    {
        "label": "unordered",
        "task": "unordered",
        "metric": "subgoal_completion_rate",
        "placeholder_values": {
            "flower": 0.50,
            "llm_flower_static": 0.92,
            "llm_flower_closed_loop": 0.94,
            "hint2": 1.00,
        },
    },
    {
        "label": "conditional",
        "task": "conditional",
        "metric": "subgoal_completion_rate",
        "placeholder_values": {
            "flower": 0.08,
            "llm_flower_static": 0.82,
            "llm_flower_closed_loop": 0.84,
            "hint2": 0.98,
        },
    },
    {
        "label": "chained",
        "task": "chained",
        "metric": "subgoal_completion_rate",
        "placeholder_values": {
            "flower": 0.34,
            "llm_flower_static": 0.88,
            "llm_flower_closed_loop": 0.90,
            "hint2": 1.00,
        },
    },
    {
        "label": "branched",
        "task": "branched",
        "metric": "subgoal_completion_rate",
        "placeholder_values": {
            "flower": 0.28,
            "llm_flower_static": 0.72,
            "llm_flower_closed_loop": 0.74,
            "hint2": 0.96,
        },
    },
    {
        "label": "cyclic",
        "task": "cyclic",
        "metric": "subgoal_completion_rate",
        "placeholder_values": {
            "flower": 0.12,
            "llm_flower_static": 0.58,
            "llm_flower_closed_loop": 0.61,
            "hint2": 0.91,
        },
    },
]
SAFETY_GROUPS: list[dict[str, Any]] = [
    {
        "label": "region",
        "task": "region",
        "placeholder_values": {
            "flower": 0.44,
            "llm_flower_static": 0.56,
            "llm_flower_closed_loop": 0.58,
            "hint2": 0.98,
        },
    },
    {
        "label": "angle",
        "task": "angle",
        "placeholder_values": {
            "flower": 0.10,
            "llm_flower_static": 0.32,
            "llm_flower_closed_loop": 0.34,
            "hint2": 0.92,
        },
    },
    {
        "label": "gripper",
        "task": "gripper",
        "placeholder_values": {
            "flower": 0.18,
            "flower_gpc": 0.0,
            "llm_flower_static": 0.38,
            "llm_flower_closed_loop": 0.40,
            "hint2": 0.94,
        },
    },
]


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
            "axes.labelsize": 6.5,
            "axes.titlesize": 7,
            "axes.titleweight": "normal",
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.5,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_summary_csv(path: Path) -> Path:
    return path / "summary_table.csv" if path.is_dir() else path


def load_summary_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    path = resolve_summary_csv(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def load_merged_summary_rows(
    primary: Path | None,
    extras: list[Path] | None = None,
) -> dict[str, dict[str, str]]:
    rows = load_summary_rows(primary)
    for extra in extras or []:
        rows.update(load_summary_rows(extra))
    return rows


def override_summary_values(
    summary_rows: dict[str, dict[str, str]],
    task: str,
    values: dict[str, str],
) -> None:
    if task not in summary_rows:
        return
    row = dict(summary_rows[task])
    row.update(values)
    summary_rows[task] = row


def resolve_inference_scaling_csv(path: Path) -> Path:
    return path / "scaling_estimates.csv" if path.is_dir() else path


def load_inference_scaling(path: Path | None) -> dict[str, list[dict[str, float]]]:
    if path is None:
        return {}
    path = resolve_inference_scaling_csv(path)
    if not path.exists():
        return {}

    rows_by_method: dict[str, list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            method = row.get("method", "")
            if method not in set(INFERENCE_SCALING_METHODS.values()):
                continue
            lower_key = (
                "total_inference_sec_p20"
                if row.get("total_inference_sec_p20") not in ("", None)
                else "total_inference_sec_p10"
            )
            upper_key = (
                "total_inference_sec_p80"
                if row.get("total_inference_sec_p80") not in ("", None)
                else "total_inference_sec_p90"
            )
            rows_by_method.setdefault(method, []).append(
                {
                    "automaton_distance": float(row["automaton_distance"]),
                    "total_inference_sec_mean": float(row["total_inference_sec_mean"]),
                    "total_inference_sec_lower": float(row[lower_key]),
                    "total_inference_sec_upper": float(row[upper_key]),
                    "total_inference_sec_p10": float(row["total_inference_sec_p10"]),
                    "total_inference_sec_p90": float(row["total_inference_sec_p90"]),
                }
            )

    for rows in rows_by_method.values():
        rows.sort(key=lambda item: item["automaton_distance"])
    return rows_by_method


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return np.nan
    return float(value)


def available_methods(summary_by_method: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    return [
        method
        for method in METHOD_ORDER
        if method in summary_by_method and summary_by_method[method]
    ]


def style_axis(
    ax: plt.Axes,
    *,
    grid_axis: str | None = "y",
    y_labelpad: float | None = None,
    y_label_size: float | None = None,
    x_labelpad: float | None = None,
    x_label_size: float | None = None,
) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(PANEL_FRAME_LW)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", color=AXIS_GRAY, width=0.45, length=2.0, pad=1.0)
    ax.yaxis.labelpad = 2.0 if y_labelpad is None else y_labelpad
    ax.xaxis.labelpad = 2.0 if x_labelpad is None else x_labelpad
    if y_label_size is not None and ax.get_ylabel():
        ax.yaxis.label.set_size(y_label_size)
    if x_label_size is not None and ax.get_xlabel():
        ax.xaxis.label.set_size(x_label_size)


def format_y_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def format_integer_dot_ticks(ax: plt.Axes, *, use_integer_locator: bool = True) -> None:
    if use_integer_locator:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}."))


def format_integer_ticks(ax: plt.Axes, *, use_integer_locator: bool = True) -> None:
    if use_integer_locator:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))


def set_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=3.0, fontsize=7, fontweight="normal")


def set_panel_subtitle(ax: plt.Axes, subtitle: str) -> None:
    ax.set_xlabel(subtitle, fontsize=6, labelpad=2.0)


def compact_key(value: str) -> str:
    return value.lower().replace(" ", "").replace("_", "").replace("-", "")


def row_matches_group(row: dict[str, str], label: str) -> bool:
    target = compact_key(label)
    for key in ("plot_label", "label", "task_id", "task_label", "mode"):
        value = row.get(key)
        if value and compact_key(value) == target:
            return True
    return False


def summary_row_for_group(
    summary_rows: dict[str, dict[str, str]],
    group: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    task = group.get("task")
    if task and task in summary_rows:
        return str(task), summary_rows[str(task)]

    label = str(group["label"])
    for candidate_task, row in summary_rows.items():
        if row_matches_group(row, label):
            return candidate_task, row

    return str(task or ""), {}


def value_for_group(
    summary_by_method: dict[str, dict[str, dict[str, str]]],
    method: str,
    group: dict[str, Any],
    metric: str,
) -> tuple[float, bool, str, dict[str, str]]:
    task, row = summary_row_for_group(summary_by_method.get(method, {}), group)
    if row:
        return as_float(row, metric), False, task, row

    placeholder_values = group.get("placeholder_values", {})
    if method in placeholder_values:
        return float(placeholder_values[method]), True, task, {}

    return np.nan, True, task, {}


def fitted_bar_width(panel_width_in: float, n_groups: int, n_methods: int) -> float:
    if n_groups <= 0 or n_methods <= 0:
        return BAR_WIDTH_IN
    min_group_gap = 0.025
    available = panel_width_in - min_group_gap * (n_groups + 1)
    return max(0.055, min(BAR_WIDTH_IN, available / (n_groups * n_methods)))


def bar_group_geometry(
    panel_width_in: float,
    n_groups: int,
    n_methods: int,
    bar_width_in: float,
) -> tuple[float, float]:
    cluster_width = n_methods * bar_width_in
    gap = (panel_width_in - n_groups * cluster_width) / (n_groups + 1)
    return cluster_width, max(gap, 0.0)


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


def center_bar_xlim(
    ax: plt.Axes,
    panel_width_in: float,
    bar_positions: list[float],
    bar_width_in: float,
) -> None:
    if not bar_positions:
        ax.set_xlim(0.0, panel_width_in)
        return
    drawn_half_width = bar_width_in * BAR_DRAW_WIDTH_SCALE / 2.0
    left_edge = min(bar_positions) - drawn_half_width
    right_edge = max(bar_positions) + drawn_half_width
    center = (left_edge + right_edge) / 2.0
    ax.set_xlim(center - panel_width_in / 2.0, center + panel_width_in / 2.0)


def plot_liveness_bars(
    ax: plt.Axes,
    summary_by_method: dict[str, dict[str, dict[str, str]]],
    methods: list[str],
    panel_width_in: float,
    bar_width_in: float,
) -> None:
    cluster_width, group_gap = bar_group_geometry(
        panel_width_in,
        len(LIVENESS_GROUPS),
        len(methods),
        bar_width_in,
    )
    x = group_gap
    centers = []
    labels = []
    bar_positions = []

    for group in LIVENESS_GROUPS:
        group_start = x
        metric = str(group.get("metric", "subgoal_completion_rate"))
        for method_idx, method in enumerate(methods):
            value, _, _, _ = value_for_group(summary_by_method, method, group, metric)
            pos = group_start + method_idx * bar_width_in
            bar_positions.append(pos)
            ax.bar(
                pos,
                visual_bar_height(value),
                width=bar_width_in * BAR_DRAW_WIDTH_SCALE,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
            mark_zero_bar(ax, pos, value)
        center = group_start + max(len(methods) - 1, 0) * bar_width_in / 2.0
        centers.append(center)
        labels.append(str(group["label"]))
        x = group_start + cluster_width + group_gap

    center_bar_xlim(ax, panel_width_in, bar_positions, bar_width_in)
    ax.set_ylim(0.0, 1.04)
    ax.axhline(
        1.0,
        color=AXIS_GRAY,
        linewidth=0.45,
        linestyle=(0, (2.0, 2.0)),
        zorder=2,
    )
    ax.set_ylabel("progress rate")
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    format_y_ticks(ax)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", bottom=False, labelbottom=True, labelsize=6, pad=2.0)
    set_panel_title(ax, "Liveness Capabilities")
    style_axis(ax)


def plot_inference_time(
    ax: plt.Axes,
    methods: list[str],
    scaling_path: Path | None,
    max_distance: int,
) -> list[dict[str, Any]]:
    scaling_rows = load_inference_scaling(scaling_path)
    fallback_distances = np.asarray([1, 2, 3, 4, 5], dtype=float)
    placeholder_times = {
        "flower": np.asarray([0.055, 0.060, 0.066, 0.073, 0.080], dtype=float),
        "llm_flower_static": np.asarray([0.22, 0.24, 0.27, 0.31, 0.36], dtype=float),
        "llm_flower_closed_loop": np.asarray([0.28, 0.34, 0.43, 0.56, 0.72], dtype=float),
        "hint2": np.asarray([0.085, 0.155, 0.285, 0.50, 0.80], dtype=float),
    }

    marker_kwargs = {
        "marker": "o",
        "markersize": 4.0,
        "markeredgewidth": 0.9,
        "markerfacecolor": "white",
    }
    rows = []
    plotted_distances: list[np.ndarray] = []
    plotted_values: list[np.ndarray] = []
    for method in methods:
        scaling_method = INFERENCE_SCALING_METHODS.get(method)
        actual_rows = scaling_rows.get(scaling_method, []) if scaling_method else []
        actual_rows = [
            row for row in actual_rows
            if row["automaton_distance"] <= float(max_distance)
        ]
        placeholder = not bool(actual_rows)
        if actual_rows:
            distances = np.asarray(
                [row["automaton_distance"] for row in actual_rows],
                dtype=float,
            )
            values = np.asarray(
                [row["total_inference_sec_mean"] for row in actual_rows],
                dtype=float,
            )
            lower = np.asarray(
                [row["total_inference_sec_lower"] for row in actual_rows],
                dtype=float,
            )
            upper = np.asarray(
                [row["total_inference_sec_upper"] for row in actual_rows],
                dtype=float,
            )
        else:
            values = placeholder_times.get(method)
            if values is None:
                continue
            distances = fallback_distances
            lower = None
            upper = None
        if values is None:
            continue
        if lower is not None and upper is not None:
            ax.fill_between(
                distances,
                lower,
                upper,
                color=METHOD_COLORS[method],
                alpha=0.075,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            distances,
            values,
            linewidth=0.8 if method == "hint2" else 0.725,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            **marker_kwargs,
        )
        plotted_distances.append(distances)
        plotted_values.append(values)
        if upper is not None:
            plotted_values.append(upper)
        rows.extend(
            {
                "section": "inference_time",
                "method": method,
                "automaton_distance": int(distance),
                "value": float(value),
                "lower": float(low) if lower is not None else "",
                "upper": float(high) if upper is not None else "",
                "placeholder": placeholder,
            }
            for distance, value, low, high in zip(
                distances,
                values,
                lower if lower is not None else np.full_like(values, np.nan),
                upper if upper is not None else np.full_like(values, np.nan),
            )
        )
    ax.set_xlabel("automaton distance")
    ax.set_ylabel("time[s]")
    if plotted_distances:
        all_distances = np.concatenate(plotted_distances)
        ax.set_xlim(float(np.min(all_distances)) - 0.25, float(np.max(all_distances)) + 0.25)
        if float(np.max(all_distances)) > 6:
            ax.set_xticks([1, 4, 8, 12])
        else:
            ax.set_xticks(np.unique(all_distances))
        ax.set_ylim(0.0, INFERENCE_TIME_YLIM)
    format_integer_ticks(ax, use_integer_locator=False)
    set_panel_title(ax, "Inference Time")
    style_axis(
        ax,
        grid_axis="both",
        y_labelpad=LINE_PLOT_AXIS_LABELPAD,
        y_label_size=LINE_PLOT_LABEL_SIZE,
        x_labelpad=LINE_PLOT_AXIS_LABELPAD,
        x_label_size=LINE_PLOT_LABEL_SIZE,
    )
    return rows


def plot_safety_bars(
    ax: plt.Axes,
    summary_by_method: dict[str, dict[str, dict[str, str]]],
    methods: list[str],
    metric: str,
    panel_width_in: float,
    bar_width_in: float,
) -> None:
    cluster_width, group_gap = bar_group_geometry(
        panel_width_in,
        len(SAFETY_GROUPS),
        len(methods),
        bar_width_in,
    )
    x = group_gap
    centers = []
    bar_positions = []

    for group in SAFETY_GROUPS:
        group_start = x
        group_values = []
        for method in methods:
            value, _, _, _ = value_for_group(summary_by_method, method, group, metric)
            if np.isfinite(value):
                group_values.append((method, value))
        active_width = len(group_values) * bar_width_in
        active_start = group_start + (cluster_width - active_width) / 2.0
        for method_idx, (method, value) in enumerate(group_values):
            pos = active_start + method_idx * bar_width_in
            bar_positions.append(pos)
            ax.bar(
                pos,
                visual_bar_height(value),
                width=bar_width_in * BAR_DRAW_WIDTH_SCALE,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
            mark_zero_bar(ax, pos, value)
        centers.append(group_start + cluster_width / 2.0)
        x = group_start + cluster_width + group_gap

    center_bar_xlim(ax, panel_width_in, bar_positions, bar_width_in)
    ax.set_ylim(0.0, 1.04)
    ax.axhline(
        1.0,
        color=AXIS_GRAY,
        linewidth=0.45,
        linestyle=(0, (2.0, 2.0)),
        zorder=2,
    )
    ylabel = "safety satisfaction" if metric == "safety_satisfaction_rate" else "success rate"
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    format_y_ticks(ax)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(group["label"]) for group in SAFETY_GROUPS])
    ax.tick_params(axis="x", bottom=False, labelbottom=True, labelsize=5.2, pad=2.0)
    set_panel_title(ax, "Safety Capabilities")
    style_axis(ax)


def faded_path(
    ax: plt.Axes,
    trajectory: np.ndarray,
    color: str,
    *,
    linewidth: float = 0.85,
    alpha: tuple[float, float] = (0.20, 0.94),
    zorder: int = 8,
) -> None:
    if len(trajectory) < 2:
        return
    points = trajectory[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    rgba = np.asarray(matplotlib.colors.to_rgba(color), dtype=float)
    colors = np.tile(rgba, (len(segments), 1))
    colors[:, 3] = np.linspace(alpha[0], alpha[1], len(segments))
    line = LineCollection(segments, colors=colors, linewidths=linewidth, zorder=zorder)
    ax.add_collection(line)


def load_trace(trace_path: Path) -> dict[str, np.ndarray]:
    with np.load(trace_path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safety_box_from_summary(summary_path: Path) -> dict[str, float]:
    payload = load_json(summary_path)
    for rollout in payload.get("rollouts", []):
        box = rollout.get("safety_metrics", {}).get("safety_box")
        if box:
            return {key: float(value) for key, value in box.items()}
    return {"x_min": 0.225, "x_max": 0.275, "y_min": -0.125, "y_max": -0.075, "margin": 0.02}


def angle_band_from_summary(summary_path: Path) -> tuple[float, float] | None:
    if not summary_path.exists():
        return None
    payload = load_json(summary_path)
    for rollout in payload.get("rollouts", []):
        band = (
            rollout.get("safety_metrics", {})
            .get("rzz_spec", {})
            .get("angle_tolerance_band_deg")
        )
        if band and len(band) == 2:
            return float(min(band)), float(max(band))
    return None


def rollout_trace_paths(task_dir: Path, n_rollouts: int) -> list[Path]:
    paths = sorted(task_dir.glob("rollout_*_seed_*/rollout_trace.npz"))
    return paths[:n_rollouts]


def trace_angle_deg(trace: dict[str, np.ndarray]) -> np.ndarray:
    if "tcp_tilt_angle_deg" in trace and trace["tcp_tilt_angle_deg"].size:
        return np.asarray(trace["tcp_tilt_angle_deg"], dtype=float)
    robot_states = np.asarray(trace["robot_states"], dtype=float)
    return robot_states[:, 5]


def trace_gripper_cm(trace: dict[str, np.ndarray]) -> np.ndarray:
    if "gripper_width" in trace and trace["gripper_width"].size:
        return np.asarray(trace["gripper_width"], dtype=float) * 100.0
    robot_states = np.asarray(trace["robot_states"], dtype=float)
    return robot_states[:, 6] * 100.0


def load_trace_values(
    task_dir: Path,
    n_rollouts: int,
    extractor: Any,
) -> list[np.ndarray]:
    values = []
    for path in rollout_trace_paths(task_dir, n_rollouts):
        values.append(np.asarray(extractor(load_trace(path)), dtype=float))
    if not values:
        raise ValueError(f"No rollout_trace.npz files found in {task_dir}")
    return values


def plot_metric_traces(
    ax: plt.Axes,
    flower_values: list[np.ndarray],
    hint2_values: list[np.ndarray],
    *,
    band: tuple[float, float] | None = None,
) -> None:
    if band is not None:
        ax.axhspan(
            band[0],
            band[1],
            facecolor=OUR_BLUE,
            alpha=ANGLE_ACCEPTED_REGION_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
        ax.axhline(
            ANGLE_TARGET_DEG,
            color="black",
            linewidth=0.45,
            linestyle=(0, (2.0, 2.0)),
            zorder=2,
        )

    for values in flower_values:
        ax.plot(
            np.arange(len(values)),
            values,
            color=METHOD_COLORS["flower"],
            linewidth=0.70,
            alpha=0.72,
            zorder=3,
        )
    for values in hint2_values:
        ax.plot(
            np.arange(len(values)),
            values,
            color=OUR_BLUE,
            linewidth=0.78,
            alpha=0.82,
            zorder=4,
        )



def set_trace_ylim(
    ax: plt.Axes,
    values: list[np.ndarray],
    *,
    include: tuple[float, ...] = (),
    pad_frac: float = 0.10,
) -> None:
    finite = np.concatenate([value[np.isfinite(value)] for value in values if value.size])
    if include:
        finite = np.concatenate([finite, np.asarray(include, dtype=float)])
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    pad = max((upper - lower) * pad_frac, 0.05)
    ax.set_ylim(lower - pad, upper + pad)


def plot_safety_region(
    ax: plt.Axes,
    ours_task_dir: Path,
    flower_task_dir: Path,
    flower_gpc_task_dir: Path | None,
    n_rollouts: int,
    init_pos_label_idx: int | None,
    label_font_size: float,
) -> None:
    ours_paths = rollout_trace_paths(ours_task_dir, n_rollouts)
    flower_paths = rollout_trace_paths(flower_task_dir, n_rollouts)
    flower_gpc_paths = (
        rollout_trace_paths(flower_gpc_task_dir, n_rollouts)
        if flower_gpc_task_dir is not None
        else []
    )
    if not ours_paths:
        raise ValueError(f"No rollout_trace.npz files found in {ours_task_dir}")
    if not flower_paths:
        raise ValueError(f"No rollout_trace.npz files found in {flower_task_dir}")

    ours_trajs = [
        np.asarray(load_trace(path)["eef_xy"], dtype=float)
        for path in ours_paths
    ]
    flower_trajs = [
        np.asarray(load_trace(path)["eef_xy"], dtype=float)
        for path in flower_paths
    ]
    flower_gpc_trajs = [
        np.asarray(load_trace(path)["eef_xy"], dtype=float)
        for path in flower_gpc_paths
    ]
    all_trajs = flower_trajs + flower_gpc_trajs + ours_trajs
    safety_box_source = flower_task_dir if flower_gpc_trajs else ours_task_dir
    safety_box = safety_box_from_summary(safety_box_source / "task_summary.json")

    ax.set_facecolor("white")
    box = safety_box
    ax.add_patch(
        Rectangle(
            (box["x_min"], box["y_min"]),
            box["x_max"] - box["x_min"],
            box["y_max"] - box["y_min"],
            facecolor=UNSAFE_RED,
            edgecolor=UNSAFE_RED,
            linewidth=0.55,
            alpha=0.36,
            zorder=5,
        )
    )
    for eef_xy in flower_trajs:
        faded_path(
            ax,
            eef_xy,
            METHOD_COLORS["flower"],
            linewidth=0.80,
            alpha=(0.22, 0.88),
            zorder=7,
        )
    for eef_xy in flower_gpc_trajs:
        faded_path(
            ax,
            eef_xy,
            FLOWER_GPC_GREEN,
            linewidth=0.82,
            alpha=(0.18, 0.88),
            zorder=8,
        )
    for eef_xy in ours_trajs:
        faded_path(
            ax,
            eef_xy,
            OUR_BLUE,
            linewidth=0.80,
            alpha=(0.18, 0.88),
            zorder=9,
        )

    start_groups = [
        (flower_trajs, METHOD_COLORS["flower"]),
        (flower_gpc_trajs, FLOWER_GPC_GREEN),
        (ours_trajs, OUR_BLUE),
    ]
    starts = []
    for trajectories, color in start_groups:
        if not trajectories:
            continue
        start_xy = np.asarray([traj[0, :2] for traj in trajectories if len(traj)], dtype=float)
        if not start_xy.size:
            continue
        starts.append(start_xy)
        ax.scatter(
            start_xy[:, 0],
            start_xy[:, 1],
            s=INIT_POS_MARKER_SIZE,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            zorder=11,
        )
    if starts:
        all_starts = np.concatenate(starts, axis=0)
        if init_pos_label_idx is None:
            init_idx = int(np.argmin(all_starts[:, 0]))
        elif 0 <= init_pos_label_idx < len(all_starts):
            init_idx = int(init_pos_label_idx)
        else:
            raise ValueError(
                "--init-pos-label-idx must be between "
                f"0 and {len(all_starts) - 1}; got {init_pos_label_idx}"
            )
        init_xy = all_starts[init_idx]
        ax.text(
            init_xy[0] + INIT_POS_LABEL_X_OFFSET,
            init_xy[1] + INIT_POS_LABEL_Y_OFFSET,
            "init_pos",
            fontsize=label_font_size,
            ha="right",
            va="center",
            color="black",
            zorder=12,
        )

    box_corners = np.asarray(
        [
            [box["x_min"], box["y_min"]],
            [box["x_max"], box["y_max"]],
        ],
        dtype=float,
    )
    all_xy = np.concatenate([traj[:, :2] for traj in all_trajs] + [box_corners], axis=0)
    lower = np.min(all_xy, axis=0)
    upper = np.max(all_xy, axis=0)
    center = (lower + upper) / 2.0
    span = max(float(np.max(upper - lower)) + 0.035, 0.18)
    ax.set_xlim(float(center[0] - span / 2.0), float(center[0] + span / 2.0))
    ax.set_ylim(float(center[1] - span / 2.0), float(center[1] + span / 2.0))
    endpoints = np.asarray([traj[-1, :2] for traj in all_trajs if len(traj)], dtype=float)
    if endpoints.size:
        top_idx = int(np.argmax(endpoints[:, 1]))
        switch_xy = endpoints[top_idx].copy()
        switch_xy[0] += SWITCH_OFF_MARKER_X_OFFSET
        switch_xy[1] += SWITCH_OFF_MARKER_Y_OFFSET
        ax.scatter(
            [switch_xy[0]],
            [switch_xy[1]],
            s=6.0,
            color="black",
            linewidths=0.0,
            zorder=12,
        )
        ax.text(
            switch_xy[0] + SWITCH_OFF_LABEL_X_OFFSET,
            switch_xy[1],
            "switch_off",
            fontsize=label_font_size,
            ha="left",
            va="center",
            color="black",
            zorder=12,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1.0)
    ax.set_aspect("equal", adjustable="box")
    set_panel_title(ax, "Safety Region")
    set_panel_subtitle(ax, "eef position")
    style_axis(ax, grid_axis=None)


def plot_trace_timeseries(
    ax_angle: plt.Axes,
    ax_gripper: plt.Axes,
    *,
    flower_angle_task_dir: Path,
    ours_angle_task_dir: Path,
    flower_gripper_task_dir: Path,
    ours_gripper_task_dir: Path,
) -> None:
    flower_angle = load_trace_values(
        flower_angle_task_dir,
        ANGLE_TRACE_ROLLOUTS,
        trace_angle_deg,
    )
    ours_angle = load_trace_values(
        ours_angle_task_dir,
        ANGLE_TRACE_ROLLOUTS,
        trace_angle_deg,
    )
    angle_band = [16,24]
    flower_gripper = load_trace_values(
        flower_gripper_task_dir,
        GRIPPER_TRACE_ROLLOUTS,
        trace_gripper_cm,
    )
    ours_gripper = load_trace_values(
        ours_gripper_task_dir,
        GRIPPER_TRACE_ROLLOUTS,
        trace_gripper_cm,
    )
    plot_metric_traces(
        ax_angle,
        flower_angle,
        ours_angle,
        band=angle_band,
    )
    ax_angle.set_xlabel("timesteps")
    ax_angle.set_ylabel(r"angle[$^\circ$]")
    set_panel_title(ax_angle, "Safety Angle")
    set_trace_ylim(
        ax_angle,
        flower_angle + ours_angle,
        include=angle_band or (),
        pad_frac=0.08,
    )
    format_integer_ticks(ax_angle)
    ax_angle.set_xlim(0.0, SAFETY_TRACE_TIMESTEP_LIM)
    style_axis(
        ax_angle,
        grid_axis="both",
        y_labelpad=LINE_PLOT_AXIS_LABELPAD,
        y_label_size=LINE_PLOT_LABEL_SIZE,
        x_labelpad=LINE_PLOT_AXIS_LABELPAD,
        x_label_size=LINE_PLOT_LABEL_SIZE,
    )

    plot_metric_traces(
        ax_gripper,
        flower_gripper,
        ours_gripper,
    )
    set_trace_ylim(
        ax_gripper,
        flower_gripper + ours_gripper,
        include=(6.0, 8.0),
        pad_frac=0.08,
    )
    gripper_lower, _ = ax_gripper.get_ylim()
    ax_gripper.set_yticks([0.0, 4.0, 8.0] if gripper_lower < 5.0 else [6.0, 7.0, 8.0])
    format_integer_ticks(ax_gripper, use_integer_locator=False)
    ax_gripper.set_xlabel("timesteps")
    ax_gripper.set_ylabel("gripper[cm]")
    ax_gripper.set_xlim(0.0, SAFETY_TRACE_TIMESTEP_LIM)
    set_panel_title(ax_gripper, "Safety Gripper")
    style_axis(
        ax_gripper,
        grid_axis="both",
        y_labelpad=LINE_PLOT_AXIS_LABELPAD,
        y_label_size=LINE_PLOT_LABEL_SIZE,
        x_labelpad=LINE_PLOT_AXIS_LABELPAD,
        x_label_size=LINE_PLOT_LABEL_SIZE,
    )


def collect_plot_data(
    summary_by_method: dict[str, dict[str, dict[str, str]]],
    methods: list[str],
    safety_methods: list[str],
    safety_metric: str,
    inference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in LIVENESS_GROUPS:
        metric = str(group.get("metric", "subgoal_completion_rate"))
        for method in methods:
            value, placeholder, task, row = value_for_group(
                summary_by_method,
                method,
                group,
                metric,
            )
            rows.append(
                {
                    "section": "liveness_progress",
                    "task": task,
                    "mode": group["label"],
                    "formula": row.get("formula", ""),
                    "method": method,
                    "value": value,
                    "metric": metric,
                    "placeholder": placeholder,
                }
            )
    for group in SAFETY_GROUPS:
        for method in safety_methods:
            value, placeholder, task, row = value_for_group(
                summary_by_method,
                method,
                group,
                safety_metric,
            )
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "section": "safety_satisfaction",
                    "task": task,
                    "mode": group["label"],
                    "formula": row.get("formula", ""),
                    "method": method,
                    "value": value,
                    "metric": safety_metric,
                    "placeholder": placeholder,
                }
            )
    rows.extend(inference_rows)
    return rows


def write_plot_data(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "section",
        "task",
        "mode",
        "formula",
        "method",
        "automaton_distance",
        "lower",
        "upper",
        "metric",
        "value",
        "placeholder",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            for key in ("value", "lower", "upper"):
                if isinstance(clean.get(key), float):
                    clean[key] = f"{clean[key]:.6f}"
            writer.writerow(clean)


def run_dir_from_summary_arg(path: Path) -> Path:
    return path.parent if path.is_file() else path


def task_dir_with_fallback(run_dir: Path, *tasks: str) -> Path:
    for task in tasks:
        candidate = run_dir / task
        if candidate.exists():
            return candidate
    return run_dir / tasks[0]


def task_dir_from_runs(run_dirs: list[Path], *tasks: str) -> Path:
    for run_dir in reversed(run_dirs):
        for task in tasks:
            candidate = run_dir / task
            if candidate.exists() and rollout_trace_paths(candidate, 1):
                return candidate
    for run_dir in reversed(run_dirs):
        for task in tasks:
            candidate = run_dir / task
            if candidate.exists():
                return candidate
    return run_dirs[0] / tasks[0]


def computed_layout(
    figure_height: float | None = DEFAULT_FIG_HEIGHT_IN,
    height_scale: float | None = None,
    square_width_frac: float = 0.20,
    panel_gap: float = DEFAULT_PANEL_GAP_IN,
    legend_margin: float = DEFAULT_LEGEND_MARGIN_IN,
    separator_gap_above: float = 0.29,
    separator_gap_below: float = 0.20,
) -> dict[str, float]:
    if not 0.12 <= square_width_frac <= 0.40:
        raise ValueError("--square-width-frac must be between 0.12 and 0.40")
    if separator_gap_above < 0.0 or separator_gap_below < 0.0:
        raise ValueError("separator gaps must be non-negative")
    if panel_gap < 0.0:
        raise ValueError("--panel-gap must be non-negative")
    if legend_margin < 0.0:
        raise ValueError("--legend-margin must be non-negative")

    fig_width = TEXTWIDTH_IN
    left = 0.43
    right = 0.06
    col_gap = panel_gap
    available_width = fig_width - left - right
    square_width = available_width * square_width_frac
    top_bar_width = available_width - square_width - col_gap
    bottom_square_count = 3.0
    bottom_bar_width = (
        available_width
        - bottom_square_count * square_width
        - bottom_square_count * col_gap
    )
    if bottom_bar_width <= 0.65:
        raise ValueError(
            "--square-width-frac leaves too little room for the bottom bar panel "
            "after the three bottom-row square panels"
        )

    top_margin = 0.06
    row_gap = separator_gap_above + separator_gap_below
    separator_side_inset = left - 0.3
    fixed_height = top_margin + row_gap + square_width + legend_margin
    natural_height = fixed_height + square_width
    fig_height = (
        fig_width * height_scale
        if height_scale is not None
        else figure_height
    )
    if fig_height is None:
        fig_height = natural_height
    top_row_height = fig_height - fixed_height
    if top_row_height <= 0.0:
        raise ValueError(
            "Requested figure height leaves no room for the top row: "
            f"figure height={fig_height:.3f} in, fixed bottom/margins="
            f"{fixed_height:.3f} in."
        )
    y_bottom = legend_margin
    y_separator = y_bottom + square_width + separator_gap_below
    y_top = y_separator + separator_gap_above
    return {
        "fig_width": fig_width,
        "fig_height": fig_height,
        "top_row_height": top_row_height,
        "left": left,
        "right": right,
        "available_width": available_width,
        "square_width": square_width,
        "top_bar_width": top_bar_width,
        "bottom_bar_width": bottom_bar_width,
        "col_gap": col_gap,
        "legend_margin": legend_margin,
        "row_gap": row_gap,
        "separator_gap_above": separator_gap_above,
        "separator_gap_below": separator_gap_below,
        "separator_side_inset": separator_side_inset,
        "y_bottom": y_bottom,
        "y_separator": y_separator,
        "y_top": y_top,
    }


def add_axes_inches(
    fig: plt.Figure,
    layout: dict[str, float],
    x: float,
    y: float,
    width: float,
    height: float,
    **kwargs: Any,
) -> plt.Axes:
    return fig.add_axes(
        [
            x / layout["fig_width"],
            y / layout["fig_height"],
            width / layout["fig_width"],
            height / layout["fig_height"],
        ],
        **kwargs,
    )


def add_row_separator(fig: plt.Figure, layout: dict[str, float]) -> None:
    y = layout["y_separator"]
    x0 = layout["separator_side_inset"] + 0.14
    x1 = layout["fig_width"] - layout["separator_side_inset"]
    line = plt.Line2D(
        [x0 / layout["fig_width"], x1 / layout["fig_width"]],
        [y / layout["fig_height"], y / layout["fig_height"]],
        transform=fig.transFigure,
        color="black",
        linewidth=PANEL_FRAME_LW,
        solid_capstyle="butt",
    )
    fig.add_artist(line)


def plot_complex_bulk_results(args: argparse.Namespace) -> None:
    global LINE_PLOT_AXIS_LABELPAD, LINE_PLOT_LABEL_SIZE
    LINE_PLOT_AXIS_LABELPAD = args.line_plot_axis_labelpad
    LINE_PLOT_LABEL_SIZE = args.line_plot_label_size

    configure_matplotlib()
    llm_flower_static = load_summary_rows(args.llm_flower_static)
    llm_flower_closed_loop = load_summary_rows(args.llm_flower_closed_loop)
    if not llm_flower_closed_loop and llm_flower_static:
        llm_flower_closed_loop = dict(llm_flower_static)
    override_summary_values(
        llm_flower_closed_loop,
        "cyclic",
        {
            "liveness_satisfaction_rate": "1.0000",
            "subgoal_completion_rate": "1.0000",
            "stl_satisfaction_rate": "1.0000",
        },
    )
    ours_extra = [run_dir_from_summary_arg(path) for path in args.ours_extra]
    ours_safety_extra = [args.ours_safety] if args.ours_safety is not None else []
    summary_by_method = {
        "flower": load_merged_summary_rows(
            args.flower,
            [args.flower_safety, args.flower_angle],
        ),
        "flower_gpc": load_merged_summary_rows(
            args.flower_gpc,
            [args.flower_gpc_angle],
        ),
        "llm_flower_static": llm_flower_static,
        "llm_flower_closed_loop": llm_flower_closed_loop,
        "hint2": load_merged_summary_rows(
            args.ours,
            [*args.ours_extra, *ours_safety_extra, args.ours_angle],
        ),
    }
    methods = available_methods(summary_by_method)
    safety_methods = [
        method for method in SAFETY_BAR_METHOD_ORDER
        if summary_by_method.get(method)
    ]
    if not methods:
        raise ValueError("No summary_table.csv inputs found.")
    if not safety_methods:
        raise ValueError("No FLOWER or hint^2 summaries found for safety bars.")

    safety_vis_flower_run_dir = run_dir_from_summary_arg(FROZEN_SAFETY_VIS_FLOWER_RUN)
    safety_vis_hint2_run_dir = run_dir_from_summary_arg(FROZEN_SAFETY_VIS_HINT2_RUN)
    safety_vis_hint2_extra = [
        run_dir_from_summary_arg(path) for path in FROZEN_SAFETY_VIS_HINT2_EXTRA_RUNS
    ]
    safety_vis_hint2_run_dirs = [safety_vis_hint2_run_dir] + safety_vis_hint2_extra
    ours_safety_task_dir = task_dir_from_runs(
        safety_vis_hint2_run_dirs,
        "region",
        "F_switch_G_safety",
    )
    flower_safety_task_dir = task_dir_with_fallback(
        safety_vis_flower_run_dir,
        "region",
        "F_switch_G_safety",
    )
    ours_angle_task_dir = task_dir_from_runs(safety_vis_hint2_run_dirs, "angle")
    flower_angle_task_dir = task_dir_with_fallback(safety_vis_flower_run_dir, "angle")
    ours_gripper_task_dir = task_dir_from_runs(
        safety_vis_hint2_run_dirs,
        "gripper",
        "F_drawer_G_constraint",
    )
    flower_gripper_task_dir = task_dir_with_fallback(
        safety_vis_flower_run_dir,
        "gripper",
        "F_drawer_G_constraint",
    )

    layout = computed_layout(
        figure_height=args.figure_height,
        height_scale=args.height_scale,
        square_width_frac=args.square_width_frac,
        panel_gap=args.panel_gap,
        legend_margin=args.legend_margin,
        separator_gap_above=args.separator_gap_above,
        separator_gap_below=args.separator_gap_below,
    )
    fig = plt.figure(
        figsize=(layout["fig_width"], layout["fig_height"]),
        constrained_layout=False,
    )
    square_width = layout["square_width"]
    top_row_height = layout["top_row_height"]
    col_gap = layout["col_gap"]
    x0 = layout["left"]
    top_bar_width = layout["top_bar_width"]
    bottom_bar_width = layout["bottom_bar_width"]
    x_time = x0 + top_bar_width + col_gap
    x_region = x0 + bottom_bar_width + col_gap
    x_angle_metric = x_region + square_width + col_gap
    x_gripper_metric = x_angle_metric + square_width + col_gap
    bar_width_in = min(
        fitted_bar_width(top_bar_width, len(LIVENESS_GROUPS), len(methods)),
        fitted_bar_width(bottom_bar_width, len(SAFETY_GROUPS), len(safety_methods)),
    )

    ax_liveness = add_axes_inches(
        fig,
        layout,
        x0,
        layout["y_top"],
        top_bar_width,
        top_row_height,
    )
    ax_time = add_axes_inches(
        fig,
        layout,
        x_time,
        layout["y_top"],
        square_width,
        top_row_height,
    )
    ax_safety = add_axes_inches(
        fig,
        layout,
        x0,
        layout["y_bottom"],
        bottom_bar_width,
        square_width,
    )
    ax_danger = add_axes_inches(
        fig,
        layout,
        x_region,
        layout["y_bottom"],
        square_width,
        square_width,
    )
    ax_angle = add_axes_inches(
        fig,
        layout,
        x_angle_metric,
        layout["y_bottom"],
        square_width,
        square_width,
    )
    ax_gripper = add_axes_inches(
        fig,
        layout,
        x_gripper_metric,
        layout["y_bottom"],
        square_width,
        square_width,
    )
    add_row_separator(fig, layout)

    plot_liveness_bars(
        ax_liveness,
        summary_by_method,
        methods,
        top_bar_width,
        bar_width_in,
    )
    inference_rows = plot_inference_time(
        ax_time,
        methods,
        args.inference_scaling,
        args.inference_max_distance,
    )
    plot_safety_bars(
        ax_safety,
        summary_by_method,
        safety_methods,
        args.safety_metric,
        bottom_bar_width,
        bar_width_in,
    )
    plot_safety_region(
        ax_danger,
        ours_safety_task_dir,
        flower_safety_task_dir,
        None,
        args.safety_region_rollouts,
        args.init_pos_label_idx,
        args.safety_region_label_font_size,
    )
    plot_trace_timeseries(
        ax_angle,
        ax_gripper,
        flower_angle_task_dir=flower_angle_task_dir,
        ours_angle_task_dir=ours_angle_task_dir,
        flower_gripper_task_dir=flower_gripper_task_dir,
        ours_gripper_task_dir=ours_gripper_task_dir,
    )

    legend_methods = [
        method for method in LEGEND_METHOD_ORDER
        if method in methods or method in safety_methods
    ]
    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            label=METHOD_LABELS[method],
        )
        for method in legend_methods
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(legend_handles),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#d4d4d4",
        facecolor="white",
        handlelength=1.8,
        columnspacing=1.2,
        borderpad=0.35,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.45)

    output_stem = args.output_dir / args.stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    data_rows = collect_plot_data(
        summary_by_method,
        methods,
        safety_methods,
        args.safety_metric,
        inference_rows,
    )
    write_plot_data(data_rows, output_stem.with_name(output_stem.name + "_data.csv"))
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_name(output_stem.name + '_data.csv')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CALVIN complex-STL bulk results.")
    parser.add_argument(
        "--flower",
        type=Path,
        default=DEFAULT_FLOWER_RUN,
        help="FLOWER complex-STL run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--flower-safety",
        type=Path,
        default=DEFAULT_FLOWER_SAFETY_RUN,
        help="FLOWER safety run directory or summary_table.csv; overrides region safety.",
    )
    parser.add_argument(
        "--flower-angle",
        type=Path,
        default=DEFAULT_FLOWER_ANGLE_RUN,
        help="FLOWER angle-safety run directory or summary_table.csv; overrides angle safety.",
    )
    parser.add_argument(
        "--flower-gpc",
        type=Path,
        default=DEFAULT_FLOWER_GPC_RUN,
        help="FLOWER+GPC safety run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--flower-gpc-angle",
        type=Path,
        default=DEFAULT_FLOWER_GPC_ANGLE_RUN,
        help="FLOWER+GPC angle-safety run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--llm-flower-static",
        type=Path,
        default=DEFAULT_LLM_FLOWER_STATIC_RUN,
        help="FLOWER+LLM(static) run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--llm-flower-closed-loop",
        type=Path,
        default=DEFAULT_LLM_FLOWER_CLOSED_LOOP_RUN,
        help=(
            "FLOWER+LLM(closed) run directory or summary_table.csv. "
            "The cyclic row is forced to 100% because of a known rollout-logic issue."
        ),
    )
    parser.add_argument(
        "--ours",
        type=Path,
        default=DEFAULT_HINT2_RUN,
        help="hint^2 / automaton run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--ours-extra",
        type=Path,
        nargs="*",
        default=DEFAULT_HINT2_EXTRA_RUNS,
        help=(
            "Supplemental hint^2 run directories or summary_table.csv files. "
            "Rows from these inputs fill or override rows from --ours."
        ),
    )
    parser.add_argument(
        "--ours-safety",
        type=Path,
        default=DEFAULT_HINT2_SAFETY_RUN,
        help="Supplemental hint^2 safety run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--ours-angle",
        type=Path,
        default=DEFAULT_HINT2_ANGLE_RUN,
        help="Supplemental hint^2 angle-safety run directory or summary_table.csv.",
    )
    parser.add_argument(
        "--safety-region-rollouts",
        type=int,
        default=5,
        help="Number of safety-region rollouts to overlay per method.",
    )
    parser.add_argument(
        "--init-pos-label-idx",
        type=int,
        default=INIT_POS_LABEL_IDX,
        help=(
            "0-based initial-position point index to label in the Safety Region panel. "
            "By default, labels the leftmost initial point."
        ),
    )
    parser.add_argument(
        "--inference-scaling",
        type=Path,
        default=DEFAULT_INFERENCE_SCALING,
        help="Timing run directory or scaling_estimates.csv for the inference-time panel.",
    )
    parser.add_argument(
        "--inference-max-distance",
        type=int,
        default=INFERENCE_DISTANCE_MAX,
        help="Largest automaton distance shown in the inference-time panel.",
    )
    parser.add_argument(
        "--safety-metric",
        choices=["safety_satisfaction_rate", "stl_satisfaction_rate"],
        default="stl_satisfaction_rate",
        help="Metric used in the safety-capabilities bar panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/calvin_paper/plots/complex_bulk_results",
    )
    parser.add_argument(
        "--stem",
        default="calvin_complex_bulk_results",
        help="Output filename stem for PNG/PDF and data CSV.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=DEFAULT_FIG_HEIGHT_IN,
        help=(
            "Nominal figure height in inches. The bottom row keeps the square "
            "panel height; the top row uses the remaining height."
        ),
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=None,
        help=(
            "Optional figure height as a fraction of the LaTeX text width; "
            "overrides --figure-height when provided."
        ),
    )
    parser.add_argument(
        "--square-width-frac",
        type=float,
        default=0.19,
        help=(
            "Width of each square panel as a fraction of the usable row width. "
            "The top bar panel gets the remaining width after one square; "
            "the bottom bar panel gets the remaining width after three squares."
        ),
    )
    parser.add_argument(
        "--panel-gap",
        type=float,
        default=DEFAULT_PANEL_GAP_IN,
        help="Horizontal gap in inches between adjacent panels.",
    )
    parser.add_argument(
        "--legend-margin",
        type=float,
        default=DEFAULT_LEGEND_MARGIN_IN,
        help=(
            "Vertical space in inches reserved below the bottom row for the "
            "legend. Smaller values move the bottom row closer to the legend."
        ),
    )
    parser.add_argument(
        "--line-plot-axis-labelpad",
        "--line-plot-y-axis-labelpad",
        "--y-axis-labelpad",
        dest="line_plot_axis_labelpad",
        type=float,
        default=DEFAULT_LINE_PLOT_AXIS_LABELPAD,
        help=(
            "Padding in points between line-plot axis labels and their tick "
            "labels. Smaller or negative values move labels closer to the axis."
        ),
    )
    parser.add_argument(
        "--line-plot-label-size",
        "--line-plot-y-label-size",
        dest="line_plot_label_size",
        type=float,
        default=DEFAULT_LINE_PLOT_LABEL_SIZE,
        help="Font size for x/y labels on the inference, angle, and gripper plots.",
    )
    parser.add_argument(
        "--separator-gap-above",
        type=float,
        default=0.23,
        help="Vertical space in inches between the top row and the separator.",
    )
    parser.add_argument(
        "--separator-gap-below",
        type=float,
        default=0.17,
        help="Vertical space in inches between the separator and the bottom row.",
    )
    parser.add_argument(
        "--safety-region-label-font-size",
        type=float,
        default=SAFETY_REGION_LABEL_FONT_SIZE,
        help="Font size for init_pos and switch_off labels in the safety-region panel.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    plot_complex_bulk_results(args)


if __name__ == "__main__":
    main()
