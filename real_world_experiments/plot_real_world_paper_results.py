#!/usr/bin/env python3
"""Plot compact real-world Cheez-It rollout result bars for the paper."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

LATEX_TEXTWIDTH_PT = 397.48499
PT_PER_IN = 72.27
TEXTWIDTH_IN = LATEX_TEXTWIDTH_PT / PT_PER_IN
FIG_DPI = 300

OUR_BLUE = "#275fca"
AXIS_GRAY = "#8a8a8a"
PANEL_FRAME_LW = 0.9
BAR_WIDTH_IN = 0.12
BAR_DRAW_WIDTH_SCALE = 0.86

METHOD_ORDER = ["base_policy", "stl_gpc", "hint2"]
METHOD_LABELS = {
    "base_policy": "Base Policy",
    "stl_gpc": "STL-GPC",
    "hint2": r"hint$^2$",
}
METHOD_COLORS = {
    "base_policy": "#d7d9de",
    "stl_gpc": "#5f6368",
    "hint2": OUR_BLUE,
}

PANELS = [
    {
        "title": "Behavior Selection",
        "groups": [
            ("pour_left", "Pour Left"),
            ("pour_right", "Pour Right"),
        ],
    },
    {
        "title": "Complex Instructions",
        "groups": [
            ("cyclic", "Cyclic"),
            ("safety", "Safety"),
        ],
    },
]

DEFAULT_BASE_POLICY_SUMMARY = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/base_dp_eval/base_dp_epoch160_n5_8/summary.json"
)
DEFAULT_HINT2_LEFT_SUMMARY = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_1/summary.json"
)
DEFAULT_HINT2_RIGHT_SUMMARY = (
    REPO_ROOT
    / "outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_2/summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/real_world/paper_plots/behavior_instruction_bars"


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
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, show_ylabel: bool) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(PANEL_FRAME_LW)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", color=AXIS_GRAY, width=0.45, length=2.0, pad=1.0)
    ax.set_ylim(0.0, 1.04)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)])
    ax.axhline(1.0, color=AXIS_GRAY, linewidth=0.45, linestyle=(0, (2.0, 2.0)), zorder=2)
    if show_ylabel:
        ax.set_ylabel("success rate")
    else:
        ax.tick_params(axis="y", labelleft=False)


def empty_values() -> dict[str, dict[str, float]]:
    return {
        group_key: {method: 0.0 for method in METHOD_ORDER}
        for panel in PANELS
        for group_key, _ in panel["groups"]
    }


def load_base_behavior_rates(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        return {"pour_left": 0.0, "pour_right": 0.0}
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    rollouts = summary.get("rollouts", [])
    if not rollouts:
        return {"pour_left": 0.0, "pour_right": 0.0}

    counts = {"pour_left": 0, "pour_right": 0}
    outcome_count = 0
    for rollout in rollouts:
        event_names = {event.get("label_name") for event in rollout.get("label_events", [])}
        final_label = rollout.get("final_label") or []
        reached_left = "pouring_left" in event_names or (len(final_label) > 2 and int(final_label[2]) == 1)
        reached_right = "pouring_right" in event_names or (len(final_label) > 1 and int(final_label[1]) == 1)
        if not reached_left and not reached_right:
            continue
        outcome_count += 1
        if reached_left:
            counts["pour_left"] += 1
        if reached_right:
            counts["pour_right"] += 1
    if outcome_count == 0:
        return {"pour_left": 0.0, "pour_right": 0.0}
    return {key: counts[key] / outcome_count for key in counts}


def load_observed_chain_prefix_rate(summary_path: Path | None, desired_prefix: list[str]) -> float:
    if summary_path is None or not summary_path.exists():
        return 0.0
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rollouts = summary.get("rollouts", [])
    if not rollouts:
        return 0.0
    successes = 0
    for rollout in rollouts:
        observed = [
            event.get("label_name")
            for event in rollout.get("label_events", [])
            if int(event.get("to", 0)) == 1
        ]
        if observed[: len(desired_prefix)] == desired_prefix:
            successes += 1
    return successes / len(rollouts)


def load_values_csv(path: Path | None, values: dict[str, dict[str, float]]) -> None:
    if path is None or not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group = row["group"].strip()
            method = row["method"].strip()
            if group in values and method in values[group]:
                values[group][method] = float(row["success_rate"])


def collect_values(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    values = empty_values()
    base_rates = load_base_behavior_rates(args.base_policy_summary)
    values["pour_left"]["base_policy"] = base_rates["pour_left"]
    values["pour_right"]["base_policy"] = base_rates["pour_right"]
    values["pour_left"]["hint2"] = load_observed_chain_prefix_rate(
        args.hint2_left_summary,
        ["can_grabbed", "pouring_left"],
    )
    values["pour_right"]["hint2"] = load_observed_chain_prefix_rate(
        args.hint2_right_summary,
        ["can_grabbed", "pouring_right"],
    )
    load_values_csv(args.values_csv, values)
    return values


def write_values_csv(values: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_by_group = {
        group_key: group_label
        for panel in PANELS
        for group_key, group_label in panel["groups"]
    }
    panel_by_group = {
        group_key: panel["title"]
        for panel in PANELS
        for group_key, _ in panel["groups"]
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["panel", "group", "group_label", "method", "method_label", "success_rate"],
        )
        writer.writeheader()
        for panel in PANELS:
            for group_key, _ in panel["groups"]:
                for method in METHOD_ORDER:
                    writer.writerow(
                        {
                            "panel": panel_by_group[group_key],
                            "group": group_key,
                            "group_label": label_by_group[group_key],
                            "method": method,
                            "method_label": METHOD_LABELS[method].replace("$", ""),
                            "success_rate": f"{values[group_key][method]:.6f}",
                        }
                    )


def plot_panel(
    ax: plt.Axes,
    panel: dict,
    values: dict[str, dict[str, float]],
    *,
    show_ylabel: bool,
) -> None:
    groups = panel["groups"]
    bar_width = BAR_WIDTH_IN
    cluster_width = len(METHOD_ORDER) * bar_width
    group_gap = 0.25
    centers = []
    labels = []
    x = 0.0

    for group_key, group_label in groups:
        group_start = x
        for method_idx, method in enumerate(METHOD_ORDER):
            pos = group_start + method_idx * bar_width
            value = values[group_key][method]
            ax.bar(
                pos,
                value,
                width=bar_width * BAR_DRAW_WIDTH_SCALE,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
        centers.append(group_start + (len(METHOD_ORDER) - 1) * bar_width / 2.0)
        labels.append(group_label)
        x = group_start + cluster_width + group_gap

    ax.set_xlim(-0.12, x - group_gap + 0.12)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", bottom=False, labelbottom=True, pad=2.0)
    ax.set_title(panel["title"], pad=3.0)
    style_axis(ax, show_ylabel=show_ylabel)


def plot_results(values: dict[str, dict[str, float]], output_stem: Path, height_scale: float) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TEXTWIDTH_IN, float(height_scale) * TEXTWIDTH_IN),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.12},
    )
    for idx, panel in enumerate(PANELS):
        plot_panel(axes[idx], panel, values, show_ylabel=(idx == 0))

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
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(METHOD_ORDER),
        frameon=True,
        handlelength=1.25,
        handleheight=0.75,
        columnspacing=1.4,
        borderpad=0.35,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.45)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.28)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real-world behavior-selection and complex-instruction success bars."
    )
    parser.add_argument("--base-policy-summary", type=Path, default=DEFAULT_BASE_POLICY_SUMMARY)
    parser.add_argument("--hint2-left-summary", type=Path, default=DEFAULT_HINT2_LEFT_SUMMARY)
    parser.add_argument("--hint2-right-summary", type=Path, default=DEFAULT_HINT2_RIGHT_SUMMARY)
    parser.add_argument(
        "--values-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns group,method,success_rate to override/append values.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="real_world_behavior_instruction_bars")
    parser.add_argument(
        "--height-scale",
        type=float,
        default=0.31,
        help="Figure height as a fraction of the LaTeX text width.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    values = collect_values(args)
    output_stem = args.output_dir / args.stem
    write_values_csv(values, output_stem.with_name(output_stem.name + "_data.csv"))
    plot_results(values, output_stem, args.height_scale)
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_name(output_stem.name + '_data.csv')}")


if __name__ == "__main__":
    main()
