#!/usr/bin/env python3
"""Plot CALVIN articulated-object success rates for the paper."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]

TASK_ORDER = [
    "button_on",
    "button_off",
    "switch_on",
    "switch_off",
    "drawer_open",
    "drawer_close",
    "door_left",
    "door_right",
]
TASK_ORDER_WITH_AVG = TASK_ORDER + ["average"]
TASK_LABELS = {
    "button_on": "Button On",
    "button_off": "Button Off",
    "switch_on": "Switch On",
    "switch_off": "Switch Off",
    "drawer_open": "Drawer Open",
    "drawer_close": "Drawer Close",
    "door_left": "Door Left",
    "door_right": "Door Right",
    "average": "Average",
}

METHOD_ORDER = ["base_policy", "itps", "dynaguide", "flower", "hint2"]
METHOD_LABELS = {
    "base_policy": "base policy",
    "itps": "ITPS",
    "dynaguide": "DynaGuide",
    "flower": "FLOWER",
    "hint2": r"hint$^2$",
}
METHOD_CSV_LABELS = {
    "base_policy": "base_policy",
    "itps": "ITPS",
    "dynaguide": "DynaGuide",
    "flower": "FLOWER",
    "hint2": "hint2",
}

DYNA_METHOD_MAP = {
    "base_policy": "Unguided",
    "itps": "ITPS",
    "dynaguide": "Dynaguide-Steering",
}

LATEX_TEXTWIDTH_PT = 397.48499
PT_PER_IN = 72.27
TEXTWIDTH_IN = LATEX_TEXTWIDTH_PT / PT_PER_IN
FIG_DPI = 300

OUR_BLUE = "#275fca"
LTLDOG_GRAY = "#5f6368"
LTLDOG_GRAY_LIGHT = "#9aa0a6"
AXIS_GRAY = "#8a8a8a"
PANEL_FRAME_LW = 0.9

METHOD_COLORS = {
    "base_policy": "#c8cbd0",
    "itps": LTLDOG_GRAY_LIGHT,
    "dynaguide": "#7a7f85",
    "flower": LTLDOG_GRAY,
    "hint2": OUR_BLUE,
}


def configure_matplotlib():
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
            "ytick.labelsize": 6,
            "legend.fontsize": 5.5,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_summary_csv(path):
    path = Path(path)
    if path.is_dir():
        return path / "summary_table.csv"
    return path


def read_summary_rates(path):
    path = resolve_summary_csv(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    rates = {}
    for row in rows:
        task = row["task"].strip()
        if task in TASK_ORDER:
            rates[task] = float(row["success_rate"])

    missing = [task for task in TASK_ORDER if task not in rates]
    if missing:
        raise ValueError(f"{path} is missing tasks: {', '.join(missing)}")

    rates["average"] = sum(rates[task] for task in TASK_ORDER) / len(TASK_ORDER)
    return rates


def read_dynaguide_rates(path):
    method_blocks = {}
    current_method = None

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for raw_row in csv.reader(f):
            if not raw_row:
                continue
            row = [cell.strip() for cell in raw_row]
            if row[0] == "Method":
                current_method = row[1]
                method_blocks[current_method] = []
            elif row[0].startswith("Bar") and current_method is not None:
                method_blocks[current_method].append(float(row[1]))

    rates_by_method = {}
    for method, values in method_blocks.items():
        if len(values) < len(TASK_ORDER_WITH_AVG):
            raise ValueError(
                f"{path} method {method!r} has {len(values)} bars; "
                f"expected at least {len(TASK_ORDER_WITH_AVG)}"
            )
        rates_by_method[method] = {
            task: values[i] for i, task in enumerate(TASK_ORDER_WITH_AVG)
        }
    return rates_by_method


def collect_rates(args):
    dynaguide_rates = read_dynaguide_rates(args.dynaguide_csv)
    flower_rates = read_summary_rates(args.flower_csv)
    hint2_rates = read_summary_rates(args.ours)

    all_rates = {}
    for output_method, dynaguide_method in DYNA_METHOD_MAP.items():
        if dynaguide_method not in dynaguide_rates:
            raise ValueError(f"Missing DynaGuide method block: {dynaguide_method}")
        all_rates[output_method] = dynaguide_rates[dynaguide_method]

    all_rates["flower"] = flower_rates
    all_rates["hint2"] = hint2_rates
    return all_rates


def write_combined_csv(rates, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "task_label",
                "method",
                "method_label",
                "success_rate",
            ],
        )
        writer.writeheader()
        for task in TASK_ORDER_WITH_AVG:
            for method in METHOD_ORDER:
                writer.writerow(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "method": METHOD_CSV_LABELS[method],
                        "method_label": METHOD_LABELS[method].replace("$", ""),
                        "success_rate": f"{rates[method][task]:.6f}",
                    }
                )


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(PANEL_FRAME_LW)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", color=AXIS_GRAY, width=0.45, length=2.0, pad=1.0)
    ax.yaxis.labelpad = 2.0


def plot_rates(rates, output_stem):
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.38 * TEXTWIDTH_IN))
    bar_width = 0.12
    intra_group_gap = 0.30
    group_width = len(METHOD_ORDER) * bar_width + intra_group_gap

    group_centers = []
    group_labels = []
    x = 0.0
    for task in TASK_ORDER:
        group_start = x
        for method_index, method in enumerate(METHOD_ORDER):
            pos = group_start + method_index * bar_width
            value = rates[method][task]
            ax.bar(
                pos,
                value,
                width=bar_width * 0.86,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.35,
            )

        center = group_start + (len(METHOD_ORDER) - 1) * bar_width / 2.0
        group_centers.append(center)
        group_labels.append(TASK_LABELS[task])
        x = group_start + group_width

    ax.set_xlim(-0.12, x - intra_group_gap + 0.12)
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("success rate")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=5.5)
    ax.tick_params(axis="x", bottom=False, labelbottom=True, pad=2.0)
    style_axis(ax)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(METHOD_ORDER),
        frameon=False,
        handlelength=1.25,
        handleheight=0.75,
        columnspacing=1.35,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.28)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot CALVIN articulated-object success bars."
    )
    parser.add_argument(
        "--flower-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs/calvin_paper/baselines/flower/articulated/"
        / "flower_articulated_h500_rollouts20/summary_table.csv",
    )
    parser.add_argument(
        "--dynaguide-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs/calvin_paper/baselines/dynaguide_numbers/dynaguide_bars.csv",
    )
    parser.add_argument(
        "--ours",
        type=Path,
        default=REPO_ROOT
        / "outputs/calvin_paper/articulated_objects/articulated_candidates64_h500_full",
        help="Directory containing summary_table.csv, or the CSV file itself.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/calvin_paper/plots/articulated_results",
    )
    parser.add_argument(
        "--stem",
        default="calvin_articulated_success_bars",
        help="Output filename stem for PNG/PDF and combined CSV.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    rates = collect_rates(args)
    output_stem = args.output_dir / args.stem
    write_combined_csv(rates, output_stem.with_name(output_stem.name + "_data.csv"))
    plot_rates(rates, output_stem)
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_name(output_stem.name + '_data.csv')}")


if __name__ == "__main__":
    main()
