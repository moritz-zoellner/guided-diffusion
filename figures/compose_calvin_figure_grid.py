from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
LATEX_TEXTWIDTH_PT = 397.48499
PT_PER_IN = 72.27
TEXTWIDTH_IN = LATEX_TEXTWIDTH_PT / PT_PER_IN
FIG_DPI = 300
DEFAULT_FIGSIZE = (TEXTWIDTH_IN, 4.70)
OUTPUT_STEM = "calvin_figure_grid"
RENDER_SCRIPT = REPO_ROOT / "figures/render_calvin_trajectory_figure.py"
LOCAL_BLENDER = REPO_ROOT / "tools/blender-4.2.0-linux-x64/blender"

# Layout knobs. These can also be overridden from the command line.
PANEL_COLUMN_WSPACE = 0.075
PANEL_SQUARE_SCALE = 0.50
ROW_SEPARATOR_GAP_ABOVE = 0.10
ROW_SEPARATOR_GAP_BELOW = 0.12
ROW_SEPARATOR_SIDE_INSET = 0.02
ROW_SEPARATOR_COLOR = "black"
ADD_BOTTOM_COLUMN_SEPARATOR = True
BOTTOM_COLUMN_SEPARATOR_TOP_INSET = 0.02
BOTTOM_COLUMN_SEPARATOR_BOTTOM_INSET = 0.02
ADD_PANEL_FRAMES = True
LINE_THICKNESS = 0.9
LEGEND_HEIGHT = 0.16
LEGEND_LINEWIDTH = 1.7
LEGEND_FRAME_LINEWIDTH = 0.45
TITLE_FONTSIZE = 7.0
SUBTITLE_FONTSIZE = 4.8
TITLE_WEIGHT = "bold"
OUR_BLUE = "#275fca"
FLOWER_GREEN = "#4e8b68"
BOOTSTRAP_CAMERA_ARGS = [
    "--camera-location",
    "0.009478",
    "0.40871",
    "1.4477",
    "--camera-rotation-deg",
    "26.327",
    "-0.000043",
    "-181.03",
    "--camera-fov",
    "46",
]
QUALITY_PRESETS = {
    0: {"samples": 1, "square_size": 900},
    1: {"samples": 32, "square_size": 1500},
    2: {"samples": 160, "square_size": 2400},
}

PANEL_ROWS = [
    [
        {
            "path": "outputs/paper_plots/complex_chained/complex_chained_v5.png",
            "render": {
                "figure_preset": "complex-chained",
                "output_dir": "outputs/paper_plots/complex_chained",
                "blend_path": "outputs/paper_plots/editable_blender_panels/long_horizon.blend",
            },
            "title": "Long-horizon Tasks",
            "subtitle": "Press the button, then move the sliding door right, then turn off the lightbulb with the switch, then close the drawer.",
        },
        {
            "path": "outputs/paper_plots/complex_conditional/complex_conditional_v5.png",
            "render": {
                "figure_preset": "complex-conditional",
                "output_dir": "outputs/paper_plots/complex_conditional",
                "blend_path": "outputs/paper_plots/editable_blender_panels/conditional.blend",
            },
            "title": "Conditional Execution",
            "subtitle": "Close the drawer, but only once the button and the switch are turned off.",
        },
        {
            "path": "outputs/paper_plots/complex_region_safety/complex_region_safety_v4.png",
            "render": {
                "figure_preset": "complex-region",
                "output_dir": "outputs/paper_plots/complex_region_safety",
                "blend_path": "outputs/paper_plots/editable_blender_panels/safety_constraint.blend",
            },
            "title": "Safety Constraints",
            "subtitle": "Turn off the lightbulb with the switch while keeping the robot arm out of the unsafe region.",
        },
    ],
    [
        {
            "path": "outputs/paper_plots/cyclic_repetition/cyclic_repetition_v1.png",
            "render": {
                "figure_preset": "cyclic",
                "output_dir": "outputs/paper_plots/cyclic_repetition",
                "blend_path": "outputs/paper_plots/editable_blender_panels/cyclic.blend",
            },
            "title": "Cyclic Repetition",
            "subtitle": "Repeatedly pick up the Cheez-Its, then pour in the left and the right bowl, and then put them down again.",
        },
        {
            "placeholder": True,
            "title": "Real-world Safety",
            "subtitle": "Pick up the Cheez-Its and pour them into the left bowl while avoiding the marked region.",
        },
        {
            "path": "outputs/paper_plots/base_policy_prior/base_policy_prior_v3.png",
            "render": {
                "figure_preset": "base-diverse",
                "output_dir": "outputs/paper_plots/base_policy_prior",
                "blend_path": "outputs/paper_plots/editable_blender_panels/behavior_prior.blend",
                "extra_args": [
                    "--trajectory-radius",
                    "0.0022",
                    "--trajectory-emission",
                    "0.28",
                ],
            },
            "title": "Behavior Prior",
            "subtitle": "Unconditioned diffusion policy",
            "quote_subtitle": False,
        },
    ],
]

PAPER_FONT = {
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


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def next_versioned_output(output_dir: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir.name
    for version in range(1, 10000):
        candidate = output_dir / f"{prefix}_v{version}.{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No free {prefix}_vN.{suffix} slot found in {output_dir}")


def fixed_output(output_dir: Path, stem: str, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}.{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose CALVIN render PNGs into a paper figure grid.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/paper_plots/calvin_figure_grid")
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--versioned", action="store_true", help="Write calvin_figure_grid_vN instead of overwriting the fixed output name.")
    parser.add_argument("--dpi", type=int, default=FIG_DPI)
    parser.add_argument("--figsize", nargs=2, type=float, default=DEFAULT_FIGSIZE, metavar=("W", "H"))
    parser.add_argument("--column-wspace", type=float, default=PANEL_COLUMN_WSPACE)
    parser.add_argument("--panel-square-scale", type=float, default=PANEL_SQUARE_SCALE)
    parser.add_argument("--row-separator-gap-above", type=float, default=ROW_SEPARATOR_GAP_ABOVE)
    parser.add_argument("--row-separator-gap-below", type=float, default=ROW_SEPARATOR_GAP_BELOW)
    parser.add_argument("--row-separator-side-inset", type=float, default=ROW_SEPARATOR_SIDE_INSET)
    parser.add_argument("--add-bottom-column-separator", action=argparse.BooleanOptionalAction, default=ADD_BOTTOM_COLUMN_SEPARATOR)
    parser.add_argument(
        "--bottom-column-separator-top-inset",
        type=float,
        default=BOTTOM_COLUMN_SEPARATOR_TOP_INSET,
        help="Inset for the bottom-row vertical separator as a fraction of the bottom row height.",
    )
    parser.add_argument(
        "--bottom-column-separator-bottom-inset",
        type=float,
        default=BOTTOM_COLUMN_SEPARATOR_BOTTOM_INSET,
        help="Inset for the bottom-row vertical separator as a fraction of the bottom row height.",
    )
    parser.add_argument("--line-thickness", type=float, default=LINE_THICKNESS, help="Shared linewidth for panel frames and separators.")
    parser.add_argument("--add-panel-frames", action=argparse.BooleanOptionalAction, default=ADD_PANEL_FRAMES)
    parser.add_argument("--legend-height", type=float, default=LEGEND_HEIGHT)
    parser.add_argument("--title-fontsize", type=float, default=TITLE_FONTSIZE)
    parser.add_argument("--subtitle-fontsize", type=float, default=SUBTITLE_FONTSIZE)
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--rerender-panels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rebuild-blends",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recreate editable Blender source files from CALVIN before rendering. This overwrites those editable .blend files.",
    )
    parser.add_argument("--blender", type=Path, default=LOCAL_BLENDER if LOCAL_BLENDER.exists() else Path("blender"))
    parser.add_argument("--quality", type=int, choices=sorted(QUALITY_PRESETS), default=0)
    parser.add_argument(
        "--render-square-size",
        type=int,
        default=None,
        help="Override the square Blender render resolution in pixels. Defaults to the selected quality preset.",
    )
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF next to the PNG.")
    return parser.parse_args()


def add_panel_frame(ax, args: argparse.Namespace) -> None:
    if not args.add_panel_frames:
        return
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=ROW_SEPARATOR_COLOR,
            linewidth=args.line_thickness,
            clip_on=False,
            zorder=10,
        )
    )


def latest_versioned_png(output_dir: Path) -> Path | None:
    output_dir = repo_path(output_dir)
    prefix = output_dir.name
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.png$")
    candidates = []
    for path in output_dir.glob(f"{prefix}_v*.png"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def render_panel_from_calvin(panel: dict, args: argparse.Namespace, output: Path) -> Path:
    spec = panel.get("render")
    quality = QUALITY_PRESETS[args.quality]
    square_size = int(args.render_square_size or quality["square_size"])
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--figure-preset",
        spec["figure_preset"],
        "--output-dir",
        str(repo_path(spec["output_dir"])),
        "--output",
        str(output),
        "--samples",
        str(quality["samples"]),
        "--resolution",
        str(square_size),
        str(square_size),
        *BOOTSTRAP_CAMERA_ARGS,
        *spec.get("extra_args", []),
    ]
    print("Building panel from CALVIN:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return output


def render_saved_blend(blend_path: Path, output: Path, args: argparse.Namespace) -> Path:
    quality = QUALITY_PRESETS[args.quality]
    square_size = int(args.render_square_size or quality["square_size"])
    expression = "\n".join(
        [
            "import bpy",
            f"output = {str(output)!r}",
            "scene = bpy.context.scene",
            "scene.render.engine = 'CYCLES'",
            f"scene.cycles.samples = {int(quality['samples'])}",
            "scene.cycles.use_denoising = True",
            f"scene.render.resolution_x = {square_size}",
            f"scene.render.resolution_y = {square_size}",
            "scene.render.filepath = output",
            "bpy.ops.render.render(write_still=True)",
        ]
    )
    cmd = [
        str(args.blender),
        "-b",
        str(blend_path),
        "--python-expr",
        expression,
    ]
    print("Rendering saved Blender panel:")
    print(" ".join(cmd[:3] + ["--python-expr", "<render expression>"]))
    subprocess.run(cmd, check=True)
    return output


def render_panel(panel: dict, args: argparse.Namespace) -> Path | None:
    spec = panel.get("render")
    if not spec:
        return None
    output_dir = repo_path(spec["output_dir"])
    output = next_versioned_output(output_dir, "png")
    blend_path = repo_path(spec["blend_path"]) if spec.get("blend_path") else None

    if blend_path is None:
        return render_panel_from_calvin(panel, args, output)

    if args.rebuild_blends or not blend_path.exists():
        rendered = render_panel_from_calvin(panel, args, output)
        built_blend = rendered.with_suffix(".blend")
        if not built_blend.exists():
            raise FileNotFoundError(f"Expected Blender file next to generated render: {built_blend}")
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_blend, blend_path)
        print(f"Editable Blender source: {blend_path}")
        return rendered

    return render_saved_blend(blend_path, output, args)


def prepare_panels(args: argparse.Namespace) -> list[list[dict]]:
    panel_rows = copy.deepcopy(PANEL_ROWS)
    for row in panel_rows:
        for panel in row:
            spec = panel.get("render")
            if not spec:
                continue
            if args.rerender_panels:
                rendered = render_panel(panel, args)
                if rendered is not None:
                    panel["path"] = relative_to_repo(rendered)
                    continue
            latest = latest_versioned_png(repo_path(spec["output_dir"]))
            if latest is not None:
                panel["path"] = relative_to_repo(latest)
    return panel_rows


def add_image_or_placeholder(ax, panel: dict, args: argparse.Namespace) -> None:
    ax.set_axis_off()
    ax.set_box_aspect(1)
    ax.set_anchor("C")
    if panel.get("placeholder"):
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                facecolor="#fbfbfb",
                edgecolor=ROW_SEPARATOR_COLOR if args.add_panel_frames else "#d6d6d6",
                linewidth=args.line_thickness if args.add_panel_frames else 0.6,
            )
        )
        return

    image_path = repo_path(panel["path"])
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    if not args.rerender_panels and width != height:
        print(f"Warning: stretching non-square panel image into square frame: {image_path} ({width}x{height})")
        ax.imshow(image, aspect="auto", extent=(0.0, 1.0, 0.0, 1.0))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.imshow(image)
    add_panel_frame(ax, args)


def add_text(ax, text: str, fontsize: float, weight: str = "normal", wrap: int | None = None) -> None:
    ax.set_axis_off()
    if wrap is not None:
        text = fill(text, width=wrap)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.08,
    )


def formatted_subtitle(panel: dict) -> str:
    subtitle = panel["subtitle"]
    if panel.get("quote_subtitle", True):
        return f'"{subtitle}"'
    return subtitle


def add_panel(fig, outer_slot, panel: dict, args: argparse.Namespace, show_title: bool) -> None:
    subgrid = outer_slot.subgridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.11, 1.0, 0.28],
        hspace=0.0,
    )
    if show_title:
        add_text(
            fig.add_subplot(subgrid[0, 0]),
            panel["title"],
            fontsize=args.title_fontsize,
            weight=TITLE_WEIGHT,
        )
    else:
        fig.add_subplot(subgrid[0, 0]).set_axis_off()
    image_scale = min(max(args.panel_square_scale, 0.05), 1.0)
    image_margin = (1.0 - image_scale) * 0.5
    image_grid = subgrid[1, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[image_margin, image_scale, image_margin],
        wspace=0.0,
    )
    add_image_or_placeholder(fig.add_subplot(image_grid[0, 1]), panel, args)
    add_text(
        fig.add_subplot(subgrid[2, 0]),
        formatted_subtitle(panel),
        fontsize=args.subtitle_fontsize,
        wrap=36,
    )


def add_row_separator(fig, outer_slot, args: argparse.Namespace) -> None:
    ax = fig.add_subplot(outer_slot)
    ax.set_axis_off()
    total_gap = args.row_separator_gap_above + args.row_separator_gap_below
    line_y = 0.5 if total_gap <= 0.0 else args.row_separator_gap_below / total_gap
    ax.plot(
        [args.row_separator_side_inset, 1.0 - args.row_separator_side_inset],
        [line_y, line_y],
        transform=ax.transAxes,
        color=ROW_SEPARATOR_COLOR,
        linewidth=args.line_thickness,
        solid_capstyle="butt",
        clip_on=False,
    )


def add_bottom_column_separator(fig, row_slot, left_slot, right_slot, args: argparse.Namespace) -> None:
    if not args.add_bottom_column_separator:
        return
    row_bbox = row_slot.get_position(fig)
    left_bbox = left_slot.get_position(fig)
    right_bbox = right_slot.get_position(fig)
    x = 0.5 * (left_bbox.x1 + right_bbox.x0)
    bottom_inset = max(args.bottom_column_separator_bottom_inset, 0.0)
    top_inset = max(args.bottom_column_separator_top_inset, 0.0)
    y0 = row_bbox.y0 + bottom_inset * row_bbox.height
    y1 = row_bbox.y1 - top_inset * row_bbox.height
    if y1 <= y0:
        return
    fig.add_artist(
        Line2D(
            [x, x],
            [y0, y1],
            transform=fig.transFigure,
            color=ROW_SEPARATOR_COLOR,
            linewidth=args.line_thickness,
            solid_capstyle="butt",
            clip_on=False,
        )
    )


def add_legend(fig, outer_slot) -> None:
    ax = fig.add_subplot(outer_slot)
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=OUR_BLUE, linewidth=LEGEND_LINEWIDTH, label=r"hint$^2$"),
        Line2D(
            [0],
            [0],
            color=FLOWER_GREEN,
            linewidth=LEGEND_LINEWIDTH,
            label="Vision-Language-Action Policy (FLOWER)",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
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
    legend.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


def compose(output_png: Path, args: argparse.Namespace) -> None:
    panel_rows = prepare_panels(args)
    plt.rcParams.update(PAPER_FONT)
    fig = plt.figure(figsize=args.figsize, dpi=args.dpi, facecolor="white")
    separator_height = max(args.row_separator_gap_above + args.row_separator_gap_below, 1e-6)
    grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[1.0, separator_height, 1.0, args.legend_height],
        hspace=0.0,
    )
    top_grid = grid[0, 0].subgridspec(nrows=1, ncols=3, wspace=args.column_wspace)
    bottom_grid = grid[2, 0].subgridspec(nrows=1, ncols=3, wspace=args.column_wspace)

    show_title = not args.no_titles
    for col_idx, panel in enumerate(panel_rows[0]):
        add_panel(fig, top_grid[0, col_idx], panel, args, show_title)
    add_row_separator(fig, grid[1, 0], args)
    for col_idx, panel in enumerate(panel_rows[1]):
        add_panel(fig, bottom_grid[0, col_idx], panel, args, show_title)
    add_bottom_column_separator(fig, grid[2, 0], bottom_grid[0, 1], bottom_grid[0, 2], args)
    add_legend(fig, grid[3, 0])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    if args.pdf:
        fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.versioned:
        output_png = next_versioned_output(output_dir, "png")
    else:
        output_png = fixed_output(output_dir, args.output_stem, "png")
    compose(output_png, args)
    print(output_png)
    if args.pdf:
        print(output_png.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
