from __future__ import annotations

import argparse
import copy
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath


REPO_ROOT = Path(__file__).resolve().parents[1]
LATEX_TEXTWIDTH_PT = 397.48499
PT_PER_IN = 72.27
TEXTWIDTH_IN = LATEX_TEXTWIDTH_PT / PT_PER_IN
FIG_DPI = 300
DEFAULT_FIGURE_WIDTH = TEXTWIDTH_IN
OUTPUT_STEM = "calvin_figure_grid"
RENDER_SCRIPT = REPO_ROOT / "figures/render_calvin_trajectory_figure.py"
LOCAL_BLENDER = REPO_ROOT / "tools/blender-4.2.0-linux-x64/blender"

# Layout knobs. These can also be overridden from the command line.
PAGE_SIDE_MARGIN_IN = 0.02
PANEL_COLUMN_WSPACE = 0.06
PANEL_SQUARE_SCALE = 1.
TITLE_ROW_HEIGHT_IN = 0.16
SUBTITLE_ROW_HEIGHT_IN = 0.32
ROW_SEPARATOR_COLOR = "black"
ADD_GROUP_SEPARATOR = True
GROUP_SEPARATOR_AFTER_COL = 0
GROUP_SEPARATOR_GAP_MULTIPLIER = 2.0
GROUP_SEPARATOR_TOP_INSET = 0.04
GROUP_SEPARATOR_BOTTOM_INSET = 0.1
ADD_PANEL_FRAMES = True
LINE_THICKNESS = 0.91
LEGEND_GAP_ABOVE_SCALE = 0.16
LEGEND_BOTTOM_MARGIN_IN = 0.02
LEGEND_LINEWIDTH = 1.7
LEGEND_FRAME_LINEWIDTH = 0.45
TITLE_FONTSIZE = 7.0
SUBTITLE_FONTSIZE = 5.5
TITLE_BOLD = False 
TITLE_WEIGHT = "bold" if TITLE_BOLD else "normal"
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
            "path": "outputs/paper_plots/calvin_individual_panels_export/png/behavior_prior.png",
            "render": {
                "figure_preset": "base-diverse",
                "output_dir": "outputs/paper_plots/calvin_individual_panels_export/png",
                "blend_path": "outputs/paper_plots/calvin_individual_panels_export/blend/behavior_prior.blend",
                "extra_args": [
                    "--trajectory-radius",
                    "0.0022",
                    "--trajectory-emission",
                    "0.28",
                ],
            },
            "title": "Behavior Prior",
            "subtitle": "Unconditioned, Multimodal\nDiffusion Policy",
            "quote_subtitle": False,
        },
        {
            "path": "outputs/paper_plots/calvin_individual_panels_export/png/conditional.png",
            "render": {
                "figure_preset": "complex-conditional",
                "output_dir": "outputs/paper_plots/calvin_individual_panels_export/png",
                "blend_path": "outputs/paper_plots/calvin_individual_panels_export/blend/conditional.blend",
            },
            "title": "Conditional Execution",
            "subtitle": "Close the drawer, but only once the button and the switch are off.",
        },
        {
            "path": "outputs/paper_plots/calvin_individual_panels_export/png/safety_constraint.png",
            "render": {
                "figure_preset": "complex-region",
                "output_dir": "outputs/paper_plots/calvin_individual_panels_export/png",
                "blend_path": "outputs/paper_plots/calvin_individual_panels_export/blend/safety_constraint.blend",
            },
            "title": "Safety Constraints",
            "subtitle": "Turn off the switch while keeping the robot arm out of the unsafe region.",
        },
        {
            "path": "outputs/paper_plots/calvin_individual_panels_export/png/cyclic_repetition.png",
            "render": {
                "output_dir": "outputs/paper_plots/calvin_individual_panels_export/png",
                "blend_path": "outputs/paper_plots/calvin_individual_panels_export/blend/cyclic.blend",
            },
            "title": "Cyclic Repetition",
            "subtitle": "Repeatedly press the button, flip the switch, \nand move the drawer.",
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


def add_boolean_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    dest: str | None = None,
    help: str | None = None,
) -> None:
    dest = dest or name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose CALVIN render PNGs into a paper figure grid.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/paper_plots/calvin_figure_grid")
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--versioned", action="store_true", help="Write calvin_figure_grid_vN instead of overwriting the fixed output name.")
    parser.add_argument("--dpi", type=int, default=FIG_DPI)
    parser.add_argument(
        "--figsize",
        nargs="+",
        type=float,
        default=[DEFAULT_FIGURE_WIDTH],
        metavar="INCH",
        help=(
            "Figure size in inches. Pass WIDTH for automatic height from the square panel geometry, "
            "or WIDTH HEIGHT for an explicit fixed size."
        ),
    )
    parser.add_argument("--column-wspace", type=float, default=PANEL_COLUMN_WSPACE)
    parser.add_argument("--panel-square-scale", type=float, default=PANEL_SQUARE_SCALE)
    add_boolean_arg(parser, "add-bottom-column-separator", default=ADD_GROUP_SEPARATOR, dest="add_group_separator", help=argparse.SUPPRESS)
    parser.add_argument(
        "--bottom-column-separator-top-inset",
        type=float,
        default=GROUP_SEPARATOR_TOP_INSET,
        dest="group_separator_top_inset",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bottom-column-separator-bottom-inset",
        type=float,
        default=GROUP_SEPARATOR_BOTTOM_INSET,
        dest="group_separator_bottom_inset",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--line-thickness", type=float, default=LINE_THICKNESS, help="Shared linewidth for panel frames and separators.")
    add_boolean_arg(parser, "add-panel-frames", default=ADD_PANEL_FRAMES)
    parser.add_argument("--legend-height", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--title-fontsize", type=float, default=TITLE_FONTSIZE)
    parser.add_argument("--subtitle-fontsize", type=float, default=SUBTITLE_FONTSIZE)
    parser.add_argument("--no-titles", action="store_true")
    add_boolean_arg(parser, "rerender-panels", default=False)
    add_boolean_arg(
        parser,
        "rebuild-blends",
        default=False,
        help="Recreate editable Blender source files from CALVIN before rendering. This overwrites those editable .blend files.",
    )
    parser.add_argument(
        "--blender",
        default=str(LOCAL_BLENDER if LOCAL_BLENDER.exists() else "blender"),
        help='Blender executable path or command prefix, e.g. "flatpak run org.blender.Blender".',
    )
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


def image_scale(args: argparse.Namespace) -> float:
    return min(max(args.panel_square_scale, 0.05), 1.0)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=OUR_BLUE, linewidth=LEGEND_LINEWIDTH, label=r"hint$^2$"),
        Line2D(
            [0],
            [0],
            color=FLOWER_GREEN,
            linewidth=LEGEND_LINEWIDTH,
            label="Vision-Language-Action Policy (FLOWER)",
        ),
    ]


def add_legend_to_axis(ax):
    legend = ax.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
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
    return legend


def measure_legend_height(width: float, dpi: int) -> float:
    measure_fig = plt.figure(figsize=(width, 0.5), dpi=dpi, facecolor="white")
    ax = measure_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    legend = add_legend_to_axis(ax)
    measure_fig.canvas.draw()
    height = legend.get_window_extent(measure_fig.canvas.get_renderer()).height / dpi
    plt.close(measure_fig)
    return height


def panel_layout(args: argparse.Namespace, width: float) -> dict[str, float]:
    ncols = len(PANEL_ROWS[0])
    side_margin = min(max(PAGE_SIDE_MARGIN_IN, 0.0), max(width * 0.45, 0.0))
    content_width = width - 2.0 * side_margin
    column_wspace = max(float(args.column_wspace), 0.0)
    gap_multipliers = [1.0] * max(ncols - 1, 0)
    if 0 <= GROUP_SEPARATOR_AFTER_COL < len(gap_multipliers):
        gap_multipliers[GROUP_SEPARATOR_AFTER_COL] = max(GROUP_SEPARATOR_GAP_MULTIPLIER, 0.0)
    panel_width = content_width / (ncols + sum(gap_multipliers) * column_wspace)
    column_gap = panel_width * column_wspace
    column_gaps = [column_gap * multiplier for multiplier in gap_multipliers]
    square_size = panel_width * image_scale(args)
    title_height = max(TITLE_ROW_HEIGHT_IN, 1e-6)
    subtitle_height = max(SUBTITLE_ROW_HEIGHT_IN, 1e-6)
    panel_height = title_height + square_size + subtitle_height
    legend_content_height = measure_legend_height(width, args.dpi)
    legend_gap_scale = LEGEND_GAP_ABOVE_SCALE if args.legend_height is None else args.legend_height
    legend_gap_above = max(legend_gap_scale, 0.0) * legend_content_height
    legend_bottom_margin = max(LEGEND_BOTTOM_MARGIN_IN, 0.0)
    legend_height = legend_bottom_margin + legend_content_height + legend_gap_above
    figure_height = panel_height + legend_height
    return {
        "panel_width": panel_width,
        "column_gap": column_gap,
        "column_gaps": column_gaps,
        "side_margin": side_margin,
        "content_width": content_width,
        "square_size": square_size,
        "title_height": title_height,
        "subtitle_height": subtitle_height,
        "panel_height": panel_height,
        "legend_content_height": legend_content_height,
        "legend_gap_above": legend_gap_above,
        "legend_bottom_margin": legend_bottom_margin,
        "legend_height": legend_height,
        "figure_height": figure_height,
    }


def resolve_figsize(args: argparse.Namespace, layout: dict[str, float] | None = None) -> tuple[float, float]:
    if len(args.figsize) not in {1, 2}:
        raise ValueError("--figsize expects WIDTH or WIDTH HEIGHT.")
    width = float(args.figsize[0])
    if width <= 0.0:
        raise ValueError("--figsize WIDTH must be positive.")
    if len(args.figsize) == 2:
        height = float(args.figsize[1])
        if height <= 0.0:
            raise ValueError("--figsize HEIGHT must be positive.")
        return width, height

    layout = layout or panel_layout(args, width)
    return width, layout["figure_height"]


def resolve_blender_command(blender: str) -> list[str]:
    parts = shlex.split(str(blender))
    if not parts:
        raise ValueError("Empty Blender executable command.")
    blender_path = Path(parts[0]).expanduser()
    if blender_path.is_absolute() or blender_path.parent != Path("."):
        if blender_path.exists():
            return [str(blender_path.resolve()), *parts[1:]]
        raise FileNotFoundError(f"Blender executable not found: {blender_path}")
    resolved = shutil.which(parts[0])
    if resolved is not None:
        return [resolved, *parts[1:]]
    raise FileNotFoundError(
        f"Blender executable '{parts[0]}' was not found on PATH. "
        'Flatpak installs can be passed as --blender "flatpak run org.blender.Blender".'
    )


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
        "--blender",
        str(args.blender),
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
    blender_cmd = resolve_blender_command(args.blender)
    cmd = [
        *blender_cmd,
        "-b",
        str(blend_path),
        "--python-expr",
        expression,
    ]
    print("Rendering saved Blender panel:")
    print(" ".join(blender_cmd + ["-b", str(blend_path), "--python-expr", "<render expression>"]))
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
        if "figure_preset" not in spec:
            raise RuntimeError(f"Cannot rebuild saved-only Blender panel: {blend_path}")
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


def text_width_in(text: str, fontsize: float, weight: str = "normal", renderer=None, dpi: float = FIG_DPI) -> float:
    if not text:
        return 0.0
    prop = FontProperties(family=PAPER_FONT["font.family"], size=fontsize, weight=weight)
    if renderer is not None:
        width_px, _, _ = renderer.get_text_width_height_descent(text, prop, ismath=False)
        return float(width_px) / float(dpi)
    return float(TextPath((0.0, 0.0), text, prop=prop, size=fontsize).get_extents().width) / PT_PER_IN


def wrap_text_to_width(text: str, max_width_in: float, fontsize: float, weight: str = "normal", renderer=None, dpi: float = FIG_DPI) -> str:
    if max_width_in <= 0.0:
        return text
    lines = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if text_width_in(candidate, fontsize, weight, renderer=renderer, dpi=dpi) <= max_width_in:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return "\n".join(lines)


def add_text(
    ax,
    text: str,
    fontsize: float,
    weight: str = "normal",
    max_width_in: float | None = None,
) -> None:
    ax.set_axis_off()
    if max_width_in is not None:
        renderer = ax.figure.canvas.get_renderer()
        text = wrap_text_to_width(text, max_width_in, fontsize, weight, renderer=renderer, dpi=ax.figure.dpi)
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


def add_axes_in_inches(fig, x: float, y: float, width: float, height: float, figsize: tuple[float, float]):
    fig_width, fig_height = figsize
    return fig.add_axes([x / fig_width, y / fig_height, width / fig_width, height / fig_height])


def formatted_subtitle(panel: dict) -> str:
    subtitle = panel["subtitle"]
    if panel.get("quote_subtitle", True):
        return f'"{subtitle}"'
    return subtitle


def add_panel(
    fig,
    panel: dict,
    args: argparse.Namespace,
    show_title: bool,
    layout: dict[str, float],
    figsize: tuple[float, float],
    x: float,
    y: float,
    subtitle_x: float,
    subtitle_width: float,
) -> None:
    panel_width = layout["panel_width"]
    square_size = layout["square_size"]
    subtitle_height = layout["subtitle_height"]
    title_height = layout["title_height"]

    subtitle_y = y
    image_y = subtitle_y + subtitle_height
    title_y = image_y + square_size
    image_x = x + 0.5 * (panel_width - square_size)

    if show_title:
        add_text(
            add_axes_in_inches(fig, x, title_y, panel_width, title_height, figsize),
            panel["title"],
            fontsize=args.title_fontsize,
            weight=TITLE_WEIGHT,
        )
    else:
        add_axes_in_inches(fig, x, title_y, panel_width, title_height, figsize).set_axis_off()
    add_image_or_placeholder(
        add_axes_in_inches(fig, image_x, image_y, square_size, square_size, figsize),
        panel,
        args,
    )
    add_text(
        add_axes_in_inches(fig, subtitle_x, subtitle_y, subtitle_width, subtitle_height, figsize),
        formatted_subtitle(panel),
        fontsize=args.subtitle_fontsize,
        max_width_in=subtitle_width,
    )


def add_group_separator(
    fig,
    x: float,
    y: float,
    height: float,
    figsize: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    if not args.add_group_separator:
        return
    fig_width, fig_height = figsize
    bottom_inset = max(args.group_separator_bottom_inset, 0.0)
    top_inset = max(args.group_separator_top_inset, 0.0)
    y0 = y + bottom_inset * height
    y1 = y + height - top_inset * height
    if y1 <= y0:
        return
    fig.add_artist(
        Line2D(
            [x / fig_width, x / fig_width],
            [y0 / fig_height, y1 / fig_height],
            transform=fig.transFigure,
            color=ROW_SEPARATOR_COLOR,
            linewidth=args.line_thickness,
            solid_capstyle="butt",
            clip_on=False,
        )
    )


def add_legend(fig, x: float, y: float, width: float, height: float, figsize: tuple[float, float]) -> None:
    ax = add_axes_in_inches(fig, x, y, width, height, figsize)
    ax.set_axis_off()
    add_legend_to_axis(ax)

def compose(output_png: Path, args: argparse.Namespace) -> None:
    plt.rcParams.update(PAPER_FONT)
    if len(args.figsize) not in {1, 2}:
        raise ValueError("--figsize expects WIDTH or WIDTH HEIGHT.")
    width = float(args.figsize[0])
    if width <= 0.0:
        raise ValueError("--figsize WIDTH must be positive.")
    layout = panel_layout(args, width)
    figsize = resolve_figsize(args, layout)
    panel_rows = prepare_panels(args)
    panels = panel_rows[0]
    fig = plt.figure(figsize=figsize, dpi=args.dpi, facecolor="white")
    panel_x = [layout["side_margin"]]
    for gap in layout["column_gaps"]:
        panel_x.append(panel_x[-1] + layout["panel_width"] + gap)
    legend_y = 0.0
    row_y = legend_y + layout["legend_height"]
    show_title = not args.no_titles

    for col_idx, panel in enumerate(panels):
        add_panel(
            fig,
            panel,
            args,
            show_title,
            layout,
            figsize,
            panel_x[col_idx],
            row_y,
            panel_x[col_idx],
            layout["panel_width"],
        )
    if 0 <= GROUP_SEPARATOR_AFTER_COL < len(panels) - 1:
        separator_gap = layout["column_gaps"][GROUP_SEPARATOR_AFTER_COL]
        add_group_separator(
            fig,
            panel_x[GROUP_SEPARATOR_AFTER_COL] + layout["panel_width"] + 0.5 * separator_gap,
            row_y,
            layout["panel_height"],
            figsize,
            args,
        )
    add_legend(
        fig,
        0.0,
        legend_y + layout["legend_bottom_margin"],
        width,
        layout["legend_content_height"],
        figsize,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=args.dpi, pad_inches=0.0)
    if args.pdf:
        fig.savefig(output_png.with_suffix(".pdf"), pad_inches=0.0)
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
