from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from compose_calvin_figure_grid import (
    FIG_DPI,
    FLOWER_GREEN,
    OUR_BLUE,
    PAPER_FONT,
    ROW_SEPARATOR_COLOR,
    SUBTITLE_FONTSIZE,
    TITLE_FONTSIZE,
    add_axes_in_inches,
    repo_path,
    wrap_text_to_width,
)


BACKGROUND_COLOR = "#ededed"
SOURCE_IMAGE = repo_path("outputs/paper_plots/calvin_individual_panels_export/png/safety_constraint.png")
OUTPUT_DIR = repo_path("outputs/paper_plots/safety_metrics_three_panel")
OUTPUT_STEM = "safety_metrics_three_panel"
OUTPUT_DPI = 600

# Main tweak: total output width in inches.
FIGURE_WIDTH_IN = 3.50
SIDE_MARGIN_IN = 0.035
COLUMN_GAP_IN = 0.12
TITLE_HEIGHT_IN = 0.20
SUBTITLE_HEIGHT_IN = 0.33

LEGEND_HEIGHT_IN = 0.22
BOTTOM_MARGIN_IN = 0.02
# Negative values move the legend upward without moving the panels.
LEGEND_NEGATIVE_INSET_IN = 0.0
TOP_MARGIN_IN = 0.02
FRAME_LINEWIDTH = 0.91
METRIC_LINEWIDTH = 0.70
GRID_LINEWIDTH = 0.35
AXIS_LABEL_FONTSIZE = 4.8
TICK_FONTSIZE = 4.4
LEGEND_FONTSIZE = 5.5

SAFETY_TITLE = "Safety Region"
SAFETY_SUBTITLE = '"Turn off the switch while avoiding the unsafe region."'


def background_rgb() -> np.ndarray:
    value = BACKGROUND_COLOR.removeprefix("#")
    return np.array([int(value[i : i + 2], 16) / 255.0 for i in range(0, 6, 2)])


def image_with_background(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    image = np.array(image, copy=True)
    rgb = image[..., :3]
    pure_white = np.all(rgb >= 0.995, axis=-1)
    rgb[pure_white] = background_rgb()
    if image.shape[-1] == 4:
        transparent = image[..., 3] <= 0.001
        rgb[transparent] = background_rgb()
        image[..., 3][transparent] = 1.0
    return image


def style_square_frame(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
    for spine in ax.spines.values():
        spine.set_color(ROW_SEPARATOR_COLOR)
        spine.set_linewidth(FRAME_LINEWIDTH)
    ax.tick_params(length=1.6, width=0.35, labelsize=TICK_FONTSIZE, pad=1.2)


def add_title(fig, text: str, x: float, y: float, width: float, height: float, figsize: tuple[float, float]) -> None:
    ax = add_axes_in_inches(fig, x, y, width, height, figsize)
    ax.set_axis_off()
    ax.text(0.5, 0.48, text, ha="center", va="center", fontsize=TITLE_FONTSIZE)


def add_subtitle(fig, text: str, x: float, y: float, width: float, height: float, figsize: tuple[float, float]) -> None:
    ax = add_axes_in_inches(fig, x, y, width, height, figsize)
    ax.set_axis_off()
    wrapped = wrap_text_to_width(text, width, SUBTITLE_FONTSIZE, renderer=fig.canvas.get_renderer(), dpi=fig.dpi)
    ax.text(0.5, 0.52, wrapped, ha="center", va="center", fontsize=SUBTITLE_FONTSIZE, linespacing=1.08)


def add_safety_image(fig, x: float, y: float, size: float, figsize: tuple[float, float]) -> None:
    ax = add_axes_in_inches(fig, x, y, size, size, figsize)
    ax.set_axis_off()
    ax.imshow(image_with_background(SOURCE_IMAGE))
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=ROW_SEPARATOR_COLOR,
            linewidth=FRAME_LINEWIDTH,
            clip_on=False,
            zorder=10,
        )
    )


def synthetic_angle_traces() -> tuple[list[np.ndarray], list[np.ndarray]]:
    t = np.arange(61)
    green = [
        23.0 - 0.33 * t + 1.4 * np.sin(t / 6.0),
        22.0 - 0.22 * t + 1.0 * np.cos(t / 7.5),
        20.0 + 1.2 * np.sin(t / 8.0) - 0.02 * t,
    ]
    blue = [
        20.0 + 1.0 * np.sin(t / 5.6) + 0.4 * np.sin(t / 1.9),
        18.7 + 0.8 * np.cos(t / 6.0) + 0.3 * np.sin(t / 2.4),
        21.0 + 0.7 * np.sin(t / 7.0 + 0.7) + 0.2 * np.cos(t / 1.8),
    ]
    return green, blue


def synthetic_gripper_traces() -> tuple[list[np.ndarray], list[np.ndarray]]:
    t = np.arange(61)
    green = [
        np.where(t < 25, 8.0, np.maximum(0.0, 8.0 - 0.95 * (t - 24))),
        np.where(t < 31, 8.0, np.maximum(0.0, 8.0 - 1.18 * (t - 30))),
    ]
    blue = [
        8.0 + 0.05 * np.sin(t / 6.0),
        7.95 + 0.05 * np.cos(t / 7.0),
    ]
    return green, blue


def add_metric_panel(
    fig,
    x: float,
    y: float,
    size: float,
    figsize: tuple[float, float],
    *,
    kind: str,
) -> None:
    ax = add_axes_in_inches(fig, x, y, size, size, figsize)
    style_square_frame(ax)
    ax.grid(True, color="#bcbcbc", linewidth=GRID_LINEWIDTH, alpha=0.62)
    ax.set_xlim(0, 60)
    ax.set_xlabel("timesteps", fontsize=AXIS_LABEL_FONTSIZE, labelpad=1.0)

    if kind == "angle":
        green, blue = synthetic_angle_traces()
        ax.axhspan(16, 24, color=OUR_BLUE, alpha=0.12, linewidth=0.0, zorder=1)
        ax.axhline(20, color="#202020", linewidth=0.42, linestyle=(0, (2.0, 2.0)), zorder=2)
        ax.set_ylim(0, 26)
        ax.set_yticks([0, 8, 16, 24])
        ylabel = r"angle[$^\circ$]"
    elif kind == "gripper":
        green, blue = synthetic_gripper_traces()
        ax.set_ylim(0, 8.4)
        ax.set_yticks([0, 4, 8])
        ylabel = "gripper [cm]"
    else:
        raise ValueError(f"Unknown metric panel kind: {kind}")

    ax.text(
        0.045,
        0.5,
        ylabel,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    for values in green:
        ax.plot(np.arange(len(values)), values, color=FLOWER_GREEN, linewidth=METRIC_LINEWIDTH, alpha=0.68, zorder=3)
    for values in blue:
        ax.plot(np.arange(len(values)), values, color=OUR_BLUE, linewidth=METRIC_LINEWIDTH, alpha=0.82, zorder=4)


def add_legend(fig, y: float, figsize: tuple[float, float]) -> None:
    ax = add_axes_in_inches(fig, 0.0, y, figsize[0], LEGEND_HEIGHT_IN, figsize)
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=FLOWER_GREEN, linewidth=1.8, label="Unguided Policy"),
        Line2D([0], [0], color=OUR_BLUE, linewidth=1.8, label="Our Method"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#d4d4d4",
        facecolor=BACKGROUND_COLOR,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        columnspacing=1.4,
        borderpad=0.35,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.45)


def main() -> None:
    plt.rcParams.update(PAPER_FONT)
    square_size = (FIGURE_WIDTH_IN - 2.0 * SIDE_MARGIN_IN - 2.0 * COLUMN_GAP_IN) / 3.0
    fig_height = (
        TOP_MARGIN_IN
        + TITLE_HEIGHT_IN
        + square_size
        + SUBTITLE_HEIGHT_IN
        + LEGEND_HEIGHT_IN
        + BOTTOM_MARGIN_IN
    )
    figsize = (FIGURE_WIDTH_IN, fig_height)
    fig = plt.figure(figsize=figsize, dpi=FIG_DPI, facecolor=BACKGROUND_COLOR)

    xs = [
        SIDE_MARGIN_IN,
        SIDE_MARGIN_IN + square_size + COLUMN_GAP_IN,
        SIDE_MARGIN_IN + 2.0 * (square_size + COLUMN_GAP_IN),
    ]
    legend_y = BOTTOM_MARGIN_IN
    legend_draw_y = legend_y - LEGEND_NEGATIVE_INSET_IN
    subtitle_y = legend_y + LEGEND_HEIGHT_IN
    square_y = subtitle_y + SUBTITLE_HEIGHT_IN
    title_y = square_y + square_size

    titles = [SAFETY_TITLE, "Safety Angle", "Safety Gripper"]
    for x, title in zip(xs, titles):
        add_title(fig, title, x, title_y, square_size, TITLE_HEIGHT_IN, figsize)

    add_safety_image(fig, xs[0], square_y, square_size, figsize)
    add_metric_panel(fig, xs[1], square_y, square_size, figsize, kind="angle")
    add_metric_panel(fig, xs[2], square_y, square_size, figsize, kind="gripper")
    add_subtitle(fig, SAFETY_SUBTITLE, xs[0], subtitle_y, square_size, SUBTITLE_HEIGHT_IN, figsize)
    add_legend(fig, legend_draw_y, figsize)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(output_png, dpi=OUTPUT_DPI, facecolor=BACKGROUND_COLOR, pad_inches=0.0)
    plt.close(fig)
    print(output_png)


if __name__ == "__main__":
    main()
