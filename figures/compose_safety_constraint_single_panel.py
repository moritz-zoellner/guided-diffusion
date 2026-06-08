from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from compose_calvin_figure_grid import (
    FIG_DPI,
    PAPER_FONT,
    ROW_SEPARATOR_COLOR,
    SUBTITLE_FONTSIZE,
    add_axes_in_inches,
    repo_path,
    wrap_text_to_width,
)


BACKGROUND_COLOR = "#ededed"
PANEL_WIDTH_IN = 1.62
SUBTITLE_HEIGHT_IN = 0.40
PANEL_MARGIN_IN = 0.015
LINE_THICKNESS = 0.91
SOURCE_IMAGE = repo_path("outputs/paper_plots/calvin_individual_panels_export/png/safety_constraint.png")
OUTPUT_DIR = repo_path("outputs/paper_plots/safety_constraint_single_panel")
OUTPUT_STEM = "safety_constraint_single_panel"
SUBTITLE = '"Turn off the switch while keeping the robot arm out of the unsafe region."'


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


def main() -> None:
    plt.rcParams.update(PAPER_FONT)
    fig_width = PANEL_WIDTH_IN + 2.0 * PANEL_MARGIN_IN
    fig_height = PANEL_WIDTH_IN + SUBTITLE_HEIGHT_IN + 2.0 * PANEL_MARGIN_IN
    figsize = (fig_width, fig_height)
    fig = plt.figure(figsize=figsize, dpi=FIG_DPI, facecolor=BACKGROUND_COLOR)

    image_ax = add_axes_in_inches(
        fig,
        PANEL_MARGIN_IN,
        PANEL_MARGIN_IN + SUBTITLE_HEIGHT_IN,
        PANEL_WIDTH_IN,
        PANEL_WIDTH_IN,
        figsize,
    )
    image_ax.set_axis_off()
    image_ax.imshow(image_with_background(SOURCE_IMAGE))
    image_ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=image_ax.transAxes,
            facecolor="none",
            edgecolor=ROW_SEPARATOR_COLOR,
            linewidth=LINE_THICKNESS,
            clip_on=False,
            zorder=10,
        )
    )

    text_ax = add_axes_in_inches(
        fig,
        PANEL_MARGIN_IN,
        PANEL_MARGIN_IN,
        PANEL_WIDTH_IN,
        SUBTITLE_HEIGHT_IN,
        figsize,
    )
    text_ax.set_axis_off()
    subtitle = wrap_text_to_width(
        SUBTITLE,
        PANEL_WIDTH_IN,
        SUBTITLE_FONTSIZE,
        renderer=fig.canvas.get_renderer(),
        dpi=fig.dpi,
    )
    text_ax.text(
        0.5,
        0.52,
        subtitle,
        ha="center",
        va="center",
        fontsize=SUBTITLE_FONTSIZE,
        linespacing=1.08,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    output_pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(output_png, dpi=FIG_DPI, facecolor=BACKGROUND_COLOR, pad_inches=0.0)
    fig.savefig(output_pdf, facecolor=BACKGROUND_COLOR, pad_inches=0.0)
    plt.close(fig)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
