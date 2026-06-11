from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BLENDER = REPO_ROOT / "tools/blender-4.2.0-linux-x64/blender"

QUALITY_PRESETS = {
    0: {"samples": 1, "square_size": 900},
    1: {"samples": 32, "square_size": 1500},
    2: {"samples": 160, "square_size": 2400},
}

PANELS = [
    {
        "name": "conditional",
        "blend_path": "outputs/paper_plots/editable_blender_panels/conditional.blend",
        "output_dir": "outputs/paper_plots/complex_conditional",
    },
    {
        "name": "safety",
        "blend_path": "outputs/paper_plots/editable_blender_panels/safety_constraint.blend",
        "output_dir": "outputs/paper_plots/complex_region_safety",
    },
    {
        "name": "cyclic",
        "blend_path": "outputs/paper_plots/editable_blender_panels/cyclic.blend",
        "output_dir": "outputs/paper_plots/cyclic_repetition",
    },
    {
        "name": "behavior_prior",
        "blend_path": "outputs/paper_plots/editable_blender_panels/behavior_prior.blend",
        "output_dir": "outputs/paper_plots/base_policy_prior",
    },
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and compose a plain 2x2 CALVIN Blender panel contact sheet.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/paper_plots/calvin_panel_contact_sheet")
    parser.add_argument("--output-stem", default="calvin_panel_contact_sheet_plain")
    parser.add_argument("--blender", type=Path, default=LOCAL_BLENDER if LOCAL_BLENDER.exists() else Path("blender"))
    parser.add_argument("--quality", type=int, choices=sorted(QUALITY_PRESETS), default=1)
    parser.add_argument("--render-square-size", type=int, default=None)
    parser.add_argument("--rerender", dest="rerender", action="store_true", default=True)
    parser.add_argument("--no-rerender", dest="rerender", action="store_false")
    parser.add_argument("--pdf", dest="pdf", action="store_true", default=True)
    parser.add_argument("--no-pdf", dest="pdf", action="store_false")
    parser.add_argument("--gap", type=float, default=0.018)
    return parser.parse_args()


def next_versioned_output(output_dir: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir.name
    for version in range(1, 10000):
        candidate = output_dir / f"{prefix}_v{version}.{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No free {prefix}_vN.{suffix} slot found in {output_dir}")


def latest_versioned_png(output_dir: Path) -> Path:
    output_dir = repo_path(output_dir)
    prefix = output_dir.name
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.png$")
    candidates = []
    for path in output_dir.glob(f"{prefix}_v*.png"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No versioned PNGs found in {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def render_saved_blend(panel: dict, args: argparse.Namespace) -> Path:
    quality = QUALITY_PRESETS[args.quality]
    square_size = int(args.render_square_size or quality["square_size"])
    output = next_versioned_output(repo_path(panel["output_dir"]), "png")
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
    cmd = [str(args.blender), "-b", str(repo_path(panel["blend_path"])), "--python-expr", expression]
    print(f"Rendering {panel['name']}: {output}")
    subprocess.run(cmd, check=True)
    return output


def panel_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for panel in PANELS:
        if args.rerender:
            paths.append(render_saved_blend(panel, args))
        else:
            paths.append(latest_versioned_png(repo_path(panel["output_dir"])))
    return paths


def compose(paths: list[Path], args: argparse.Namespace) -> tuple[Path, Path | None]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / f"{args.output_stem}.png"
    output_pdf = args.output_dir / f"{args.output_stem}.pdf" if args.pdf else None

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0), facecolor="white")
    for ax, path in zip(axes.flat, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=args.gap, hspace=args.gap)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.0)
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output_png, output_pdf


def main() -> None:
    args = parse_args()
    paths = panel_paths(args)
    output_png, output_pdf = compose(paths, args)
    print(output_png)
    if output_pdf is not None:
        print(output_pdf)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
