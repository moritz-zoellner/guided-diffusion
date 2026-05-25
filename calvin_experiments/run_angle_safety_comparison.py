#!/usr/bin/env python3
"""Run the angle safety task for all three methods and compare rates.

The wrapper keeps the three method runs under one comparison directory:

    <output-root>/<name>/
        hint2/
        flower/
        flower_gpc/
        logs/
        angle_comparison_rates.png
        angle_comparison_summary.csv

Each method still writes its usual rollout artifacts inside its own subfolder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/angle_comparisons"
ANGLE_RANDOM_RANGE_DEG = (20.0, 20.0)
ANGLE_TOLERANCE_DEG = 3.5

METHODS = (
    {
        "key": "hint2",
        "label": r"hint$^2$",
        "script": REPO_ROOT / "calvin_experiments/run_complex_stl_automaton.py",
        "color": "#275fca",
    },
    {
        "key": "flower",
        "label": "FLOWER",
        "script": REPO_ROOT / "calvin_experiments/paper_stls/baselines/run_flower_complex_stls.py",
        "color": "#c3c7cd",
    },
    {
        "key": "flower_gpc",
        "label": "FLOWER+GPC",
        "script": REPO_ROOT / "calvin_experiments/paper_stls/baselines/run_flower_gpc_complex_stls.py",
        "color": "#88b884",
    },
)

METRICS = (
    ("liveness_satisfaction_rate", "Liveness"),
    ("safety_satisfaction_rate", "Safety"),
    ("stl_satisfaction_rate", "STL"),
    ("subgoal_completion_rate", "Subgoal"),
)


def unique_dir(output_root: Path, name: str) -> Path:
    candidate = output_root / name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        suffixed = output_root / f"{name}_{suffix:02d}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def command_for_method(method: dict[str, Any], args: argparse.Namespace, comparison_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(method["script"]),
        "--name",
        method["key"],
        "--output-root",
        str(comparison_dir),
        "--tasks",
        "angle",
        "--n-rollouts",
        str(args.n_rollouts),
        "--seed-start",
        str(args.seed_start),
        "--device",
        args.device,
    ]
    if args.horizon is not None:
        cmd += ["--horizon", str(int(args.horizon))]
    if args.no_video:
        cmd.append("--no-video")
    if args.disable_safety_randomization:
        cmd.append("--disable-safety-randomization")
    if args.angle_goal_deg is not None:
        cmd += ["--angle-goal-deg", str(float(args.angle_goal_deg))]
    if args.angle_random_range_deg is not None:
        low, high = [float(value) for value in args.angle_random_range_deg]
        cmd += ["--angle-random-range-deg", str(low), str(high)]
    if args.angle_tolerance_deg is not None:
        cmd += ["--angle-tolerance-deg", str(float(args.angle_tolerance_deg))]
    if method["key"] == "flower_gpc":
        cmd += ["--n-candidates", str(args.gpc_n_candidates), "--chunk-horizon", str(args.gpc_chunk_horizon)]
    return cmd


def run_and_log(cmd: Sequence[str], *, log_path: Path, prefix: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{prefix}] {line}", end="")
            log_file.write(line)
        return_code = proc.wait()
        log_file.write(f"\nreturn_code={return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(cmd))


def load_angle_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    with summary_path.open() as f:
        payload = json.load(f)
    for item in payload.get("tasks", []):
        if item.get("task") == "angle":
            return item
    raise ValueError(f"No angle task summary found in {summary_path}")


def rate_to_percent(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return 100.0 * float(value)


def write_summary_files(comparison_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "label",
        "run_dir",
        "n_rollouts",
        "liveness_pct",
        "safety_pct",
        "stl_pct",
        "subgoal_pct",
        "avg_termination_step",
    ]
    with (comparison_dir / "angle_comparison_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])

    with (comparison_dir / "angle_comparison_summary.json").open("w") as f:
        json.dump(json_ready({"rows": rows}), f, indent=2)

    with (comparison_dir / "angle_comparison_summary.md").open("w") as f:
        f.write("| method | liveness | safety | STL | subgoal | avg steps |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['label_text']} | {row['liveness_pct']:.1f}% | {row['safety_pct']:.1f}% | "
                f"{row['stl_pct']:.1f}% | {row['subgoal_pct']:.1f}% | {row['avg_termination_step']:.1f} |\n"
            )


def plot_rates(comparison_dir: Path, rows: list[dict[str, Any]]) -> Path:
    labels = [row["label_text"] for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METRICS))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    metric_colors = {
        "liveness_satisfaction_rate": "#93c5fd",
        "safety_satisfaction_rate": "#86efac",
        "stl_satisfaction_rate": "#275fca",
        "subgoal_completion_rate": "#cbd5e1",
    }
    for offset, (metric, metric_label) in zip(offsets, METRICS):
        pct_key = {
            "liveness_satisfaction_rate": "liveness_pct",
            "safety_satisfaction_rate": "safety_pct",
            "stl_satisfaction_rate": "stl_pct",
            "subgoal_completion_rate": "subgoal_pct",
        }[metric]
        values = np.asarray([row[pct_key] for row in rows], dtype=np.float32)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=metric_label,
            color=metric_colors[metric],
            edgecolor="#334155",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(float(value) + 2.0, 103.0),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(f"Angle Safety Comparison ({rows[0]['n_rollouts']} rollouts)")
    ax.set_ylabel("rate [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", ncols=4, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path = comparison_dir / "angle_comparison_rates.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None, help="Override the angle rollout horizon for all methods.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--angle-goal-deg", type=float, default=None)
    parser.add_argument(
        "--angle-random-range-deg",
        type=float,
        nargs=2,
        default=list(ANGLE_RANDOM_RANGE_DEG),
        metavar=("MIN", "MAX"),
        help="Sample each rollout's angle target uniformly from this degree range.",
    )
    parser.add_argument(
        "--angle-tolerance-deg",
        type=float,
        default=ANGLE_TOLERANCE_DEG,
        help="Angle tolerance in degrees; converted to the Rzz margin used by the safety scorer.",
    )
    parser.add_argument("--gpc-n-candidates", type=int, default=16)
    parser.add_argument("--gpc-chunk-horizon", type=int, default=10)
    parser.add_argument("--device", default="auto", help="Device passed to all underlying runners.")
    parser.add_argument("--no-video", action="store_true", help="Disable videos for the underlying runs.")
    parser.add_argument("--disable-safety-randomization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_name = args.name or f"angle_compare_N{args.n_rollouts}_{time.strftime('%Y%m%d_%H%M%S')}"
    comparison_dir = unique_dir(output_root, run_name)

    commands = {
        method["key"]: command_for_method(method, args, comparison_dir)
        for method in METHODS
    }
    run_args = {
        "output_dir": comparison_dir,
        "n_rollouts": int(args.n_rollouts),
        "horizon": None if args.horizon is None else int(args.horizon),
        "seed_start": int(args.seed_start),
        "angle_goal_deg": args.angle_goal_deg,
        "angle_random_range_deg": args.angle_random_range_deg,
        "angle_tolerance_deg": args.angle_tolerance_deg,
        "device": args.device,
        "video": not bool(args.no_video),
        "disable_safety_randomization": bool(args.disable_safety_randomization),
        "commands": commands,
    }

    if args.dry_run:
        print(json.dumps(json_ready(run_args), indent=2))
        return

    comparison_dir.mkdir(parents=True, exist_ok=False)
    log_dir = comparison_dir / "logs"
    with (comparison_dir / "run_args.json").open("w") as f:
        json.dump(json_ready(run_args), f, indent=2)

    for method in METHODS:
        key = method["key"]
        run_and_log(commands[key], log_path=log_dir / f"{key}.log", prefix=key)

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        key = method["key"]
        run_dir = comparison_dir / key
        summary = load_angle_summary(run_dir)
        rows.append(
            {
                "method": key,
                "label": method["label"],
                "label_text": {"hint2": "hint^2", "flower": "FLOWER", "flower_gpc": "FLOWER+GPC"}[key],
                "run_dir": str(run_dir),
                "raw_summary": summary,
                "n_rollouts": int(summary.get("n_rollouts", args.n_rollouts)),
                "liveness_pct": rate_to_percent(summary.get("liveness_satisfaction_rate")),
                "safety_pct": rate_to_percent(summary.get("safety_satisfaction_rate")),
                "stl_pct": rate_to_percent(summary.get("stl_satisfaction_rate")),
                "subgoal_pct": rate_to_percent(summary.get("subgoal_completion_rate")),
                "avg_termination_step": float(summary.get("avg_termination_step", 0.0)),
            }
        )

    write_summary_files(comparison_dir, rows)
    plot_path = plot_rates(comparison_dir, rows)
    print("\ncomparison:", comparison_dir)
    print("plot:", plot_path)
    print("table:", comparison_dir / "angle_comparison_summary.md")


if __name__ == "__main__":
    main()
