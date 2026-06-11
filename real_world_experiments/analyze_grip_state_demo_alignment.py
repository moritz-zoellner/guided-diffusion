"""Compare post-grip rollout states to left/right training-demo manifolds.

For each real-robot rollout state after ``can_grabbed`` flips, this computes
how close the state is to the gripper-closed, pre-pour portion of left-first
and right-first training demos. The goal is to test whether DP mode support
tracks proximity to one class of demonstrations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world_experiments.analyze_dp_post_grip_mode_sampling import (
    available_reached_decisions,
    grip_to_first_resample_decisions,
    reconstruct_stacks_at_target_reached,
)
from real_world_experiments.label_real_world import LabelConfig, label_demo


OBS_KEYS = ("eef_pos", "eef_rot6d", "cheezit_pos", "cheezit_rot6d")
RIGHT_IDX = 1
LEFT_IDX = 2


def obs_vec_from_hdf_demo(demo: h5py.Group, idxs: np.ndarray) -> np.ndarray:
    obs = demo["obs"]
    return np.concatenate([obs[key][idxs].astype(np.float32) for key in OBS_KEYS], axis=-1)


def obs_vec_from_rollout_obs(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in OBS_KEYS], axis=0)


def classify_first_pour(labels: np.ndarray) -> tuple[str, int | None]:
    pour = np.where((labels[:, RIGHT_IDX] > 0.5) | (labels[:, LEFT_IDX] > 0.5))[0]
    if len(pour) == 0:
        return "none", None
    first = int(pour[0])
    cls = "right" if labels[first, RIGHT_IDX] > 0.5 else "left"
    return cls, first


def build_demo_manifolds(dataset: Path, label_config: LabelConfig):
    demos = []
    all_states = []
    with h5py.File(dataset, "r") as f:
        for demo_key in sorted(f["data"].keys(), key=lambda key: int(key.split("_")[-1])):
            demo = f[f"data/{demo_key}"]
            labels = label_demo(demo, config=label_config)
            cls, first_pour_idx = classify_first_pour(labels)
            if cls not in {"left", "right"} or first_pour_idx is None:
                continue
            idx = np.arange(len(labels))
            mask = (
                (labels[:, 0] > 0.5)
                & (labels[:, RIGHT_IDX] < 0.5)
                & (labels[:, LEFT_IDX] < 0.5)
                & (idx < first_pour_idx)
            )
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            states = obs_vec_from_hdf_demo(demo, idxs)
            demos.append({
                "demo_key": demo_key,
                "class": cls,
                "num_states": int(len(states)),
                "states": states,
            })
            all_states.append(states)
    if not demos:
        raise ValueError(f"No classified pre-pour closed demo states found in {dataset}")
    all_states = np.concatenate(all_states, axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0) + 1e-6
    for demo in demos:
        demo["states_norm"] = (demo["states"] - mean) / std
    by_class = {"left": [], "right": []}
    for demo in demos:
        by_class[demo["class"]].append(demo)
    state_cloud_by_class = {
        cls: np.concatenate([demo["states_norm"] for demo in class_demos], axis=0)
        for cls, class_demos in by_class.items()
    }
    return demos, by_class, state_cloud_by_class, mean, std


def mean_min_demo_distance(query_norm: np.ndarray, class_demos: list[dict]) -> dict:
    per_demo = []
    for demo in class_demos:
        diff = demo["states_norm"] - query_norm[None, :]
        dists = np.linalg.norm(diff, axis=1)
        per_demo.append(float(dists.min()))
    arr = np.asarray(per_demo, dtype=np.float64)
    return {
        "mean_min": float(arr.mean()),
        "median_min": float(np.median(arr)),
        "q10_min": float(np.quantile(arr, 0.10)),
        "q90_min": float(np.quantile(arr, 0.90)),
    }


def knn_fraction(query_norm: np.ndarray, state_cloud_by_class: dict[str, np.ndarray], k: int) -> dict:
    clouds = []
    labels = []
    for cls in ["left", "right"]:
        cloud = state_cloud_by_class[cls]
        clouds.append(cloud)
        labels.extend([cls] * len(cloud))
    cloud = np.concatenate(clouds, axis=0)
    labels = np.asarray(labels)
    dists = np.linalg.norm(cloud - query_norm[None, :], axis=1)
    order = np.argsort(dists)[: min(k, len(dists))]
    return {
        "k": int(len(order)),
        "left_frac": float(np.mean(labels[order] == "left")),
        "right_frac": float(np.mean(labels[order] == "right")),
        "mean_dist": float(dists[order].mean()),
    }


def reconstruct_rollout_offset_states(run_dir: Path, max_offset: int):
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = []
    for rollout in summary["rollouts"]:
        rollout_idx = int(rollout["rollout_idx"])
        events_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}" / "events.jsonl"
        decisions = grip_to_first_resample_decisions(events_path)
        grip_decision = decisions[0]
        reached = available_reached_decisions(events_path)
        wanted = {
            grip_decision + offset
            for offset in range(max_offset + 1)
            if grip_decision + offset in reached
        }
        stacks = reconstruct_stacks_at_target_reached(events_path, wanted)
        for offset in range(max_offset + 1):
            decision_idx = grip_decision + offset
            if decision_idx not in stacks:
                continue
            item = stacks[decision_idx]
            rows.append({
                "run_name": run_dir.name,
                "rollout_idx": rollout_idx,
                "offset_after_grip": int(offset),
                "decision_idx": int(decision_idx),
                "grip_decision_idx": int(grip_decision),
                "first_event_sequence": " -> ".join(e["label_name"] for e in rollout.get("label_events", [])),
                "success": bool(rollout.get("success", False)),
                "termination_reason": rollout.get("termination_reason"),
                "wrist_joint_delta_deg": item.get("wrist_joint_delta_deg"),
                "current_label": item["label"].astype(int).tolist(),
                "obs_vec": obs_vec_from_rollout_obs(item["step_obs"]),
            })
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def add_alignment(rows: list[dict], mode_rows: list[dict], by_class, state_cloud_by_class, mean, std, k: int):
    mode_lookup = {
        (row["run_name"], int(row["rollout_idx"]), int(row["offset_after_grip"])): row
        for row in mode_rows
    }
    out = []
    for row in rows:
        query_norm = (row["obs_vec"] - mean) / std
        left_dist = mean_min_demo_distance(query_norm, by_class["left"])
        right_dist = mean_min_demo_distance(query_norm, by_class["right"])
        knn = knn_fraction(query_norm, state_cloud_by_class, k=k)
        key = (row["run_name"], int(row["rollout_idx"]), int(row["offset_after_grip"]))
        mode = mode_lookup.get(key, {})
        clean = {k2: v for k2, v in row.items() if k2 != "obs_vec"}
        clean.update({
            "left_demo_distance": left_dist,
            "right_demo_distance": right_dist,
            "demo_alignment_delta": float(right_dist["mean_min"] - left_dist["mean_min"]),
            "demo_alignment_delta_median": float(right_dist["median_min"] - left_dist["median_min"]),
            "knn_demo_alignment": knn,
            "mode_support": {
                "delta_mean": mode.get("delta_mean"),
                "delta_min": mode.get("delta_min"),
                "delta_max": mode.get("delta_max"),
                "right_gt_02_frac": None if not mode else float(mode["right_gt_02"] / mode["n"]),
                "left_gt_02_frac": None if not mode else float(mode["left_gt_02"] / mode["n"]),
                "right_wins_frac": None if not mode else float(mode["right_wins"] / mode["n"]),
                "left_wins_frac": None if not mode else float(mode["left_wins"] / mode["n"]),
            },
        })
        out.append(clean)
    return out


def aggregate_and_correlate(rows: list[dict]) -> dict:
    aggregate = []
    correlations = {}
    for offset in sorted({row["offset_after_grip"] for row in rows}):
        sub = [row for row in rows if row["offset_after_grip"] == offset]
        aggregate.append({
            "offset_after_grip": int(offset),
            "n": int(len(sub)),
            "alignment_delta_mean": float(np.mean([row["demo_alignment_delta"] for row in sub])),
            "alignment_delta_std": float(np.std([row["demo_alignment_delta"] for row in sub])),
            "knn_left_frac_mean": float(np.mean([row["knn_demo_alignment"]["left_frac"] for row in sub])),
            "mode_delta_mean": float(np.mean([row["mode_support"]["delta_mean"] for row in sub if row["mode_support"]["delta_mean"] is not None])),
        })
    valid = [row for row in rows if row["mode_support"]["delta_mean"] is not None]
    pairs = {
        "mode_delta_mean_vs_demo_alignment_delta": (
            [row["mode_support"]["delta_mean"] for row in valid],
            [row["demo_alignment_delta"] for row in valid],
        ),
        "left_support_vs_demo_alignment_delta": (
            [row["mode_support"]["left_gt_02_frac"] for row in valid],
            [row["demo_alignment_delta"] for row in valid],
        ),
        "right_support_vs_demo_alignment_delta": (
            [row["mode_support"]["right_gt_02_frac"] for row in valid],
            [row["demo_alignment_delta"] for row in valid],
        ),
        "knn_left_frac_vs_mode_delta_mean": (
            [row["knn_demo_alignment"]["left_frac"] for row in valid],
            [row["mode_support"]["delta_mean"] for row in valid],
        ),
    }
    for name, (x, y) in pairs.items():
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        correlations[name] = {
            "pearson": corr(x_arr, y_arr),
            "spearman": corr(rankdata(x_arr), rankdata(y_arr)),
        }
    return {"aggregate_by_offset": aggregate, "correlations": correlations}


def plot_alignment(output_dir: Path, rows: list[dict], aggregate: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    x = [row["demo_alignment_delta"] for row in rows]
    y = [row["mode_support"]["delta_mean"] for row in rows]
    c = [row["offset_after_grip"] for row in rows]
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=28, alpha=0.8)
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("demo alignment: dist(right demos) - dist(left demos)\npositive = closer to left demos")
    ax.set_ylabel("DP mode bias: p(left) - p(right)")
    ax.set_title("DP mode bias tracks demo-manifold alignment")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="offset after can_grabbed")
    fig.tight_layout()
    fig.savefig(output_dir / "mode_bias_vs_demo_alignment.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.asarray([row["offset_after_grip"] for row in aggregate], dtype=float)
    ax.errorbar(
        offsets,
        [row["alignment_delta_mean"] for row in aggregate],
        yerr=[row["alignment_delta_std"] for row in aggregate],
        marker="o",
        label="demo alignment delta",
    )
    ax.plot(offsets, [row["mode_delta_mean"] for row in aggregate], marker="s", label="mode delta mean")
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("offset after can_grabbed")
    ax.set_ylabel("signed left-vs-right score")
    ax.set_title("Alignment and DP mode bias over leftover chunk execution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_and_mode_bias_by_offset.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/real_world/cheezit_pouring_condensed.hdf5"))
    parser.add_argument(
        "--mode-support-summary",
        type=Path,
        default=Path(
            "outputs/real_world/paper_rollouts/automaton_sequence_eval/candidate_mode_analysis"
            "/post_grip_dp_resampling/all_grip_offsets/all_grip_offset_mode_support_summary.json"
        ),
    )
    parser.add_argument(
        "--right-run",
        type=Path,
        default=Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_right_epoch160_n10"),
    )
    parser.add_argument(
        "--left-run",
        type=Path,
        default=Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3"),
    )
    parser.add_argument("--max-offset", type=int, default=8)
    parser.add_argument("--knn", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/real_world/paper_rollouts/automaton_sequence_eval/candidate_mode_analysis"
            "/post_grip_demo_alignment"
        ),
    )
    args = parser.parse_args()

    label_config = LabelConfig()
    demos, by_class, state_cloud_by_class, mean, std = build_demo_manifolds(args.dataset, label_config)
    print(
        "Training pre-pour closed demos:",
        {cls: len(items) for cls, items in by_class.items()},
        "states:",
        {cls: int(len(state_cloud_by_class[cls])) for cls in state_cloud_by_class},
    )
    rollout_rows = []
    for run_dir in [args.right_run, args.left_run]:
        rollout_rows.extend(reconstruct_rollout_offset_states(run_dir, max_offset=args.max_offset))
    mode_rows = json.loads(args.mode_support_summary.read_text())["rows"]
    aligned = add_alignment(rollout_rows, mode_rows, by_class, state_cloud_by_class, mean, std, k=args.knn)
    analysis = aggregate_and_correlate(aligned)
    report = {
        "dataset": str(args.dataset),
        "label_config": label_config.__dict__,
        "obs_keys": list(OBS_KEYS),
        "normalization_source": "all gripper-closed pre-pour demo states from both classes",
        "demo_counts": {cls: len(items) for cls, items in by_class.items()},
        "demo_state_counts": {cls: int(len(state_cloud_by_class[cls])) for cls in state_cloud_by_class},
        **analysis,
        "rows": aligned,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "post_grip_demo_alignment_summary.json"
    out_path.write_text(json.dumps(report, indent=2))
    plot_alignment(args.output_dir, aligned, analysis["aggregate_by_offset"])
    print(f"Wrote {out_path}")
    print("Correlations:")
    print(json.dumps(analysis["correlations"], indent=2))
    print("Offset-0 rows sorted by demo alignment delta:")
    for row in sorted([r for r in aligned if r["offset_after_grip"] == 0], key=lambda r: r["demo_alignment_delta"]):
        mode = row["mode_support"]
        print(
            f"{row['run_name']} r{row['rollout_idx']:03d}: "
            f"align={row['demo_alignment_delta']:+.3f}, "
            f"mode_delta={mode['delta_mean']:+.3f}, "
            f"R={mode['right_gt_02_frac']:.3f}, L={mode['left_gt_02_frac']:.3f}, "
            f"seq={row['first_event_sequence']}"
        )


if __name__ == "__main__":
    main()
