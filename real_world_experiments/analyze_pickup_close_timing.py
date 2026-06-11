"""Diagnose pickup timing, close offsets, and object-start distribution.

This script compares real-robot automaton rollouts against the condensed
training HDF. It is aimed at the specific failure mode where the policy nudges
the Cheez-It pack with an open gripper and the selected close action appears at
the first action of a chunk.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_DATASET = Path("data/real_world/cheezit_pouring_condensed.hdf5")
DEFAULT_RUN_DIRS = (
    Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_1"),
    Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_2"),
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/real_world/paper_rollouts/automaton_sequence_eval/"
    "candidate_mode_analysis/pickup_close_timing"
)

HORIZON = 8
NOOP_POS_THRESHOLD = 0.001
NOOP_ROT_THRESHOLD_RAD = np.deg2rad(0.5)


def demo_sort_key(key: str) -> int:
    return int(key.split("_")[-1])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_positive_offset(chunk: np.ndarray) -> int | None:
    idx = np.flatnonzero(np.asarray(chunk)[:, 6] > 0.0)
    return int(idx[0]) if len(idx) else None


def nearest_dist_mm(points_xy: np.ndarray, query_xy: np.ndarray) -> float:
    d = np.linalg.norm(points_xy - query_xy[None, :], axis=1)
    return float(np.min(d) * 1000.0)


def quantiles(values: list[float] | np.ndarray, qs=(0.0, 0.1, 0.5, 0.9, 1.0)) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {f"q{int(q * 100):02d}": None for q in qs}
    return {f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def summarize_numeric(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": int(len(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def summarize_vector(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def count_idle_before(actions: np.ndarray, close_t: int) -> int:
    count = 0
    for idx in range(close_t - 1, -1, -1):
        pos_norm = float(np.linalg.norm(actions[idx, :3]))
        rot_norm = float(np.linalg.norm(actions[idx, 3:6]))
        if actions[idx, 6] < 0.0 and pos_norm < NOOP_POS_THRESHOLD and rot_norm < NOOP_ROT_THRESHOLD_RAD:
            count += 1
        else:
            break
    return count


def count_idle_after(actions: np.ndarray, close_t: int) -> int:
    count = 0
    for idx in range(close_t + 1, len(actions)):
        pos_norm = float(np.linalg.norm(actions[idx, :3]))
        rot_norm = float(np.linalg.norm(actions[idx, 3:6]))
        if actions[idx, 6] > 0.0 and pos_norm < NOOP_POS_THRESHOLD and rot_norm < NOOP_ROT_THRESHOLD_RAD:
            count += 1
        else:
            break
    return count


def analyze_training_hdf(dataset: Path) -> tuple[dict, dict]:
    start_obj_xy = []
    preclose_obj_xy = []
    preclose_state9 = []
    preclose_rel = []
    close_offsets = Counter()
    close_action_pos_norms = []
    close_action_rot_norms = []
    idle_before = []
    idle_after = []
    close_transition_count = 0

    with h5py.File(dataset, "r") as f:
        for demo_key in sorted(f["data"].keys(), key=demo_sort_key):
            demo = f[f"data/{demo_key}"]
            obs = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            eef = np.asarray(obs["eef_pos"], dtype=np.float32)
            obj = np.asarray(obs["cheezit_pos"], dtype=np.float32)
            grip = np.asarray(obs["gripper_binary"], dtype=np.float32).reshape(-1)

            start_obj_xy.append(obj[0, :2])
            transitions = np.flatnonzero((grip[:-1] < 0.0) & (grip[1:] > 0.0))
            if len(transitions):
                close_t = int(transitions[0])
                close_transition_count += 1
                preclose_obj_xy.append(obj[close_t, :2])
                rel = obj[close_t] - eef[close_t]
                preclose_rel.append(rel)
                preclose_state9.append(np.concatenate([eef[close_t], obj[close_t], rel], axis=0))
                close_action_pos_norms.append(float(np.linalg.norm(actions[close_t, :3])))
                close_action_rot_norms.append(float(np.linalg.norm(actions[close_t, 3:6])))
                idle_before.append(count_idle_before(actions, close_t))
                idle_after.append(count_idle_after(actions, close_t))

            for start in range(0, max(0, len(actions) - HORIZON + 1)):
                if grip[start] > 0.0:
                    continue
                offset = first_positive_offset(actions[start : start + HORIZON])
                if offset is not None:
                    close_offsets[offset] += 1

    start_obj_xy = np.asarray(start_obj_xy, dtype=np.float64)
    preclose_obj_xy = np.asarray(preclose_obj_xy, dtype=np.float64)
    preclose_state9 = np.asarray(preclose_state9, dtype=np.float64)
    preclose_rel = np.asarray(preclose_rel, dtype=np.float64)

    state9_mean = preclose_state9.mean(axis=0)
    state9_std = preclose_state9.std(axis=0) + 1e-6

    summary = {
        "dataset": str(dataset),
        "num_demos": int(len(start_obj_xy)),
        "close_transition_count": int(close_transition_count),
        "start_object_xy_m": summarize_vector(start_obj_xy),
        "preclose_object_xy_m": summarize_vector(preclose_obj_xy),
        "preclose_object_minus_eef_m": summarize_vector(preclose_rel),
        "training_close_offset_counts_open_start_chunks": {str(k): int(v) for k, v in sorted(close_offsets.items())},
        "first_close_action_pos_norm_m": summarize_numeric(close_action_pos_norms),
        "first_close_action_rot_norm_rad": summarize_numeric(close_action_rot_norms),
        "idle_open_actions_before_close": summarize_numeric(idle_before),
        "idle_closed_actions_after_close": summarize_numeric(idle_after),
    }
    arrays = {
        "start_obj_xy": start_obj_xy,
        "preclose_obj_xy": preclose_obj_xy,
        "preclose_state9": preclose_state9,
        "preclose_rel": preclose_rel,
        "state9_mean": state9_mean,
        "state9_std": state9_std,
    }
    return summary, arrays


def rollout_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "rollouts").glob("rollout_*/events.jsonl"))


def analyze_run(run_dir: Path, train_arrays: dict) -> tuple[dict, list[dict]]:
    run_config = json.loads((run_dir / "run_config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    target_chain = run_config.get("target_chain_parsed") or []

    selected_offsets = []
    flushed_counts = []
    start_nearest_mm = []
    preclose_nearest_mm = []
    push_mm = []
    state9_nearest_norm = []
    selected_p_can = []
    selected_p_target = []
    candidate_offset_counts = Counter()
    p_can_by_offset = defaultdict(list)
    preclose_selected_offsets = []
    preclose_selected_p_can = []
    preclose_candidate_close_counts = []
    preclose_candidate_none_counts = []
    preclose_no_close_p_can_max = []
    preclose_close_p_can_max = []
    rollout_rows = []

    for events_path in rollout_paths(run_dir):
        rows = load_jsonl(events_path)
        chunks = {int(r["chunk_idx"]): r for r in rows if r.get("type") == "chunk_sample"}
        start_row = next((r for r in rows if r.get("type") == "rollout_start"), None)
        close_decision = next(
            (
                r
                for r in rows
                if r.get("type") == "decision"
                and int(r.get("flushed_actions_after_gripper_close", 0)) > 0
            ),
            None,
        )
        if start_row is None or close_decision is None:
            continue

        chunk = chunks.get(int(close_decision["chunk_idx"]))
        if chunk is None:
            continue

        for chunk_idx in range(int(close_decision["chunk_idx"])):
            pre_chunk = chunks.get(chunk_idx)
            if pre_chunk is None:
                continue
            pre_action_chunks = np.asarray(pre_chunk["candidate_action_chunks"], dtype=np.float32)
            pre_probs = np.asarray(pre_chunk["label_probs"], dtype=np.float32)
            pre_selected_idx = int(pre_chunk["selected_candidate"])
            pre_offsets = [first_positive_offset(candidate) for candidate in pre_action_chunks]
            pre_none_mask = np.asarray([offset is None for offset in pre_offsets], dtype=bool)
            pre_close_mask = ~pre_none_mask
            preclose_selected_offsets.append(
                "none"
                if pre_offsets[pre_selected_idx] is None
                else str(int(pre_offsets[pre_selected_idx]))
            )
            preclose_selected_p_can.append(float(pre_probs[pre_selected_idx, 0]))
            preclose_candidate_close_counts.append(int(np.sum(pre_close_mask)))
            preclose_candidate_none_counts.append(int(np.sum(pre_none_mask)))
            if np.any(pre_none_mask):
                preclose_no_close_p_can_max.append(float(np.max(pre_probs[pre_none_mask, 0])))
            if np.any(pre_close_mask):
                preclose_close_p_can_max.append(float(np.max(pre_probs[pre_close_mask, 0])))

        selected_chunk = np.asarray(chunk["selected_chunk"], dtype=np.float32)
        selected_offset = first_positive_offset(selected_chunk)
        selected_offsets.append(-1 if selected_offset is None else selected_offset)
        flushed_counts.append(int(close_decision.get("flushed_actions_after_gripper_close", 0)))

        label_probs = np.asarray(chunk["label_probs"], dtype=np.float32)
        action_chunks = np.asarray(chunk["candidate_action_chunks"], dtype=np.float32)
        for cand_idx, cand in enumerate(action_chunks):
            offset = first_positive_offset(cand)
            key = "none" if offset is None else int(offset)
            candidate_offset_counts[key] += 1
            if offset is not None:
                p_can_by_offset[int(offset)].append(float(label_probs[cand_idx, 0]))

        selected_idx = int(chunk["selected_candidate"])
        selected_p_can.append(float(label_probs[selected_idx, 0]))
        target_idx = None
        selection = chunk.get("selection") or {}
        if selection.get("target_label_idx") is not None:
            target_idx = int(selection["target_label_idx"])
            selected_p_target.append(float(label_probs[selected_idx, target_idx]))

        start_obj = np.asarray(start_row["obs"]["cheezit_pos"], dtype=np.float64)
        preclose_obj = np.asarray(close_decision["obs"]["cheezit_pos"], dtype=np.float64)
        preclose_eef = np.asarray(close_decision["obs"]["eef_pos"], dtype=np.float64)
        rel = preclose_obj - preclose_eef
        state9 = np.concatenate([preclose_eef, preclose_obj, rel], axis=0)
        norm_state9 = (state9 - train_arrays["state9_mean"]) / train_arrays["state9_std"]
        train_norm = (train_arrays["preclose_state9"] - train_arrays["state9_mean"]) / train_arrays["state9_std"]
        state9_dist = np.linalg.norm(train_norm - norm_state9[None, :], axis=1)

        start_dist = nearest_dist_mm(train_arrays["start_obj_xy"], start_obj[:2])
        preclose_dist = nearest_dist_mm(train_arrays["preclose_obj_xy"], preclose_obj[:2])
        push_dist = float(np.linalg.norm(preclose_obj[:2] - start_obj[:2]) * 1000.0)
        start_nearest_mm.append(start_dist)
        preclose_nearest_mm.append(preclose_dist)
        push_mm.append(push_dist)
        state9_nearest_norm.append(float(np.min(state9_dist)))

        rollout_idx = int(events_path.parent.name.split("_")[-1])
        rollout_rows.append(
            {
                "run_name": run_dir.name,
                "rollout_idx": rollout_idx,
                "success": bool((summary.get("rollouts") or [{}])[rollout_idx].get("success", False))
                if rollout_idx < len(summary.get("rollouts", []))
                else None,
                "target_chain": target_chain,
                "close_chunk_idx": int(chunk["chunk_idx"]),
                "close_decision_idx": int(close_decision["decision_idx"]),
                "selected_close_offset": None if selected_offset is None else int(selected_offset),
                "flushed_actions_after_gripper_close": int(close_decision.get("flushed_actions_after_gripper_close", 0)),
                "selected_p_can_grabbed": float(label_probs[selected_idx, 0]),
                "selected_p_target": None if target_idx is None else float(label_probs[selected_idx, target_idx]),
                "start_cheezit_xy_m": start_obj[:2].astype(float).tolist(),
                "preclose_cheezit_xy_m": preclose_obj[:2].astype(float).tolist(),
                "preclose_eef_xy_m": preclose_eef[:2].astype(float).tolist(),
                "preclose_object_minus_eef_m": rel.astype(float).tolist(),
                "start_nearest_train_start_object_xy_mm": start_dist,
                "preclose_nearest_train_preclose_object_xy_mm": preclose_dist,
                "object_push_start_to_preclose_xy_mm": push_dist,
                "preclose_nearest_train_state9_norm": float(np.min(state9_dist)),
            }
        )

    p_can_by_offset_summary = {
        str(k): {
            "n": int(len(v)),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "max": float(np.max(v)),
            "quantiles": quantiles(v),
        }
        for k, v in sorted(p_can_by_offset.items())
    }

    run_summary = {
        "run_dir": str(run_dir),
        "success_rate": summary.get("success_rate"),
        "target_chain_parsed": target_chain,
        "num_rollouts_with_close": int(len(rollout_rows)),
        "selected_close_offset_counts": {str(k): int(v) for k, v in sorted(Counter(selected_offsets).items())},
        "flushed_actions_after_gripper_close": summarize_numeric(flushed_counts),
        "candidate_close_offset_counts_at_close_chunks": {str(k): int(v) for k, v in sorted(candidate_offset_counts.items(), key=lambda item: str(item[0]))},
        "candidate_p_can_grabbed_by_close_offset": p_can_by_offset_summary,
        "preclose_chunks_before_first_grip": {
            "num_chunks": int(len(preclose_selected_offsets)),
            "selected_close_offset_counts": {
                str(k): int(v) for k, v in sorted(Counter(preclose_selected_offsets).items(), key=lambda item: str(item[0]))
            },
            "selected_p_can_grabbed": summarize_numeric(preclose_selected_p_can),
            "candidate_close_count_per_chunk": summarize_numeric(preclose_candidate_close_counts),
            "candidate_no_close_count_per_chunk": summarize_numeric(preclose_candidate_none_counts),
            "max_p_can_among_no_close_candidates": summarize_numeric(preclose_no_close_p_can_max),
            "max_p_can_among_close_candidates": summarize_numeric(preclose_close_p_can_max),
        },
        "selected_p_can_grabbed": summarize_numeric(selected_p_can),
        "selected_p_current_target": summarize_numeric(selected_p_target),
        "start_nearest_train_start_object_xy_mm": summarize_numeric(start_nearest_mm),
        "preclose_nearest_train_preclose_object_xy_mm": summarize_numeric(preclose_nearest_mm),
        "object_push_start_to_preclose_xy_mm": summarize_numeric(push_mm),
        "preclose_nearest_train_state9_norm": summarize_numeric(state9_nearest_norm),
        "preclose_object_minus_eef_m": summarize_vector(
            np.asarray([row["preclose_object_minus_eef_m"] for row in rollout_rows], dtype=np.float64)
        ),
    }
    return run_summary, rollout_rows


def plot_object_distribution(output_path: Path, train_arrays: dict, rollout_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        train_arrays["start_obj_xy"][:, 0],
        train_arrays["start_obj_xy"][:, 1],
        s=18,
        alpha=0.35,
        color="tab:blue",
        label="train object start",
    )
    ax.scatter(
        train_arrays["preclose_obj_xy"][:, 0],
        train_arrays["preclose_obj_xy"][:, 1],
        s=18,
        alpha=0.35,
        color="tab:green",
        label="train object pre-close",
    )
    for row in rollout_rows:
        start = np.asarray(row["start_cheezit_xy_m"])
        close = np.asarray(row["preclose_cheezit_xy_m"])
        ax.plot([start[0], close[0]], [start[1], close[1]], color="tab:red", alpha=0.45, linewidth=1.2)
    if rollout_rows:
        starts = np.asarray([row["start_cheezit_xy_m"] for row in rollout_rows], dtype=float)
        closes = np.asarray([row["preclose_cheezit_xy_m"] for row in rollout_rows], dtype=float)
        ax.scatter(starts[:, 0], starts[:, 1], s=55, color="tab:red", marker="x", label="rollout object start")
        ax.scatter(closes[:, 0], closes[:, 1], s=55, color="tab:orange", marker="o", label="rollout object pre-close")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title("Object XY: training starts/pre-close vs rollout push before grip")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_close_offsets(output_path: Path, train_summary: dict, run_summaries: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(HORIZON)
    train_counts = train_summary["training_close_offset_counts_open_start_chunks"]
    axes[0].bar(xs, [train_counts.get(str(x), 0) for x in xs], color="0.45")
    axes[0].set_title("training chunks: first close offset")
    axes[0].set_xlabel("offset inside H=8 chunk")
    axes[0].set_ylabel("count")
    axes[0].set_xticks(xs)

    width = 0.35
    for idx, summary in enumerate(run_summaries):
        counts = summary["selected_close_offset_counts"]
        offset = (idx - (len(run_summaries) - 1) / 2) * width
        axes[1].bar(
            xs + offset,
            [counts.get(str(x), 0) for x in xs],
            width=width,
            label=Path(summary["run_dir"]).name,
        )
    axes[1].set_title("selected rollout close offset")
    axes[1].set_xlabel("offset inside selected chunk")
    axes[1].set_ylabel("rollout count")
    axes[1].set_xticks(xs)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pcan_by_offset(output_path: Path, run_summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for run_idx, summary in enumerate(run_summaries):
        offsets = []
        medians = []
        q10 = []
        q90 = []
        for offset_str, stats in summary["candidate_p_can_grabbed_by_close_offset"].items():
            offsets.append(int(offset_str))
            medians.append(stats["median"])
            q10.append(stats["quantiles"]["q10"])
            q90.append(stats["quantiles"]["q90"])
        order = np.argsort(offsets)
        offsets = np.asarray(offsets)[order]
        medians = np.asarray(medians)[order]
        q10 = np.asarray(q10)[order]
        q90 = np.asarray(q90)[order]
        jitter = (run_idx - (len(run_summaries) - 1) / 2) * 0.04
        color = colors[run_idx % len(colors)]
        ax.plot(offsets + jitter, medians, marker="o", color=color, label=Path(summary["run_dir"]).name)
        ax.fill_between(offsets + jitter, q10, q90, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("candidate first close offset")
    ax.set_ylabel("automaton p(can_grabbed)")
    ax.set_title("Candidate grabbed score is high for many offsets, but argmax favors earliest close")
    ax.set_xticks(np.arange(HORIZON))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, nargs="*", default=list(DEFAULT_RUN_DIRS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_summary, train_arrays = analyze_training_hdf(args.dataset)
    run_summaries = []
    rollout_rows = []
    for run_dir in args.run_dir:
        summary, rows = analyze_run(run_dir, train_arrays)
        run_summaries.append(summary)
        rollout_rows.extend(rows)

    result = {
        "training": train_summary,
        "runs": run_summaries,
        "rollouts": rollout_rows,
    }
    (args.output_dir / "pickup_close_timing_summary.json").write_text(json.dumps(result, indent=2))
    plot_object_distribution(args.output_dir / "object_start_preclose_distribution.png", train_arrays, rollout_rows)
    plot_close_offsets(args.output_dir / "close_offset_histograms.png", train_summary, run_summaries)
    plot_pcan_by_offset(args.output_dir / "candidate_p_can_by_close_offset.png", run_summaries)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "training_close_offset_counts": train_summary["training_close_offset_counts_open_start_chunks"],
        "runs": [
            {
                "run_dir": summary["run_dir"],
                "selected_close_offset_counts": summary["selected_close_offset_counts"],
                "flushed": summary["flushed_actions_after_gripper_close"],
                "start_nearest_train_start_object_xy_mm": summary["start_nearest_train_start_object_xy_mm"],
                "preclose_nearest_train_preclose_object_xy_mm": summary["preclose_nearest_train_preclose_object_xy_mm"],
                "object_push_start_to_preclose_xy_mm": summary["object_push_start_to_preclose_xy_mm"],
            }
            for summary in run_summaries
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
