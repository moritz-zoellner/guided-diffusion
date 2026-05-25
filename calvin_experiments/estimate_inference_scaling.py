#!/usr/bin/env python3
"""Estimate inference-time scaling over automaton distance.

The estimate is grounded in observed rollout logs:

1. Extract the distribution of environment steps per observed automaton subgoal.
2. Benchmark or provide one inference call for:
   - our automaton sample-and-rank method, including candidate diffusion samples
     and automaton scoring
   - FLOWER VLA, one language-conditioned action inference
3. Bootstrap automaton distances by sampling observed subgoal durations.

Only model inference time is estimated here. Environment stepping/rendering time is
not included; rollout logs only provide the number of environment steps taken per
subgoal.

Outputs are CSV/JSON files that plotting scripts can consume, plus a quick-look
PNG showing estimated inference time over automaton distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FLOWER_ROOT = REPO_ROOT / "flower_vla_calvin"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/timing"

for path in [
    REPO_ROOT,
    REPO_ROOT / "robomimic",
    REPO_ROOT / "calvin" / "calvin_env",
    REPO_ROOT / "calvin_experiments",
    REPO_ROOT / "calvin_experiments" / "paper_stls" / "baselines",
    FLOWER_ROOT,
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from calvin_experiments import calvin_rollout_utils as CRU
from calvin_experiments.complex_stl_experiment_utils import COMPLEX_STL_SPECS, unique_run_dir
from calvin_experiments.run_dynaguide_articulated_automaton import repo_path, resolve_existing_path, write_json
from calvin_experiments.run_complex_stl_automaton import (
    DEFAULT_COMPLEX_AUTOMATON_CKPT,
    DEFAULT_COMPLEX_POLICY_CKPT,
    score_candidate_batch,
)
from calvin_experiments.paper_stls.baselines.flower_our_env_rollout import (
    DEFAULT_ENV_CHECKPOINT,
    DEFAULT_FLOWER_CHECKPOINT,
    FlowerPolicyAdapter,
    load_flower_model,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--task-id", default="chained", choices=tuple(COMPLEX_STL_SPECS.keys()))
    parser.add_argument("--task-dir", type=Path, action="append", default=[], help="Shared duration source for all methods.")
    parser.add_argument("--chained-task-dir", type=Path, action="append", default=[], help="Legacy alias for --task-dir.")
    parser.add_argument("--ours-task-dir", type=Path, action="append", default=[], help="Duration source for our automaton method.")
    parser.add_argument("--vla-task-dir", type=Path, action="append", default=[], help="Duration source for base FLOWER/VLA.")
    parser.add_argument("--llm-static-task-dir", type=Path, action="append", default=[], help="Optional duration source for VLA+LLM static.")
    parser.add_argument("--llm-closed-task-dir", type=Path, action="append", default=[], help="Optional duration source for VLA+LLM closed-loop.")
    parser.add_argument("--search-root", type=Path, default=REPO_ROOT / "outputs/calvin_paper/complex-behaviors")
    parser.add_argument("--include-baselines-in-step-stats", action="store_true")
    parser.add_argument("--include-incomplete-rollouts", action="store_true")
    parser.add_argument("--max-distance", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--chunk-horizon", type=int, default=8, help="Fallback action chunk horizon if benchmarking is skipped.")
    parser.add_argument("--benchmark-iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-ckpt", type=Path, default=DEFAULT_COMPLEX_POLICY_CKPT)
    parser.add_argument("--automaton-ckpt", type=Path, default=DEFAULT_COMPLEX_AUTOMATON_CKPT)
    parser.add_argument("--flower-checkpoint", type=Path, default=DEFAULT_FLOWER_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=DEFAULT_ENV_CHECKPOINT)
    parser.add_argument("--scene-config", type=Path, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-flower-benchmark", action="store_true")
    parser.add_argument("--our-call-ms", type=float, default=None, help="Manual override for our per-chunk inference time.")
    parser.add_argument("--vla-call-ms", type=float, default=None, help="Manual override for FLOWER per-action inference time.")
    parser.add_argument("--llm-latency-sec", type=float, default=1.5)
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face downloads/checks instead of cache-only mode.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        return json.load(f)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def resolve_task_dir(path: Path, *, task_id: str) -> Path:
    path = repo_path(path)
    if path.name == "task_summary.json":
        return path.parent
    if (path / "task_summary.json").exists():
        return path
    if (path / task_id / "task_summary.json").exists():
        return path / task_id
    raise FileNotFoundError(f"Could not resolve {task_id!r} task directory from {path}")


def discover_task_dirs(search_root: Path, *, task_id: str, include_baselines: bool) -> list[Path]:
    formula = COMPLEX_STL_SPECS[task_id].formula
    root = repo_path(search_root)
    task_dirs = []
    for summary_path in sorted(root.glob(f"**/{task_id}/task_summary.json")):
        if not include_baselines and "baselines" in summary_path.parts:
            continue
        try:
            payload = load_json(summary_path)
        except Exception:
            continue
        if payload.get("formula") != formula:
            continue
        task_dirs.append(summary_path.parent)
    return task_dirs


def extract_subgoal_duration_rows(task_dirs: Sequence[Path], *, complete_only: bool) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for task_dir in task_dirs:
        summary = load_json(Path(task_dir) / "task_summary.json")
        for rollout in summary.get("rollouts", []):
            if complete_only and not rollout.get("stl_satisfied", False):
                continue
            events = sorted(rollout.get("target_events", []), key=lambda item: int(item.get("step", 0)))
            previous_step = 0
            for event_idx, event in enumerate(events):
                event_step = int(event["step"])
                duration = int(event_step - previous_step)
                if duration < 0:
                    previous_step = event_step
                    continue
                rows.append(
                    {
                        "source_task_dir": str(task_dir),
                        "seed": rollout.get("seed"),
                        "rollout_stl_satisfied": bool(rollout.get("stl_satisfied", False)),
                        "event_idx": int(event_idx),
                        "target_name": event.get("target_name"),
                        "previous_step": int(previous_step),
                        "event_step": int(event_step),
                        "env_steps": int(duration),
                    }
                )
                previous_step = event_step
    return rows


def method_duration_rows(
    method: str,
    task_dirs: Sequence[Path],
    *,
    complete_only: bool,
) -> tuple[list[Dict[str, Any]], bool]:
    rows = extract_subgoal_duration_rows(task_dirs, complete_only=complete_only)
    used_complete_only = bool(complete_only)
    if not rows and complete_only:
        rows = extract_subgoal_duration_rows(task_dirs, complete_only=False)
        used_complete_only = False
    for row in rows:
        row["duration_source_method"] = method
        row["complete_rollouts_only_for_this_source"] = bool(used_complete_only)
    return rows, used_complete_only


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p20": float(np.quantile(arr, 0.20)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p80": float(np.quantile(arr, 0.80)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def sync_if_needed(device) -> None:
    import torch

    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def time_callable(fn: Callable[[], Any], *, device, warmup_iters: int, benchmark_iters: int) -> tuple[list[float], Any]:
    last_result = None
    times_ms: list[float] = []
    for idx in range(int(warmup_iters) + int(benchmark_iters)):
        sync_if_needed(device)
        t0 = time.perf_counter()
        last_result = fn()
        sync_if_needed(device)
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        if idx >= int(warmup_iters):
            times_ms.append(float(elapsed_ms))
    return times_ms, last_result


def prepare_env_obs(env_ckpt_dict: Dict[str, Any], scene_config: Path, seed: int):
    env, base_env_state = CRU.load_fresh_env_from_checkpoint(env_ckpt_dict, seed=int(seed), suppress_output=True)
    fixed_scene, fixed_robot, _ = CRU.fixed_scene_robot_from_config(base_env_state, scene_config)
    obs = CRU.reset_env_to_scene_robot(env, fixed_scene, fixed_robot)
    return env, obs


def benchmark_our_method(args: argparse.Namespace, device) -> Dict[str, Any]:
    import robomimic.envs  # noqa: F401
    import robomimic.utils.file_utils as FileUtils

    from calvin_experiments.run_dynaguide_articulated_automaton import AutomatonGuidance

    policy_ckpt = resolve_existing_path(repo_path(args.policy_ckpt))
    automaton_ckpt = resolve_existing_path(repo_path(args.automaton_ckpt))
    scene_config = resolve_existing_path(repo_path(args.scene_config))
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(policy_ckpt), device=device, verbose=False)
    policy.start_episode()
    guidance = AutomatonGuidance(automaton_ckpt, device)
    env, obs = prepare_env_obs(ckpt_dict, scene_config, args.seed)
    try:
        def call():
            return score_candidate_batch(policy, guidance, obs, env, int(args.n_candidates))

        times_ms, result = time_callable(
            call,
            device=device,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        action_chunks, _, automaton_horizon, _ = result
        return {
            "method": "ours_automaton_sample_rank_chunk",
            "call_unit": "action_chunk",
            "n_candidates": int(args.n_candidates),
            "automaton_horizon": int(automaton_horizon),
            "action_chunk_shape": list(np.asarray(action_chunks).shape),
            **summarize_values(times_ms),
            "times_ms": times_ms,
        }
    finally:
        CRU.close_env_quietly(env)


def benchmark_flower(args: argparse.Namespace, device) -> Dict[str, Any]:
    import robomimic.envs  # noqa: F401
    import robomimic.utils.file_utils as FileUtils

    flower_checkpoint = resolve_existing_path(repo_path(args.flower_checkpoint))
    env_checkpoint = resolve_existing_path(repo_path(args.env_checkpoint))
    scene_config = resolve_existing_path(repo_path(args.scene_config))
    env_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(env_checkpoint))
    flower_model = load_flower_model(flower_checkpoint, device)
    instruction = COMPLEX_STL_SPECS[args.task_id].prompt
    flower_policy = FlowerPolicyAdapter(flower_model, instruction=instruction, device=device)
    env, obs = prepare_env_obs(env_ckpt_dict, scene_config, args.seed)
    try:
        flower_policy.reset(instruction)

        def call():
            return flower_policy(obs)

        times_ms, action = time_callable(
            call,
            device=device,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        return {
            "method": "flower_vla_action",
            "call_unit": "env_action",
            "instruction": instruction,
            "action_shape": list(np.asarray(action).shape),
            **summarize_values(times_ms),
            "times_ms": times_ms,
        }
    finally:
        CRU.close_env_quietly(env)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_stats_columns(row: Dict[str, Any], prefix: str, values: Sequence[float]) -> None:
    stats = summarize_values(values)
    for key in ("mean", "p10", "p20", "median", "p80", "p90"):
        row[f"{prefix}_{key}"] = stats[key]


def bootstrap_scaling_by_method(
    duration_sets: Dict[str, np.ndarray],
    *,
    max_distance: int,
    bootstrap_samples: int,
    seed: int,
    chunk_horizon: int,
    our_call_ms: float,
    vla_call_ms: float,
    llm_latency_sec: float,
) -> list[Dict[str, Any]]:
    method_order = [
        ("ours", "ours", "ours"),
        ("vla", "VLA", "vla"),
        ("llm_static", "VLA + LLM static", "llm_static"),
        ("llm_closed_loop", "VLA + LLM closed-loop", "llm_closed_loop"),
    ]
    rows: list[Dict[str, Any]] = []
    for method_idx, (method, label, source_key) in enumerate(method_order):
        durations = duration_sets[source_key]
        if durations.size == 0:
            continue
        rng = np.random.default_rng(int(seed) + 1009 * method_idx)
        for distance in range(1, int(max_distance) + 1):
            sampled = rng.choice(durations, size=(int(bootstrap_samples), int(distance)), replace=True)
            env_steps = np.sum(sampled, axis=1)
            if method == "ours":
                inference_calls = np.sum(np.ceil(sampled / float(chunk_horizon)), axis=1)
                action_model_sec = inference_calls * float(our_call_ms) / 1000.0
                llm_calls = np.zeros_like(env_steps)
                total_sec = action_model_sec
                call_unit = "action_chunk"
            elif method == "vla":
                inference_calls = env_steps
                action_model_sec = inference_calls * float(vla_call_ms) / 1000.0
                llm_calls = np.zeros_like(env_steps)
                total_sec = action_model_sec
                call_unit = "env_action"
            elif method == "llm_static":
                inference_calls = env_steps
                action_model_sec = inference_calls * float(vla_call_ms) / 1000.0
                llm_calls = np.ones_like(env_steps)
                total_sec = action_model_sec + float(llm_latency_sec)
                call_unit = "env_action_plus_one_llm_call"
            elif method == "llm_closed_loop":
                inference_calls = env_steps
                action_model_sec = inference_calls * float(vla_call_ms) / 1000.0
                llm_calls = np.full_like(env_steps, fill_value=float(distance))
                total_sec = action_model_sec + float(distance) * float(llm_latency_sec)
                call_unit = "env_action_plus_llm_call_per_subgoal"
            else:
                raise AssertionError(method)

            row: Dict[str, Any] = {
                "method": method,
                "method_label": label,
                "duration_source_method": source_key,
                "call_unit": call_unit,
                "automaton_distance": int(distance),
                "bootstrap_samples": int(bootstrap_samples),
                "chunk_horizon": int(chunk_horizon),
                "our_call_ms": float(our_call_ms),
                "vla_call_ms": float(vla_call_ms),
                "llm_latency_sec": float(llm_latency_sec),
                "band": "p20_p80",
                "inference_time_includes_env_runtime": False,
            }
            add_stats_columns(row, "env_steps", env_steps)
            add_stats_columns(row, "action_model_inference_calls", inference_calls)
            add_stats_columns(row, "action_model_inference_sec", action_model_sec)
            add_stats_columns(row, "llm_calls", llm_calls)
            add_stats_columns(row, "llm_inference_sec", llm_calls * float(llm_latency_sec))
            add_stats_columns(row, "total_inference_sec", total_sec)
            rows.append(row)
    return rows


def write_scaling_plot(out_dir: Path, rows: Sequence[Dict[str, Any]], *, task_id: str) -> Optional[Path]:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series = [
        ("ours", "ours", "#2563eb"),
        ("vla", "VLA", "#111827"),
        ("llm_static", "VLA + LLM static", "#16a34a"),
        ("llm_closed_loop", "VLA + LLM closed-loop", "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, label, color in series:
        method_rows = sorted(
            [row for row in rows if row.get("method") == method],
            key=lambda row: int(row["automaton_distance"]),
        )
        if not method_rows:
            continue
        x = np.asarray([row["automaton_distance"] for row in method_rows], dtype=np.float64)
        mean = np.asarray([row["total_inference_sec_mean"] for row in method_rows], dtype=np.float64)
        p20 = np.asarray([row["total_inference_sec_p20"] for row in method_rows], dtype=np.float64)
        p80 = np.asarray([row["total_inference_sec_p80"] for row in method_rows], dtype=np.float64)
        ax.plot(x, mean, label=label, color=color, linewidth=2.0)
        ax.fill_between(x, p20, p80, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("automaton distance / number of high-level subgoals")
    ax.set_ylabel("estimated inference time (s)")
    ax.set_title(f"Estimated inference-time scaling from {task_id} rollout subgoal durations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "inference_time_over_automaton_distance.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def write_env_step_plot(out_dir: Path, rows: Sequence[Dict[str, Any]], *, task_id: str) -> Optional[Path]:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series = [
        ("ours", "ours rollout durations", "#2563eb"),
        ("vla", "VLA rollout durations", "#111827"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for method, label, color in series:
        method_rows = sorted(
            [row for row in rows if row.get("method") == method],
            key=lambda row: int(row["automaton_distance"]),
        )
        if not method_rows:
            continue
        x = np.asarray([row["automaton_distance"] for row in method_rows], dtype=np.float64)
        mean = np.asarray([row["env_steps_mean"] for row in method_rows], dtype=np.float64)
        p10 = np.asarray([row["env_steps_p10"] for row in method_rows], dtype=np.float64)
        p90 = np.asarray([row["env_steps_p90"] for row in method_rows], dtype=np.float64)
        ax.plot(x, mean, color=color, label=label, linewidth=2.0)
        ax.fill_between(x, p10, p90, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("automaton distance / number of high-level subgoals")
    ax.set_ylabel("estimated environment steps")
    ax.set_title(f"Bootstrap environment-step estimate from {task_id} rollouts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "env_steps_over_automaton_distance.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    task_spec = COMPLEX_STL_SPECS[args.task_id]
    if args.n_candidates is None:
        args.n_candidates = int(task_spec.default_n_candidates)
    if args.scene_config is None:
        args.scene_config = task_spec.scene_config
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    run_name = args.name or f"inference_scaling_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = unique_run_dir(repo_path(args.output_root), run_name)
    out_dir.mkdir(parents=True, exist_ok=False)

    def resolve_task_dir_list(paths: Sequence[Path]) -> list[Path]:
        return [resolve_task_dir(path, task_id=args.task_id) for path in paths]

    shared_task_dirs = resolve_task_dir_list([*args.task_dir, *args.chained_task_dir])
    explicit_method_dirs = bool(
        args.ours_task_dir
        or args.vla_task_dir
        or args.llm_static_task_dir
        or args.llm_closed_task_dir
    )
    if not shared_task_dirs and not explicit_method_dirs:
        shared_task_dirs = discover_task_dirs(
            args.search_root,
            task_id=args.task_id,
            include_baselines=bool(args.include_baselines_in_step_stats),
        )
    if not shared_task_dirs and not explicit_method_dirs:
        raise FileNotFoundError(
            f"No {args.task_id} task_summary.json files found. Pass --task-dir or a method-specific task dir."
        )

    ours_task_dirs = resolve_task_dir_list(args.ours_task_dir) or shared_task_dirs
    vla_task_dirs = resolve_task_dir_list(args.vla_task_dir) or shared_task_dirs
    llm_static_task_dirs = resolve_task_dir_list(args.llm_static_task_dir) or vla_task_dirs
    llm_closed_task_dirs = resolve_task_dir_list(args.llm_closed_task_dir) or vla_task_dirs
    if not ours_task_dirs:
        raise FileNotFoundError("No duration source for ours. Pass --ours-task-dir or --task-dir.")
    if not vla_task_dirs:
        raise FileNotFoundError("No duration source for VLA. Pass --vla-task-dir or --task-dir.")

    duration_sources = {
        "ours": ours_task_dirs,
        "vla": vla_task_dirs,
        "llm_static": llm_static_task_dirs,
        "llm_closed_loop": llm_closed_task_dirs,
    }
    complete_only = not bool(args.include_incomplete_rollouts)
    duration_rows: list[Dict[str, Any]] = []
    duration_sets: Dict[str, np.ndarray] = {}
    duration_source_summaries: list[Dict[str, Any]] = []
    for method, task_dirs in duration_sources.items():
        rows, used_complete_only = method_duration_rows(method, task_dirs, complete_only=complete_only)
        if not rows:
            raise RuntimeError(f"No target-event durations found for {method} in {task_dirs}")
        duration_rows.extend(rows)
        durations = np.asarray([row["env_steps"] for row in rows], dtype=np.float64)
        duration_sets[method] = durations
        duration_source_summaries.append(
            {
                "duration_source_method": method,
                "task_dirs": ";".join(str(path) for path in task_dirs),
                "complete_rollouts_only": bool(used_complete_only),
                **summarize_values(durations),
            }
        )

    per_event_rows = []
    for method in sorted({str(row["duration_source_method"]) for row in duration_rows}):
        method_rows = [row for row in duration_rows if str(row["duration_source_method"]) == method]
        for event_idx in sorted({int(row["event_idx"]) for row in method_rows}):
            event_rows = [row for row in method_rows if int(row["event_idx"]) == event_idx]
            values = [row["env_steps"] for row in event_rows]
            stats = summarize_values(values)
            target_names = sorted({str(row.get("target_name")) for row in event_rows})
            per_event_rows.append(
                {
                    "duration_source_method": method,
                    "event_idx": int(event_idx),
                    "target_names": ",".join(target_names),
                    **stats,
                }
            )

    device = None
    timing_rows = []
    our_timing = None
    flower_timing = None
    if not args.skip_benchmark:
        device = resolve_device(args.device)
        our_timing = benchmark_our_method(args, device)
        timing_rows.append({k: v for k, v in our_timing.items() if k != "times_ms"})
        if not args.skip_flower_benchmark:
            flower_timing = benchmark_flower(args, device)
            timing_rows.append({k: v for k, v in flower_timing.items() if k != "times_ms"})

    if args.our_call_ms is not None:
        our_call_ms = float(args.our_call_ms)
    elif our_timing is not None:
        our_call_ms = float(our_timing["mean"])
    else:
        raise ValueError("Need --our-call-ms when --skip-benchmark is used.")

    if args.vla_call_ms is not None:
        vla_call_ms = float(args.vla_call_ms)
    elif flower_timing is not None:
        vla_call_ms = float(flower_timing["mean"])
    else:
        raise ValueError("Need --vla-call-ms when FLOWER benchmark is skipped.")

    chunk_horizon = int(
        our_timing.get("automaton_horizon", args.chunk_horizon) if our_timing is not None else args.chunk_horizon
    )
    if not any(row.get("method") == "ours_automaton_sample_rank_chunk" for row in timing_rows):
        timing_rows.append(
            {
                "method": "ours_automaton_sample_rank_chunk",
                "call_unit": "action_chunk",
                "source": "manual_override",
                "n_candidates": int(args.n_candidates),
                "automaton_horizon": int(chunk_horizon),
                "mean": float(our_call_ms),
            }
        )
    if not any(row.get("method") == "flower_vla_action" for row in timing_rows):
        timing_rows.append(
            {
                "method": "flower_vla_action",
                "call_unit": "env_action",
                "source": "manual_override",
                "mean": float(vla_call_ms),
            }
        )
    timing_rows.append(
        {
            "method": "llm_planner",
            "call_unit": "planning_request",
            "source": "assumption",
            "mean": 1000.0 * float(args.llm_latency_sec),
        }
    )
    scaling_rows = bootstrap_scaling_by_method(
        duration_sets,
        max_distance=args.max_distance,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        chunk_horizon=chunk_horizon,
        our_call_ms=our_call_ms,
        vla_call_ms=vla_call_ms,
        llm_latency_sec=float(args.llm_latency_sec),
    )

    write_json(
        out_dir / "run_args.json",
        json_ready({
            **vars(args),
            "output_dir": str(out_dir),
            "resolved_duration_sources": {
                method: [str(path) for path in task_dirs]
                for method, task_dirs in duration_sources.items()
            },
            "complete_rollouts_only_requested_for_step_stats": bool(complete_only),
        }),
    )
    write_csv(out_dir / "subgoal_step_distribution.csv", duration_rows)
    write_csv(out_dir / "duration_source_summary.csv", duration_source_summaries)
    write_csv(out_dir / "subgoal_step_distribution_by_event.csv", per_event_rows)
    write_csv(out_dir / "timing_measurements.csv", timing_rows)
    write_csv(out_dir / "scaling_estimates.csv", scaling_rows)
    write_json(
        out_dir / "summary.json",
        json_ready({
            "assumptions": {
                "x_axis": "automaton distance / number of high-level subgoals",
                "inference_only": True,
                "env_runtime_included": False,
                "env_step_distribution": f"bootstrap samples from observed {args.task_id} rollout subgoal durations",
                "ours_scaling": "ceil(subgoal_env_steps / automaton_horizon) inference calls per subgoal",
                "vla_scaling": "one FLOWER inference call per environment step",
                "llm_static": "one LLM planning call plus VLA action inference",
                "llm_closed_loop": "one LLM planning call per automaton step plus VLA action inference",
                "uncertainty_band": "p20-p80 bootstrap interval",
                "llm_latency_sec": float(args.llm_latency_sec),
            },
            "duration_source_summaries": duration_source_summaries,
            "timing": {
                "ours_call_ms": our_call_ms,
                "vla_call_ms": vla_call_ms,
                "automaton_horizon": chunk_horizon,
                "our_raw": our_timing,
                "flower_raw": flower_timing,
            },
            "outputs": {
                "subgoal_step_distribution_csv": str(out_dir / "subgoal_step_distribution.csv"),
                "duration_source_summary_csv": str(out_dir / "duration_source_summary.csv"),
                "subgoal_step_distribution_by_event_csv": str(out_dir / "subgoal_step_distribution_by_event.csv"),
                "timing_measurements_csv": str(out_dir / "timing_measurements.csv"),
                "scaling_estimates_csv": str(out_dir / "scaling_estimates.csv"),
            },
        }),
    )
    scaling_plot = write_scaling_plot(out_dir, scaling_rows, task_id=args.task_id)
    env_plot = write_env_step_plot(out_dir, scaling_rows, task_id=args.task_id)
    with (out_dir / "README.md").open("w") as f:
        f.write("# Inference Scaling Estimate\n\n")
        f.write("This folder contains a bootstrap estimate of inference time over automaton distance.\n")
        f.write("It uses inference-call timing only; environment rollout/runtime is not included.\n\n")
        for summary in duration_source_summaries:
            f.write(
                f"- {summary['duration_source_method']} observed subgoal duration mean: "
                f"{summary['mean']:.2f} env steps "
                f"(p10-p90: {summary['p10']:.2f}-{summary['p90']:.2f})\n"
            )
        f.write(f"- ours per chunk: {our_call_ms:.3f} ms, chunk horizon: {chunk_horizon}\n")
        f.write(f"- FLOWER VLA per action: {vla_call_ms:.3f} ms\n")
        f.write(f"- LLM latency assumption: {float(args.llm_latency_sec):.3f} s per planning call\n\n")
        f.write("Main files:\n")
        f.write("- `scaling_estimates.csv`\n")
        f.write("- `subgoal_step_distribution.csv`\n")
        f.write("- `duration_source_summary.csv`\n")
        f.write("- `timing_measurements.csv`\n")
        if scaling_plot is not None:
            f.write(f"- `{scaling_plot.name}`\n")
        if env_plot is not None:
            f.write(f"- `{env_plot.name}`\n")

    print("output:", out_dir)
    print("inference-only estimate: env rollout/runtime is not included")
    print("duration sources:", json.dumps(duration_source_summaries, indent=2))
    print(f"ours: {our_call_ms:.3f} ms/chunk, horizon={chunk_horizon}")
    print(f"FLOWER: {vla_call_ms:.3f} ms/action")
    print(f"LLM estimate: {float(args.llm_latency_sec):.3f} s/call")
    print("scaling:", out_dir / "scaling_estimates.csv")


if __name__ == "__main__":
    main()
