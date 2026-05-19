#!/usr/bin/env python3
"""Run FLOWER VLA on the DynaGuide-style CALVIN articulated-object tasks.

This mirrors `calvin_experiments/run_dynaguide_articulated_automaton.py` for
the eight articulated-object tasks, but replaces automaton-guided DP with the
language-conditioned FLOWER policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
FLOWER_ROOT = REPO_ROOT / "flower_vla_calvin"
DEFAULT_FLOWER_CHECKPOINT = REPO_ROOT / "outputs/calvin/baselines/flower/flower_calvin_d"
DEFAULT_ENV_CHECKPOINT = REPO_ROOT / "outputs/calvin/base_policy/calvin_D_base_dp/20260501015147/models/model_epoch_280.pth"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin_paper/baselines/flower/articulated"
DEFAULT_ANNOTATION_FILE = FLOWER_ROOT / "conf/annotations/new_playtable_validation.yaml"


for path in [
    REPO_ROOT,
    REPO_ROOT / "robomimic",
    REPO_ROOT / "calvin" / "calvin_env",
    REPO_ROOT / "calvin_experiments",
    Path(__file__).resolve().parent,
    FLOWER_ROOT,
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


import robomimic.envs  # noqa: F401
import robomimic.utils.file_utils as FileUtils

from calvin_experiments import calvin_rollout_utils as CRU
from calvin_experiments.label_calvin_world_model import LABEL_NAMES, LABEL_THRESHOLDS, label_scene_states_for_names
from calvin_experiments.run_dynaguide_articulated_automaton import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_RESET_ROBOT_Y_MIN,
    DEFAULT_RESET_SWITCH_CLEARANCE,
    DEFAULT_SETTLE_GRIPPER,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_VISUALIZATION_CONFIG,
    RELEVANT_BEHAVIORS,
    dynaguide_scene_from_base,
    filter_reset_poses,
    format_onehot,
    guidance_label_name_for_task,
    idle_action_from_checkpoint,
    load_json,
    load_reset_poses,
    repo_path,
    resolve_existing_path,
    settle_metrics,
    write_json,
)
from flower_our_env_rollout import FlowerPolicyAdapter, load_flower_model, resolve_device


CALVIN_TASK_FOR_DYNAGUIDE_TASK = {
    "button_on": "turn_on_led",
    "button_off": "turn_off_led",
    "switch_on": "turn_on_lightbulb",
    "switch_off": "turn_off_lightbulb",
    "drawer_open": "open_drawer",
    "drawer_close": "close_drawer",
    "door_left": "move_slider_left",
    "door_right": "move_slider_right",
}

FALLBACK_INSTRUCTIONS = {
    "turn_on_led": "press the button to turn on the led light",
    "turn_off_led": "press the button to turn off the led light",
    "turn_on_lightbulb": "use the switch to turn on the light bulb",
    "turn_off_lightbulb": "use the switch to turn off the light bulb",
    "open_drawer": "pull the handle to open the drawer",
    "close_drawer": "push the handle to close the drawer",
    "move_slider_left": "push the sliding door to the left side",
    "move_slider_right": "push the sliding door to the right side",
}


def parse_simple_annotation_yaml(path: Path) -> Dict[str, list[str]]:
    """Parse CALVIN's simple annotation yaml without requiring PyYAML."""

    if not path.exists():
        return {}
    text = path.read_text()
    annotations: Dict[str, list[str]] = {}
    key_pattern = re.compile(r"^([A-Za-z0-9_]+):\s*(.*)$")
    current_key: Optional[str] = None
    current_value: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = key_pattern.match(line)
        if match and not raw_line.startswith((" ", "\t")):
            if current_key is not None:
                annotations[current_key] = re.findall(r'"([^"]+)"|\'([^\']+)\'', "\n".join(current_value))
                annotations[current_key] = [a or b for a, b in annotations[current_key]]
            current_key = match.group(1)
            current_value = [match.group(2)]
        elif current_key is not None:
            current_value.append(line)
    if current_key is not None:
        annotations[current_key] = re.findall(r'"([^"]+)"|\'([^\']+)\'', "\n".join(current_value))
        annotations[current_key] = [a or b for a, b in annotations[current_key]]
    return annotations


def instruction_for_task(task_name: str, annotations: Dict[str, list[str]]) -> tuple[str, str]:
    calvin_task = CALVIN_TASK_FOR_DYNAGUIDE_TASK[task_name]
    candidates = annotations.get(calvin_task) or []
    if candidates:
        return candidates[0], calvin_task
    return FALLBACK_INSTRUCTIONS[calvin_task], calvin_task


def labels_for_scene(scene: Sequence[float]) -> np.ndarray:
    return label_scene_states_for_names(
        np.asarray(scene, dtype=np.float32)[None, :],
        LABEL_NAMES,
        LABEL_THRESHOLDS,
    )[0].astype(np.float32)


def save_trace_without_video(rollout: Dict[str, Any], output_dir: Path, rollout_tag: str) -> None:
    rollout_dir = output_dir / rollout_tag
    rollout_dir.mkdir(parents=True, exist_ok=True)
    trace_path = rollout_dir / "rollout_trace.npz"
    scene_snapshot_path = rollout_dir / "scene_snapshot.json"
    trace = {
        "actions": np.asarray(rollout["actions"], dtype=np.float32),
        "rewards": np.asarray(rollout["rewards"], dtype=np.float32),
        "dones": np.asarray(rollout["dones"], dtype=bool),
        "scene_states": np.asarray(rollout["scene_states"], dtype=np.float32),
        "robot_states": np.asarray(rollout["robot_states"], dtype=np.float32),
        "eef_xy": np.asarray(rollout["eef_xy"], dtype=np.float32),
        "detected_behavior": np.asarray(rollout["behavior"]),
        "detected_behavior_step": np.asarray(rollout["behavior_step"], dtype=np.int32),
        "termination_step": np.asarray(rollout["termination_step"], dtype=np.int32),
        "termination_reason": np.asarray(rollout["termination_reason"]),
        "rollout_seed": np.asarray(-1 if rollout["seed"] is None else rollout["seed"], dtype=np.int32),
        "scene_config": np.asarray(rollout["scene_config"]),
        "initial_label": np.asarray(rollout["initial_label"], dtype=np.int32),
        "final_label": np.asarray(rollout["final_label"], dtype=np.int32),
        "pre_settle_label": np.asarray(rollout["pre_settle_label"], dtype=np.int32),
        "settle_scene_states": np.asarray(rollout["settle_scene_states"], dtype=np.float32),
        "settle_robot_states": np.asarray(rollout["settle_robot_states"], dtype=np.float32),
        "settle_action": np.asarray(rollout["settle_action"], dtype=np.float32),
    }
    np.savez_compressed(trace_path, **trace)
    CRU.save_scene_snapshot(rollout["scene_snapshot"], scene_snapshot_path)
    rollout["video"] = None
    rollout["trace"] = trace_path
    rollout["scene_snapshot_path"] = scene_snapshot_path
    rollout["rollout_dir"] = rollout_dir


def rollout_flower_once(
    *,
    seed: int,
    env_ckpt_dict: Dict[str, Any],
    flower_policy: FlowerPolicyAdapter,
    task_name: str,
    instruction: str,
    calvin_task_name: str,
    task_config: Dict[str, Any],
    task_config_path: Path,
    output_dir: Path,
    rollout_tag: str,
    horizon: int,
    video_cfg: Dict[str, Any],
    save_video: bool,
    reset_robot_y_min: Optional[float],
    reset_robot_y_max: Optional[float],
    reset_switch_clearance: Optional[float],
    settle_steps: int,
    settle_gripper: float,
    fps: int,
) -> Dict[str, Any]:
    CRU.seed_everything(seed)
    env, base_env_state = CRU.load_fresh_env_from_checkpoint(env_ckpt_dict, seed=int(seed), suppress_output=True)
    try:
        target_label_name = guidance_label_name_for_task(task_name)
        target_label_idx = LABEL_NAMES.index(target_label_name)
        reset_poses, reset_pose_filter = filter_reset_poses(
            load_reset_poses(task_config, task_config_path),
            robot_y_min=reset_robot_y_min,
            robot_y_max=reset_robot_y_max,
            switch_clearance=reset_switch_clearance,
        )
        scene, binaries = dynaguide_scene_from_base(base_env_state["scene"], task_config.get("env_setup", {}))
        robot = np.asarray(base_env_state["robot"], dtype=np.float32).copy()
        if reset_poses:
            robot = np.asarray(random.choice(reset_poses), dtype=np.float32).copy()

        obs = CRU.reset_env_to_scene_robot(env, scene, robot)
        pre_settle_state = env.get_state()
        pre_settle_scene = np.asarray(pre_settle_state["scene"], dtype=np.float32).copy()
        pre_settle_robot = np.asarray(pre_settle_state["robot"], dtype=np.float32).copy()
        pre_settle_label = labels_for_scene(pre_settle_scene)

        settle_action = idle_action_from_checkpoint(env_ckpt_dict, gripper=settle_gripper)
        settle_scene_states = [pre_settle_scene.copy()]
        settle_robot_states = [pre_settle_robot.copy()]
        settle_rewards, settle_dones = [], []
        for _ in range(max(0, int(settle_steps))):
            obs, reward, done, _ = env.step(settle_action)
            settle_rewards.append(float(reward))
            settle_dones.append(bool(done))
            state_after_settle = env.get_state()
            settle_scene_states.append(np.asarray(state_after_settle["scene"], dtype=np.float32).copy())
            settle_robot_states.append(np.asarray(state_after_settle["robot"], dtype=np.float32).copy())

        flower_policy.reset(instruction)
        scene_snapshot = CRU.capture_scene_snapshot(env)
        frames = [CRU.render_visual_camera(env, video_cfg)] if save_video else []

        start_state = env.get_state()
        start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
        label0 = labels_for_scene(start_scene)
        target_label_initial = bool(label0[target_label_idx] >= 0.5)

        actions, rewards, dones = [], [], []
        scene_states = [start_scene.copy()]
        robot_states = [np.asarray(start_state["robot"], dtype=np.float32).copy()]
        eef_xy = [robot_states[-1][:2].copy()]
        detected_behavior = "none"
        detected_step = -1
        termination_reason = "horizon"
        target_label_flipped = False
        target_label_flip_step = -1
        ignored_behavior_events = []
        last_ignored_behavior = "other"
        env_done_step = -1
        total_reward = 0.0

        for step in range(int(horizon)):
            action = flower_policy(obs)
            next_obs, reward, done, _ = env.step(action.copy())
            total_reward += float(reward)

            state_now = env.get_state()
            scene_now = np.asarray(state_now["scene"], dtype=np.float32).copy()
            robot_now = np.asarray(state_now["robot"], dtype=np.float32).copy()

            actions.append(np.asarray(action, dtype=np.float32).copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            scene_states.append(scene_now.copy())
            robot_states.append(robot_now.copy())
            eef_xy.append(robot_now[:2].copy())

            if save_video:
                frames.append(CRU.render_visual_camera(env, video_cfg))

            behavior_now = CRU.classify_behavior(start_scene, scene_now, robot_now[:3], binaries, for_display=False)
            if behavior_now == task_name:
                if detected_step < 0:
                    detected_behavior = behavior_now
                    detected_step = int(step + 1)
            elif behavior_now != "other" and behavior_now != last_ignored_behavior:
                ignored_behavior_events.append({"behavior": behavior_now, "step": int(step + 1)})
                last_ignored_behavior = behavior_now

            if done and env_done_step < 0:
                env_done_step = int(step + 1)

            label_now = labels_for_scene(scene_now)
            target_label_current = bool(label_now[target_label_idx] >= 0.5)
            if (not target_label_initial) and target_label_current:
                target_label_flipped = True
                target_label_flip_step = int(step + 1)
                termination_reason = "target_label_flip"
                break
            obs = next_obs
        else:
            step = int(horizon) - 1

        labelf = labels_for_scene(scene_states[-1])
        target_label_final = bool(labelf[target_label_idx] >= 0.5)
        rollout = {
            "task": task_name,
            "scene_config": task_config.get("name", task_name),
            "seed": int(seed),
            "policy": "flower_vla",
            "instruction": instruction,
            "calvin_task_name": calvin_task_name,
            "target_behavior_name": task_name,
            "target_label_name": target_label_name,
            "target_label_idx": int(target_label_idx),
            "target_label_initial": bool(target_label_initial),
            "target_label_final": bool(target_label_final),
            "label_flipped": bool(target_label_flipped),
            "label_flip_step": int(target_label_flip_step),
            "behavior": detected_behavior,
            "behavior_step": int(detected_step),
            "behavior_success": bool(detected_behavior == task_name),
            "success": bool(target_label_flipped),
            "termination_step": int(step + 1),
            "termination_reason": termination_reason,
            "env_done_step": int(env_done_step),
            "return": float(total_reward),
            "initial_label": label0.astype(int).tolist(),
            "final_label": labelf.astype(int).tolist(),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "scene_states": np.asarray(scene_states, dtype=np.float32),
            "robot_states": np.asarray(robot_states, dtype=np.float32),
            "eef_xy": np.asarray(eef_xy, dtype=np.float32),
            "ignored_behavior_events": ignored_behavior_events,
            "scene_snapshot": scene_snapshot,
            "reset_env_setup": dict(task_config.get("env_setup", {})),
            "reset_robot_from_pose_file": bool(reset_poses),
            "reset_pose_filter": reset_pose_filter,
            "settle_steps": int(max(0, int(settle_steps))),
            "settle_action": settle_action.astype(float).tolist(),
            "pre_settle_label": pre_settle_label.astype(int).tolist(),
            "settle_scene_states": settle_scene_states,
            "settle_robot_states": settle_robot_states,
            "settle_rewards": settle_rewards,
            "settle_dones": settle_dones,
            "settle_metrics": settle_metrics(settle_scene_states),
        }
        if save_video:
            CRU.save_rollout_artifacts(rollout, frames, output_dir, rollout_tag, video_cfg, fps=fps)
        else:
            save_trace_without_video(rollout, output_dir, rollout_tag)

        rollout_dir = Path(rollout["rollout_dir"])
        label_status = f"{target_label_name} flip @ {target_label_flip_step}" if target_label_flipped else f"no {target_label_name} flip"
        CRU.plot_rollout_xy(
            [rollout],
            rollout["scene_snapshot"],
            f"FLOWER {task_name} seed {seed} -> {label_status}",
            save_path=rollout_dir / "rollout_xy.png",
            display_inline=False,
        )
        write_json(
            rollout_dir / "rollout_summary.json",
            {
                "task": task_name,
                "seed": int(seed),
                "policy": "flower_vla",
                "instruction": instruction,
                "calvin_task_name": calvin_task_name,
                "target_label_name": target_label_name,
                "target_label_idx": int(target_label_idx),
                "target_label_initial": bool(target_label_initial),
                "target_label_final": bool(target_label_final),
                "label_flipped": bool(target_label_flipped),
                "label_flip_step": int(target_label_flip_step),
                "success": bool(target_label_flipped),
                "behavior": detected_behavior,
                "behavior_step": int(detected_step),
                "behavior_success": bool(detected_behavior == task_name),
                "termination_step": int(step + 1),
                "termination_reason": termination_reason,
                "env_done_step": int(env_done_step),
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": None if rollout.get("video") is None else str(rollout.get("video")),
                "trace": str(rollout.get("trace")),
                "topdown_plot": str(rollout_dir / "rollout_xy.png"),
                "ignored_behavior_events": ignored_behavior_events,
                "reset_pose_filter": reset_pose_filter,
                "settle_steps": rollout["settle_steps"],
                "settle_action": rollout["settle_action"],
                "pre_settle_label": rollout["pre_settle_label"],
                "settle_metrics": rollout["settle_metrics"],
            },
        )
        return rollout
    finally:
        CRU.close_env_quietly(env)


def task_summary(task_name: str, instruction: str, rollouts: Sequence[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    behavior_counts = Counter(rollout["behavior"] for rollout in rollouts)
    ignored_behavior_counts = Counter(
        event["behavior"]
        for rollout in rollouts
        for event in rollout.get("ignored_behavior_events", [])
    )
    successes = sum(1 for rollout in rollouts if rollout["success"])
    behavior_successes = sum(1 for rollout in rollouts if rollout.get("behavior_success", rollout["behavior"] == task_name))
    return {
        "task": task_name,
        "instruction": instruction,
        "n_rollouts": len(rollouts),
        "horizon": int(horizon),
        "evaluation": "target_label_flip",
        "success_count": int(successes),
        "success_rate": float(successes / len(rollouts)) if rollouts else 0.0,
        "behavior_success_count": int(behavior_successes),
        "behavior_counts": dict(behavior_counts),
        "ignored_behavior_counts": dict(ignored_behavior_counts),
        "avg_termination_step": float(np.mean([rollout["termination_step"] for rollout in rollouts])) if rollouts else 0.0,
        "rollouts": [
            {
                "seed": rollout["seed"],
                "success": bool(rollout["success"]),
                "label_flipped": bool(rollout["label_flipped"]),
                "label_flip_step": rollout["label_flip_step"],
                "target_label_name": rollout["target_label_name"],
                "target_label_initial": rollout["target_label_initial"],
                "target_label_final": rollout["target_label_final"],
                "behavior": rollout["behavior"],
                "behavior_step": rollout["behavior_step"],
                "behavior_success": bool(rollout["behavior_success"]),
                "termination_step": rollout["termination_step"],
                "termination_reason": rollout["termination_reason"],
                "env_done_step": rollout.get("env_done_step", -1),
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": None if rollout.get("video") is None else str(rollout.get("video")),
                "trace": str(rollout.get("trace")),
                "topdown_plot": str(Path(rollout.get("rollout_dir", "")) / "rollout_xy.png"),
                "ignored_behavior_events": rollout.get("ignored_behavior_events", []),
                "settle_steps": rollout.get("settle_steps", 0),
                "settle_metrics": rollout.get("settle_metrics", {}),
            }
            for rollout in rollouts
        ],
    }


def write_summary_tables(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    rows = [
        {
            "task": item["task"],
            "instruction": item["instruction"],
            "evaluation": item["evaluation"],
            "success_rate": f"{item['success_rate']:.4f}",
            "success_count": item["success_count"],
            "behavior_success_count": item["behavior_success_count"],
            "n_rollouts": item["n_rollouts"],
            "avg_termination_step": f"{item['avg_termination_step']:.2f}",
            "behavior_counts": json.dumps(item["behavior_counts"], sort_keys=True),
            "ignored_behavior_counts": json.dumps(item["ignored_behavior_counts"], sort_keys=True),
        }
        for item in summaries
    ]
    csv_path = run_dir / "summary_table.csv"
    md_path = run_dir / "summary_table.md"
    fieldnames = [
        "task",
        "instruction",
        "evaluation",
        "success_rate",
        "success_count",
        "behavior_success_count",
        "n_rollouts",
        "avg_termination_step",
        "behavior_counts",
        "ignored_behavior_counts",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w") as f:
        f.write("| task | instruction | evaluation | success_rate | success_count | behavior_success_count | n_rollouts | avg_termination_step | behavior_counts | ignored_behavior_counts |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['task']} | {row['instruction']} | {row['evaluation']} | "
                f"{row['success_rate']} | {row['success_count']} | {row['behavior_success_count']} | "
                f"{row['n_rollouts']} | {row['avg_termination_step']} | "
                f"`{row['behavior_counts']}` | `{row['ignored_behavior_counts']}` |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=FLOWER_ROOT)
    parser.add_argument("--flower-checkpoint", type=Path, default=DEFAULT_FLOWER_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=DEFAULT_ENV_CHECKPOINT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--visualization-config", type=Path, default=DEFAULT_VISUALIZATION_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--tasks", "--run-for", nargs="*", default=None, choices=RELEVANT_BEHAVIORS)
    parser.add_argument("--n-rollouts", "--num-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=DEFAULT_SETTLE_GRIPPER)
    parser.add_argument("--reset-robot-y-min", type=float, default=DEFAULT_RESET_ROBOT_Y_MIN)
    parser.add_argument("--reset-robot-y-max", type=float, default=None)
    parser.add_argument("--reset-switch-clearance", type=float, default=DEFAULT_RESET_SWITCH_CLEARANCE)
    parser.add_argument("--disable-reset-pose-filter", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face downloads/checks instead of cache-only mode.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tasks = list(args.tasks) if args.tasks else list(RELEVANT_BEHAVIORS)
    reset_robot_y_min = None if args.disable_reset_pose_filter else args.reset_robot_y_min
    reset_robot_y_max = None if args.disable_reset_pose_filter else args.reset_robot_y_max
    reset_switch_clearance = None if args.disable_reset_pose_filter else args.reset_switch_clearance
    config_dir = resolve_existing_path(repo_path(args.config_dir))
    visualization_config = resolve_existing_path(repo_path(args.visualization_config))
    flower_checkpoint = resolve_existing_path(args.flower_checkpoint)
    env_checkpoint = resolve_existing_path(args.env_checkpoint)
    annotation_file = Path(args.annotation_file).expanduser()
    if not annotation_file.is_absolute():
        annotation_file = repo_path(annotation_file)
    task_config_paths = {task: resolve_existing_path(config_dir / f"{task}.json") for task in tasks}
    annotations = parse_simple_annotation_yaml(annotation_file)
    instructions = {task: instruction_for_task(task, annotations) for task in tasks}
    run_name = args.name or f"flower_articulated_rollouts{args.n_rollouts}_h{args.horizon}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = run_dir / run_name

    planned = {
        "policy": "flower_vla",
        "flower_root": str(Path(args.flower_root).expanduser()),
        "flower_checkpoint": str(flower_checkpoint),
        "env_checkpoint": str(env_checkpoint),
        "config_dir": str(config_dir),
        "annotation_file": str(annotation_file),
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "tasks": list(tasks),
        "task_configs": {task: str(path) for task, path in task_config_paths.items()},
        "instructions": {
            task: {"instruction": instruction, "calvin_task_name": calvin_task}
            for task, (instruction, calvin_task) in instructions.items()
        },
        "n_rollouts": int(args.n_rollouts),
        "horizon": int(args.horizon),
        "seed_start": int(args.seed_start),
        "evaluation": "target_label_flip",
        "stop_condition": "target_label_flip_or_horizon",
        "settle_steps": int(args.settle_steps),
        "settle_gripper": float(args.settle_gripper),
        "reset_robot_y_min": reset_robot_y_min,
        "reset_robot_y_max": reset_robot_y_max,
        "reset_switch_clearance": reset_switch_clearance,
        "disable_reset_pose_filter": bool(args.disable_reset_pose_filter),
        "save_video": not args.no_video,
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return

    video_cfg = load_json(visualization_config)
    device = resolve_device(args.device)
    env_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(env_checkpoint))
    flower_model = load_flower_model(flower_checkpoint, device)
    flower_policy = FlowerPolicyAdapter(flower_model, device=device)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_args.json", planned)
    print("device:", device)
    print("flower checkpoint:", flower_checkpoint)
    print("env checkpoint:", env_checkpoint)
    print("output:", run_dir)

    all_task_summaries = []
    for task in tasks:
        instruction, calvin_task = instructions[task]
        task_config_path = task_config_paths[task]
        task_config = load_json(task_config_path)
        task_dir = run_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "task_config_resolved.json", task_config)

        print(f"\nTask {task}: instruction={instruction!r} config={task_config_path}")
        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            rollout = rollout_flower_once(
                seed=seed,
                env_ckpt_dict=env_ckpt_dict,
                flower_policy=flower_policy,
                task_name=task,
                instruction=instruction,
                calvin_task_name=calvin_task,
                task_config=task_config,
                task_config_path=task_config_path,
                output_dir=task_dir,
                rollout_tag=tag,
                horizon=int(args.horizon),
                video_cfg=video_cfg,
                save_video=not args.no_video,
                reset_robot_y_min=reset_robot_y_min,
                reset_robot_y_max=reset_robot_y_max,
                reset_switch_clearance=reset_switch_clearance,
                settle_steps=int(args.settle_steps),
                settle_gripper=float(args.settle_gripper),
                fps=int(args.fps),
            )
            rollouts.append(rollout)
            print(
                f"  seed {seed:03d}: success={rollout['success']} "
                f"label_flip={rollout['label_flip_step']:>3}, "
                f"behavior={rollout['behavior']:>12} @ {rollout['behavior_step']:>3}, "
                f"steps={rollout['termination_step']:>3}, final={format_onehot(rollout['final_label'])}"
            )

        summary = task_summary(task, instruction, rollouts, args.horizon)
        write_json(task_dir / "task_summary.json", summary)
        CRU.plot_rollout_xy(
            rollouts,
            rollouts[0]["scene_snapshot"],
            f"FLOWER {task} | success {summary['success_count']}/{summary['n_rollouts']}",
            save_path=task_dir / "task_rollouts_xy.png",
            display_inline=False,
        )
        all_task_summaries.append(summary)

    write_json(
        run_dir / "summary.json",
        {
            "run": planned,
            "tasks": all_task_summaries,
        },
    )
    write_summary_tables(run_dir, all_task_summaries)
    print("\nSummary:", run_dir / "summary_table.md")


if __name__ == "__main__":
    main()
