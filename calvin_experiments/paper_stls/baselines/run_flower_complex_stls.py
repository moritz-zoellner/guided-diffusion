#!/usr/bin/env python3
"""Run FLOWER VLA on the complex CALVIN paper-STL tasks.

This mirrors `calvin_experiments/run_complex_stl_automaton.py`, but executes a
single language-conditioned FLOWER policy command for each STL task.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/baselines/flower"

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
from calvin_experiments.run_complex_stl_automaton import (
    COMPLEX_STL_SPECS,
    DEFAULT_RESET_POSE_FILES,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_VISUALIZATION_CONFIG,
    GripperOpenSpec,
    SafetyBox,
    TASK_ORDER,
    evaluate_safety,
    evaluate_subgoals,
    load_reset_pose_pool,
    make_fixed_scene_robot,
    resolve_reset_pose_paths,
    save_rollout_diagnostics,
    save_trace_without_video,
    task_summary,
    unique_run_dir,
    write_summary_tables,
)
from calvin_experiments.run_dynaguide_articulated_automaton import (
    DEFAULT_RESET_ROBOT_Y_MIN,
    DEFAULT_RESET_SWITCH_CLEARANCE,
    DEFAULT_SETTLE_GRIPPER,
    DEFAULT_SETTLE_STEPS,
    filter_reset_poses,
    format_onehot,
    idle_action_from_checkpoint,
    load_json,
    repo_path,
    resolve_existing_path,
    settle_metrics,
    write_json,
)
from flower_our_env_rollout import FlowerPolicyAdapter, load_flower_model, resolve_device


def labels_for_scene(scene: Sequence[float]) -> np.ndarray:
    return label_scene_states_for_names(
        np.asarray(scene, dtype=np.float32)[None, :],
        LABEL_NAMES,
        LABEL_THRESHOLDS,
    )[0].astype(np.float32)


class SpecMonitor:
    def __init__(self, spec):
        self.spec = spec
        self.events: list[Dict[str, Any]] = []
        self.violations: list[Dict[str, Any]] = []
        self.done = False
        self.pos = 0
        self.stage_pos = 0
        self.achieved = set()
        self.stage_achieved = [set() for _ in spec.stage_target_names]
        self.violation_keys = set()

    def idx(self, name: str) -> int:
        if name not in LABEL_NAMES:
            raise ValueError(f"Unknown label {name}; labels={LABEL_NAMES}")
        return LABEL_NAMES.index(name)

    def sync(self, label: np.ndarray, step: int) -> bool:
        if self.spec.mode == "or":
            return self._sync_or(label, step)
        if self.spec.mode == "and":
            return self._sync_and(label, step)
        if self.spec.mode == "chain":
            return self._sync_chain(label, step)
        if self.spec.mode == "ordered_stage":
            return self._sync_ordered_stage(label, step)
        if self.spec.mode == "target":
            return self._sync_target(label, step)
        raise ValueError(f"Unsupported mode: {self.spec.mode}")

    def _sync_or(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        for name in self.spec.target_names:
            idx = self.idx(name)
            if float(label[idx]) > 0.5:
                self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
                self.done = True
                return True
        return False

    def _sync_and(self, label: np.ndarray, step: int) -> bool:
        advanced = False
        for name in self.spec.target_names:
            idx = self.idx(name)
            if idx not in self.achieved and float(label[idx]) > 0.5:
                self.achieved.add(idx)
                self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
                advanced = True
        self.done = len(self.achieved) == len(self.spec.target_names)
        return advanced

    def _sync_chain(self, label: np.ndarray, step: int) -> bool:
        advanced = False
        while self.pos < len(self.spec.target_names):
            name = self.spec.target_names[self.pos]
            idx = self.idx(name)
            if float(label[idx]) <= 0.5:
                break
            self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
            self.pos += 1
            advanced = True
        self.done = self.pos >= len(self.spec.target_names)
        return advanced

    def _record_future_stage_violations(self, label: np.ndarray, step: int) -> None:
        for future_stage_idx in range(max(1, self.stage_pos + 1), len(self.spec.stage_target_names)):
            for name in self.spec.stage_target_names[future_stage_idx]:
                idx = self.idx(name)
                key = (future_stage_idx, idx)
                if key not in self.violation_keys and float(label[idx]) > 0.5:
                    self.violation_keys.add(key)
                    self.violations.append(
                        {
                            "step": int(step),
                            "stage_idx": int(future_stage_idx),
                            "target_idx": int(idx),
                            "target_name": name,
                            "message": "future-stage target became true before prior stages completed",
                        }
                    )

    def _sync_ordered_stage(self, label: np.ndarray, step: int) -> bool:
        advanced = False
        if self.stage_pos < len(self.spec.stage_target_names):
            self._record_future_stage_violations(label, step)
        while self.stage_pos < len(self.spec.stage_target_names):
            stage = self.spec.stage_target_names[self.stage_pos]
            for name in stage:
                idx = self.idx(name)
                if idx not in self.stage_achieved[self.stage_pos] and float(label[idx]) > 0.5:
                    self.stage_achieved[self.stage_pos].add(idx)
                    self.events.append(
                        {
                            "step": int(step),
                            "stage_idx": int(self.stage_pos),
                            "target_idx": int(idx),
                            "target_name": name,
                        }
                    )
                    advanced = True
            if len(self.stage_achieved[self.stage_pos]) == len(stage):
                self.stage_pos += 1
                advanced = True
                continue
            break
        self.done = self.stage_pos >= len(self.spec.stage_target_names)
        return advanced

    def _sync_target(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        name = self.spec.target_names[0]
        idx = self.idx(name)
        if float(label[idx]) > 0.5:
            self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
            self.done = True
            return True
        return False


def rollout_flower_once(
    *,
    seed: int,
    env_ckpt_dict: Dict[str, Any],
    flower_policy: FlowerPolicyAdapter,
    spec,
    instruction: str,
    scene_config_path: Path,
    reset_poses: Sequence[np.ndarray],
    reset_pose_filter: Dict[str, Any],
    fixed_reset_pose_index: Optional[int],
    output_dir: Path,
    rollout_tag: str,
    horizon: int,
    video_cfg: Dict[str, Any],
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
    save_video: bool,
    settle_steps: int,
    settle_gripper: float,
    stop_when_complete: bool,
    stop_on_env_done: bool,
    fps: int,
) -> Dict[str, Any]:
    CRU.seed_everything(seed)
    env, base_env_state = CRU.load_fresh_env_from_checkpoint(env_ckpt_dict, seed=int(seed), suppress_output=True)
    try:
        fixed_scene, fixed_robot, scene_cfg, robot_from_pose_file = make_fixed_scene_robot(
            base_env_state, scene_config_path, reset_poses, fixed_reset_pose_index
        )
        obs = CRU.reset_env_to_scene_robot(env, fixed_scene, fixed_robot)
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
            settled_state = env.get_state()
            settle_scene_states.append(np.asarray(settled_state["scene"], dtype=np.float32).copy())
            settle_robot_states.append(np.asarray(settled_state["robot"], dtype=np.float32).copy())

        flower_policy.reset(instruction)
        monitor = SpecMonitor(spec)
        scene_snapshot = CRU.capture_scene_snapshot(env)
        frames = [CRU.render_visual_camera(env, video_cfg)] if save_video else []

        start_state = env.get_state()
        start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
        binaries = CRU.articulated_binaries_from_start_state(start_scene)
        label0 = labels_for_scene(start_scene)
        monitor.sync(label0, 0)

        actions, rewards, dones, records = [], [], [], []
        scene_states = [start_scene.copy()]
        robot_states = [np.asarray(start_state["robot"], dtype=np.float32).copy()]
        labels_over_time = [label0.astype(int).copy()]
        eef_xy = [robot_states[-1][:2].copy()]
        first_behavior, first_behavior_step = "none", -1
        behavior_events = []
        last_behavior = "other"
        termination_reason = "horizon"
        env_done_step = -1
        total_reward = 0.0
        last_step = -1

        for step in range(int(horizon)):
            last_step = step
            action = np.asarray(flower_policy(obs), dtype=np.float32).copy()
            next_obs, reward, done, _ = env.step(action.copy())
            total_reward += float(reward)

            state_now = env.get_state()
            scene_now = np.asarray(state_now["scene"], dtype=np.float32).copy()
            robot_now = np.asarray(state_now["robot"], dtype=np.float32).copy()
            label_now = labels_for_scene(scene_now)

            actions.append(action.copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            scene_states.append(scene_now.copy())
            robot_states.append(robot_now.copy())
            labels_over_time.append(label_now.astype(int).copy())
            eef_xy.append(robot_now[:2].copy())

            if save_video:
                frames.append(CRU.render_visual_camera(env, video_cfg))

            behavior_now = CRU.classify_behavior(start_scene, scene_now, robot_now[:3], binaries, for_display=False)
            if behavior_now != "other" and behavior_now != last_behavior:
                behavior_events.append({"behavior": behavior_now, "step": int(step + 1)})
                last_behavior = behavior_now
                if first_behavior_step < 0:
                    first_behavior = behavior_now
                    first_behavior_step = int(step + 1)

            events_before = len(monitor.events)
            advanced = monitor.sync(label_now, step + 1)
            records.append(
                {
                    "t": int(step + 1),
                    "mode": spec.mode,
                    "current_label": label_now.astype(int).tolist(),
                    "advanced": bool(advanced),
                    "new_events": monitor.events[events_before:],
                    "done": bool(monitor.done),
                    "violations": list(monitor.violations),
                }
            )
            if stop_when_complete and monitor.done:
                termination_reason = "task_complete"
                break

            if done and env_done_step < 0:
                env_done_step = int(step + 1)
                if stop_on_env_done:
                    termination_reason = "env_done"
                    break
            obs = next_obs

        target_events = list(monitor.events)
        order_violations = list(monitor.violations)
        completed_subgoals, total_subgoals, subgoal_rate = evaluate_subgoals(spec, target_events)
        robot_states_np = np.asarray(robot_states, dtype=np.float32)
        eef_xy_np = np.asarray(eef_xy, dtype=np.float32)
        safety_satisfied, safety_metrics, safety_distances = evaluate_safety(
            spec, robot_states_np, eef_xy_np, safety_box, gripper_spec
        )
        liveness_satisfied = bool(monitor.done)
        order_violation = bool(order_violations)
        stl_satisfied = bool(liveness_satisfied and (safety_satisfied is not False) and not order_violation)
        behavior = "stl_satisfied" if stl_satisfied else "liveness_satisfied" if liveness_satisfied else first_behavior
        behavior_step = int(target_events[-1]["step"] if target_events else first_behavior_step)

        rollout = {
            "task": spec.name,
            "formula": spec.formula,
            "policy": "flower_vla",
            "instruction": instruction,
            "scene_config": scene_cfg["name"],
            "seed": int(seed),
            "behavior": behavior,
            "first_behavior": first_behavior,
            "first_behavior_step": int(first_behavior_step),
            "behavior_events": behavior_events,
            "behavior_step": behavior_step,
            "termination_step": int(max(0, last_step + 1)),
            "termination_reason": termination_reason,
            "env_done_step": int(env_done_step),
            "return": float(total_reward),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "scene_states": np.asarray(scene_states, dtype=np.float32),
            "robot_states": robot_states_np,
            "eef_xy": eef_xy_np,
            "gripper_width": robot_states_np[:, 6].astype(np.float32),
            "labels_over_time": np.asarray(labels_over_time, dtype=np.int32),
            "initial_label": labels_over_time[0].astype(int).tolist(),
            "final_label": labels_over_time[-1].astype(int).tolist(),
            "records": records,
            "target_events": target_events,
            "order_violations": order_violations,
            "order_violation": order_violation,
            "liveness_satisfied": liveness_satisfied,
            "safety_satisfied": None if safety_satisfied is None else bool(safety_satisfied),
            "stl_satisfied": stl_satisfied,
            "success": stl_satisfied,
            "completed_subgoals": completed_subgoals,
            "total_subgoals": total_subgoals,
            "subgoal_completion_rate": subgoal_rate,
            "safety_kind": spec.safety_kind,
            "safety_metrics": safety_metrics,
            "safety_distances": safety_distances,
            "scene_snapshot": scene_snapshot,
            "reset_robot_from_pose_file": bool(robot_from_pose_file),
            "fixed_reset_pose_index": None if fixed_reset_pose_index is None else int(fixed_reset_pose_index),
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
        save_rollout_diagnostics(rollout)
        rollout_dir = Path(rollout["rollout_dir"])
        CRU.plot_rollout_xy(
            [rollout],
            rollout["scene_snapshot"],
            f"FLOWER {spec.name} seed {seed} | stl={stl_satisfied} subgoals={completed_subgoals}/{total_subgoals}",
            save_path=rollout_dir / "rollout_xy.png",
            display_inline=False,
        )
        write_json(rollout_dir / "records.json", {"records": records})
        write_json(
            rollout_dir / "rollout_summary.json",
            {
                "task": spec.name,
                "formula": spec.formula,
                "seed": int(seed),
                "policy": "flower_vla",
                "instruction": instruction,
                "liveness_satisfied": bool(liveness_satisfied),
                "safety_satisfied": rollout["safety_satisfied"],
                "stl_satisfied": bool(stl_satisfied),
                "success": bool(stl_satisfied),
                "subgoal_completion_rate": float(subgoal_rate),
                "completed_subgoals": int(completed_subgoals),
                "total_subgoals": int(total_subgoals),
                "target_events": target_events,
                "order_violation": order_violation,
                "order_violations": order_violations,
                "safety_kind": spec.safety_kind,
                "safety_metrics": safety_metrics,
                "behavior": behavior,
                "first_behavior": first_behavior,
                "first_behavior_step": int(first_behavior_step),
                "behavior_step": behavior_step,
                "termination_step": rollout["termination_step"],
                "termination_reason": termination_reason,
                "env_done_step": int(env_done_step),
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": None if rollout.get("video") is None else str(rollout["video"]),
                "trace": str(rollout["trace"]),
                "diagnostics": str(rollout["diagnostics"]),
                "topdown_plot": str(rollout_dir / "rollout_xy.png"),
                "reset_robot_from_pose_file": bool(robot_from_pose_file),
                "fixed_reset_pose_index": None if fixed_reset_pose_index is None else int(fixed_reset_pose_index),
                "reset_pose_filter": reset_pose_filter,
                "settle_steps": rollout["settle_steps"],
                "settle_action": rollout["settle_action"],
                "pre_settle_label": rollout["pre_settle_label"],
                "settle_metrics": rollout["settle_metrics"],
                "records": records,
            },
        )
        return rollout
    finally:
        CRU.close_env_quietly(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=FLOWER_ROOT)
    parser.add_argument("--flower-checkpoint", type=Path, default=DEFAULT_FLOWER_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=DEFAULT_ENV_CHECKPOINT)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--visualization-config", type=Path, default=DEFAULT_VISUALIZATION_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--tasks",
        "--run-for",
        nargs="*",
        default=None,
        choices=TASK_ORDER,
        help=(
            "Optional subset of paper STL tasks to run. Omit this flag, or pass it with no values, "
            f"to run all tasks: {', '.join(TASK_ORDER)}."
        ),
    )
    parser.add_argument("--n-rollouts", "--num-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reset-pose-files", type=Path, nargs="*", default=list(DEFAULT_RESET_POSE_FILES))
    parser.add_argument(
        "--fixed-reset-pose-index",
        type=int,
        default=0,
        help="Use one deterministic robot reset pose from the filtered pose pool. Default keeps every rollout at the same robot start.",
    )
    parser.add_argument(
        "--sample-reset-poses",
        action="store_true",
        help="Restore the older behavior: sample a reset pose from the filtered pool for each rollout seed.",
    )
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=DEFAULT_SETTLE_GRIPPER)
    parser.add_argument("--reset-robot-y-min", type=float, default=DEFAULT_RESET_ROBOT_Y_MIN)
    parser.add_argument("--reset-robot-y-max", type=float, default=None)
    parser.add_argument("--reset-switch-clearance", type=float, default=DEFAULT_RESET_SWITCH_CLEARANCE)
    parser.add_argument("--disable-reset-pose-filter", action="store_true")
    parser.add_argument("--safety-box", type=float, nargs=4, default=None, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--gripper-min-width", type=float, default=0.06)
    parser.add_argument("--gripper-margin", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--stop-on-env-done", action="store_true")
    parser.add_argument("--dont-stop-when-complete", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face downloads/checks instead of cache-only mode.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tasks = list(args.tasks) if args.tasks else list(TASK_ORDER)
    specs = [COMPLEX_STL_SPECS[task] for task in tasks]
    reset_robot_y_min = None if args.disable_reset_pose_filter else args.reset_robot_y_min
    reset_robot_y_max = None if args.disable_reset_pose_filter else args.reset_robot_y_max
    reset_switch_clearance = None if args.disable_reset_pose_filter else args.reset_switch_clearance

    scene_config_path = resolve_existing_path(repo_path(args.scene_config))
    visualization_config = resolve_existing_path(repo_path(args.visualization_config))
    flower_checkpoint = resolve_existing_path(args.flower_checkpoint)
    env_checkpoint = resolve_existing_path(args.env_checkpoint)
    reset_pose_paths = resolve_reset_pose_paths(args.reset_pose_files)
    reset_pose_pool, reset_pose_meta = load_reset_pose_pool(reset_pose_paths)
    reset_poses, reset_pose_filter = filter_reset_poses(
        reset_pose_pool,
        robot_y_min=reset_robot_y_min,
        robot_y_max=reset_robot_y_max,
        switch_clearance=reset_switch_clearance,
    )
    reset_pose_filter = {**reset_pose_filter, **reset_pose_meta}
    fixed_reset_pose_index = None if args.sample_reset_poses else int(args.fixed_reset_pose_index)

    if args.safety_box is None:
        safety_box = SafetyBox()
    else:
        safety_box = SafetyBox(
            x_min=float(args.safety_box[0]),
            x_max=float(args.safety_box[1]),
            y_min=float(args.safety_box[2]),
            y_max=float(args.safety_box[3]),
            margin=SafetyBox().margin,
        )
    gripper_spec = GripperOpenSpec(min_width=float(args.gripper_min_width), margin=float(args.gripper_margin))
    run_name = args.name or f"flower_complex_stls_rollouts{args.n_rollouts}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = unique_run_dir(run_dir, run_name)

    planned = {
        "policy": "flower_vla",
        "flower_root": str(Path(args.flower_root).expanduser()),
        "flower_checkpoint": str(flower_checkpoint),
        "env_checkpoint": str(env_checkpoint),
        "scene_config": str(scene_config_path),
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "requested_name": args.name,
        "run_name": run_dir.name,
        "tasks": tasks,
        "instructions": {spec.name: spec.prompt for spec in specs},
        "task_specs": [
            {
                "name": spec.name,
                "mode": spec.mode,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "stage_target_names": [list(stage) for stage in spec.stage_target_names],
                "safety_kind": spec.safety_kind,
                "horizon": int(args.horizon if args.horizon is not None else spec.default_horizon),
            }
            for spec in specs
        ],
        "n_rollouts": int(args.n_rollouts),
        "seed_start": int(args.seed_start),
        "reset_pose_filter": reset_pose_filter,
        "reset_pose_selection": "sample_per_rollout_seed" if fixed_reset_pose_index is None else "fixed_index",
        "fixed_reset_pose_index": fixed_reset_pose_index,
        "settle_steps": int(args.settle_steps),
        "settle_gripper": float(args.settle_gripper),
        "safety_box": asdict(safety_box.normalized()),
        "gripper_spec": asdict(gripper_spec.normalized()),
        "save_video": not args.no_video,
        "stop_when_complete": not args.dont_stop_when_complete,
        "stop_on_env_done": bool(args.stop_on_env_done),
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return

    video_cfg = load_json(visualization_config)
    device = resolve_device(args.device)
    env_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(env_checkpoint))
    flower_model = load_flower_model(flower_checkpoint, device)
    flower_policy = FlowerPolicyAdapter(flower_model, device=device)

    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run_args.json", planned)
    print("device:", device)
    print("flower checkpoint:", flower_checkpoint)
    print("env checkpoint:", env_checkpoint)
    print("output:", run_dir)

    all_task_summaries = []
    for spec in specs:
        task_horizon = int(args.horizon if args.horizon is not None else spec.default_horizon)
        task_dir = run_dir / spec.name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            task_dir / "task_spec.json",
            {
                "name": spec.name,
                "mode": spec.mode,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "stage_target_names": [list(stage) for stage in spec.stage_target_names],
                "safety_kind": spec.safety_kind,
                "horizon": task_horizon,
                "instruction": spec.prompt,
            },
        )
        write_json(task_dir / "scene_config_resolved.json", load_json(scene_config_path))

        print(f"\nTask {spec.name}: instruction={spec.prompt!r}")
        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            rollout = rollout_flower_once(
                seed=seed,
                env_ckpt_dict=env_ckpt_dict,
                flower_policy=flower_policy,
                spec=spec,
                instruction=spec.prompt,
                scene_config_path=scene_config_path,
                reset_poses=reset_poses or [],
                reset_pose_filter=reset_pose_filter,
                fixed_reset_pose_index=fixed_reset_pose_index,
                output_dir=task_dir,
                rollout_tag=tag,
                horizon=task_horizon,
                video_cfg=video_cfg,
                safety_box=safety_box,
                gripper_spec=gripper_spec,
                save_video=not args.no_video,
                settle_steps=int(args.settle_steps),
                settle_gripper=float(args.settle_gripper),
                stop_when_complete=not args.dont_stop_when_complete,
                stop_on_env_done=bool(args.stop_on_env_done),
                fps=int(args.fps),
            )
            rollouts.append(rollout)
            safety_str = "na" if rollout["safety_satisfied"] is None else str(rollout["safety_satisfied"])
            events = ",".join(event["target_name"] for event in rollout["target_events"]) or "none"
            print(
                f"  seed {seed:03d}: stl={rollout['stl_satisfied']} live={rollout['liveness_satisfied']} "
                f"safe={safety_str} subgoals={rollout['completed_subgoals']}/{rollout['total_subgoals']} "
                f"steps={rollout['termination_step']:>3} events=[{events}] final={format_onehot(rollout['final_label'])}"
            )

        summary = task_summary(spec, rollouts, n_candidates=0, horizon=task_horizon)
        summary["instruction"] = spec.prompt
        write_json(task_dir / "task_summary.json", summary)
        if rollouts:
            CRU.plot_rollout_xy(
                rollouts,
                rollouts[0]["scene_snapshot"],
                f"FLOWER {spec.name} | STL {summary['stl_satisfied_count']}/{summary['n_rollouts']} | {spec.formula}",
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
