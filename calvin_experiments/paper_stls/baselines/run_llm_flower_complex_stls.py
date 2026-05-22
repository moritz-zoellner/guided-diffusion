#!/usr/bin/env python3
"""Run static and in-loop LLM-planned FLOWER VLA baselines.

Both LLM baselines know the same set of FLOWER/CALVIN primitive language
commands.  They decompose the STL into the executable liveness/order part and do
not add safety language to the primitive commands.  Safety is still evaluated
after rollout, so the comparison shows whether the VLA itself can honor safety
from the plain VLA full-language prompt, while the LLM variants isolate planning.

Static mode plans once before execution, producing a sequence like
``[[a, b], c]``.  Lists mean an explicit OR; the rollout samples one option.
In-loop mode replans at high-level command boundaries from the observed history
of completed primitive propositions.
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/baselines/llm_flower"

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
from calvin_experiments.complex_stl_experiment_utils import (
    COMPLEX_STL_SPECS,
    DEFAULT_COMPLEX_RESET_ROBOT_X_MAX,
    DEFAULT_COMPLEX_RESET_ROBOT_X_MIN,
    DEFAULT_COMPLEX_RESET_ROBOT_Y_MAX,
    DEFAULT_COMPLEX_RESET_ROBOT_Y_MIN,
    DEFAULT_RESET_POSE_FILES,
    DEFAULT_SCENE_CONFIG,
    GripperOpenSpec,
    SafetyBox,
    SpecMonitor,
    evaluate_safety,
    evaluate_subgoals,
    filter_complex_reset_poses,
    labels_for_scene,
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
    DEFAULT_RESET_SWITCH_CLEARANCE,
    DEFAULT_SETTLE_GRIPPER,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_VISUALIZATION_CONFIG,
    format_onehot,
    idle_action_from_checkpoint,
    load_json,
    repo_path,
    resolve_existing_path,
    settle_metrics,
    write_json,
)
from flower_our_env_rollout import FlowerPolicyAdapter, load_flower_model, resolve_device


LLM_PRIMITIVE_GROUNDING: Dict[str, str] = {
    "button_pressed": "press the button to turn on the led light",
    "door_left": "push the sliding door to the left side",
    "door_right": "push the sliding door to the right side",
    "switch_off": "use the switch to turn off the light bulb",
    "switch_on": "use the switch to turn on the light bulb",
    "drawer_closed": "push the handle to close the drawer",
    "drawer_open": "pull the handle to open the drawer",
}

LLM_PRIMITIVE_ALIASES: Dict[str, str] = {
    "button_on": "button_pressed",
}

DEFAULT_LLM_TASK_ORDER = (
    "selection",
    "unordered",
    "conditional",
    "chained",
    "branched",
    "cyclic",
)

GENERATED_LLM_PLANS: Dict[str, Dict[str, Any]] = {
    "selection": {
        "task_name": "selection",
        "formula": "F(door_left OR switch_off OR button_on)",
        "open_loop_plan": [["door_left", "switch_off", "button_on"]],
        "closed_loop_policy": [
            {
                "if_history_contains": [],
                "and_history_not_contains": ["door_left", "switch_off", "button_on"],
                "next": ["door_left", "switch_off", "button_on"],
            }
        ],
        "notes": "This is a simple liveness disjunction: eventually complete any one of the three executable primitives.",
    },
    "unordered": {
        "task_name": "unordered",
        "formula": "F door_left AND F switch_off AND F button_on",
        "open_loop_plan": ["door_left", "switch_off", "button_on"],
        "closed_loop_policy": [
            {"if_history_contains": [], "and_history_not_contains": ["door_left"], "next": "door_left"},
            {"if_history_contains": ["door_left"], "and_history_not_contains": ["switch_off"], "next": "switch_off"},
            {
                "if_history_contains": ["door_left", "switch_off"],
                "and_history_not_contains": ["button_on"],
                "next": "button_on",
            },
        ],
        "notes": "The task requires all three liveness goals eventually, with no ordering constraint. The open-loop plan chooses one valid order.",
    },
    "conditional": {
        "task_name": "conditional",
        "formula": "F(button_on AND F(switch_off AND F drawer_closed))",
        "open_loop_plan": ["button_on", "switch_off", "drawer_closed"],
        "closed_loop_policy": [
            {"if_history_contains": [], "and_history_not_contains": ["button_on"], "next": "button_on"},
            {"if_history_contains": ["button_on"], "and_history_not_contains": ["switch_off"], "next": "switch_off"},
            {
                "if_history_contains": ["button_on", "switch_off"],
                "and_history_not_contains": ["drawer_closed"],
                "next": "drawer_closed",
            },
        ],
        "notes": (
            "The executable temporal structure is an ordering constraint: button_on and switch_off must occur before "
            "drawer_closed. The open-loop plan chooses button_on before switch_off, since their relative order is not otherwise constrained."
        ),
    },
    "chained": {
        "task_name": "chained",
        "formula": "F(button_on AND F(door_right AND F(switch_off AND F drawer_closed)))",
        "open_loop_plan": ["button_on", "door_right", "switch_off", "drawer_closed"],
        "closed_loop_policy": [
            {"if_history_contains": [], "and_history_not_contains": ["button_on"], "next": "button_on"},
            {"if_history_contains": ["button_on"], "and_history_not_contains": ["door_right"], "next": "door_right"},
            {
                "if_history_contains": ["button_on", "door_right"],
                "and_history_not_contains": ["switch_off"],
                "next": "switch_off",
            },
            {
                "if_history_contains": ["button_on", "door_right", "switch_off"],
                "and_history_not_contains": ["drawer_closed"],
                "next": "drawer_closed",
            },
        ],
        "notes": "This is a strict temporal chain: each primitive must eventually occur after the previous one.",
    },
    "branched": {
        "task_name": "branched",
        "formula": "F((button_on AND F(drawer_closed AND F switch_off)) OR (switch_off AND F(drawer_closed AND F button_on)))",
        "open_loop_plan": [["button_on", "switch_off"], "drawer_closed"],
        "closed_loop_policy": [
            {
                "if_history_contains": [],
                "and_history_not_contains": ["button_on", "switch_off"],
                "next": ["button_on", "switch_off"],
            },
            {"if_history_contains": ["button_on"], "and_history_not_contains": ["drawer_closed"], "next": "drawer_closed"},
            {"if_history_contains": ["switch_off"], "and_history_not_contains": ["drawer_closed"], "next": "drawer_closed"},
            {
                "if_history_contains": ["button_on", "drawer_closed"],
                "and_history_not_contains": ["switch_off"],
                "next": "switch_off",
            },
            {
                "if_history_contains": ["switch_off", "drawer_closed"],
                "and_history_not_contains": ["button_on"],
                "next": "button_on",
            },
        ],
        "notes": (
            "The previous open-loop plan incorrectly used a non-primitive placeholder. This version keeps only allowed primitives. "
            "The closed-loop policy handles the branch-dependent remaining option after drawer_closed."
        ),
    },
    "cyclic": {
        "task_name": "cyclic",
        "formula": "G F(drawer_open AND F(switch_on AND F(drawer_closed AND F switch_off)))",
        "open_loop_plan": ["drawer_open", "switch_on", "drawer_closed", "switch_off"],
        "closed_loop_policy": [
            {"if_history_contains": [], "and_history_not_contains": ["drawer_open"], "next": "drawer_open"},
            {"if_history_contains": ["drawer_open"], "and_history_not_contains": ["switch_on"], "next": "switch_on"},
            {
                "if_history_contains": ["drawer_open", "switch_on"],
                "and_history_not_contains": ["drawer_closed"],
                "next": "drawer_closed",
            },
            {
                "if_history_contains": ["drawer_open", "switch_on", "drawer_closed"],
                "and_history_not_contains": ["switch_off"],
                "next": "switch_off",
            },
        ],
        "notes": (
            "The executable cycle is drawer_open, then switch_on, then drawer_closed, then switch_off. Since the requested schema "
            "tracks only containment history and has no reset or loop counter, the open-loop plan returns one cycle and the closed-loop "
            "policy completes one cycle from an empty history."
        ),
    },
}


def normalize_primitive_options(step: str | Sequence[str]) -> list[str]:
    if isinstance(step, str):
        options = [canonical_primitive_name(step)]
    else:
        options = [canonical_primitive_name(str(item)) for item in step]
    if not options:
        raise ValueError("LLM plan step contains no primitive options")
    for primitive in options:
        if primitive not in LLM_PRIMITIVE_GROUNDING:
            raise ValueError(f"Unknown LLM primitive {primitive!r}")
    return options


def canonical_primitive_name(name: str) -> str:
    return LLM_PRIMITIVE_ALIASES.get(str(name), str(name))


def canonicalize_condition_names(names: Sequence[str]) -> list[str]:
    return [canonical_primitive_name(name) for name in names]


def canonicalize_policy_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "if_history_contains": canonicalize_condition_names(rule.get("if_history_contains", [])),
        "and_history_not_contains": canonicalize_condition_names(rule.get("and_history_not_contains", [])),
        "next": normalize_primitive_options(rule["next"]),
    }


def canonicalized_generated_plan(task_name: str) -> Dict[str, Any]:
    if task_name not in GENERATED_LLM_PLANS:
        raise KeyError(f"No generated LLM plan registered for task {task_name}")
    raw_plan = GENERATED_LLM_PLANS[task_name]
    canonical_steps = [normalize_primitive_options(step) for step in raw_plan["open_loop_plan"]]
    canonical_policy = [canonicalize_policy_rule(rule) for rule in raw_plan["closed_loop_policy"]]
    return {
        **raw_plan,
        "task_name": task_name,
        "canonical_open_loop_plan": canonical_steps,
        "canonical_closed_loop_policy": canonical_policy,
        "primitive_aliases": dict(LLM_PRIMITIVE_ALIASES),
    }


def primitive_options_payload(options: Sequence[str]) -> Dict[str, Any]:
    return {
        "primitive_options": list(options),
        "instruction_options": [LLM_PRIMITIVE_GROUNDING[name] for name in options],
    }


def plan_for_spec(spec, llm_mode: str) -> Dict[str, Any]:
    generated_plan = canonicalized_generated_plan(spec.name)
    steps = list(generated_plan["canonical_open_loop_plan"])
    return {
        "task": spec.name,
        "formula": spec.formula,
        "planner_input": generated_plan["formula"],
        "llm_mode": llm_mode,
        "planning_scope": "primitive liveness/order planning only",
        "safety_terms_not_planned": [],
        "strategy": generated_plan["notes"],
        "llm_generated_plan": generated_plan,
        "static_steps": [primitive_options_payload(options) for options in steps],
    }


def make_command_schedule(plan: Dict[str, Any], horizon: int, subtask_steps: Optional[int]) -> list[Dict[str, Any]]:
    steps = list(plan["static_steps"])
    if not steps:
        raise ValueError(f"LLM plan for {plan['task']} contains no steps")
    if subtask_steps is not None:
        budgets = [int(subtask_steps)] * len(steps)
    else:
        base = int(horizon) // len(steps)
        remainder = int(horizon) % len(steps)
        budgets = [base + (1 if idx < remainder else 0) for idx in range(len(steps))]
    if any(steps <= 0 for steps in budgets):
        raise ValueError(f"Invalid command schedule for horizon={horizon}, budgets={budgets}")
    schedule = []
    start = 0
    for idx, (step_payload, budget) in enumerate(zip(steps, budgets)):
        end = start + int(budget)
        schedule.append(
            {
                "idx": int(idx),
                "start_step": int(start),
                "end_step_exclusive": int(end),
                "budget": int(budget),
                **step_payload,
            }
        )
        start = end
    return schedule


def command_for_step(schedule: Sequence[Dict[str, Any]], step: int) -> Dict[str, Any]:
    for command in schedule:
        if int(command["start_step"]) <= int(step) < int(command["end_step_exclusive"]):
            return command
    return schedule[-1]


def choose_primitive_command(command: Dict[str, Any]) -> Dict[str, Any]:
    options = list(command["primitive_options"])
    primitive = random.choice(options)
    return {
        **command,
        "selected_primitive": primitive,
        "target": primitive,
        "instruction": LLM_PRIMITIVE_GROUNDING[primitive],
    }


def default_command_budget(spec, horizon: int, subtask_steps: Optional[int]) -> int:
    if subtask_steps is not None:
        return int(subtask_steps)
    step_count = max(1, len(GENERATED_LLM_PLANS[spec.name]["open_loop_plan"]))
    return max(1, int(horizon) // step_count)


def closed_loop_primitive_options(spec, monitor: SpecMonitor) -> list[str]:
    if monitor.done:
        return []
    generated_plan = canonicalized_generated_plan(spec.name)
    achieved = set(canonicalize_condition_names(monitor.achieved_names()))
    for rule in generated_plan["canonical_closed_loop_policy"]:
        required = set(rule["if_history_contains"])
        forbidden = set(rule["and_history_not_contains"])
        if required.issubset(achieved) and not forbidden.intersection(achieved):
            return list(rule["next"])
    return []


def make_closed_loop_command(
    *,
    spec,
    monitor: SpecMonitor,
    step: int,
    command_idx: int,
    budget: int,
    horizon: int,
) -> Optional[Dict[str, Any]]:
    options = closed_loop_primitive_options(spec, monitor)
    if not options:
        return None
    command = {
        "idx": int(command_idx),
        "start_step": int(step),
        "end_step_exclusive": int(min(horizon, step + int(budget))),
        "budget": int(budget),
        **primitive_options_payload(options),
        "history": list(monitor.achieved_names()),
    }
    return choose_primitive_command(command)


def rollout_llm_flower_once(
    *,
    seed: int,
    env_ckpt_dict: Dict[str, Any],
    flower_policy: FlowerPolicyAdapter,
    spec,
    llm_plan: Dict[str, Any],
    llm_mode: str,
    command_schedule: Optional[Sequence[Dict[str, Any]]],
    command_budget: int,
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
    policy_name = "llm_static_flower_vla" if llm_mode == "static" else "llm_in_loop_flower_vla"
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
        command_switches = []
        selected_static_commands: Dict[int, Dict[str, Any]] = {}
        active_command: Optional[Dict[str, Any]] = None
        planner_call_idx = 0
        last_behavior = "other"
        last_command_idx = None
        termination_reason = "horizon"
        env_done_step = -1
        total_reward = 0.0
        last_step = -1

        for step in range(int(horizon)):
            last_step = step
            if llm_mode == "static":
                if command_schedule is None:
                    raise ValueError("Static LLM mode requires a command schedule")
                scheduled_command = command_for_step(command_schedule, step)
                command_idx = int(scheduled_command["idx"])
                if command_idx not in selected_static_commands:
                    selected_static_commands[command_idx] = choose_primitive_command(scheduled_command)
                command = selected_static_commands[command_idx]
                if last_command_idx != command_idx:
                    flower_policy.reset(command["instruction"])
                    command_switches.append(
                        {
                            "step": int(step),
                            "command_idx": command_idx,
                            "llm_mode": llm_mode,
                            "primitive_options": list(command["primitive_options"]),
                            "selected_primitive": command["selected_primitive"],
                            "instruction": command["instruction"],
                        }
                    )
                    last_command_idx = command_idx
            elif llm_mode == "in_loop":
                if active_command is None or step >= int(active_command["end_step_exclusive"]):
                    active_command = make_closed_loop_command(
                        spec=spec,
                        monitor=monitor,
                        step=step,
                        command_idx=planner_call_idx,
                        budget=command_budget,
                        horizon=int(horizon),
                    )
                    if active_command is None:
                        termination_reason = "task_complete" if monitor.done else "planner_no_command"
                        break
                    planner_call_idx += 1
                    flower_policy.reset(active_command["instruction"])
                    command_switches.append(
                        {
                            "step": int(step),
                            "command_idx": int(active_command["idx"]),
                            "llm_mode": llm_mode,
                            "history": list(active_command.get("history", [])),
                            "primitive_options": list(active_command["primitive_options"]),
                            "selected_primitive": active_command["selected_primitive"],
                            "instruction": active_command["instruction"],
                        }
                    )
                command = active_command
            else:
                raise ValueError(f"Unknown LLM mode: {llm_mode}")

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
                    "llm_mode": llm_mode,
                    "command_idx": int(command["idx"]),
                    "command_target": command.get("selected_primitive"),
                    "primitive_options": list(command.get("primitive_options", [])),
                    "instruction": command["instruction"],
                    "history": list(command.get("history", [])),
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
            if llm_mode == "in_loop" and advanced:
                active_command = None

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
        instruction_sequence = [item["instruction"] for item in command_switches]

        rollout = {
            "task": spec.name,
            "formula": spec.formula,
            "policy": policy_name,
            "llm_mode": llm_mode,
            "instruction": " | ".join(instruction_sequence),
            "instruction_sequence": instruction_sequence,
            "llm_plan": llm_plan,
            "command_schedule": [] if command_schedule is None else list(command_schedule),
            "command_switches": command_switches,
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
            f"LLM+FLOWER {spec.name} seed {seed} | stl={stl_satisfied} subgoals={completed_subgoals}/{total_subgoals}",
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
                "policy": policy_name,
                "llm_mode": llm_mode,
                "instruction_sequence": instruction_sequence,
                "llm_plan": llm_plan,
                "command_schedule": [] if command_schedule is None else list(command_schedule),
                "command_switches": command_switches,
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
    parser.add_argument("--scene-config", type=Path, default=None)
    parser.add_argument("--visualization-config", type=Path, default=DEFAULT_VISUALIZATION_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--tasks",
        "--run-for",
        nargs="*",
        default=None,
        choices=tuple(COMPLEX_STL_SPECS),
        help=(
            "Optional subset of complex STL tasks to run. Omit this flag, or pass it with no values, "
            f"to run the LLM planning tasks: {', '.join(DEFAULT_LLM_TASK_ORDER)}."
        ),
    )
    parser.add_argument("--n-rollouts", "--num-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None, help="Override total horizon per STL task.")
    parser.add_argument(
        "--llm-mode",
        choices=("static", "in_loop"),
        default="static",
        help="Use a one-shot static LLM plan or replan from high-level history at command boundaries.",
    )
    parser.add_argument(
        "--subtask-steps",
        type=int,
        default=None,
        help="Use a fixed budget for every LLM primitive command. By default the task horizon is split over the static plan length.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reset-pose-files", type=Path, nargs="*", default=list(DEFAULT_RESET_POSE_FILES))
    parser.add_argument(
        "--fixed-reset-pose-index",
        type=int,
        default=None,
        help="Use one deterministic robot reset pose from the filtered pose pool. By default, each rollout samples from the constrained pool.",
    )
    parser.add_argument(
        "--sample-reset-poses",
        action="store_true",
        help="Sample a reset pose from the filtered pool for each rollout seed. This is now the default.",
    )
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=DEFAULT_SETTLE_GRIPPER)
    parser.add_argument("--reset-robot-x-min", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_X_MIN)
    parser.add_argument("--reset-robot-x-max", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_X_MAX)
    parser.add_argument("--reset-robot-y-min", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_Y_MIN)
    parser.add_argument("--reset-robot-y-max", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_Y_MAX)
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

    tasks = list(args.tasks) if args.tasks else list(DEFAULT_LLM_TASK_ORDER)
    unsupported = [task for task in tasks if task not in GENERATED_LLM_PLANS]
    if unsupported:
        raise ValueError(
            "The generated LLM planner is registered only for "
            f"{', '.join(DEFAULT_LLM_TASK_ORDER)}; unsupported requested tasks: {', '.join(unsupported)}"
        )
    specs = [COMPLEX_STL_SPECS[task] for task in tasks]
    reset_robot_x_min = None if args.disable_reset_pose_filter else args.reset_robot_x_min
    reset_robot_x_max = None if args.disable_reset_pose_filter else args.reset_robot_x_max
    reset_robot_y_min = None if args.disable_reset_pose_filter else args.reset_robot_y_min
    reset_robot_y_max = None if args.disable_reset_pose_filter else args.reset_robot_y_max
    reset_switch_clearance = None if args.disable_reset_pose_filter else args.reset_switch_clearance

    scene_config_args = {
        spec.name: (args.scene_config or spec.scene_config or DEFAULT_SCENE_CONFIG)
        for spec in specs
    }
    scene_config_paths = {
        spec.name: resolve_existing_path(repo_path(scene_config_args[spec.name]))
        for spec in specs
    }
    visualization_config = resolve_existing_path(repo_path(args.visualization_config))
    flower_checkpoint = resolve_existing_path(args.flower_checkpoint)
    env_checkpoint = resolve_existing_path(args.env_checkpoint)
    reset_pose_paths = resolve_reset_pose_paths(args.reset_pose_files)
    reset_pose_pool, reset_pose_meta = load_reset_pose_pool(reset_pose_paths)
    reset_poses, reset_pose_filter = filter_complex_reset_poses(
        reset_pose_pool,
        robot_x_min=reset_robot_x_min,
        robot_x_max=reset_robot_x_max,
        robot_y_min=reset_robot_y_min,
        robot_y_max=reset_robot_y_max,
        switch_clearance=reset_switch_clearance,
    )
    reset_pose_filter = {**reset_pose_filter, **reset_pose_meta}
    fixed_reset_pose_index = None if args.sample_reset_poses or args.fixed_reset_pose_index is None else int(args.fixed_reset_pose_index)

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
    policy_name = "llm_static_flower_vla" if args.llm_mode == "static" else "llm_in_loop_flower_vla"
    run_name = args.name or f"{args.llm_mode}_llm_flower_complex_stls_rollouts{args.n_rollouts}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = unique_run_dir(run_dir, run_name)

    plans = {}
    for spec in specs:
        task_horizon = int(args.horizon if args.horizon is not None else spec.default_horizon)
        plan = plan_for_spec(spec, args.llm_mode)
        command_budget = default_command_budget(spec, task_horizon, args.subtask_steps)
        plan["command_budget"] = int(command_budget)
        if args.llm_mode == "static":
            plan["command_schedule"] = make_command_schedule(plan, task_horizon, args.subtask_steps)
            plan["horizon"] = int(plan["command_schedule"][-1]["end_step_exclusive"])
        else:
            plan["command_schedule"] = []
            plan["horizon"] = int(task_horizon)
        plans[spec.name] = plan

    planned = {
        "policy": policy_name,
        "llm_mode": args.llm_mode,
        "planner": "static_llm_decomposition_by_codex" if args.llm_mode == "static" else "history_conditioned_llm_decomposition_by_codex",
        "fairness_note": (
            "The LLM only emits executable VLA primitives. OR steps are represented as primitive option sets "
            "and sampled uniformly. Safety terms are not translated into extra commands or controller logic."
        ),
        "primitive_grounding": LLM_PRIMITIVE_GROUNDING,
        "primitive_aliases": LLM_PRIMITIVE_ALIASES,
        "flower_root": str(Path(args.flower_root).expanduser()),
        "flower_checkpoint": str(flower_checkpoint),
        "env_checkpoint": str(env_checkpoint),
        "scene_config": "override" if args.scene_config is not None else "per_task",
        "scene_configs": {spec.name: str(scene_config_paths[spec.name]) for spec in specs},
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "requested_name": args.name,
        "run_name": run_dir.name,
        "scene_config_arg": None if args.scene_config is None else str(args.scene_config),
        "tasks": tasks,
        "llm_plans": plans,
        "task_specs": [
            {
                "name": spec.name,
                "mode": spec.mode,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "stage_target_names": [list(stage) for stage in spec.stage_target_names],
                "safety_kind": spec.safety_kind,
                "scene_config": str(scene_config_paths[spec.name]),
                "horizon": int(plans[spec.name]["horizon"]),
                "static_step_count": len(plans[spec.name]["static_steps"]),
                "command_budget": int(plans[spec.name]["command_budget"]),
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
        "generated_llm_plans": {
            name: canonicalized_generated_plan(name)
            for name in DEFAULT_LLM_TASK_ORDER
        },
        "subtask_steps": None if args.subtask_steps is None else int(args.subtask_steps),
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
    write_json(run_dir / "llm_plans.json", plans)
    print("device:", device)
    print("flower checkpoint:", flower_checkpoint)
    print("env checkpoint:", env_checkpoint)
    print("output:", run_dir)

    all_task_summaries = []
    for spec in specs:
        plan = plans[spec.name]
        task_horizon = int(plan["horizon"])
        task_scene_config_path = scene_config_paths[spec.name]
        command_schedule = plan["command_schedule"] if args.llm_mode == "static" else None
        command_budget = int(plan["command_budget"])
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
                "scene_config": str(task_scene_config_path),
                "horizon": task_horizon,
                "llm_mode": args.llm_mode,
                "command_budget": command_budget,
                "llm_plan": plan,
            },
        )
        write_json(task_dir / "scene_config_resolved.json", load_json(task_scene_config_path))

        print(f"\nTask {spec.name} ({args.llm_mode}):")
        if args.llm_mode == "static":
            for command in command_schedule or []:
                print(
                    f"  command {command['idx']}: steps {command['start_step']}..{command['end_step_exclusive'] - 1} "
                    f"options={command['primitive_options']}"
                )
        else:
            print(f"  replans every {command_budget} steps or when a high-level event is observed")

        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            rollout = rollout_llm_flower_once(
                seed=seed,
                env_ckpt_dict=env_ckpt_dict,
                flower_policy=flower_policy,
                spec=spec,
                llm_plan=plan,
                llm_mode=args.llm_mode,
                command_schedule=command_schedule,
                command_budget=command_budget,
                scene_config_path=task_scene_config_path,
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
        summary["llm_plan"] = plan
        summary["llm_mode"] = args.llm_mode
        summary["static_steps"] = plan["static_steps"]
        summary["command_budget"] = command_budget
        summary["command_count"] = len(command_schedule or [])
        write_json(task_dir / "task_summary.json", summary)
        if rollouts:
            CRU.plot_rollout_xy(
                rollouts,
                rollouts[0]["scene_snapshot"],
                f"LLM+FLOWER {spec.name} | STL {summary['stl_satisfied_count']}/{summary['n_rollouts']} | {spec.formula}",
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
