#!/usr/bin/env python3
"""Run FLOWER+GPC on the complex CALVIN safety tasks.

FLOWER proposes multiple stochastic action chunks in parallel from the same
language-conditioned observation.  A dynamics world model rolls out those
chunks, evaluates the safety robustness term, and the controller executes the
best chunk.  This is a safety-only baseline: the VLA language prompt still
contains the task/liveness request, while GPC only ranks candidates by predicted
safety robustness.
"""

from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/baselines/flower_gpc"
SAFETY_TASK_ORDER = ("region", "angle", "gripper")

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
    GRIPPER_WIDTH_RAW_ROBOT_IDX,
    GripperOpenSpec,
    SafetyBox,
    SpecMonitor,
    evaluate_safety,
    evaluate_subgoals,
    filter_complex_reset_poses,
    labels_for_scene,
    load_reset_pose_pool,
    make_angle_stl_spec,
    make_fixed_scene_robot,
    randomized_safety_context_for_rollout,
    resolve_reset_pose_paths,
    rollout_summary_payload,
    rzz_from_euler_xyz_np,
    rzz_to_tilt_angle_deg_np,
    save_rollout_diagnostics,
    save_trace_without_video,
    safety_randomization_plan,
    task_summary,
    unique_run_dir,
    write_rollout_rzz_angle_diagnostic_plot,
    write_rzz_angle_diagnostic_plot,
    write_summary_tables,
)
from calvin_experiments.run_complex_stl_automaton import (
    DEFAULT_DYNAMICS_CKPT,
    DynamicsRefiner,
    apply_rzz_action_warmup,
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


class FlowerCandidateAdapter(FlowerPolicyAdapter):
    """FLOWER adapter with batched stochastic action-chunk sampling."""

    def sample_action_chunks(
        self,
        obs,
        *,
        instruction: str,
        n_candidates: int,
        chunk_horizon: Optional[int] = None,
    ) -> np.ndarray:
        import torch

        flower_obs = self.observation_to_flower(obs)
        n_candidates = int(n_candidates)
        if n_candidates <= 0:
            raise ValueError(f"n_candidates must be positive, got {n_candidates}")
        rgb_static = flower_obs["rgb_obs"]["rgb_static"].repeat(n_candidates, 1, 1, 1, 1)
        rgb_gripper = flower_obs["rgb_obs"]["rgb_gripper"].repeat(n_candidates, 1, 1, 1, 1)
        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper,
            },
            "lang_text": [instruction] * n_candidates,
        }
        with torch.no_grad():
            features = self.model.encode_observations(batch)
            noise = torch.randn(
                n_candidates,
                self.model.act_window_size,
                self.model.action_dim,
                device=features["features"].device,
            )
            chunks = self.model.sample_actions(noise, features, inference=True)
        chunks_np = chunks.detach().cpu().numpy().astype(np.float32)
        horizon = self.model.act_window_size if chunk_horizon is None else int(chunk_horizon)
        horizon = max(1, min(int(horizon), int(chunks_np.shape[1])))
        chunks_np = chunks_np[:, :horizon, :]
        chunks_np[..., -1] = np.where(chunks_np[..., -1] > 0.0, 1.0, -1.0)
        return np.clip(chunks_np, -1.0, 1.0).astype(np.float32)


def score_safety_candidates(
    *,
    dynamics_refiner: DynamicsRefiner,
    spec,
    robot: Sequence[float],
    scene: Sequence[float],
    candidate_chunks: np.ndarray,
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
    smooth_min_tau: float,
) -> tuple[int, Dict[str, Any]]:
    import torch

    state_dyn = dynamics_refiner.raw_env_state_to_dynamics_state_torch(robot, scene)
    action_t = torch.as_tensor(candidate_chunks, device=dynamics_refiner.device, dtype=torch.float32)
    with torch.no_grad():
        if spec.safety_kind == "eef_avoid_box":
            scores, pred_xy, signed_dist = dynamics_refiner.safety_robustness(
                state_dyn,
                action_t,
                safety_box,
                smooth_min_tau,
            )
            score_record = {
                "score_kind": "eef_avoid_box_smooth_min_signed_distance",
                "candidate_scores": scores.detach().cpu().numpy().astype(float).tolist(),
                "candidate_min_signed_distance": signed_dist.min(dim=1).values.detach().cpu().numpy().astype(float).tolist(),
                "candidate_mean_signed_distance": signed_dist.mean(dim=1).detach().cpu().numpy().astype(float).tolist(),
                "selected_pred_xy": None,
            }
        elif spec.safety_kind == "gripper_open":
            scores, width = dynamics_refiner.gripper_open_robustness(
                state_dyn,
                action_t,
                gripper_spec,
                smooth_min_tau,
            )
            score_record = {
                "score_kind": "gripper_open_smooth_min_width_margin",
                "candidate_scores": scores.detach().cpu().numpy().astype(float).tolist(),
                "candidate_min_gripper_width": width.min(dim=1).values.detach().cpu().numpy().astype(float).tolist(),
                "candidate_mean_gripper_width": width.mean(dim=1).detach().cpu().numpy().astype(float).tolist(),
            }
        elif spec.safety_kind in {"tcp_rzz_30deg", "tcp_rzz_angle"}:
            scores, pred_rzz, abs_error, mse = dynamics_refiner.rzz_robustness(
                state_dyn,
                action_t,
                spec.rzz_spec,
            )
            score_record = {
                "score_kind": "tcp_rzz_angle_smooth_min_margin",
                "target_rzz": float(spec.rzz_spec.target),
                "target_angle_deg": float(spec.rzz_spec.angle_deg),
                "tolerance_deg": None if spec.rzz_spec.tolerance_deg is None else float(spec.rzz_spec.tolerance_deg),
                "tolerance_rzz": float(spec.rzz_spec.tolerance),
                "candidate_scores": scores.detach().cpu().numpy().astype(float).tolist(),
                "candidate_max_abs_rzz_error": abs_error.max(dim=1).values.detach().cpu().numpy().astype(float).tolist(),
                "candidate_mean_abs_rzz_error": abs_error.mean(dim=1).detach().cpu().numpy().astype(float).tolist(),
                "candidate_mse": mse.detach().cpu().numpy().astype(float).tolist(),
            }
        else:
            raise ValueError(f"FLOWER+GPC is safety-only; unsupported safety kind {spec.safety_kind!r}")
    selected_idx = int(torch.argmax(scores).detach().cpu())
    score_record.update(
        {
            "selected_idx": selected_idx,
            "selected_score": float(scores[selected_idx].detach().cpu()),
            "n_candidates": int(candidate_chunks.shape[0]),
            "chunk_horizon": int(candidate_chunks.shape[1]),
        }
    )
    return selected_idx, score_record


def rollout_flower_gpc_once(
    *,
    seed: int,
    env_ckpt_dict: Dict[str, Any],
    flower_policy: FlowerCandidateAdapter,
    dynamics_refiner: DynamicsRefiner,
    spec,
    instruction: str,
    scene_config_path: Path,
    reset_poses: Sequence[np.ndarray],
    reset_pose_filter: Dict[str, Any],
    fixed_reset_pose_index: Optional[int],
    output_dir: Path,
    rollout_tag: str,
    horizon: int,
    n_candidates: int,
    chunk_horizon: int,
    video_cfg: Dict[str, Any],
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
    safety_randomization: Optional[Dict[str, Any]],
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

        rzz_warmup: Dict[str, Any] = {"enabled": False, "mode": spec.rzz_init_mode}
        warmup_frames: list[np.ndarray] = []
        if spec.rzz_init_mode == "warmup":
            obs, rzz_warmup, warmup_frames = apply_rzz_action_warmup(
                env,
                obs,
                spec.rzz_spec,
                max_steps=int(spec.rzz_warmup_max_steps),
                tolerance=spec.rzz_warmup_tolerance,
                save_video=save_video,
                video_cfg=video_cfg,
                restack_obs=bool(spec.restack_after_warmup),
            )
        elif spec.rzz_init_mode not in {"none", ""}:
            raise ValueError(f"Unsupported rzz_init_mode for {spec.name}: {spec.rzz_init_mode}")

        flower_policy.reset(instruction)
        monitor = SpecMonitor(spec)
        scene_snapshot = CRU.capture_scene_snapshot(env)
        frames = list(warmup_frames)
        if save_video:
            frames.append(CRU.render_visual_camera(env, video_cfg))

        start_state = env.get_state()
        start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
        start_robot = np.asarray(start_state["robot"], dtype=np.float32).copy()
        reset_rzz = float(rzz_from_euler_xyz_np(start_robot[3:6]))
        reset_tcp_tilt_angle_deg = float(rzz_to_tilt_angle_deg_np(reset_rzz, spec.rzz_spec))
        binaries = CRU.articulated_binaries_from_start_state(start_scene)
        label0 = labels_for_scene(start_scene)
        monitor.sync(label0, 0)

        actions, rewards, dones, records = [], [], [], []
        scene_states = [start_scene.copy()]
        robot_states = [start_robot.copy()]
        labels_over_time = [label0.astype(int).copy()]
        eef_xy = [robot_states[-1][:2].copy()]
        tcp_rzz = [reset_rzz]
        tcp_tilt_angle_deg = [reset_tcp_tilt_angle_deg]
        action_queue: list[np.ndarray] = []
        first_behavior, first_behavior_step = "none", -1
        behavior_events = []
        last_behavior = "other"
        termination_reason = "horizon"
        env_done_step = -1
        total_reward = 0.0
        last_step = -1

        for step in range(int(horizon)):
            last_step = step
            if not action_queue:
                state_for_score = env.get_state()
                candidate_chunks = flower_policy.sample_action_chunks(
                    obs,
                    instruction=instruction,
                    n_candidates=int(n_candidates),
                    chunk_horizon=int(chunk_horizon),
                )
                selected_idx, score_record = score_safety_candidates(
                    dynamics_refiner=dynamics_refiner,
                    spec=spec,
                    robot=state_for_score["robot"],
                    scene=state_for_score["scene"],
                    candidate_chunks=candidate_chunks,
                    safety_box=safety_box,
                    gripper_spec=gripper_spec,
                    smooth_min_tau=float(spec.smooth_min_tau or 0.02),
                )
                selected_chunk = np.asarray(candidate_chunks[selected_idx], dtype=np.float32)
                action_queue.extend(selected_chunk)
                records.append(
                    {
                        "t": int(step),
                        "mode": "flower_gpc_safety_rank",
                        "instruction": instruction,
                        "safety_kind": spec.safety_kind,
                        "current_label": labels_over_time[-1].astype(int).tolist(),
                        **score_record,
                    }
                )

            action = np.asarray(action_queue.pop(0), dtype=np.float32).copy()
            next_obs, reward, done, _ = env.step(action)
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
            rzz_now = float(rzz_from_euler_xyz_np(robot_now[3:6]))
            tcp_rzz.append(rzz_now)
            tcp_tilt_angle_deg.append(float(rzz_to_tilt_angle_deg_np(rzz_now, spec.rzz_spec)))

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
            if advanced:
                action_queue.clear()
                records.append(
                    {
                        "t": int(step + 1),
                        "mode": "monitor_event",
                        "advanced": True,
                        "new_events": monitor.events[events_before:],
                        "done": bool(monitor.done),
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
            "policy": "flower_gpc_vla",
            "instruction": instruction,
            "scene_config": scene_cfg["name"],
            "scene_config_path": str(scene_config_path),
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
            "tcp_rzz": np.asarray(tcp_rzz, dtype=np.float32),
            "tcp_tilt_angle_deg": np.asarray(tcp_tilt_angle_deg, dtype=np.float32),
            "gripper_width": robot_states_np[:, GRIPPER_WIDTH_RAW_ROBOT_IDX].astype(np.float32),
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
            "safety_box": asdict(safety_box.normalized()) if spec.safety_kind == "eef_avoid_box" else None,
            "rzz_spec": asdict(spec.rzz_spec),
            "safety_randomization": safety_randomization or {"enabled": False},
            "scene_snapshot": scene_snapshot,
            "reset_robot_from_pose_file": bool(robot_from_pose_file),
            "fixed_reset_pose_index": None if fixed_reset_pose_index is None else int(fixed_reset_pose_index),
            "reset_pose_filter": reset_pose_filter,
            "reset_rzz": reset_rzz,
            "reset_tcp_tilt_angle_deg": reset_tcp_tilt_angle_deg,
            "rzz_warmup": rzz_warmup,
            "warmup_actions": np.asarray(rzz_warmup.get("actions", []), dtype=np.float32),
            "warmup_robot_states": np.asarray(rzz_warmup.get("robot_states", []), dtype=np.float32),
            "settle_steps": int(max(0, int(settle_steps))),
            "settle_action": settle_action.astype(float).tolist(),
            "pre_settle_label": pre_settle_label.astype(int).tolist(),
            "settle_scene_states": settle_scene_states,
            "settle_robot_states": settle_robot_states,
            "settle_rewards": settle_rewards,
            "settle_dones": settle_dones,
            "settle_metrics": settle_metrics(settle_scene_states),
            "n_candidates": int(n_candidates),
            "gpc_chunk_horizon": int(chunk_horizon),
        }
        if save_video:
            CRU.save_rollout_artifacts(rollout, frames, output_dir, rollout_tag, video_cfg, fps=fps)
        else:
            save_trace_without_video(rollout, output_dir, rollout_tag)
        save_rollout_diagnostics(rollout)
        write_rollout_rzz_angle_diagnostic_plot(rollout)
        rollout_dir = Path(rollout["rollout_dir"])
        CRU.plot_rollout_xy(
            [rollout],
            rollout["scene_snapshot"],
            f"FLOWER+GPC {spec.name} seed {seed} | stl={stl_satisfied} subgoals={completed_subgoals}/{total_subgoals}",
            save_path=rollout_dir / "rollout_xy.png",
            display_inline=False,
        )
        write_json(rollout_dir / "records.json", {"records": json_ready(records)})
        write_json(rollout_dir / "rollout_summary.json", json_ready(rollout_summary_payload(rollout)))
        return rollout
    finally:
        CRU.close_env_quietly(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=FLOWER_ROOT)
    parser.add_argument("--flower-checkpoint", type=Path, default=DEFAULT_FLOWER_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=DEFAULT_ENV_CHECKPOINT)
    parser.add_argument("--dynamics-ckpt", type=Path, default=DEFAULT_DYNAMICS_CKPT)
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
        help=f"Safety task subset. Omit for: {', '.join(SAFETY_TASK_ORDER)}.",
    )
    parser.add_argument("--n-rollouts", "--num-rollouts", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--chunk-horizon", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--angle-goal-deg", type=float, default=None)
    parser.add_argument("--angle-random-range-deg", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--angle-tolerance-deg", type=float, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reset-pose-files", type=Path, nargs="*", default=list(DEFAULT_RESET_POSE_FILES))
    parser.add_argument("--fixed-reset-pose-index", type=int, default=None)
    parser.add_argument("--sample-reset-poses", action="store_true")
    parser.add_argument("--use-scene-config-robot", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=DEFAULT_SETTLE_GRIPPER)
    parser.add_argument("--reset-robot-x-min", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_X_MIN)
    parser.add_argument("--reset-robot-x-max", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_X_MAX)
    parser.add_argument("--reset-robot-y-min", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_Y_MIN)
    parser.add_argument("--reset-robot-y-max", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_Y_MAX)
    parser.add_argument("--reset-switch-clearance", type=float, default=DEFAULT_RESET_SWITCH_CLEARANCE)
    parser.add_argument("--disable-reset-pose-filter", action="store_true")
    parser.add_argument("--safety-box", type=float, nargs=4, default=None, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--disable-safety-randomization", action="store_true")
    parser.add_argument("--gripper-min-width", type=float, default=0.06)
    parser.add_argument("--gripper-margin", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--stop-on-env-done", action="store_true")
    parser.add_argument("--dont-stop-when-complete", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    spec_by_name = dict(COMPLEX_STL_SPECS)
    if args.angle_goal_deg is not None:
        spec_by_name["angle"] = make_angle_stl_spec(float(args.angle_goal_deg))
    tasks = list(args.tasks) if args.tasks else list(SAFETY_TASK_ORDER)
    unsupported = [task for task in tasks if spec_by_name[task].category != "safety"]
    if unsupported:
        raise ValueError(f"FLOWER+GPC is safety-only; unsupported non-safety tasks: {', '.join(unsupported)}")
    specs = [spec_by_name[task] for task in tasks]

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
    dynamics_checkpoint = resolve_existing_path(repo_path(args.dynamics_ckpt))

    if args.use_scene_config_robot:
        reset_poses = []
        reset_pose_filter = {
            "enabled": False,
            "using_scene_config_robot": True,
            "filtered_count": 0,
            "sources": [],
            "total_count": 0,
        }
        fixed_reset_pose_index = None
    else:
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
        reset_pose_filter = {**reset_pose_filter, **reset_pose_meta, "using_scene_config_robot": False}
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

    run_name = args.name or f"flower_gpc_safety_N{args.n_rollouts}_candidates{args.n_candidates}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = unique_run_dir(run_dir, run_name)

    planned = {
        "policy": "flower_gpc_vla",
        "tasks": tasks,
        "flower_root": str(Path(args.flower_root).expanduser()),
        "flower_checkpoint": str(flower_checkpoint),
        "env_checkpoint": str(env_checkpoint),
        "dynamics_checkpoint": str(dynamics_checkpoint),
        "scene_config": "override" if args.scene_config is not None else "per_task",
        "scene_configs": {spec.name: str(scene_config_paths[spec.name]) for spec in specs},
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "requested_name": args.name,
        "run_name": run_dir.name,
        "n_rollouts": int(args.n_rollouts),
        "n_candidates": int(args.n_candidates),
        "chunk_horizon": int(args.chunk_horizon),
        "seed_start": int(args.seed_start),
        "angle_goal_deg_override": None if args.angle_goal_deg is None else float(args.angle_goal_deg),
        "instructions": {spec.name: spec.prompt for spec in specs},
        "task_specs": [
            {
                "name": spec.name,
                "mode": spec.mode,
                "category": spec.category,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "safety_kind": spec.safety_kind,
                "scene_config": str(scene_config_paths[spec.name]),
                "horizon": int(args.horizon if args.horizon is not None else spec.default_horizon),
                "smooth_min_tau": spec.smooth_min_tau,
                "rzz_spec": asdict(spec.rzz_spec),
                "rzz_init_mode": spec.rzz_init_mode,
                "rzz_warmup_max_steps": int(spec.rzz_warmup_max_steps),
                "rzz_warmup_tolerance": spec.rzz_warmup_tolerance,
                "restack_after_warmup": bool(spec.restack_after_warmup),
                "prompt": spec.prompt,
            }
            for spec in specs
        ],
        "use_scene_config_robot": bool(args.use_scene_config_robot),
        "reset_pose_filter": reset_pose_filter,
        "reset_pose_selection": (
            "scene_config_robot"
            if args.use_scene_config_robot
            else ("sample_per_rollout_seed" if fixed_reset_pose_index is None else "fixed_index")
        ),
        "fixed_reset_pose_index": fixed_reset_pose_index,
        "settle_steps": int(args.settle_steps),
        "settle_gripper": float(args.settle_gripper),
        "safety_box": asdict(safety_box.normalized()),
        "safety_randomization": safety_randomization_plan(
            safety_box,
            enabled=not bool(args.disable_safety_randomization),
            rzz_angle_deg_range=args.angle_random_range_deg,
            rzz_tolerance_deg=args.angle_tolerance_deg,
        ),
        "gripper_spec": asdict(gripper_spec.normalized()),
        "save_video": not args.no_video,
        "stop_when_complete": not args.dont_stop_when_complete,
        "stop_on_env_done": bool(args.stop_on_env_done),
        "implementation_note": (
            "FLOWER candidate chunks are sampled in one batched diffusion call by repeating the observation "
            "and language instruction n_candidates times."
        ),
    }
    if args.dry_run:
        print(json.dumps(json_ready(planned), indent=2))
        return

    video_cfg = load_json(visualization_config)
    device = resolve_device(args.device)
    env_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(env_checkpoint))
    flower_model = load_flower_model(flower_checkpoint, device)
    flower_policy = FlowerCandidateAdapter(flower_model, device=device)
    dynamics_refiner = DynamicsRefiner(dynamics_checkpoint, device)

    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run_args.json", json_ready(planned))
    print("device:", device)
    print("flower checkpoint:", flower_checkpoint)
    print("dynamics:", dynamics_refiner.meta["checkpoint_path"])
    print("output:", run_dir)

    all_task_summaries = []
    for spec in specs:
        task_horizon = int(args.horizon if args.horizon is not None else spec.default_horizon)
        task_scene_config_path = scene_config_paths[spec.name]
        task_dir = run_dir / spec.name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            task_dir / "task_spec.json",
            json_ready(
                {
                    "name": spec.name,
                    "mode": spec.mode,
                    "category": spec.category,
                    "formula": spec.formula,
                    "target_names": list(spec.target_names),
                    "safety_kind": spec.safety_kind,
                    "scene_config": str(task_scene_config_path),
                    "horizon": task_horizon,
                    "n_candidates": int(args.n_candidates),
                    "chunk_horizon": int(args.chunk_horizon),
                    "rzz_spec": asdict(spec.rzz_spec),
                    "rzz_init_mode": spec.rzz_init_mode,
                    "rzz_warmup_max_steps": int(spec.rzz_warmup_max_steps),
                    "rzz_warmup_tolerance": spec.rzz_warmup_tolerance,
                    "restack_after_warmup": bool(spec.restack_after_warmup),
                    "safety_randomization": safety_randomization_plan(
                        safety_box,
                        enabled=not bool(args.disable_safety_randomization),
                        rzz_angle_deg_range=args.angle_random_range_deg,
                        rzz_tolerance_deg=args.angle_tolerance_deg,
                    ),
                    "prompt": spec.prompt,
                }
            ),
        )
        write_json(task_dir / "scene_config_resolved.json", load_json(task_scene_config_path))

        print(f"\nTask {spec.name}: prompt={spec.prompt!r}")
        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            rollout_spec, rollout_safety_box, safety_randomization = randomized_safety_context_for_rollout(
                spec,
                safety_box,
                seed=seed,
                enabled=not bool(args.disable_safety_randomization),
                rzz_angle_deg_range=args.angle_random_range_deg,
                rzz_tolerance_deg=args.angle_tolerance_deg,
            )
            rollout = rollout_flower_gpc_once(
                seed=seed,
                env_ckpt_dict=env_ckpt_dict,
                flower_policy=flower_policy,
                dynamics_refiner=dynamics_refiner,
                spec=rollout_spec,
                instruction=rollout_spec.prompt,
                scene_config_path=task_scene_config_path,
                reset_poses=reset_poses or [],
                reset_pose_filter=reset_pose_filter,
                fixed_reset_pose_index=fixed_reset_pose_index,
                output_dir=task_dir,
                rollout_tag=tag,
                horizon=task_horizon,
                n_candidates=int(args.n_candidates),
                chunk_horizon=int(args.chunk_horizon),
                video_cfg=video_cfg,
                safety_box=rollout_safety_box,
                gripper_spec=gripper_spec,
                safety_randomization=safety_randomization,
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

        summary = task_summary(spec, rollouts, n_candidates=int(args.n_candidates), horizon=task_horizon)
        summary["policy"] = "flower_gpc_vla"
        summary["chunk_horizon"] = int(args.chunk_horizon)
        write_json(task_dir / "task_summary.json", json_ready(summary))
        if rollouts:
            CRU.plot_rollout_xy(
                rollouts,
                rollouts[0]["scene_snapshot"],
                f"FLOWER+GPC {spec.name} | STL {summary['stl_satisfied_count']}/{summary['n_rollouts']} | {spec.formula}",
                save_path=task_dir / "task_rollouts_xy.png",
                display_inline=False,
            )
        write_rzz_angle_diagnostic_plot(task_dir, spec, rollouts)
        all_task_summaries.append(summary)

    write_json(
        run_dir / "summary.json",
        json_ready(
            {
                "run": planned,
                "tasks": all_task_summaries,
                "dynamics": dynamics_refiner.meta,
            }
        ),
    )
    write_summary_tables(run_dir, all_task_summaries)
    print("\nSummary:", run_dir / "summary_table.md")


if __name__ == "__main__":
    main()
