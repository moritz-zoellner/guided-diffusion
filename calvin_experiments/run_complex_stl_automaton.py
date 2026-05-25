#!/usr/bin/env python3
"""Run bulk CALVIN complex-STL experiments with automaton sample-and-rank.

This packages the six `calvin_experiments/paper_stls` notebook workflows into a
single repeatable paper run.  It keeps the same rollout artifacts as the recent
articulated-object experiments, but evaluates richer STL-style objectives:

- eventual OR: satisfy one of several target propositions
- eventual AND: satisfy all target propositions, greedily choosing among the
  remaining targets
- ordered chains / ordered stages
- eventual target plus a global safety-style constraint
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

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in [
    REPO_ROOT,
    REPO_ROOT / "robomimic",
    REPO_ROOT / "calvin" / "calvin_env",
    REPO_ROOT / "calvin_experiments",
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import robomimic.envs  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

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
    TASK_ORDER,
    VIDEO_FPS,
    ComplexSTLSpec,
    GripperOpenSpec,
    RzzSpec,
    SafetyBox,
    evaluate_safety,
    evaluate_subgoals,
    filter_complex_reset_poses,
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
from calvin_experiments.label_calvin_world_model import label_scene_states_for_names
from calvin_experiments.run_dynaguide_articulated_automaton import (
    DEFAULT_RESET_SWITCH_CLEARANCE,
    DEFAULT_SETTLE_GRIPPER,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_VISUALIZATION_CONFIG,
    AutomatonGuidance,
    format_onehot,
    idle_action_from_checkpoint,
    load_json,
    repeat_obs_batch,
    repo_path,
    resolve_existing_path,
    settle_metrics,
    unnormalize_action_sequence,
    write_json,
)
from calvin_experiments.train_dynamics_world_model import load_dynamics_model_for_eval


DEFAULT_OUTPUT_ROOT = Path("outputs/calvin_paper/complex-behaviors")
DEFAULT_COMPLEX_POLICY_CKPT = Path("outputs/calvin/base_policy/calvin_D_base_dp/20260501015147/models/model_epoch_280.pth")
DEFAULT_COMPLEX_AUTOMATON_CKPT = Path(
    "outputs/calvin/automaton_world_model/h8_sh64_ah96_lh16_hh128_lr0.0003_epochs80_2026-05-05_20-38-38"
)
DEFAULT_DYNAMICS_CKPT = Path(
    "outputs/calvin/dynamics_world_model/hd512_depth4_drop0.02_lr0.0005_epochs70_2026-05-06_01-42-08"
)


def euler_xyz_to_rot6d_torch(euler_xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = euler_xyz[..., 0], euler_xyz[..., 1], euler_xyz[..., 2]
    sx, cx = torch.sin(x), torch.cos(x)
    sy, cy = torch.sin(y), torch.cos(y)
    sz, cz = torch.sin(z), torch.cos(z)
    return torch.stack(
        [
            cz * cy,
            sz * cy,
            -sy,
            cz * sy * sx - sz * cx,
            sz * sy * sx + cz * cx,
            cy * sx,
        ],
        dim=-1,
    )


def project_rot6d_torch(rot6d: torch.Tensor) -> torch.Tensor:
    r1 = rot6d[..., 0:3]
    r2 = rot6d[..., 3:6]
    r1 = r1 / (torch.linalg.norm(r1, dim=-1, keepdim=True) + 1e-8)
    r2 = r2 - torch.sum(r1 * r2, dim=-1, keepdim=True) * r1
    r2 = r2 / (torch.linalg.norm(r2, dim=-1, keepdim=True) + 1e-8)
    return torch.cat([r1, r2], dim=-1)


def project_dynamics_state_torch(state: torch.Tensor) -> torch.Tensor:
    pieces = []
    cursor = 0
    for start, end in [(3, 9), (27, 33), (36, 42), (45, 51)]:
        pieces += [state[..., cursor:start], project_rot6d_torch(state[..., start:end])]
        cursor = end
    return torch.cat(pieces + [state[..., cursor:]], dim=-1)


def rot6d_to_rzz_torch(rot6d: torch.Tensor) -> torch.Tensor:
    rot6d = project_rot6d_torch(rot6d)
    r1 = rot6d[..., 0:3]
    r2 = rot6d[..., 3:6]
    r3 = torch.cross(r1, r2, dim=-1)
    return r3[..., 2]


def euler_xyz_to_matrix_np(euler_xyz: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(euler_xyz, dtype=np.float32)]
    sx, cx = np.sin(roll), np.cos(roll)
    sy, cy = np.sin(pitch), np.cos(pitch)
    sz, cz = np.sin(yaw), np.cos(yaw)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float32,
    )


def matrix_to_euler_xyz_np(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64)
    pitch = np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) < 1e-6:
        roll = 0.0
        yaw = np.arctan2(-rot[0, 1], rot[1, 1])
    else:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def target_tcp_z_axis_np(current_euler_xyz: Sequence[float], rzz_spec: RzzSpec) -> np.ndarray:
    target_z = float(rzz_spec.target)
    horizontal = float(np.sqrt(max(0.0, 1.0 - target_z ** 2)))
    if horizontal < 1e-8:
        return np.asarray([0.0, 0.0, np.sign(target_z) if target_z != 0 else 1.0], dtype=np.float32)
    current_rot = euler_xyz_to_matrix_np(current_euler_xyz)
    direction = np.asarray(current_rot[:2, 2], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.asarray(current_rot[:2, 0], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    return np.asarray([horizontal * direction[0], horizontal * direction[1], target_z], dtype=np.float32)


def euler_with_target_tcp_z_axis(current_euler_xyz: Sequence[float], target_z_axis: Sequence[float]) -> np.ndarray:
    r3 = np.asarray(target_z_axis, dtype=np.float32)
    r3 = r3 / (np.linalg.norm(r3) + 1e-8)
    current_rot = euler_xyz_to_matrix_np(current_euler_xyz)
    r1 = current_rot[:, 0]
    r1 = r1 - np.dot(r1, r3) * r3
    if np.linalg.norm(r1) < 1e-6:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(fallback, r3)) > 0.95:
            fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        r1 = fallback - np.dot(fallback, r3) * r3
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = np.cross(r3, r1)
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    rot = np.stack([r1, r2, r3], axis=1)
    return matrix_to_euler_xyz_np(rot)


def angle_diff(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def restack_current_observation(env):
    if not hasattr(env, "_get_initial_obs_history"):
        return None
    robomimic_env = env.env
    raw_obs = getattr(robomimic_env, "_current_obs", None)
    obs = robomimic_env.get_observation(raw_obs)
    env.timestep = 0
    env.update_obs(obs, reset=True)
    env.obs_history = env._get_initial_obs_history(init_obs=obs)
    return env._get_stacked_obs_from_history()


def apply_rzz_action_warmup(
    env,
    obs,
    rzz_spec: RzzSpec,
    *,
    max_steps: int,
    tolerance: Optional[float],
    save_video: bool,
    video_cfg: Dict[str, Any],
    restack_obs: bool,
) -> tuple[Any, Dict[str, Any], list[np.ndarray]]:
    tolerance = float(min(0.005, max(1e-4, 0.1 * float(rzz_spec.tolerance))) if tolerance is None else tolerance)
    start_robot = np.asarray(env.get_state()["robot"], dtype=np.float32).copy()
    target_axis = target_tcp_z_axis_np(start_robot[3:6], rzz_spec)
    target_euler = euler_with_target_tcp_z_axis(start_robot[3:6], target_axis)
    max_rel_orn = float(getattr(CRU.get_calvin_unwrapped_env(env).robot, "max_rel_orn", 0.05))
    warmup_actions, warmup_states, frames = [], [start_robot.copy()], []
    reached = abs(float(rzz_from_euler_xyz_np(start_robot[3:6])) - float(rzz_spec.target)) <= tolerance
    done = False
    for _ in range(int(max_steps)):
        if reached or done:
            break
        robot = np.asarray(env.get_state()["robot"], dtype=np.float32).copy()
        delta_euler = np.asarray(
            [angle_diff(float(target_euler[i]), float(robot[3 + i])) for i in range(3)],
            dtype=np.float32,
        )
        action = np.zeros(7, dtype=np.float32)
        action[3:6] = np.clip(delta_euler / max(max_rel_orn, 1e-6), -1.0, 1.0)
        action[6] = 1.0
        warmup_actions.append(action.copy())
        obs, _, done, _ = env.step(action.copy())
        new_robot = np.asarray(env.get_state()["robot"], dtype=np.float32).copy()
        warmup_states.append(new_robot.copy())
        if save_video:
            frames.append(CRU.render_visual_camera(env, video_cfg))
        reached = abs(float(rzz_from_euler_xyz_np(new_robot[3:6])) - float(rzz_spec.target)) <= tolerance
    if restack_obs:
        restacked = restack_current_observation(env)
        if restacked is not None:
            obs = restacked
    final_robot = np.asarray(env.get_state()["robot"], dtype=np.float32).copy()
    info = {
        "enabled": True,
        "mode": "warmup",
        "target_rzz": float(rzz_spec.target),
        "tolerance": tolerance,
        "target_tcp_z_axis": [float(v) for v in target_axis],
        "target_euler_xyz": [float(v) for v in target_euler],
        "start_xyz": start_robot[:3].astype(float).tolist(),
        "start_rzz": float(rzz_from_euler_xyz_np(start_robot[3:6])),
        "start_angle_deg": float(rzz_to_tilt_angle_deg_np(rzz_from_euler_xyz_np(start_robot[3:6]), rzz_spec)),
        "final_xyz": final_robot[:3].astype(float).tolist(),
        "final_rzz": float(rzz_from_euler_xyz_np(final_robot[3:6])),
        "final_angle_deg": float(rzz_to_tilt_angle_deg_np(rzz_from_euler_xyz_np(final_robot[3:6]), rzz_spec)),
        "final_angle_error_deg": float(abs(rzz_to_tilt_angle_deg_np(rzz_from_euler_xyz_np(final_robot[3:6]), rzz_spec) - float(rzz_spec.angle_deg))),
        "xyz_drift": (final_robot[:3] - start_robot[:3]).astype(float).tolist(),
        "n_steps": len(warmup_actions),
        "reached_tolerance": bool(reached),
        "restacked_obs": bool(restack_obs),
        "actions": np.asarray(warmup_actions, dtype=np.float32).tolist(),
        "robot_states": np.asarray(warmup_states, dtype=np.float32).tolist(),
        "rzz_trace": [float(rzz_from_euler_xyz_np(state[3:6])) for state in warmup_states],
        "angle_deg_trace": [float(rzz_to_tilt_angle_deg_np(rzz_from_euler_xyz_np(state[3:6]), rzz_spec)) for state in warmup_states],
        "xyz_trace": [state[:3].astype(float).tolist() for state in warmup_states],
    }
    return obs, info, frames


class DynamicsRefiner:
    def __init__(self, checkpoint_path: Path, device):
        self.model, self.stats, self.ckpt, self.meta = load_dynamics_model_for_eval(checkpoint_path, device=device)
        self.model.eval()
        self.device = device
        self.stats_t = {
            key: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0)
            for key, value in self.stats.items()
        }

    def raw_env_state_to_dynamics_state_torch(self, robot: Sequence[float], scene: Sequence[float]) -> torch.Tensor:
        robot_t = torch.as_tensor(robot, device=self.device, dtype=torch.float32).reshape(1, -1)
        scene_t = torch.as_tensor(scene, device=self.device, dtype=torch.float32).reshape(1, -1)
        robot_dyn = torch.cat([robot_t[..., :3], euler_xyz_to_rot6d_torch(robot_t[..., 3:6]), robot_t[..., 6:]], dim=-1)
        scene_parts = [scene_t[..., :6]]
        for start in (6, 12, 18):
            scene_parts += [scene_t[..., start:start + 3], euler_xyz_to_rot6d_torch(scene_t[..., start + 3:start + 6])]
        return torch.cat([robot_dyn, torch.cat(scene_parts, dim=-1)], dim=-1)

    def rollout_torch(self, state_dyn: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        state = state_dyn.expand(action_chunk.shape[0], -1)
        states = []
        for t in range(action_chunk.shape[1]):
            state_n = (state - self.stats_t["state_mean"]) / self.stats_t["state_std"]
            action_n = (action_chunk[:, t, :] - self.stats_t["action_mean"]) / self.stats_t["action_std"]
            delta_n = self.model(state_n, action_n)
            delta = delta_n * self.stats_t["delta_std"] + self.stats_t["delta_mean"]
            state = project_dynamics_state_torch(state + delta)
            states.append(state)
        return torch.stack(states, dim=1)

    def rzz_robustness(
        self,
        state_dyn: torch.Tensor,
        action_chunk: torch.Tensor,
        rzz_spec: RzzSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_states = self.rollout_torch(state_dyn, action_chunk)
        pred_rzz = rot6d_to_rzz_torch(pred_states[..., 3:9])
        target = torch.as_tensor(rzz_spec.target, device=pred_rzz.device, dtype=pred_rzz.dtype)
        abs_error = torch.abs(pred_rzz - target)
        if rzz_spec.tolerance_deg is not None:
            signed_cos = float(rzz_spec.axis_sign) * pred_rzz
            pred_angle = torch.acos(torch.clamp(signed_cos, -1.0 + 1e-6, 1.0 - 1e-6)) * (180.0 / np.pi)
            angle_error = torch.abs(pred_angle - float(rzz_spec.angle_deg))
            margins = float(rzz_spec.tolerance_deg) - angle_error
        else:
            margins = float(rzz_spec.tolerance) - abs_error
        # Keep the optimizer in Rzz space even when safety is reported in degrees.
        # Otherwise --angle-tolerance-deg silently rescales guidance relative to the notebook.
        mean_sq_error = torch.mean((pred_rzz - target) ** 2, dim=-1)
        tau = max(float(rzz_spec.smooth_min_tau), 1e-6)
        smooth_min_margin = -tau * torch.logsumexp(-margins / tau, dim=-1)
        return smooth_min_margin, pred_rzz, abs_error, mean_sq_error

    def refine_for_rzz(
        self,
        robot: Sequence[float],
        scene: Sequence[float],
        action_chunk: np.ndarray,
        rzz_spec: RzzSpec,
        guidance_scale: float,
        gradient_steps: int,
        step_size: float,
        action_reg: float,
        action_dims: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        state_dyn = self.raw_env_state_to_dynamics_state_torch(robot, scene)
        original = torch.as_tensor(action_chunk[None], device=self.device, dtype=torch.float32)
        actions = original.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=float(step_size))
        action_dims = None if action_dims is None else tuple(int(dim) for dim in action_dims)
        history = []
        with torch.no_grad():
            robust_before, pred_rzz_before, abs_error_before, mse_before = self.rzz_robustness(
                state_dyn, original, rzz_spec
            )
        if float(guidance_scale) <= 0.0 or int(gradient_steps) <= 0:
            pred_angle = rzz_to_tilt_angle_deg_np(pred_rzz_before[0].detach().cpu().numpy(), rzz_spec)
            record = {
                "enabled": False,
                "target_rzz": float(rzz_spec.target),
                "target_angle_deg": float(rzz_spec.angle_deg),
                "tolerance": float(rzz_spec.tolerance),
                "tolerance_deg": None if rzz_spec.tolerance_deg is None else float(rzz_spec.tolerance_deg),
                "robust_before": float(robust_before[0].cpu()),
                "robust_after": float(robust_before[0].cpu()),
                "mse_before": float(mse_before[0].cpu()),
                "mse_after": float(mse_before[0].cpu()),
                "mean_abs_rzz_error_before": float(abs_error_before[0].mean().cpu()),
                "mean_abs_rzz_error_after": float(abs_error_before[0].mean().cpu()),
                "pred_rzz_before": pred_rzz_before[0].detach().cpu().numpy().tolist(),
                "pred_rzz_after": pred_rzz_before[0].detach().cpu().numpy().tolist(),
                "pred_angle_deg_before": np.asarray(pred_angle, dtype=np.float32).tolist(),
                "pred_angle_deg_after": np.asarray(pred_angle, dtype=np.float32).tolist(),
                "action_l2_change": 0.0,
                "guided_action_dims": None if action_dims is None else list(action_dims),
                "history": [],
            }
            return np.asarray(action_chunk, dtype=np.float32), record

        for _ in range(int(gradient_steps)):
            opt.zero_grad(set_to_none=True)
            robust, _, _, mse = self.rzz_robustness(state_dyn, actions, rzz_spec)
            action_penalty = torch.mean((actions - original) ** 2, dim=(1, 2))
            objective = -float(guidance_scale) * mse - float(action_reg) * action_penalty
            (-objective.mean()).backward()
            opt.step()
            with torch.no_grad():
                if action_dims is not None:
                    keep = original.clone()
                    keep[..., list(action_dims)] = actions[..., list(action_dims)]
                    actions.copy_(keep)
                actions.clamp_(-1.0, 1.0)
                history.append(
                    {
                        "robustness": float(robust[0].detach().cpu()),
                        "mse": float(mse[0].detach().cpu()),
                        "objective": float(objective[0].detach().cpu()),
                    }
                )
        with torch.no_grad():
            robust_after, pred_rzz_after, abs_error_after, mse_after = self.rzz_robustness(
                state_dyn, actions, rzz_spec
            )
            non_guided_l2 = 0.0
            if action_dims is not None:
                mask = torch.ones(actions.shape[-1], device=self.device, dtype=torch.bool)
                mask[list(action_dims)] = False
                non_guided_l2 = float(torch.linalg.norm((actions - original)[..., mask]).detach().cpu())
        pred_angle_before = rzz_to_tilt_angle_deg_np(pred_rzz_before[0].detach().cpu().numpy(), rzz_spec)
        pred_angle_after = rzz_to_tilt_angle_deg_np(pred_rzz_after[0].detach().cpu().numpy(), rzz_spec)
        angle_error_before = np.abs(pred_angle_before - float(rzz_spec.angle_deg))
        angle_error_after = np.abs(pred_angle_after - float(rzz_spec.angle_deg))
        record = {
            "enabled": True,
            "target_rzz": float(rzz_spec.target),
            "target_angle_deg": float(rzz_spec.angle_deg),
            "tolerance": float(rzz_spec.tolerance),
            "tolerance_deg": None if rzz_spec.tolerance_deg is None else float(rzz_spec.tolerance_deg),
            "robust_before": float(robust_before[0].cpu()),
            "robust_after": float(robust_after[0].cpu()),
            "mse_before": float(mse_before[0].cpu()),
            "mse_after": float(mse_after[0].cpu()),
            "mean_abs_rzz_error_before": float(abs_error_before[0].mean().cpu()),
            "mean_abs_rzz_error_after": float(abs_error_after[0].mean().cpu()),
            "mean_abs_angle_error_deg_before": float(np.mean(angle_error_before)),
            "mean_abs_angle_error_deg_after": float(np.mean(angle_error_after)),
            "pred_rzz_before": pred_rzz_before[0].detach().cpu().numpy().tolist(),
            "pred_rzz_after": pred_rzz_after[0].detach().cpu().numpy().tolist(),
            "pred_angle_deg_before": np.asarray(pred_angle_before, dtype=np.float32).tolist(),
            "pred_angle_deg_after": np.asarray(pred_angle_after, dtype=np.float32).tolist(),
            "action_l2_change": float(torch.linalg.norm(actions - original).detach().cpu()),
            "non_guided_action_l2_change": non_guided_l2,
            "guided_action_dims": None if action_dims is None else list(action_dims),
            "history": history,
        }
        return actions[0].detach().cpu().numpy().astype(np.float32), record

    def gripper_open_robustness(
        self,
        state_dyn: torch.Tensor,
        action_chunk: torch.Tensor,
        spec: GripperOpenSpec,
        smooth_min_tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = spec.normalized()
        pred_states = self.rollout_torch(state_dyn, action_chunk)
        gripper_width = pred_states[..., 9]
        width_margin = gripper_width - float(spec.min_width)
        capped = torch.clamp(width_margin, max=float(spec.margin))
        tau = max(float(smooth_min_tau), 1e-6)
        smooth_min = -tau * torch.logsumexp(-capped / tau, dim=-1)
        return smooth_min, gripper_width

    def raw_gripper_width_objective(
        self,
        state_dyn: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_states = self.rollout_torch(state_dyn, action_chunk)
        gripper_width = pred_states[..., 9]
        return gripper_width.mean(dim=-1), gripper_width

    def refine_for_gripper_open(
        self,
        robot: Sequence[float],
        scene: Sequence[float],
        action_chunk: np.ndarray,
        spec: GripperOpenSpec,
        guidance_scale: float,
        gradient_steps: int,
        step_size: float,
        action_reg: float,
        smooth_min_tau: float,
        mode: str = "smooth_min_all_actions",
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        state_dyn = self.raw_env_state_to_dynamics_state_torch(robot, scene)
        original = torch.as_tensor(action_chunk[None], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            robust_before, width_before = self.gripper_open_robustness(state_dyn, original, spec, smooth_min_tau)
        if float(guidance_scale) <= 0.0 or int(gradient_steps) <= 0:
            record = {
                "enabled": False,
                "mode": mode,
                "robust_before": float(robust_before[0].cpu()),
                "robust_after": float(robust_before[0].cpu()),
                "min_width_before": float(width_before[0].min().cpu()),
                "min_width_after": float(width_before[0].min().cpu()),
                "pred_width_before": width_before[0].detach().cpu().numpy().tolist(),
                "pred_width_after": width_before[0].detach().cpu().numpy().tolist(),
                "action_l2_change": 0.0,
                "non_gripper_l2_change": 0.0,
                "gripper_l2_change": 0.0,
                "history": [],
            }
            return np.asarray(action_chunk, dtype=np.float32), record

        if mode not in {
            "smooth_min_all_actions",
            "world_model_all_actions",
            "world_model_gripper_only",
            "world_model_gripper_value_only",
            "world_model_gripper_value_all_actions",
        }:
            raise ValueError(f"Unknown gripper guidance mode: {mode}")
        actions = original.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=float(step_size))
        history = []
        for _ in range(int(gradient_steps)):
            opt.zero_grad(set_to_none=True)
            action_eval = actions
            if mode in {"world_model_gripper_only", "world_model_gripper_value_only"}:
                action_eval = original.clone()
                action_eval[..., -1] = actions[..., -1]
            if mode in {"world_model_gripper_value_only", "world_model_gripper_value_all_actions"}:
                robust, width = self.raw_gripper_width_objective(state_dyn, action_eval)
            else:
                robust, width = self.gripper_open_robustness(state_dyn, action_eval, spec, smooth_min_tau)
            delta = action_eval - original
            action_penalty = torch.mean(delta ** 2, dim=(1, 2))
            objective = float(guidance_scale) * robust - float(action_reg) * action_penalty
            (-objective.mean()).backward()
            opt.step()
            with torch.no_grad():
                if mode in {"world_model_gripper_only", "world_model_gripper_value_only"}:
                    keep = original.clone()
                    keep[..., -1] = actions[..., -1]
                    actions.copy_(keep)
                actions.clamp_(-1.0, 1.0)
                history.append(
                    {
                        "robustness": float(robust[0].detach().cpu()),
                        "min_width": float(width[0].min().detach().cpu()),
                        "action_penalty": float(action_penalty[0].detach().cpu()),
                        "objective": float(objective[0].detach().cpu()),
                    }
                )
        with torch.no_grad():
            if mode in {"world_model_gripper_value_only", "world_model_gripper_value_all_actions"}:
                robust_after, width_after = self.raw_gripper_width_objective(state_dyn, actions)
            else:
                robust_after, width_after = self.gripper_open_robustness(state_dyn, actions, spec, smooth_min_tau)
        delta = actions - original
        nongrip = delta.clone()
        nongrip[..., -1] = 0.0
        grip = delta[..., -1]
        record = {
            "enabled": True,
            "mode": mode,
            "robust_before": float(robust_before[0].cpu()),
            "robust_after": float(robust_after[0].cpu()),
            "min_width_before": float(width_before[0].min().cpu()),
            "min_width_after": float(width_after[0].min().cpu()),
            "pred_width_before": width_before[0].detach().cpu().numpy().tolist(),
            "pred_width_after": width_after[0].detach().cpu().numpy().tolist(),
            "action_l2_change": float(torch.linalg.norm(delta).detach().cpu()),
            "non_gripper_l2_change": float(torch.linalg.norm(nongrip).detach().cpu()),
            "gripper_l2_change": float(torch.linalg.norm(grip).detach().cpu()),
            "gripper_action_before": original[0, :, -1].detach().cpu().numpy().tolist(),
            "gripper_action_after": actions[0, :, -1].detach().cpu().numpy().tolist(),
            "history": history,
        }
        return actions[0].detach().cpu().numpy().astype(np.float32), record

    def safety_robustness(
        self,
        state_dyn: torch.Tensor,
        action_chunk: torch.Tensor,
        safety_box: SafetyBox,
        smooth_min_tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_states = self.rollout_torch(state_dyn, action_chunk)
        pred_xy = pred_states[..., 0:2]
        box = safety_box.normalized()
        center = torch.tensor([(box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2], device=pred_xy.device, dtype=pred_xy.dtype)
        half = torch.tensor([(box.x_max - box.x_min) / 2, (box.y_max - box.y_min) / 2], device=pred_xy.device, dtype=pred_xy.dtype)
        q = torch.abs(pred_xy - center) - half
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.minimum(torch.maximum(q[..., 0], q[..., 1]), torch.zeros((), device=pred_xy.device, dtype=pred_xy.dtype))
        signed_dist = outside + inside
        capped = torch.clamp(signed_dist, max=float(box.margin))
        tau = max(float(smooth_min_tau), 1e-6)
        smooth_min = -tau * torch.logsumexp(-capped / tau, dim=-1)
        return smooth_min, pred_xy, signed_dist

    def refine_for_safety_box(
        self,
        robot: Sequence[float],
        scene: Sequence[float],
        action_chunk: np.ndarray,
        safety_box: SafetyBox,
        guidance_scale: float,
        gradient_steps: int,
        step_size: float,
        action_reg: float,
        smooth_min_tau: float,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        state_dyn = self.raw_env_state_to_dynamics_state_torch(robot, scene)
        original = torch.as_tensor(action_chunk[None], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            robust_before, pred_xy_before, signed_before = self.safety_robustness(
                state_dyn, original, safety_box, smooth_min_tau
            )
        if float(guidance_scale) <= 0.0 or int(gradient_steps) <= 0:
            record = {
                "enabled": False,
                "robust_before": float(robust_before[0].cpu()),
                "robust_after": float(robust_before[0].cpu()),
                "min_signed_dist_before": float(signed_before[0].min().cpu()),
                "min_signed_dist_after": float(signed_before[0].min().cpu()),
                "pred_xy_before": pred_xy_before[0].detach().cpu().numpy().tolist(),
                "pred_xy_after": pred_xy_before[0].detach().cpu().numpy().tolist(),
                "action_l2_change": 0.0,
                "history": [],
            }
            return np.asarray(action_chunk, dtype=np.float32), record

        actions = original.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=float(step_size))
        history = []
        for _ in range(int(gradient_steps)):
            opt.zero_grad(set_to_none=True)
            robust, _, _ = self.safety_robustness(state_dyn, actions, safety_box, smooth_min_tau)
            action_penalty = torch.mean((actions - original) ** 2, dim=(1, 2))
            objective = float(guidance_scale) * robust - float(action_reg) * action_penalty
            (-objective.mean()).backward()
            opt.step()
            with torch.no_grad():
                actions.clamp_(-1.0, 1.0)
                history.append(
                    {
                        "robustness": float(robust[0].detach().cpu()),
                        "action_penalty": float(action_penalty[0].detach().cpu()),
                        "objective": float(objective[0].detach().cpu()),
                    }
                )
        with torch.no_grad():
            robust_after, pred_xy_after, signed_after = self.safety_robustness(
                state_dyn, actions, safety_box, smooth_min_tau
            )
        record = {
            "enabled": True,
            "robust_before": float(robust_before[0].cpu()),
            "robust_after": float(robust_after[0].cpu()),
            "min_signed_dist_before": float(signed_before[0].min().cpu()),
            "min_signed_dist_after": float(signed_after[0].min().cpu()),
            "pred_xy_before": pred_xy_before[0].detach().cpu().numpy().tolist(),
            "pred_xy_after": pred_xy_after[0].detach().cpu().numpy().tolist(),
            "action_l2_change": float(torch.linalg.norm(actions - original).detach().cpu()),
            "history": history,
        }
        return actions[0].detach().cpu().numpy().astype(np.float32), record


def target_idxs_from_names(guidance: AutomatonGuidance, names: Sequence[str]) -> list[int]:
    return [guidance.label_index(name) for name in names]


def score_candidate_batch(
    policy,
    guidance: AutomatonGuidance,
    obs,
    env,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    automaton_state, automaton_label = guidance.current_state_and_label(env)
    obs_tensor = policy._prepare_observation(obs)
    obs_tensor_rank = repeat_obs_batch(obs_tensor, int(n_candidates))
    with torch.no_grad():
        action_chunk_n = policy.policy._get_action_trajectory(obs_dict=obs_tensor_rank)
    action_chunks = unnormalize_action_sequence(policy, action_chunk_n)
    candidate_probs, automaton_horizon = guidance.predict_future_label_probs(
        automaton_state, automaton_label, action_chunks
    )
    return action_chunks, candidate_probs, automaton_horizon, automaton_label


def make_or_action_sampler(policy, guidance: AutomatonGuidance, target_names: Sequence[str], n_candidates: int):
    option_idxs = target_idxs_from_names(guidance, target_names)
    state = {"done": False, "event": None}

    def sync(label, step):
        if state["done"]:
            return False
        achieved = [idx for idx in option_idxs if float(label[idx]) > 0.5]
        if not achieved:
            return False
        target_idx = int(achieved[0])
        state["event"] = {"step": int(step), "target_idx": target_idx, "target_name": guidance.label_names[target_idx]}
        state["done"] = True
        action_sampler.done = True
        action_sampler.events = [state["event"]]
        return True

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        option_score_columns = np.stack([guidance.score_label_probs(candidate_probs, idx) for idx in option_idxs], axis=1)
        candidate_scores = option_score_columns.max(axis=1)
        candidate_option_pos = option_score_columns.argmax(axis=1)
        selected_idx = int(np.argmax(candidate_scores))
        selected_target_idx = int(option_idxs[int(candidate_option_pos[selected_idx])])
        record = {
            "t": int(step),
            "mode": "or",
            "option_idxs": [int(idx) for idx in option_idxs],
            "option_names": [guidance.label_names[idx] for idx in option_idxs],
            "option_score_rules": [guidance.score_rule_name(idx) for idx in option_idxs],
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_option_idx": selected_target_idx,
            "selected_option_name": guidance.label_names[selected_target_idx],
            "selected_option_scores": {
                guidance.label_names[idx]: float(option_score_columns[selected_idx, pos])
                for pos, idx in enumerate(option_idxs)
            },
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    action_sampler.done = False
    action_sampler.events = []
    action_sampler.violations = []
    return action_sampler


def make_remaining_and_action_sampler(policy, guidance: AutomatonGuidance, target_names: Sequence[str], n_candidates: int):
    target_idxs = target_idxs_from_names(guidance, target_names)
    achieved: set[int] = set()
    events: list[Dict[str, Any]] = []

    def sync(label, step):
        advanced = False
        for idx in target_idxs:
            if idx not in achieved and float(label[idx]) > 0.5:
                achieved.add(idx)
                events.append({"step": int(step), "target_idx": int(idx), "target_name": guidance.label_names[idx]})
                advanced = True
        action_sampler.events = events
        action_sampler.done = len(achieved) == len(target_idxs)
        return advanced

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        remaining = [idx for idx in target_idxs if idx not in achieved]
        if not remaining:
            remaining = [target_idxs[-1]]
        remaining_score_columns = np.stack([guidance.score_label_probs(candidate_probs, idx) for idx in remaining], axis=1)
        candidate_scores = remaining_score_columns.max(axis=1)
        candidate_target_pos = remaining_score_columns.argmax(axis=1)
        selected_idx = int(np.argmax(candidate_scores))
        selected_target_idx = int(remaining[int(candidate_target_pos[selected_idx])])
        record = {
            "t": int(step),
            "mode": "remaining_or_and",
            "target_idxs": [int(idx) for idx in target_idxs],
            "target_names": [guidance.label_names[idx] for idx in target_idxs],
            "remaining_names": [guidance.label_names[idx] for idx in remaining],
            "achieved_names": [guidance.label_names[idx] for idx in sorted(achieved)],
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_idx": selected_target_idx,
            "selected_target_name": guidance.label_names[selected_target_idx],
            "selected_remaining_scores": {
                guidance.label_names[idx]: float(remaining_score_columns[selected_idx, pos])
                for pos, idx in enumerate(remaining)
            },
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    action_sampler.done = False
    action_sampler.events = events
    action_sampler.violations = []
    return action_sampler


def make_chain_action_sampler(policy, guidance: AutomatonGuidance, target_names: Sequence[str], n_candidates: int):
    target_chain = target_idxs_from_names(guidance, target_names)
    state = {"pos": 0, "events": []}

    def sync(label, step):
        advanced = False
        while state["pos"] < len(target_chain) and float(label[target_chain[state["pos"]]]) > 0.5:
            target_idx = int(target_chain[state["pos"]])
            state["events"].append({"step": int(step), "target_idx": target_idx, "target_name": guidance.label_names[target_idx]})
            state["pos"] += 1
            advanced = True
        action_sampler.pos = state["pos"]
        action_sampler.events = state["events"]
        action_sampler.done = state["pos"] >= len(target_chain)
        return advanced

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        target_label_idx = int(target_chain[min(state["pos"], len(target_chain) - 1)])
        candidate_scores = guidance.score_label_probs(candidate_probs, target_label_idx)
        selected_idx = int(np.argmax(candidate_scores))
        opp = guidance.opposite_label_idx(target_label_idx)
        record = {
            "t": int(step),
            "mode": "chain",
            "chain_pos": int(state["pos"]),
            "chain_names": [guidance.label_names[idx] for idx in target_chain],
            "target_label_idx": target_label_idx,
            "target_label_name": guidance.label_names[target_label_idx],
            "opposite_label_idx": None if opp is None else int(opp),
            "opposite_label_name": None if opp is None else guidance.label_names[opp],
            "score_rule": guidance.score_rule_name(target_label_idx),
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_prob": float(candidate_probs[selected_idx, target_label_idx]),
            "selected_opposite_prob": None if opp is None else float(candidate_probs[selected_idx, opp]),
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    action_sampler.done = False
    action_sampler.events = state["events"]
    action_sampler.violations = []
    action_sampler.pos = 0
    return action_sampler


def make_ordered_stage_action_sampler(
    policy,
    guidance: AutomatonGuidance,
    stage_target_names: Sequence[Sequence[str]],
    n_candidates: int,
    guard_stage_idx: int = 1,
):
    stage_target_idxs = [target_idxs_from_names(guidance, stage) for stage in stage_target_names]
    achieved = [set() for _ in stage_target_idxs]
    events: list[Dict[str, Any]] = []
    violations: list[Dict[str, Any]] = []
    violation_keys: set[tuple[int, int]] = set()
    state = {"stage_pos": 0}

    def record_future_stage_violations(label, step):
        for future_stage_idx in range(max(int(guard_stage_idx), state["stage_pos"] + 1), len(stage_target_idxs)):
            for idx in stage_target_idxs[future_stage_idx]:
                key = (future_stage_idx, idx)
                if key not in violation_keys and float(label[idx]) > 0.5:
                    violation_keys.add(key)
                    violations.append(
                        {
                            "step": int(step),
                            "stage_idx": int(future_stage_idx),
                            "target_idx": int(idx),
                            "target_name": guidance.label_names[idx],
                            "message": "future-stage target became true before prior stages completed",
                        }
                    )

    def sync(label, step):
        advanced = False
        if state["stage_pos"] < len(stage_target_idxs):
            record_future_stage_violations(label, step)
        while state["stage_pos"] < len(stage_target_idxs):
            stage_idx = state["stage_pos"]
            for idx in stage_target_idxs[stage_idx]:
                if idx not in achieved[stage_idx] and float(label[idx]) > 0.5:
                    achieved[stage_idx].add(idx)
                    events.append(
                        {
                            "step": int(step),
                            "stage_idx": int(stage_idx),
                            "target_idx": int(idx),
                            "target_name": guidance.label_names[idx],
                        }
                    )
                    advanced = True
            if len(achieved[stage_idx]) == len(stage_target_idxs[stage_idx]):
                state["stage_pos"] += 1
                advanced = True
                continue
            break
        action_sampler.stage_pos = state["stage_pos"]
        action_sampler.events = events
        action_sampler.violations = violations
        action_sampler.done = state["stage_pos"] >= len(stage_target_idxs)
        return advanced

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        if state["stage_pos"] >= len(stage_target_idxs):
            current_stage = stage_target_idxs[-1]
            remaining = current_stage
        else:
            current_stage = stage_target_idxs[state["stage_pos"]]
            remaining = [idx for idx in current_stage if idx not in achieved[state["stage_pos"]]]
            if not remaining:
                remaining = current_stage
        remaining_score_columns = np.stack([guidance.score_label_probs(candidate_probs, idx) for idx in remaining], axis=1)
        candidate_scores = remaining_score_columns.max(axis=1)
        candidate_target_pos = remaining_score_columns.argmax(axis=1)
        selected_idx = int(np.argmax(candidate_scores))
        selected_target_idx = int(remaining[int(candidate_target_pos[selected_idx])])
        record = {
            "t": int(step),
            "mode": "ordered_stage_remaining_or",
            "stage_pos": int(state["stage_pos"]),
            "stage_target_names": [[guidance.label_names[idx] for idx in stage] for stage in stage_target_idxs],
            "current_stage_names": [guidance.label_names[idx] for idx in current_stage],
            "remaining_names": [guidance.label_names[idx] for idx in remaining],
            "achieved_by_stage": [[guidance.label_names[idx] for idx in sorted(stage)] for stage in achieved],
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_idx": selected_target_idx,
            "selected_target_name": guidance.label_names[selected_target_idx],
            "selected_remaining_scores": {
                guidance.label_names[idx]: float(remaining_score_columns[selected_idx, pos])
                for pos, idx in enumerate(remaining)
            },
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    action_sampler.done = False
    action_sampler.stage_pos = 0
    action_sampler.events = events
    action_sampler.violations = violations
    return action_sampler


def make_selection_then_target_action_sampler(
    policy,
    guidance: AutomatonGuidance,
    first_option_names: Sequence[str],
    middle_target_name: str,
    n_candidates: int,
):
    first_option_idxs = target_idxs_from_names(guidance, first_option_names)
    middle_idx = target_idxs_from_names(guidance, [middle_target_name])[0]
    state = {"phase": "first_or", "events": [], "last_selected_option_idx": None}

    def mark_event(step, role, target_idx):
        event = {
            "step": int(step),
            "role": role,
            "target_idx": int(target_idx),
            "target_name": guidance.label_names[int(target_idx)],
        }
        state["events"].append(event)
        return event

    def expose_state():
        action_sampler.phase = state["phase"]
        action_sampler.events = state["events"]
        action_sampler.done = state["phase"] == "done"

    def sync(label, step):
        advanced = False
        while state["phase"] != "done":
            if state["phase"] == "first_or":
                achieved = [idx for idx in first_option_idxs if float(label[idx]) > 0.5]
                if not achieved:
                    break
                last_selected = state.get("last_selected_option_idx")
                chosen = last_selected if last_selected in achieved else achieved[0]
                mark_event(step, "first_or", chosen)
                state["phase"] = "middle"
                advanced = True
                continue
            if state["phase"] == "middle":
                if float(label[middle_idx]) <= 0.5:
                    break
                mark_event(step, "middle", middle_idx)
                state["phase"] = "done"
                advanced = True
                continue
        expose_state()
        return advanced

    def current_target_idxs():
        if state["phase"] == "first_or":
            return list(first_option_idxs)
        return [middle_idx]

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        target_idxs = current_target_idxs()
        target_score_columns = np.stack([guidance.score_label_probs(candidate_probs, idx) for idx in target_idxs], axis=1)
        candidate_scores = target_score_columns.max(axis=1)
        candidate_target_pos = target_score_columns.argmax(axis=1)
        selected_idx = int(np.argmax(candidate_scores))
        selected_target_pos = int(candidate_target_pos[selected_idx])
        selected_target_idx = int(target_idxs[selected_target_pos])
        if state["phase"] == "first_or":
            state["last_selected_option_idx"] = selected_target_idx
        opp = guidance.opposite_label_idx(selected_target_idx)
        record = {
            "t": int(step),
            "mode": "selection_then_target",
            "phase": state["phase"],
            "first_option_names": list(first_option_names),
            "middle_target_name": middle_target_name,
            "target_names": [guidance.label_names[idx] for idx in target_idxs],
            "target_score_rules": [guidance.score_rule_name(idx) for idx in target_idxs],
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_idx": selected_target_idx,
            "selected_target_name": guidance.label_names[selected_target_idx],
            "selected_target_score": float(target_score_columns[selected_idx, selected_target_pos]),
            "selected_target_prob": float(candidate_probs[selected_idx, selected_target_idx]),
            "selected_opposite_idx": None if opp is None else int(opp),
            "selected_opposite_name": None if opp is None else guidance.label_names[opp],
            "selected_opposite_prob": None if opp is None else float(candidate_probs[selected_idx, opp]),
            "selected_target_scores": {
                guidance.label_names[idx]: float(target_score_columns[selected_idx, pos])
                for pos, idx in enumerate(target_idxs)
            },
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    expose_state()
    action_sampler.violations = []
    return action_sampler


def make_branch_remaining_action_sampler(
    policy,
    guidance: AutomatonGuidance,
    first_option_names: Sequence[str],
    middle_target_name: str,
    n_candidates: int,
):
    first_option_idxs = target_idxs_from_names(guidance, first_option_names)
    middle_idx = target_idxs_from_names(guidance, [middle_target_name])[0]
    state = {
        "phase": "first_or",
        "events": [],
        "first_choice_idx": None,
        "remaining_first_idxs": [],
        "last_selected_option_idx": None,
    }

    def mark_event(step, role, target_idx):
        event = {
            "step": int(step),
            "role": role,
            "target_idx": int(target_idx),
            "target_name": guidance.label_names[int(target_idx)],
        }
        state["events"].append(event)
        return event

    def expose_state():
        action_sampler.phase = state["phase"]
        action_sampler.events = state["events"]
        action_sampler.done = state["phase"] == "done"
        action_sampler.first_choice_name = None if state["first_choice_idx"] is None else guidance.label_names[state["first_choice_idx"]]
        action_sampler.remaining_first_name = None if not state["remaining_first_idxs"] else guidance.label_names[state["remaining_first_idxs"][0]]

    def sync(label, step):
        advanced = False
        while state["phase"] != "done":
            if state["phase"] == "first_or":
                achieved = [idx for idx in first_option_idxs if float(label[idx]) > 0.5]
                if not achieved:
                    break
                last_selected = state.get("last_selected_option_idx")
                chosen = last_selected if last_selected in achieved else achieved[0]
                state["first_choice_idx"] = int(chosen)
                state["remaining_first_idxs"] = [idx for idx in first_option_idxs if idx != chosen]
                mark_event(step, "first_or", chosen)
                state["phase"] = "middle"
                advanced = True
                continue
            if state["phase"] == "middle":
                if float(label[middle_idx]) <= 0.5:
                    break
                mark_event(step, "middle", middle_idx)
                state["phase"] = "remaining_first" if state["remaining_first_idxs"] else "done"
                advanced = True
                continue
            if state["phase"] == "remaining_first":
                current_remaining = state["remaining_first_idxs"][0]
                if float(label[current_remaining]) <= 0.5:
                    break
                mark_event(step, "remaining_first", current_remaining)
                state["remaining_first_idxs"].pop(0)
                state["phase"] = "remaining_first" if state["remaining_first_idxs"] else "done"
                advanced = True
                continue
        expose_state()
        return advanced

    def current_target_idxs():
        if state["phase"] == "first_or":
            return list(first_option_idxs)
        if state["phase"] == "middle":
            return [middle_idx]
        if state["phase"] == "remaining_first":
            return list(state["remaining_first_idxs"])
        return [middle_idx]

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        target_idxs = current_target_idxs()
        target_score_columns = np.stack([guidance.score_label_probs(candidate_probs, idx) for idx in target_idxs], axis=1)
        candidate_scores = target_score_columns.max(axis=1)
        candidate_target_pos = target_score_columns.argmax(axis=1)
        selected_idx = int(np.argmax(candidate_scores))
        selected_target_pos = int(candidate_target_pos[selected_idx])
        selected_target_idx = int(target_idxs[selected_target_pos])
        state["last_selected_option_idx"] = selected_target_idx if state["phase"] == "first_or" else state.get("last_selected_option_idx")
        selected_opp = guidance.opposite_label_idx(selected_target_idx)
        expose_state()
        record = {
            "t": int(step),
            "mode": "branch_remaining",
            "phase": state["phase"],
            "first_option_names": list(first_option_names),
            "middle_target_name": middle_target_name,
            "first_choice_name": action_sampler.first_choice_name,
            "remaining_first_name": action_sampler.remaining_first_name,
            "target_names": [guidance.label_names[idx] for idx in target_idxs],
            "target_score_rules": [guidance.score_rule_name(idx) for idx in target_idxs],
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_idx": selected_target_idx,
            "selected_target_name": guidance.label_names[selected_target_idx],
            "selected_target_score": float(target_score_columns[selected_idx, selected_target_pos]),
            "selected_target_prob": float(candidate_probs[selected_idx, selected_target_idx]),
            "selected_opposite_idx": None if selected_opp is None else int(selected_opp),
            "selected_opposite_name": None if selected_opp is None else guidance.label_names[selected_opp],
            "selected_opposite_prob": None if selected_opp is None else float(candidate_probs[selected_idx, selected_opp]),
            "selected_target_scores": {
                guidance.label_names[idx]: float(target_score_columns[selected_idx, pos])
                for pos, idx in enumerate(target_idxs)
            },
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    expose_state()
    action_sampler.violations = []
    return action_sampler


def make_cyclic_action_sampler(
    policy,
    guidance: AutomatonGuidance,
    target_names: Sequence[str],
    n_candidates: int,
    target_timeout_steps: int,
    max_target_events: int,
):
    target_cycle = target_idxs_from_names(guidance, target_names)
    state = {"pos": 0, "events": [], "target_start_step": 0, "cycles_completed": 0, "timed_out": False, "timeout_event": None}

    def refresh_public(step):
        action_sampler.cycle_pos = int(state["pos"])
        action_sampler.cycles_completed = int(state["cycles_completed"])
        action_sampler.events = state["events"]
        action_sampler.done = len(state["events"]) >= int(max_target_events)
        action_sampler.timed_out = bool(state["timed_out"])
        action_sampler.timeout_event = state["timeout_event"]
        action_sampler.current_target_idx = int(target_cycle[state["pos"]])
        action_sampler.current_target_name = guidance.label_names[target_cycle[state["pos"]]]
        action_sampler.target_elapsed = int(step - state["target_start_step"])

    def sync(label, step):
        advanced = False
        if state["timed_out"] or len(state["events"]) >= int(max_target_events):
            refresh_public(step)
            return False
        while len(state["events"]) < int(max_target_events):
            target_idx = target_cycle[state["pos"]]
            if float(label[target_idx]) <= 0.5:
                break
            elapsed = int(step - state["target_start_step"])
            state["events"].append(
                {
                    "step": int(step),
                    "elapsed": elapsed,
                    "event_idx": len(state["events"]),
                    "cycle_idx": int(state["cycles_completed"]),
                    "cycle_pos": int(state["pos"]),
                    "target_idx": int(target_idx),
                    "target_name": guidance.label_names[target_idx],
                }
            )
            state["pos"] = (state["pos"] + 1) % len(target_cycle)
            if state["pos"] == 0:
                state["cycles_completed"] += 1
            state["target_start_step"] = int(step)
            advanced = True
        if len(state["events"]) < int(max_target_events):
            elapsed = int(step - state["target_start_step"])
            if elapsed >= int(target_timeout_steps):
                target_idx = target_cycle[state["pos"]]
                state["timed_out"] = True
                state["timeout_event"] = {
                    "step": int(step),
                    "elapsed": elapsed,
                    "cycle_idx": int(state["cycles_completed"]),
                    "cycle_pos": int(state["pos"]),
                    "target_idx": int(target_idx),
                    "target_name": guidance.label_names[target_idx],
                    "message": f"target not reached within {int(target_timeout_steps)} steps",
                }
        refresh_public(step)
        return advanced

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        target_label_idx = target_cycle[state["pos"]]
        candidate_scores = guidance.score_label_probs(candidate_probs, target_label_idx)
        selected_idx = int(np.argmax(candidate_scores))
        opp = guidance.opposite_label_idx(target_label_idx)
        record = {
            "t": int(step),
            "mode": "cyclic_ltl",
            "cycle_names": [guidance.label_names[idx] for idx in target_cycle],
            "cycle_pos": int(state["pos"]),
            "cycle_idx": int(state["cycles_completed"]),
            "event_count": len(state["events"]),
            "target_elapsed": int(step - state["target_start_step"]),
            "target_label_idx": int(target_label_idx),
            "target_label_name": guidance.label_names[target_label_idx],
            "opposite_label_idx": None if opp is None else int(opp),
            "opposite_label_name": None if opp is None else guidance.label_names[opp],
            "score_rule": guidance.score_rule_name(target_label_idx),
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_prob": float(candidate_probs[selected_idx, target_label_idx]),
            "selected_opposite_prob": None if opp is None else float(candidate_probs[selected_idx, opp]),
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32), record

    action_sampler.sync = sync
    action_sampler.violations = []
    refresh_public(0)
    return action_sampler


def make_target_action_sampler(
    policy,
    guidance: AutomatonGuidance,
    spec: ComplexSTLSpec,
    target_name: str,
    n_candidates: int,
    dynamics_refiner: Optional[DynamicsRefiner],
    safety_kind: Optional[str],
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
    safety_guidance_scale: float,
    gripper_guidance_scale: float,
    gradient_steps: int,
    step_size: float,
    action_reg: float,
    smooth_min_tau: float,
    gripper_guidance_mode: str,
):
    target_label_idx = guidance.label_index(target_name)
    events: list[Dict[str, Any]] = []

    def sync(label, step):
        if not events and float(label[target_label_idx]) > 0.5:
            events.append({"step": int(step), "target_idx": int(target_label_idx), "target_name": target_name})
            action_sampler.done = True
            action_sampler.events = events
            return True
        return False

    def action_sampler(obs, env, step):
        action_chunks, candidate_probs, automaton_horizon, automaton_label = score_candidate_batch(
            policy, guidance, obs, env, n_candidates
        )
        sync(automaton_label, step)
        candidate_scores = guidance.score_label_probs(candidate_probs, target_label_idx)
        selected_idx = int(np.argmax(candidate_scores))
        selected = np.asarray(action_chunks[selected_idx, :automaton_horizon, :], dtype=np.float32)
        refinement_record: Dict[str, Any] = {"enabled": False}
        if dynamics_refiner is not None and safety_kind is not None:
            state = env.get_state()
            if safety_kind == "eef_avoid_box":
                selected, refinement_record = dynamics_refiner.refine_for_safety_box(
                    state["robot"],
                    state["scene"],
                    selected,
                    safety_box,
                    guidance_scale=safety_guidance_scale,
                    gradient_steps=gradient_steps,
                    step_size=step_size,
                    action_reg=action_reg,
                    smooth_min_tau=smooth_min_tau,
                )
            elif safety_kind == "gripper_open":
                selected, refinement_record = dynamics_refiner.refine_for_gripper_open(
                    state["robot"],
                    state["scene"],
                    selected,
                    gripper_spec,
                    guidance_scale=gripper_guidance_scale,
                    gradient_steps=gradient_steps,
                    step_size=step_size,
                    action_reg=action_reg,
                    smooth_min_tau=smooth_min_tau,
                    mode=gripper_guidance_mode,
                )
            elif safety_kind in {"tcp_rzz_30deg", "tcp_rzz_angle"}:
                selected, refinement_record = dynamics_refiner.refine_for_rzz(
                    state["robot"],
                    state["scene"],
                    selected,
                    spec.rzz_spec,
                    guidance_scale=safety_guidance_scale,
                    gradient_steps=gradient_steps,
                    step_size=step_size,
                    action_reg=action_reg,
                    action_dims=None,
                )
        opp = guidance.opposite_label_idx(target_label_idx)
        record = {
            "t": int(step),
            "mode": "target_with_optional_safety",
            "target_idx": int(target_label_idx),
            "target_name": target_name,
            "opposite_label_idx": None if opp is None else int(opp),
            "opposite_label_name": None if opp is None else guidance.label_names[opp],
            "score_rule": guidance.score_rule_name(target_label_idx),
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_prob": float(candidate_probs[selected_idx, target_label_idx]),
            "selected_opposite_prob": None if opp is None else float(candidate_probs[selected_idx, opp]),
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
            "safety_kind": safety_kind,
            "refinement": refinement_record,
        }
        return selected, record

    action_sampler.sync = sync
    action_sampler.done = False
    action_sampler.events = events
    action_sampler.violations = []
    return action_sampler


def make_action_sampler(
    spec: ComplexSTLSpec,
    policy,
    guidance: AutomatonGuidance,
    n_candidates: int,
    dynamics_refiner: Optional[DynamicsRefiner],
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
    args: argparse.Namespace,
):
    if spec.mode == "or":
        return make_or_action_sampler(policy, guidance, spec.target_names, n_candidates)
    if spec.mode == "and":
        return make_remaining_and_action_sampler(policy, guidance, spec.target_names, n_candidates)
    if spec.mode == "chain":
        return make_chain_action_sampler(policy, guidance, spec.target_names, n_candidates)
    if spec.mode == "ordered_stage":
        return make_ordered_stage_action_sampler(policy, guidance, spec.stage_target_names, n_candidates)
    if spec.mode == "selection_then_target":
        if spec.middle_target_name is None:
            raise ValueError(f"{spec.name} requires middle_target_name")
        return make_selection_then_target_action_sampler(
            policy, guidance, spec.first_option_names, spec.middle_target_name, n_candidates
        )
    if spec.mode == "branch_remaining":
        if spec.middle_target_name is None:
            raise ValueError(f"{spec.name} requires middle_target_name")
        return make_branch_remaining_action_sampler(
            policy, guidance, spec.first_option_names, spec.middle_target_name, n_candidates
        )
    if spec.mode == "cyclic":
        return make_cyclic_action_sampler(
            policy,
            guidance,
            spec.cycle_target_names,
            n_candidates,
            target_timeout_steps=int(spec.target_timeout_steps),
            max_target_events=int(spec.max_target_events),
        )
    if spec.mode == "target":
        safety_scale = float(spec.safety_guidance_scale if spec.safety_guidance_scale is not None else args.safety_guidance_scale)
        gripper_scale = float(spec.gripper_guidance_scale if spec.gripper_guidance_scale is not None else args.gripper_guidance_scale)
        gradient_steps = int(spec.gradient_steps if spec.gradient_steps is not None else args.gradient_steps)
        step_size = float(spec.step_size if spec.step_size is not None else args.step_size)
        action_reg = float(spec.action_reg if spec.action_reg is not None else args.action_reg)
        smooth_min_tau = float(spec.smooth_min_tau if spec.smooth_min_tau is not None else args.smooth_min_tau)
        return make_target_action_sampler(
            policy,
            guidance,
            spec,
            spec.target_names[0],
            n_candidates,
            dynamics_refiner=dynamics_refiner,
            safety_kind=None if args.disable_safety_refinement else spec.safety_kind,
            safety_box=safety_box,
            gripper_spec=gripper_spec,
            safety_guidance_scale=safety_scale,
            gripper_guidance_scale=gripper_scale,
            gradient_steps=gradient_steps,
            step_size=step_size,
            action_reg=action_reg,
            smooth_min_tau=smooth_min_tau,
            gripper_guidance_mode=spec.gripper_guidance_mode,
        )
    raise ValueError(f"Unsupported complex STL mode: {spec.mode}")


def rollout_policy_once(
    *,
    seed: int,
    policy,
    ckpt_dict: Dict[str, Any],
    spec: ComplexSTLSpec,
    action_sampler,
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
    safety_randomization: Optional[Dict[str, Any]],
    save_video: bool,
    settle_steps: int,
    settle_gripper: float,
    stop_on_env_done: bool,
    fps: int,
) -> Dict[str, Any]:
    CRU.seed_everything(seed)
    env, base_env_state = CRU.load_fresh_env_from_checkpoint(ckpt_dict, seed=int(seed), suppress_output=True)
    try:
        fixed_scene, fixed_robot, scene_cfg, robot_from_pose_file = make_fixed_scene_robot(
            base_env_state, scene_config_path, reset_poses, fixed_reset_pose_index
        )
        obs = CRU.reset_env_to_scene_robot(env, fixed_scene, fixed_robot)
        pre_settle_state = env.get_state()
        pre_settle_scene = np.asarray(pre_settle_state["scene"], dtype=np.float32).copy()
        pre_settle_robot = np.asarray(pre_settle_state["robot"], dtype=np.float32).copy()

        settle_action = idle_action_from_checkpoint(ckpt_dict, gripper=settle_gripper)
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

        policy.start_episode()
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
        _, label0 = action_sampler.guidance.current_state_and_label(env) if hasattr(action_sampler, "guidance") else (None, None)
        if label0 is None:
            raise RuntimeError("Action sampler is missing attached guidance.")
        pre_settle_label = label_scene_states_for_names(
            pre_settle_scene[None, :],
            action_sampler.guidance.label_names,
            action_sampler.guidance.label_thresholds,
        )[0].astype(np.float32)
        if hasattr(action_sampler, "sync"):
            action_sampler.sync(label0, 0)

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
                new_actions, record = action_sampler(obs, env, step)
                action_queue.extend(np.asarray(new_actions, dtype=np.float32))
                records.append(record)
            action = np.asarray(action_queue.pop(0), dtype=np.float32).copy()
            next_obs, reward, done, _ = env.step(action)
            total_reward += float(reward)

            state_now = env.get_state()
            scene_now = np.asarray(state_now["scene"], dtype=np.float32).copy()
            robot_now = np.asarray(state_now["robot"], dtype=np.float32).copy()
            _, label_now = action_sampler.guidance.current_state_and_label(env)

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

            if done and env_done_step < 0:
                env_done_step = int(step + 1)
                if stop_on_env_done:
                    termination_reason = "env_done"
                    break

            if hasattr(action_sampler, "sync"):
                advanced = action_sampler.sync(label_now, step + 1)
                if advanced:
                    action_queue.clear()
                if getattr(action_sampler, "timed_out", False):
                    termination_reason = "target_timeout"
                    break
                if action_sampler.done:
                    termination_reason = "task_complete"
                    break
            obs = next_obs

        labelf = labels_over_time[-1]
        target_events = list(getattr(action_sampler, "events", []))
        order_violations = list(getattr(action_sampler, "violations", []))
        completed_subgoals, total_subgoals, subgoal_rate = evaluate_subgoals(spec, target_events)
        robot_states_np = np.asarray(robot_states, dtype=np.float32)
        eef_xy_np = np.asarray(eef_xy, dtype=np.float32)
        safety_satisfied, safety_metrics, safety_distances = evaluate_safety(
            spec, robot_states_np, eef_xy_np, safety_box, gripper_spec
        )
        liveness_satisfied = bool(getattr(action_sampler, "done", False))
        order_violation = bool(order_violations)
        stl_satisfied = bool(liveness_satisfied and (safety_satisfied is not False) and not order_violation)
        behavior = "stl_satisfied" if stl_satisfied else "liveness_satisfied" if liveness_satisfied else first_behavior
        behavior_step = int(target_events[-1]["step"] if target_events else first_behavior_step)

        rollout = {
            "task": spec.name,
            "formula": spec.formula,
            "policy": "automaton_sample_rank",
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
            "gripper_width": robot_states_np[:, GRIPPER_WIDTH_RAW_ROBOT_IDX].astype(np.float32),
            "tcp_rzz": np.asarray(tcp_rzz, dtype=np.float32),
            "tcp_tilt_angle_deg": np.asarray(tcp_tilt_angle_deg, dtype=np.float32),
            "labels_over_time": np.asarray(labels_over_time, dtype=np.int32),
            "initial_label": labels_over_time[0].astype(int).tolist(),
            "final_label": labelf.astype(int).tolist(),
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
            "pre_settle_label": np.asarray(pre_settle_label, dtype=np.float32).astype(int).tolist(),
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
        write_rollout_rzz_angle_diagnostic_plot(rollout)
        rollout_dir = Path(rollout["rollout_dir"])
        CRU.plot_rollout_xy(
            [rollout],
            rollout["scene_snapshot"],
            f"{spec.name} seed {seed} | stl={stl_satisfied} subgoals={completed_subgoals}/{total_subgoals}",
            save_path=rollout_dir / "rollout_xy.png",
            display_inline=False,
        )
        write_json(rollout_dir / "records.json", {"records": records})
        write_json(rollout_dir / "rollout_summary.json", rollout_summary_payload(rollout))
        return rollout
    finally:
        CRU.close_env_quietly(env)


def attach_guidance(action_sampler, guidance: AutomatonGuidance):
    action_sampler.guidance = guidance
    return action_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-ckpt", type=Path, default=DEFAULT_COMPLEX_POLICY_CKPT)
    parser.add_argument("--automaton-ckpt", type=Path, default=DEFAULT_COMPLEX_AUTOMATON_CKPT)
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
        help=(
            "Optional subset of complex STL tasks to run. Omit this flag, or pass it with no values, "
            f"to run all tasks: {', '.join(TASK_ORDER)}."
        ),
    )
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=None, help="Override all task-specific candidate counts.")
    parser.add_argument("--horizon", type=int, default=None, help="Override all task-specific horizons.")
    parser.add_argument("--angle-goal-deg", type=float, default=None, help="Override the angle safety task target in degrees.")
    parser.add_argument("--angle-random-range-deg", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--angle-tolerance-deg", type=float, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', or a torch device like 'cuda:0'.")
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
    parser.add_argument(
        "--use-scene-config-robot",
        action="store_true",
        help="Use the robot pose from the scene config instead of sampling from the reset-pose pool.",
    )
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=DEFAULT_SETTLE_GRIPPER)
    parser.add_argument("--reset-robot-x-min", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_X_MIN)
    parser.add_argument("--reset-robot-x-max", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_X_MAX)
    parser.add_argument("--reset-robot-y-min", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_Y_MIN)
    parser.add_argument("--reset-robot-y-max", type=float, default=DEFAULT_COMPLEX_RESET_ROBOT_Y_MAX)
    parser.add_argument("--reset-switch-clearance", type=float, default=DEFAULT_RESET_SWITCH_CLEARANCE)
    parser.add_argument("--disable-reset-pose-filter", action="store_true")
    parser.add_argument("--disable-safety-refinement", action="store_true")
    parser.add_argument("--disable-safety-randomization", action="store_true")
    parser.add_argument("--safety-guidance-scale", type=float, default=0.5)
    parser.add_argument("--gripper-guidance-scale", type=float, default=10.0)
    parser.add_argument("--gradient-steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--action-reg", type=float, default=0.05)
    parser.add_argument("--smooth-min-tau", type=float, default=0.02)
    parser.add_argument("--safety-box", type=float, nargs=4, default=None, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--gripper-min-width", type=float, default=0.06)
    parser.add_argument("--gripper-margin", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--stop-on-env-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_by_name = dict(COMPLEX_STL_SPECS)
    if args.angle_goal_deg is not None:
        spec_by_name["angle"] = make_angle_stl_spec(float(args.angle_goal_deg))
    tasks = list(args.tasks) if args.tasks else list(TASK_ORDER)
    specs = [spec_by_name[task] for task in tasks]
    reset_robot_x_min = None if args.disable_reset_pose_filter else args.reset_robot_x_min
    reset_robot_x_max = None if args.disable_reset_pose_filter else args.reset_robot_x_max
    reset_robot_y_min = None if args.disable_reset_pose_filter else args.reset_robot_y_min
    reset_robot_y_max = None if args.disable_reset_pose_filter else args.reset_robot_y_max
    reset_switch_clearance = None if args.disable_reset_pose_filter else args.reset_switch_clearance

    policy_ckpt_candidate = repo_path(args.policy_ckpt)
    automaton_ckpt_candidate = repo_path(args.automaton_ckpt)
    dynamics_ckpt_candidate = repo_path(args.dynamics_ckpt)
    visualization_config = resolve_existing_path(repo_path(args.visualization_config))
    scene_config_args = {
        spec.name: (args.scene_config or spec.scene_config or DEFAULT_SCENE_CONFIG)
        for spec in specs
    }
    scene_config_paths = {
        spec.name: resolve_existing_path(repo_path(scene_config_args[spec.name]))
        for spec in specs
    }
    if args.use_scene_config_robot:
        reset_pose_paths = []
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
    run_name = args.name or f"complex_stls_rollouts{args.n_rollouts}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = unique_run_dir(repo_path(args.output_root), run_name)

    planned = {
        "policy": "automaton_sample_rank",
        "policy_ckpt": str(policy_ckpt_candidate),
        "policy_ckpt_exists": policy_ckpt_candidate.exists(),
        "automaton_ckpt": str(automaton_ckpt_candidate),
        "automaton_ckpt_exists": automaton_ckpt_candidate.exists(),
        "dynamics_ckpt": str(dynamics_ckpt_candidate),
        "dynamics_ckpt_exists": dynamics_ckpt_candidate.exists(),
        "scene_config": "override" if args.scene_config is not None else "per_task",
        "scene_configs": {spec.name: str(scene_config_paths[spec.name]) for spec in specs},
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "requested_name": args.name,
        "run_name": run_dir.name,
        "scene_config_arg": None if args.scene_config is None else str(args.scene_config),
        "tasks": tasks,
        "angle_goal_deg_override": None if args.angle_goal_deg is None else float(args.angle_goal_deg),
        "task_specs": [
            {
                "name": spec.name,
                "mode": spec.mode,
                "category": spec.category,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "stage_target_names": [list(stage) for stage in spec.stage_target_names],
                "first_option_names": list(spec.first_option_names),
                "middle_target_name": spec.middle_target_name,
                "cycle_target_names": list(spec.cycle_target_names),
                "safety_kind": spec.safety_kind,
                "scene_config": str(scene_config_paths[spec.name]),
                "horizon": int(args.horizon if args.horizon is not None else spec.default_horizon),
                "n_candidates": int(args.n_candidates if args.n_candidates is not None else spec.default_n_candidates),
                "target_timeout_steps": int(spec.target_timeout_steps),
                "max_target_events": int(spec.max_target_events),
                "safety_guidance_scale": spec.safety_guidance_scale,
                "gripper_guidance_scale": spec.gripper_guidance_scale,
                "gradient_steps": spec.gradient_steps,
                "step_size": spec.step_size,
                "action_reg": spec.action_reg,
                "smooth_min_tau": spec.smooth_min_tau,
                "gripper_guidance_mode": spec.gripper_guidance_mode,
                "rzz_spec": asdict(spec.rzz_spec),
                "rzz_init_mode": spec.rzz_init_mode,
                "rzz_warmup_max_steps": int(spec.rzz_warmup_max_steps),
                "rzz_warmup_tolerance": spec.rzz_warmup_tolerance,
                "restack_after_warmup": bool(spec.restack_after_warmup),
                "prompt": spec.prompt,
            }
            for spec in specs
        ],
        "n_rollouts": int(args.n_rollouts),
        "seed_start": int(args.seed_start),
        "device": args.device,
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
        "disable_safety_refinement": bool(args.disable_safety_refinement),
        "safety_randomization": safety_randomization_plan(
            safety_box,
            enabled=not bool(args.disable_safety_randomization),
            rzz_angle_deg_range=args.angle_random_range_deg,
            rzz_tolerance_deg=args.angle_tolerance_deg,
        ),
        "safety_box": asdict(safety_box.normalized()),
        "gripper_spec": asdict(gripper_spec.normalized()),
        "safety_guidance_scale": float(args.safety_guidance_scale),
        "gripper_guidance_scale": float(args.gripper_guidance_scale),
        "gradient_steps": int(args.gradient_steps),
        "step_size": float(args.step_size),
        "action_reg": float(args.action_reg),
        "smooth_min_tau": float(args.smooth_min_tau),
        "save_video": not args.no_video,
        "stop_on_env_done": bool(args.stop_on_env_done),
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return

    if not policy_ckpt_candidate.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_ckpt_candidate}")
    if not automaton_ckpt_candidate.exists():
        raise FileNotFoundError(f"Automaton checkpoint not found: {automaton_ckpt_candidate}")
    needs_safety_refiner = any(spec.safety_kind is not None for spec in specs) and not args.disable_safety_refinement
    if needs_safety_refiner and not dynamics_ckpt_candidate.exists():
        raise FileNotFoundError(f"Dynamics checkpoint not found: {dynamics_ckpt_candidate}")

    policy_ckpt = resolve_existing_path(policy_ckpt_candidate)
    automaton_ckpt = resolve_existing_path(automaton_ckpt_candidate)
    video_cfg = load_json(visualization_config)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True) if args.device == "auto" else torch.device(args.device)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(policy_ckpt), device=device, verbose=False)
    policy_epoch = CRU.policy_epoch_from_checkpoint(policy_ckpt)
    if policy_epoch == "epoch_unknown" and ckpt_dict.get("variable_state", {}).get("epoch") is not None:
        policy_epoch = f"epoch{int(ckpt_dict['variable_state']['epoch'])}"
    guidance = AutomatonGuidance(automaton_ckpt, device)
    dynamics_refiner = DynamicsRefiner(resolve_existing_path(dynamics_ckpt_candidate), device) if needs_safety_refiner else None

    for spec in specs:
        for target in spec.flattened_targets:
            guidance.label_index(target)

    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run_args.json", planned)
    print("device:", device)
    print("policy:", policy_ckpt)
    print("policy epoch:", policy_epoch)
    print("automaton:", guidance.meta["ckpt_path"])
    print("label order:", guidance.label_names)
    print("dynamics:", None if dynamics_refiner is None else dynamics_refiner.meta["checkpoint_path"])
    print("output:", run_dir)

    all_task_summaries = []
    for spec in specs:
        task_horizon = int(args.horizon if args.horizon is not None else spec.default_horizon)
        task_n_candidates = int(args.n_candidates if args.n_candidates is not None else spec.default_n_candidates)
        task_scene_config_path = scene_config_paths[spec.name]
        task_dir = run_dir / spec.name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            task_dir / "task_spec.json",
            {
                "name": spec.name,
                "mode": spec.mode,
                "category": spec.category,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "stage_target_names": [list(stage) for stage in spec.stage_target_names],
                "first_option_names": list(spec.first_option_names),
                "middle_target_name": spec.middle_target_name,
                "cycle_target_names": list(spec.cycle_target_names),
                "safety_kind": spec.safety_kind,
                "scene_config": str(task_scene_config_path),
                "horizon": task_horizon,
                "n_candidates": task_n_candidates,
                "target_timeout_steps": int(spec.target_timeout_steps),
                "max_target_events": int(spec.max_target_events),
                "safety_guidance_scale": spec.safety_guidance_scale,
                "gripper_guidance_scale": spec.gripper_guidance_scale,
                "gradient_steps": spec.gradient_steps,
                "step_size": spec.step_size,
                "action_reg": spec.action_reg,
                "smooth_min_tau": spec.smooth_min_tau,
                "gripper_guidance_mode": spec.gripper_guidance_mode,
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
            },
        )
        write_json(task_dir / "scene_config_resolved.json", load_json(task_scene_config_path))

        print(f"\nTask {spec.name}: {spec.formula}")
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
            action_sampler = make_action_sampler(
                rollout_spec,
                policy,
                guidance,
                task_n_candidates,
                dynamics_refiner,
                rollout_safety_box,
                gripper_spec,
                args,
            )
            attach_guidance(action_sampler, guidance)
            rollout = rollout_policy_once(
                seed=seed,
                policy=policy,
                ckpt_dict=ckpt_dict,
                spec=rollout_spec,
                action_sampler=action_sampler,
                scene_config_path=task_scene_config_path,
                reset_poses=reset_poses or [],
                reset_pose_filter=reset_pose_filter,
                fixed_reset_pose_index=fixed_reset_pose_index,
                output_dir=task_dir,
                rollout_tag=tag,
                horizon=task_horizon,
                video_cfg=video_cfg,
                safety_box=rollout_safety_box,
                gripper_spec=gripper_spec,
                safety_randomization=safety_randomization,
                save_video=not args.no_video,
                settle_steps=int(args.settle_steps),
                settle_gripper=float(args.settle_gripper),
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

        summary = task_summary(spec, rollouts, task_n_candidates, task_horizon)
        write_json(task_dir / "task_summary.json", summary)
        if rollouts:
            CRU.plot_rollout_xy(
                rollouts,
                rollouts[0]["scene_snapshot"],
                f"{spec.name} | STL {summary['stl_satisfied_count']}/{summary['n_rollouts']} | {spec.formula}",
                save_path=task_dir / "task_rollouts_xy.png",
                display_inline=False,
            )
            write_rzz_angle_diagnostic_plot(task_dir, spec, rollouts)
        all_task_summaries.append(summary)
        write_summary_tables(run_dir, all_task_summaries)
        write_json(
            run_dir / "summary_partial.json",
            {
                "run": planned,
                "policy_epoch": policy_epoch,
                "automaton": guidance.meta,
                "dynamics": None if dynamics_refiner is None else dynamics_refiner.meta,
                "completed_tasks": [item["task"] for item in all_task_summaries],
                "missing_tasks": [item.name for item in specs if item.name not in {s["task"] for s in all_task_summaries}],
                "tasks": all_task_summaries,
            },
        )

    write_json(
        run_dir / "summary.json",
        {
            "run": planned,
            "policy_epoch": policy_epoch,
            "automaton": guidance.meta,
            "dynamics": None if dynamics_refiner is None else dynamics_refiner.meta,
            "tasks": all_task_summaries,
        },
    )
    write_summary_tables(run_dir, all_task_summaries)
    print("\nSummary:", run_dir / "summary_table.md")


if __name__ == "__main__":
    main()
