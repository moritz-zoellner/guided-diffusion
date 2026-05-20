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
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
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
from calvin_experiments.label_calvin_world_model import label_scene_states_for_names
from calvin_experiments.run_dynaguide_articulated_automaton import (
    DEFAULT_RESET_ROBOT_Y_MIN,
    DEFAULT_RESET_SWITCH_CLEARANCE,
    DEFAULT_SETTLE_GRIPPER,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_VISUALIZATION_CONFIG,
    AutomatonGuidance,
    filter_reset_poses,
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


DEFAULT_SCENE_CONFIG = Path("calvin_experiments/configs/blocks_hidden.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/calvin_paper/complex-behaviors")
DEFAULT_COMPLEX_POLICY_CKPT = Path("outputs/calvin/base_policy/calvin_D_base_dp/20260501015147/models/model_epoch_280.pth")
DEFAULT_COMPLEX_AUTOMATON_CKPT = Path(
    "outputs/calvin/automaton_world_model/h8_sh64_ah96_lh16_hh128_lr0.0003_epochs80_2026-05-05_20-38-38"
)
DEFAULT_DYNAMICS_CKPT = Path(
    "outputs/calvin/dynamics_world_model/hd512_depth4_drop0.02_lr0.0005_epochs70_2026-05-06_01-42-08"
)
DEFAULT_RESET_POSE_DIR = Path("calvin_experiments/configs/dynaguide_articulated_objects/reset_poses")
DEFAULT_RESET_POSE_FILES = (
    DEFAULT_RESET_POSE_DIR / "initial_calvin_robot_states_midpoint.json",
    DEFAULT_RESET_POSE_DIR / "initial_calvin_robot_states_right_side_midpoint.json",
)
VIDEO_FPS = 30
GRIPPER_WIDTH_RAW_ROBOT_IDX = 6


@dataclass(frozen=True)
class SafetyBox:
    x_min: float = 0.225
    x_max: float = 0.275
    y_min: float = -0.125
    y_max: float = -0.075
    margin: float = 0.02

    def normalized(self) -> "SafetyBox":
        return SafetyBox(
            min(self.x_min, self.x_max),
            max(self.x_min, self.x_max),
            min(self.y_min, self.y_max),
            max(self.y_min, self.y_max),
            float(max(self.margin, 0.0)),
        )


@dataclass(frozen=True)
class GripperOpenSpec:
    min_width: float = 0.06
    margin: float = 0.02

    def normalized(self) -> "GripperOpenSpec":
        return GripperOpenSpec(float(self.min_width), float(max(self.margin, 0.0)))


@dataclass(frozen=True)
class ComplexSTLSpec:
    name: str
    mode: str
    formula: str
    target_names: tuple[str, ...] = ()
    stage_target_names: tuple[tuple[str, ...], ...] = ()
    safety_kind: Optional[str] = None
    default_horizon: int = 400
    default_n_candidates: int = 16
    prompt: str = ""

    @property
    def flattened_targets(self) -> tuple[str, ...]:
        if self.stage_target_names:
            return tuple(name for stage in self.stage_target_names for name in stage)
        return tuple(self.target_names)

    @property
    def required_subgoal_count(self) -> int:
        if self.mode == "or":
            return 1
        return len(self.flattened_targets)


COMPLEX_STL_SPECS: Dict[str, ComplexSTLSpec] = {
    "F_a_or_F_b": ComplexSTLSpec(
        name="F_a_or_F_b",
        mode="or",
        formula="F switch_on OR F button_on",
        target_names=("switch_on", "button_on"),
        default_horizon=250,
        default_n_candidates=16,
        prompt="turn on either the lightbulb with the switch or the LED with the button",
    ),
    "F_a_and_F_b": ComplexSTLSpec(
        name="F_a_and_F_b",
        mode="and",
        formula="F switch_on AND F button_on",
        target_names=("switch_on", "button_on"),
        default_horizon=350,
        default_n_candidates=16,
        prompt="turn on the lightbulb with the switch and turn on the LED with the button",
    ),
    "F_button_then_F_drawer": ComplexSTLSpec(
        name="F_button_then_F_drawer",
        mode="chain",
        formula="F(button_on -> F drawer_open -> F switch_on -> F button_pressed -> F door_left -> F drawer_closed)",
        target_names=("button_on", "drawer_open", "switch_on", "button_pressed", "door_left", "drawer_closed"),
        default_horizon=700,
        default_n_candidates=16,
        prompt=(
            "first turn on the LED with the button, then open the drawer, then turn on the lightbulb "
            "with the switch, then press the button, then move the sliding door left, and finally close the drawer"
        ),
    ),
    "F_drawer_after_button_switch": ComplexSTLSpec(
        name="F_drawer_after_button_switch",
        mode="ordered_stage",
        formula="F drawer_open AND (!drawer_open U (button_on AND switch_on))",
        stage_target_names=(("button_on", "switch_on"), ("drawer_open",)),
        default_horizon=500,
        default_n_candidates=16,
        prompt="turn on the LED with the button and turn on the lightbulb with the switch before opening the drawer",
    ),
    "F_drawer_G_constraint": ComplexSTLSpec(
        name="F_drawer_G_constraint",
        mode="target",
        formula="F drawer_open AND G gripper_open",
        target_names=("drawer_open",),
        safety_kind="gripper_open",
        default_horizon=250,
        default_n_candidates=16,
        prompt="open the drawer while keeping the gripper open",
    ),
    "F_switch_G_safety": ComplexSTLSpec(
        name="F_switch_G_safety",
        mode="target",
        formula="F switch_on AND G avoid_unsafe_square",
        target_names=("switch_on",),
        safety_kind="eef_avoid_box",
        default_horizon=220,
        default_n_candidates=16,
        prompt="turn on the lightbulb with the switch while keeping the robot arm out of the unsafe square",
    ),
}
TASK_ORDER = tuple(COMPLEX_STL_SPECS)


def unique_run_dir(output_root: Path, run_name: str) -> Path:
    candidate = output_root / run_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        suffixed = output_root / f"{run_name}_{suffix:02d}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def resolve_reset_pose_paths(raw_paths: Sequence[Path | str]) -> list[Path]:
    return [
        resolve_existing_path(path, base_dir=repo_path(DEFAULT_RESET_POSE_DIR))
        for path in raw_paths
    ]


def load_reset_pose_pool(paths: Sequence[Path]) -> tuple[list[np.ndarray], Dict[str, Any]]:
    poses: list[np.ndarray] = []
    sources = []
    for path in paths:
        payload = load_json(path)
        states = [np.asarray(item, dtype=np.float32) for item in payload["robot_states"]]
        poses.extend(states)
        sources.append({"path": str(path), "count": len(states)})
    return poses, {"sources": sources, "total_count": len(poses)}


def make_fixed_scene_robot(
    base_env_state: Dict[str, np.ndarray],
    scene_config_path: Path,
    reset_poses: Sequence[np.ndarray],
    fixed_reset_pose_index: Optional[int],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], bool]:
    fixed_scene, fixed_robot, scene_cfg = CRU.fixed_scene_robot_from_config(base_env_state, scene_config_path)
    if reset_poses:
        if fixed_reset_pose_index is None:
            selected_pose = random.choice(list(reset_poses))
        else:
            selected_pose = list(reset_poses)[int(fixed_reset_pose_index) % len(reset_poses)]
        fixed_robot = np.asarray(selected_pose, dtype=np.float32).copy()
        return fixed_scene, fixed_robot, scene_cfg, True
    return fixed_scene, fixed_robot, scene_cfg, False


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


def signed_distance_to_box_np(xy: np.ndarray, safety_box: SafetyBox) -> np.ndarray:
    box = safety_box.normalized()
    xy = np.asarray(xy, dtype=np.float32)
    center = np.asarray([(box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2], dtype=np.float32)
    half = np.asarray([(box.x_max - box.x_min) / 2, (box.y_max - box.y_min) / 2], dtype=np.float32)
    q = np.abs(xy - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside


def path_enters_safety_box(eef_xy: np.ndarray, safety_box: SafetyBox) -> bool:
    return bool(np.any(signed_distance_to_box_np(eef_xy, safety_box) <= 0.0))


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
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        state_dyn = self.raw_env_state_to_dynamics_state_torch(robot, scene)
        original = torch.as_tensor(action_chunk[None], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            robust_before, width_before = self.gripper_open_robustness(state_dyn, original, spec, smooth_min_tau)
        if float(guidance_scale) <= 0.0 or int(gradient_steps) <= 0:
            record = {
                "enabled": False,
                "robust_before": float(robust_before[0].cpu()),
                "robust_after": float(robust_before[0].cpu()),
                "min_width_before": float(width_before[0].min().cpu()),
                "min_width_after": float(width_before[0].min().cpu()),
                "pred_width_before": width_before[0].detach().cpu().numpy().tolist(),
                "pred_width_after": width_before[0].detach().cpu().numpy().tolist(),
                "action_l2_change": 0.0,
                "history": [],
            }
            return np.asarray(action_chunk, dtype=np.float32), record

        actions = original.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=float(step_size))
        history = []
        for _ in range(int(gradient_steps)):
            opt.zero_grad(set_to_none=True)
            robust, width = self.gripper_open_robustness(state_dyn, actions, spec, smooth_min_tau)
            action_penalty = torch.mean((actions - original) ** 2, dim=(1, 2))
            objective = float(guidance_scale) * robust - float(action_reg) * action_penalty
            (-objective.mean()).backward()
            opt.step()
            with torch.no_grad():
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
            robust_after, width_after = self.gripper_open_robustness(state_dyn, actions, spec, smooth_min_tau)
        record = {
            "enabled": True,
            "robust_before": float(robust_before[0].cpu()),
            "robust_after": float(robust_after[0].cpu()),
            "min_width_before": float(width_before[0].min().cpu()),
            "min_width_after": float(width_after[0].min().cpu()),
            "pred_width_before": width_before[0].detach().cpu().numpy().tolist(),
            "pred_width_after": width_after[0].detach().cpu().numpy().tolist(),
            "action_l2_change": float(torch.linalg.norm(actions - original).detach().cpu()),
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


def make_target_action_sampler(
    policy,
    guidance: AutomatonGuidance,
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
    if spec.mode == "target":
        return make_target_action_sampler(
            policy,
            guidance,
            spec.target_names[0],
            n_candidates,
            dynamics_refiner=dynamics_refiner,
            safety_kind=None if args.disable_safety_refinement else spec.safety_kind,
            safety_box=safety_box,
            gripper_spec=gripper_spec,
            safety_guidance_scale=float(args.safety_guidance_scale),
            gripper_guidance_scale=float(args.gripper_guidance_scale),
            gradient_steps=int(args.gradient_steps),
            step_size=float(args.step_size),
            action_reg=float(args.action_reg),
            smooth_min_tau=float(args.smooth_min_tau),
        )
    raise ValueError(f"Unsupported complex STL mode: {spec.mode}")


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
        "rollout_seed": np.asarray(rollout["seed"], dtype=np.int32),
        "scene_config": np.asarray(rollout["scene_config"]),
        "initial_label": np.asarray(rollout["initial_label"], dtype=np.int32),
        "final_label": np.asarray(rollout["final_label"], dtype=np.int32),
        "labels_over_time": np.asarray(rollout["labels_over_time"], dtype=np.int32),
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


def save_rollout_diagnostics(rollout: Dict[str, Any]) -> None:
    rollout_dir = Path(rollout["rollout_dir"])
    diagnostics_path = rollout_dir / "diagnostics.npz"
    diagnostics = {
        "labels_over_time": np.asarray(rollout["labels_over_time"], dtype=np.int32),
        "gripper_width": np.asarray(rollout.get("gripper_width", []), dtype=np.float32),
    }
    if rollout.get("safety_distances") is not None:
        diagnostics["safety_distances"] = np.asarray(rollout["safety_distances"], dtype=np.float32)
    np.savez_compressed(diagnostics_path, **diagnostics)
    rollout["diagnostics"] = diagnostics_path


def rollout_summary_payload(rollout: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task": rollout["task"],
        "formula": rollout["formula"],
        "seed": int(rollout["seed"]),
        "policy": "automaton_sample_rank",
        "scene_config": rollout["scene_config"],
        "liveness_satisfied": bool(rollout["liveness_satisfied"]),
        "safety_satisfied": rollout["safety_satisfied"],
        "stl_satisfied": bool(rollout["stl_satisfied"]),
        "success": bool(rollout["success"]),
        "subgoal_completion_rate": float(rollout["subgoal_completion_rate"]),
        "completed_subgoals": int(rollout["completed_subgoals"]),
        "total_subgoals": int(rollout["total_subgoals"]),
        "target_events": rollout["target_events"],
        "order_violation": bool(rollout.get("order_violation", False)),
        "order_violations": rollout.get("order_violations", []),
        "safety_kind": rollout.get("safety_kind"),
        "safety_metrics": rollout.get("safety_metrics", {}),
        "behavior": rollout["behavior"],
        "first_behavior": rollout["first_behavior"],
        "first_behavior_step": int(rollout["first_behavior_step"]),
        "behavior_step": int(rollout["behavior_step"]),
        "termination_step": int(rollout["termination_step"]),
        "termination_reason": rollout["termination_reason"],
        "env_done_step": int(rollout["env_done_step"]),
        "return": float(rollout["return"]),
        "initial_label": rollout["initial_label"],
        "final_label": rollout["final_label"],
        "video": None if rollout.get("video") is None else str(rollout["video"]),
        "trace": str(rollout["trace"]),
        "diagnostics": str(rollout["diagnostics"]),
        "topdown_plot": str(Path(rollout["rollout_dir"]) / "rollout_xy.png"),
        "reset_robot_from_pose_file": bool(rollout["reset_robot_from_pose_file"]),
        "fixed_reset_pose_index": rollout.get("fixed_reset_pose_index"),
        "reset_pose_filter": rollout["reset_pose_filter"],
        "settle_steps": int(rollout["settle_steps"]),
        "settle_action": rollout["settle_action"],
        "pre_settle_label": rollout["pre_settle_label"],
        "settle_metrics": rollout["settle_metrics"],
        "records": rollout["records"],
    }


def evaluate_subgoals(spec: ComplexSTLSpec, events: Sequence[Dict[str, Any]]) -> tuple[int, int, float]:
    total = spec.required_subgoal_count
    if spec.mode == "or":
        completed = 1 if events else 0
    else:
        target_names = set(spec.flattened_targets)
        completed = len({event["target_name"] for event in events if event["target_name"] in target_names})
    rate = float(completed / total) if total else 0.0
    return int(completed), int(total), rate


def evaluate_safety(
    spec: ComplexSTLSpec,
    robot_states: np.ndarray,
    eef_xy: np.ndarray,
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
) -> tuple[Optional[bool], Dict[str, Any], Optional[np.ndarray]]:
    if spec.safety_kind == "eef_avoid_box":
        distances = signed_distance_to_box_np(eef_xy, safety_box)
        violation = bool(np.any(distances <= 0.0))
        metrics = {
            "kind": "eef_avoid_box",
            "safety_box": asdict(safety_box.normalized()),
            "violation": violation,
            "min_signed_distance": float(np.min(distances)),
        }
        return (not violation), metrics, distances
    if spec.safety_kind == "gripper_open":
        widths = np.asarray(robot_states, dtype=np.float32)[:, GRIPPER_WIDTH_RAW_ROBOT_IDX]
        spec_norm = gripper_spec.normalized()
        violation = bool(np.any(widths < spec_norm.min_width))
        metrics = {
            "kind": "gripper_open",
            "gripper_spec": asdict(spec_norm),
            "violation": violation,
            "min_gripper_width": float(np.min(widths)),
        }
        return (not violation), metrics, None
    return None, {}, None


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

        policy.start_episode()
        scene_snapshot = CRU.capture_scene_snapshot(env)
        frames = [CRU.render_visual_camera(env, video_cfg)] if save_video else []

        start_state = env.get_state()
        start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
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
        robot_states = [np.asarray(start_state["robot"], dtype=np.float32).copy()]
        labels_over_time = [label0.astype(int).copy()]
        eef_xy = [robot_states[-1][:2].copy()]
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
            "scene_snapshot": scene_snapshot,
            "reset_robot_from_pose_file": bool(robot_from_pose_file),
            "fixed_reset_pose_index": None if fixed_reset_pose_index is None else int(fixed_reset_pose_index),
            "reset_pose_filter": reset_pose_filter,
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


def task_summary(spec: ComplexSTLSpec, rollouts: Sequence[Dict[str, Any]], n_candidates: int, horizon: int) -> Dict[str, Any]:
    behavior_counts = Counter(rollout["behavior"] for rollout in rollouts)
    first_behavior_counts = Counter(rollout["first_behavior"] for rollout in rollouts)
    event_patterns = Counter(tuple(event["target_name"] for event in rollout["target_events"]) for rollout in rollouts)
    liveness_count = sum(1 for rollout in rollouts if rollout["liveness_satisfied"])
    stl_count = sum(1 for rollout in rollouts if rollout["stl_satisfied"])
    safety_values = [rollout["safety_satisfied"] for rollout in rollouts if rollout["safety_satisfied"] is not None]
    safety_count = sum(1 for value in safety_values if value)
    order_violation_count = sum(1 for rollout in rollouts if rollout.get("order_violation", False))
    return {
        "task": spec.name,
        "formula": spec.formula,
        "mode": spec.mode,
        "safety_kind": spec.safety_kind,
        "n_rollouts": len(rollouts),
        "n_candidates": int(n_candidates),
        "horizon": int(horizon),
        "liveness_satisfied_count": int(liveness_count),
        "liveness_satisfaction_rate": float(liveness_count / len(rollouts)) if rollouts else 0.0,
        "safety_satisfied_count": None if not safety_values else int(safety_count),
        "safety_satisfaction_rate": None if not safety_values else float(safety_count / len(safety_values)),
        "stl_satisfied_count": int(stl_count),
        "stl_satisfaction_rate": float(stl_count / len(rollouts)) if rollouts else 0.0,
        "subgoal_completion_rate": float(np.mean([rollout["subgoal_completion_rate"] for rollout in rollouts])) if rollouts else 0.0,
        "order_violation_count": int(order_violation_count),
        "order_violation_rate": float(order_violation_count / len(rollouts)) if rollouts else 0.0,
        "avg_termination_step": float(np.mean([rollout["termination_step"] for rollout in rollouts])) if rollouts else 0.0,
        "behavior_counts": dict(behavior_counts),
        "first_behavior_counts": dict(first_behavior_counts),
        "event_count_patterns": {
            " -> ".join(pattern) if pattern else "none": count
            for pattern, count in event_patterns.items()
        },
        "rollouts": [
            {
                "seed": rollout["seed"],
                "liveness_satisfied": bool(rollout["liveness_satisfied"]),
                "safety_satisfied": rollout["safety_satisfied"],
                "stl_satisfied": bool(rollout["stl_satisfied"]),
                "subgoal_completion_rate": float(rollout["subgoal_completion_rate"]),
                "completed_subgoals": int(rollout["completed_subgoals"]),
                "total_subgoals": int(rollout["total_subgoals"]),
                "target_events": rollout["target_events"],
                "order_violation": bool(rollout.get("order_violation", False)),
                "safety_metrics": rollout.get("safety_metrics", {}),
                "behavior": rollout["behavior"],
                "first_behavior": rollout["first_behavior"],
                "termination_step": rollout["termination_step"],
                "termination_reason": rollout["termination_reason"],
                "env_done_step": rollout["env_done_step"],
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": None if rollout.get("video") is None else str(rollout["video"]),
                "trace": str(rollout.get("trace")),
                "diagnostics": str(rollout.get("diagnostics")),
                "topdown_plot": str(Path(rollout.get("rollout_dir", "")) / "rollout_xy.png"),
            }
            for rollout in rollouts
        ],
    }


def write_summary_tables(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "mode",
        "formula",
        "n_rollouts",
        "n_candidates",
        "horizon",
        "liveness_satisfaction_rate",
        "safety_satisfaction_rate",
        "subgoal_completion_rate",
        "stl_satisfaction_rate",
        "order_violation_rate",
        "avg_termination_step",
        "event_count_patterns",
    ]
    rows = []
    for item in summaries:
        safety_rate = item["safety_satisfaction_rate"]
        rows.append(
            {
                "task": item["task"],
                "mode": item["mode"],
                "formula": item["formula"],
                "n_rollouts": item["n_rollouts"],
                "n_candidates": item["n_candidates"],
                "horizon": item["horizon"],
                "liveness_satisfaction_rate": f"{item['liveness_satisfaction_rate']:.4f}",
                "safety_satisfaction_rate": "" if safety_rate is None else f"{safety_rate:.4f}",
                "subgoal_completion_rate": f"{item['subgoal_completion_rate']:.4f}",
                "stl_satisfaction_rate": f"{item['stl_satisfaction_rate']:.4f}",
                "order_violation_rate": f"{item['order_violation_rate']:.4f}",
                "avg_termination_step": f"{item['avg_termination_step']:.2f}",
                "event_count_patterns": json.dumps(item["event_count_patterns"], sort_keys=True),
            }
        )
    with (run_dir / "summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "summary_table.md").open("w") as f:
        f.write("| task | mode | liveness | safety | subgoal | STL | order violations | n | horizon | events |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['task']} | {row['mode']} | {row['liveness_satisfaction_rate']} | "
                f"{row['safety_satisfaction_rate']} | {row['subgoal_completion_rate']} | "
                f"{row['stl_satisfaction_rate']} | {row['order_violation_rate']} | "
                f"{row['n_rollouts']} | {row['horizon']} | `{row['event_count_patterns']}` |\n"
            )
    write_summary_table_image(run_dir, rows)


def _rate_cell(value: str) -> str:
    if value is None or value == "":
        return ""
    return f"{100.0 * float(value):.0f}%"


def _count_events(value: str) -> str:
    try:
        events = json.loads(value)
    except Exception:
        return value
    compact = []
    for name, count in sorted(events.items(), key=lambda item: (-int(item[1]), item[0])):
        if len(name) > 42:
            name = name[:39] + "..."
        compact.append(f"{name}: {count}")
    return "\n".join(compact[:3])


def _task_label(task: str) -> str:
    labels = {
        "F_a_or_F_b": "F a OR F b",
        "F_a_and_F_b": "F a AND F b",
        "F_button_then_F_drawer": "6-step chain",
        "F_drawer_after_button_switch": "drawer after\nbutton+switch",
        "F_drawer_G_constraint": "drawer +\ngripper safe",
        "F_switch_G_safety": "switch +\navoid box",
    }
    return labels.get(task, task)


def _color_for_rate(text: str, *, invert: bool = False) -> str:
    if not text:
        return "#f8fafc"
    value = float(text.rstrip("%")) / 100.0
    if invert:
        value = 1.0 - value
    if value >= 0.8:
        return "#dcfce7"
    if value >= 0.5:
        return "#fef9c3"
    return "#fee2e2"


def write_summary_table_image(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt

    core_headers = ["Task", "Mode", "Live", "Subgoal", "STL", "Order\nviol.", "Avg\nsteps", "Top events"]
    core_rows = [
        [
            _task_label(row["task"]),
            row["mode"],
            _rate_cell(row["liveness_satisfaction_rate"]),
            _rate_cell(row["subgoal_completion_rate"]),
            _rate_cell(row["stl_satisfaction_rate"]),
            _rate_cell(row["order_violation_rate"]),
            row["avg_termination_step"],
            _count_events(row["event_count_patterns"]),
        ]
        for row in rows
    ]
    safety_rows = [
        [_task_label(row["task"]), _rate_cell(row["safety_satisfaction_rate"])]
        for row in rows
        if row["safety_satisfaction_rate"] != ""
    ]

    fig_height = 1.4 + 0.58 * len(core_rows) + (0.9 + 0.28 * len(safety_rows) if safety_rows else 0.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    ax.axis("off")
    title = f"Complex STL Summary | {rows[0]['n_rollouts']} rollouts"
    ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")

    core_table = ax.table(
        cellText=core_rows,
        colLabels=core_headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.09, 0.07, 0.08, 0.07, 0.08, 0.07, 0.38],
        bbox=[0.0, 0.32 if safety_rows else 0.0, 1.0, 0.65 if safety_rows else 0.92],
    )
    core_table.auto_set_font_size(False)
    core_table.set_fontsize(9)
    for (r, c), cell in core_table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
            if c in (2, 3, 4):
                cell.set_facecolor(_color_for_rate(core_rows[r - 1][c]))
            if c == 5:
                cell.set_facecolor(_color_for_rate(core_rows[r - 1][c], invert=True))
            if c in (0, 7):
                cell.get_text().set_ha("left")

    if safety_rows:
        ax.text(0.0, 0.22, "Safety metrics (only tasks where safety is defined)", transform=ax.transAxes, fontsize=11, fontweight="bold")
        safety_table = ax.table(
            cellText=safety_rows,
            colLabels=["Task", "Safety"],
            cellLoc="center",
            colLoc="center",
            colWidths=[0.3, 0.12],
            bbox=[0.0, 0.0, 0.42, 0.20],
        )
        safety_table.auto_set_font_size(False)
        safety_table.set_fontsize(9)
        for (r, c), cell in safety_table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#334155")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
                if c == 1:
                    cell.set_facecolor(_color_for_rate(safety_rows[r - 1][1]))
                if c == 0:
                    cell.get_text().set_ha("left")

    out_path = run_dir / "summary_table.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-ckpt", type=Path, default=DEFAULT_COMPLEX_POLICY_CKPT)
    parser.add_argument("--automaton-ckpt", type=Path, default=DEFAULT_COMPLEX_AUTOMATON_CKPT)
    parser.add_argument("--dynamics-ckpt", type=Path, default=DEFAULT_DYNAMICS_CKPT)
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
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=None, help="Override all task-specific candidate counts.")
    parser.add_argument("--horizon", type=int, default=None, help="Override all task-specific horizons.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', or a torch device like 'cuda:0'.")
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
    parser.add_argument("--disable-safety-refinement", action="store_true")
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
    tasks = list(args.tasks) if args.tasks else list(TASK_ORDER)
    specs = [COMPLEX_STL_SPECS[task] for task in tasks]
    reset_robot_y_min = None if args.disable_reset_pose_filter else args.reset_robot_y_min
    reset_robot_y_max = None if args.disable_reset_pose_filter else args.reset_robot_y_max
    reset_switch_clearance = None if args.disable_reset_pose_filter else args.reset_switch_clearance

    policy_ckpt_candidate = repo_path(args.policy_ckpt)
    automaton_ckpt_candidate = repo_path(args.automaton_ckpt)
    dynamics_ckpt_candidate = repo_path(args.dynamics_ckpt)
    scene_config_path = resolve_existing_path(repo_path(args.scene_config))
    visualization_config = resolve_existing_path(repo_path(args.visualization_config))
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
        "scene_config": str(scene_config_path),
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "requested_name": args.name,
        "run_name": run_dir.name,
        "tasks": tasks,
        "task_specs": [
            {
                "name": spec.name,
                "mode": spec.mode,
                "formula": spec.formula,
                "target_names": list(spec.target_names),
                "stage_target_names": [list(stage) for stage in spec.stage_target_names],
                "safety_kind": spec.safety_kind,
                "horizon": int(args.horizon if args.horizon is not None else spec.default_horizon),
                "n_candidates": int(args.n_candidates if args.n_candidates is not None else spec.default_n_candidates),
                "prompt": spec.prompt,
            }
            for spec in specs
        ],
        "n_rollouts": int(args.n_rollouts),
        "seed_start": int(args.seed_start),
        "device": args.device,
        "reset_pose_filter": reset_pose_filter,
        "reset_pose_selection": "sample_per_rollout_seed" if fixed_reset_pose_index is None else "fixed_index",
        "fixed_reset_pose_index": fixed_reset_pose_index,
        "settle_steps": int(args.settle_steps),
        "settle_gripper": float(args.settle_gripper),
        "disable_safety_refinement": bool(args.disable_safety_refinement),
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
                "n_candidates": task_n_candidates,
            },
        )
        write_json(task_dir / "scene_config_resolved.json", load_json(scene_config_path))

        print(f"\nTask {spec.name}: {spec.formula}")
        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            action_sampler = make_action_sampler(
                spec,
                policy,
                guidance,
                task_n_candidates,
                dynamics_refiner,
                safety_box,
                gripper_spec,
                args,
            )
            attach_guidance(action_sampler, guidance)
            rollout = rollout_policy_once(
                seed=seed,
                policy=policy,
                ckpt_dict=ckpt_dict,
                spec=spec,
                action_sampler=action_sampler,
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
        all_task_summaries.append(summary)

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
