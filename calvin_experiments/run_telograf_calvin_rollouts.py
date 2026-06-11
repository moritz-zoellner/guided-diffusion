#!/usr/bin/env python3
"""Open-loop TeLoGraF rollouts on DynaGuide-style CALVIN articulated tasks.

This script reuses the CALVIN environment reset, behavior detection, and
artifact utilities from the existing articulated-object rollout scripts. The
only policy source is a trained TeLoGraF GaussianFlow checkpoint: one H-step
trajectory is sampled at rollout start, then its actions are executed open-loop.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/scratch/gilbreth/zoellner/guided-diffusion/outputs/telograf/.cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/scratch/gilbreth/zoellner/guided-diffusion/outputs/telograf/.cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import h5py
import numpy as np
import torch


def find_repo_root(start: Path | str | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start)
    for path in (start_path, *start_path.parents):
        if (path / "calvin_experiments" / "calvin_rollout_utils.py").exists():
            return path
    raise FileNotFoundError(f"Could not find guided-diffusion repo root from {start_path}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SCRATCH_ROOT = Path("/scratch/gilbreth/zoellner/guided-diffusion")
DEFAULT_TELOGRAF_CHECKPOINT = (
    SCRATCH_ROOT
    / "outputs/telograf/calvin/runs/calvin_play_eventual_h64_telograf/checkpoint.pt"
)
DEFAULT_ENV_CHECKPOINT = (
    REPO_ROOT
    / "outputs/calvin/base_policy/calvin_D_base_dp/20260501015147/models/model_epoch_280.pth"
)
DEFAULT_ENV_DATASET = SCRATCH_ROOT / "data/calvin.hdf5"
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / "outputs/telograf/calvin_rollouts"
DEFAULT_CONFIG_DIR = REPO_ROOT / "calvin_experiments/configs/dynaguide_articulated_objects"
DEFAULT_VISUALIZATION_CONFIG = REPO_ROOT / "calvin_experiments/configs/visualization_freiburg_style.json"
DEFAULT_RESET_ROBOT_Y_MIN = -0.25
DEFAULT_RESET_SWITCH_CLEARANCE = 0.01
DEFAULT_SETTLE_STEPS = 10
DEFAULT_SETTLE_GRIPPER = 1.0
VIDEO_FPS = 30

for path in [
    REPO_ROOT,
    REPO_ROOT / "robomimic",
    REPO_ROOT / "calvin" / "calvin_env",
    REPO_ROOT / "calvin_experiments",
    REPO_ROOT / "TeLoGraF" / "code",
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import robomimic.envs  # noqa: F401
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from z_diffuser import GaussianFlow, TemporalUnet

from calvin_experiments import calvin_rollout_utils as CRU
from calvin_experiments.label_calvin_world_model import (
    LABEL_NAMES,
    LABEL_THRESHOLDS,
    label_scene_states_for_names,
)
from calvin_experiments.run_dynaguide_articulated_automaton import (
    dynaguide_scene_from_base,
    filter_reset_poses,
    load_json,
    load_reset_poses,
    settle_metrics,
)
from telograf_calvin.paper_specs import evaluate_spec_sequence, make_spec, spec_to_vector


TARGET_LABELS = [
    "switch_on",
    "switch_off",
    "button_on",
    "button_off",
    "button_pressed",
    "drawer_open",
    "drawer_closed",
    "door_left",
    "door_right",
]

LABEL_TO_RESET_TASK = {
    "switch_on": "switch_on",
    "switch_off": "switch_off",
    "button_on": "button_on",
    "button_off": "button_off",
    "button_pressed": "button_on",
    "drawer_open": "drawer_open",
    "drawer_closed": "drawer_close",
    "door_left": "door_left",
    "door_right": "door_right",
}

LABEL_TO_ACCEPTABLE_BEHAVIORS = {
    "switch_on": ("switch_on",),
    "switch_off": ("switch_off",),
    "button_on": ("button_on", "button_off"),
    "button_off": ("button_off", "button_on"),
    "button_pressed": ("button_on", "button_off"),
    "drawer_open": ("drawer_open",),
    "drawer_closed": ("drawer_close",),
    "door_left": ("door_left",),
    "door_right": ("door_right",),
}


def repo_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_existing_path(path: Path | str) -> Path:
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path.resolve()


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def format_onehot(values: Sequence[int | float]) -> str:
    return "[" + " ".join(f"{int(v):1d}" for v in values) + "]"


def labels_for_scene(scene: Sequence[float]) -> np.ndarray:
    return label_scene_states_for_names(
        np.asarray(scene, dtype=np.float32)[None, :],
        LABEL_NAMES,
        LABEL_THRESHOLDS,
    )[0].astype(np.float32)


def load_env_metadata_from_dataset(dataset_path: Path) -> Dict[str, Any]:
    with h5py.File(dataset_path, "r") as h5:
        env_args = h5["data"].attrs.get("env_args")
    if env_args is None:
        raise KeyError(f"No data.attrs['env_args'] found in {dataset_path}")
    if isinstance(env_args, bytes):
        env_args = env_args.decode("utf-8")
    env_meta = json.loads(str(env_args))
    env_meta.setdefault("env_kwargs", {})["use_egl"] = False
    return env_meta


def load_fresh_env_from_metadata(
    env_meta: Dict[str, Any],
    seed: Optional[int] = None,
    existing_env: Any = None,
    render_offscreen: bool = False,
):
    CRU.close_env_quietly(existing_env)
    CRU.seed_everything(seed)

    def _load():
        return EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=render_offscreen,
            use_image_obs=False,
            use_depth_obs=False,
        )

    try:
        with CRU.suppress_native_output():
            env = _load()
    except Exception as exc:
        print(f"CALVIN env creation from dataset metadata failed: {exc!r}", flush=True)
        env = _load()
    if not CRU.is_env_connected(env):
        raise RuntimeError("Fresh CALVIN env is disconnected immediately after metadata env creation.")
    base_env_state = {key: np.asarray(value, dtype=np.float32).copy() for key, value in env.get_state().items()}
    return env, base_env_state


class CalvinEnvFactory:
    def __init__(
        self,
        *,
        ckpt_dict: Optional[Dict[str, Any]] = None,
        env_meta: Optional[Dict[str, Any]] = None,
        action_dim: int = 7,
    ):
        if ckpt_dict is None and env_meta is None:
            raise ValueError("Need either ckpt_dict or env_meta")
        self.ckpt_dict = ckpt_dict
        self.env_meta = env_meta
        self.action_dim = int(action_dim)

    @property
    def source(self) -> str:
        return "checkpoint" if self.ckpt_dict is not None else "dataset_env_metadata"

    def fresh(self, seed: int):
        if self.ckpt_dict is not None:
            env, base_state = CRU.load_fresh_env_from_checkpoint(
                self.ckpt_dict,
                seed=int(seed),
                suppress_output=True,
            )
            shape_meta = self.ckpt_dict.get("shape_metadata", {})
            self.action_dim = int(shape_meta.get("ac_dim", self.action_dim))
            return env, base_state
        return load_fresh_env_from_metadata(self.env_meta, seed=int(seed), render_offscreen=False)

    def idle_action(self, gripper: float) -> np.ndarray:
        action = np.zeros((self.action_dim,), dtype=np.float32)
        if self.action_dim > 0:
            action[-1] = float(gripper)
        return action


def save_trace_without_video(rollout: Dict[str, Any], output_dir: Path, rollout_tag: str) -> None:
    rollout_dir = output_dir / rollout_tag
    rollout_dir.mkdir(parents=True, exist_ok=True)
    trace_path = rollout_dir / "rollout_trace.npz"
    scene_snapshot_path = rollout_dir / "scene_snapshot.json"
    np.savez_compressed(
        trace_path,
        actions=np.asarray(rollout["actions"], dtype=np.float32),
        rewards=np.asarray(rollout["rewards"], dtype=np.float32),
        dones=np.asarray(rollout["dones"], dtype=bool),
        scene_states=np.asarray(rollout["scene_states"], dtype=np.float32),
        robot_states=np.asarray(rollout["robot_states"], dtype=np.float32),
        eef_xy=np.asarray(rollout["eef_xy"], dtype=np.float32),
        planned_actions=np.asarray(rollout["planned_actions"], dtype=np.float32),
        planned_transitions=np.asarray(rollout["planned_transitions"], dtype=np.float32),
        initial_label=np.asarray(rollout["initial_label"], dtype=np.int32),
        final_label=np.asarray(rollout["final_label"], dtype=np.int32),
        pre_settle_label=np.asarray(rollout["pre_settle_label"], dtype=np.int32),
        settle_scene_states=np.asarray(rollout["settle_scene_states"], dtype=np.float32),
        settle_robot_states=np.asarray(rollout["settle_robot_states"], dtype=np.float32),
        settle_action=np.asarray(rollout["settle_action"], dtype=np.float32),
        detected_behavior=np.asarray(rollout["behavior"]),
        detected_behavior_step=np.asarray(rollout["behavior_step"], dtype=np.int32),
        termination_step=np.asarray(rollout["termination_step"], dtype=np.int32),
        termination_reason=np.asarray(rollout["termination_reason"]),
        rollout_seed=np.asarray(rollout["seed"], dtype=np.int32),
        target_label_name=np.asarray(rollout["target_label_name"]),
        scene_config=np.asarray(rollout["scene_config"]),
    )
    CRU.save_scene_snapshot(rollout["scene_snapshot"], scene_snapshot_path)
    rollout["video"] = None
    rollout["trace"] = trace_path
    rollout["scene_snapshot_path"] = scene_snapshot_path
    rollout["rollout_dir"] = rollout_dir


class TelografCalvinPolicy:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.checkpoint = checkpoint
        self.stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
        self.horizon = int(checkpoint["horizon"])
        self.state_dim = int(checkpoint["state_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.condition_dim = int(checkpoint["condition_dim"])
        self.spec_dim = int(checkpoint["spec_dim"])

        if self.horizon != 64:
            raise ValueError(f"Expected H=64 checkpoint, got horizon={self.horizon}")
        if self.state_dim != 39 or self.action_dim != 7:
            raise ValueError(
                f"Expected CALVIN low-dim state/action dims 39/7, got {self.state_dim}/{self.action_dim}"
            )

        ckpt_args = checkpoint.get("args", {})
        if not isinstance(ckpt_args, dict):
            ckpt_args = vars(ckpt_args)
        self.n_timesteps = int(ckpt_args.get("n_timesteps", 100))
        dim = int(ckpt_args.get("dim", 64))
        dim_mults = tuple(int(x) for x in ckpt_args.get("dim_mults", [1, 2, 4, 8]))

        model = TemporalUnet(
            horizon=self.horizon,
            transition_dim=self.state_dim + self.action_dim,
            cond_dim=self.condition_dim,
            dim=dim,
            dim_mults=dim_mults,
            attention=True,
        ).to(device)
        self.diffuser = GaussianFlow(
            model,
            horizon=self.horizon,
            observation_dim=self.state_dim,
            action_dim=self.action_dim,
            n_timesteps=self.n_timesteps,
        ).to(device)
        self.diffuser.load_state_dict(checkpoint["model"])
        self.diffuser.eval()

    def _sample_transition_norm(
        self,
        cond: torch.Tensor,
        transition_state_norm: np.ndarray,
        *,
        flow_pattern: int,
        anchor_initial_state: bool,
    ) -> torch.Tensor:
        if not anchor_initial_state:
            return self.diffuser.conditional_sample(
                cond,
                args=SimpleNamespace(flow_pattern=flow_pattern),
            ).trajectories[0]

        if flow_pattern != 13:
            raise ValueError("Initial-state anchoring is currently implemented for flow_pattern=13 only.")

        batch_size = len(cond)
        shape = (batch_size, self.horizon, self.state_dim + self.action_dim)
        x = torch.randn(shape, device=self.device)
        anchor = torch.from_numpy(np.asarray(transition_state_norm, dtype=np.float32)).to(self.device).view(1, self.state_dim)
        anchor = anchor.expand(batch_size, -1)
        x[:, 0, : self.state_dim] = anchor
        t = torch.full((batch_size,), self.n_timesteps - 1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            x = x + self.diffuser.model(x, cond, t)
        x[:, 0, : self.state_dim] = anchor
        return x

    def sample_open_loop_actions(
        self,
        state: np.ndarray,
        target_label: str,
        *,
        flow_pattern: int = 13,
        anchor_initial_state: bool = True,
        num_candidates: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, Dict]:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape {(self.state_dim,)}, got {state.shape}")
        spec = make_spec(f"eventual_{target_label}", "eventual", labels=[target_label])
        spec_vec = spec_to_vector(spec).astype(np.float32)
        if len(spec_vec) != self.spec_dim:
            raise ValueError(f"Spec vector dim mismatch: expected {self.spec_dim}, got {len(spec_vec)}")
        state_norm = (state - self.stats["state_mean"]) / self.stats["state_std"]
        transition_state_norm = (
            (state - self.stats["transition_mean"][: self.state_dim])
            / self.stats["transition_std"][: self.state_dim]
        )
        cond_np = np.concatenate([state_norm, spec_vec], axis=0).astype(np.float32)
        if len(cond_np) != self.condition_dim:
            raise ValueError(f"Condition dim mismatch: expected {self.condition_dim}, got {len(cond_np)}")

        n_candidates = max(1, int(num_candidates))
        cond = torch.from_numpy(cond_np[None, :]).to(self.device)
        cond_batch = cond.expand(n_candidates, -1).contiguous()
        samples = self._sample_transition_norm(
            cond_batch,
            transition_state_norm,
            flow_pattern=int(flow_pattern),
            anchor_initial_state=bool(anchor_initial_state),
        )
        transitions_norm = samples.detach().cpu().numpy().astype(np.float32)
        transitions = transitions_norm * self.stats["transition_std"] + self.stats["transition_mean"]
        if transitions.ndim == 3:
            safe_state_std = np.maximum(self.stats["state_std"], 1e-4)
            start_errors = np.linalg.norm(
                (transitions[:, 0, : self.state_dim] - state[None, :]) / safe_state_std[None, :],
                axis=1,
            )
            spec_scores = []
            for candidate in transitions:
                _, score = evaluate_spec_sequence(spec, candidate[:, :15], candidate[:, 15:self.state_dim])
                spec_scores.append(float(score))
            spec_scores = np.asarray(spec_scores, dtype=np.float32)
            feasible = start_errors <= (1e-4 if anchor_initial_state else np.inf)
            if np.any(feasible):
                masked_scores = np.where(feasible, spec_scores, -np.inf)
                selected = int(np.argmax(masked_scores))
            else:
                selected = int(np.lexsort((-spec_scores, start_errors))[0])
            transitions = transitions[selected]
        actions = transitions[:, self.state_dim : self.state_dim + self.action_dim].astype(np.float32)
        return actions, transitions.astype(np.float32), spec


def rollout_telograf_once(
    *,
    seed: int,
    env_factory: CalvinEnvFactory,
    telograf: TelografCalvinPolicy,
    target_label_name: str,
    task_config: Dict[str, Any],
    task_config_path: Path,
    output_dir: Path,
    rollout_tag: str,
    video_cfg: Dict[str, Any],
    save_video: bool,
    reset_robot_y_min: Optional[float],
    reset_robot_y_max: Optional[float],
    reset_switch_clearance: Optional[float],
    settle_steps: int,
    settle_gripper: float,
    fps: int,
    clip_actions: bool,
    flow_pattern: int,
    anchor_initial_state: bool,
    num_candidates: int,
    fixed_reset_robot: Optional[np.ndarray],
    fixed_reset_pose_source: Optional[str],
) -> Dict[str, Any]:
    CRU.seed_everything(seed)
    env, base_env_state = env_factory.fresh(seed=int(seed))
    try:
        reset_task_name = LABEL_TO_RESET_TASK[target_label_name]
        acceptable_behaviors = LABEL_TO_ACCEPTABLE_BEHAVIORS[target_label_name]
        target_label_idx = LABEL_NAMES.index(target_label_name)
        if fixed_reset_robot is not None:
            reset_poses = [np.asarray(fixed_reset_robot, dtype=np.float32).copy()]
            reset_pose_filter = {
                "source": str(fixed_reset_pose_source),
                "fixed_reset_pose": True,
                "robot_y_min": reset_robot_y_min,
                "robot_y_max": reset_robot_y_max,
                "switch_clearance": reset_switch_clearance,
                "input_count": 1,
                "kept_count": 1,
            }
        else:
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

        settle_action = env_factory.idle_action(gripper=settle_gripper)
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

        scene_snapshot = CRU.capture_scene_snapshot(env)
        frames = [CRU.render_visual_camera(env, video_cfg)] if save_video else []

        start_state = env.get_state()
        start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
        start_robot = np.asarray(start_state["robot"], dtype=np.float32).copy()
        lowdim_state = np.concatenate([start_robot, start_scene], axis=0).astype(np.float32)
        planned_actions, planned_transitions, target_spec = telograf.sample_open_loop_actions(
            lowdim_state,
            target_label_name,
            flow_pattern=flow_pattern,
            anchor_initial_state=anchor_initial_state,
            num_candidates=num_candidates,
        )

        label0 = labels_for_scene(start_scene)
        target_label_initial = bool(label0[target_label_idx] >= 0.5)
        actions, rewards, dones = [], [], []
        scene_states = [start_scene.copy()]
        robot_states = [start_robot.copy()]
        eef_xy = [start_robot[:2].copy()]
        detected_behavior = "none"
        detected_step = -1
        termination_reason = "horizon"
        label_success = bool(target_label_initial)
        target_label_flipped = False
        target_label_success_step = 0 if label_success else -1
        ignored_behavior_events = []
        last_ignored_behavior = "other"
        env_done_step = -1
        total_reward = 0.0

        executed_plan = planned_actions[: telograf.horizon]
        for step, action in enumerate(executed_plan):
            action_to_step = np.clip(action, -1.0, 1.0).astype(np.float32) if clip_actions else action.astype(np.float32)
            next_obs, reward, done, _ = env.step(action_to_step.copy())
            total_reward += float(reward)

            state_now = env.get_state()
            scene_now = np.asarray(state_now["scene"], dtype=np.float32).copy()
            robot_now = np.asarray(state_now["robot"], dtype=np.float32).copy()

            actions.append(np.asarray(action_to_step, dtype=np.float32).copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            scene_states.append(scene_now.copy())
            robot_states.append(robot_now.copy())
            eef_xy.append(robot_now[:2].copy())

            if save_video:
                frames.append(CRU.render_visual_camera(env, video_cfg))

            behavior_now = CRU.classify_behavior(start_scene, scene_now, robot_now[:3], binaries, for_display=False)
            if behavior_now in acceptable_behaviors and detected_step < 0:
                detected_behavior = behavior_now
                detected_step = int(step + 1)
            elif behavior_now != "other" and behavior_now != last_ignored_behavior:
                ignored_behavior_events.append({"behavior": behavior_now, "step": int(step + 1)})
                last_ignored_behavior = behavior_now

            if done and env_done_step < 0:
                env_done_step = int(step + 1)

            label_now = labels_for_scene(scene_now)
            target_label_current = bool(label_now[target_label_idx] >= 0.5)
            if target_label_current:
                label_success = True
                if target_label_success_step < 0:
                    target_label_success_step = int(step + 1)
                if not target_label_initial:
                    target_label_flipped = True
                    termination_reason = "target_label_success"
                    break
            if detected_step >= 0:
                termination_reason = "behavior_success"
                break
            if done:
                termination_reason = "env_done"
                break
            obs = next_obs
        else:
            step = len(planned_actions[: telograf.horizon]) - 1

        labelf = labels_for_scene(scene_states[-1])
        target_label_final = bool(labelf[target_label_idx] >= 0.5)
        behavior_success = bool(detected_step >= 0)
        rollout = {
            "task": reset_task_name,
            "scene_config": task_config.get("name", reset_task_name),
            "seed": int(seed),
            "policy": "telograf_calvin_open_loop_h64",
            "target_label_name": target_label_name,
            "target_spec": target_spec,
            "target_label_idx": int(target_label_idx),
            "target_label_initial": bool(target_label_initial),
            "target_label_final": bool(target_label_final),
            "label_success": bool(label_success),
            "label_flipped": bool(target_label_flipped),
            "label_success_step": int(target_label_success_step),
            "label_flip_step": int(target_label_success_step if target_label_flipped else -1),
            "target_behavior_names": list(acceptable_behaviors),
            "behavior": detected_behavior,
            "behavior_step": int(detected_step),
            "behavior_success": behavior_success,
            "success": bool(label_success),
            "termination_step": int(step + 1),
            "termination_reason": termination_reason,
            "env_done_step": int(env_done_step),
            "return": float(total_reward),
            "initial_label": label0.astype(int).tolist(),
            "final_label": labelf.astype(int).tolist(),
            "actions": np.asarray(actions, dtype=np.float32),
            "planned_actions": np.asarray(planned_actions, dtype=np.float32),
            "clip_actions": bool(clip_actions),
            "flow_pattern": int(flow_pattern),
            "anchor_initial_state": bool(anchor_initial_state),
            "num_candidates": int(num_candidates),
            "planned_transitions": np.asarray(planned_transitions, dtype=np.float32),
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
            "fixed_reset_pose_source": fixed_reset_pose_source,
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
        np.savez_compressed(
            rollout_dir / "telograf_open_loop_plan.npz",
            planned_actions=np.asarray(planned_actions, dtype=np.float32),
            planned_transitions=np.asarray(planned_transitions, dtype=np.float32),
            target_label_name=np.asarray(target_label_name),
        )
        label_status = (
            f"{target_label_name} @ {target_label_success_step}"
            if label_success
            else f"no {target_label_name}"
        )
        CRU.plot_rollout_xy(
            [rollout],
            rollout["scene_snapshot"],
            f"TeLoGraF {target_label_name} seed {seed} -> {label_status}",
            save_path=rollout_dir / "rollout_xy.png",
            display_inline=False,
        )
        write_json(
            rollout_dir / "rollout_summary.json",
            {
                "target_label_name": target_label_name,
                "reset_task": reset_task_name,
                "seed": int(seed),
                "label_success": bool(label_success),
                "label_flipped": bool(target_label_flipped),
                "label_success_step": int(target_label_success_step),
                "target_label_initial": bool(target_label_initial),
                "target_label_final": bool(target_label_final),
                "behavior": detected_behavior,
                "behavior_step": int(detected_step),
                "behavior_success": behavior_success,
                "target_behavior_names": list(acceptable_behaviors),
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
                "settle_metrics": rollout["settle_metrics"],
            },
        )
        return rollout
    finally:
        CRU.close_env_quietly(env)


def task_summary(target_label_name: str, rollouts: Sequence[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    behavior_counts = Counter(rollout["behavior"] for rollout in rollouts)
    ignored_behavior_counts = Counter(
        event["behavior"]
        for rollout in rollouts
        for event in rollout.get("ignored_behavior_events", [])
    )
    label_successes = sum(1 for rollout in rollouts if rollout["label_success"])
    label_flips = sum(1 for rollout in rollouts if rollout["label_flipped"])
    behavior_successes = sum(1 for rollout in rollouts if rollout["behavior_success"])
    return {
        "target_label": target_label_name,
        "reset_task": LABEL_TO_RESET_TASK[target_label_name],
        "target_behavior_names": list(LABEL_TO_ACCEPTABLE_BEHAVIORS[target_label_name]),
        "n_rollouts": len(rollouts),
        "horizon": int(horizon),
        "controller": "open_loop_h64_sample_once",
        "label_success_count": int(label_successes),
        "label_success_rate": float(label_successes / len(rollouts)) if rollouts else 0.0,
        "label_flip_count": int(label_flips),
        "behavior_success_count": int(behavior_successes),
        "behavior_success_rate": float(behavior_successes / len(rollouts)) if rollouts else 0.0,
        "behavior_counts": dict(behavior_counts),
        "ignored_behavior_counts": dict(ignored_behavior_counts),
        "avg_termination_step": float(np.mean([rollout["termination_step"] for rollout in rollouts])) if rollouts else 0.0,
        "rollouts": [
            {
                "seed": rollout["seed"],
                "label_success": bool(rollout["label_success"]),
                "label_flipped": bool(rollout["label_flipped"]),
                "label_success_step": rollout["label_success_step"],
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
            "target_label": item["target_label"],
            "reset_task": item["reset_task"],
            "controller": item["controller"],
            "label_success_rate": f"{item['label_success_rate']:.4f}",
            "label_success_count": item["label_success_count"],
            "label_flip_count": item["label_flip_count"],
            "behavior_success_rate": f"{item['behavior_success_rate']:.4f}",
            "behavior_success_count": item["behavior_success_count"],
            "n_rollouts": item["n_rollouts"],
            "avg_termination_step": f"{item['avg_termination_step']:.2f}",
            "behavior_counts": json.dumps(item["behavior_counts"], sort_keys=True),
            "ignored_behavior_counts": json.dumps(item["ignored_behavior_counts"], sort_keys=True),
        }
        for item in summaries
    ]
    fieldnames = [
        "target_label",
        "reset_task",
        "controller",
        "label_success_rate",
        "label_success_count",
        "label_flip_count",
        "behavior_success_rate",
        "behavior_success_count",
        "n_rollouts",
        "avg_termination_step",
        "behavior_counts",
        "ignored_behavior_counts",
    ]
    with (run_dir / "summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "summary_table.md").open("w") as f:
        f.write(
            "| target_label | reset_task | controller | label_success_rate | label_success_count | "
            "label_flip_count | behavior_success_rate | behavior_success_count | n_rollouts | "
            "avg_termination_step | behavior_counts | ignored_behavior_counts |\n"
        )
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['target_label']} | {row['reset_task']} | {row['controller']} | "
                f"{row['label_success_rate']} | {row['label_success_count']} | {row['label_flip_count']} | "
                f"{row['behavior_success_rate']} | {row['behavior_success_count']} | {row['n_rollouts']} | "
                f"{row['avg_termination_step']} | `{row['behavior_counts']}` | `{row['ignored_behavior_counts']}` |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telograf-checkpoint", type=Path, default=DEFAULT_TELOGRAF_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=DEFAULT_ENV_CHECKPOINT)
    parser.add_argument(
        "--env-dataset",
        type=Path,
        default=DEFAULT_ENV_DATASET,
        help="Fallback source for CALVIN env_args when --env-checkpoint is unavailable.",
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--visualization-config", type=Path, default=DEFAULT_VISUALIZATION_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--targets", "--tasks", nargs="*", default=None, choices=TARGET_LABELS)
    parser.add_argument("--n-rollouts", "--num-rollouts", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=DEFAULT_SETTLE_GRIPPER)
    parser.add_argument("--reset-robot-y-min", type=float, default=DEFAULT_RESET_ROBOT_Y_MIN)
    parser.add_argument("--reset-robot-y-max", type=float, default=None)
    parser.add_argument("--reset-switch-clearance", type=float, default=DEFAULT_RESET_SWITCH_CLEARANCE)
    parser.add_argument(
        "--fixed-reset-pose-file",
        type=Path,
        default=None,
        help="Use one fixed robot state from this reset-pose JSON for every target and rollout.",
    )
    parser.add_argument("--fixed-reset-pose-index", type=int, default=0)
    parser.add_argument("--disable-reset-pose-filter", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-action-clip", action="store_true", help="Execute raw TeLoGraF actions instead of clipping to [-1, 1].")
    parser.add_argument("--flow-pattern", type=int, default=13)
    parser.add_argument(
        "--no-initial-state-anchor",
        action="store_true",
        help="Use the original free trajectory sampler without hard-clamping the first generated state.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Sample this many TeLoGraF trajectories and execute the one whose first generated state is closest to the env state.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    targets = list(args.targets) if args.targets else list(TARGET_LABELS)
    reset_robot_y_min = None if args.disable_reset_pose_filter else args.reset_robot_y_min
    reset_robot_y_max = None if args.disable_reset_pose_filter else args.reset_robot_y_max
    reset_switch_clearance = None if args.disable_reset_pose_filter else args.reset_switch_clearance

    telograf_checkpoint = repo_path(args.telograf_checkpoint)
    env_checkpoint = repo_path(args.env_checkpoint)
    env_dataset = repo_path(args.env_dataset)
    config_dir = resolve_existing_path(args.config_dir)
    visualization_config = resolve_existing_path(args.visualization_config)
    fixed_reset_pose_file = repo_path(args.fixed_reset_pose_file) if args.fixed_reset_pose_file else None
    fixed_reset_robot = None
    fixed_reset_pose_source = None
    if fixed_reset_pose_file is not None:
        if not fixed_reset_pose_file.exists():
            raise FileNotFoundError(f"Fixed reset pose file not found: {fixed_reset_pose_file}")
        fixed_payload = load_json(fixed_reset_pose_file)
        fixed_robot_states = fixed_payload.get("robot_states", [])
        if not fixed_robot_states:
            raise ValueError(f"No robot_states in fixed reset pose file: {fixed_reset_pose_file}")
        fixed_idx = int(args.fixed_reset_pose_index)
        if fixed_idx < 0 or fixed_idx >= len(fixed_robot_states):
            raise IndexError(
                f"--fixed-reset-pose-index {fixed_idx} out of range for {fixed_reset_pose_file} "
                f"with {len(fixed_robot_states)} states"
            )
        fixed_reset_robot = np.asarray(fixed_robot_states[fixed_idx], dtype=np.float32).copy()
        fixed_reset_pose_source = f"{fixed_reset_pose_file}[{fixed_idx}]"
    task_config_paths = {
        target: resolve_existing_path(config_dir / f"{LABEL_TO_RESET_TASK[target]}.json")
        for target in targets
    }
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_name = args.name or f"telograf_open_loop_h64_rollouts{args.n_rollouts}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name

    planned = {
        "policy": "telograf_calvin_open_loop_h64",
        "telograf_checkpoint": str(telograf_checkpoint),
        "telograf_checkpoint_exists": telograf_checkpoint.exists(),
        "env_checkpoint": str(env_checkpoint),
        "env_checkpoint_exists": env_checkpoint.exists(),
        "env_dataset": str(env_dataset),
        "env_dataset_exists": env_dataset.exists(),
        "config_dir": str(config_dir),
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "targets": targets,
        "task_configs": {target: str(path) for target, path in task_config_paths.items()},
        "n_rollouts": int(args.n_rollouts),
        "seed_start": int(args.seed_start),
        "horizon": 64,
        "controller": "open_loop_sample_once",
        "stop_condition": "target_label_or_behavior_or_done_or_horizon",
        "settle_steps": int(args.settle_steps),
        "settle_gripper": float(args.settle_gripper),
        "reset_robot_y_min": reset_robot_y_min,
        "reset_robot_y_max": reset_robot_y_max,
        "reset_switch_clearance": reset_switch_clearance,
        "disable_reset_pose_filter": bool(args.disable_reset_pose_filter),
        "fixed_reset_pose_file": str(fixed_reset_pose_file) if fixed_reset_pose_file else None,
        "fixed_reset_pose_index": int(args.fixed_reset_pose_index),
        "fixed_reset_pose_source": fixed_reset_pose_source,
        "save_video": not args.no_video,
        "clip_actions": not args.no_action_clip,
        "flow_pattern": int(args.flow_pattern),
        "anchor_initial_state": not args.no_initial_state_anchor,
        "num_candidates": int(args.num_candidates),
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return

    if not telograf_checkpoint.exists():
        raise FileNotFoundError(f"TeLoGraF checkpoint not found: {telograf_checkpoint}")
    if not env_checkpoint.exists() and not env_dataset.exists():
        raise FileNotFoundError(
            f"Neither CALVIN robomimic env checkpoint nor env dataset exists. "
            f"Missing checkpoint: {env_checkpoint}; missing dataset: {env_dataset}"
        )

    video_cfg = load_json(visualization_config)
    device = resolve_device(args.device)
    telograf = TelografCalvinPolicy(telograf_checkpoint, device)
    if env_checkpoint.exists():
        env_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(env_checkpoint))
        env_factory = CalvinEnvFactory(ckpt_dict=env_ckpt_dict)
    else:
        env_meta = load_env_metadata_from_dataset(env_dataset)
        env_factory = CalvinEnvFactory(env_meta=env_meta, action_dim=telograf.action_dim)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_args.json", planned)
    print("device:", device)
    print("telograf checkpoint:", telograf_checkpoint)
    print("env source:", env_factory.source)
    print("env checkpoint:", env_checkpoint if env_checkpoint.exists() else "missing")
    print("env dataset:", env_dataset if env_dataset.exists() else "missing")
    print("output:", run_dir)

    all_summaries = []
    for target in targets:
        reset_task = LABEL_TO_RESET_TASK[target]
        task_config_path = task_config_paths[target]
        task_config = load_json(task_config_path)
        task_dir = run_dir / target
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "task_config_resolved.json", task_config)

        print(f"\nTarget {target}: reset_task={reset_task} config={task_config_path}")
        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            rollout = rollout_telograf_once(
                seed=seed,
                env_factory=env_factory,
                telograf=telograf,
                target_label_name=target,
                task_config=task_config,
                task_config_path=task_config_path,
                output_dir=task_dir,
                rollout_tag=tag,
                video_cfg=video_cfg,
                save_video=not args.no_video,
                reset_robot_y_min=reset_robot_y_min,
                reset_robot_y_max=reset_robot_y_max,
                reset_switch_clearance=reset_switch_clearance,
                settle_steps=int(args.settle_steps),
                settle_gripper=float(args.settle_gripper),
                fps=int(args.fps),
                clip_actions=not args.no_action_clip,
                flow_pattern=int(args.flow_pattern),
                anchor_initial_state=not args.no_initial_state_anchor,
                num_candidates=int(args.num_candidates),
                fixed_reset_robot=fixed_reset_robot,
                fixed_reset_pose_source=fixed_reset_pose_source,
            )
            rollouts.append(rollout)
            print(
                f"  seed {seed:03d}: label_success={rollout['label_success']} "
                f"label_step={rollout['label_success_step']:>3}, "
                f"behavior_success={rollout['behavior_success']} "
                f"behavior={rollout['behavior']:>12} @ {rollout['behavior_step']:>3}, "
                f"steps={rollout['termination_step']:>3}, final={format_onehot(rollout['final_label'])}"
            )

        summary = task_summary(target, rollouts, telograf.horizon)
        write_json(task_dir / "task_summary.json", summary)
        CRU.plot_rollout_xy(
            rollouts,
            rollouts[0]["scene_snapshot"],
            (
                f"TeLoGraF {target} | label {summary['label_success_count']}/{summary['n_rollouts']} | "
                f"behavior {summary['behavior_success_count']}/{summary['n_rollouts']}"
            ),
            save_path=task_dir / "task_rollouts_xy.png",
            display_inline=False,
        )
        all_summaries.append(summary)

    write_json(
        run_dir / "summary.json",
        {
            "run": planned,
            "telograf": {
                "checkpoint": str(telograf_checkpoint),
                "horizon": telograf.horizon,
                "state_dim": telograf.state_dim,
                "action_dim": telograf.action_dim,
                "condition_dim": telograf.condition_dim,
                "spec_dim": telograf.spec_dim,
                "n_timesteps": telograf.n_timesteps,
                "backend": telograf.checkpoint.get("backend"),
            },
            "targets": all_summaries,
        },
    )
    write_summary_tables(run_dir, all_summaries)
    print("\nSummary:", run_dir / "summary_table.md")


if __name__ == "__main__":
    main()
