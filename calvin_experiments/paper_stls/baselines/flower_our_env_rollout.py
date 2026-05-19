#!/usr/bin/env python
"""Roll out FLOWER VLA in this repo's robomimic-style CALVIN env.

This intentionally adapts at the boundary:
  our env obs: third_person / eye_in_hand C,H,W float images
  FLOWER obs:  rgb_obs.rgb_static / rgb_gripper B,T,C,H,W normalized tensors

Run:
    /home/moritz/.conda/envs/calvin/bin/python \
      calvin_experiments/paper_stls/baselines/flower_our_env_rollout.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
FLOWER_ROOT = REPO_ROOT / "flower_vla_calvin"
DEFAULT_FLOWER_CHECKPOINT = REPO_ROOT / "outputs/calvin/baselines/flower/flower_calvin_d"
DEFAULT_ENV_CHECKPOINT = REPO_ROOT / "outputs/calvin/base_policy/calvin_D_base_dp/20260501015147/models/model_epoch_280.pth"
DEFAULT_SCENE_CONFIG = REPO_ROOT / "calvin_experiments/configs/blocks_hidden.json"
DEFAULT_VIDEO_CONFIG = REPO_ROOT / "calvin_experiments/configs/visualization_freiburg_style.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/calvin/baselines/flower/our_env_rollouts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=FLOWER_ROOT)
    parser.add_argument("--flower-checkpoint", type=Path, default=DEFAULT_FLOWER_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=DEFAULT_ENV_CHECKPOINT)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--video-config", type=Path, default=DEFAULT_VIDEO_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--instruction", default="pull the handle to open the drawer")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stop-on-behavior", action="store_true")
    parser.add_argument(
        "--expected-behavior",
        action="append",
        default=[],
        help="Behavior that must be observed before stopping. Pass multiple times for a sequence/set.",
    )
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face downloads/checks instead of cache-only mode.")
    return parser.parse_args()


def add_repo_paths(repo_root: Path, flower_root: Path) -> None:
    for path in [
        repo_root,
        repo_root / "robomimic",
        repo_root / "calvin" / "calvin_env",
        repo_root / "calvin_experiments",
        flower_root,
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name != "cpu" and not torch.cuda.is_available():
        print(f"Requested {device_name}, but torch cannot see CUDA here; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_flower_model(checkpoint: Path, device):
    from flower.evaluation.utils import load_mode_from_safetensor
    from transformers.dynamic_module_utils import get_imports

    def get_imports_without_flash_attn(filename):
        imports = get_imports(filename)
        if filename.endswith("modeling_florence2.py") and "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

    with patch("transformers.dynamic_module_utils.get_imports", get_imports_without_flash_attn):
        model = load_mode_from_safetensor(
            checkpoint,
            overwrite_cfg={
                "num_sampling_steps": 4,
                "multistep": 10,
                "query_seq_len": 120,
                "return_act_chunk": False,
            },
        )
    model.freeze()
    model.to(device)
    model.eval()
    model.reset()
    return model


class FlowerPolicyAdapter:
    """Thin policy adapter from this repo's CALVIN obs format to FLOWER."""

    def __init__(self, model, instruction: str = "", device=None):
        import torch

        if device is None:
            device = next(model.parameters()).device
        self.model = model
        self.instruction = instruction
        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def reset(self, instruction: str | None = None) -> None:
        if instruction is not None:
            self.instruction = instruction
        self.model.reset()

    def _image_to_flower_tensor(self, image):
        import torch
        from torchvision.transforms import functional as TF

        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[-1]
        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")
        if arr.shape[0] == 3:
            tensor = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)
        elif arr.shape[-1] == 3:
            tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Cannot identify channel dimension in image shape {arr.shape}")
        if tensor.max() <= 2.0:
            tensor = tensor.clamp(0.0, 1.0) * 255.0
        tensor = tensor.clamp(0.0, 255.0).round().to(torch.uint8)
        tensor = TF.resize(tensor, [224, 224], antialias=True).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)  # B,T,C,H,W with B=1,T=1

    def observation_to_flower(self, obs):
        if "third_person" not in obs or "eye_in_hand" not in obs:
            raise KeyError(f"Expected robomimic CALVIN obs keys third_person and eye_in_hand; got {sorted(obs.keys())}")
        return {
            "rgb_obs": {
                "rgb_static": self._image_to_flower_tensor(obs["third_person"]),
                "rgb_gripper": self._image_to_flower_tensor(obs["eye_in_hand"]),
            }
        }

    def __call__(self, obs):
        import torch

        flower_obs = self.observation_to_flower(obs)
        goal = {"lang_text": self.instruction}
        with torch.no_grad():
            action = self.model.step(flower_obs, goal)
        action_np = action.detach().cpu().view(-1).numpy().astype(np.float32)
        if action_np.shape != (7,):
            raise ValueError(f"FLOWER produced action shape {action_np.shape}; expected (7,)")
        action_np[-1] = 1.0 if action_np[-1] > 0 else -1.0
        return np.clip(action_np, -1.0, 1.0)


def detect_behaviors_from_state(start_state, state, robot_pos, binaries=None, for_display: bool = False) -> list[str]:
    """Return every behavior currently true relative to the rollout start state."""

    from calvin_experiments.calvin_rollout_utils import (
        ADJUSTABLE_BEHAVIORS,
        ADJUSTABLE_INDEX,
        ADJUSTABLE_LIMITS,
        BLOCK_POS_SLICES,
        articulated_binaries_from_start_state,
    )

    start_state = np.asarray(start_state)
    state = np.asarray(state)
    if binaries is None:
        binaries = articulated_binaries_from_start_state(start_state)

    detected = []
    for binary, idx, limits, names in zip(binaries, ADJUSTABLE_INDEX, ADJUSTABLE_LIMITS, ADJUSTABLE_BEHAVIORS):
        span = limits[1] - limits[0]
        midpoint = limits[0] + span / 2
        low_threshold = limits[0] + 0.25 * span if for_display else midpoint
        high_threshold = limits[0] + 0.75 * span if for_display else midpoint
        high_to_low_name, low_to_high_name = names

        if binary and state[idx] < low_threshold:
            detected.append(high_to_low_name)
        if not binary and state[idx] > high_threshold:
            detected.append(low_to_high_name)

    robot_pos = np.asarray(robot_pos)
    block_threshold = 0.03 if for_display else 0.001
    for color, pos_slice in BLOCK_POS_SLICES.items():
        if np.linalg.norm(robot_pos - state[pos_slice]) < 0.06:
            xy_delta = state[pos_slice.start : pos_slice.start + 2] - start_state[pos_slice.start : pos_slice.start + 2]
            if np.linalg.norm(xy_delta) > block_threshold:
                detected.append(f"{color}_displace")

    return detected


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    add_repo_paths(REPO_ROOT, args.flower_root.resolve())

    import robomimic.envs  # noqa: F401
    import robomimic.utils.file_utils as FileUtils

    import calvin_experiments.calvin_rollout_utils as CalvinRolloutUtils
    from calvin_experiments.calvin_rollout_utils import (
        articulated_binaries_from_start_state,
        capture_scene_snapshot,
        close_env_quietly,
        fixed_scene_robot_from_config,
        load_fresh_env_from_checkpoint,
        load_json_config,
        plot_rollouts_from_trace_summaries,
        render_visual_camera,
        reset_env_to_scene_robot,
        run_folder_name,
        save_rollout_artifacts,
        seed_everything,
    )

    device = resolve_device(args.device)
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(args.env_checkpoint))
    flower_model = load_flower_model(args.flower_checkpoint, device)
    flower_policy = FlowerPolicyAdapter(flower_model, device=device)
    expected_behaviors = list(dict.fromkeys(args.expected_behavior))

    video_cfg = load_json_config(args.video_config)
    scene_cfg_json = load_json_config(args.scene_config)
    scene_name = scene_cfg_json["name"]
    output_dir = args.output_root / run_folder_name(
        "flower_vla",
        f"scene_{scene_name}",
        f"rollouts{args.num_rollouts}",
        f"horizon{args.horizon}",
        f"seed{args.seed}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"FLOWER checkpoint: {args.flower_checkpoint}")
    print(f"Env checkpoint:    {args.env_checkpoint}")
    print(f"Instruction:       {args.instruction!r}")
    print(f"Expected behavior: {expected_behaviors if expected_behaviors else 'none'}")
    print(f"Device:            {device}")
    print(f"Output:            {output_dir}")

    summaries = []
    env = None
    try:
        seed_everything(args.seed)
        for rollout_idx in range(args.num_rollouts):
            rollout_seed = None if args.seed is None else args.seed + rollout_idx
            env, base_env_state = load_fresh_env_from_checkpoint(
                ckpt_dict,
                seed=rollout_seed,
                existing_env=env,
                suppress_output=True,
            )
            fixed_scene, fixed_robot, scene_cfg = fixed_scene_robot_from_config(base_env_state, args.scene_config)
            seed_everything(rollout_seed)
            flower_policy.reset(args.instruction)
            obs = reset_env_to_scene_robot(env, fixed_scene, fixed_robot)
            scene_snapshot = capture_scene_snapshot(env)
            start_state = env.get_state()
            start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
            binaries = articulated_binaries_from_start_state(start_scene)

            frames = [render_visual_camera(env, video_cfg)]
            actions, rewards, dones = [], [], []
            scene_states = [start_scene.copy()]
            robot_states = [np.asarray(start_state["robot"], dtype=np.float32).copy()]
            eef_xy = [robot_states[-1][:2].copy()]
            detected_behavior = "none"
            detected_step = -1
            detected_behaviors = []
            behavior_steps = {}
            termination_reason = "horizon"

            for step in range(int(args.horizon)):
                action = flower_policy(obs)
                actions.append(action.copy())
                obs, reward, done, info = env.step(action.copy())
                state = env.get_state()
                scene = np.asarray(state["scene"], dtype=np.float32).copy()
                robot = np.asarray(state["robot"], dtype=np.float32).copy()

                rewards.append(float(reward))
                dones.append(bool(done))
                scene_states.append(scene)
                robot_states.append(robot)
                eef_xy.append(robot[:2].copy())
                frames.append(render_visual_camera(env, video_cfg))

                current_behaviors = detect_behaviors_from_state(start_scene, scene, robot[:3], binaries)
                for behavior in current_behaviors:
                    if behavior not in behavior_steps:
                        behavior_steps[behavior] = step + 1
                        detected_behaviors.append(behavior)
                if detected_step < 0 and detected_behaviors:
                    detected_behavior = detected_behaviors[0]
                    detected_step = behavior_steps[detected_behavior]
                if expected_behaviors and set(expected_behaviors).issubset(behavior_steps):
                    termination_reason = "expected_behaviors"
                    break
                if args.stop_on_behavior and not expected_behaviors and detected_behaviors:
                    termination_reason = "behavior"
                    break
                if done:
                    termination_reason = "env_done"
                    break

            rollout = {
                "scene_config": scene_name,
                "seed": rollout_seed,
                "instruction": args.instruction,
                "behavior": detected_behavior,
                "behavior_step": detected_step,
                "detected_behaviors": detected_behaviors,
                "behavior_steps": behavior_steps,
                "expected_behaviors": expected_behaviors,
                "termination_step": len(actions),
                "termination_reason": termination_reason,
                "return": float(np.sum(rewards)),
                "actions": np.asarray(actions, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "dones": np.asarray(dones, dtype=bool),
                "scene_states": np.asarray(scene_states, dtype=np.float32),
                "robot_states": np.asarray(robot_states, dtype=np.float32),
                "eef_xy": np.asarray(eef_xy, dtype=np.float32),
                "scene_snapshot": scene_snapshot,
            }
            save_rollout_artifacts(rollout, frames, output_dir, f"rollout_{rollout_idx:03d}", video_cfg, fps=args.fps)
            summaries.append(
                {
                    "scene_config": scene_name,
                    "rollout": rollout_idx,
                    "seed": rollout_seed,
                    "instruction": args.instruction,
                    "behavior": detected_behavior,
                    "step": detected_step,
                    "detected_behaviors": detected_behaviors,
                    "behavior_steps": behavior_steps,
                    "expected_behaviors": expected_behaviors,
                    "termination_step": len(actions),
                    "termination_reason": termination_reason,
                    "video": str(rollout["video"]),
                    "trace": str(rollout["trace"]),
                    "scene_snapshot": str(rollout["scene_snapshot_path"]),
                }
            )
            print(
                f"rollout {rollout_idx:03d}: behavior={detected_behavior} "
                f"step={detected_step} detected={detected_behaviors} "
                f"termination={termination_reason} actions={len(actions)}"
            )
    finally:
        close_env_quietly(env)

    summary_path = output_dir / "flower_rollout_summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    if summaries:
        plot_rollouts_from_trace_summaries(
            summaries,
            title=f"FLOWER VLA | {scene_name} | {args.instruction}",
            save_path=output_dir / "rollout_scene_overlay_all.png",
            display_inline=False,
        )
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
