#!/usr/bin/env python
"""Minimal isolated FLOWER rollout in FLOWER's CALVIN environment.

Run this with the FLOWER conda env, for example:

    /home/moritz/.conda/envs/flower_cal/bin/python \
      /home/moritz/src/guided-diffusion/calvin_experiments/paper_stls/baselines/flower_setup.py

The script intentionally avoids importing this repository's CALVIN or DP code.  It
uses only flower_vla_calvin, its local calvin_env, and the downloaded FLOWER
checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]
FLOWER_ROOT = REPO_ROOT / "flower_vla_calvin"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/calvin/baselines/flower/flower_calvin_d"
DEFAULT_ENV_DIR = REPO_ROOT / "outputs/calvin/baselines/flower/calvin_env_config"
DEFAULT_ROLLOUT_ROOT = REPO_ROOT / "outputs/calvin/baselines/flower/rollouts/setup_tests"


OBSERVATION_SPACE = {
    "rgb_obs": ["rgb_static", "rgb_gripper"],
    "depth_obs": [],
    "state_obs": ["robot_obs"],
    "actions": ["rel_actions"],
    "language": ["language"],
}

PROPRIO_STATE = {
    "n_state_obs": 8,
    "keep_indices": [[0, 7], [14, 15]],
    "robot_orientation_idx": [3, 6],
    "normalize": True,
    "normalize_robot_orientation": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=FLOWER_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--instruction", default="pull the handle to open the drawer")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0", help="Use cuda:0 for normal FLOWER inference, or cpu for smoke tests.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face network checks/downloads instead of using the local cache only.")
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", default="calvin_scene_D")
    return parser.parse_args()


def add_flower_to_path(flower_root: Path) -> None:
    sys.path.insert(0, str(flower_root))
    sys.path.insert(0, str(flower_root / "calvin_env"))


def make_calvin_merged_config(env_dir: Path, flower_root: Path, scene: str, seed: int) -> Path:
    """Write the tiny render config expected by calvin_env.envs.play_table_env.get_env."""
    from omegaconf import OmegaConf

    conf_dir = flower_root / "calvin_env/conf"
    scene_cfg = OmegaConf.load(conf_dir / "scene" / f"{scene}.yaml")
    robot_cfg = OmegaConf.merge(
        OmegaConf.load(conf_dir / "robot/panda.yaml"),
        OmegaConf.load(conf_dir / "robot/panda_longer_finger.yaml"),
    )
    robot_cfg.pop("defaults", None)
    static_camera = OmegaConf.load(conf_dir / "cameras/cameras/static.yaml")
    gripper_camera = OmegaConf.load(conf_dir / "cameras/cameras/gripper.yaml")

    data_path = str((flower_root / "calvin_env/data").resolve())
    scene_cfg.data_path = data_path
    scene_cfg.euler_obs = robot_cfg.euler_obs
    robot_cfg.base_position = scene_cfg.robot_base_position
    robot_cfg.base_orientation = scene_cfg.robot_base_orientation
    robot_cfg.initial_joint_positions = scene_cfg.robot_initial_joint_positions

    cfg = OmegaConf.create(
        {
            "env": {
                "_target_": "calvin_env.envs.play_table_env.PlayTableSimEnv",
                "_recursive_": False,
                "robot_cfg": robot_cfg,
                "seed": seed,
                "use_vr": False,
                "bullet_time_step": 240,
                "cameras": {"static": static_camera, "gripper": gripper_camera},
                "scene_cfg": scene_cfg,
                "use_egl": False,
                "control_freq": 30,
            },
            "cameras": {"static": static_camera, "gripper": gripper_camera},
            "scene": scene_cfg,
            "robot": robot_cfg,
            "data_path": data_path,
        }
    )

    hydra_dir = env_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    out_path = hydra_dir / "merged_config.yaml"
    OmegaConf.save(cfg, out_path)
    return out_path


def make_val_transforms():
    import torchvision.transforms as T

    from flower.utils.transforms import ScaleImageTensor

    image_transform = T.Compose(
        [
            T.Resize(224, antialias=True),
            ScaleImageTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return {"rgb_static": image_transform, "rgb_gripper": image_transform}


def make_dataset_adapter(env_dir: Path):
    return SimpleNamespace(
        abs_datasets_dir=env_dir,
        observation_space=OBSERVATION_SPACE,
        transforms=make_val_transforms(),
        proprio_state=SimpleNamespace(**PROPRIO_STATE),
    )


def resolve_device(device_name: str):
    import torch

    if device_name != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU. This is only practical as a smoke test.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_flower_model(checkpoint: Path, device, steps_per_chunk: int):
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
                "multistep": steps_per_chunk,
                "query_seq_len": 120,
                "return_act_chunk": False,
            },
        )
    model.freeze()
    model.to(device)
    model.eval()
    model.reset()
    return model


def get_initial_state():
    from flower.evaluation.utils import get_env_state_for_initial_condition

    initial_condition = {
        "slider": "right",
        "drawer": "closed",
        "lightbulb": 0,
        "led": 0,
        "red_block": "table",
        "blue_block": "table",
        "pink_block": "table",
    }
    return get_env_state_for_initial_condition(initial_condition)


def make_rollout_dir(rollout_root: Path) -> Path:
    rollout_dir = rollout_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rollout_dir.mkdir(parents=True, exist_ok=False)
    return rollout_dir


def drawer_joint(info) -> float:
    return float(info["scene_info"]["doors"]["base__drawer"]["current_state"])


def render_frame(env, text: str | None = None):
    import cv2

    frame = env.render(mode="rgb_array")
    if text:
        frame = frame.copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def write_video(frames, rollout_dir: Path, fps: int) -> Path | None:
    if not frames:
        return None

    import imageio.v2 as imageio

    mp4_path = rollout_dir / "rollout.mp4"
    try:
        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
        return mp4_path
    except Exception as exc:
        print(f"MP4 write failed ({exc}); writing GIF fallback.")

    gif_path = rollout_dir / "rollout.gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    return gif_path


def write_actions_csv(actions, rollout_dir: Path) -> Path:
    out_path = rollout_dir / "actions.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"])
        for row in actions:
            writer.writerow([row["step"], *row["action"]])
    return out_path


def write_stats_json(stats: dict, rollout_dir: Path) -> Path:
    out_path = rollout_dir / "stats.json"
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    return out_path


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    add_flower_to_path(args.flower_root.resolve())

    import torch

    from flower.wrappers.hulc_wrapper import HulcWrapper

    device = resolve_device(args.device)
    merged_config = make_calvin_merged_config(args.env_dir, args.flower_root, args.scene, args.seed)
    dataset = make_dataset_adapter(args.env_dir)
    env = HulcWrapper(dataset, device, show_gui=args.show_gui)
    model = load_flower_model(args.checkpoint, device, steps_per_chunk=10)
    rollout_dir = make_rollout_dir(args.rollout_root)

    robot_obs, scene_obs = get_initial_state()
    obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    goal = {"lang_text": args.instruction}

    print(f"FLOWER root: {args.flower_root}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Env config:  {merged_config}")
    print(f"Instruction: {args.instruction!r}")
    print(f"Device:      {device}")
    print(f"Steps:       {args.steps}")
    print(f"Rollout dir: {rollout_dir}")

    start_info = env.get_info()
    info = start_info
    actions = []
    frames = []
    if not args.no_video:
        frames.append(render_frame(env, f"start | {args.instruction}"))

    with torch.no_grad():
        for step_idx in range(args.steps):
            action = model.step(obs, goal)
            obs, _, done, info = env.step(action)
            action_np = action.detach().cpu().view(-1).numpy()
            action_list = action_np.round(6).tolist()
            actions.append({"step": step_idx, "action": action_list, "drawer_joint": drawer_joint(info)})
            print(f"step {step_idx:03d}: action={action_np.round(4).tolist()}")
            if not args.no_video:
                frames.append(render_frame(env, f"step {step_idx:03d} | drawer {drawer_joint(info):.3f}"))
            if done:
                print(f"Environment returned done=True at step {step_idx}.")
                break

    drawer_start = drawer_joint(start_info)
    drawer_end = drawer_joint(info)
    video_path = None if args.no_video else write_video(frames, rollout_dir, args.fps)
    actions_path = write_actions_csv(actions, rollout_dir)
    stats_path = write_stats_json(
        {
            "instruction": args.instruction,
            "requested_steps": args.steps,
            "executed_steps": len(actions),
            "device": str(device),
            "checkpoint": str(args.checkpoint),
            "flower_root": str(args.flower_root),
            "env_config": str(merged_config),
            "video": str(video_path) if video_path else None,
            "actions_csv": str(actions_path),
            "drawer_joint_start": drawer_start,
            "drawer_joint_end": drawer_end,
            "drawer_joint_delta": drawer_end - drawer_start,
            "open_drawer_success_proxy": (drawer_end - drawer_start) >= 0.12,
        },
        rollout_dir,
    )

    print(f"Drawer joint: {drawer_start:.4f} -> {drawer_end:.4f}")
    print(f"Video:        {video_path}")
    print(f"Actions CSV:  {actions_path}")
    print(f"Stats JSON:   {stats_path}")


if __name__ == "__main__":
    main()
