#!/usr/bin/env python3
"""Open-loop H64 TeLoGraF rollouts in the Toy Squares TouchCube env."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/scratch/gilbreth/zoellner/guided-diffusion/outputs/telograf/.cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/scratch/gilbreth/zoellner/guided-diffusion/outputs/telograf/.cache")

import imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/scratch/gilbreth/zoellner/guided-diffusion")
DEFAULT_CHECKPOINT = (
    SCRATCH_ROOT
    / "outputs/telograf/toy_squares/runs/toy_squares_reach_h64_full_10k_telograf/checkpoint.pt"
)
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / "outputs/telograf/toy_squares_rollouts"
DEFAULT_ENV_CONFIG = REPO_ROOT / "toy_squares/touchcubes.json"

for path in [REPO_ROOT, REPO_ROOT / "robomimic", REPO_ROOT / "TeLoGraF" / "code"]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gym  # noqa: E402
import robomimic.envs  # noqa: F401,E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
from z_diffuser import GaussianFlow, TemporalUnet  # noqa: E402

from telograf_toy_squares.toy_specs import (  # noqa: E402
    ACTION_DIM,
    DATA_DIM,
    DEFAULT_RADIUS,
    LABEL_NAMES,
    STATE_DIM,
    evaluate_spec_sequence,
    label_states,
    spec_to_vector,
)
from toy_squares.toy_squares_utils import early_decision_cube_setup  # noqa: E402


def paper_early_decision_setup_state(
    *,
    deterministic: bool = True,
    setup_variant: str = "compact_deterministic",
    compact_block_scale: float = 0.55,
) -> np.ndarray:
    """Match toy_squares/baselines/paper_horizon_test.py rollout_setup_state."""
    setup_state = np.asarray(early_decision_cube_setup(deterministic=deterministic), dtype=np.float32).copy()
    variant = str(setup_variant).strip().lower()
    if variant in {"early", "early_decision"}:
        return setup_state
    if variant in {"compact", "compact_early", "compact_deterministic"}:
        positions = setup_state[:10].reshape(5, 2).astype(np.float32)
        center = np.array([256.0, 256.0], dtype=np.float32)
        positions[1:] = center + float(compact_block_scale) * (positions[1:] - center)
        setup_state[:10] = positions.reshape(-1)
        if deterministic and setup_state.shape[0] >= 14:
            setup_state[10:14] = 0.0
        return setup_state
    raise ValueError(f"Unknown setup_variant={setup_variant!r}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def obs_to_state(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)[:2],
            np.asarray(obs["states"], dtype=np.float32).reshape(-1)[:8],
        ],
        axis=0,
    ).astype(np.float32)


def make_eventual_spec(label: str) -> Dict[str, Any]:
    return {
        "id": f"eventual_{label}",
        "type": "eventual",
        "labels": [str(label)],
        "formula": f"F reach_{label}",
        "radius": float(DEFAULT_RADIUS),
    }


class TelografToySquaresPolicy:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.checkpoint = checkpoint
        self.stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
        self.horizon = int(checkpoint["horizon"])
        self.state_dim = int(checkpoint["state_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.condition_dim = int(checkpoint["condition_dim"])
        self.spec_dim = int(checkpoint["spec_dim"])

        if self.state_dim != STATE_DIM or self.action_dim != ACTION_DIM:
            raise ValueError(
                f"Expected ToySquares state/action dims {STATE_DIM}/{ACTION_DIM}, "
                f"got {self.state_dim}/{self.action_dim}"
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

    def sample_open_loop_actions(self, state: np.ndarray, target_label: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape {(self.state_dim,)}, got {state.shape}")
        spec = make_eventual_spec(target_label)
        spec_vec = spec_to_vector(spec).astype(np.float32)
        if len(spec_vec) != self.spec_dim:
            raise ValueError(f"Spec vector dim mismatch: expected {self.spec_dim}, got {len(spec_vec)}")
        state_norm = (state - self.stats["state_mean"]) / self.stats["state_std"]
        cond_np = np.concatenate([state_norm, spec_vec], axis=0).astype(np.float32)
        if len(cond_np) != self.condition_dim:
            raise ValueError(f"Condition dim mismatch: expected {self.condition_dim}, got {len(cond_np)}")

        cond = torch.from_numpy(cond_np[None, :]).to(self.device)
        with torch.no_grad():
            sample = self.diffuser.conditional_sample(
                cond,
                args=SimpleNamespace(flow_pattern=13),
            ).trajectories[0]
        transitions_norm = sample.detach().cpu().numpy().astype(np.float32)
        transitions = transitions_norm * self.stats["transition_std"] + self.stats["transition_mean"]
        actions = transitions[:, self.state_dim : self.state_dim + self.action_dim].astype(np.float32)
        return actions, transitions.astype(np.float32), spec


def create_env(env_config: Path):
    with Path(env_config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    ObsUtils.initialize_obs_utils_with_obs_specs(cfg["obs_specs"])
    return gym.make(cfg["env_meta"]["env_name"])


def render_frame(env, size: int) -> np.ndarray:
    frame = env.render(mode="rgb_array", height=int(size), width=int(size))
    return np.asarray(frame, dtype=np.uint8)


def rollout_once(
    *,
    env,
    policy: TelografToySquaresPolicy,
    target_label: str,
    seed: int,
    output_dir: Path,
    video_size: int,
    video_fps: int,
    clip_actions: bool,
    setup_state: np.ndarray,
) -> Dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if hasattr(env, "seed"):
        env.seed(int(seed))

    rollout_dir = output_dir / target_label / f"rollout_seed_{int(seed):03d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    video_path = rollout_dir / f"{target_label}_rollout_seed_{int(seed):03d}.mp4"

    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset_to"):
        env.reset()
        obs = env.unwrapped.reset_to(np.asarray(setup_state, dtype=np.float32).copy())
    elif hasattr(env, "reset_to"):
        obs = env.reset_to(np.asarray(setup_state, dtype=np.float32).copy())
    else:
        obs = env.reset(np.asarray(setup_state, dtype=np.float32).copy())
    initial_state = obs_to_state(obs)
    actions, planned_transitions, spec = policy.sample_open_loop_actions(initial_state, target_label)

    states = [initial_state.copy()]
    executed_actions = []
    rewards = []
    dones = []
    contacts = []
    frames = [render_frame(env, video_size)]
    label_success = bool(label_states(np.asarray(states), radius=float(spec["radius"]))[-1, LABEL_NAMES.index(target_label)] > 0.5)
    behavior_success = False
    done = False

    for step_idx, raw_action in enumerate(actions):
        action = np.asarray(raw_action, dtype=np.float32).reshape(policy.action_dim)
        if clip_actions:
            action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action)
        state = obs_to_state(obs)
        states.append(state.copy())
        executed_actions.append(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        contact = int(info.get("cube_contacted", -1)) if isinstance(info, dict) else -1
        contacts.append(contact)
        frames.append(render_frame(env, video_size))

        labels = label_states(np.asarray(states), radius=float(spec["radius"]))
        label_success = bool(labels[-1, LABEL_NAMES.index(target_label)] > 0.5)
        behavior_success = behavior_success or contact == LABEL_NAMES.index(target_label)
        if label_success or behavior_success or done:
            break

    imageio.mimsave(video_path, frames, fps=int(video_fps))
    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(executed_actions, dtype=np.float32)
    labels_np = label_states(states_np, radius=float(spec["radius"]))
    offline_ok, offline_score = evaluate_spec_sequence(spec, states_np)
    planned_states = planned_transitions[:, : policy.state_dim]
    planned_ok, planned_score = evaluate_spec_sequence(spec, planned_states)

    np.savez_compressed(
        rollout_dir / "rollout_trace.npz",
        states=states_np,
        actions=actions_np,
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        contacts=np.asarray(contacts, dtype=np.int32),
        labels=labels_np.astype(np.float32),
        planned_actions=np.asarray(actions, dtype=np.float32),
        planned_transitions=planned_transitions.astype(np.float32),
        setup_state=np.asarray(setup_state, dtype=np.float32),
    )
    np.save(rollout_dir / "setup_state.npy", np.asarray(setup_state, dtype=np.float32))
    summary = {
        "target_label": target_label,
        "seed": int(seed),
        "steps": int(len(executed_actions)),
        "done": bool(done),
        "actual_label_success": bool(label_success),
        "label_success": bool(label_success),
        "behavior_success": bool(behavior_success),
        "imagined_telograf_success": bool(planned_ok),
        "offline_spec_success": bool(offline_ok),
        "offline_spec_score": float(offline_score),
        "planned_spec_success": bool(planned_ok),
        "planned_spec_score": float(planned_score),
        "return": float(np.sum(rewards)) if rewards else 0.0,
        "video_path": str(video_path),
        "rollout_dir": str(rollout_dir),
        "clip_actions": bool(clip_actions),
        "setup_variant": "paper_early_decision",
    }
    write_json(rollout_dir / "rollout_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--targets", nargs="+", default=list(LABEL_NAMES), choices=list(LABEL_NAMES))
    parser.add_argument("--n-rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--video-size", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--no-action-clip", action="store_true")
    parser.add_argument(
        "--setup-variant",
        default="compact_deterministic",
        choices=["early_decision", "compact_deterministic"],
        help="Paper ToySquares layout variant. compact_deterministic matches the paper_horizon_test default.",
    )
    parser.add_argument("--compact-block-scale", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    run_name = args.name or f"telograf_toy_squares_h64_open_loop_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    policy = TelografToySquaresPolicy(args.checkpoint, device)
    env = create_env(args.env_config)
    setup_state = paper_early_decision_setup_state(
        deterministic=True,
        setup_variant=args.setup_variant,
        compact_block_scale=float(args.compact_block_scale),
    )
    np.save(output_dir / "setup_state.npy", setup_state.astype(np.float32))

    rows = []
    try:
        for target in args.targets:
            for rollout_idx in range(int(args.n_rollouts)):
                seed = int(args.seed) + rollout_idx
                print(f"rollout target={target} idx={rollout_idx} seed={seed}", flush=True)
                rows.append(
                    rollout_once(
                        env=env,
                        policy=policy,
                        target_label=target,
                        seed=seed,
                        output_dir=output_dir,
                        video_size=args.video_size,
                        video_fps=args.video_fps,
                        clip_actions=not args.no_action_clip,
                        setup_state=setup_state,
                    )
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    by_target: Dict[str, Dict[str, Any]] = {}
    for target in args.targets:
        target_rows = [row for row in rows if row["target_label"] == target]
        by_target[target] = {
            "n": int(len(target_rows)),
            "actual_label_success_rate": float(np.mean([row["actual_label_success"] for row in target_rows])) if target_rows else 0.0,
            "label_success_rate": float(np.mean([row["actual_label_success"] for row in target_rows])) if target_rows else 0.0,
            "behavior_success_rate": float(np.mean([row["behavior_success"] for row in target_rows])) if target_rows else 0.0,
            "imagined_telograf_success_rate": float(np.mean([row["imagined_telograf_success"] for row in target_rows])) if target_rows else 0.0,
            "offline_success_rate": float(np.mean([row["offline_spec_success"] for row in target_rows])) if target_rows else 0.0,
            "planned_success_rate": float(np.mean([row["planned_spec_success"] for row in target_rows])) if target_rows else 0.0,
            "mean_steps": float(np.mean([row["steps"] for row in target_rows])) if target_rows else 0.0,
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(output_dir),
        "targets": list(args.targets),
        "n_rollouts": int(args.n_rollouts),
        "horizon": int(policy.horizon),
        "device": str(device),
        "setup_variant": str(args.setup_variant),
        "setup_source": "toy_squares/baselines/paper_horizon_test.py rollout_setup_state default logic",
        "setup_state": setup_state.astype(float).tolist(),
        "by_target": by_target,
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["target_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output_dir": str(output_dir), "by_target": by_target}, indent=2), flush=True)


if __name__ == "__main__":
    main()
