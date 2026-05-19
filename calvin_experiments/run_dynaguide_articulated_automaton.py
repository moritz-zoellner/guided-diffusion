#!/usr/bin/env python3
"""Run DynaGuide-style CALVIN articulated-object tasks with automaton sample-and-rank.

This is the notebook workflow from `automaton_guidance_calvin.ipynb` packaged as a
repeatable paper experiment. It keeps your rollout / automaton world model code,
but matches the DynaGuide articulated-object reset conditions:

- task-specific articulated state from the DynaGuide JSON setup
- random binary endpoints for unspecified articulated objects
- robot starts sampled from DynaGuide's reset-pose files
- randomized block poses from a fresh CALVIN reset, instead of a fixed block setup
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
from typing import Any, Dict, Iterable, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch


RELEVANT_BEHAVIORS = [
    "button_on",
    "button_off",
    "switch_on",
    "switch_off",
    "drawer_open",
    "drawer_close",
    "door_left",
    "door_right",
]

OPPOSITE_LABEL_NAMES = {
    "switch_on": "switch_off",
    "switch_off": "switch_on",
    "button_on": "button_off",
    "button_off": "button_on",
    "drawer_open": "drawer_closed",
    "drawer_closed": "drawer_open",
    "door_left": "door_right",
    "door_right": "door_left",
}

DEFAULT_CONFIG_DIR = Path("calvin_experiments/configs/dynaguide_articulated_objects")
DEFAULT_OUTPUT_ROOT = Path("outputs/calvin_paper/articulated_objects")
DEFAULT_VISUALIZATION_CONFIG = Path("calvin_experiments/configs/visualization_freiburg_style.json")
DEFAULT_POLICY_CKPT = Path("models/model_epoch_280.pth")
DEFAULT_AUTOMATON_CKPT = Path(
    "calvin_experiments/checkpoints/automaton/best_model.pt"
)
VIDEO_FPS = 30


def find_repo_root(start: Path | str | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start)
    for path in (start_path, *start_path.parents):
        if (path / "calvin_experiments" / "calvin_rollout_utils.py").exists():
            return path
    raise FileNotFoundError(f"Could not find guided-diffusion repo root from {start_path}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
for path in [
    REPO_ROOT,
    REPO_ROOT / "robomimic",
    REPO_ROOT / "calvin" / "calvin_env",
    REPO_ROOT / "calvin_experiments",
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.torch_utils as TorchUtils
from calvin_experiments import calvin_rollout_utils as CRU
from calvin_experiments.label_calvin_world_model import label_scene_states_for_names


def repo_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_existing_path(path: Path | str, base_dir: Path | None = None) -> Path:
    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        if base_dir is not None:
            candidates.append(base_dir / raw)
        candidates.append(REPO_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Path not found. Tried: {tried}")


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def format_onehot(values: Sequence[int | float]) -> str:
    return "[" + " ".join(f"{int(v):1d}" for v in values) + "]"


def repeat_obs_batch(obs_tensor: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    if int(n) == 1:
        return obs_tensor
    return {key: value.repeat((int(n),) + (1,) * (value.ndim - 1)) for key, value in obs_tensor.items()}


def unnormalize_action_sequence(policy, action_sequence: torch.Tensor | np.ndarray) -> np.ndarray:
    action_np = CRU.to_numpy(action_sequence).astype(np.float32)
    if policy.action_normalization_stats is None:
        return action_np

    original_shape = action_np.shape
    flat_actions = action_np.reshape(-1, original_shape[-1])
    action_keys = policy.policy.global_config.train.action_keys
    action_shapes = {
        key: policy.action_normalization_stats[key]["offset"].shape[1:]
        for key in policy.action_normalization_stats
    }
    action_dict = PyUtils.vector_to_action_dict(flat_actions, action_shapes=action_shapes, action_keys=action_keys)
    action_dict = ObsUtils.unnormalize_dict(action_dict, normalization_stats=policy.action_normalization_stats)
    return PyUtils.action_dict_to_vector(action_dict, action_keys=action_keys).reshape(original_shape)


class AutomatonGuidance:
    def __init__(self, automaton_path: Path, device):
        self.model, self.stats, self.meta = CRU.automaton_model_for_eval(automaton_path, device)
        self.label_names = list(self.meta["label_names"])
        self.label_thresholds = self.meta.get("label_thresholds")
        self.device = device

    def label_index(self, name: str) -> int:
        if name not in self.label_names:
            raise ValueError(f"Automaton checkpoint has labels {self.label_names}, missing requested target '{name}'")
        return self.label_names.index(name)

    def current_state_and_label(self, env):
        state = env.get_state()
        robot = np.asarray(state["robot"], dtype=np.float32).reshape(-1)
        scene = np.asarray(state["scene"], dtype=np.float32).reshape(-1)
        automaton_state = np.concatenate([robot, scene]).astype(np.float32)
        automaton_label = label_scene_states_for_names(
            scene[None, :], self.label_names, self.label_thresholds
        )[0].astype(np.float32)
        return automaton_state, automaton_label

    def opposite_label_idx(self, target_label_idx: int) -> Optional[int]:
        target_name = self.label_names[int(target_label_idx)]
        opposite_name = OPPOSITE_LABEL_NAMES.get(target_name)
        if opposite_name is None or opposite_name not in self.label_names:
            return None
        return self.label_names.index(opposite_name)

    def score_label_probs(self, label_probs: np.ndarray, target_label_idx: int) -> np.ndarray:
        scores = np.asarray(label_probs, dtype=np.float32)[..., int(target_label_idx)].copy()
        opposite_idx = self.opposite_label_idx(target_label_idx)
        if opposite_idx is not None:
            scores -= np.asarray(label_probs, dtype=np.float32)[..., opposite_idx]
        return scores

    def score_rule_name(self, target_label_idx: int) -> str:
        target_name = self.label_names[int(target_label_idx)]
        opposite_idx = self.opposite_label_idx(target_label_idx)
        if opposite_idx is None:
            return f"p({target_name})"
        return f"p({target_name}) - p({self.label_names[opposite_idx]})"

    def predict_future_label_probs(
        self,
        automaton_state: np.ndarray,
        automaton_label: np.ndarray,
        action_chunks: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        action_chunks = np.asarray(action_chunks, dtype=np.float32)
        n_candidates, _, action_dim = action_chunks.shape
        action_chunk_dim = len(self.stats["actions_mean"])
        if action_chunk_dim % action_dim != 0:
            raise ValueError(f"Automaton action chunk dim {action_chunk_dim} is not divisible by action dim {action_dim}")
        automaton_horizon = action_chunk_dim // action_dim
        if action_chunks.shape[1] < automaton_horizon:
            raise ValueError(f"Policy produced {action_chunks.shape[1]} actions, automaton expects horizon {automaton_horizon}")

        scored_chunks = action_chunks[:, :automaton_horizon, :].reshape(n_candidates, -1)
        states = np.repeat(np.asarray(automaton_state, dtype=np.float32)[None, :], n_candidates, axis=0)
        labels = np.repeat(np.asarray(automaton_label, dtype=np.float32)[None, :], n_candidates, axis=0)

        states_t = torch.as_tensor(
            (states - self.stats["states_mean"]) / self.stats["states_std"],
            device=self.device,
            dtype=torch.float32,
        )
        actions_t = torch.as_tensor(
            (scored_chunks - self.stats["actions_mean"]) / self.stats["actions_std"],
            device=self.device,
            dtype=torch.float32,
        )
        labels_t = torch.as_tensor(labels, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(states_t, actions_t, labels_t)).detach().cpu().numpy()
        return probs, automaton_horizon


def load_reset_poses(task_config: Dict[str, Any], config_path: Path) -> Optional[list[np.ndarray]]:
    reset_poses = task_config.get("reset_poses")
    if reset_poses is None:
        return None
    reset_path = resolve_existing_path(reset_poses, base_dir=config_path.parent)
    poses = load_json(reset_path)
    return [np.asarray(robot_state, dtype=np.float32) for robot_state in poses["robot_states"]]


def dynaguide_scene_from_base(base_scene: Sequence[float], env_setup: Dict[str, float]) -> tuple[np.ndarray, list[bool]]:
    """Apply DynaGuide's articulated reset while preserving sampled block poses."""

    base_scene = np.asarray(base_scene, dtype=np.float32).copy()
    dynaguide_scene, binaries = CRU.generate_reset_state(sim_hold=env_setup)
    # Copy DynaGuide-controlled scalars, but preserve the fresh env's sampled block
    # poses. The original DynaGuide CALVIN fork shuffled movable objects on reset.
    base_scene[:6] = dynaguide_scene[:6]
    return base_scene, binaries


def make_sample_rank_action_provider(policy, guidance: AutomatonGuidance, target_label_name: str, n_candidates: int):
    target_label_idx = guidance.label_index(target_label_name)
    n_candidates = int(n_candidates)
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    def action_provider(obs, env, step):
        automaton_state, automaton_label = guidance.current_state_and_label(env)
        obs_tensor = policy._prepare_observation(obs)
        obs_tensor_rank = repeat_obs_batch(obs_tensor, n_candidates)
        with torch.no_grad():
            action_chunk_n = policy.policy._get_action_trajectory(obs_dict=obs_tensor_rank)
        action_chunk = unnormalize_action_sequence(policy, action_chunk_n)
        candidate_probs, automaton_horizon = guidance.predict_future_label_probs(
            automaton_state, automaton_label, action_chunk
        )
        candidate_scores = guidance.score_label_probs(candidate_probs, target_label_idx)
        selected_idx = int(np.argmax(candidate_scores))
        opposite_idx = guidance.opposite_label_idx(target_label_idx)
        record = {
            "t": int(step),
            "target_label_idx": int(target_label_idx),
            "target_label_name": target_label_name,
            "opposite_label_idx": None if opposite_idx is None else int(opposite_idx),
            "opposite_label_name": None if opposite_idx is None else guidance.label_names[opposite_idx],
            "score_rule": guidance.score_rule_name(target_label_idx),
            "current_label": automaton_label.astype(int).tolist(),
            "selected_idx": selected_idx,
            "selected_score": float(candidate_scores[selected_idx]),
            "selected_target_prob": float(candidate_probs[selected_idx, target_label_idx]),
            "selected_opposite_prob": None if opposite_idx is None else float(candidate_probs[selected_idx, opposite_idx]),
            "pred_probs": candidate_probs[selected_idx].tolist(),
            "candidate_scores": candidate_scores.tolist(),
        }
        return action_chunk[selected_idx, :automaton_horizon, :], record

    return action_provider


def rollout_policy_once(
    *,
    seed: int,
    policy,
    ckpt_dict: Dict[str, Any],
    task_name: str,
    task_config: Dict[str, Any],
    task_config_path: Path,
    guidance: AutomatonGuidance,
    n_candidates: int,
    output_dir: Path,
    rollout_tag: str,
    horizon: int,
    video_cfg: Dict[str, Any],
    save_video: bool = True,
):
    CRU.seed_everything(seed)
    env, base_env_state = CRU.load_fresh_env_from_checkpoint(ckpt_dict, seed=int(seed), suppress_output=True)
    try:
        reset_poses = load_reset_poses(task_config, task_config_path)
        scene, binaries = dynaguide_scene_from_base(base_env_state["scene"], task_config.get("env_setup", {}))
        robot = np.asarray(base_env_state["robot"], dtype=np.float32).copy()
        if reset_poses:
            robot = random.choice(reset_poses).copy()

        policy.start_episode()
        obs = CRU.reset_env_to_scene_robot(env, scene, robot)
        scene_snapshot = CRU.capture_scene_snapshot(env)
        frames = [CRU.render_visual_camera(env, video_cfg)] if save_video else []

        start_state = env.get_state()
        start_scene = np.asarray(start_state["scene"], dtype=np.float32).copy()
        _, label0 = guidance.current_state_and_label(env)
        action_provider = make_sample_rank_action_provider(policy, guidance, task_name, n_candidates)

        actions, rewards, dones, records = [], [], [], []
        scene_states = [start_scene.copy()]
        robot_states = [np.asarray(start_state["robot"], dtype=np.float32).copy()]
        eef_xy = [robot_states[-1][:2].copy()]
        action_queue: list[np.ndarray] = []
        detected_behavior = "none"
        detected_step = -1
        termination_reason = "horizon"
        total_reward = 0.0

        for step in range(int(horizon)):
            if not action_queue:
                new_actions, record = action_provider(obs, env, step)
                action_queue.extend(np.asarray(new_actions, dtype=np.float32))
                records.append(record)
            action = action_queue.pop(0)
            next_obs, reward, done, _ = env.step(action)
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
            if behavior_now != "other":
                detected_behavior = behavior_now
                detected_step = int(step + 1)
                termination_reason = "behavior"
                break

            if done:
                termination_reason = "env_done"
                break
            obs = next_obs
        else:
            step = int(horizon) - 1

        _, labelf = guidance.current_state_and_label(env)
        rollout = {
            "task": task_name,
            "scene_config": task_config.get("name", task_name),
            "seed": int(seed),
            "target_label_name": task_name,
            "behavior": detected_behavior,
            "behavior_step": int(detected_step),
            "success": bool(detected_behavior == task_name),
            "termination_step": int(step + 1),
            "termination_reason": termination_reason,
            "return": float(total_reward),
            "initial_label": label0.astype(int).tolist(),
            "final_label": labelf.astype(int).tolist(),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "scene_states": scene_states,
            "robot_states": robot_states,
            "eef_xy": eef_xy,
            "records": records,
            "scene_snapshot": scene_snapshot,
            "reset_env_setup": dict(task_config.get("env_setup", {})),
            "reset_robot_from_pose_file": bool(reset_poses),
        }
        if save_video:
            CRU.save_rollout_artifacts(rollout, frames, output_dir, rollout_tag, video_cfg, fps=VIDEO_FPS)
        rollout_dir = Path(rollout.get("rollout_dir", output_dir / rollout_tag))
        CRU.plot_rollout_xy(
            [rollout],
            rollout["scene_snapshot"],
            f"{task_name} seed {seed} -> {detected_behavior}",
            save_path=rollout_dir / "rollout_xy.png",
            display_inline=False,
        )
        write_json(
            rollout_dir / "rollout_summary.json",
            {
                "task": task_name,
                "seed": int(seed),
                "success": bool(detected_behavior == task_name),
                "behavior": detected_behavior,
                "behavior_step": int(detected_step),
                "termination_step": int(step + 1),
                "termination_reason": termination_reason,
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": str(rollout.get("video")),
                "trace": str(rollout.get("trace")),
                "topdown_plot": str(rollout_dir / "rollout_xy.png"),
                "records": records,
            },
        )
        return rollout
    finally:
        CRU.close_env_quietly(env)


def task_summary(task_name: str, rollouts: Sequence[Dict[str, Any]], n_candidates: int, horizon: int) -> Dict[str, Any]:
    behavior_counts = Counter(rollout["behavior"] for rollout in rollouts)
    successes = sum(1 for rollout in rollouts if rollout["behavior"] == task_name)
    return {
        "task": task_name,
        "n_rollouts": len(rollouts),
        "n_candidates": int(n_candidates),
        "horizon": int(horizon),
        "success_count": int(successes),
        "success_rate": float(successes / len(rollouts)) if rollouts else 0.0,
        "behavior_counts": dict(behavior_counts),
        "avg_termination_step": float(np.mean([rollout["termination_step"] for rollout in rollouts])) if rollouts else 0.0,
        "rollouts": [
            {
                "seed": rollout["seed"],
                "success": bool(rollout["behavior"] == task_name),
                "behavior": rollout["behavior"],
                "behavior_step": rollout["behavior_step"],
                "termination_step": rollout["termination_step"],
                "termination_reason": rollout["termination_reason"],
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": str(rollout.get("video")),
                "trace": str(rollout.get("trace")),
                "topdown_plot": str(Path(rollout.get("rollout_dir", "")) / "rollout_xy.png"),
            }
            for rollout in rollouts
        ],
    }


def write_summary_tables(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    csv_path = run_dir / "summary_table.csv"
    md_path = run_dir / "summary_table.md"
    rows = [
        {
            "task": item["task"],
            "success_rate": f"{item['success_rate']:.4f}",
            "success_count": item["success_count"],
            "n_rollouts": item["n_rollouts"],
            "avg_termination_step": f"{item['avg_termination_step']:.2f}",
            "behavior_counts": json.dumps(item["behavior_counts"], sort_keys=True),
        }
        for item in summaries
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "success_rate", "success_count", "n_rollouts", "avg_termination_step", "behavior_counts"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w") as f:
        f.write("| task | success_rate | success_count | n_rollouts | avg_termination_step | behavior_counts |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['task']} | {row['success_rate']} | {row['success_count']} | "
                f"{row['n_rollouts']} | {row['avg_termination_step']} | `{row['behavior_counts']}` |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-ckpt", type=Path, default=DEFAULT_POLICY_CKPT)
    parser.add_argument("--automaton-ckpt", type=Path, default=DEFAULT_AUTOMATON_CKPT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--visualization-config", type=Path, default=DEFAULT_VISUALIZATION_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None, help="Run folder name under output-root. Defaults to timestamped sample_rank name.")
    parser.add_argument("--tasks", nargs="+", default=RELEVANT_BEHAVIORS, choices=RELEVANT_BEHAVIORS)
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and planned tasks without loading models or running CALVIN.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = resolve_existing_path(repo_path(args.config_dir))
    visualization_config = resolve_existing_path(repo_path(args.visualization_config))
    policy_ckpt_candidate = repo_path(args.policy_ckpt)
    automaton_ckpt_candidate = repo_path(args.automaton_ckpt)
    task_config_paths = {task: resolve_existing_path(config_dir / f"{task}.json") for task in args.tasks}

    run_name = args.name or f"sample_rank_candidates{args.n_candidates}_horizon{args.horizon}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = repo_path(args.output_root) / run_name

    planned = {
        "policy_ckpt": str(policy_ckpt_candidate),
        "policy_ckpt_exists": policy_ckpt_candidate.exists(),
        "automaton_ckpt": str(automaton_ckpt_candidate),
        "automaton_ckpt_exists": automaton_ckpt_candidate.exists(),
        "config_dir": str(config_dir),
        "visualization_config": str(visualization_config),
        "output_dir": str(run_dir),
        "tasks": list(args.tasks),
        "task_configs": {task: str(path) for task, path in task_config_paths.items()},
        "n_rollouts": int(args.n_rollouts),
        "n_candidates": int(args.n_candidates),
        "horizon": int(args.horizon),
        "seed_start": int(args.seed_start),
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return

    if not policy_ckpt_candidate.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_ckpt_candidate}")
    if not automaton_ckpt_candidate.exists():
        raise FileNotFoundError(
            f"Automaton checkpoint not found: {automaton_ckpt_candidate}. "
            "Pass --automaton-ckpt PATH to a CALVIN automaton run directory or best_model.pt."
        )
    policy_ckpt = resolve_existing_path(policy_ckpt_candidate)
    automaton_ckpt = resolve_existing_path(automaton_ckpt_candidate)
    video_cfg = load_json(visualization_config)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    try:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(policy_ckpt), device=device, verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load --policy-ckpt as a robomimic diffusion-policy checkpoint: {policy_ckpt}. "
            "This file should contain keys like 'algo_name', 'model', 'config', and 'shape_metadata'."
        ) from exc
    policy_epoch = CRU.policy_epoch_from_checkpoint(policy_ckpt)
    if policy_epoch == "epoch_unknown" and ckpt_dict.get("variable_state", {}).get("epoch") is not None:
        policy_epoch = f"epoch{int(ckpt_dict['variable_state']['epoch'])}"
    guidance = AutomatonGuidance(automaton_ckpt, device)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_args.json", planned)

    print("device:", device)
    print("policy:", policy_ckpt)
    print("policy epoch:", policy_epoch)
    print("automaton:", guidance.meta["ckpt_path"])
    print("label order:", guidance.label_names)
    print("output:", run_dir)

    all_task_summaries = []
    for task in args.tasks:
        task_config_path = task_config_paths[task]
        task_config = load_json(task_config_path)
        task_dir = run_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "task_config_resolved.json", task_config)

        print(f"\nTask {task}: config={task_config_path}")
        rollouts = []
        for rollout_idx in range(int(args.n_rollouts)):
            seed = int(args.seed_start) + rollout_idx
            tag = f"rollout_{rollout_idx:03d}_seed_{seed:03d}"
            rollout = rollout_policy_once(
                seed=seed,
                policy=policy,
                ckpt_dict=ckpt_dict,
                task_name=task,
                task_config=task_config,
                task_config_path=task_config_path,
                guidance=guidance,
                n_candidates=int(args.n_candidates),
                output_dir=task_dir,
                rollout_tag=tag,
                horizon=int(args.horizon),
                video_cfg=video_cfg,
                save_video=not args.no_video,
            )
            rollouts.append(rollout)
            print(
                f"  seed {seed:03d}: success={rollout['success']} "
                f"behavior={rollout['behavior']:>12} @ {rollout['behavior_step']:>3}, "
                f"steps={rollout['termination_step']:>3}, final={format_onehot(rollout['final_label'])}"
            )

        summary = task_summary(task, rollouts, args.n_candidates, args.horizon)
        write_json(task_dir / "task_summary.json", summary)
        CRU.plot_rollout_xy(
            rollouts,
            rollouts[0]["scene_snapshot"],
            f"{task} | success {summary['success_count']}/{summary['n_rollouts']} | candidates={args.n_candidates}",
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
            "tasks": all_task_summaries,
        },
    )
    write_summary_tables(run_dir, all_task_summaries)
    print("\nSummary:", run_dir / "summary_table.md")


if __name__ == "__main__":
    main()
