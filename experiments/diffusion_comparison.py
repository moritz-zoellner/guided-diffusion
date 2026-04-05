import argparse
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

import robosuite


REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDANCE_DIR = REPO_ROOT / "guidance"
for _path in (REPO_ROOT, GUIDANCE_DIR):
	if str(_path) not in sys.path:
		sys.path.insert(0, str(_path))

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import torch_utils as TorchUtils

from world_model_utils import (
	build_state_from_obs_dict,
	get_mujoco_rzz_and_pos,
	load_init_state_json,
	load_model_for_eval,
)


DEFAULT_CKPT_PATH = str(REPO_ROOT / "models/model_epoch_1100_low_dim_v15_success_0.7.pth")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/can_rollouts/comparison"
DEFAULT_POLICY_CONFIG_PATH = REPO_ROOT / "configs/diffusion_policy.json"
DEFAULT_WORLD_MODEL_RUN_PATH = REPO_ROOT / "models/dynamics/baseline_mlp"
DEFAULT_INIT_STATE_DIRS = [
	REPO_ROOT / "outputs/can_rollouts/snr/n16h300r60g1bYfYv1",
	REPO_ROOT / "outputs/can_rollouts/snr/n16h300r60g1bYfYv3",
	REPO_ROOT / "outputs/can_rollouts/snr/n16h300r60g1bYfYv4",
	REPO_ROOT / "outputs/can_rollouts/snr/n16h300r60g1bYfYv5",
	REPO_ROOT / "outputs/can_rollouts/snr/n8h300r30g1bYfY",
]


@dataclass(frozen=True)
class MethodSpec:
	slug: str
	label: str
	kind: str
	guidance_scale: Optional[float] = None
	rank_k: int = 8
	rank_recompute_interval: int = 8
	rank_reinject_horizon: int = 8


def _ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def _to_numpy_tree(value: Any) -> Any:
	if isinstance(value, dict):
		return {key: _to_numpy_tree(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_to_numpy_tree(item) for item in value]
	if isinstance(value, tuple):
		return tuple(_to_numpy_tree(item) for item in value)
	if torch.is_tensor(value):
		return value.detach().cpu().numpy().copy()
	if isinstance(value, np.ndarray):
		return value.copy()
	return value


def to_jsonable(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {key: to_jsonable(item) for key, item in value.items()}
	if isinstance(value, list):
		return [to_jsonable(item) for item in value]
	if isinstance(value, tuple):
		return [to_jsonable(item) for item in value]
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if torch.is_tensor(value):
		return value.detach().cpu().numpy().tolist()
	return value


def reseed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _load_ckpt_and_initialize_obs_utils(ckpt_path: str) -> Dict[str, Any]:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	cfg = json.loads(ckpt["config"])

	if DEFAULT_POLICY_CONFIG_PATH.exists():
		override_cfg = json.loads(DEFAULT_POLICY_CONFIG_PATH.read_text())
		if "algo" in override_cfg:
			cfg.setdefault("algo", {})
			for key in ("ddpm", "ddim"):
				if key in override_cfg["algo"]:
					cfg["algo"][key] = override_cfg["algo"][key]

	ObsUtils.initialize_obs_utils_with_obs_specs(cfg["observation"]["modalities"])
	ckpt["config"] = json.dumps(cfg)
	return ckpt


def load_policy_and_env(ckpt_path: str, device: str, record_video: bool) -> Tuple[Any, Any, Dict[str, Any]]:
	if not getattr(robosuite, "__version__", None):
		robosuite.__version__ = "1.5.1"

	ckpt = _load_ckpt_and_initialize_obs_utils(ckpt_path)
	env, _ = FileUtils.env_from_checkpoint(
		ckpt_dict=ckpt,
		render=record_video,
		render_offscreen=record_video,
	)
	policy, _ = FileUtils.policy_from_checkpoint(
		ckpt_dict=ckpt,
		device=device,
		verbose=True,
	)
	return env, policy, ckpt


def _success_from_env(env: Any) -> bool:
	value = env.is_success()
	if isinstance(value, dict) and "task" in value:
		return bool(value["task"])
	return bool(value)


def _extract_current_obs_dict(obs_queue: Dict[str, Any]) -> Dict[str, np.ndarray]:
	current = {}
	for key, value in obs_queue.items():
		arr = _to_numpy_tree(value)
		if isinstance(arr, np.ndarray):
			if arr.ndim >= 3:
				current[key] = np.asarray(arr[0, -1]).copy()
			elif arr.ndim == 2:
				current[key] = np.asarray(arr[-1]).copy()
			else:
				current[key] = arr.copy()
		else:
			current[key] = arr
	return current


def _rotation_6d_to_rzz_torch(rot6d: torch.Tensor) -> torch.Tensor:
	r1 = rot6d[..., 0:3]
	r2 = rot6d[..., 3:6]

	r1 = r1 / (torch.linalg.norm(r1, dim=-1, keepdim=True) + 1e-8)
	proj = torch.sum(r2 * r1, dim=-1, keepdim=True)
	r2 = r2 - proj * r1
	r2 = r2 / (torch.linalg.norm(r2, dim=-1, keepdim=True) + 1e-8)
	r3 = torch.cross(r1, r2, dim=-1)
	return r3[..., 2]


def _make_world_model_scoring_helpers(
	predictor: torch.nn.Module,
	stats: Dict[str, np.ndarray],
	device: str,
	rollout_steps: int = 8,
) -> Tuple[Any, Any]:
	state_mean = torch.tensor(stats["state_mean"], device=device, dtype=torch.float32).unsqueeze(0)
	state_std = torch.tensor(stats["state_std"], device=device, dtype=torch.float32).unsqueeze(0)
	action_mean = torch.tensor(stats["action_mean"], device=device, dtype=torch.float32).unsqueeze(0)
	action_std = torch.tensor(stats["action_std"], device=device, dtype=torch.float32).unsqueeze(0)
	delta_mean = torch.tensor(stats["delta_mean"], device=device, dtype=torch.float32).unsqueeze(0)
	delta_std = torch.tensor(stats["delta_std"], device=device, dtype=torch.float32).unsqueeze(0)

	def rollout_score_from_state(state_now: np.ndarray, actions: torch.Tensor) -> torch.Tensor:
		state = torch.as_tensor(state_now, device=device, dtype=torch.float32).unsqueeze(0)
		state = state.expand(actions.shape[0], -1).contiguous()

		horizon = min(int(rollout_steps), int(actions.shape[1]))
		rzz_traj = []

		for step_idx in range(horizon):
			action_t = actions[:, step_idx, :]
			state_n = (state - state_mean) / state_std
			action_n = (action_t - action_mean) / action_std
			delta_n = predictor(state_n, action_n)
			delta = delta_n * delta_std + delta_mean
			state = state + delta
			rzz_traj.append(_rotation_6d_to_rzz_torch(state[:, 12:18]))

		rzz_stack = torch.stack(rzz_traj, dim=1)
		return rzz_stack.mean(dim=1)

	def guidance_function(states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
		current_obs = _extract_current_obs_dict(states)
		state_now = build_state_from_obs_dict(current_obs)
		objective = rollout_score_from_state(state_now, actions).mean()
		return torch.autograd.grad(objective, actions, retain_graph=False, create_graph=False)[0]

	return rollout_score_from_state, guidance_function


def _sample_full_action_candidates(policy: Any, obs: Dict[str, Any], num_candidates: int) -> torch.Tensor:
	obs_torch = policy._prepare_observation(obs)
	candidates = []
	for _ in range(num_candidates):
		with torch.no_grad():
			candidates.append(policy.policy.get_full_action(obs_torch))
	return torch.cat(candidates, dim=0)


def _write_json(path: Path, payload: Any) -> None:
	path.write_text(json.dumps(to_jsonable(payload), indent=2))


def _summarize_rollout(
	method: MethodSpec,
	init_dir: Path,
	seed: int,
	rollout: Dict[str, Any],
	combo_dir: Path,
) -> Dict[str, Any]:
	rzz_values = np.asarray(rollout["rzz_mujoco"], dtype=np.float32)
	pos_values = np.asarray(rollout["can_pos_mujoco"], dtype=np.float32)
	return {
		"method_slug": method.slug,
		"method_label": method.label,
		"method_kind": method.kind,
		"guidance_scale": method.guidance_scale,
		"rank_k": method.rank_k,
		"init_dir": str(init_dir),
		"init_name": init_dir.name,
		"seed": int(seed),
		"combo_dir": str(combo_dir),
		"video_path": rollout["video_path"],
		"steps_executed": int(rollout["steps_executed"]),
		"horizon_requested": int(rollout["horizon_requested"]),
		"success": bool(rollout["success"]),
		"total_reward": float(rollout["total_reward"]),
		"final_rzz": float(rzz_values[-1]) if len(rzz_values) else None,
		"mean_rzz": float(np.mean(rzz_values)) if len(rzz_values) else None,
		"max_rzz": float(np.max(rzz_values)) if len(rzz_values) else None,
		"min_rzz": float(np.min(rzz_values)) if len(rzz_values) else None,
		"final_can_pos_x": float(pos_values[-1, 0]) if len(pos_values) else None,
		"final_can_pos_y": float(pos_values[-1, 1]) if len(pos_values) else None,
		"final_can_pos_z": float(pos_values[-1, 2]) if len(pos_values) else None,
	}


def _save_rollout_artifacts(combo_dir: Path, rollout: Dict[str, Any], summary: Dict[str, Any]) -> None:
	_write_json(combo_dir / "metadata.json", summary)
	_write_json(combo_dir / "rollout_stats.json", summary)

	with open(combo_dir / "rollout_full.pkl", "wb") as f:
		pickle.dump(rollout, f, protocol=pickle.HIGHEST_PROTOCOL)

	np.savez_compressed(
		combo_dir / "rollout_numeric.npz",
		states=np.asarray(rollout["states"], dtype=np.float32),
		actions=np.asarray(rollout["actions"], dtype=np.float32),
		rewards=np.asarray(rollout["rewards"], dtype=np.float32),
		dones=np.asarray(rollout["dones"], dtype=np.bool_),
		success_flags=np.asarray(rollout["success_flags"], dtype=np.bool_),
		rzz_mujoco=np.asarray(rollout["rzz_mujoco"], dtype=np.float32),
		can_pos_mujoco=np.asarray(rollout["can_pos_mujoco"], dtype=np.float32),
	)


def _run_rollout_episode(
	env: Any,
	policy: Any,
	init_state: Dict[str, Any],
	seed: int,
	horizon: int,
	video_path: Optional[Path],
	camera_names: Sequence[str],
	video_skip: int,
	method: MethodSpec,
	score_fn: Any,
	guidance_fn: Any,
	body_name: str = "Can_main",
) -> Dict[str, Any]:
	reseed(seed)
	policy.start_episode()

	env.reset()
	obs = env.reset_to(init_state)

	observations = [_to_numpy_tree(obs)]
	states = [build_state_from_obs_dict(obs)]
	actions: List[np.ndarray] = []
	rewards: List[float] = []
	dones: List[bool] = []
	success_flags: List[bool] = []
	rzz_mujoco: List[float] = []
	can_pos_mujoco: List[np.ndarray] = []
	step_records: List[Dict[str, Any]] = []

	rzz0, pos0 = get_mujoco_rzz_and_pos(env, body_name=body_name)
	rzz_mujoco.append(float(rzz0))
	can_pos_mujoco.append(np.asarray(pos0, dtype=np.float32))

	total_reward = 0.0
	success = False
	video_writer = imageio.get_writer(str(video_path), fps=20) if video_path is not None else None

	try:
		for step_idx in tqdm(range(horizon), leave=False):
			selected_chunk = None
			candidate_scores = None
			selected_candidate_index = None

			if method.kind == "sample_and_rank" and step_idx % method.rank_recompute_interval == 0:
				candidate_actions = _sample_full_action_candidates(policy, obs, method.rank_k)
				current_state = build_state_from_obs_dict(obs)
				candidate_scores_t = score_fn(current_state, candidate_actions)
				selected_candidate_index = int(torch.argmax(candidate_scores_t).item())
				selected_chunk = candidate_actions[selected_candidate_index].detach().cpu()
				policy.policy.set_full_action(selected_chunk[1 : 1 + method.rank_reinject_horizon])
				candidate_scores = candidate_scores_t.detach().cpu().numpy()

			if method.kind == "base":
				act = policy(ob=obs)
			elif method.kind == "guided":
				act = policy(
					ob=obs,
					guidance_function=guidance_fn,
					guidance_type="diffusion",
					guidance_scale=float(method.guidance_scale),
				)
			elif method.kind == "sample_and_rank":
				act = policy(ob=obs)
			else:
				raise ValueError(f"Unknown method kind: {method.kind}")

			next_obs, reward, done, _ = env.step(act)
			next_state = build_state_from_obs_dict(next_obs)
			rzz_now, pos_now = get_mujoco_rzz_and_pos(env, body_name=body_name)
			success = _success_from_env(env)

			actions.append(np.asarray(act, dtype=np.float32))
			rewards.append(float(reward))
			dones.append(bool(done))
			success_flags.append(bool(success))
			states.append(next_state)
			observations.append(_to_numpy_tree(next_obs))
			rzz_mujoco.append(float(rzz_now))
			can_pos_mujoco.append(np.asarray(pos_now, dtype=np.float32))
			total_reward += float(reward)

			step_record = {
				"step_index": int(step_idx),
				"state": states[-2],
				"action": np.asarray(act, dtype=np.float32),
				"next_state": next_state,
				"reward": float(reward),
				"done": bool(done),
				"success": bool(success),
				"rzz_mujoco": float(rzz_now),
				"can_pos_mujoco": np.asarray(pos_now, dtype=np.float32),
			}
			if selected_candidate_index is not None:
				step_record["selected_candidate_index"] = int(selected_candidate_index)
				step_record["candidate_scores"] = np.asarray(candidate_scores, dtype=np.float32)
				step_record["selected_action_chunk"] = np.asarray(selected_chunk, dtype=np.float32)
			step_records.append(step_record)

			if video_writer is not None and step_idx % video_skip == 0:
				frame_parts = [
					env.render(mode="rgb_array", height=512, width=512, camera_name=camera_name)
					for camera_name in camera_names
				]
				frame = frame_parts[0] if len(frame_parts) == 1 else np.concatenate(frame_parts, axis=1)
				video_writer.append_data(frame)

			if done or success:
				break

			obs = next_obs
	finally:
		if video_writer is not None:
			video_writer.close()

	return {
		"method_slug": method.slug,
		"method_label": method.label,
		"method_kind": method.kind,
		"guidance_scale": method.guidance_scale,
		"rank_k": method.rank_k,
		"seed": int(seed),
		"horizon_requested": int(horizon),
		"steps_executed": int(len(actions)),
		"total_reward": float(total_reward),
		"success": bool(success),
		"observations": observations,
		"states": states,
		"actions": actions,
		"rewards": rewards,
		"dones": dones,
		"success_flags": success_flags,
		"rzz_mujoco": rzz_mujoco,
		"can_pos_mujoco": can_pos_mujoco,
		"step_records": step_records,
		"video_path": str(video_path) if video_path is not None else None,
	}


def _build_method_specs(guidance_scales: Sequence[float], sample_and_rank_k: int) -> List[MethodSpec]:
	specs = [MethodSpec(slug="dp_base", label="Base DP", kind="base")]
	specs.append(
		MethodSpec(
			slug=f"snr_k{sample_and_rank_k}",
			label=f"SNR k={sample_and_rank_k}",
			kind="sample_and_rank",
			rank_k=sample_and_rank_k,
			rank_recompute_interval=8,
			rank_reinject_horizon=8,
		)
	)
	for scale in guidance_scales:
		scale_label = int(scale) if float(scale).is_integer() else scale
		specs.append(
			MethodSpec(
				slug=f"dp_guidance_l{scale_label}",
				label=f"Guidance λ={scale_label}",
				kind="guided",
				guidance_scale=float(scale),
			)
		)
	return specs


def run_comparison(args: argparse.Namespace) -> None:
	device = TorchUtils.get_torch_device(try_to_use_cuda=True)
	run_name = args.run_name
	output_root = _ensure_dir(Path(args.output_path) if run_name is None else Path(args.output_path) / run_name)

	env, policy, ckpt = load_policy_and_env(args.ckpt_path, device=device, record_video=(args.record_video == "y"))
	predictor, stats, _, eval_meta = load_model_for_eval(
		model_or_run_path=args.world_model_run_path,
		predictor_kind="learned",
		device=device,
		load_val_trajectories=False,
	)
	score_fn, guidance_fn = _make_world_model_scoring_helpers(
		predictor=predictor,
		stats=stats,
		device=device,
		rollout_steps=args.guidance_rollout_steps,
	)

	method_specs = _build_method_specs(args.guidance_scales, args.sample_and_rank_k)
	init_state_dirs = [Path(item) for item in args.init_state_dirs]
	seeds = [int(seed) for seed in args.seeds]
	init_states = {}
	for init_dir in init_state_dirs:
		init_state_path = init_dir / "init_state.json"
		if not init_state_path.exists():
			raise FileNotFoundError(f"Missing init_state.json: {init_state_path}")
		init_states[str(init_dir)] = load_init_state_json(str(init_state_path))

	print(f"Saving comparison outputs to: {output_root.resolve()}")
	total_jobs = len(method_specs) * len(init_state_dirs) * len(seeds)
	job_index = 0

	for method in method_specs:
		method_dir = _ensure_dir(output_root / method.slug)
		method_rows_path = method_dir / "rollout_rows.jsonl"
		for init_dir in init_state_dirs:
			init_state = init_states[str(init_dir)]
			init_name = init_dir.name
			for seed in seeds:
				job_index += 1
				combo_dir = _ensure_dir(method_dir / f"{init_name}__seed_{seed:02d}")
				video_path = combo_dir / "video.mp4" if args.record_video == "y" else None

				print(f"[{job_index}/{total_jobs}] {method.label} | init={init_name} | seed={seed}")
				rollout = _run_rollout_episode(
					env=env,
					policy=policy,
					init_state=init_state,
					seed=seed,
					horizon=args.horizon,
					video_path=video_path,
					camera_names=[args.camera_name],
					video_skip=args.video_skip,
					method=method,
					score_fn=score_fn,
					guidance_fn=guidance_fn,
				)
				summary = _summarize_rollout(method, init_dir, seed, rollout, combo_dir)
				_save_rollout_artifacts(combo_dir, rollout, summary)
				with open(method_rows_path, "a", encoding="utf-8") as f:
					f.write(json.dumps(to_jsonable(summary)) + "\n")

	print("\nComparison complete.")
	print(f"Raw rollout rows saved under: {output_root}")


def _rollout_single_method(
	args: argparse.Namespace,
	method: MethodSpec,
	output_path: str,
	run_name: Optional[str],
	use_world_model_guidance: bool,
) -> None:
	device = TorchUtils.get_torch_device(try_to_use_cuda=True)
	run_id = run_name or args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
	run_output_path = _ensure_dir(Path(output_path) / run_id)

	env, policy, _ = load_policy_and_env(args.ckpt_path, device=device, record_video=(args.record_video == "y"))

	score_fn = None
	guidance_fn = None
	if use_world_model_guidance:
		predictor, stats, _, _ = load_model_for_eval(
			model_or_run_path=args.world_model_run_path,
			predictor_kind="learned",
			device=device,
			load_val_trajectories=False,
		)
		score_fn, guidance_fn = _make_world_model_scoring_helpers(
			predictor=predictor,
			stats=stats,
			device=device,
			rollout_steps=args.guidance_rollout_steps,
		)

	init_state = None
	if args.fix_init_state == "y":
		env.reset()
		init_state = env.get_state()
		_write_json(run_output_path / "init_state.json", init_state)

	video_dir = None
	if args.record_video == "y":
		video_dir = _ensure_dir(run_output_path / "video")

	all_stats: List[Dict[str, Any]] = []
	print("Starting Rollouts")

	for rollout_i in range(args.n_rollouts):
		rollout_num = rollout_i + 1
		make_video = args.record_video == "y" and rollout_num % args.n_step_rollout_video == 0
		video_path = video_dir / f"rollout_{rollout_num}.mp4" if make_video and video_dir is not None else None

		rollout_start_time = time.perf_counter()
		rollout = _run_rollout_episode(
			env=env,
			policy=policy,
			init_state=init_state,
			seed=getattr(args, "seed", 0),
			horizon=args.horizon,
			video_path=video_path,
			camera_names=args.camera_names,
			video_skip=args.video_skip,
			method=method,
			score_fn=score_fn,
			guidance_fn=guidance_fn,
		)
		rollout_end_time = time.perf_counter()

		rollout_row = {
			"Return": rollout["total_reward"],
			"Horizon": rollout["steps_executed"],
			"Success_Rate": float(rollout["success"]),
			"Obs_List": rollout["observations"],
			"State_List": rollout["states"],
			"Action_List": rollout["actions"],
			"Rzz_List": rollout["rzz_mujoco"],
			"Pos_List": rollout["can_pos_mujoco"],
			"Step_Records": rollout["step_records"],
			"Mode": method.slug,
		}
		all_stats.append(rollout_row)
		_write_json(run_output_path / "rollout_stats.json", all_stats)

		summary = _summarize_rollout(method, run_output_path, getattr(args, "seed", 0), rollout, run_output_path)
		_save_rollout_artifacts(run_output_path, rollout, summary)

		rollout_seconds = rollout_end_time - rollout_start_time
		print(
			f"[rollout {rollout_num}/{args.n_rollouts}] "
			f"time_s={rollout_seconds:.2f} "
			f"video_made={make_video} "
			f"stats: { {key: (len(value) if key in ['Obs_List', 'State_List', 'Action_List', 'Rzz_List', 'Pos_List', 'Step_Records'] else value) for key, value in rollout_row.items()} }"
		)

	if video_dir is not None:
		print(
			f"Completed {args.n_rollouts} rollouts. Outputs: stats={run_output_path / 'rollout_stats.json'}, "
			f"videos={video_dir}, run_dir={run_output_path}"
		)
	else:
		print(
			f"Completed {args.n_rollouts} rollouts. Outputs: stats={run_output_path / 'rollout_stats.json'}, run_dir={run_output_path}"
		)


def run_base_policy(args: argparse.Namespace) -> None:
	method = MethodSpec(slug="base_policy", label="Base Policy", kind="base")
	_rollout_single_method(
		args=args,
		method=method,
		output_path=args.output_path,
		run_name=args.name,
		use_world_model_guidance=False,
	)


def run_sample_and_rank(args: argparse.Namespace) -> None:
	use_world_model_guidance = True
	method = MethodSpec(
		slug=f"sample_and_rank_k{args.num_samples}",
		label=f"Sample-and-rank k={args.num_samples}",
		kind="sample_and_rank",
		rank_k=args.num_samples,
		rank_recompute_interval=args.Ta,
		rank_reinject_horizon=args.Ta,
	)
	_rollout_single_method(
		args=args,
		method=method,
		output_path=args.output_path,
		run_name=args.name,
		use_world_model_guidance=use_world_model_guidance,
	)


def main_base_policy() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--horizon", type=int, default=700)
	parser.add_argument("--n_rollouts", type=int, default=1)
	parser.add_argument("--n_step_rollout_video", type=int, default=10)
	parser.add_argument("--record_video", type=str, choices=["y", "n"], default="y")
	parser.add_argument("--video_skip", type=int, default=1)
	parser.add_argument("--camera_names", nargs="+", default=["frontview"])
	parser.add_argument("--output_path", type=str, default=str(REPO_ROOT / "outputs/sr_rollout"))
	parser.add_argument("--name", type=str, default=None)
	parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
	parser.add_argument("--fix_init_state", type=str, choices=["y", "n"], default="n")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--world_model_run_path", type=str, default=str(DEFAULT_WORLD_MODEL_RUN_PATH))
	parser.add_argument("--guidance_rollout_steps", type=int, default=8)
	args = parser.parse_args()
	run_base_policy(args)


def main_sample_and_rank() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_samples", type=int, default=16)
	parser.add_argument("--goal", type=int, default=1)
	parser.add_argument("--Ta", type=int, default=8)
	parser.add_argument("--T_eval", type=int, default=15)
	parser.add_argument("--horizon", type=int, default=700)
	parser.add_argument("--n_rollouts", type=int, default=1)
	parser.add_argument("--n_step_rollout_video", type=int, default=1)
	parser.add_argument("--record_video", type=str, choices=["y", "n"], default="y")
	parser.add_argument("--video_skip", type=int, default=1)
	parser.add_argument("--camera_names", nargs="+", default=["frontview"])
	parser.add_argument("--output_path", type=str, default=str(REPO_ROOT / "runs/sr_rollout"))
	parser.add_argument("--name", type=str, default=None)
	parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
	parser.add_argument("--fix_init_state", type=str, choices=["y", "n"], default="n")
	parser.add_argument("--get_baseline", type=str, choices=["y", "n"], default="n")
	parser.add_argument("--ranking_mode", type=str, choices=["env", "world_model"], default="world_model")
	parser.add_argument("--world_model_run_path", type=str, default=str(DEFAULT_WORLD_MODEL_RUN_PATH))
	parser.add_argument("--guidance_rollout_steps", type=int, default=8)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()
	run_sample_and_rank(args)


def _comparison_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()
	parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_ROOT))
	parser.add_argument("--run_name", type=str, default=None)
	parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
	parser.add_argument("--world_model_run_path", type=str, default=str(DEFAULT_WORLD_MODEL_RUN_PATH))
	parser.add_argument("--horizon", type=int, default=300)
	parser.add_argument("--video_skip", type=int, default=1)
	parser.add_argument("--camera_name", type=str, default="frontview")
	parser.add_argument("--record_video", type=str, choices=["y", "n"], default="y")
	parser.add_argument("--sample_and_rank_k", type=int, default=8)
	parser.add_argument("--guidance_rollout_steps", type=int, default=8)
	parser.add_argument(
		"--guidance_scales",
		nargs="+",
		type=float,
		default=[25.0, 50.0, 75.0, 100.0],
	)
	parser.add_argument(
		"--init_state_dirs",
		nargs="+",
		default=[str(path) for path in DEFAULT_INIT_STATE_DIRS],
	)
	parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
	return parser


def main_comparison() -> None:
	args = _comparison_arg_parser().parse_args()
	run_comparison(args)


if __name__ == "__main__":
	main_comparison()
