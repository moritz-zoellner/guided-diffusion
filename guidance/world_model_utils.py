import json
import os
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch


# ==============================
# State / geometry helpers
# ==============================

def xyzw_to_wxyz_batch(q: np.ndarray) -> np.ndarray:
	q = np.asarray(q)
	return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_to_6d(q: np.ndarray) -> np.ndarray:
	w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

	r00 = 1 - 2 * (y ** 2 + z ** 2)
	r01 = 2 * (x * y - w * z)

	r10 = 2 * (x * y + w * z)
	r11 = 1 - 2 * (x ** 2 + z ** 2)

	r20 = 2 * (x * z - w * y)
	r21 = 2 * (y * z + w * x)

	return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)


def latest_1d(x: np.ndarray) -> np.ndarray:
	x = np.asarray(x)
	if x.ndim <= 1:
		return x
	return x[-1]


def build_state_from_obs_dict(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
	obj = latest_1d(obs_dict["object"])
	p_can_to_eef = obj[0:3]
	q_can_to_eef = quat_to_6d(xyzw_to_wxyz_batch(obj[3:7][None]))[0]
	p_can = obj[7:10]
	q_can = quat_to_6d(xyzw_to_wxyz_batch(obj[10:14][None]))[0]
	p_eef = latest_1d(obs_dict["robot0_eef_pos"])
	q_eef = quat_to_6d(xyzw_to_wxyz_batch(latest_1d(obs_dict["robot0_eef_quat"])[None]))[0]
	g_pos = latest_1d(obs_dict["robot0_gripper_qpos"])
	if g_pos.size == 0:
		g_pos = np.zeros(2, dtype=np.float32)
	return np.concatenate(
		[p_can_to_eef, q_can_to_eef, p_can, q_can, p_eef, q_eef, g_pos], axis=-1
	).astype(np.float32)


def deconstruct_state(state_29d):
    """
    Breaks down 29D state into semantic components.

    State layout:
      [0:3]   p_can_to_eef (position, relative)
      [3:9]   q_can_to_eef (rotation, 6D)
      [9:12]  p_can (position, absolute)
      [12:18] q_can (rotation, 6D)
      [18:21] p_eef (position, absolute)
      [21:27] q_eef (rotation, 6D)
      [27:29] g_pos (gripper, 2D)

    Returns dict with keys: p_can_to_eef, q_can_to_eef, p_can, q_can, p_eef, q_eef, g_pos
    """
    components = {
        "p_can_to_eef": state_29d[0:3],
        "q_can_to_eef": state_29d[3:9],
        "p_can": state_29d[9:12],
        "q_can": state_29d[12:18],
        "p_eef": state_29d[18:21],
        "q_eef": state_29d[21:27],
        "g_pos": state_29d[27:29],
    }
    return components


def reconstruct_rotation_matrix_from_6d(rotation_6d: np.ndarray) -> np.ndarray:
	r1 = rotation_6d[:3]
	r2 = rotation_6d[3:6]

	r1_norm = r1 / (np.linalg.norm(r1) + 1e-8)
	r2_orth = r2 - np.dot(r1_norm, r2) * r1_norm
	r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-8)
	r3_norm = np.cross(r1_norm, r2_norm)

	return np.stack([r1_norm, r2_norm, r3_norm], axis=1)


def rzz_from_state_29d(state_29d: np.ndarray) -> float:
	q_can_6d = deconstruct_state(state_29d)["q_can"]
	r_can = reconstruct_rotation_matrix_from_6d(q_can_6d)
	return float(r_can[2, 2])


def predict_next_state_from_raw(
	s_t_raw: np.ndarray,
	a_t_raw: np.ndarray,
	predictor: torch.nn.Module,
	stats: Dict[str, np.ndarray],
	device: str,
) -> np.ndarray:
	s_t_norm = (s_t_raw - stats["state_mean"]) / stats["state_std"]
	a_t_norm = (a_t_raw - stats["action_mean"]) / stats["action_std"]

	s_t_norm_t = torch.from_numpy(s_t_norm).float().unsqueeze(0).to(device)
	a_t_norm_t = torch.from_numpy(a_t_norm).float().unsqueeze(0).to(device)

	with torch.no_grad():
		d_pred_n = predictor(s_t_norm_t, a_t_norm_t)[0].cpu().numpy()

	d_pred = d_pred_n * stats["delta_std"] + stats["delta_mean"]
	return s_t_raw + d_pred


# ==============================
# Dynamics model helpers
# ==============================

class DynamicsMLP(torch.nn.Module):
	def __init__(self, state_dim: int = 29, action_dim: int = 7, hidden_dim: int = 256):
		super().__init__()
		input_dim = state_dim + action_dim
		output_dim = state_dim

		self.net = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.SiLU(),
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.SiLU(),
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.SiLU(),
			torch.nn.Linear(hidden_dim, output_dim),
		)

	def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
		x = torch.cat([s_t, a_t], dim=-1)
		return self.net(x)


class BaselineDeltaPredictor(torch.nn.Module):
	def __init__(self, delta_value: np.ndarray):
		super().__init__()
		self.register_buffer("delta", torch.from_numpy(delta_value.astype(np.float32)))

	def forward(self, s_t_n: torch.Tensor, a_t_n: torch.Tensor) -> torch.Tensor:
		batch_size = s_t_n.shape[0]
		return self.delta.unsqueeze(0).expand(batch_size, -1)


def load_model_for_eval(
	model_or_run_path: str,
	device: str,
	predictor_kind: str = "learned",
	state_dim: int = 29,
	action_dim: int = 7,
	hidden_dim: int = 256,
	load_val_trajectories: bool = False,
	build_trajectory_dataset_fn: Optional[Any] = None,
	fallback_val_trajectories: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.nn.Module, Dict[str, np.ndarray], Optional[List[Dict[str, Any]]], Dict[str, str]]:
	if os.path.isdir(model_or_run_path):
		run_dir_local = model_or_run_path
		ckpt_path_local = os.path.join(run_dir_local, "best_model.pt")
	else:
		ckpt_path_local = model_or_run_path
		run_dir_local = os.path.dirname(ckpt_path_local)

	if not os.path.exists(ckpt_path_local):
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_local}")

	provenance_path_local = os.path.join(run_dir_local, "data_provenance.json")
	norm_path_local = os.path.join(run_dir_local, "normalization_stats.npz")

	ckpt_local = torch.load(ckpt_path_local, map_location=device)

	if "normalization_stats" in ckpt_local:
		eval_stats_local = ckpt_local["normalization_stats"]
	else:
		if not os.path.exists(norm_path_local):
			raise FileNotFoundError(
				"No normalization stats in checkpoint and no normalization_stats.npz found."
			)
		z = np.load(norm_path_local)
		eval_stats_local = {
			"state_mean": z["state_mean"],
			"state_std": z["state_std"],
			"action_mean": z["action_mean"],
			"action_std": z["action_std"],
			"delta_mean": z["delta_mean"],
			"delta_std": z["delta_std"],
		}

	for k in ["state_mean", "state_std", "action_mean", "action_std", "delta_mean", "delta_std"]:
		eval_stats_local[k] = np.asarray(eval_stats_local[k], dtype=np.float32)

	if predictor_kind == "learned":
		model_cfg = ckpt_local.get("model_config", {})
		state_dim_local = model_cfg.get("state_dim", state_dim)
		action_dim_local = model_cfg.get("action_dim", action_dim)
		hidden_dim_local = model_cfg.get("hidden_dim", hidden_dim)

		predictor_local = DynamicsMLP(
			state_dim=state_dim_local,
			action_dim=action_dim_local,
			hidden_dim=hidden_dim_local,
		).to(device)
		predictor_local.load_state_dict(ckpt_local["model_state_dict"])
		predictor_local.eval()
	elif predictor_kind == "baseline_mean":
		predictor_local = BaselineDeltaPredictor(np.zeros(state_dim, dtype=np.float32)).to(device)
		predictor_local.eval()
	elif predictor_kind == "baseline_zero":
		zero_delta_value = (-eval_stats_local["delta_mean"] / eval_stats_local["delta_std"]).astype(np.float32)
		predictor_local = BaselineDeltaPredictor(zero_delta_value).to(device)
		predictor_local.eval()
	else:
		raise ValueError(
			f"Unknown predictor_kind='{predictor_kind}'. Use one of: learned, baseline_mean, baseline_zero"
		)

	val_trajectories_local = None
	if load_val_trajectories:
		if os.path.exists(provenance_path_local):
			with open(provenance_path_local, "r", encoding="utf-8") as f:
				prov_local = json.load(f)
			val_demo_refs_local = [(x["path"], x["demo_id"]) for x in prov_local["split"]["val_demos"]]
			if build_trajectory_dataset_fn is None:
				raise RuntimeError(
					"build_trajectory_dataset_fn is required when load_val_trajectories=True and provenance exists."
				)
			val_trajectories_local = build_trajectory_dataset_fn(val_demo_refs_local)
		elif fallback_val_trajectories is not None:
			val_trajectories_local = fallback_val_trajectories
		else:
			raise RuntimeError(
				"No data_provenance.json found and no fallback_val_trajectories were provided."
			)

	eval_meta_local = {
		"run_dir": run_dir_local,
		"ckpt_path": ckpt_path_local,
		"provenance_path": provenance_path_local,
		"norm_path": norm_path_local,
		"predictor_kind": predictor_kind,
	}
	return predictor_local, eval_stats_local, val_trajectories_local, eval_meta_local


# ==============================
# Rollout helpers
# ==============================

def _unwrap_sim(env: Any) -> Any:
	probe = env
	for _ in range(8):
		if hasattr(probe, "sim"):
			return probe.sim
		if not hasattr(probe, "env"):
			break
		probe = probe.env
	raise RuntimeError("Could not locate MuJoCo sim on wrapped env.")


def get_mujoco_rzz_and_pos(env: Any, body_name: str = "Can_main") -> Tuple[float, np.ndarray]:
	sim = _unwrap_sim(env)
	bid = sim.model.body_name2id(body_name)
	rzz = float(sim.data.body_xmat[bid].reshape(3, 3)[2, 2])
	pos = sim.data.body_xpos[bid].copy()
	return rzz, pos


def _json_to_numpy_tree(obj: Any) -> Any:
	if isinstance(obj, dict):
		return {k: _json_to_numpy_tree(v) for k, v in obj.items()}
	if isinstance(obj, list):
		if len(obj) == 0:
			return np.array([], dtype=np.float32)
		if all(isinstance(v, (int, float, bool)) for v in obj):
			return np.asarray(obj)
		if all(isinstance(v, list) for v in obj):
			try:
				return np.asarray(obj)
			except Exception:
				return [_json_to_numpy_tree(v) for v in obj]
		return [_json_to_numpy_tree(v) for v in obj]
	return obj


def load_init_state_json(init_state_path: str) -> Dict[str, Any]:
	with open(init_state_path, "r", encoding="utf-8") as f:
		init_state = json.load(f)
	return _json_to_numpy_tree(init_state)


def rollout_policy_for_rzz_analysis(
	env: Any,
	policy: Any,
	build_state_from_obs_fn: Any,
	horizon: int,
	video_path: Optional[str] = None,
	video_skip: int = 1,
	camera_name: str = "frontview",
	mujoco_body_name: str = "Can_main",
	init_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	policy.start_episode()
	obs = env.reset()
	if init_state is not None:
		obs = env.reset_to(init_state)
	else:
		state0 = env.get_state()
		obs = env.reset_to(state0)

	states = [build_state_from_obs_fn(obs)]
	actions: List[np.ndarray] = []
	rewards: List[float] = []
	rzz_mujoco: List[float] = []
	can_pos_mujoco: List[np.ndarray] = []

	rzz0, pos0 = get_mujoco_rzz_and_pos(env, body_name=mujoco_body_name)
	rzz_mujoco.append(rzz0)
	can_pos_mujoco.append(pos0)

	video_writer = imageio.get_writer(video_path, fps=20) if video_path is not None else None

	total_reward = 0.0
	success = False
	steps_executed = 0

	try:
		for step_i in range(horizon):
			act = policy(obs)
			next_obs, r, done, _ = env.step(act)

			actions.append(np.asarray(act, dtype=np.float32))
			rewards.append(float(r))
			states.append(build_state_from_obs_fn(next_obs))

			rzz_now, pos_now = get_mujoco_rzz_and_pos(env, body_name=mujoco_body_name)
			rzz_mujoco.append(rzz_now)
			can_pos_mujoco.append(pos_now)

			total_reward += float(r)
			steps_executed = step_i + 1

			if video_writer is not None and step_i % video_skip == 0:
				frame = env.render(mode="rgb_array", height=512, width=512, camera_name=camera_name)
				video_writer.append_data(frame)

			is_success = env.is_success()
			success = bool(is_success["task"]) if isinstance(is_success, dict) and "task" in is_success else bool(is_success)
			if done or success:
				break
			obs = next_obs
	finally:
		if video_writer is not None:
			video_writer.close()

	return {
		"states": states,
		"actions": actions,
		"rewards": rewards,
		"rzz_mujoco": rzz_mujoco,
		"can_pos_mujoco": can_pos_mujoco,
		"total_reward": total_reward,
		"success": success,
		"steps_executed": steps_executed,
		"video_path": video_path,
	}


# ==============================
# Plot helpers
# ==============================

def plot_rzz_world_model_branches(
	states: List[np.ndarray],
	actions: List[np.ndarray],
	predictor: torch.nn.Module,
	stats: Dict[str, np.ndarray],
	device: str,
	branch_horizon: int = 8,
	branch_stride: int = 8,
	branch_colors: Optional[List[str]] = None,
	title: str = "Rzz: true rollout with world-model branches",
	ax: Optional[Any] = None,
	show: bool = True,
	true_label: str = "true Rzz (29D)",
	branch_label: str = "8-step prediction branches",
	branch_alpha: float = 1.0,
	show_legend: bool = True,
) -> Any:
	import matplotlib.pyplot as plt

	true_rzz_29d = np.asarray([rzz_from_state_29d(s) for s in states], dtype=np.float32)
	t_actions = len(actions)
	colors = branch_colors if branch_colors is not None else ["#e41a1c", "#ff7f00"]

	if ax is None:
		fig, ax = plt.subplots(figsize=(12, 5))
	else:
		fig = ax.figure
	ax.plot(
		np.arange(len(true_rzz_29d)),
		true_rzz_29d,
		color="black",
		linewidth=2.3,
		label=true_label,
		zorder=2,
	)

	first_branch = True
	for branch_idx, t0 in enumerate(range(0, t_actions, branch_stride)):
		s_roll = np.asarray(states[t0], dtype=np.float32).copy()
		branch_rzz = [true_rzz_29d[t0]]
		k_max = min(branch_horizon, t_actions - t0)

		for k in range(k_max):
			a_k = actions[t0 + k]
			s_roll = predict_next_state_from_raw(s_roll, a_k, predictor, stats, device)
			branch_rzz.append(rzz_from_state_29d(s_roll))

		x_branch = np.arange(t0, t0 + len(branch_rzz))
		color = colors[branch_idx % len(colors)]
		ax.plot(
			x_branch,
			np.asarray(branch_rzz, dtype=np.float32),
			color=color,
			linewidth=1.8,
			alpha=branch_alpha,
			label=branch_label if first_branch else None,
			zorder=4,
		)
		first_branch = False

	ax.set_title(title)
	ax.set_xlabel("timestep")
	ax.set_ylabel("Rzz")
	ax.grid(alpha=0.3)
	if show_legend:
		ax.legend()
	fig.tight_layout()
	if show:
		plt.show()
	return ax
