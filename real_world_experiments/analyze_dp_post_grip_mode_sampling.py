"""Re-sample the real-world DP at post-grip decision states.

This script is for diagnosing whether automaton sequence failures come from
missing diffusion-policy candidate support or from automaton mis-ranking.
It reconstructs the exact two-frame observation stack used at a logged
``chunk_sample`` row, queries the DP repeatedly from that frozen input, and
scores the fresh samples with the automaton world model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROBOMIMIC_ROOT = REPO_ROOT / "robomimic"
if str(ROBOMIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOMIMIC_ROOT))

from calvin_experiments.calvin_rollout_utils import automaton_model_for_eval
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import python_utils as PyUtils


OBS_KEYS = ("eef_pos", "eef_rot6d", "gripper_binary", "cheezit_pos", "cheezit_rot6d")
LABEL_NAMES = ("can_grabbed", "pouring_right", "pouring_left")
RIGHT_IDX = 1
LEFT_IDX = 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def obs_from_json(obs: dict) -> dict[str, np.ndarray]:
    return {key: np.asarray(obs[key], dtype=np.float32) for key in OBS_KEYS}


def obs_to_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs[key].reshape(-1) for key in OBS_KEYS]).astype(np.float32)


def reconstruct_stack_for_chunk(events_path: Path, target_chunk_idx: int, obs_horizon: int = 2):
    """Return the online obs stack and row for a logged chunk_sample.

    The ROS node appends observations at rollout_start and target_reached.
    It samples the next chunk from the existing deque before logging the
    chunk_sample row, so this replay mirrors that ordering.
    """

    obs_deque: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
    for row in load_jsonl(events_path):
        row_type = row.get("type")
        if row_type == "rollout_start":
            obs_deque.append(obs_from_json(row["obs"]))
        elif row_type == "target_reached":
            obs_deque.append(obs_from_json(row["reached_obs"]))
        elif row_type == "chunk_sample" and int(row["chunk_idx"]) == int(target_chunk_idx):
            fallback = obs_from_json(row["obs"])
            obs_list = list(obs_deque) or [fallback]
            while len(obs_list) < obs_horizon:
                obs_list = [obs_list[0]] + obs_list
            stacked = {
                key: np.stack([obs[key] for obs in obs_list], axis=0).astype(np.float32)
                for key in OBS_KEYS
            }
            return stacked, obs_from_json(row["obs"]), np.asarray(row["current_label"], dtype=np.float32), row
    raise ValueError(f"chunk_sample chunk_idx={target_chunk_idx} not found in {events_path}")


def first_chain_step_chunk(events_path: Path, chain_pos: int = 1) -> dict:
    for row in load_jsonl(events_path):
        if row.get("type") != "chunk_sample":
            continue
        selection = row.get("selection") or {}
        if selection.get("mode") != "automaton_sample_rank":
            continue
        if int(selection.get("chain_pos", -1)) == int(chain_pos):
            return row
    raise ValueError(f"No chunk_sample with chain_pos={chain_pos} in {events_path}")


def unnormalize_action_chunks(policy, action_tensor: torch.Tensor) -> np.ndarray:
    action_np = action_tensor.detach().cpu().numpy().astype(np.float32)
    if policy.action_normalization_stats is None:
        return action_np
    original_shape = action_np.shape
    flat = action_np.reshape(-1, original_shape[-1])
    action_keys = policy.policy.global_config.train.action_keys
    action_shapes = {
        key: policy.action_normalization_stats[key]["offset"].shape[1:]
        for key in policy.action_normalization_stats
    }
    action_dict = PyUtils.vector_to_action_dict(flat, action_shapes=action_shapes, action_keys=action_keys)
    action_dict = ObsUtils.unnormalize_dict(action_dict, normalization_stats=policy.action_normalization_stats)
    return PyUtils.action_dict_to_vector(action_dict, action_keys=action_keys).reshape(original_shape)


def repeat_obs_batch(obs_tensor: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    if n == 1:
        return obs_tensor
    return {key: value.repeat((n,) + (1,) * (value.ndim - 1)) for key, value in obs_tensor.items()}


def score_automaton(model, stats, device, step_obs, label, action_chunks):
    n_candidates, _, action_dim = action_chunks.shape
    action_chunk_dim = len(stats["actions_mean"])
    automaton_horizon = action_chunk_dim // action_dim
    flat_chunks = action_chunks[:, :automaton_horizon, :].reshape(n_candidates, -1)
    state = obs_to_state(step_obs)
    states = np.repeat(state[None, :], n_candidates, axis=0)
    labels = np.repeat(label[None, :], n_candidates, axis=0)
    states_t = torch.as_tensor(
        (states - stats["states_mean"]) / stats["states_std"],
        device=device,
        dtype=torch.float32,
    )
    actions_t = torch.as_tensor(
        (flat_chunks - stats["actions_mean"]) / stats["actions_std"],
        device=device,
        dtype=torch.float32,
    )
    labels_t = torch.as_tensor(labels, device=device, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(states_t, actions_t, labels_t)).detach().cpu().numpy()
    return probs, automaton_horizon


def summarize_samples(action_chunks: np.ndarray, probs: np.ndarray, target_idx: int) -> dict:
    opposite_idx = LEFT_IDX if target_idx == RIGHT_IDX else RIGHT_IDX
    right = probs[:, RIGHT_IDX]
    left = probs[:, LEFT_IDX]
    target = probs[:, target_idx]
    opposite = probs[:, opposite_idx]
    winner = np.where(right >= left, "right", "left")
    cum = action_chunks.sum(axis=1)
    mean = action_chunks.mean(axis=1)
    return {
        "n": int(len(action_chunks)),
        "winner_counts": dict(Counter(winner.tolist())),
        "right_gt_08": int((right > 0.8).sum()),
        "left_gt_08": int((left > 0.8).sum()),
        "both_right_left_gt_08": int(((right > 0.8) & (left > 0.8)).sum()),
        "target_gt_08": int((target > 0.8).sum()),
        "opposite_gt_08": int((opposite > 0.8).sum()),
        "max_target": float(target.max()),
        "max_opposite": float(opposite.max()),
        "mean_target": float(target.mean()),
        "mean_opposite": float(opposite.mean()),
        "right_prob_quantiles": np.quantile(right, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist(),
        "left_prob_quantiles": np.quantile(left, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist(),
        "mean_dy_quantiles": np.quantile(mean[:, 1], [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist(),
        "mean_drot_x_quantiles": np.quantile(mean[:, 3], [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist(),
        "cum_dy_quantiles": np.quantile(cum[:, 1], [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist(),
        "cum_drot_x_quantiles": np.quantile(cum[:, 3], [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist(),
    }


def sample_case(policy, automaton_model, automaton_stats, device, case, n_queries: int, n_candidates: int):
    obs_tensor = policy._prepare_observation(case["stacked_obs"])
    query_summaries = []
    all_actions = []
    all_probs = []
    for query_idx in range(n_queries):
        obs_batch = repeat_obs_batch(obs_tensor, n_candidates)
        with torch.no_grad():
            action_t = policy.policy._get_action_trajectory(obs_dict=obs_batch)
        actions = unnormalize_action_chunks(policy, action_t)
        probs, automaton_horizon = score_automaton(
            automaton_model,
            automaton_stats,
            device,
            case["step_obs"],
            case["label"],
            actions,
        )
        query_summary = summarize_samples(actions, probs, case["target_idx"])
        query_summary["query_idx"] = int(query_idx)
        query_summary["automaton_horizon"] = int(automaton_horizon)
        query_summaries.append(query_summary)
        all_actions.append(actions)
        all_probs.append(probs)
    all_actions = np.concatenate(all_actions, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    return {
        "queries": query_summaries,
        "aggregate": summarize_samples(all_actions, all_probs, case["target_idx"]),
        "all_actions": all_actions,
        "all_probs": all_probs,
    }


def build_case(run_dir: Path, rollout_idx: int, name: str) -> dict:
    run_config = json.loads((run_dir / "run_config.json").read_text())
    target_idx = int(run_config["target_chain_parsed"][1]["label_idx"])
    events_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}" / "events.jsonl"
    chain_row = first_chain_step_chunk(events_path, chain_pos=1)
    stacked_obs, step_obs, label, row = reconstruct_stack_for_chunk(events_path, chain_row["chunk_idx"])
    return {
        "name": name,
        "run_dir": str(run_dir),
        "rollout_idx": int(rollout_idx),
        "events_path": str(events_path),
        "target_idx": int(target_idx),
        "target_name": LABEL_NAMES[target_idx],
        "chunk_idx": int(row["chunk_idx"]),
        "decision_idx_before": int(row["decision_idx_before"]),
        "logged_selected_candidate": int(row["selected_candidate"]),
        "logged_selection": row["selection"],
        "logged_label_probs": row["label_probs"],
        "logged_candidate_summary": summarize_samples(
            np.asarray(row["candidate_action_chunks"], dtype=np.float32),
            np.asarray(row["label_probs"], dtype=np.float32),
            target_idx,
        ),
        "current_label": label.astype(int).tolist(),
        "wrist_joint_delta_deg": row.get("wrist_joint_delta_deg"),
        "stacked_obs": stacked_obs,
        "step_obs": step_obs,
        "label": label,
    }


def plot_case(output_path: Path, case: dict, sampled: dict):
    probs = sampled["all_probs"]
    actions = sampled["all_actions"]
    mean = actions.mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(probs[:, RIGHT_IDX], probs[:, LEFT_IDX], s=18, alpha=0.7)
    axes[0].axvline(0.8, color="tab:orange", linestyle="--", linewidth=1)
    axes[0].axhline(0.8, color="tab:blue", linestyle="--", linewidth=1)
    axes[0].set_xlabel("p(pouring_right)")
    axes[0].set_ylabel("p(pouring_left)")
    axes[0].set_title(f"{case['name']}: automaton scores")
    axes[0].grid(True, alpha=0.3)

    color = np.where(probs[:, RIGHT_IDX] >= probs[:, LEFT_IDX], "tab:orange", "tab:blue")
    axes[1].scatter(mean[:, 1], mean[:, 3], c=color, s=18, alpha=0.7)
    axes[1].axvline(0.0, color="0.4", linewidth=1)
    axes[1].axhline(0.0, color="0.4", linewidth=1)
    axes[1].set_xlabel("mean dy over chunk")
    axes[1].set_ylabel("mean drot_x over chunk")
    axes[1].set_title("fresh DP chunks colored by score winner")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def reconstruct_stacks_at_target_reached(events_path: Path, decision_indices: set[int], obs_horizon: int = 2):
    obs_deque: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
    out = {}
    for row in load_jsonl(events_path):
        row_type = row.get("type")
        if row_type == "rollout_start":
            obs_deque.append(obs_from_json(row["obs"]))
        elif row_type == "target_reached":
            step_obs = obs_from_json(row["reached_obs"])
            obs_deque.append(step_obs)
            decision_idx = int(row["decision_idx"])
            if decision_idx in decision_indices:
                obs_list = list(obs_deque)
                while len(obs_list) < obs_horizon:
                    obs_list = [obs_list[0]] + obs_list
                stacked = {
                    key: np.stack([obs[key] for obs in obs_list], axis=0).astype(np.float32)
                    for key in OBS_KEYS
                }
                out[decision_idx] = {
                    "stacked_obs": stacked,
                    "step_obs": step_obs,
                    "label": np.asarray(row["label"], dtype=np.float32),
                    "wrist_joint_delta_deg": row.get("wrist_joint_delta_deg"),
                    "label_events": row.get("label_events", []),
                }
    missing = sorted(decision_indices - set(out))
    if missing:
        raise ValueError(f"Could not reconstruct target_reached stacks for decisions {missing} in {events_path}")
    return out


def grip_to_first_resample_decisions(events_path: Path):
    """Return decisions from the can-grabbed flip through first chain-1 sample."""

    grip_decision = None
    first_chain1_decision_before = None
    for row in load_jsonl(events_path):
        if row.get("type") == "target_reached":
            for event in row.get("label_events", []):
                if (
                    int(event.get("label_idx", -1)) == 0
                    and int(event.get("from", -1)) == 0
                    and int(event.get("to", -1)) == 1
                ):
                    grip_decision = int(row["decision_idx"])
        elif row.get("type") == "chunk_sample":
            selection = row.get("selection") or {}
            if selection.get("mode") == "automaton_sample_rank" and int(selection.get("chain_pos", -1)) == 1:
                first_chain1_decision_before = int(row["decision_idx_before"])
                break
    if grip_decision is None or first_chain1_decision_before is None:
        raise ValueError(f"Could not find grip->chain1 interval in {events_path}")
    return list(range(grip_decision, first_chain1_decision_before + 1))


def sample_timeline(policy, automaton_model, automaton_stats, device, run_dir: Path, rollout_idx: int, target_idx: int, n_queries: int, n_candidates: int):
    events_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}" / "events.jsonl"
    decisions = grip_to_first_resample_decisions(events_path)
    stacks = reconstruct_stacks_at_target_reached(events_path, set(decisions))
    rows = []
    for decision_idx in decisions:
        item = stacks[decision_idx]
        case = {
            "name": f"rollout_{rollout_idx:03d}_decision_{decision_idx:03d}",
            "target_idx": target_idx,
            "target_name": LABEL_NAMES[target_idx],
            "stacked_obs": item["stacked_obs"],
            "step_obs": item["step_obs"],
            "label": item["label"],
        }
        sampled = sample_case(
            policy,
            automaton_model,
            automaton_stats,
            device,
            case,
            n_queries=n_queries,
            n_candidates=n_candidates,
        )
        agg = sampled["aggregate"]
        opposite_idx = LEFT_IDX if target_idx == RIGHT_IDX else RIGHT_IDX
        probs = sampled["all_probs"]
        target = probs[:, target_idx]
        opposite = probs[:, opposite_idx]
        margin = target - opposite
        rows.append({
            "decision_idx": int(decision_idx),
            "current_label": item["label"].astype(int).tolist(),
            "wrist_joint_delta_deg": item["wrist_joint_delta_deg"],
            "label_events": item["label_events"],
            "n": int(len(target)),
            "target_name": LABEL_NAMES[target_idx],
            "opposite_name": LABEL_NAMES[opposite_idx],
            "target_gt_02": int((target > 0.2).sum()),
            "target_gt_05": int((target > 0.5).sum()),
            "target_gt_08": int((target > 0.8).sum()),
            "opposite_gt_02": int((opposite > 0.2).sum()),
            "opposite_gt_05": int((opposite > 0.5).sum()),
            "opposite_gt_08": int((opposite > 0.8).sum()),
            "target_wins": int((target > opposite).sum()),
            "max_target": float(target.max()),
            "q50_target": float(np.quantile(target, 0.5)),
            "q90_target": float(np.quantile(target, 0.9)),
            "max_opposite": float(opposite.max()),
            "q50_opposite": float(np.quantile(opposite, 0.5)),
            "q90_opposite": float(np.quantile(opposite, 0.9)),
            "max_margin": float(margin.max()),
            "q50_margin": float(np.quantile(margin, 0.5)),
            "q90_margin": float(np.quantile(margin, 0.9)),
            "winner_counts": agg["winner_counts"],
            "mean_dy_quantiles": agg["mean_dy_quantiles"],
            "mean_drot_x_quantiles": agg["mean_drot_x_quantiles"],
        })
    return rows


def plot_timeline(output_path: Path, title: str, rows: list[dict]):
    x = np.asarray([row["decision_idx"] for row in rows], dtype=float)
    n = np.asarray([row["n"] for row in rows], dtype=float)
    target_frac_02 = np.asarray([row["target_gt_02"] / row["n"] for row in rows], dtype=float)
    target_frac_win = np.asarray([row["target_wins"] / row["n"] for row in rows], dtype=float)
    max_target = np.asarray([row["max_target"] for row in rows], dtype=float)
    q90_target = np.asarray([row["q90_target"] for row in rows], dtype=float)
    max_margin = np.asarray([row["max_margin"] for row in rows], dtype=float)
    q90_margin = np.asarray([row["q90_margin"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, max_target, marker="o", label="max p(target)", color="tab:green")
    axes[0].plot(x, q90_target, marker=".", label="q90 p(target)", color="tab:olive")
    axes[0].plot(x, target_frac_02, marker="s", label="frac p(target)>0.2", color="tab:blue")
    axes[0].plot(x, target_frac_win, marker="^", label="frac target wins", color="tab:purple")
    axes[0].axhline(0.2, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("target support")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(x, max_margin, marker="o", label="max margin target-opposite", color="tab:red")
    axes[1].plot(x, q90_margin, marker=".", label="q90 margin", color="tab:pink")
    axes[1].axhline(0.0, color="0.3", linestyle="--", linewidth=1)
    axes[1].set_xlabel("decision index after target_reached")
    axes[1].set_ylabel("p(target)-p(opposite)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def mode_delta_summary(action_chunks: np.ndarray, probs: np.ndarray) -> dict:
    right = probs[:, RIGHT_IDX]
    left = probs[:, LEFT_IDX]
    delta = left - right
    mean_action = action_chunks.mean(axis=1)
    return {
        "n": int(len(delta)),
        "delta_min": float(delta.min()),
        "delta_q10": float(np.quantile(delta, 0.10)),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.quantile(delta, 0.50)),
        "delta_q90": float(np.quantile(delta, 0.90)),
        "delta_max": float(delta.max()),
        "right_prob_min": float(right.min()),
        "right_prob_mean": float(right.mean()),
        "right_prob_max": float(right.max()),
        "left_prob_min": float(left.min()),
        "left_prob_mean": float(left.mean()),
        "left_prob_max": float(left.max()),
        "right_gt_02": int((right > 0.2).sum()),
        "right_gt_05": int((right > 0.5).sum()),
        "right_gt_08": int((right > 0.8).sum()),
        "left_gt_02": int((left > 0.2).sum()),
        "left_gt_05": int((left > 0.5).sum()),
        "left_gt_08": int((left > 0.8).sum()),
        "right_wins": int((right > left).sum()),
        "left_wins": int((left > right).sum()),
        "both_abs_support_gt_02": bool((right > 0.2).any() and (left > 0.2).any()),
        "both_win_support": bool((right > left).any() and (left > right).any()),
        "locked_left_by_02": bool((right > 0.2).sum() == 0 and (left > 0.2).sum() > 0),
        "locked_right_by_02": bool((left > 0.2).sum() == 0 and (right > 0.2).sum() > 0),
        "all_left_wins": bool((left > right).all()),
        "all_right_wins": bool((right > left).all()),
        "mean_dy_mean": float(mean_action[:, 1].mean()),
        "mean_drot_x_mean": float(mean_action[:, 3].mean()),
    }


def sample_mode_delta_at_state(policy, automaton_model, automaton_stats, device, stacked_obs, step_obs, label, n_queries: int, n_candidates: int):
    obs_tensor = policy._prepare_observation(stacked_obs)
    all_actions = []
    all_probs = []
    for _ in range(n_queries):
        obs_batch = repeat_obs_batch(obs_tensor, n_candidates)
        with torch.no_grad():
            action_t = policy.policy._get_action_trajectory(obs_dict=obs_batch)
        actions = unnormalize_action_chunks(policy, action_t)
        probs, _ = score_automaton(automaton_model, automaton_stats, device, step_obs, label, actions)
        all_actions.append(actions)
        all_probs.append(probs)
    actions = np.concatenate(all_actions, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    return mode_delta_summary(actions, probs)


def available_reached_decisions(events_path: Path) -> set[int]:
    out = set()
    for row in load_jsonl(events_path):
        if row.get("type") == "target_reached":
            out.add(int(row["decision_idx"]))
    return out


def sample_grip_offsets_for_run(policy, automaton_model, automaton_stats, device, run_dir: Path, n_queries: int, max_offset: int):
    run_config = json.loads((run_dir / "run_config.json").read_text())
    n_candidates = int(run_config["n_candidates"])
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = []
    for rollout in summary["rollouts"]:
        rollout_idx = int(rollout["rollout_idx"])
        events_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}" / "events.jsonl"
        decisions = grip_to_first_resample_decisions(events_path)
        grip_decision = decisions[0]
        reached = available_reached_decisions(events_path)
        wanted = {
            grip_decision + offset
            for offset in range(max_offset + 1)
            if grip_decision + offset in reached
        }
        stacks = reconstruct_stacks_at_target_reached(events_path, wanted)
        for offset in range(max_offset + 1):
            decision_idx = grip_decision + offset
            if decision_idx not in stacks:
                continue
            item = stacks[decision_idx]
            stats = sample_mode_delta_at_state(
                policy,
                automaton_model,
                automaton_stats,
                device,
                item["stacked_obs"],
                item["step_obs"],
                item["label"],
                n_queries=n_queries,
                n_candidates=n_candidates,
            )
            rows.append({
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "rollout_idx": rollout_idx,
                "offset_after_grip": int(offset),
                "decision_idx": int(decision_idx),
                "grip_decision_idx": int(grip_decision),
                "n_queries": int(n_queries),
                "n_candidates_per_query": int(n_candidates),
                "first_event_sequence": " -> ".join(e["label_name"] for e in rollout.get("label_events", [])),
                "success": bool(rollout.get("success", False)),
                "termination_reason": rollout.get("termination_reason"),
                "current_label": item["label"].astype(int).tolist(),
                "wrist_joint_delta_deg": item["wrist_joint_delta_deg"],
                **stats,
            })
            print(
                f"{run_dir.name} rollout {rollout_idx:03d} offset {offset}: "
                f"delta[min/mean/max]={stats['delta_min']:.3f}/{stats['delta_mean']:.3f}/{stats['delta_max']:.3f}, "
                f"right>0.2={stats['right_gt_02']}/{stats['n']}, left>0.2={stats['left_gt_02']}/{stats['n']}"
            )
    return rows


def aggregate_grip_offset_rows(rows: list[dict]) -> list[dict]:
    out = []
    offsets = sorted({row["offset_after_grip"] for row in rows})
    for offset in offsets:
        sub = [row for row in rows if row["offset_after_grip"] == offset]
        if not sub:
            continue
        out.append({
            "offset_after_grip": int(offset),
            "n_states": int(len(sub)),
            "delta_mean_mean": float(np.mean([row["delta_mean"] for row in sub])),
            "delta_mean_std": float(np.std([row["delta_mean"] for row in sub])),
            "delta_min_min": float(np.min([row["delta_min"] for row in sub])),
            "delta_max_max": float(np.max([row["delta_max"] for row in sub])),
            "frac_both_abs_support_gt_02": float(np.mean([row["both_abs_support_gt_02"] for row in sub])),
            "frac_both_win_support": float(np.mean([row["both_win_support"] for row in sub])),
            "frac_locked_left_by_02": float(np.mean([row["locked_left_by_02"] for row in sub])),
            "frac_locked_right_by_02": float(np.mean([row["locked_right_by_02"] for row in sub])),
            "frac_all_left_wins": float(np.mean([row["all_left_wins"] for row in sub])),
            "frac_all_right_wins": float(np.mean([row["all_right_wins"] for row in sub])),
            "mean_right_gt_02_frac": float(np.mean([row["right_gt_02"] / row["n"] for row in sub])),
            "mean_left_gt_02_frac": float(np.mean([row["left_gt_02"] / row["n"] for row in sub])),
            "mean_right_wins_frac": float(np.mean([row["right_wins"] / row["n"] for row in sub])),
            "mean_left_wins_frac": float(np.mean([row["left_wins"] / row["n"] for row in sub])),
        })
    return out


def plot_all_grip_offsets(output_dir: Path, rows: list[dict], aggregate: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        x = row["offset_after_grip"]
        color = "tab:blue" if "left" in row["first_event_sequence"].split(" -> ")[-1] else "tab:orange"
        ax.vlines(x, row["delta_min"], row["delta_max"], color=color, alpha=0.18, linewidth=1.0)
        ax.scatter([x], [row["delta_mean"]], color=color, alpha=0.45, s=18)
    ax.axhline(0.0, color="0.2", linestyle="--", linewidth=1)
    ax.set_xlabel("offset after can_grabbed flip [executed actions]")
    ax.set_ylabel("p(left) - p(right); min/max bars, mean dot")
    ax.set_title("Fresh DP mode support after gripping, all rollouts")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "all_rollouts_grip_offset_delta_minmax_mean.png", dpi=180)
    plt.close(fig)

    agg_x = np.asarray([row["offset_after_grip"] for row in aggregate], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(agg_x, [row["frac_both_abs_support_gt_02"] for row in aggregate], marker="o", label="both p>0.2 support")
    axes[0].plot(agg_x, [row["frac_both_win_support"] for row in aggregate], marker="o", label="both win support")
    axes[0].plot(agg_x, [row["frac_locked_left_by_02"] for row in aggregate], marker="x", label="locked left by p>0.2")
    axes[0].plot(agg_x, [row["frac_locked_right_by_02"] for row in aggregate], marker="x", label="locked right by p>0.2")
    axes[0].set_ylabel("fraction of rollout states")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].errorbar(
        agg_x,
        [row["delta_mean_mean"] for row in aggregate],
        yerr=[row["delta_mean_std"] for row in aggregate],
        marker="o",
        label="mean delta across states",
    )
    axes[1].axhline(0.0, color="0.2", linestyle="--", linewidth=1)
    axes[1].set_xlabel("offset after can_grabbed flip [executed actions]")
    axes[1].set_ylabel("mean p(left)-p(right)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle("How quickly post-grip states become mode-locked")
    fig.tight_layout()
    fig.savefig(output_dir / "all_rollouts_grip_offset_aggregate.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--right-run",
        type=Path,
        default=Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_right_epoch160_n10"),
    )
    parser.add_argument("--success-rollout", type=int, default=0)
    parser.add_argument("--wrong-route-rollout", type=int, default=7)
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/candidate_mode_analysis/post_grip_dp_resampling"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timeline", action="store_true", help="Also query every reached state from grip flip to first chain-1 resample.")
    parser.add_argument(
        "--all-grip-offsets",
        action="store_true",
        help="Sample all rollouts from both sequence runs at offsets after can_grabbed flips.",
    )
    parser.add_argument(
        "--left-run",
        type=Path,
        default=Path("outputs/real_world/paper_rollouts/automaton_sequence_eval/automaton_left_epoch160_n10_3"),
    )
    parser.add_argument("--max-offset", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((args.right_run / "run_config.json").read_text())
    n_candidates = int(args.n_candidates or run_config["n_candidates"])
    ckpt_path = run_config["ckpt_path"]
    automaton_ckpt_path = run_config["automaton_ckpt_path"]
    device = torch.device(args.device)

    print(f"Loading DP checkpoint on {device}: {ckpt_path}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=str(device), verbose=False)
    policy.start_episode()
    print(f"Loading automaton: {automaton_ckpt_path}")
    automaton_model, automaton_stats, automaton_meta = automaton_model_for_eval(automaton_ckpt_path, device)

    cases = [
        build_case(args.right_run, args.success_rollout, f"go_right_success_rollout_{args.success_rollout:03d}"),
        build_case(args.right_run, args.wrong_route_rollout, f"go_right_wrong_left_rollout_{args.wrong_route_rollout:03d}"),
    ]
    report = {
        "run_dir": str(args.right_run),
        "policy_ckpt": ckpt_path,
        "automaton_ckpt": automaton_ckpt_path,
        "automaton_meta": automaton_meta,
        "n_queries": int(args.n_queries),
        "n_candidates_per_query": int(n_candidates),
        "cases": [],
    }
    for case in cases:
        print(
            f"Sampling {case['name']} at chunk={case['chunk_idx']} "
            f"decision={case['decision_idx_before']} target={case['target_name']}"
        )
        sampled = sample_case(
            policy,
            automaton_model,
            automaton_stats,
            device,
            case,
            n_queries=args.n_queries,
            n_candidates=n_candidates,
        )
        plot_case(args.output_dir / f"{case['name']}_fresh_dp_scores.png", case, sampled)
        clean_case = {
            key: value
            for key, value in case.items()
            if key not in {"stacked_obs", "step_obs", "label"}
        }
        clean_case["aggregate_fresh_dp_summary"] = sampled["aggregate"]
        clean_case["per_query_fresh_dp_summary"] = sampled["queries"]
        clean_case["stacked_obs_debug"] = {
            key: value.astype(float).tolist() for key, value in case["stacked_obs"].items()
        }
        clean_case["step_obs_debug"] = {
            key: value.astype(float).tolist() for key, value in case["step_obs"].items()
        }
        report["cases"].append(clean_case)
        agg = sampled["aggregate"]
        print(
            f"  logged winner_counts={case['logged_candidate_summary']['winner_counts']}, "
            f"fresh winner_counts={agg['winner_counts']}, "
            f"fresh right>0.8={agg['right_gt_08']}/{agg['n']}, "
            f"left>0.8={agg['left_gt_08']}/{agg['n']}, "
            f"target>0.8={agg['target_gt_08']}/{agg['n']}"
        )

    if args.timeline:
        timeline_report = {}
        target_idx = int(run_config["target_chain_parsed"][1]["label_idx"])
        for rollout_idx, name in [
            (args.success_rollout, f"go_right_success_rollout_{args.success_rollout:03d}"),
            (args.wrong_route_rollout, f"go_right_wrong_left_rollout_{args.wrong_route_rollout:03d}"),
        ]:
            print(f"Timeline sampling {name}: grip flip -> first pour-target resample")
            rows = sample_timeline(
                policy,
                automaton_model,
                automaton_stats,
                device,
                args.right_run,
                rollout_idx,
                target_idx,
                n_queries=args.n_queries,
                n_candidates=n_candidates,
            )
            timeline_report[name] = rows
            plot_timeline(
                args.output_dir / f"{name}_grip_to_first_resample_timeline.png",
                f"{name}: target support before first pour-target resample",
                rows,
            )
            for row in rows:
                print(
                    f"  d={row['decision_idx']:03d} "
                    f"target>0.2={row['target_gt_02']}/{row['n']} "
                    f"target_wins={row['target_wins']}/{row['n']} "
                    f"max_target={row['max_target']:.3f} "
                    f"max_margin={row['max_margin']:.3f}"
                )
        timeline_path = args.output_dir / "grip_to_first_resample_timeline_summary.json"
        timeline_path.write_text(json.dumps(timeline_report, indent=2))
        print(f"Wrote {timeline_path}")

    if args.all_grip_offsets:
        all_rows = []
        for run_dir in [args.right_run, args.left_run]:
            all_rows.extend(
                sample_grip_offsets_for_run(
                    policy,
                    automaton_model,
                    automaton_stats,
                    device,
                    run_dir,
                    n_queries=args.n_queries,
                    max_offset=args.max_offset,
                )
            )
        aggregate = aggregate_grip_offset_rows(all_rows)
        out = {
            "policy_ckpt": ckpt_path,
            "automaton_ckpt": automaton_ckpt_path,
            "n_queries": int(args.n_queries),
            "max_offset": int(args.max_offset),
            "rows": all_rows,
            "aggregate_by_offset": aggregate,
            "just_gripped_locked_cases": [
                row for row in all_rows
                if row["offset_after_grip"] == 0
                and (row["locked_left_by_02"] or row["locked_right_by_02"] or row["all_left_wins"] or row["all_right_wins"])
            ],
        }
        grip_dir = args.output_dir / "all_grip_offsets"
        grip_dir.mkdir(parents=True, exist_ok=True)
        (grip_dir / "all_grip_offset_mode_support_summary.json").write_text(json.dumps(out, indent=2))
        plot_all_grip_offsets(grip_dir, all_rows, aggregate)
        print(f"Wrote {grip_dir / 'all_grip_offset_mode_support_summary.json'}")
        print(f"Just-gripped locked-ish cases: {len(out['just_gripped_locked_cases'])}")
        for row in out["just_gripped_locked_cases"]:
            print(
                f"  {row['run_name']} rollout {row['rollout_idx']:03d}: "
                f"delta[min/mean/max]={row['delta_min']:.3f}/{row['delta_mean']:.3f}/{row['delta_max']:.3f}, "
                f"right>0.2={row['right_gt_02']}/{row['n']}, left>0.2={row['left_gt_02']}/{row['n']}, "
                f"right_wins={row['right_wins']}/{row['n']}, left_wins={row['left_wins']}/{row['n']}"
            )

    summary_path = args.output_dir / "post_grip_dp_resampling_summary.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
