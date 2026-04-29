from __future__ import annotations

import json
import os
from datetime import datetime
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.collections import LineCollection
from matplotlib.patches import RegularPolygon


class AutomatonMLP(nn.Module):
    def __init__(
        self,
        state_dim=10,
        label_dim=4,
        action_chunk_dim=16,
        state_hidden=16,
        label_hidden=4,
        action_hidden=16,
        hidden_dim=32,
    ):
        super().__init__()
        self.state_enc = nn.Sequential(nn.Linear(state_dim, state_hidden), nn.SiLU(), nn.Linear(state_hidden, state_hidden), nn.SiLU())
        self.label_enc = nn.Sequential(nn.Linear(label_dim, label_hidden), nn.SiLU(), nn.Linear(label_hidden, label_hidden), nn.SiLU())
        self.action_enc = nn.Sequential(nn.Linear(action_chunk_dim, action_hidden), nn.SiLU(), nn.Linear(action_hidden, action_hidden), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(state_hidden + label_hidden + action_hidden, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, s_t, a_chunk, label_t):
        z = torch.cat([self.state_enc(s_t), self.action_enc(a_chunk), self.label_enc(label_t)], dim=-1)
        return self.head(z)


class BaselineLogitPredictor(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, s_t, a_chunk, label_t):
        batch_size = s_t.shape[0]
        return self.logits.unsqueeze(0).expand(batch_size, -1)


def _label_mean_from_trajectories(trajectories, key="next_labels"):
    label_arrays = [np.asarray(tr[key], dtype=np.float32) for tr in trajectories if key in tr]
    if not label_arrays:
        raise ValueError(f"Could not compute label mean: no '{key}' arrays were found.")
    return np.mean(np.concatenate(label_arrays, axis=0), axis=0)


def load_automaton_model_for_eval(
    model_or_run_path,
    device,
    predictor_kind="learned",
    state_dim=10,
    label_dim=4,
    action_chunk_dim=16,
    state_hidden=128,
    label_hidden=32,
    action_hidden=128,
    hidden_dim=256,
    load_val_trajectories=False,
    build_trajectory_dataset_fn=None,
    fallback_val_trajectories=None,
    state_components=None,
    add_labels_fn=None,
    label_predicates=None,
):
    if os.path.isdir(model_or_run_path):
        run_dir_local = Path(model_or_run_path)
        ckpt_path_local = run_dir_local / "best_model.pt"
    else:
        ckpt_path_local = Path(model_or_run_path)
        run_dir_local = ckpt_path_local.parent

    if not ckpt_path_local.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_local}")

    ckpt_local = torch.load(ckpt_path_local, map_location=device, weights_only=False)
    state_dict_local = ckpt_local["model_state_dict"]

    eval_stats_local = ckpt_local.get("normalization_stats")
    if eval_stats_local is None:
        norm_path_local = run_dir_local / "normalization_stats.npz"
        if not norm_path_local.exists():
            raise FileNotFoundError(
                f"No normalization stats found in checkpoint and no file at {norm_path_local}."
            )
        z = np.load(norm_path_local)
        eval_stats_local = {k: z[k] for k in z.files}

    for k in [
        "states_mean",
        "states_std",
        "actions_mean",
        "actions_std",
        "labels_mean",
        "labels_std",
        "next_labels_mean",
        "next_labels_std",
    ]:
        if k in eval_stats_local:
            eval_stats_local[k] = np.asarray(eval_stats_local[k], dtype=np.float32)

    state_dim_local = int(state_dict_local["state_enc.0.weight"].shape[1])
    state_hidden_local = int(state_dict_local["state_enc.0.weight"].shape[0])
    label_dim_local = int(state_dict_local["label_enc.0.weight"].shape[1])
    label_hidden_local = int(state_dict_local["label_enc.0.weight"].shape[0])
    action_chunk_dim_local = int(state_dict_local["action_enc.0.weight"].shape[1])
    action_hidden_local = int(state_dict_local["action_enc.0.weight"].shape[0])
    hidden_dim_local = int(state_dict_local["head.0.weight"].shape[0])

    if predictor_kind == "learned":
        predictor_local = AutomatonMLP(
            state_dim=state_dim_local,
            label_dim=label_dim_local,
            action_chunk_dim=action_chunk_dim_local,
            state_hidden=state_hidden_local,
            label_hidden=label_hidden_local,
            action_hidden=action_hidden_local,
            hidden_dim=hidden_dim_local,
        ).to(device)
        predictor_local.load_state_dict(state_dict_local)
        predictor_local.eval()
    elif predictor_kind == "baseline_zero":
        predictor_local = BaselineLogitPredictor(np.zeros(label_dim_local, dtype=np.float32)).to(device)
        predictor_local.eval()
    elif predictor_kind == "baseline_mean":
        label_mean = ckpt_local.get("label_mean")
        if label_mean is None:
            if fallback_val_trajectories is not None:
                label_mean = _label_mean_from_trajectories(fallback_val_trajectories, key="next_labels")
            else:
                raise RuntimeError(
                    "No label_mean found in the checkpoint and no trajectories were provided to compute it."
                )
        label_mean = np.asarray(label_mean, dtype=np.float32)
        label_mean = np.clip(label_mean, 1e-4, 1.0 - 1e-4)
        label_logits = np.log(label_mean / (1.0 - label_mean))
        predictor_local = BaselineLogitPredictor(label_logits.astype(np.float32)).to(device)
        predictor_local.eval()
    else:
        raise ValueError(
            f"Unknown predictor_kind='{predictor_kind}'. Use one of: learned, baseline_zero, baseline_mean"
        )

    val_trajectories_local = None
    if load_val_trajectories:
        provenance_path_local = run_dir_local / "data_provenance.json"
        if provenance_path_local.exists():
            with open(provenance_path_local, "r", encoding="utf-8") as f:
                prov_local = json.load(f)
            val_demo_refs_local = [(x["path"], x["demo_id"]) for x in prov_local["val_demos"]]
            if build_trajectory_dataset_fn is None:
                raise RuntimeError(
                    "build_trajectory_dataset_fn is required when load_val_trajectories=True and provenance exists."
                )
            val_trajectories_local = build_trajectory_dataset_fn(
                val_demo_refs_local,
                state_components=state_components if state_components is not None else [],
            )
            if val_trajectories_local and "labels" not in val_trajectories_local[0]:
                if add_labels_fn is None or label_predicates is None:
                    raise RuntimeError(
                        "Loaded trajectories are missing labels; provide add_labels_fn and label_predicates."
                    )
                val_trajectories_local = add_labels_fn(val_trajectories_local, label_predicates)
        elif fallback_val_trajectories is not None:
            val_trajectories_local = fallback_val_trajectories
        else:
            raise RuntimeError(
                "No data_provenance.json found and no fallback_val_trajectories were provided."
            )

    eval_meta_local = {
        "run_dir": str(run_dir_local),
        "ckpt_path": str(ckpt_path_local),
        "provenance_path": str(run_dir_local / "data_provenance.json"),
        "norm_path": str(run_dir_local / "normalization_stats.npz"),
        "predictor_kind": predictor_kind,
    }
    return predictor_local, eval_stats_local, val_trajectories_local, eval_meta_local


def jiggle(deterministic: bool = False, scale: float = 1.0) -> float:
    if deterministic:
        return 0.0
    return scale * (2.0 * (np.random.random() - 0.5))


def early_decision_cube_setup(deterministic: bool = False) -> np.ndarray:
    cube_1 = [128 + 30 * jiggle(deterministic), 128 + 30 * jiggle(deterministic)]
    cube_2 = [128 + 30 * jiggle(deterministic), 384 + 30 * jiggle(deterministic)]
    cube_4 = [384 + 30 * jiggle(deterministic), 128 + 30 * jiggle(deterministic)]
    cube_3 = [384 + 30 * jiggle(deterministic), 384 + 30 * jiggle(deterministic)]
    agent = [255 + 60 * jiggle(deterministic), 255 + 60 * jiggle(deterministic)]

    loc_list = [agent, cube_1, cube_2, cube_3, cube_4]
    rand_rots = np.random.rand(4) * 2 * np.pi - np.pi
    loc_list.append(rand_rots)
    return np.concatenate(loc_list)


def late_decision_cube_setup(deterministic: bool = False) -> np.ndarray:
    cube_1 = [100, 340 + 30 * jiggle(deterministic)]
    cube_2 = [100, 420 + 30 * jiggle(deterministic)]
    cube_3 = [150 + 30 * jiggle(deterministic), 460]
    cube_4 = [230 + 30 * jiggle(deterministic), 460]
    agent = [420 + 30 * jiggle(deterministic), 120 + 30 * jiggle(deterministic)]

    loc_list = [agent, cube_1, cube_2, cube_3, cube_4]
    rand_rots = np.random.rand(4) * 2 * np.pi - np.pi
    loc_list.append(rand_rots)
    return np.concatenate(loc_list)


def _resolve_rollout_dir(rollout):
    if isinstance(rollout, (str, Path)):
        return Path(rollout)
    if isinstance(rollout, dict) and "run_dir" in rollout:
        return Path(rollout["run_dir"])
    raise ValueError("Each rollout must be a path or a dict containing 'run_dir'.")


def _load_rollout_trace(run_dir):
    with np.load(Path(run_dir) / "low_level_obs.npz", allow_pickle=True) as data:
        states = np.asarray(data["states"])
        agent_pos = np.asarray(data["agent_pos"])

    if states.dtype == object:
        states = np.stack([np.asarray(item, dtype=float) for item in states])
    else:
        states = states.astype(float)
    if agent_pos.dtype == object:
        agent_pos = np.stack([np.asarray(item, dtype=float) for item in agent_pos])
    else:
        agent_pos = agent_pos.astype(float)
    return states, agent_pos


def _canonical_layout(setup_fn):
    layout = np.asarray(setup_fn(deterministic=True), dtype=float)
    agent_pos = layout[:2] / 256.0 - 1.0
    block_positions = layout[2:10].reshape(4, 2) / 256.0 - 1.0
    return agent_pos, block_positions


def _collapse_latest_frame(value):
    array = np.asarray(value, dtype=float)
    if array.ndim <= 1:
        return array
    return array[-1]


def _trajectory_segments(points):
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)


def _flip_y(points):
    arr = np.asarray(points, dtype=float).copy()
    arr[..., 1] *= -1.0
    return arr


def _load_setup_layout(run_dir):
    setup_state = np.load(Path(run_dir) / "setup_state.npy")
    blocks = setup_state[2:10].reshape(4, 2) / 256.0 - 1.0
    angles = np.asarray(setup_state[10:14], dtype=float) if setup_state.shape[0] >= 14 else np.zeros(4)
    return blocks, angles


def _draw_blocks(ax, centers, angles, colors, radius=0.07, alpha=1.0, zorder=3):
    for center, angle, color in zip(centers, angles, colors):
        patch = RegularPolygon(
            (center[0], center[1]),
            numVertices=4,
            radius=radius,
            orientation=angle + np.pi / 4.0,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(patch)


def _match_blocks_to_canonical(blocks, canonical_blocks):
    blocks = np.asarray(blocks, dtype=float)
    canonical_blocks = np.asarray(canonical_blocks, dtype=float)
    best_perm = None
    best_cost = np.inf
    for perm in permutations(range(4)):
        candidate = blocks[list(perm)]
        cost = np.sum((candidate - canonical_blocks) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return blocks[list(best_perm)]


def _plot_rollouts(ax, rollouts, setup_fn, title):
    palette = ["#0a84ff", "#ff3b30", "#34c759", "#ffd60a"]
    canonical_agent_raw, canonical_blocks_raw = _canonical_layout(setup_fn)
    canonical_agent = _flip_y(canonical_agent_raw)
    canonical_blocks = _flip_y(canonical_blocks_raw)
    canonical_angles = np.zeros(4)

    for rollout in rollouts:
        run_dir = _resolve_rollout_dir(rollout)
        states, agent_pos = _load_rollout_trace(run_dir)

        setup_blocks_raw, setup_angles_raw = _load_setup_layout(run_dir)

        start_blocks = _match_blocks_to_canonical(setup_blocks_raw, canonical_blocks_raw)
        permuted_angles = np.zeros(4)
        for idx, block in enumerate(start_blocks):
            distances = np.sum((setup_blocks_raw - block) ** 2, axis=1)
            permuted_angles[idx] = setup_angles_raw[int(np.argmin(distances))]
        start_blocks = _flip_y(start_blocks)
        start_angles = -permuted_angles
        start_agent = _flip_y(_collapse_latest_frame(agent_pos[0]))
        trajectory = np.stack([_collapse_latest_frame(step) for step in agent_pos], axis=0)
        trajectory = _flip_y(trajectory)

        if len(trajectory) > 1:
            segments = _trajectory_segments(trajectory)
            line = LineCollection(
                segments,
                cmap="plasma",
                norm=plt.Normalize(0.0, 1.0),
                linewidths=1.4,
                alpha=0.68,
                zorder=2,
            )
            line.set_array(np.linspace(0.0, 1.0, len(segments)))
            ax.add_collection(line)

        _draw_blocks(ax, start_blocks, start_angles, palette, radius=0.11, alpha=0.2, zorder=3)
        ax.scatter(
            [start_agent[0]],
            [start_agent[1]],
            s=150,
            marker="o",
            c="#666666",
            alpha=0.16,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            [trajectory[-1, 0]],
            [trajectory[-1, 1]],
            s=8,
            marker="o",
            c="#000000",
            alpha=0.7,
            edgecolors="none",
            linewidths=0.0,
            zorder=6,
        )

    _draw_blocks(ax, canonical_blocks, canonical_angles, palette, radius=0.11, alpha=0.95, zorder=4)
    ax.scatter(
        [canonical_agent[0]],
        [canonical_agent[1]],
        s=220,
        marker="o",
        c="#2f2f2f",
        alpha=0.92,
        edgecolors="white",
        linewidths=0.4,
        zorder=5,
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    frame = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#9a9a9a", linewidth=2.2, zorder=10)
    ax.add_patch(frame)


def plot_early_late_rollouts_overlay(
    early_rollouts,
    late_rollouts,
    output_dir,
    early_setup_fn=early_decision_cube_setup,
    late_setup_fn=late_decision_cube_setup,
    early_title="Early Decision",
    late_title="Late Decision",
    figsize=(4.5, 3.8),
    dpi=170,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"trajectory_overlay_{timestamp}.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    _plot_rollouts(ax1, early_rollouts, early_setup_fn, early_title)
    _plot_rollouts(ax2, late_rollouts, late_setup_fn, late_title)
    fig.savefig(comparison_path, bbox_inches="tight")

    return fig, comparison_path
