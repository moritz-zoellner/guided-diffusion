"""Toy Squares safety-square rollout with low-level dynamics guidance.

The rollout follows the early-decision setup from ``automaton_guidance.ipynb``:
sample action chunks from the diffusion policy, rank candidates with the
automaton model toward blue, then optionally refine the selected chunk with a
low-level dynamics model. The dynamics objective is deliberately simple:

    maximize blue progress + safety-square signed distance

There is no action regularization term.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse, Polygon, Rectangle

plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.monospace": ["Computer Modern Typewriter", "CMU Typewriter Text", "DejaVu Sans Mono"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 6.5,
        "axes.titlesize": 7,
        "axes.titleweight": "normal",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toy_squares.toy_squares_utils import (  # noqa: E402
    _draw_blocks as draw_toy_blocks,
)
from toy_squares.toy_squares_utils import (  # noqa: E402
    _flip_y as flip_toy_y,
)
from toy_squares.toy_squares_utils import (  # noqa: E402
    early_decision_cube_setup,
    load_automaton_model_for_eval,
)
from toy_squares.train_dynamics_world_model import load_dynamics_model_for_eval  # noqa: E402


DEFAULT_DP_CKPT = Path(
    "/home/moritz/data/diffusion_runs/toy_squares_dp_n500/20260422190540/models/"
    "model_epoch_320_best_validation_0.005075077305082232.pth"
)
DEFAULT_AUTOMATON_RUN = REPO_ROOT / "outputs/automaton_world_model/training-run_2026-04-28_17-52-27"
DEFAULT_DYNAMICS_ROOT = REPO_ROOT / "outputs/toy_squares/dynamics_world_model"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/toy_squares_rollouts/dynamics_safety_guidance"

STATE_BLOCK_NAMES = ["blue", "red", "green", "yellow"]
LABEL_NAMES = ["at_green", "at_blue", "at_red", "at_yellow"]
LABEL_TO_STATE_BLOCK_IDX = [2, 0, 1, 3]
STATE_BLOCK_TO_LABEL_IDX = [1, 2, 0, 3]
BLUE_LABEL_IDX = 1
BLUE_BLOCK_IDX = 0
LABEL_NAME_TO_IDX = {
    "green": 0,
    "at_green": 0,
    "blue": 1,
    "at_blue": 1,
    "red": 2,
    "at_red": 2,
    "yellow": 3,
    "at_yellow": 3,
}

PAPER_BLOCK_COLORS = {
    "blue": "#4f79a7",
    "red": "#b85f5a",
    "green": "#5e9c73",
    "yellow": "#c8a84a",
}
PAPER_XLIM = (-0.76, 0.76)
PAPER_YLIM = (-0.76, 0.76)
PAPER_BLOCK_RADIUS = 0.16
PAPER_FRAME_LW = 0.9
PAPER_BLUE = "#275fca"
PAPER_GRAY = "#5f6368"
PAPER_DARK = "#2f2f2f"
SAFETY_FACE = "#d58a82"
SAFETY_EDGE = "#9b3f3a"
SAFETY_FAINT_FACE = "#c9c9c9"
SAFETY_FAINT_EDGE = "#9d9d9d"


@dataclass(frozen=True)
class SafetyBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    angle_degrees: float = 0.0
    name: str = "safety_square"

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    def expanded(self, margin: float) -> "SafetyBox":
        margin = float(margin)
        cx, cy = self.center
        width, height = self.size
        return SafetyBox(
            x_min=cx - 0.5 * width - margin,
            x_max=cx + 0.5 * width + margin,
            y_min=cy - 0.5 * height - margin,
            y_max=cy + 0.5 * height + margin,
            angle_degrees=self.angle_degrees,
            name=self.name,
        )


@dataclass(frozen=True)
class SafetyEllipse:
    cx: float
    cy: float
    rx: float
    ry: float
    name: str = "safety_ellipse"

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def size(self) -> tuple[float, float]:
        return (2.0 * self.rx, 2.0 * self.ry)

    def expanded(self, margin: float) -> "SafetyEllipse":
        margin = float(margin)
        return SafetyEllipse(
            cx=self.cx,
            cy=self.cy,
            rx=max(self.rx + margin, 1e-6),
            ry=max(self.ry + margin, 1e-6),
            name=self.name,
        )


SafetyRegion = SafetyBox | SafetyEllipse


@dataclass(frozen=True)
class StageSpec:
    mode: str
    label_idxs: tuple[int, ...]
    raw: str

    @property
    def names(self) -> list[str]:
        return [LABEL_NAMES[int(idx)] for idx in self.label_idxs]

    @property
    def display(self) -> str:
        joiner = " OR " if self.mode == "any" else " AND "
        return joiner.join(self.names)


def reseed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def latest_low_level_vector(value) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=np.float32)
    if array.ndim >= 2:
        array = array[-1]
    return array.reshape(-1)


def obs_low_level_snapshot(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: to_numpy(obs[key]).copy() for key in ("agent_pos", "states") if key in obs}


def state_vector_from_low_level_obs(obs: dict[str, Any]) -> np.ndarray:
    state = np.concatenate(
        [
            latest_low_level_vector(obs["agent_pos"]),
            latest_low_level_vector(obs["states"]),
        ]
    ).astype(np.float32)
    if state.shape != (10,):
        raise ValueError(f"Expected 10D low-level state, got {state.shape}")
    return state


def latest_obs_tensor(obs_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = obs_dict[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    value = value.float()
    expected_dim = {"agent_pos": 2, "states": 8}[key]
    if value.shape[-1] != expected_dim:
        raise ValueError(f"Expected obs['{key}'] last dim {expected_dim}, got {tuple(value.shape)}")
    if value.ndim == 3:
        return value[:, -1, :]
    if value.ndim == 2:
        return value if value.shape[0] == 1 else value[-1:, :]
    if value.ndim == 1:
        return value.unsqueeze(0)
    raise ValueError(f"Unexpected obs['{key}'] shape {tuple(value.shape)}")


def automaton_state_from_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([latest_obs_tensor(obs_dict, "agent_pos"), latest_obs_tensor(obs_dict, "states")], dim=-1)


def automaton_label_from_state(state: torch.Tensor, cube_radius: float = 0.2) -> torch.Tensor:
    agent = state[:, 0:2]
    cubes = state[:, 2:10].reshape(state.shape[0], 4, 2)
    distances = torch.linalg.norm(agent[:, None, :] - cubes, dim=-1)
    contact = (distances <= float(cube_radius)).float()
    has_contact = contact.bool().any(dim=-1)
    active_distances = torch.where(contact.bool(), distances, torch.full_like(distances, float("inf")))
    nearest_active = active_distances.argmin(dim=-1)
    contact_onehot = torch.zeros_like(contact)
    contact_onehot.scatter_(1, nearest_active[:, None], has_contact.to(contact.dtype)[:, None])
    return torch.stack(
        [contact_onehot[:, 2], contact_onehot[:, 0], contact_onehot[:, 1], contact_onehot[:, 3]],
        dim=-1,
    )


def label_from_low_level_obs(obs: dict[str, Any], cube_radius: float = 0.2) -> dict[str, Any]:
    state_np = state_vector_from_low_level_obs(obs)
    state = torch.as_tensor(state_np, dtype=torch.float32).unsqueeze(0)
    label = automaton_label_from_state(state, cube_radius=cube_radius)[0].detach().cpu().numpy().astype(int)
    agent = state_np[0:2]
    blocks = state_np[2:10].reshape(4, 2)
    distances = np.linalg.norm(blocks - agent[None, :], axis=-1)
    nearest_block = int(np.argmin(distances))
    return {
        "onehot": label.tolist(),
        "active_labels": [LABEL_NAMES[idx] for idx, value in enumerate(label) if value],
        "nearest_state_block_idx": nearest_block,
        "nearest_state_block": STATE_BLOCK_NAMES[nearest_block],
        "nearest_label_idx": int(STATE_BLOCK_TO_LABEL_IDX[nearest_block]),
        "distances_by_state_block": {name: float(distances[idx]) for idx, name in enumerate(STATE_BLOCK_NAMES)},
    }


def repeat_obs_batch(obs_tensor: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    if int(n) == 1:
        return obs_tensor
    return {key: value.repeat((int(n),) + (1,) * (value.ndim - 1)) for key, value in obs_tensor.items()}


def policy_action_normalization_parts(policy, device: torch.device, dtype: torch.dtype):
    if policy.action_normalization_stats is None:
        return None
    action_keys = policy.policy.global_config.train.action_keys
    offsets = []
    scales = []
    for key in action_keys:
        offset = torch.as_tensor(policy.action_normalization_stats[key]["offset"].reshape(-1), device=device, dtype=dtype)
        scale = torch.as_tensor(policy.action_normalization_stats[key]["scale"].reshape(-1), device=device, dtype=dtype)
        offsets.append(offset)
        scales.append(scale)
    return torch.cat(offsets, dim=0), torch.cat(scales, dim=0)


def unnormalize_action_sequence_torch(action_sequence: torch.Tensor, policy) -> torch.Tensor:
    parts = policy_action_normalization_parts(policy, action_sequence.device, action_sequence.dtype)
    if parts is None:
        return action_sequence
    offset, scale = parts
    return action_sequence * scale.view(1, 1, -1) + offset.view(1, 1, -1)


def point_signed_distance_to_box_np(point: np.ndarray, box: SafetyBox) -> float:
    point = np.asarray(point, dtype=np.float32)
    center = np.asarray(box.center, dtype=np.float32)
    half = np.asarray(box.size, dtype=np.float32) / 2.0
    delta = point - center
    if float(box.angle_degrees) != 0.0:
        theta = math.radians(float(box.angle_degrees))
        c, s = math.cos(theta), math.sin(theta)
        delta = np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]], dtype=np.float32)
    q = np.abs(delta) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(float(q[0]), float(q[1])), 0.0)
    return float(outside + inside)


def point_signed_distance_to_ellipse_np(point: np.ndarray, ellipse: SafetyEllipse) -> float:
    point = np.asarray(point, dtype=np.float32)
    center = np.asarray(ellipse.center, dtype=np.float32)
    radii = np.asarray([ellipse.rx, ellipse.ry], dtype=np.float32)
    scaled_radius = float(np.linalg.norm((point - center) / np.maximum(radii, 1e-6)))
    return float((scaled_radius - 1.0) * float(np.min(radii)))


def point_signed_distance_to_region_np(point: np.ndarray, region: SafetyRegion) -> float:
    if isinstance(region, SafetyBox):
        return point_signed_distance_to_box_np(point, region)
    if isinstance(region, SafetyEllipse):
        return point_signed_distance_to_ellipse_np(point, region)
    raise TypeError(f"Unsupported safety region: {type(region)!r}")


def region_to_dict(region: SafetyRegion) -> dict[str, Any]:
    data = asdict(region)
    data["kind"] = "box" if isinstance(region, SafetyBox) else "ellipse"
    return data


def densify_polyline_np(points_xy: np.ndarray, segment_samples: int) -> tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    segment_samples = max(int(segment_samples), 1)
    if len(points_xy) <= 1:
        return points_xy.copy(), np.zeros(len(points_xy), dtype=np.int64)

    fractions = np.linspace(0.0, 1.0, segment_samples + 1, dtype=np.float32)[1:]
    dense = [points_xy[0]]
    dense_segment_idxs = [0]
    for idx in range(len(points_xy) - 1):
        start = points_xy[idx]
        end = points_xy[idx + 1]
        samples = start[None, :] + fractions[:, None] * (end - start)[None, :]
        dense.extend(samples)
        dense_segment_idxs.extend([idx + 1] * len(samples))
    return np.asarray(dense, dtype=np.float32), np.asarray(dense_segment_idxs, dtype=np.int64)


def trajectory_safety_metrics(agent_xy: np.ndarray, regions: list[SafetyRegion], segment_samples: int) -> dict[str, Any]:
    dense_xy, dense_step_idxs = densify_polyline_np(agent_xy, segment_samples=segment_samples)
    per_region_signed = []
    region_metrics = []
    for region in regions:
        signed_i = np.asarray([point_signed_distance_to_region_np(point, region) for point in dense_xy], dtype=np.float32)
        per_region_signed.append(signed_i)
        region_violations = np.where(signed_i < 0.0)[0]
        region_metrics.append(
            {
                "name": region.name,
                "kind": "box" if isinstance(region, SafetyBox) else "ellipse",
                "violated": bool(len(region_violations) > 0),
                "first_violation_step": None
                if len(region_violations) == 0
                else int(dense_step_idxs[int(region_violations[0])]),
                "first_violation_sample": None if len(region_violations) == 0 else int(region_violations[0]),
                "min_signed_distance": float(np.min(signed_i)) if len(signed_i) else float("nan"),
            }
        )
    if per_region_signed:
        signed_by_region = np.stack(per_region_signed, axis=0)
        signed = np.min(signed_by_region, axis=0)
        closest_region = np.argmin(signed_by_region, axis=0)
    else:
        signed = np.full(len(dense_xy), float("inf"), dtype=np.float32)
        closest_region = np.full(len(dense_xy), -1, dtype=np.int64)
    first_violation = np.where(signed < 0.0)[0]
    first_violation_idx = None if len(first_violation) == 0 else int(first_violation[0])
    return {
        "violated": bool(len(first_violation) > 0),
        "first_violation_step": None if first_violation_idx is None else int(dense_step_idxs[first_violation_idx]),
        "first_violation_sample": first_violation_idx,
        "first_violation_region": None
        if first_violation_idx is None
        else regions[int(closest_region[first_violation_idx])].name,
        "min_signed_distance": float(np.min(signed)) if len(signed) else float("nan"),
        "signed_distances": signed.tolist(),
        "regions": region_metrics,
        "segment_samples": int(max(segment_samples, 1)),
    }


class DynamicsSafetyRefiner:
    def __init__(self, model, stats: dict[str, np.ndarray], policy, device: str):
        self.model = model
        self.stats_t = {
            key: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0)
            for key, value in stats.items()
        }
        self.policy = policy
        self.device = device

    def rollout_torch(self, state0: torch.Tensor, action_chunk_raw: torch.Tensor) -> torch.Tensor:
        if action_chunk_raw.ndim == 2:
            action_chunk_raw = action_chunk_raw.unsqueeze(0)
        state = state0.expand(action_chunk_raw.shape[0], -1)
        states = []
        for t in range(action_chunk_raw.shape[1]):
            state_n = (state - self.stats_t["state_mean"].to(state.dtype)) / self.stats_t["state_std"].to(state.dtype)
            action_n = (
                action_chunk_raw[:, t, :] - self.stats_t["action_mean"].to(action_chunk_raw.dtype)
            ) / self.stats_t["action_std"].to(action_chunk_raw.dtype)
            delta_n = self.model(state_n, action_n)
            delta = delta_n * self.stats_t["delta_std"].to(delta_n.dtype) + self.stats_t["delta_mean"].to(delta_n.dtype)
            state = state + delta
            state = torch.clamp(state, -1.05, 1.05)
            states.append(state)
        return torch.stack(states, dim=1)

    @staticmethod
    def signed_distance_to_box_torch(points_xy: torch.Tensor, box: SafetyBox) -> torch.Tensor:
        center = torch.tensor(box.center, device=points_xy.device, dtype=points_xy.dtype)
        half = torch.tensor(box.size, device=points_xy.device, dtype=points_xy.dtype) / 2.0
        delta = points_xy - center
        if float(box.angle_degrees) != 0.0:
            theta = math.radians(float(box.angle_degrees))
            c = torch.as_tensor(math.cos(theta), device=points_xy.device, dtype=points_xy.dtype)
            s = torch.as_tensor(math.sin(theta), device=points_xy.device, dtype=points_xy.dtype)
            delta = torch.stack(
                [c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]],
                dim=-1,
            )
        q = torch.abs(delta) - half
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.minimum(torch.maximum(q[..., 0], q[..., 1]), torch.zeros((), device=points_xy.device, dtype=points_xy.dtype))
        return outside + inside

    @staticmethod
    def signed_distance_to_ellipse_torch(points_xy: torch.Tensor, ellipse: SafetyEllipse) -> torch.Tensor:
        center = torch.tensor(ellipse.center, device=points_xy.device, dtype=points_xy.dtype)
        radii = torch.tensor([ellipse.rx, ellipse.ry], device=points_xy.device, dtype=points_xy.dtype).clamp_min(1e-6)
        scaled_radius = torch.linalg.norm((points_xy - center) / radii, dim=-1)
        return (scaled_radius - 1.0) * torch.min(radii)

    @staticmethod
    def signed_distance_to_region_torch(points_xy: torch.Tensor, region: SafetyRegion) -> torch.Tensor:
        if isinstance(region, SafetyBox):
            return DynamicsSafetyRefiner.signed_distance_to_box_torch(points_xy, region)
        if isinstance(region, SafetyEllipse):
            return DynamicsSafetyRefiner.signed_distance_to_ellipse_torch(points_xy, region)
        raise TypeError(f"Unsupported safety region: {type(region)!r}")

    @staticmethod
    def densify_polyline_torch(points_xy: torch.Tensor, segment_samples: int) -> torch.Tensor:
        segment_samples = max(int(segment_samples), 1)
        if points_xy.shape[1] <= 1:
            return points_xy
        start = points_xy[:, :-1, :]
        end = points_xy[:, 1:, :]
        fractions = torch.linspace(
            0.0,
            1.0,
            segment_samples + 1,
            device=points_xy.device,
            dtype=points_xy.dtype,
        )[1:]
        dense = start[:, :, None, :] + fractions[None, None, :, None] * (end - start)[:, :, None, :]
        return torch.cat([points_xy[:, :1, :], dense.reshape(points_xy.shape[0], -1, points_xy.shape[-1])], dim=1)

    def objective_terms(
        self,
        state0: torch.Tensor,
        action_chunk_n: torch.Tensor,
        safety_regions: list[SafetyRegion],
        target_xy: torch.Tensor | None,
        safety_margin: float,
        safety_clearance_cap: float,
        segment_samples: int,
        max_step_length: float,
        smooth_min_tau: float,
        goal_tau: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        action_raw = unnormalize_action_sequence_torch(action_chunk_n, self.policy)
        pred_states = self.rollout_torch(state0, action_raw)
        pred_xy = pred_states[..., 0:2]
        polyline_xy = torch.cat([state0[:, None, 0:2], pred_xy], dim=1)
        dense_xy = self.densify_polyline_torch(polyline_xy, segment_samples=segment_samples)
        segment_lengths = torch.linalg.norm(polyline_xy[:, 1:, :] - polyline_xy[:, :-1, :], dim=-1)
        excess_length = torch.relu(segment_lengths - float(max_step_length))
        length_penalty = torch.mean(excess_length ** 2, dim=-1)
        mean_length = torch.mean(segment_lengths, dim=-1, keepdim=True)
        length_variance = torch.mean((segment_lengths - mean_length) ** 2, dim=-1)

        if target_xy is None:
            target_xy = state0[:, 2 + BLUE_BLOCK_IDX * 2 : 2 + BLUE_BLOCK_IDX * 2 + 2]
        target_xy = target_xy.to(device=pred_xy.device, dtype=pred_xy.dtype).reshape(pred_xy.shape[0], 1, 2)
        dist_to_target = torch.linalg.norm(pred_xy - target_xy, dim=-1)
        goal_margin = 0.2 - dist_to_target
        goal_score = float(goal_tau) * torch.logsumexp(goal_margin / max(float(goal_tau), 1e-6), dim=-1)

        signed_parts = [
            self.signed_distance_to_region_torch(dense_xy, region.expanded(float(safety_margin)))
            for region in safety_regions
        ]
        if signed_parts:
            signed = torch.stack(signed_parts, dim=1).reshape(dense_xy.shape[0], -1)
        else:
            signed = torch.full(
                (dense_xy.shape[0], dense_xy.shape[1]),
                float("inf"),
                device=dense_xy.device,
                dtype=dense_xy.dtype,
            )
        capped = torch.clamp(signed, max=max(float(safety_clearance_cap), 1e-6))
        tau = max(float(smooth_min_tau), 1e-6)
        safety_score = -tau * torch.logsumexp(-capped / tau, dim=-1)
        return pred_states, {
            "goal_score": goal_score,
            "safety_score": safety_score,
            "signed": signed,
            "dist_to_target": dist_to_target,
            "segment_lengths": segment_lengths,
            "length_penalty": length_penalty,
            "length_variance": length_variance,
        }

    def refine(
        self,
        obs_tensor: dict[str, torch.Tensor],
        base_chunk_n: torch.Tensor,
        safety_regions: list[SafetyRegion],
        target_xy: np.ndarray | None,
        guidance_steps: int,
        adam_lr: float,
        safety_scale: float,
        goal_scale: float,
        safety_margin: float,
        safety_clearance_cap: float,
        segment_samples: int,
        max_step_length: float,
        length_scale: float,
        length_variance_scale: float,
        smooth_min_tau: float,
        goal_tau: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state0 = automaton_state_from_obs(obs_tensor).to(device=self.device, dtype=torch.float32)
        target_xy_t = None
        if target_xy is not None:
            target_xy_t = torch.as_tensor(target_xy, device=self.device, dtype=torch.float32).reshape(1, 2)
        original = base_chunk_n.detach().clone().to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            pred_before, terms_before = self.objective_terms(
                state0,
                original,
                safety_regions,
                target_xy_t,
                safety_margin=safety_margin,
                safety_clearance_cap=safety_clearance_cap,
                segment_samples=segment_samples,
                max_step_length=max_step_length,
                smooth_min_tau=smooth_min_tau,
                goal_tau=goal_tau,
            )

        if int(guidance_steps) <= 0 or float(safety_scale) == 0.0 and float(goal_scale) == 0.0:
            record = self._record(
                enabled=False,
                before=terms_before,
                after=terms_before,
                pred_before=pred_before,
                pred_after=pred_before,
                actions_before=original,
                actions_after=original,
                history=[],
            )
            return original, record

        actions = original.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=float(adam_lr))
        history = []
        for _ in range(int(guidance_steps)):
            opt.zero_grad(set_to_none=True)
            _, terms = self.objective_terms(
                state0,
                actions,
                safety_regions,
                target_xy_t,
                safety_margin=safety_margin,
                safety_clearance_cap=safety_clearance_cap,
                segment_samples=segment_samples,
                max_step_length=max_step_length,
                smooth_min_tau=smooth_min_tau,
                goal_tau=goal_tau,
            )
            objective = (
                float(safety_scale) * terms["safety_score"]
                + float(goal_scale) * terms["goal_score"]
                - float(length_scale) * terms["length_penalty"]
                - float(length_variance_scale) * terms["length_variance"]
            )
            (-objective.mean()).backward()
            opt.step()
            with torch.no_grad():
                actions.clamp_(-1.0, 1.0)
                history.append(
                    {
                        "objective": float(objective[0].detach().cpu()),
                        "safety_score": float(terms["safety_score"][0].detach().cpu()),
                        "goal_score": float(terms["goal_score"][0].detach().cpu()),
                        "min_signed_distance": float(terms["signed"][0].min().detach().cpu()),
                        "min_dist_to_target": float(terms["dist_to_target"][0].min().detach().cpu()),
                        "max_segment_length": float(terms["segment_lengths"][0].max().detach().cpu()),
                        "length_penalty": float(terms["length_penalty"][0].detach().cpu()),
                        "length_variance": float(terms["length_variance"][0].detach().cpu()),
                    }
                )

        with torch.no_grad():
            pred_after, terms_after = self.objective_terms(
                state0,
                actions,
                safety_regions,
                target_xy_t,
                safety_margin=safety_margin,
                safety_clearance_cap=safety_clearance_cap,
                segment_samples=segment_samples,
                max_step_length=max_step_length,
                smooth_min_tau=smooth_min_tau,
                goal_tau=goal_tau,
            )
        record = self._record(
            enabled=True,
            before=terms_before,
            after=terms_after,
            pred_before=pred_before,
            pred_after=pred_after,
            actions_before=original,
            actions_after=actions,
            history=history,
        )
        return actions.detach(), record

    @staticmethod
    def _record(
        enabled: bool,
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor],
        pred_before: torch.Tensor,
        pred_after: torch.Tensor,
        actions_before: torch.Tensor,
        actions_after: torch.Tensor,
        history: list[dict[str, float]],
    ) -> dict[str, Any]:
        delta = actions_after - actions_before
        return {
            "enabled": bool(enabled),
            "safety_score_before": float(before["safety_score"][0].detach().cpu()),
            "safety_score_after": float(after["safety_score"][0].detach().cpu()),
            "goal_score_before": float(before["goal_score"][0].detach().cpu()),
            "goal_score_after": float(after["goal_score"][0].detach().cpu()),
            "min_signed_before": float(before["signed"][0].min().detach().cpu()),
            "min_signed_after": float(after["signed"][0].min().detach().cpu()),
            "min_dist_to_target_before": float(before["dist_to_target"][0].min().detach().cpu()),
            "min_dist_to_target_after": float(after["dist_to_target"][0].min().detach().cpu()),
            "max_segment_length_before": float(before["segment_lengths"][0].max().detach().cpu()),
            "max_segment_length_after": float(after["segment_lengths"][0].max().detach().cpu()),
            "length_penalty_before": float(before["length_penalty"][0].detach().cpu()),
            "length_penalty_after": float(after["length_penalty"][0].detach().cpu()),
            "length_variance_before": float(before["length_variance"][0].detach().cpu()),
            "length_variance_after": float(after["length_variance"][0].detach().cpu()),
            "pred_xy_before": pred_before[0, :, 0:2].detach().cpu().numpy().tolist(),
            "pred_xy_after": pred_after[0, :, 0:2].detach().cpu().numpy().tolist(),
            "action_l2_change": float(torch.linalg.norm(delta).detach().cpu()),
            "max_abs_action_change": float(torch.max(torch.abs(delta)).detach().cpu()),
            "history": history,
        }


def score_action_chunks(
    automaton_model,
    automaton_stats: dict[str, np.ndarray],
    obs_tensor_rank: dict[str, torch.Tensor],
    action_chunks_raw: torch.Tensor,
    device: str,
) -> np.ndarray:
    state = automaton_state_from_obs(obs_tensor_rank).to(device=device, dtype=torch.float32)
    label = automaton_label_from_state(state).to(device=device, dtype=torch.float32)
    action_flat = action_chunks_raw.reshape(action_chunks_raw.shape[0], -1).to(device=device, dtype=torch.float32)
    state_mean = torch.as_tensor(automaton_stats["states_mean"], device=device, dtype=torch.float32).unsqueeze(0)
    state_std = torch.as_tensor(automaton_stats["states_std"], device=device, dtype=torch.float32).unsqueeze(0)
    action_mean = torch.as_tensor(automaton_stats["actions_mean"], device=device, dtype=torch.float32).unsqueeze(0)
    action_std = torch.as_tensor(automaton_stats["actions_std"], device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = automaton_model((state - state_mean) / state_std, (action_flat - action_mean) / action_std, label)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def score_stage_candidates(
    candidate_probs: np.ndarray,
    stage: StageSpec,
    achieved_idxs: set[int],
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    if stage.mode == "all":
        active_target_idxs = [idx for idx in stage.label_idxs if int(idx) not in achieved_idxs]
        if not active_target_idxs:
            active_target_idxs = [int(stage.label_idxs[-1])]
    else:
        active_target_idxs = [int(idx) for idx in stage.label_idxs]

    target_columns = np.stack([candidate_probs[:, int(idx)] for idx in active_target_idxs], axis=1)
    if stage.mode == "all":
        candidate_target_pos = target_columns.argmax(axis=1)
        candidate_scores = target_columns.max(axis=1)
    elif stage.mode == "any":
        candidate_target_pos = target_columns.argmax(axis=1)
        candidate_scores = target_columns.max(axis=1)
    else:
        raise ValueError(f"Unsupported stage mode {stage.mode!r}")

    selected_target_idxs = np.asarray([active_target_idxs[int(pos)] for pos in candidate_target_pos], dtype=np.int64)
    return candidate_scores, selected_target_idxs, active_target_idxs, target_columns


def target_xy_from_state(state: np.ndarray, target_label_idx: int) -> np.ndarray:
    block_idx = LABEL_TO_STATE_BLOCK_IDX[int(target_label_idx)]
    start = 2 + block_idx * 2
    return np.asarray(state[start : start + 2], dtype=np.float32)


def stage_to_dict(stage: StageSpec) -> dict[str, Any]:
    return {
        "mode": stage.mode,
        "label_idxs": [int(idx) for idx in stage.label_idxs],
        "label_names": stage.names,
        "raw": stage.raw,
        "display": stage.display,
    }


def extract_agent_xy(low_level_obs: list[dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack([latest_low_level_vector(obs["agent_pos"]) for obs in low_level_obs], axis=0)


def run_rollout(
    name: str,
    policy,
    env,
    automaton_model,
    automaton_stats,
    refiner: DynamicsSafetyRefiner,
    setup_state: np.ndarray,
    rollout_seed: int,
    safety_regions: list[SafetyRegion],
    horizon: int,
    n_candidates: int,
    stage_specs: list[StageSpec],
    use_dynamics_guidance: bool,
    guidance_steps: int,
    adam_lr: float,
    safety_scale: float,
    goal_scale: float,
    safety_margin: float,
    safety_clearance_cap: float,
    segment_samples: int,
    max_step_length: float,
    length_scale: float,
    length_variance_scale: float,
    route_waypoint: np.ndarray | None,
    stage_route_waypoints: dict[int, np.ndarray],
    route_waypoint_radius: float,
    smooth_min_tau: float,
    goal_tau: float,
    capture_env_frames: bool,
    device: str,
) -> dict[str, Any]:
    reseed(int(rollout_seed))
    obs = env.reset_to(setup_state)
    policy.start_episode()
    old_debug = getattr(policy.policy, "debug_guidance_actions", False)
    policy.policy.debug_guidance_actions = False

    records = []
    action_queue = []
    low_level_obs = [obs_low_level_snapshot(obs)]
    env_frames = []
    if capture_env_frames:
        env_frames.append(render_env_frame(env))

    total_reward = 0.0
    steps = 0
    target_reached = False
    reaches = []
    stage_completions = []
    stage_pos = 0
    stage_achieved: set[int] = set()
    route_waypoint_reached = False
    stage_route_waypoints_reached = {int(idx): False for idx in stage_route_waypoints}

    def active_route_waypoint(current_stage: int, current_state_np: np.ndarray) -> tuple[np.ndarray | None, bool]:
        if int(current_stage) == 0 and route_waypoint is not None and not route_waypoint_reached:
            return np.asarray(route_waypoint, dtype=np.float32), False
        if int(current_stage) in stage_route_waypoints and not stage_route_waypoints_reached[int(current_stage)]:
            waypoint = np.asarray(stage_route_waypoints[int(current_stage)], dtype=np.float32)
            reached = bool(np.linalg.norm(current_state_np[0:2] - waypoint) <= float(route_waypoint_radius))
            if reached:
                stage_route_waypoints_reached[int(current_stage)] = True
                return None, True
            return waypoint, False
        return None, False

    def sync_stage_progress(current_label: dict[str, Any], step: int) -> bool:
        nonlocal stage_pos, stage_achieved, target_reached
        active = {idx for idx, value in enumerate(current_label["onehot"]) if int(value) == 1}
        advanced = False
        while stage_pos < len(stage_specs):
            stage = stage_specs[stage_pos]
            matched = [int(idx) for idx in stage.label_idxs if int(idx) in active]
            if stage.mode == "any":
                if not matched:
                    break
                label_idx = matched[0]
                event = {
                    "t": int(step),
                    "stage": int(stage_pos),
                    "stage_mode": stage.mode,
                    "stage_display": stage.display,
                    "label_idx": int(label_idx),
                    "label_name": LABEL_NAMES[int(label_idx)],
                }
                reaches.append(event)
                stage_completions.append(
                    {
                        "t": int(step),
                        "stage": int(stage_pos),
                        "stage_mode": stage.mode,
                        "stage_display": stage.display,
                        "satisfied_label_idxs": [int(label_idx)],
                        "satisfied_label_names": [LABEL_NAMES[int(label_idx)]],
                    }
                )
                stage_pos += 1
                stage_achieved = set()
                advanced = True
                continue

            if stage.mode == "all":
                for label_idx in matched:
                    if label_idx not in stage_achieved:
                        stage_achieved.add(label_idx)
                        reaches.append(
                            {
                                "t": int(step),
                                "stage": int(stage_pos),
                                "stage_mode": stage.mode,
                                "stage_display": stage.display,
                                "label_idx": int(label_idx),
                                "label_name": LABEL_NAMES[int(label_idx)],
                                "achieved_label_idxs": [int(idx) for idx in sorted(stage_achieved)],
                                "achieved_label_names": [LABEL_NAMES[int(idx)] for idx in sorted(stage_achieved)],
                            }
                        )
                if len(stage_achieved) >= len(stage.label_idxs):
                    achieved_sorted = [int(idx) for idx in sorted(stage_achieved)]
                    stage_completions.append(
                        {
                            "t": int(step),
                            "stage": int(stage_pos),
                            "stage_mode": stage.mode,
                            "stage_display": stage.display,
                            "satisfied_label_idxs": achieved_sorted,
                            "satisfied_label_names": [LABEL_NAMES[int(idx)] for idx in achieved_sorted],
                        }
                    )
                    stage_pos += 1
                    stage_achieved = set()
                    advanced = True
                    continue
                break

            raise ValueError(f"Unsupported stage mode {stage.mode!r}")

        if stage_pos >= len(stage_specs):
            target_reached = True
        return advanced

    try:
        for t in range(int(horizon)):
            current_label = label_from_low_level_obs(obs_low_level_snapshot(obs))
            if sync_stage_progress(current_label, steps):
                action_queue.clear()
            if stage_pos >= len(stage_specs):
                target_reached = True
                break
            if not action_queue:
                stage = stage_specs[stage_pos]
                current_state_np = state_vector_from_low_level_obs(obs_low_level_snapshot(obs))
                if stage_pos == 0 and route_waypoint is not None and not route_waypoint_reached:
                    route_waypoint_reached = bool(
                        np.linalg.norm(current_state_np[0:2] - np.asarray(route_waypoint, dtype=np.float32)) <= float(route_waypoint_radius)
                    )
                waypoint_target, waypoint_just_reached = active_route_waypoint(stage_pos, current_state_np)
                if waypoint_just_reached:
                    action_queue.clear()
                obs_tensor = policy._prepare_observation(obs)
                obs_tensor_rank = repeat_obs_batch(obs_tensor, int(n_candidates))
                with torch.no_grad():
                    candidate_chunks_n = policy.policy._get_action_trajectory(obs_dict=obs_tensor_rank).detach()
                candidate_chunks_raw = unnormalize_action_sequence_torch(candidate_chunks_n, policy).detach()
                candidate_probs = score_action_chunks(
                    automaton_model,
                    automaton_stats,
                    obs_tensor_rank,
                    candidate_chunks_raw,
                    device=device,
                )
                candidate_scores, candidate_target_idxs, active_target_idxs, target_columns = score_stage_candidates(
                    candidate_probs,
                    stage,
                    achieved_idxs=stage_achieved,
                )
                selected_idx = int(np.argmax(candidate_scores))
                target_label_idx = int(candidate_target_idxs[selected_idx])
                if waypoint_target is not None:
                    refinement_target_xy = waypoint_target
                    refinement_target_kind = f"stage_{stage_pos}_route_waypoint"
                else:
                    refinement_target_xy = target_xy_from_state(current_state_np, target_label_idx)
                    refinement_target_kind = LABEL_NAMES[target_label_idx]
                selected_chunk_n = candidate_chunks_n[selected_idx : selected_idx + 1].detach()
                refinement = None
                if use_dynamics_guidance:
                    selected_chunk_n, refinement = refiner.refine(
                        obs_tensor,
                        selected_chunk_n,
                        safety_regions=safety_regions,
                        target_xy=refinement_target_xy,
                        guidance_steps=guidance_steps,
                        adam_lr=adam_lr,
                        safety_scale=safety_scale,
                        goal_scale=goal_scale,
                        safety_margin=safety_margin,
                        safety_clearance_cap=safety_clearance_cap,
                        segment_samples=segment_samples,
                        max_step_length=max_step_length,
                        length_scale=length_scale,
                        length_variance_scale=length_variance_scale,
                        smooth_min_tau=smooth_min_tau,
                        goal_tau=goal_tau,
                    )
                selected_chunk_raw = unnormalize_action_sequence_torch(selected_chunk_n, policy).detach().cpu().numpy()[0]
                records.append(
                    {
                        "t": int(t),
                        "stage": int(stage_pos),
                        "stage_mode": stage.mode,
                        "stage_display": stage.display,
                        "stage_label_idxs": [int(idx) for idx in stage.label_idxs],
                        "stage_label_names": stage.names,
                        "stage_achieved_idxs": [int(idx) for idx in sorted(stage_achieved)],
                        "stage_achieved_names": [LABEL_NAMES[int(idx)] for idx in sorted(stage_achieved)],
                        "active_target_idxs": [int(idx) for idx in active_target_idxs],
                        "active_target_names": [LABEL_NAMES[int(idx)] for idx in active_target_idxs],
                        "target_label_idx": int(target_label_idx),
                        "target_label": LABEL_NAMES[int(target_label_idx)],
                        "current_label": current_label["onehot"],
                        "selected_idx": selected_idx,
                        "selected_score": float(candidate_scores[selected_idx]),
                        "selected_target_scores": {
                            LABEL_NAMES[int(idx)]: float(target_columns[selected_idx, pos])
                            for pos, idx in enumerate(active_target_idxs)
                        },
                        "pred_probs": candidate_probs[selected_idx].astype(float).tolist(),
                        "candidate_scores": candidate_scores.astype(float).tolist(),
                        "refinement_target_kind": refinement_target_kind,
                        "refinement_target_xy": np.asarray(refinement_target_xy, dtype=float).tolist(),
                        "dynamics_refinement": refinement,
                    }
                )
                action_queue.extend(selected_chunk_raw.astype(np.float32))

            action = np.asarray(action_queue.pop(0), dtype=np.float32)
            obs, reward, done, info = env.step(action)
            low_level_obs.append(obs_low_level_snapshot(obs))
            if capture_env_frames:
                env_frames.append(render_env_frame(env))
            total_reward += float(reward)
            steps = t + 1

            current_label = label_from_low_level_obs(obs_low_level_snapshot(obs))
            if stage_pos == 0 and route_waypoint is not None and not route_waypoint_reached:
                current_state_np = state_vector_from_low_level_obs(obs_low_level_snapshot(obs))
                route_waypoint_reached = bool(
                    np.linalg.norm(current_state_np[0:2] - np.asarray(route_waypoint, dtype=np.float32)) <= float(route_waypoint_radius)
                )
                if route_waypoint_reached:
                    action_queue.clear()
            if stage_pos in stage_route_waypoints and not stage_route_waypoints_reached[int(stage_pos)]:
                current_state_np = state_vector_from_low_level_obs(obs_low_level_snapshot(obs))
                waypoint = np.asarray(stage_route_waypoints[int(stage_pos)], dtype=np.float32)
                stage_route_waypoints_reached[int(stage_pos)] = bool(
                    np.linalg.norm(current_state_np[0:2] - waypoint) <= float(route_waypoint_radius)
                )
                if stage_route_waypoints_reached[int(stage_pos)]:
                    action_queue.clear()
            if sync_stage_progress(current_label, steps):
                action_queue.clear()
                if stage_pos >= len(stage_specs):
                    target_reached = True
                    break
            if done and len(stage_specs) <= 1:
                break
    finally:
        policy.policy.debug_guidance_actions = old_debug

    agent_xy = extract_agent_xy(low_level_obs)
    safety = trajectory_safety_metrics(agent_xy, safety_regions, segment_samples=segment_samples)
    final_label = label_from_low_level_obs(low_level_obs[-1])
    return {
        "name": name,
        "use_dynamics_guidance": bool(use_dynamics_guidance),
        "steps": int(steps),
        "return": float(total_reward),
        "target_reached": bool(target_reached),
        "chain_pos": int(stage_pos),
        "stage_pos": int(stage_pos),
        "stage_specs": [stage_to_dict(stage) for stage in stage_specs],
        "label_chain": [[int(idx) for idx in stage.label_idxs] for stage in stage_specs],
        "label_chain_names": [stage.display for stage in stage_specs],
        "reaches": reaches,
        "stage_completions": stage_completions,
        "final_label": final_label,
        "safety": safety,
        "records": records,
        "route_waypoint": None if route_waypoint is None else np.asarray(route_waypoint, dtype=float).tolist(),
        "route_waypoint_reached": bool(route_waypoint_reached),
        "stage_route_waypoints": {
            str(int(idx)): np.asarray(waypoint, dtype=float).tolist()
            for idx, waypoint in sorted(stage_route_waypoints.items())
        },
        "stage_route_waypoints_reached": {str(int(idx)): bool(value) for idx, value in sorted(stage_route_waypoints_reached.items())},
        "low_level_obs": low_level_obs,
        "agent_xy": agent_xy,
        "env_frames": env_frames,
    }


def render_env_frame(env, height: int = 512, width: int = 512, camera_name: str = "agentview") -> np.ndarray:
    try:
        return env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
    except TypeError:
        return env.render(mode="rgb_array", height=height, width=width)


def setup_blocks_from_state(setup_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(setup_state[2:10], dtype=np.float32).reshape(4, 2) / 256.0 - 1.0
    angles = np.asarray(setup_state[10:14], dtype=np.float32) if len(setup_state) >= 14 else np.zeros(4, dtype=np.float32)
    return blocks, angles


def draw_topdown_base(
    ax,
    setup_state: np.ndarray,
    safety_regions: list[SafetyRegion],
    title: str | None,
    show_title: bool = True,
    show_frame: bool = True,
    safety_style: str = "normal",
) -> None:
    palette = [PAPER_BLOCK_COLORS[name] for name in STATE_BLOCK_NAMES]
    blocks, angles = setup_blocks_from_state(setup_state)
    draw_toy_blocks(ax, flip_toy_y(blocks), -angles, palette, radius=PAPER_BLOCK_RADIUS, alpha=0.96, zorder=4)
    if safety_style == "faint":
        face_alpha, edge_alpha = 0.12, 0.34
        facecolor, edgecolor = SAFETY_FAINT_FACE, SAFETY_FAINT_EDGE
    else:
        face_alpha, edge_alpha = 0.32, 0.95
        facecolor, edgecolor = SAFETY_FACE, SAFETY_EDGE
    for region in safety_regions:
        draw_safety_region(
            ax,
            region,
            face_alpha=face_alpha,
            edge_alpha=edge_alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
    start_agent = flip_toy_y(setup_state[:2] / 256.0 - 1.0)
    ax.scatter(
        [start_agent[0]],
        [start_agent[1]],
        s=30,
        marker="o",
        c=PAPER_DARK,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.35,
        zorder=6,
    )
    if show_title and title:
        ax.set_title(title, fontsize=7, fontweight="normal", pad=3.0)
    ax.set_xlim(*PAPER_XLIM)
    ax.set_ylim(*PAPER_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    if show_frame:
        frame = plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#000000",
            linewidth=PAPER_FRAME_LW,
            zorder=10,
        )
        ax.add_patch(frame)


def draw_safety_region(
    ax,
    region: SafetyRegion,
    face_alpha: float = 0.18,
    edge_alpha: float = 0.95,
    facecolor: str = SAFETY_FACE,
    edgecolor: str = SAFETY_EDGE,
) -> None:
    if isinstance(region, SafetyBox):
        draw_safety_box(ax, region, face_alpha=face_alpha, edge_alpha=edge_alpha, facecolor=facecolor, edgecolor=edgecolor)
    elif isinstance(region, SafetyEllipse):
        draw_safety_ellipse(ax, region, face_alpha=face_alpha, edge_alpha=edge_alpha, facecolor=facecolor, edgecolor=edgecolor)
    else:
        raise TypeError(f"Unsupported safety region: {type(region)!r}")


def safety_box_display_vertices(box: SafetyBox) -> np.ndarray:
    cx, cy = box.center
    width, height = box.size
    half_w, half_h = 0.5 * width, 0.5 * height
    local = np.asarray(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
        dtype=np.float32,
    )
    theta = math.radians(float(box.angle_degrees))
    c, s = math.cos(theta), math.sin(theta)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    physical = local @ rot.T + np.asarray([cx, cy], dtype=np.float32)
    return flip_toy_y(physical)


def draw_safety_box(
    ax,
    box: SafetyBox,
    face_alpha: float = 0.18,
    edge_alpha: float = 0.95,
    facecolor: str = SAFETY_FACE,
    edgecolor: str = SAFETY_EDGE,
) -> None:
    vertices = safety_box_display_vertices(box)
    fill = Polygon(
        vertices,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        alpha=face_alpha,
        zorder=1,
    )
    ax.add_patch(fill)
    outline = Polygon(
        vertices,
        closed=True,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=0.95,
        alpha=edge_alpha,
        zorder=7,
    )
    ax.add_patch(outline)


def draw_safety_ellipse(
    ax,
    ellipse: SafetyEllipse,
    face_alpha: float = 0.18,
    edge_alpha: float = 0.95,
    facecolor: str = SAFETY_FACE,
    edgecolor: str = SAFETY_EDGE,
) -> None:
    center = (ellipse.cx, -ellipse.cy)
    fill = Ellipse(
        center,
        width=2.0 * ellipse.rx,
        height=2.0 * ellipse.ry,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        alpha=face_alpha,
        zorder=1,
    )
    ax.add_patch(fill)
    outline = Ellipse(
        center,
        width=2.0 * ellipse.rx,
        height=2.0 * ellipse.ry,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=0.95,
        alpha=edge_alpha,
        zorder=7,
    )
    ax.add_patch(outline)


def add_faded_path(
    ax,
    xy: np.ndarray,
    color: str,
    linewidth: float = 0.8,
    alpha_start: float = 0.22,
    alpha_end: float = 0.95,
    zorder: int = 3,
) -> None:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        if len(xy) == 1:
            ax.scatter([xy[0, 0]], [xy[0, 1]], s=8, c=color, zorder=zorder)
        return
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    rgba = np.asarray(to_rgba(color), dtype=float)
    colors = np.tile(rgba, (len(segments), 1))
    colors[:, 3] = np.linspace(float(alpha_start), float(alpha_end), len(segments))
    collection = LineCollection(segments, colors=colors, linewidths=linewidth, zorder=zorder)
    try:
        collection.set_capstyle("round")
        collection.set_joinstyle("round")
    except AttributeError:
        pass
    ax.add_collection(collection)


def save_topdown_comparison(
    baseline: dict[str, Any],
    guided: dict[str, Any],
    setup_state: np.ndarray,
    safety_regions: list[SafetyRegion],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.2), dpi=300, constrained_layout=True)
    rows = [
        (baseline, axes[0], "High Level Guidance", PAPER_GRAY, "faint"),
        (guided, axes[1], "Hierarchical Guidance", PAPER_BLUE, "normal"),
    ]
    for rollout, ax, title, color, safety_style in rows:
        draw_topdown_base(ax, setup_state, safety_regions, title, safety_style=safety_style)
        traj = flip_toy_y(rollout["agent_xy"])
        add_faded_path(ax, traj, color=color, linewidth=0.85, alpha_start=0.22, alpha_end=0.95, zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=9, marker="o", c=color, edgecolors="white", linewidths=0.35, zorder=8)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def animation_frame(
    baseline: dict[str, Any],
    guided: dict[str, Any],
    setup_state: np.ndarray,
    safety_regions: list[SafetyRegion],
    frame_idx: int,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.2), dpi=180, constrained_layout=True)
    rows = [
        (baseline, axes[0], "High Level Guidance", PAPER_GRAY, "faint"),
        (guided, axes[1], "Hierarchical Guidance", PAPER_BLUE, "normal"),
    ]
    for rollout, ax, title, color, safety_style in rows:
        draw_topdown_base(ax, setup_state, safety_regions, title, safety_style=safety_style)
        idx = min(int(frame_idx), len(rollout["agent_xy"]) - 1)
        traj = flip_toy_y(rollout["agent_xy"][: idx + 1])
        add_faded_path(ax, traj, color=color, linewidth=0.9, alpha_start=0.28, alpha_end=0.95, zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=11, marker="o", c=color, edgecolors="white", linewidths=0.35, zorder=8)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image


def single_animation_frame(
    rollout: dict[str, Any],
    setup_state: np.ndarray,
    safety_regions: list[SafetyRegion],
    frame_idx: int,
    title: str | None,
    color: str,
    show_title: bool = True,
    show_frame: bool = True,
    show_status: bool = True,
    safety_style: str = "normal",
) -> np.ndarray:
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6), dpi=180, constrained_layout=False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94 if show_title else 1)
    draw_topdown_base(
        ax,
        setup_state,
        safety_regions,
        title,
        show_title=show_title,
        show_frame=show_frame,
        safety_style=safety_style,
    )
    idx = min(int(frame_idx), len(rollout["agent_xy"]) - 1)
    traj = flip_toy_y(rollout["agent_xy"][: idx + 1])
    add_faded_path(ax, traj, color=color, linewidth=1.05, alpha_start=0.26, alpha_end=0.95, zorder=3)
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=14, marker="o", c=color, edgecolors="white", linewidths=0.4, zorder=8)
    if show_status:
        reached = [event["label_name"].replace("at_", "") for event in rollout.get("reaches", []) if int(event["t"]) <= idx]
        status = "safe" if not rollout["safety"]["violated"] or idx < int(rollout["safety"].get("first_violation_step") or 10**9) else "hit"
        ax.text(
            0.03,
            0.04,
            f"t={idx:03d}\n{status}\n{' -> '.join(reached[-3:]) if reached else 'running'}",
            transform=ax.transAxes,
            fontsize=6,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.82},
            zorder=20,
        )
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image


def save_single_rollout_video(
    rollout: dict[str, Any],
    setup_state: np.ndarray,
    safety_regions: list[SafetyRegion],
    output_path: Path,
    title: str | None,
    color: str,
    fps: int = 12,
    show_title: bool = True,
    show_frame: bool = True,
    show_status: bool = True,
    safety_style: str = "normal",
) -> Path:
    frames = [
        single_animation_frame(
            rollout,
            setup_state,
            safety_regions,
            idx,
            title=title,
            color=color,
            show_title=show_title,
            show_frame=show_frame,
            show_status=show_status,
            safety_style=safety_style,
        )
        for idx in range(len(rollout["agent_xy"]))
    ]
    try:
        imageio.mimsave(output_path, frames, fps=int(fps), macro_block_size=8)
        return output_path
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, duration=1.0 / max(int(fps), 1))
        return gif_path


def save_comparison_video(
    baseline: dict[str, Any],
    guided: dict[str, Any],
    setup_state: np.ndarray,
    safety_regions: list[SafetyRegion],
    output_path: Path,
    fps: int = 12,
) -> Path:
    max_len = max(len(baseline["agent_xy"]), len(guided["agent_xy"]))
    frame_indices = list(range(max_len))
    frames = [animation_frame(baseline, guided, setup_state, safety_regions, idx) for idx in frame_indices]
    try:
        imageio.mimsave(output_path, frames, fps=int(fps), macro_block_size=8)
        return output_path
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, duration=1.0 / max(int(fps), 1))
        return gif_path


def stack_or_object(values: list[Any]) -> np.ndarray:
    if not values:
        return np.array([])
    try:
        return np.stack(values)
    except Exception:
        return np.asarray(values, dtype=object)


def save_rollout_npz(run_dir: Path, rollout: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    low_level_obs = rollout["low_level_obs"]
    np.savez_compressed(
        run_dir / "low_level_obs.npz",
        states=stack_or_object([obs.get("states") for obs in low_level_obs]),
        agent_pos=stack_or_object([obs.get("agent_pos") for obs in low_level_obs]),
    )
    summary = {
        key: rollout[key]
        for key in [
            "name",
            "use_dynamics_guidance",
            "steps",
            "return",
            "target_reached",
            "chain_pos",
            "stage_pos",
            "stage_specs",
            "label_chain",
            "label_chain_names",
            "reaches",
            "stage_completions",
            "final_label",
            "safety",
            "route_waypoint",
            "route_waypoint_reached",
            "stage_route_waypoints",
            "stage_route_waypoints_reached",
            "records",
        ]
    }
    (run_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2))
    if rollout.get("env_frames"):
        imageio.mimsave(run_dir / "env_render.mp4", rollout["env_frames"], fps=12, macro_block_size=8)


def latest_dynamics_run(root: Path) -> Path:
    candidates = [path for path in Path(root).glob("*/best_model.pt") if path.exists()]
    if not candidates:
        raise FileNotFoundError(f"No dynamics checkpoints found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def load_components(args):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    device_str = str(device)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(args.dp_ckpt), device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=True, verbose=False)
    env.reset()
    automaton_model, automaton_stats, _, automaton_meta = load_automaton_model_for_eval(
        model_or_run_path=str(args.automaton_run),
        predictor_kind="learned",
        device=device,
        load_val_trajectories=False,
    )
    dynamics_path = args.dynamics_run if args.dynamics_run is not None else latest_dynamics_run(args.dynamics_root)
    dynamics_model, dynamics_stats, dynamics_ckpt, dynamics_meta = load_dynamics_model_for_eval(dynamics_path, device=device_str)
    dynamics_model.eval()
    refiner = DynamicsSafetyRefiner(dynamics_model, dynamics_stats, policy, device_str)
    return {
        "device": device_str,
        "policy": policy,
        "env": env,
        "automaton_model": automaton_model,
        "automaton_stats": automaton_stats,
        "automaton_meta": automaton_meta,
        "dynamics_path": Path(dynamics_meta["run_dir"]),
        "dynamics_ckpt": dynamics_ckpt,
        "refiner": refiner,
    }


def make_setup_state(env_seed: int, deterministic_setup: bool) -> np.ndarray:
    reseed(int(env_seed))
    return np.asarray(early_decision_cube_setup(deterministic=bool(deterministic_setup)), dtype=np.float32).copy()


def parse_label_chain(value: str | None, fallback_target_label_idx: int) -> list[int]:
    if value is None or not str(value).strip():
        return [int(fallback_target_label_idx)]
    chain = []
    for item in str(value).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in LABEL_NAME_TO_IDX:
            chain.append(LABEL_NAME_TO_IDX[token])
        else:
            chain.append(int(token))
    if not chain:
        raise ValueError("label chain must contain at least one target")
    for idx in chain:
        if idx < 0 or idx >= len(LABEL_NAMES):
            raise ValueError(f"Invalid label index {idx}; valid labels are 0..{len(LABEL_NAMES) - 1}")
    return chain


def parse_label_token(value: str) -> int:
    token = value.strip().lower()
    if not token:
        raise ValueError("Empty label token")
    if token in LABEL_NAME_TO_IDX:
        return int(LABEL_NAME_TO_IDX[token])
    return int(token)


def parse_stage_specs(stage_spec: str | None, label_chain: str | None, fallback_target_label_idx: int) -> list[StageSpec]:
    if stage_spec is None or not str(stage_spec).strip():
        return [
            StageSpec(mode="any", label_idxs=(int(idx),), raw=LABEL_NAMES[int(idx)])
            for idx in parse_label_chain(label_chain, fallback_target_label_idx)
        ]

    raw_spec = str(stage_spec).strip()
    normalized = (
        raw_spec.replace("(", "")
        .replace(")", "")
        .replace(" OR ", "|")
        .replace(" or ", "|")
        .replace(" AND ", "&")
        .replace(" and ", "&")
    )
    stage_tokens = [token.strip() for token in normalized.split(";") if token.strip()]
    if not stage_tokens:
        raise ValueError("stage spec must contain at least one stage")

    stages = []
    for token in stage_tokens:
        if "&" in token:
            mode = "all"
            label_tokens = [part.strip() for part in token.split("&") if part.strip()]
        elif "|" in token:
            mode = "any"
            label_tokens = [part.strip() for part in token.split("|") if part.strip()]
        else:
            mode = "any"
            label_tokens = [token.strip()]
        label_idxs = tuple(parse_label_token(part) for part in label_tokens)
        if not label_idxs:
            raise ValueError(f"Stage {token!r} has no labels")
        for idx in label_idxs:
            if idx < 0 or idx >= len(LABEL_NAMES):
                raise ValueError(f"Invalid label index {idx}; valid labels are 0..{len(LABEL_NAMES) - 1}")
        stages.append(StageSpec(mode=mode, label_idxs=label_idxs, raw=token))
    return stages


def build_safety_regions(args) -> list[SafetyRegion]:
    regions: list[SafetyRegion] = []
    if not args.disable_safety_box:
        regions.append(
            SafetyBox(
                *[float(v) for v in args.safety_box],
                angle_degrees=float(args.safety_box_angle_deg),
                name="safety_square",
            )
        )
    extra_angles = list(args.extra_safety_box_angle_deg or [])
    for idx, values in enumerate(args.extra_safety_box or []):
        angle = float(extra_angles[idx]) if idx < len(extra_angles) else 0.0
        regions.append(SafetyBox(*[float(v) for v in values], angle_degrees=angle, name=f"extra_box_{idx}"))
    if args.extra_safety_ellipse is not None:
        cx, cy, rx, ry = [float(v) for v in args.extra_safety_ellipse]
        regions.append(SafetyEllipse(cx=cx, cy=cy, rx=rx, ry=ry, name="yellow_green_ellipse"))
    return regions


def build_stage_route_waypoints(args) -> dict[int, np.ndarray]:
    waypoints: dict[int, np.ndarray] = {}
    for values in args.stage_route_waypoint or []:
        stage_idx = int(values[0])
        waypoints[stage_idx] = np.asarray([float(values[1]), float(values[2])], dtype=np.float32)
    return waypoints


def run_pair(args, components, output_dir: Path | None = None, capture_env_frames: bool = False) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    safety_regions = build_safety_regions(args)
    stage_specs = parse_stage_specs(args.stage_spec, args.label_chain, args.target_label_idx)
    stage_route_waypoints = build_stage_route_waypoints(args)
    setup_state = make_setup_state(args.env_seed, args.deterministic_setup)
    common = {
        "policy": components["policy"],
        "env": components["env"],
        "automaton_model": components["automaton_model"],
        "automaton_stats": components["automaton_stats"],
        "refiner": components["refiner"],
        "setup_state": setup_state,
        "rollout_seed": args.rollout_seed,
        "safety_regions": safety_regions,
        "horizon": args.horizon,
        "n_candidates": args.n_candidates,
        "stage_specs": stage_specs,
        "guidance_steps": args.guidance_steps,
        "adam_lr": args.adam_lr,
        "safety_scale": args.safety_scale,
        "goal_scale": args.goal_scale,
        "safety_margin": args.safety_margin,
        "safety_clearance_cap": args.safety_clearance_cap,
        "segment_samples": args.segment_samples,
        "max_step_length": args.max_step_length,
        "length_scale": args.length_scale,
        "length_variance_scale": args.length_variance_scale,
        "route_waypoint": None if args.route_waypoint is None else np.asarray(args.route_waypoint, dtype=np.float32),
        "stage_route_waypoints": stage_route_waypoints,
        "route_waypoint_radius": args.route_waypoint_radius,
        "smooth_min_tau": args.smooth_min_tau,
        "goal_tau": args.goal_tau,
        "capture_env_frames": capture_env_frames,
        "device": components["device"],
    }
    baseline = run_rollout(name="baseline_no_dynamics_safety", use_dynamics_guidance=False, **common)
    guided = run_rollout(name="guided_dynamics_safety", use_dynamics_guidance=True, **common)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "setup_state.npy", setup_state)
        save_rollout_npz(output_dir / "baseline", baseline)
        save_rollout_npz(output_dir / "guided", guided)
        plot_path = output_dir / "topdown_safety_comparison.png"
        save_topdown_comparison(baseline, guided, setup_state, safety_regions, plot_path)
        video_path = save_comparison_video(
            baseline,
            guided,
            setup_state,
            safety_regions,
            output_dir / "safety_guidance_comparison.mp4",
            fps=args.video_fps,
        )
        baseline_topdown_video_path = save_single_rollout_video(
            baseline,
            setup_state,
            safety_regions,
            output_dir / "baseline" / "topdown_rollout.mp4",
            title="High Level Guidance",
            color=PAPER_GRAY,
            fps=args.video_fps,
            show_title=True,
            show_frame=False,
            show_status=False,
            safety_style="faint",
        )
        guided_topdown_video_path = save_single_rollout_video(
            guided,
            setup_state,
            safety_regions,
            output_dir / "guided" / "topdown_rollout.mp4",
            title="Hierarchical Guidance",
            color=PAPER_BLUE,
            fps=args.video_fps,
            show_title=True,
            show_frame=False,
            show_status=False,
            safety_style="normal",
        )
        summary = build_summary(
            args,
            components,
            safety_regions,
            stage_specs,
            baseline,
            guided,
            output_dir,
            plot_path,
            video_path,
            baseline_topdown_video_path,
            guided_topdown_video_path,
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return baseline, guided, setup_state


def build_summary(
    args,
    components,
    safety_regions: list[SafetyRegion],
    stage_specs: list[StageSpec],
    baseline: dict[str, Any],
    guided: dict[str, Any],
    output_dir: Path,
    plot_path: Path,
    video_path: Path,
    baseline_topdown_video_path: Path | None = None,
    guided_topdown_video_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "plot_path": str(plot_path),
        "video_path": str(video_path),
        "baseline_topdown_video_path": None if baseline_topdown_video_path is None else str(baseline_topdown_video_path),
        "guided_topdown_video_path": None if guided_topdown_video_path is None else str(guided_topdown_video_path),
        "dp_ckpt": str(args.dp_ckpt),
        "automaton_run": str(args.automaton_run),
        "dynamics_run": str(components["dynamics_path"]),
        "target_label_idx": int(args.target_label_idx),
        "target_label": LABEL_NAMES[int(args.target_label_idx)],
        "stage_spec": args.stage_spec,
        "stage_specs": [stage_to_dict(stage) for stage in stage_specs],
        "formula_display": " THEN ".join(stage.display for stage in stage_specs),
        "label_chain": [[int(idx) for idx in stage.label_idxs] for stage in stage_specs],
        "label_chain_names": [stage.display for stage in stage_specs],
        "safety_regions": [region_to_dict(region) for region in safety_regions],
        "env_seed": int(args.env_seed),
        "rollout_seed": int(args.rollout_seed),
        "n_candidates": int(args.n_candidates),
        "guidance": {
            "guidance_steps": int(args.guidance_steps),
            "adam_lr": float(args.adam_lr),
            "safety_scale": float(args.safety_scale),
            "goal_scale": float(args.goal_scale),
            "safety_margin": float(args.safety_margin),
            "safety_clearance_cap": float(args.safety_clearance_cap),
            "segment_samples": int(args.segment_samples),
            "max_step_length": float(args.max_step_length),
            "length_scale": float(args.length_scale),
            "length_variance_scale": float(args.length_variance_scale),
            "route_waypoint": None if args.route_waypoint is None else [float(v) for v in args.route_waypoint],
            "stage_route_waypoints": {
                str(int(values[0])): [float(values[1]), float(values[2])]
                for values in args.stage_route_waypoint or []
            },
            "route_waypoint_radius": float(args.route_waypoint_radius),
            "smooth_min_tau": float(args.smooth_min_tau),
            "goal_tau": float(args.goal_tau),
            "action_regularization": 0.0,
        },
        "baseline": {
            "target_reached": bool(baseline["target_reached"]),
            "chain_pos": int(baseline["chain_pos"]),
            "stage_pos": int(baseline["stage_pos"]),
            "reaches": baseline["reaches"],
            "stage_completions": baseline["stage_completions"],
            "steps": int(baseline["steps"]),
            "safety_violated": bool(baseline["safety"]["violated"]),
            "first_violation_step": baseline["safety"]["first_violation_step"],
            "first_violation_region": baseline["safety"]["first_violation_region"],
            "min_signed_distance": float(baseline["safety"]["min_signed_distance"]),
        },
        "guided": {
            "target_reached": bool(guided["target_reached"]),
            "chain_pos": int(guided["chain_pos"]),
            "stage_pos": int(guided["stage_pos"]),
            "reaches": guided["reaches"],
            "stage_completions": guided["stage_completions"],
            "steps": int(guided["steps"]),
            "safety_violated": bool(guided["safety"]["violated"]),
            "first_violation_step": guided["safety"]["first_violation_step"],
            "first_violation_region": guided["safety"]["first_violation_region"],
            "min_signed_distance": float(guided["safety"]["min_signed_distance"]),
        },
        "success_condition_met": bool(
            baseline["safety"]["violated"] and guided["target_reached"] and not guided["safety"]["violated"]
        ),
    }


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def search(args, components) -> Path | None:
    base_output = args.output_root / f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_output.mkdir(parents=True, exist_ok=True)
    rows = []
    original = argparse.Namespace(**vars(args))

    for env_seed in parse_int_list(args.search_env_seeds):
        for rollout_seed in parse_int_list(args.search_rollout_seeds):
            args.env_seed = env_seed
            args.rollout_seed = rollout_seed
            baseline, _, _ = run_pair(args, components, output_dir=None, capture_env_frames=False)
            baseline_ok = bool(baseline["safety"]["violated"])
            for steps in parse_int_list(args.search_guidance_steps):
                for adam_lr in parse_float_list(args.search_adam_lrs):
                    for safety_scale in parse_float_list(args.search_safety_scales):
                        for goal_scale in parse_float_list(args.search_goal_scales):
                            args.guidance_steps = steps
                            args.adam_lr = adam_lr
                            args.safety_scale = safety_scale
                            args.goal_scale = goal_scale
                            baseline2, guided, _ = run_pair(args, components, output_dir=None, capture_env_frames=False)
                            row = {
                                "env_seed": env_seed,
                                "rollout_seed": rollout_seed,
                                "guidance_steps": steps,
                                "adam_lr": adam_lr,
                                "safety_scale": safety_scale,
                                "goal_scale": goal_scale,
                                "baseline_violated": bool(baseline2["safety"]["violated"]),
                                "baseline_reached": bool(baseline2["target_reached"]),
                                "guided_violated": bool(guided["safety"]["violated"]),
                                "guided_reached": bool(guided["target_reached"]),
                                "baseline_min_signed": float(baseline2["safety"]["min_signed_distance"]),
                                "guided_min_signed": float(guided["safety"]["min_signed_distance"]),
                                "guided_steps": int(guided["steps"]),
                            }
                            rows.append(row)
                            (base_output / "search_rows.json").write_text(json.dumps(rows, indent=2))
                            print(
                                "search "
                                f"env={env_seed} roll={rollout_seed} steps={steps} lr={adam_lr:g} "
                                f"safety={safety_scale:g} goal={goal_scale:g} | "
                                f"base_hit={row['baseline_violated']} guided_hit={row['guided_violated']} "
                                f"guided_blue={row['guided_reached']} min={row['guided_min_signed']:.4f}",
                                flush=True,
                            )
                            if baseline_ok and row["baseline_violated"] and row["guided_reached"] and not row["guided_violated"]:
                                final_dir = base_output / (
                                    f"winner_env{env_seed:03d}_roll{rollout_seed:03d}_"
                                    f"steps{steps}_lr{adam_lr:g}_safety{safety_scale:g}_goal{goal_scale:g}"
                                )
                                run_pair(args, components, output_dir=final_dir, capture_env_frames=args.save_env_videos)
                                print(f"Found tuned rollout: {final_dir}", flush=True)
                                return final_dir

    args.__dict__.update(vars(original))
    print(f"No successful tuned rollout found. Search rows: {base_output / 'search_rows.json'}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dp-ckpt", type=Path, default=DEFAULT_DP_CKPT)
    parser.add_argument("--automaton-run", type=Path, default=DEFAULT_AUTOMATON_RUN)
    parser.add_argument("--dynamics-run", type=Path, default=None)
    parser.add_argument("--dynamics-root", type=Path, default=DEFAULT_DYNAMICS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--rollout-seed", type=int, default=4)
    parser.add_argument("--deterministic-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--target-label-idx", type=int, default=BLUE_LABEL_IDX)
    parser.add_argument("--label-chain", default=None)
    parser.add_argument("--stage-spec", default=None)
    parser.add_argument("--safety-box", type=float, nargs=4, default=[-0.36, -0.14, -0.36, -0.14])
    parser.add_argument("--safety-box-angle-deg", type=float, default=0.0)
    parser.add_argument("--disable-safety-box", action="store_true")
    parser.add_argument("--extra-safety-box", type=float, nargs=4, action="append", default=None)
    parser.add_argument("--extra-safety-box-angle-deg", type=float, action="append", default=None)
    parser.add_argument("--extra-safety-ellipse", type=float, nargs=4, default=None)
    parser.add_argument("--guidance-steps", type=int, default=40)
    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument("--safety-scale", type=float, default=10.0)
    parser.add_argument("--goal-scale", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--safety-clearance-cap", type=float, default=0.02)
    parser.add_argument("--segment-samples", type=int, default=12)
    parser.add_argument("--max-step-length", type=float, default=0.10)
    parser.add_argument("--length-scale", type=float, default=50.0)
    parser.add_argument("--length-variance-scale", type=float, default=5.0)
    parser.add_argument("--route-waypoint", type=float, nargs=2, default=None)
    parser.add_argument("--stage-route-waypoint", type=float, nargs=3, action="append", default=None)
    parser.add_argument("--route-waypoint-radius", type=float, default=0.08)
    parser.add_argument("--smooth-min-tau", type=float, default=0.03)
    parser.add_argument("--goal-tau", type=float, default=0.04)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--save-env-videos", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-env-seeds", default="0,1,2,3,4")
    parser.add_argument("--search-rollout-seeds", default="0,1,2,3,4,5,6")
    parser.add_argument("--search-guidance-steps", default="20,40,60,80")
    parser.add_argument("--search-adam-lrs", default="0.01,0.03,0.05,0.08")
    parser.add_argument("--search-safety-scales", default="5,10,20,40")
    parser.add_argument("--search-goal-scales", default="0.5,1,2")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    components = load_components(args)
    stage_specs = parse_stage_specs(args.stage_spec, args.label_chain, args.target_label_idx)
    safety_regions = build_safety_regions(args)
    print(f"Loaded dynamics: {components['dynamics_path']}", flush=True)
    print(f"Formula: {' THEN '.join(stage.display for stage in stage_specs)}", flush=True)
    print(f"Safety regions: {[region_to_dict(region) for region in safety_regions]}", flush=True)

    if args.search:
        winner = search(args, components)
        if winner is None:
            raise SystemExit(1)
        return

    output_dir = args.output_root / (
        f"formula_{'-'.join(stage.raw.replace('|', 'or').replace('&', 'and') for stage in stage_specs)}_"
        f"env{args.env_seed:03d}_roll{args.rollout_seed:03d}_"
        f"steps{args.guidance_steps}_lr{args.adam_lr:g}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    baseline, guided, _ = run_pair(args, components, output_dir=output_dir, capture_env_frames=args.save_env_videos)
    summary_path = output_dir / "summary.json"
    print(
        "baseline: "
        f"complete={baseline['target_reached']} safety_hit={baseline['safety']['violated']} "
        f"min_signed={baseline['safety']['min_signed_distance']:.4f}",
        flush=True,
    )
    print(
        "guided: "
        f"complete={guided['target_reached']} safety_hit={guided['safety']['violated']} "
        f"min_signed={guided['safety']['min_signed_distance']:.4f}",
        flush=True,
    )
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
