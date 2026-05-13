"""Predicate labels for the real-world Cheez-It pouring task.

These predicates intentionally describe clean geometric events, not real
granular pouring. Bowl positions are fixed in the table frame; Cheez-It pose is
estimated by FoundationPose and converted to the same table/robot frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

try:
    from real_world_data import get_demo_keys, read_obs_array, rotation_zz_from_rot6d
except ModuleNotFoundError:
    from real_world_experiments.real_world_data import get_demo_keys, read_obs_array, rotation_zz_from_rot6d


LABEL_NAMES = ["pour_left_bowl", "pour_right_bowl", "cheezits_released"]


@dataclass(frozen=True)
class BowlConfig:
    left_center_xy: tuple[float, float] = (-0.18, 0.35)
    right_center_xy: tuple[float, float] = (0.18, 0.35)
    radius: float = 0.09


@dataclass(frozen=True)
class LabelConfig:
    bowl: BowlConfig = BowlConfig()
    pour_rzz_on: float = -0.05
    pour_rzz_off: float = 0.02
    release_gripper_width: float = 100.0
    release_table_z: float = 0.08
    release_table_z_margin: float = 0.025
    debounce_steps: int = 3


def load_label_config(path: Path | str | None) -> LabelConfig:
    if path is None:
        return LabelConfig()
    payload = json.loads(Path(path).read_text())
    bowl_payload = payload.pop("bowl", {})
    return LabelConfig(bowl=BowlConfig(**bowl_payload), **payload)


def _debounce_binary(values: np.ndarray, steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    steps = int(steps)
    if steps <= 1 or len(values) == 0:
        return values.astype(np.float32)
    out = np.zeros_like(values, dtype=bool)
    run = 0
    for idx, value in enumerate(values):
        run = run + 1 if value else 0
        out[idx] = run >= steps
    return out.astype(np.float32)


def _hysteresis_less_than(values: np.ndarray, on_threshold: float, off_threshold: float) -> np.ndarray:
    active = False
    out = np.zeros(len(values), dtype=bool)
    for idx, value in enumerate(values):
        if active:
            active = value < off_threshold
        else:
            active = value < on_threshold
        out[idx] = active
    return out


def _inside_bowl_xy(positions: np.ndarray, center_xy: Sequence[float], radius: float) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float32)
    dist = np.linalg.norm(positions[:, :2] - center[None, :], axis=-1)
    return dist <= float(radius)


def label_arrays(
    cheezit_pos: np.ndarray,
    cheezit_rot6d: np.ndarray,
    gripper_width: np.ndarray,
    config: LabelConfig | None = None,
) -> np.ndarray:
    """Return [pour_left, pour_right, released] labels for every timestep."""
    config = LabelConfig() if config is None else config
    cheezit_pos = np.asarray(cheezit_pos, dtype=np.float32)
    cheezit_rot6d = np.asarray(cheezit_rot6d, dtype=np.float32)
    gripper_width = np.asarray(gripper_width, dtype=np.float32).reshape(len(cheezit_pos), -1)[:, 0]

    rzz = rotation_zz_from_rot6d(cheezit_rot6d)
    tilted = _hysteresis_less_than(rzz, config.pour_rzz_on, config.pour_rzz_off)
    left_region = _inside_bowl_xy(cheezit_pos, config.bowl.left_center_xy, config.bowl.radius)
    right_region = _inside_bowl_xy(cheezit_pos, config.bowl.right_center_xy, config.bowl.radius)

    pour_left = _debounce_binary(tilted & left_region, config.debounce_steps)
    pour_right = _debounce_binary(tilted & right_region, config.debounce_steps)

    released_raw = (
        (gripper_width >= float(config.release_gripper_width))
        & (cheezit_pos[:, 2] <= float(config.release_table_z + config.release_table_z_margin))
    )
    released = _debounce_binary(released_raw, config.debounce_steps)

    return np.stack([pour_left, pour_right, released], axis=-1).astype(np.float32)


def label_demo(
    demo: h5py.Group,
    config: LabelConfig | None = None,
    cheezit_pos_key: str = "cheezit_pos",
    cheezit_rot_key: str = "cheezit_rot6d",
    gripper_width_key: str = "gripper_width",
) -> np.ndarray:
    return label_arrays(
        read_obs_array(demo, cheezit_pos_key),
        read_obs_array(demo, cheezit_rot_key),
        read_obs_array(demo, gripper_width_key),
        config=config,
    )


def next_changed_labels(labels: np.ndarray, immediate_next_labels: np.ndarray) -> np.ndarray:
    """Target the next future label configuration after the current plateau."""
    if len(labels) != len(immediate_next_labels):
        raise ValueError("labels and immediate_next_labels must have the same length")
    if len(labels) == 0:
        return labels.copy()
    if len(labels) == 1:
        return immediate_next_labels.copy()

    out = np.empty_like(labels)
    previous = immediate_next_labels[-2].copy()
    carry = immediate_next_labels[-1].copy()
    for idx in range(len(labels) - 1, -1, -1):
        current = labels[idx]
        if np.array_equal(current, previous):
            out[idx] = carry
        else:
            out[idx] = previous
            carry = previous.copy()
        previous = current.copy()
    return out


def horizon_eventual_labels(labels: np.ndarray, next_labels: np.ndarray, action_horizon: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    next_labels = np.asarray(next_labels, dtype=np.float32)
    n_chunks = len(labels) - int(action_horizon)
    if n_chunks <= 0:
        return np.empty((0, labels.shape[-1]), dtype=np.float32)
    targets = np.empty((n_chunks, labels.shape[-1]), dtype=np.float32)
    for idx in range(n_chunks):
        targets[idx] = next_labels[idx : idx + action_horizon + 1].max(axis=0)
    return targets


TARGET_RULE_DESCRIPTIONS = {
    "max_next": "max(next_changed_labels[t:t+H+1])",
    "next_tH": "next_changed_labels[t+H]",
}


def horizon_targets(labels: np.ndarray, next_labels: np.ndarray, action_horizon: int, target_rule: str = "max_next"):
    if target_rule == "max_next":
        return horizon_eventual_labels(labels, next_labels, action_horizon)
    if target_rule != "next_tH":
        raise ValueError(f"Unknown target_rule {target_rule!r}")
    n_chunks = len(labels) - int(action_horizon)
    if n_chunks <= 0:
        return np.empty((0, labels.shape[-1]), dtype=np.float32)
    return np.asarray(next_labels[action_horizon:], dtype=np.float32)


def write_label_diagnostics(args):
    config = load_label_config(args.label_config)
    summary = {"label_names": LABEL_NAMES, "label_config": asdict(config), "demos": {}}
    keys = get_demo_keys(args.dataset, mask=args.mask)
    with h5py.File(args.dataset, "r") as f:
        for key in keys:
            labels = label_demo(
                f[f"data/{key}"],
                config=config,
                cheezit_pos_key=args.cheezit_pos_key,
                cheezit_rot_key=args.cheezit_rot_key,
                gripper_width_key=args.gripper_width_key,
            )
            summary["demos"][key] = {
                name: int(labels[:, idx].sum()) for idx, name in enumerate(LABEL_NAMES)
            } | {"num_steps": int(len(labels))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote label diagnostics: {args.output}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/real_world/label_diagnostics.json"))
    parser.add_argument("--mask", default=None)
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--cheezit-pos-key", default="cheezit_pos")
    parser.add_argument("--cheezit-rot-key", default="cheezit_rot6d")
    parser.add_argument("--gripper-width-key", default="gripper_width")
    args = parser.parse_args()
    write_label_diagnostics(args)


if __name__ == "__main__":
    main()
