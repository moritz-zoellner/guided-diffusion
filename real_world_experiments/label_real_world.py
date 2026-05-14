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
import h5py
import numpy as np

try:
    from real_world_data import (
        get_demo_keys,
        quat_wxyz_conjugate,
        quat_wxyz_multiply,
        quat_wxyz_to_rotvec,
        read_obs_array,
    )
except ModuleNotFoundError:
    from real_world_experiments.real_world_data import (
        get_demo_keys,
        quat_wxyz_conjugate,
        quat_wxyz_multiply,
        quat_wxyz_to_rotvec,
        read_obs_array,
    )


LABEL_NAMES = ["can_grabbed", "pouring_right", "pouring_left"]


@dataclass(frozen=True)
class LabelConfig:
    gripper_closed_threshold: float = 128.5
    pour_twist_on_deg: float = 90.0
    pour_twist_off_deg: float = 75.0
    positive_twist_is_right: bool = True
    debounce_steps: int = 3


def load_label_config(path: Path | str | None) -> LabelConfig:
    if path is None:
        return LabelConfig()
    payload = json.loads(Path(path).read_text())
    known_fields = set(LabelConfig.__dataclass_fields__)
    payload = {key: value for key, value in payload.items() if key in known_fields}
    return LabelConfig(**payload)


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


def _hysteresis_greater_than(values: np.ndarray, on_threshold: float, off_threshold: float) -> np.ndarray:
    active = False
    out = np.zeros(len(values), dtype=bool)
    for idx, value in enumerate(values):
        if active:
            active = value > off_threshold
        else:
            active = value > on_threshold
        out[idx] = active
    return out


def label_arrays(
    eef_quat_wxyz: np.ndarray,
    gripper_width: np.ndarray,
    config: LabelConfig | None = None,
) -> np.ndarray:
    """Return [can_grabbed, pouring_right, pouring_left] labels for every timestep.

    Pouring labels use the relative twist of the gripper around its local z axis
    from the start pose of each trajectory:

        q_t = q_start * q_delta_local

    so positive / negative ``rotvec_z(q_delta_local)`` correspond to the two
    repeatable wrist-twist directions used to pour into the two bowls.
    """
    config = LabelConfig() if config is None else config
    eef_quat_wxyz = np.asarray(eef_quat_wxyz, dtype=np.float32)
    gripper_width = np.asarray(gripper_width, dtype=np.float32).reshape(len(eef_quat_wxyz), -1)[:, 0]

    q_start = np.repeat(eef_quat_wxyz[:1], len(eef_quat_wxyz), axis=0)
    q_delta_local = quat_wxyz_multiply(quat_wxyz_conjugate(q_start), eef_quat_wxyz)
    local_twist = quat_wxyz_to_rotvec(q_delta_local)[:, 2]

    on = np.deg2rad(float(config.pour_twist_on_deg))
    off = np.deg2rad(float(config.pour_twist_off_deg))
    positive_twist = _hysteresis_greater_than(local_twist, on, off)
    negative_twist = _hysteresis_less_than(local_twist, -on, -off)

    grabbed_raw = gripper_width >= float(config.gripper_closed_threshold)
    grabbed = _debounce_binary(grabbed_raw, config.debounce_steps)
    positive_pour = _debounce_binary(grabbed_raw & positive_twist, config.debounce_steps)
    negative_pour = _debounce_binary(grabbed_raw & negative_twist, config.debounce_steps)

    if config.positive_twist_is_right:
        pouring_right, pouring_left = positive_pour, negative_pour
    else:
        pouring_right, pouring_left = negative_pour, positive_pour

    return np.stack([grabbed, pouring_right, pouring_left], axis=-1).astype(np.float32)


def label_demo(
    demo: h5py.Group,
    config: LabelConfig | None = None,
    eef_quat_key: str = "eef_quat_wxyz",
    gripper_width_key: str = "gripper_width",
    **_: object,
) -> np.ndarray:
    return label_arrays(
        read_obs_array(demo, eef_quat_key),
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
                eef_quat_key=args.eef_quat_key,
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
    parser.add_argument("--eef-quat-key", default="eef_quat_wxyz")
    parser.add_argument("--gripper-width-key", default="gripper_width")
    args = parser.parse_args()
    write_label_diagnostics(args)


if __name__ == "__main__":
    main()
