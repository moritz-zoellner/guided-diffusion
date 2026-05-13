"""Shared HDF5 and pose helpers for the real-world Cheez-It experiments.

Expected converted dataset layout is robomimic-like:

    data/demo_i/actions
    data/demo_i/obs/<observation_key>
    mask/train, mask/valid        optional

The final selection-pipeline converter can choose exact observation key names;
the training scripts expose the key lists as command-line arguments.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


DEFAULT_STATE_KEYS = (
    "eef_pos",
    "eef_rot6d",
    "gripper_width",
    "cheezit_pos",
    "cheezit_rot6d",
)
DEFAULT_ACTION_KEY = "actions"


def demo_sort_key(name: str) -> int:
    match = re.search(r"\d+", name)
    return int(match.group()) if match else 0


def get_demo_keys(hdf5_path: Path | str, mask: str | None = None) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        if mask is None:
            keys = list(f["data"].keys())
        else:
            keys = [key.decode("utf-8") for key in np.asarray(f[f"mask/{mask}"])]
    return sorted(keys, key=demo_sort_key)


def split_trajectories(trajectories: Sequence[dict], val_ratio: float, seed: int):
    if len(trajectories) < 2:
        raise ValueError("Need at least 2 trajectories for a train/val split.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(trajectories))
    n_val = max(1, int(round(len(trajectories) * val_ratio)))
    n_val = min(len(trajectories) - 1, n_val)
    val_indices = set(int(idx) for idx in permutation[:n_val])
    train = [traj for idx, traj in enumerate(trajectories) if idx not in val_indices]
    val = [traj for idx, traj in enumerate(trajectories) if idx in val_indices]
    return train, val


def parse_key_list(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) in wxyz order to the first two rotation columns."""
    q = normalize_quat_wxyz(quat)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r20 = 2.0 * (x * z - y * w)

    r01 = 2.0 * (x * y - z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r21 = 2.0 * (y * z + x * w)

    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation columns to a 3x3 matrix using Gram-Schmidt."""
    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2_proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def rotation_zz_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    return rot6d_to_matrix(rot6d)[..., 2, 2].astype(np.float32)


def _read_obs_key(demo: h5py.Group, key: str) -> np.ndarray:
    obs = demo["obs"]
    if key in obs:
        return obs[key][:].astype(np.float32)

    if key.endswith("_rot6d"):
        quat_key = key[: -len("_rot6d")] + "_quat"
        if quat_key in obs:
            return quat_wxyz_to_rot6d(obs[quat_key][:])

    raise KeyError(
        f"Observation key obs/{key!r} not found in demo {demo.name}. "
        "If the source data stores quaternions, name the requested key '*_rot6d' "
        "and provide the corresponding '*_quat' key in wxyz order."
    )


def build_state_from_demo(demo: h5py.Group, state_keys: Sequence[str]) -> np.ndarray:
    values = [_read_obs_key(demo, key) for key in state_keys]
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"State observation lengths do not align in {demo.name}: {sorted(lengths)}")
    return np.concatenate(values, axis=-1).astype(np.float32)


def read_obs_array(demo: h5py.Group, key: str) -> np.ndarray:
    return _read_obs_key(demo, key)

