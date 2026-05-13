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
import argparse
import json
import pickle
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
DEFAULT_SOURCE_DIRS = (
    Path("/home/moritz/src/ros-ws/saved_data/20260512_231419"),
    Path("/home/moritz/src/ros-ws/saved_data/20260513_004156"),
    Path("/home/moritz/src/ros-ws/saved_data/20260513_015147"),
)
DEFAULT_OUTPUT_PATH = Path("data/real_world/cheezit_pouring.hdf5")


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


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    return np.concatenate([quat[..., 1:4], quat[..., 0:1]], axis=-1).astype(np.float32)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    quat = np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1)
    return normalize_quat_wxyz(quat)


def quat_wxyz_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    out = quat.copy()
    out[..., 1:4] *= -1.0
    return out


def quat_wxyz_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_quat_wxyz(left)
    right = normalize_quat_wxyz(right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    quat = np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )
    return normalize_quat_wxyz(quat.astype(np.float32))


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat).astype(np.float64)
    quat = np.where(quat[..., 0:1] < 0.0, -quat, quat)
    vector = quat[..., 1:4]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.maximum(quat[..., 0:1], 1e-12))
    axis = vector / np.maximum(vector_norm, 1e-12)
    rotvec = angle * axis
    small = vector_norm[..., 0] < 1e-8
    if np.any(small):
        rotvec[small] = 2.0 * vector[small]
    return rotvec.astype(np.float32)


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    axis = rotvec / np.maximum(angle, 1e-12)
    quat = np.concatenate([np.cos(half_angle), axis * np.sin(half_angle)], axis=-1)
    small = angle[..., 0] < 1e-8
    if np.any(small):
        quat[small, 0] = 1.0
        quat[small, 1:4] = 0.5 * rotvec[small]
    return normalize_quat_wxyz(quat.astype(np.float32))


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


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix/matrices to normalized quaternions in wxyz order."""
    matrix = np.asarray(matrix, dtype=np.float64)
    flat = matrix.reshape(-1, 3, 3)
    quats = np.empty((len(flat), 4), dtype=np.float64)

    for idx, rot in enumerate(flat):
        trace = np.trace(rot)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
        quats[idx] = [qw, qx, qy, qz]

    quats = quats.reshape(matrix.shape[:-2] + (4,))
    return normalize_quat_wxyz(quats.astype(np.float32))


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1).astype(np.float32)


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


def episode_sort_key(path: Path) -> tuple[str, int]:
    return (path.parent.name, demo_sort_key(path.stem))


def discover_episode_paths(source_dirs: Sequence[Path | str]) -> list[Path]:
    episode_paths = []
    for source_dir in source_dirs:
        source_dir = Path(source_dir).expanduser()
        episode_paths.extend(source_dir.glob("episode_*.pkl"))
    return sorted(episode_paths, key=episode_sort_key)


def _load_episode(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _column(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value[:, None] if value.ndim == 1 else value


def _synthesized_pose_actions(
    eef_pos: np.ndarray,
    eef_quat_wxyz: np.ndarray,
    gripper_pos: np.ndarray,
    gripper_mid: float,
) -> np.ndarray:
    """Build [dpos, drotvec, next_gripper_binary] actions from measured states."""
    n_steps = len(eef_pos)
    dpos = np.zeros((n_steps, 3), dtype=np.float32)
    drot = np.zeros((n_steps, 3), dtype=np.float32)

    if n_steps > 1:
        dpos[:-1] = eef_pos[1:] - eef_pos[:-1]
        q_current = eef_quat_wxyz[:-1]
        q_next = eef_quat_wxyz[1:]
        q_delta = quat_wxyz_multiply(q_next, quat_wxyz_conjugate(q_current))
        drot[:-1] = quat_wxyz_to_rotvec(q_delta)

    gripper_binary = _gripper_binary(gripper_pos, gripper_mid)
    next_gripper = np.empty((n_steps, 1), dtype=np.float32)
    if n_steps > 1:
        next_gripper[:-1, 0] = gripper_binary[1:]
    next_gripper[-1, 0] = gripper_binary[-1]

    return np.concatenate([dpos, drot, next_gripper], axis=-1).astype(np.float32)


def _gripper_binary(gripper_pos: np.ndarray, gripper_mid: float) -> np.ndarray:
    gripper_raw = np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
    return np.where(gripper_raw < gripper_mid, -1.0, 1.0).astype(np.float32)


def _episode_length(episode: dict) -> int:
    lengths = []
    for key in ["ee_pos", "ee_quat_wxyz", "gripper_pos", "action", "obj_pose_4x4_world"]:
        if key not in episode:
            raise KeyError(f"Episode missing required key: {key}")
        lengths.append(len(episode[key]))
    if len(set(lengths)) != 1:
        raise ValueError(f"Episode arrays have inconsistent lengths: {dict(zip(['ee_pos','ee_quat_wxyz','gripper_pos','action','obj_pose_4x4_world'], lengths))}")
    return lengths[0]


def _compact_episode_arrays(episode: dict, gripper_mid: float) -> dict[str, np.ndarray]:
    n_steps = _episode_length(episode)

    eef_pos = _as_float32(episode["ee_pos"])
    eef_quat_wxyz = normalize_quat_wxyz(episode["ee_quat_wxyz"])
    eef_quat_xyzw = quat_wxyz_to_xyzw(eef_quat_wxyz)
    eef_rot6d = quat_wxyz_to_rot6d(eef_quat_wxyz)
    gripper = _column(episode["gripper_pos"])
    gripper_binary = _column(_gripper_binary(episode["gripper_pos"], gripper_mid))

    obj_pose_world = np.asarray(episode["obj_pose_4x4_world"], dtype=np.float32)
    obj_pos = obj_pose_world[:, :3, 3].astype(np.float32)
    obj_rot = obj_pose_world[:, :3, :3].astype(np.float32)
    obj_quat_wxyz = matrix_to_quat_wxyz(obj_rot)
    obj_quat_xyzw = quat_wxyz_to_xyzw(obj_quat_wxyz)
    obj_rot6d = matrix_to_rot6d(obj_rot)
    gripper_to_obj_pos = (obj_pos - eef_pos).astype(np.float32)
    object_obs = np.concatenate([obj_pos, obj_quat_xyzw, gripper_to_obj_pos], axis=-1).astype(np.float32)

    actions = _synthesized_pose_actions(eef_pos, eef_quat_wxyz, episode["gripper_pos"], gripper_mid)
    logged_actions = _as_float32(episode["action"])
    dones = np.zeros(n_steps, dtype=bool)
    dones[-1] = True

    return {
        "actions": actions,
        "actions_logged_xyz": logged_actions,
        "rewards": np.zeros(n_steps, dtype=np.float32),
        "dones": dones,
        "obs/robot0_eef_pos": eef_pos,
        "obs/robot0_eef_quat": eef_quat_xyzw,
        "obs/robot0_gripper_qpos": gripper_binary,
        "obs/object": object_obs,
        "obs/eef_pos": eef_pos,
        "obs/eef_quat_wxyz": eef_quat_wxyz,
        "obs/eef_rot6d": eef_rot6d,
        "obs/gripper_width": gripper,
        "obs/gripper_binary": gripper_binary,
        "obs/cheezit_pos": obj_pos,
        "obs/cheezit_quat_wxyz": obj_quat_wxyz,
        "obs/cheezit_quat": obj_quat_xyzw,
        "obs/cheezit_rot6d": obj_rot6d,
        "obs/cheezit_to_eef_pos": gripper_to_obj_pos,
        "states": np.empty((0,), dtype=np.float32),
    }


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray):
    value = np.asarray(value)
    compression = "gzip" if value.size > 0 and value.ndim > 0 else None
    if "/" in name:
        parent_name, dataset_name = name.rsplit("/", 1)
        parent = group.require_group(parent_name)
    else:
        parent = group
        dataset_name = name
    parent.create_dataset(dataset_name, data=value, compression=compression)


def _stratified_train_valid_masks(demo_records: Sequence[dict], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    train, valid = [], []
    by_session: dict[str, list[str]] = {}
    for record in demo_records:
        by_session.setdefault(record["session"], []).append(record["demo_key"])
    for keys in by_session.values():
        keys = list(keys)
        rng.shuffle(keys)
        n_val = max(1, int(round(len(keys) * val_ratio))) if len(keys) > 1 else 0
        n_val = min(len(keys) - 1, n_val) if len(keys) > 1 else 0
        valid.extend(keys[:n_val])
        train.extend(keys[n_val:])
    return sorted(train, key=demo_sort_key), sorted(valid, key=demo_sort_key)


def convert_saved_episodes_to_hdf5(
    source_dirs: Sequence[Path | str] = DEFAULT_SOURCE_DIRS,
    output: Path | str = DEFAULT_OUTPUT_PATH,
    val_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
    min_length: int = 2,
) -> dict:
    """Read ROS saved episode pickle files and write a compact training HDF5.

    Source ``episode_*.pkl`` files are opened read-only and never modified.
    """
    output = Path(output).expanduser()
    episode_paths = discover_episode_paths(source_dirs)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.pkl files found under {list(source_dirs)}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")

    gripper_mins = []
    gripper_maxs = []
    for path in episode_paths:
        episode = _load_episode(path)
        n_steps = _episode_length(episode)
        if n_steps >= min_length:
            gripper = np.asarray(episode["gripper_pos"], dtype=np.float32)
            gripper_mins.append(float(np.min(gripper)))
            gripper_maxs.append(float(np.max(gripper)))
    if not gripper_mins:
        raise ValueError(f"No episodes with at least {min_length} samples found.")
    gripper_min = float(np.min(gripper_mins))
    gripper_max = float(np.max(gripper_maxs))
    gripper_mid = 0.5 * (gripper_min + gripper_max)

    output.parent.mkdir(parents=True, exist_ok=True)
    demo_records = []
    total_samples = 0
    skipped = []

    env_meta = {
        "env_name": "RealWorldCheezitPouring",
        "type": 2,
        "env_kwargs": {
            "source": "ros_saved_data",
            "obs_quat_order": "xyzw for robomimic keys; *_wxyz aliases preserve source order",
            "object_obs": "cheezit_pos(3), cheezit_quat_xyzw(4), cheezit_pos_minus_eef_pos(3)",
            "action": "dpos_world(3), drotvec_world(3), next_gripper_binary(1)",
            "rotation_delta": "q_delta = q_next * conjugate(q_current); q_next = q_delta * q_current",
        },
    }

    with h5py.File(output, "w") as h5:
        data_group = h5.create_group("data")
        data_group.attrs["env_args"] = json.dumps(env_meta, indent=2)

        for demo_idx, path in enumerate(episode_paths):
            episode = _load_episode(path)
            n_steps = _episode_length(episode)
            if n_steps < min_length:
                skipped.append({"path": str(path), "num_samples": int(n_steps), "reason": "too_short"})
                continue

            arrays = _compact_episode_arrays(episode, gripper_mid=gripper_mid)
            demo_key = f"demo_{len(demo_records)}"
            demo = data_group.create_group(demo_key)
            demo.attrs["num_samples"] = int(n_steps)
            demo.attrs["source_path"] = str(path)
            demo.attrs["source_session"] = path.parent.name
            demo.attrs["source_episode"] = path.stem
            demo.attrs["behavior"] = path.parent.name
            for name, value in arrays.items():
                _write_dataset(demo, name, value)

            demo_records.append(
                {
                    "demo_key": demo_key,
                    "source_path": str(path),
                    "session": path.parent.name,
                    "source_episode": path.stem,
                    "num_samples": int(n_steps),
                    "action_dim": int(arrays["actions"].shape[-1]),
                }
            )
            total_samples += int(n_steps)

        data_group.attrs["total"] = int(total_samples)

        train_keys, valid_keys = _stratified_train_valid_masks(demo_records, val_ratio=val_ratio, seed=seed)
        mask = h5.create_group("mask")
        mask.create_dataset("train", data=np.asarray([key.encode("utf-8") for key in train_keys]))
        mask.create_dataset("valid", data=np.asarray([key.encode("utf-8") for key in valid_keys]))

        metadata = {
            "source_dirs": [str(Path(path).expanduser()) for path in source_dirs],
            "output": str(output),
            "num_demos": len(demo_records),
            "total_samples": total_samples,
            "val_ratio": val_ratio,
            "seed": seed,
            "train_demos": len(train_keys),
            "valid_demos": len(valid_keys),
            "skipped": skipped,
            "obs_keys_for_dp": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            "obs_keys_for_real_world_dp_config": [
                "eef_pos",
                "eef_rot6d",
                "gripper_binary",
                "cheezit_pos",
                "cheezit_rot6d",
            ],
            "obs_keys_for_world_model": list(DEFAULT_STATE_KEYS),
            "action_key": DEFAULT_ACTION_KEY,
            "action_dim": int(demo_records[0]["action_dim"]) if demo_records else None,
            "action_representation": [
                "dx_world",
                "dy_world",
                "dz_world",
                "drotvec_x_world",
                "drotvec_y_world",
                "drotvec_z_world",
                "next_gripper_binary",
            ],
            "action_note": (
                "actions are synthesized from measured consecutive EEF poses so they reconstruct the next state; "
                "the original logged 3D position-delta command is preserved in actions_logged_xyz"
            ),
            "rotation_delta_convention": "q_delta = q_next * conjugate(q_current); q_next = q_delta * q_current",
            "gripper_binary_midpoint": gripper_mid,
            "gripper_raw_min": gripper_min,
            "gripper_raw_max": gripper_max,
            "demos": demo_records,
        }
        metadata_group = h5.create_group("metadata")
        metadata_group.create_dataset("json", data=json.dumps(metadata, indent=2).encode("utf-8"))

    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dirs", nargs="+", type=Path, default=list(DEFAULT_SOURCE_DIRS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = convert_saved_episodes_to_hdf5(
        source_dirs=args.source_dirs,
        output=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        min_length=args.min_length,
    )
    print(f"Wrote {metadata['num_demos']} demos / {metadata['total_samples']} samples to {metadata['output']}")
    print(f"Train demos: {metadata['train_demos']} | Valid demos: {metadata['valid_demos']}")
    print(f"Action dim: {metadata['action_dim']}")


if __name__ == "__main__":
    main()
