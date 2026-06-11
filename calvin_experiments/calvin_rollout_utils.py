"""DynaGuide-style CALVIN rollout reset and behavior labeling helpers.

These utilities intentionally work on CALVIN low-dimensional `scene_obs` and
`robot_obs` arrays, so they can be reused for base-policy rollouts, STL world
model labeling, and post-hoc trace analysis without depending on DynaGuide.
"""

from __future__ import annotations

import json
import gc
import os
import random
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


#####
# low-dimensional CALVIN state constants
#####

SCENE_INDEX = {
    "sliding_door": 0,
    "drawer": 1,
    "button": 2,
    "switch": 3,
    "lightbulb": 4,
    "green_light": 5,
}

ADJUSTABLES = ("sliding_door", "drawer", "switch", "green_light")
ADJUSTABLE_INDEX = (0, 1, 3, 5)
ADJUSTABLE_LIMITS = ((0.0, 0.27), (0.0, 0.16), (0.0, 0.09), (0.0, 1.0))
ADJUSTABLE_BEHAVIORS = (
    ("door_right", "door_left"),
    ("drawer_close", "drawer_open"),
    ("switch_off", "switch_on"),
    ("button_off", "button_on"),
)

BLOCK_POS_SLICES = {
    "red": slice(6, 9),
    "blue": slice(12, 15),
    "pink": slice(18, 21),
}
BLOCK_POSE_SLICES = {
    "block_red": (6, 9, 9, 12),
    "block_blue": (12, 15, 15, 18),
    "block_pink": (18, 21, 21, 24),
}


@dataclass(frozen=True)
class BehaviorEvent:
    """First behavior detected in a rollout relative to its start state."""

    name: str
    step: int


#####
# notebook setup and small formatting helpers
#####

def find_repo_root(start: Path | str | None = None) -> Path:
    """Find this guided-diffusion checkout from a notebook or script path."""

    start_path = Path.cwd() if start is None else Path(start)
    for path in (start_path, *start_path.parents):
        if (path / "calvin_experiments" / "calvin_rollout_utils.py").exists():
            return path
    raise FileNotFoundError(f"Could not find guided-diffusion repo root from {start_path}")


def add_calvin_repo_paths(repo_root: Path) -> None:
    """Add the local repo, robomimic, CALVIN env, and experiment paths to sys.path."""

    for path in [
        repo_root,
        repo_root / "robomimic",
        repo_root / "calvin" / "calvin_env",
        repo_root / "calvin_experiments",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def seed_everything(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and torch if torch is already imported."""

    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def to_numpy(value: Any) -> np.ndarray:
    """Detach torch tensors and otherwise coerce values to NumPy arrays."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def format_onehot(values: Sequence[int]) -> str:
    return "[" + " ".join(f"{int(v):1d}" for v in values) + "]"


def format_probs(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{float(v):6.3f}" for v in values) + "]"


def load_json_config(path: Path | str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


@contextmanager
def suppress_native_output(enabled: bool = True):
    """Suppress Python and native C/C++ writes to stdout/stderr."""

    if not enabled:
        yield
        return

    stdout_fd = stderr_fd = devnull_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if stdout_fd is not None:
            os.dup2(stdout_fd, 1)
            os.close(stdout_fd)
        if stderr_fd is not None:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


#####
# rollout folder naming helpers
#####

def readable_timestamp() -> str:
    """Timestamp suffix for run folders, with readable date/time separators."""

    return time.strftime("%d-%m-%y__%H-%M-%S")


def policy_epoch_from_checkpoint(checkpoint_path: Path | str) -> str:
    """Extract `epoch280` from checkpoint names like `model_epoch_280.pth`."""

    match = re.search(r"epoch[_-]?(\d+)", Path(checkpoint_path).stem)
    return f"epoch{match.group(1)}" if match else "epoch_unknown"


def run_folder_name(prefix: str, *parts: object, timestamp: Optional[str] = None) -> str:
    """Build readable run folder names with config first and date/time last."""

    clean_parts = [str(part).strip("_") for part in (prefix, *parts) if part is not None and str(part) != ""]
    if timestamp is None:
        timestamp = readable_timestamp()
    return "_".join([*clean_parts, timestamp])


#####
# model and checkpoint loading
#####

def resolve_checkpoint_path(checkpoint_path: Path | str, repo_root: Path | None = None) -> Path:
    """Resolve a policy checkpoint file or newest .pth inside a models directory."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        if repo_root is None:
            repo_root = find_repo_root()
        checkpoint_path = repo_root / checkpoint_path
    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("*.pth"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoints found in {checkpoint_path}")
        checkpoint_path = candidates[-1]
        print(f"Using newest checkpoint in models folder: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def automaton_model_for_eval(model_or_run_path: Path | str, device):
    """Load an AutomatonMLP checkpoint plus normalization stats for evaluation."""

    import json
    import torch

    from calvin_experiments.label_calvin_world_model import LABEL_NAMES
    from calvin_experiments.train_automaton_world_model import AutomatonMLP

    model_or_run_path = Path(model_or_run_path)
    ckpt_path = model_or_run_path / "best_model.pt" if model_or_run_path.is_dir() else model_or_run_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Automaton checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = dict(ckpt.get("model_config", {}))
    if not model_config:
        state_dict = ckpt["model_state_dict"]
        model_config = {
            "state_dim": int(state_dict["state_enc.0.weight"].shape[1]),
            "label_dim": int(state_dict["label_enc.0.weight"].shape[1]),
            "action_chunk_dim": int(state_dict["action_enc.0.weight"].shape[1]),
            "state_hidden": int(state_dict["state_enc.0.weight"].shape[0]),
            "label_hidden": int(state_dict["label_enc.0.weight"].shape[0]),
            "action_hidden": int(state_dict["action_enc.0.weight"].shape[0]),
            "head_hidden": int(state_dict["head.0.weight"].shape[0]),
            "dropout": 0.0,
        }

    stats = ckpt.get("normalization_stats")
    if stats is None:
        stats_path = ckpt_path.parent / "normalization_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"No normalization stats in checkpoint or at {stats_path}")
        z = np.load(stats_path)
        stats = {key: z[key] for key in z.files}
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}

    run_config = {}
    for metadata_name in ["train_config.json", "data_provenance.json"]:
        metadata_path = ckpt_path.parent / metadata_name
        if metadata_path.exists():
            with open(metadata_path) as f:
                run_config.update(json.load(f))

    training_config = ckpt.get("training_config", {})
    label_names = (
        ckpt.get("label_names")
        or training_config.get("label_names")
        or run_config.get("label_names")
    )
    if label_names is None:
        label_dim = int(model_config["label_dim"])
        if label_dim == 3:
            label_names = ["switch_on", "button_on", "drawer_open"]
        elif label_dim == len(LABEL_NAMES):
            label_names = list(LABEL_NAMES)
        else:
            raise ValueError(f"Could not infer label names for automaton label_dim={label_dim}")
    label_names = list(label_names)
    if len(label_names) != int(model_config["label_dim"]):
        raise ValueError(
            f"Checkpoint label_names has length {len(label_names)}, "
            f"but model label_dim is {model_config['label_dim']}"
        )

    label_thresholds = (
        ckpt.get("label_thresholds")
        or training_config.get("label_thresholds")
        or run_config.get("label_thresholds")
    )

    model = AutomatonMLP(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "ckpt_path": str(ckpt_path),
        "run_dir": str(ckpt_path.parent),
        "model_config": model_config,
        "state_keys": training_config.get("state_keys") or run_config.get("state_keys"),
        "action_key": training_config.get("action_key") or run_config.get("action_key"),
        "target_rule": training_config.get("target_rule") or run_config.get("target_rule"),
        "label_names": label_names,
        "label_thresholds": label_thresholds,
        "target_index_mapping": {idx: name for idx, name in enumerate(label_names)},
    }
    return model, stats, meta


#####
# environment reset and scene initialization
#####

def get_calvin_unwrapped_env(env):
    """Return the underlying CALVIN gym env through robomimic wrappers."""

    current = env
    while hasattr(current, "env"):
        child = current.env
        if hasattr(child, "unwrapped"):
            return child.unwrapped
        current = child
    raise AttributeError("Could not find underlying CALVIN env")


def is_env_connected(env) -> bool:
    try:
        calvin_env = get_calvin_unwrapped_env(env)
        info = calvin_env.p.getConnectionInfo(physicsClientId=calvin_env.cid)
        return bool(info.get("isConnected", False))
    except Exception:
        return False


def close_env_quietly(env) -> None:
    """Close a wrapped CALVIN env if it is still connected."""

    if env is None:
        return
    try:
        if is_env_connected(env):
            with suppress_native_output():
                get_calvin_unwrapped_env(env).close()
                gc.collect()
    except Exception:
        pass


def drop_env_quietly(namespace: Dict[str, Any], name: str = "env") -> None:
    """Close and clear an env reference while suppressing EGL/PyBullet teardown chatter."""

    env = namespace.get(name)
    with suppress_native_output():
        close_env_quietly(env)
        namespace[name] = None
        del env
        gc.collect()


def load_fresh_env_from_checkpoint(
    ckpt_dict: Dict[str, Any],
    seed: Optional[int] = None,
    existing_env: Any = None,
    render: bool = False,
    render_offscreen: bool = True,
    verbose: bool = False,
    suppress_output: bool = True,
):
    """Create a fresh CALVIN env and return it with a copy of its base state.

    Fresh envs avoid hidden PyBullet / CALVIN state leaking across repeated
    rollouts. Output suppression keeps notebook sweeps readable.
    """

    import robomimic.utils.file_utils as FileUtils

    close_env_quietly(existing_env)
    seed_everything(seed)

    def _load():
        return FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict,
            render=render,
            render_offscreen=render_offscreen,
            verbose=verbose,
        )

    if suppress_output:
        with suppress_native_output():
            env, _ = _load()
    else:
        env, _ = _load()

    if not is_env_connected(env):
        raise RuntimeError("Fresh CALVIN env is disconnected immediately after env_from_checkpoint.")

    base_env_state = {key: np.asarray(value, dtype=np.float32).copy() for key, value in env.get_state().items()}
    return env, base_env_state


def _calvin_scene_from_wrapped_env(env):
    current = env
    for _ in range(8):
        scene = getattr(current, "scene", None)
        if scene is not None and hasattr(scene, "buttons"):
            return scene
        current = getattr(current, "env", None)
        if current is None:
            break
    return None


def sync_button_logic_to_lights(env) -> None:
    """Keep CALVIN's hidden edge-triggered button state consistent with scene_obs lights."""

    scene = _calvin_scene_from_wrapped_env(env)
    if scene is None:
        return
    for button in getattr(scene, "buttons", []):
        light = getattr(button, "light", None)
        if light is None or not hasattr(button, "state"):
            continue
        state_cls = button.state.__class__
        button.state = state_cls.ON if float(light.get_state()) > 0.5 else state_cls.OFF
        if hasattr(button, "prev_is_pressed"):
            button.prev_is_pressed = button._is_pressed


def reset_env_to_scene_robot(env, scene: Sequence[float], robot: Sequence[float]):
    """Reset CALVIN to an explicit scene / robot state while keeping wrappers in sync."""

    state = {
        "scene": np.asarray(scene, dtype=np.float32).copy(),
        "robot": np.asarray(robot, dtype=np.float32).copy(),
    }
    robomimic_env = env
    frame_stack_env = env if hasattr(env, "_get_initial_obs_history") else None
    if frame_stack_env is not None:
        robomimic_env = frame_stack_env.env
    gym_env = getattr(robomimic_env, "env", None)
    if gym_env is not None and hasattr(gym_env, "reset") and hasattr(robomimic_env, "get_observation"):
        raw_obs = gym_env.reset(scene_obs=state["scene"], robot_obs=state["robot"])
        sync_button_logic_to_lights(env)
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        robomimic_env._current_obs = raw_obs
        robomimic_env._current_reward = None
        robomimic_env._current_done = None
        obs = robomimic_env.get_observation(raw_obs)
        if frame_stack_env is not None:
            frame_stack_env.timestep = 0
            frame_stack_env.update_obs(obs, reset=True)
            frame_stack_env.obs_history = frame_stack_env._get_initial_obs_history(init_obs=obs)
            return frame_stack_env._get_stacked_obs_from_history()
        return obs
    obs = env.reset_to(state)
    sync_button_logic_to_lights(env)
    return obs


def render_visual_camera(env, video_cfg: Dict[str, Any], suppress_output: bool = True) -> np.ndarray:
    """Render the notebook visualization camera without changing policy observations."""

    if suppress_output:
        with suppress_native_output():
            return render_visual_camera(env, video_cfg, suppress_output=False)

    if video_cfg.get("perspective") == "third_person":
        return env.render(
            mode="rgb_array",
            height=int(video_cfg["height"]),
            width=int(video_cfg["width"]),
        )

    import pybullet as p

    calvin_env = get_calvin_unwrapped_env(env)
    width = int(video_cfg["width"])
    height = int(video_cfg["height"])
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=video_cfg["look_from"],
        cameraTargetPosition=video_cfg["look_at"],
        cameraUpVector=video_cfg["up_vector"],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=float(video_cfg["fov"]),
        aspect=width / height,
        nearVal=float(video_cfg["nearval"]),
        farVal=float(video_cfg["farval"]),
    )
    image = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        physicsClientId=calvin_env.cid,
    )
    return np.reshape(image[2], (height, width, 4))[:, :, :3].astype(np.uint8)


def apply_block_scene_overrides(scene: np.ndarray, blocks_cfg: Dict[str, Any]) -> list[str]:
    """Set block poses through scene_obs before reset_to, preserving env structure."""

    applied = []
    for name, pose_cfg in blocks_cfg.get("poses", {}).items():
        if name not in BLOCK_POSE_SLICES:
            print(f"Unknown block name in blocks.poses: {name}")
            continue
        pos_start, pos_end, rot_start, rot_end = BLOCK_POSE_SLICES[name]
        if isinstance(pose_cfg, dict):
            position = pose_cfg.get("position")
            rotation = pose_cfg.get("rotation", [0.0, 0.0, 0.0])
        else:
            position = pose_cfg
            rotation = [0.0, 0.0, 0.0]
        scene[pos_start:pos_end] = np.asarray(position, dtype=np.float32)
        scene[rot_start:rot_end] = np.asarray(rotation, dtype=np.float32)
        applied.append(name)
    return applied


def fixed_scene_robot_from_config(
    base_env_state: Dict[str, np.ndarray],
    scene_config_path: Path | str,
    robot_state_override: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create fixed scene and robot reset arrays from a CALVIN scene config."""

    scene_cfg = load_json_config(scene_config_path)
    fixed_scene = np.asarray(base_env_state["scene"], dtype=np.float32).copy()
    fixed_robot = np.asarray(base_env_state["robot"], dtype=np.float32).copy()
    if robot_state_override is not None:
        robot_override = np.asarray(robot_state_override, dtype=np.float32)
        if robot_override.shape != fixed_robot.shape:
            raise ValueError(f"ROBOT_STATE_OVERRIDE must have shape {fixed_robot.shape}, got {robot_override.shape}")
        fixed_robot = robot_override.copy()
    for name, value in scene_cfg.get("env_setup", {}).items():
        if name not in SCENE_INDEX:
            raise KeyError(f"Unknown env_setup key: {name}")
        fixed_scene[SCENE_INDEX[name]] = float(value)
    apply_block_scene_overrides(fixed_scene, scene_cfg.get("blocks", {}))
    return fixed_scene, fixed_robot, scene_cfg


#####
# scene snapshot helpers for top-down plotting
#####

def _pose_from_joint_object(obj):
    link_state = obj.p.getLinkState(obj.uid, obj.joint_index, physicsClientId=obj.cid)
    return np.asarray(link_state[4], dtype=np.float32), np.asarray(link_state[5], dtype=np.float32)


def _pose_from_link_object(obj, link_index):
    link_state = obj.p.getLinkState(obj.uid, link_index, physicsClientId=obj.cid)
    return np.asarray(link_state[4], dtype=np.float32), np.asarray(link_state[5], dtype=np.float32)


def _pose_from_body_object(obj):
    position, orientation = obj.p.getBasePositionAndOrientation(obj.uid, physicsClientId=obj.cid)
    return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)


def _aabb_from_object(obj, link_index=None):
    if link_index is None:
        lower, upper = obj.p.getAABB(obj.uid, physicsClientId=obj.cid)
    else:
        lower, upper = obj.p.getAABB(obj.uid, link_index, physicsClientId=obj.cid)
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def capture_scene_snapshot(env) -> Dict[str, Any]:
    """Capture top-down plotting geometry from the currently reset CALVIN scene."""

    scene = get_calvin_unwrapped_env(env).scene
    snapshot = {"fixed_objects": [], "controls": [], "blocks": []}

    for obj in scene.fixed_objects:
        position, orientation = _pose_from_body_object(obj)
        lower, upper = _aabb_from_object(obj)
        snapshot["fixed_objects"].append(
            {
                "name": obj.name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "aabb": [lower.tolist(), upper.tolist()],
            }
        )

    for obj in scene.buttons:
        position, orientation = _pose_from_joint_object(obj)
        lower, upper = _aabb_from_object(obj, obj.joint_index)
        snapshot["controls"].append(
            {
                "kind": "button",
                "name": obj.name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "aabb": [lower.tolist(), upper.tolist()],
                "state": float(obj.get_state()),
            }
        )

    for obj in scene.doors:
        position, orientation = _pose_from_joint_object(obj)
        lower, upper = _aabb_from_object(obj, obj.joint_index)
        kind = "drawer" if "drawer" in obj.name else "slider" if "slide" in obj.name else "door"
        snapshot["controls"].append(
            {
                "kind": kind,
                "name": obj.name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "aabb": [lower.tolist(), upper.tolist()],
                "state": float(obj.get_state()),
            }
        )

    for obj in scene.switches:
        position, orientation = _pose_from_joint_object(obj)
        lower, upper = _aabb_from_object(obj, obj.joint_index)
        snapshot["controls"].append(
            {
                "kind": "switch",
                "name": obj.name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "aabb": [lower.tolist(), upper.tolist()],
                "state": float(obj.get_state()),
            }
        )

    for obj in scene.lights:
        position, orientation = _pose_from_link_object(obj, obj.link_id)
        lower, upper = _aabb_from_object(obj, obj.link_id)
        snapshot["controls"].append(
            {
                "kind": "light",
                "name": obj.name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "aabb": [lower.tolist(), upper.tolist()],
                "state": float(obj.get_state()),
            }
        )

    for obj in scene.movable_objects:
        position, orientation = _pose_from_body_object(obj)
        lower, upper = _aabb_from_object(obj)
        snapshot["blocks"].append(
            {
                "name": obj.name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "aabb": [lower.tolist(), upper.tolist()],
            }
        )

    return snapshot


def save_scene_snapshot(snapshot: Dict[str, Any], path: Path | str) -> None:
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


#####
# top-down rollout plot helpers
#####

def _rect_from_aabb(ax, lower, upper, **kwargs):
    from matplotlib.patches import Rectangle

    width = float(upper[0] - lower[0])
    height = float(upper[1] - lower[1])
    patch = Rectangle((float(lower[0]), float(lower[1])), width, height, **kwargs)
    ax.add_patch(patch)
    return patch


def _circle_from_aabb(ax, lower, upper, **kwargs):
    from matplotlib.patches import Circle

    center = ((float(lower[0]) + float(upper[0])) / 2, (float(lower[1]) + float(upper[1])) / 2)
    radius = 0.5 * min(float(upper[0] - lower[0]), float(upper[1] - lower[1]))
    patch = Circle(center, radius=radius, **kwargs)
    ax.add_patch(patch)
    return patch


def _safety_box_from_rollout(rollout: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if rollout.get("safety_kind") != "eef_avoid_box" and rollout.get("safety_metrics", {}).get("kind") != "eef_avoid_box":
        return None
    box = rollout.get("safety_box") or rollout.get("safety_metrics", {}).get("safety_box")
    if not isinstance(box, dict):
        return None
    required = ("x_min", "x_max", "y_min", "y_max")
    if not all(key in box for key in required):
        return None
    return {key: float(box[key]) for key in required}


def scene_limits_from_snapshot(
    snapshot: Dict[str, Any],
    eef_xy: np.ndarray,
    padding: float = 0.05,
    safety_boxes: Optional[Sequence[Dict[str, float]]] = None,
):
    boxes = []
    for item in snapshot.get("controls", []):
        if "aabb" in item:
            lower = np.asarray(item["aabb"][0], dtype=np.float32)
            upper = np.asarray(item["aabb"][1], dtype=np.float32)
            boxes.append((lower, upper))
    if not boxes:
        return (-0.4, 0.4), (-0.4, 0.25)

    lower_xy = np.min([box[0][:2] for box in boxes], axis=0)
    upper_xy = np.max([box[1][:2] for box in boxes], axis=0)
    lower_xy = np.minimum(lower_xy, np.min(eef_xy[:, :2], axis=0))
    upper_xy = np.maximum(upper_xy, np.max(eef_xy[:, :2], axis=0))
    for box in safety_boxes or []:
        lower_xy = np.minimum(lower_xy, np.asarray([box["x_min"], box["y_min"]], dtype=np.float32))
        upper_xy = np.maximum(upper_xy, np.asarray([box["x_max"], box["y_max"]], dtype=np.float32))
    return (float(lower_xy[0] - padding), float(upper_xy[0] + padding)), (
        float(lower_xy[1] - padding),
        float(upper_xy[1] + padding),
    )


def draw_scene_snapshot(ax, snapshot: Dict[str, Any], eef_xy: Optional[np.ndarray] = None) -> None:
    kind_styles = {
        "button": {"facecolor": "#111111", "edgecolor": "#000000", "alpha": 0.95},
        "slider": {"facecolor": "#4c566a", "edgecolor": "#1f2937", "alpha": 0.92},
        "drawer": {"facecolor": "#7a5c4d", "edgecolor": "#2f241f", "alpha": 0.9},
        "switch": {"facecolor": "#6b7280", "edgecolor": "black", "alpha": 0.9},
        "light": {"facecolor": "#9ca3af", "edgecolor": "black", "alpha": 0.75},
    }

    for control in snapshot.get("controls", []):
        lower = np.asarray(control["aabb"][0], dtype=np.float32)
        upper = np.asarray(control["aabb"][1], dtype=np.float32)
        kind = control["kind"]
        style = kind_styles.get(kind, {"facecolor": "#adb5bd", "edgecolor": "black", "alpha": 0.8})
        patch_kwargs = {
            "facecolor": style["facecolor"],
            "edgecolor": style["edgecolor"],
            "linewidth": 1.0,
            "alpha": style["alpha"],
        }
        if kind == "button":
            _circle_from_aabb(ax, lower, upper, **patch_kwargs)
        else:
            _rect_from_aabb(ax, lower, upper, **patch_kwargs)
        label_x = float((lower[0] + upper[0]) / 2)
        label_y = float(upper[1] + 0.01)
        ax.text(label_x, label_y, control["name"].replace("base__", ""), ha="center", va="bottom", fontsize=7, color="black")

    if eef_xy is not None and len(eef_xy) > 0:
        ax.plot(eef_xy[:, 0], eef_xy[:, 1], color="#ff3b30", linewidth=2.5, label="EEF XY path")
        ax.scatter(eef_xy[0, 0], eef_xy[0, 1], c="#4b5563", s=55, edgecolors="white", linewidths=1.0, label="start")
        ax.scatter(eef_xy[-1, 0], eef_xy[-1, 1], c="#111827", s=55, edgecolors="white", linewidths=1.0, label="end")

    ax.set_aspect("equal")
    ax.grid(color="white", alpha=0.2, linewidth=0.8)


def plot_rollout_xy(
    rollouts: Sequence[Dict[str, Any]],
    scene_snapshot: Dict[str, Any],
    title: str,
    save_path: Path | str | None = None,
    display_inline: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    paths = [np.asarray(item["eef_xy"], dtype=np.float32) for item in rollouts]
    behaviors = [item["behavior"] for item in rollouts]
    safety_boxes = [box for box in (_safety_box_from_rollout(item) for item in rollouts) if box is not None]
    unique_behaviors = sorted(set(behaviors))
    cmap = plt.get_cmap("tab10")
    behavior_colors = {behavior: cmap(idx % cmap.N) for idx, behavior in enumerate(unique_behaviors)}
    all_eef_xy = np.concatenate(paths, axis=0)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#fbf7ef")
    draw_scene_snapshot(ax, scene_snapshot)
    for idx, box in enumerate(safety_boxes):
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (box["x_min"], box["y_min"]),
            box["x_max"] - box["x_min"],
            box["y_max"] - box["y_min"],
            facecolor="#ef4444",
            edgecolor="#991b1b",
            linewidth=1.2,
            linestyle="--",
            alpha=0.18,
            label="avoid box" if idx == 0 else None,
        )
        ax.add_patch(rect)
    for item, eef_xy in zip(rollouts, paths):
        color = behavior_colors[item["behavior"]]
        ax.plot(eef_xy[:, 0], eef_xy[:, 1], color=color, linewidth=2.0, alpha=0.75)
        ax.scatter(eef_xy[0, 0], eef_xy[0, 1], c=[color], s=28, edgecolors="white", linewidths=0.7, alpha=0.85)
        ax.scatter(eef_xy[-1, 0], eef_xy[-1, 1], c=[color], s=42, edgecolors="black", linewidths=0.7, alpha=0.9)

    xlim, ylim = scene_limits_from_snapshot(scene_snapshot, all_eef_xy, safety_boxes=safety_boxes)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    handles = [
        Line2D([0], [0], color=behavior_colors[behavior], lw=2.5, label=f"{behavior} ({behaviors.count(behavior)})")
        for behavior in unique_behaviors
    ]
    if safety_boxes:
        from matplotlib.patches import Patch

        handles.append(Patch(facecolor="#ef4444", edgecolor="#991b1b", alpha=0.18, linestyle="--", label="avoid box"))
    ax.legend(
        handles=handles,
        loc="upper right",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print("plot:", save_path)

    if display_inline:
        png = BytesIO()
        fig.savefig(png, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        from IPython.display import Image, display

        display(Image(data=png.getvalue()))
    else:
        plt.close(fig)


def plot_rollouts_from_trace_summaries(
    rollout_summaries: Sequence[Dict[str, Any]],
    title: Optional[str] = None,
    save_path: Path | str | None = None,
    display_inline: bool = True,
):
    """Load trace files from env_playground summaries and draw the shared XY overlay."""

    if not rollout_summaries:
        print("Run rollout generation first; it writes trace .npz files for plotting.")
        return

    first_item = rollout_summaries[0]
    scene_snapshot = load_json_config(first_item["scene_snapshot"])
    rollouts = []
    for item in rollout_summaries:
        with np.load(item["trace"]) as trace:
            rollout = {"eef_xy": trace["eef_xy"], "behavior": item["behavior"]}
            if item.get("safety_kind") is not None:
                rollout["safety_kind"] = item["safety_kind"]
            if item.get("safety_metrics") is not None:
                rollout["safety_metrics"] = item["safety_metrics"]
            if item.get("safety_box") is not None:
                rollout["safety_box"] = item["safety_box"]
            elif "safety_box" in trace:
                x_min, x_max, y_min, y_max = [float(value) for value in np.asarray(trace["safety_box"]).tolist()]
                rollout["safety_kind"] = "eef_avoid_box"
                rollout["safety_box"] = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
            rollouts.append(rollout)

    if title is None:
        title = f"Top-down rollouts | {first_item['scene_config']} | {len(rollout_summaries)} trajectories"
    if save_path is None:
        save_path = Path(first_item["trace"]).parents[1] / "rollout_scene_overlay_all.png"
    plot_rollout_xy(rollouts, scene_snapshot, title, save_path=save_path, display_inline=display_inline)


#####
# rollout artifact saving
#####

def save_rollout_artifacts(
    rollout: Dict[str, Any],
    frames: Sequence[np.ndarray],
    output_dir: Path | str,
    rollout_tag: str,
    video_cfg: Dict[str, Any],
    fps: int = 30,
) -> Dict[str, Any]:
    """Save a rollout video, trace npz, and per-rollout scene snapshot."""

    import imageio.v2 as imageio

    rollout_dir = Path(output_dir) / rollout_tag
    rollout_dir.mkdir(parents=True, exist_ok=True)
    scene_name = rollout["scene_config"]
    video_path = rollout_dir / f"{scene_name}_{rollout_tag}_{video_cfg['perspective']}.mp4"
    trace_path = rollout_dir / "rollout_trace.npz"
    scene_snapshot_path = rollout_dir / "scene_snapshot.json"

    imageio.mimsave(video_path, frames, fps=int(fps))
    trace = {
        "actions": np.asarray(rollout["actions"], dtype=np.float32),
        "rewards": np.asarray(rollout["rewards"], dtype=np.float32),
        "dones": np.asarray(rollout["dones"], dtype=bool),
        "scene_states": np.asarray(rollout["scene_states"], dtype=np.float32),
        "robot_states": np.asarray(rollout["robot_states"], dtype=np.float32),
        "eef_xy": np.asarray(rollout["eef_xy"], dtype=np.float32),
        "detected_behavior": np.asarray(rollout["behavior"]),
        "detected_behavior_step": np.asarray(rollout["behavior_step"], dtype=np.int32),
        "termination_step": np.asarray(rollout["termination_step"], dtype=np.int32),
        "termination_reason": np.asarray(rollout["termination_reason"]),
        "rollout_seed": np.asarray(-1 if rollout["seed"] is None else rollout["seed"], dtype=np.int32),
        "scene_config": np.asarray(scene_name),
    }
    if "initial_label" in rollout:
        trace["initial_label"] = np.asarray(rollout["initial_label"], dtype=np.int32)
    if "final_label" in rollout:
        trace["final_label"] = np.asarray(rollout["final_label"], dtype=np.int32)
    if "labels_over_time" in rollout:
        trace["labels_over_time"] = np.asarray(rollout["labels_over_time"], dtype=np.int32)
    if "pre_settle_label" in rollout:
        trace["pre_settle_label"] = np.asarray(rollout["pre_settle_label"], dtype=np.int32)
    if "settle_scene_states" in rollout:
        trace["settle_scene_states"] = np.asarray(rollout["settle_scene_states"], dtype=np.float32)
    if "settle_robot_states" in rollout:
        trace["settle_robot_states"] = np.asarray(rollout["settle_robot_states"], dtype=np.float32)
    if "settle_action" in rollout:
        trace["settle_action"] = np.asarray(rollout["settle_action"], dtype=np.float32)
    if "tcp_rzz" in rollout:
        trace["tcp_rzz"] = np.asarray(rollout["tcp_rzz"], dtype=np.float32)
    if "tcp_tilt_angle_deg" in rollout:
        trace["tcp_tilt_angle_deg"] = np.asarray(rollout["tcp_tilt_angle_deg"], dtype=np.float32)
    if rollout.get("safety_box") is not None:
        box = rollout["safety_box"]
        trace["safety_box"] = np.asarray(
            [box["x_min"], box["x_max"], box["y_min"], box["y_max"]],
            dtype=np.float32,
        )
    if rollout.get("rzz_spec") is not None:
        rzz_spec = rollout["rzz_spec"]
        trace["rzz_spec"] = np.asarray(
            [
                rzz_spec["angle_deg"],
                rzz_spec["axis_sign"],
                rzz_spec["tolerance"],
                rzz_spec["smooth_min_tau"],
            ],
            dtype=np.float32,
        )
    if "warmup_actions" in rollout:
        trace["warmup_actions"] = np.asarray(rollout["warmup_actions"], dtype=np.float32)
    if "warmup_robot_states" in rollout:
        trace["warmup_robot_states"] = np.asarray(rollout["warmup_robot_states"], dtype=np.float32)
    np.savez_compressed(trace_path, **trace)
    save_scene_snapshot(rollout["scene_snapshot"], scene_snapshot_path)

    rollout["video"] = video_path
    rollout["trace"] = trace_path
    rollout["scene_snapshot_path"] = scene_snapshot_path
    rollout["rollout_dir"] = rollout_dir
    return rollout


#####
# initial state sampling and rollout classification
#####

def generate_reset_state(sim_hold: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, list[bool]]:
    """Match DynaGuide's randomized articulated-object reset state.

    `sim_hold` can pin a subset of the adjustable states. Unspecified states
    are sampled at their binary endpoints, which is what DynaGuide uses for
    behavior-direction tracking.
    """

    sim_hold = sim_hold or {}
    state = np.zeros((24,), dtype=np.float32)
    binary_list = []
    for adjustable, idx, limits in zip(ADJUSTABLES, ADJUSTABLE_INDEX, ADJUSTABLE_LIMITS):
        if adjustable in sim_hold:
            state[idx] = float(sim_hold[adjustable])
            binary_list.append(state[idx] > (limits[1] - limits[0]) / 2)
        else:
            select = random.random() > 0.5
            state[idx] = limits[1] if select else limits[0]
            binary_list.append(select)
    return state, binary_list


def articulated_binaries_from_start_state(start_state: Sequence[float]) -> list[bool]:
    """Return DynaGuide-style direction flags for adjustable scene states."""

    start_state = np.asarray(start_state)
    binary_list = []
    for idx, limits in zip(ADJUSTABLE_INDEX, ADJUSTABLE_LIMITS):
        midpoint = (limits[1] - limits[0]) / 2
        binary_list.append(start_state[idx] > midpoint)
    return binary_list


def classify_behavior(
    start_state: Sequence[float],
    state: Sequence[float],
    robot_pos: Optional[Sequence[float]] = None,
    binaries: Optional[Iterable[bool]] = None,
    for_display: bool = False,
) -> str:
    """Classify behavior with the same criteria used for rollout stopping."""

    start_state = np.asarray(start_state)
    state = np.asarray(state)
    if binaries is None:
        binaries = articulated_binaries_from_start_state(start_state)

    for binary, idx, limits, names in zip(binaries, ADJUSTABLE_INDEX, ADJUSTABLE_LIMITS, ADJUSTABLE_BEHAVIORS):
        span = limits[1] - limits[0]
        midpoint = limits[0] + span / 2
        low_threshold = limits[0] + 0.25 * span if for_display else midpoint
        high_threshold = limits[0] + 0.75 * span if for_display else midpoint
        high_to_low_name, low_to_high_name = names

        if binary and state[idx] < low_threshold:
            return high_to_low_name
        if not binary and state[idx] > high_threshold:
            return low_to_high_name

    if robot_pos is not None:
        robot_pos = np.asarray(robot_pos)
        block_threshold = 0.03 if for_display else 0.001
        for color, pos_slice in BLOCK_POS_SLICES.items():
            if np.linalg.norm(robot_pos - state[pos_slice]) < 0.06:
                xy_delta = state[pos_slice.start : pos_slice.start + 2] - start_state[pos_slice.start : pos_slice.start + 2]
                if np.linalg.norm(xy_delta) > block_threshold:
                    return f"{color}_displace"

    return "other"


def check_state_difference(
    start_state: Sequence[float],
    state: Sequence[float],
    robot_pos: Sequence[float],
    binaries: Optional[Iterable[bool]] = None,
    for_display: bool = False,
) -> bool:
    """Return True once `classify_behavior` sees a stopping event."""

    return classify_behavior(start_state, state, robot_pos, binaries, for_display) != "other"


def first_behavior_event(
    scene_states: np.ndarray,
    robot_states: np.ndarray,
    for_display: bool = False,
) -> Optional[BehaviorEvent]:
    """Find the first DynaGuide-style behavior event in a recorded trace."""

    if len(scene_states) == 0:
        return None
    start_state = np.asarray(scene_states[0])
    binaries = articulated_binaries_from_start_state(start_state)
    for step in range(1, len(scene_states)):
        if check_state_difference(start_state, scene_states[step], robot_states[step, :3], binaries, for_display):
            name = classify_behavior(start_state, scene_states[step], robot_states[step, :3], binaries, for_display)
            return BehaviorEvent(name, step)
    return None


def stl_atomic_labels(scene_state: Sequence[float]) -> Dict[str, bool]:
    """Current-state predicates useful for STL / LTL label world models."""

    state = np.asarray(scene_state)
    return {
        "drawer_open": bool(state[SCENE_INDEX["drawer"]] > 0.08),
        "drawer_closed": bool(state[SCENE_INDEX["drawer"]] <= 0.08),
        "button_on": bool(state[SCENE_INDEX["green_light"]] > 0.5),
        "button_off": bool(state[SCENE_INDEX["green_light"]] <= 0.5),
        "switch_on": bool(state[SCENE_INDEX["switch"]] > 0.045),
        "switch_off": bool(state[SCENE_INDEX["switch"]] <= 0.045),
    }
