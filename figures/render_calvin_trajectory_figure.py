from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_DIR = REPO_ROOT / (
    "outputs/calvin/paper_stls/F_button_then_F_drawer/20260506_133900/"
    "chain_button_on_then_drawer_open_then_switch_on_then_button_pressed_then_door_left_004_seed_004"
)
ENV_CHECKPOINT = REPO_ROOT / "outputs/calvin/base_policy/calvin_D_base_dp/20260501015147/last.pth"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/paper_plots/5-step-figure"
TEXTURE_DIR = REPO_ROOT / "assets/calvin_render_textures"
LOCAL_BLENDER = REPO_ROOT / "tools/blender-4.2.0-linux-x64/blender"
TASK_COLOR_HEX = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#AA3377"]
SAFETY_SWITCH_DIR = REPO_ROOT / "outputs/calvin/paper_stls/F_switch_G_safety/20260506_125354"
FLOWER_ROLLOUT_DIR = REPO_ROOT / (
    "outputs/calvin/baselines/flower/our_env_rollouts/"
    "flower_vla_scene_blocks_hidden_rollouts1_horizon300_seed0_07-05-26__00-15-13/rollout_000"
)
ORDERED_STAGE_DIR = REPO_ROOT / (
    "outputs/calvin/paper_stls/F_drawer_after_button_switch/ordered_stage_20260507_191217/"
    "ordered_button_on_and_switch_on_then_drawer_open_start02_x0p10_ym0p20_seed_000"
)
BASE_DIVERSE_BATCHES = [
    (
        "button",
        REPO_ROOT
        / "outputs/calvin/guidance_test/sample_rank_batch_policy_epoch280_scene_blocks_hidden_rollouts20_target4_candidates16_horizon200_05-05-26__21-05-58",
        0,
        False,
    ),
    (
        "drawer",
        REPO_ROOT
        / "outputs/calvin/guidance_test/sample_rank_batch_policy_epoch280_scene_blocks_hidden_rollouts20_target2_candidates16_horizon100_05-05-26__22-49-10",
        1,
        False,
    ),
    (
        "switch",
        REPO_ROOT
        / "outputs/calvin/guidance_test/sample_rank_batch_policy_epoch280_scene_blocks_hidden_rollouts20_target0_candidates16_horizon100_05-05-26__22-49-48",
        2,
        False,
    ),
    (
        "door_left",
        REPO_ROOT
        / "outputs/calvin/guidance_test/sample_rank_batch_policy_epoch280_scene_blocks_hidden_rollouts10_target7_candidates16_horizon500_05-05-26__23-25-03",
        4,
        True,
    ),
]


CAMERA_PRESETS = {
    "dynaguide_side": {
        "width": 2400,
        "height": 1800,
        "fov": 46.0,
        "nearval": 0.01,
        "farval": 10.0,
        "look_from": [-1.03, -0.84, 0.98],
        "look_at": [-0.03, -0.045, 0.50],
        "up_vector": [0.0, 0.0, 1.0],
    },
    "dynaguide_close": {
        "width": 2400,
        "height": 1800,
        "fov": 43.0,
        "nearval": 0.01,
        "farval": 10.0,
        "look_from": [-0.86, -0.74, 0.90],
        "look_at": [-0.02, -0.075, 0.49],
        "up_vector": [0.0, 0.0, 1.0],
    },
    "topdown": {
        "width": 2700,
        "height": 1875,
        "fov": 34.0,
        "nearval": 0.01,
        "farval": 10.0,
        "look_from": [0.02, -0.11, 2.35],
        "look_at": [0.02, -0.11, 0.45],
        "up_vector": [0.0, 1.0, 0.0],
    },
}


GEOM_SPHERE = 2
GEOM_BOX = 3
GEOM_CYLINDER = 4
GEOM_MESH = 5

bpy = None
Quaternion = None
Vector = None


def argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def is_blender_render_mode() -> bool:
    return "--blender-render" in argv_after_double_dash()


def parse_vec3(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(values[0]), float(values[1]), float(values[2])]


def hex_to_rgba(hex_color: str) -> list[float]:
    hex_color = hex_color.strip().lstrip("#")
    def srgb_to_linear(value: float) -> float:
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    return [srgb_to_linear(int(hex_color[i : i + 2], 16) / 255.0) for i in (0, 2, 4)] + [1.0]


def color_ramp(index: int, count: int) -> list[float]:
    start = [0.36, 0.08, 0.95, 1.0]
    end = [1.00, 0.48, 0.05, 1.0]
    t = 0.0 if count <= 1 else index / float(count - 1)
    eased = t * t * (3.0 - 2.0 * t)
    return [(1.0 - eased) * start[i] + eased * end[i] for i in range(4)]


def task_color(index: int, count: int) -> list[float]:
    if index < len(TASK_COLOR_HEX):
        return hex_to_rgba(TASK_COLOR_HEX[index])
    return color_ramp(index, count)


def next_versioned_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir.name
    for version in range(1, 10000):
        candidate = output_dir / f"{prefix}_v{version}.png"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No free {prefix}_vN.png slot found in {output_dir}")


def default_blender_path() -> str:
    return str(LOCAL_BLENDER if LOCAL_BLENDER.exists() else "blender")


def parse_driver_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a CALVIN rollout as a publication-style Blender figure. "
            "Run this with pixi/python, not directly with blender."
        )
    )
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument(
        "--figure-preset",
        choices=["single", "safety-switch", "flower", "base-diverse", "ordered-stage"],
        default="single",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Optional exact output PNG path. Must not already exist.")
    parser.add_argument("--blender", default=default_blender_path())
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--view", choices=sorted(CAMERA_PRESETS), default="dynaguide_side")
    parser.add_argument("--camera-look-from", nargs=3, type=float, default=None)
    parser.add_argument("--camera-look-at", nargs=3, type=float, default=None)
    parser.add_argument("--camera-location", nargs=3, type=float, default=None)
    parser.add_argument("--camera-rotation-deg", nargs=3, type=float, default=None)
    parser.add_argument("--camera-fov", type=float, default=None)
    parser.add_argument("--resolution", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--debug-frames", action="store_true")
    parser.add_argument("--export-only", action="store_true")

    parser.add_argument("--key-light-energy", type=float, default=185.0)
    parser.add_argument("--bulb-light-energy", type=float, default=8.0)
    parser.add_argument("--led-light-energy", type=float, default=3.0)
    parser.add_argument("--world-strength", type=float, default=0.38)
    parser.add_argument("--camera-bg-strength", type=float, default=10.0)
    parser.add_argument("--exposure", type=float, default=-0.35)

    parser.add_argument("--wood-roughness", type=float, default=0.72)
    parser.add_argument("--wood-saturation", type=float, default=1.24)
    parser.add_argument("--wood-value", type=float, default=0.55)
    parser.add_argument("--robot-value", type=float, default=0.82)

    parser.add_argument("--trajectory-radius", type=float, default=0.0036)
    parser.add_argument("--trajectory-halo-radius", type=float, default=0.0046)
    parser.add_argument("--trajectory-halo-alpha", type=float, default=0.0)
    parser.add_argument("--trajectory-emission", type=float, default=0.35)
    parser.add_argument("--event-emission", type=float, default=0.7)
    parser.add_argument("--transition-steps", type=int, default=0)
    parser.add_argument("--event-radius", type=float, default=0.0072)
    return parser.parse_args()


def build_camera_config(args: argparse.Namespace) -> dict:
    camera = dict(CAMERA_PRESETS[args.view])
    if args.camera_location is not None:
        camera["look_from"] = parse_vec3(args.camera_location)
    elif args.camera_look_from is not None:
        camera["look_from"] = parse_vec3(args.camera_look_from)
    if args.camera_look_at is not None:
        camera["look_at"] = parse_vec3(args.camera_look_at)
    if args.camera_rotation_deg is not None:
        camera["rotation_euler_deg"] = parse_vec3(args.camera_rotation_deg)
    if args.camera_fov is not None:
        camera["fov"] = float(args.camera_fov)
    if args.resolution is not None:
        camera["width"] = int(args.resolution[0])
        camera["height"] = int(args.resolution[1])
    return camera


def add_repo_import_paths() -> None:
    for path in [
        REPO_ROOT,
        REPO_ROOT / "robomimic",
        REPO_ROOT / "calvin" / "calvin_env",
        REPO_ROOT / "calvin_experiments",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def body_names(env, p, cru) -> dict[int, str]:
    calvin_env = cru.get_calvin_unwrapped_env(env)
    names = {}
    for body_index in range(p.getNumBodies(physicsClientId=calvin_env.cid)):
        body_id = p.getBodyUniqueId(body_index, physicsClientId=calvin_env.cid)
        info = p.getBodyInfo(body_id, physicsClientId=calvin_env.cid)
        names[int(body_id)] = info[1].decode("utf-8", "replace") or info[0].decode("utf-8", "replace")
    return names


def link_names(env, body_id: int, p, cru) -> dict[int, str]:
    calvin_env = cru.get_calvin_unwrapped_env(env)
    info = p.getBodyInfo(body_id, physicsClientId=calvin_env.cid)
    names = {-1: info[0].decode("utf-8", "replace") or "base_link"}
    for joint_index in range(p.getNumJoints(body_id, physicsClientId=calvin_env.cid)):
        joint_info = p.getJointInfo(body_id, joint_index, physicsClientId=calvin_env.cid)
        names[joint_index] = joint_info[12].decode("utf-8", "replace")
    return names


def link_world_transform(env, body_id: int, link_index: int, p, cru) -> tuple[tuple, tuple]:
    calvin_env = cru.get_calvin_unwrapped_env(env)
    if link_index == -1:
        base_pos, base_orn = p.getBasePositionAndOrientation(body_id, physicsClientId=calvin_env.cid)
        dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=calvin_env.cid)
        inertial_pos, inertial_orn = dynamics[3], dynamics[4]
        inv_inertial_pos, inv_inertial_orn = p.invertTransform(inertial_pos, inertial_orn)
        return p.multiplyTransforms(base_pos, base_orn, inv_inertial_pos, inv_inertial_orn)
    state = p.getLinkState(body_id, link_index, computeForwardKinematics=True, physicsClientId=calvin_env.cid)
    return state[4], state[5]


def export_visual_shapes(env, p, cru) -> list[dict]:
    calvin_env = cru.get_calvin_unwrapped_env(env)
    shapes = []
    for body_id, body_name in body_names(env, p, cru).items():
        links = link_names(env, body_id, p, cru)
        for visual_index, visual in enumerate(p.getVisualShapeData(body_id, physicsClientId=calvin_env.cid)):
            _, link_index, geom_type, dimensions, filename, local_pos, local_orn, rgba = visual
            mesh_path = filename.decode("utf-8", "replace") if isinstance(filename, bytes) else str(filename)
            if mesh_path and (geom_type != p.GEOM_MESH or not Path(mesh_path).exists()):
                continue
            link_pos, link_orn = link_world_transform(env, body_id, link_index, p, cru)
            world_pos, world_orn = p.multiplyTransforms(link_pos, link_orn, local_pos, local_orn)
            shapes.append(
                {
                    "body_id": int(body_id),
                    "body_name": body_name,
                    "link_index": int(link_index),
                    "link_name": links.get(link_index, f"link_{link_index}"),
                    "visual_index": int(visual_index),
                    "geometry_type": int(geom_type),
                    "mesh_path": str(Path(mesh_path).resolve()) if mesh_path else None,
                    "link_world_position": [float(x) for x in link_pos],
                    "link_world_orientation_xyzw": [float(x) for x in link_orn],
                    "local_visual_position": [float(x) for x in local_pos],
                    "local_visual_orientation_xyzw": [float(x) for x in local_orn],
                    "position": [float(x) for x in world_pos],
                    "orientation_xyzw": [float(x) for x in world_orn],
                    "scale": [float(x) for x in dimensions],
                    "rgba": [float(x) for x in rgba],
                }
            )
    return shapes


def first_crossing(values, threshold: float, direction: str = "above") -> int | None:
    import numpy as np

    mask = values > threshold if direction == "above" else values < threshold
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def summary_target_events(rollout_dir: Path, eef_xyz) -> list[dict] | None:
    return summary_target_events_for_trace(rollout_dir / "rollout_trace.npz", eef_xyz)


def summary_target_events_for_trace(trace_path: Path, eef_xyz, colors: list[list[float]] | None = None) -> list[dict] | None:
    summary_path = trace_path.parent.parent / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for rollout in summary.get("rollouts", []):
        if Path(rollout.get("trace", "")).resolve() != trace_path.resolve():
            continue
        raw_events = rollout.get("target_events", [])
        events = []
        for idx, event in enumerate(raw_events):
            step = min(int(event["step"]), len(eef_xyz) - 1)
            color = colors[idx] if colors is not None and idx < len(colors) else task_color(idx, len(raw_events))
            events.append(
                {
                    "name": str(event["target_name"]),
                    "step": step,
                    "position": eef_xyz[step].astype(float).tolist(),
                    "color": color,
                }
            )
        return events
    return None


def event_markers(scene_states, eef_xyz, rollout_dir: Path) -> list[dict]:
    summary_events = summary_target_events(rollout_dir, eef_xyz)
    if summary_events is not None:
        return summary_events

    events = []
    candidates = [
        ("button_on", first_crossing(scene_states[:, 5], 0.5), task_color(0, 2)),
        ("drawer_open", first_crossing(scene_states[:, 1], 0.08), task_color(1, 2)),
    ]
    for name, step, color in candidates:
        if step is None:
            continue
        step = min(step, len(eef_xyz) - 1)
        events.append({"name": name, "step": int(step), "position": eef_xyz[step].astype(float).tolist(), "color": color})
    return events


def texture_paths() -> dict[str, str]:
    paths = {
        "wood_base": TEXTURE_DIR / "wood_base.jpg",
        "wood_handle_atlas": TEXTURE_DIR / "wood_handle_atlas.png",
        "wood_roughness": TEXTURE_DIR / "wood_roughness.jpg",
        "wood_normal_gl": TEXTURE_DIR / "wood_normal_gl.jpg",
        "wood_ao": TEXTURE_DIR / "wood_ao.jpg",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing render texture files:\n" + "\n".join(missing))
    return {key: str(path) for key, path in paths.items()}


def load_trace_points(trace_path: Path, horizon: int | None = None):
    import numpy as np

    trace = np.load(trace_path, allow_pickle=True)
    points = np.asarray(trace["robot_states"], dtype=np.float32)[:, :3]
    if horizon is not None:
        points = points[: min(len(points), horizon + 1)]
    return trace, points


def trajectory_from_trace(
    name: str,
    trace_path: Path,
    color: list[float],
    horizon: int | None = None,
    events: list[dict] | None = None,
) -> tuple[dict, object]:
    trace, points = load_trace_points(trace_path, horizon=horizon)
    return (
        {
            "name": name,
            "trace_path": str(trace_path.resolve()),
            "points": points.astype(float).tolist(),
            "color": color,
            "events": events or [],
        },
        trace,
    )


def rollout_trace_path(rollout_dir: Path) -> Path:
    return rollout_dir.expanduser().resolve() / "rollout_trace.npz"


def safety_box_from_summary(summary_dir: Path) -> list[dict]:
    summary_path = summary_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    box = summary.get("safety_box")
    if not box:
        return []
    margin = float(box.get("margin", 0.0))
    return [
        {
            "name": "safety_box",
            "x_min": float(box["x_min"]) - margin,
            "x_max": float(box["x_max"]) + margin,
            "y_min": float(box["y_min"]) - margin,
            "y_max": float(box["y_max"]) + margin,
            "z_min": 0.44,
            "z_max": 0.63,
            "color": task_color(2, 5),
            "alpha": 0.24,
        }
    ]


def selected_batch_rollouts(batch_dir: Path, take_lowest_steps: bool = False, n: int = 5) -> list[dict]:
    summary = json.loads((batch_dir / "batch_summary.json").read_text(encoding="utf-8"))
    horizon = int(summary.get("horizon", 10**9))
    rollouts = list(summary.get("rollouts", []))
    if take_lowest_steps:
        candidates = rollouts
    else:
        candidates = [rollout for rollout in rollouts if int(rollout.get("termination_step", horizon)) < horizon]
        if len(candidates) < n:
            candidates = rollouts
    return sorted(candidates, key=lambda rollout: int(rollout.get("termination_step", horizon)))[:n]


def build_plot_spec(args: argparse.Namespace) -> dict:
    import numpy as np

    if args.figure_preset == "safety-switch":
        rollout_dirs = sorted(SAFETY_SWITCH_DIR.glob("F_switch_G_safety_scale_10_seed_*"))
        trajectories = []
        primary_trace_path = rollout_trace_path(rollout_dirs[0])
        for rollout_dir in rollout_dirs:
            trace_path = rollout_trace_path(rollout_dir)
            traj, _ = trajectory_from_trace(rollout_dir.name, trace_path, task_color(2, 5))
            trajectories.append(traj)
        return {
            "primary_trace_path": primary_trace_path,
            "trajectories": trajectories,
            "safety_boxes": safety_box_from_summary(SAFETY_SWITCH_DIR),
        }

    if args.figure_preset == "flower":
        trace_path = rollout_trace_path(FLOWER_ROLLOUT_DIR)
        traj, _ = trajectory_from_trace("flower_vla_h150", trace_path, task_color(0, 5), horizon=150)
        return {"primary_trace_path": trace_path, "trajectories": [traj], "safety_boxes": []}

    if args.figure_preset == "base-diverse":
        trajectories = []
        primary_trace_path = None
        for label, batch_dir, color_idx, take_lowest in BASE_DIVERSE_BATCHES:
            for rollout in selected_batch_rollouts(batch_dir, take_lowest_steps=take_lowest, n=5):
                trace_path = Path(rollout["trace"]).resolve()
                primary_trace_path = primary_trace_path or trace_path
                name = f"{label}_seed_{rollout.get('seed')}_step_{rollout.get('termination_step')}"
                traj, _ = trajectory_from_trace(name, trace_path, task_color(color_idx, 5))
                trajectories.append(traj)
        return {"primary_trace_path": primary_trace_path, "trajectories": trajectories, "safety_boxes": []}

    if args.figure_preset == "ordered-stage":
        trace_path = rollout_trace_path(ORDERED_STAGE_DIR)
        trace, points = load_trace_points(trace_path)
        colors = [task_color(2, 5), task_color(0, 5), task_color(1, 5)]
        events = summary_target_events_for_trace(trace_path, points, colors=colors) or []
        return {
            "primary_trace_path": trace_path,
            "trajectories": [
                {
                    "name": "ordered_switch_button_drawer",
                    "trace_path": str(trace_path),
                    "points": points.astype(float).tolist(),
                    "color": colors[0],
                    "events": events,
                }
            ],
            "safety_boxes": [],
        }

    rollout_dir = args.rollout_dir.expanduser().resolve()
    trace_path = rollout_trace_path(rollout_dir)
    trace, points = load_trace_points(trace_path)
    events = event_markers(np.asarray(trace["scene_states"], dtype=np.float32), points, rollout_dir)
    return {
        "primary_trace_path": trace_path,
        "trajectories": [
            {
                "name": "eef_trajectory",
                "trace_path": str(trace_path),
                "points": points.astype(float).tolist(),
                "color": task_color(0, max(1, len(events))),
                "events": events,
            }
        ],
        "safety_boxes": [],
    }


def build_manifest(args: argparse.Namespace, output: Path) -> Path:
    import numpy as np
    import pybullet as p
    import robomimic.utils.file_utils as FileUtils

    add_repo_import_paths()
    import calvin_experiments.calvin_rollout_utils as CRU

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    plot_spec = build_plot_spec(args)
    primary_trace_path = Path(plot_spec["primary_trace_path"]).resolve()
    primary_trace = np.load(primary_trace_path, allow_pickle=True)

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(ENV_CHECKPOINT))
    env = None
    try:
        env, _ = CRU.load_fresh_env_from_checkpoint(
            ckpt_dict,
            seed=int(np.asarray(primary_trace["rollout_seed"]).item()),
            suppress_output=True,
        )
        CRU.reset_env_to_scene_robot(env, primary_trace["scene_states"][0], primary_trace["robot_states"][0])
        visual_shapes = export_visual_shapes(env, p, CRU)
    finally:
        CRU.close_env_quietly(env)

    trajectories = plot_spec["trajectories"]
    first_points = trajectories[0]["points"]
    first_events = trajectories[0].get("events", [])

    manifest = {
        "repo_root": str(REPO_ROOT),
        "figure_preset": args.figure_preset,
        "rollout_dir": str(primary_trace_path.parent),
        "trace_path": str(primary_trace_path),
        "env_checkpoint": str(ENV_CHECKPOINT),
        "view": args.view,
        "camera": build_camera_config(args),
        "eef_xyz": first_points,
        "events": first_events,
        "trajectories": trajectories,
        "safety_boxes": plot_spec.get("safety_boxes", []),
        "visual_shapes": visual_shapes,
        "texture_paths": texture_paths(),
        "style": {
            "key_light_energy": args.key_light_energy,
            "bulb_light_energy": args.bulb_light_energy,
            "led_light_energy": args.led_light_energy,
            "world_strength": args.world_strength,
            "camera_bg_strength": args.camera_bg_strength,
            "exposure": args.exposure,
            "wood_roughness": args.wood_roughness,
            "wood_saturation": args.wood_saturation,
            "wood_value": args.wood_value,
            "robot_value": args.robot_value,
            "trajectory_radius": args.trajectory_radius,
            "trajectory_halo_radius": args.trajectory_halo_radius,
            "trajectory_halo_alpha": args.trajectory_halo_alpha,
            "trajectory_emission": args.trajectory_emission,
            "event_emission": args.event_emission,
            "transition_steps": args.transition_steps,
            "event_radius": args.event_radius,
        },
    }
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Visual shapes: {len(visual_shapes)}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"First EEF points: {len(manifest['eef_xyz'])}")
    if manifest["events"]:
        print("Events: " + ", ".join(f"{event['name']}@{event['step']}" for event in manifest["events"]))
    return manifest_path


def run_blender(args: argparse.Namespace, manifest_path: Path, output: Path) -> None:
    cmd = [
        args.blender,
        "-b",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--blender-render",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        "--samples",
        str(args.samples),
        "--resolution-scale",
        str(args.resolution_scale),
    ]
    if args.debug_frames:
        cmd.append("--debug-frames")
    print("Render command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def driver_main() -> None:
    args = parse_driver_args()
    output = args.output.expanduser().resolve() if args.output is not None else next_versioned_output(args.output_dir.expanduser().resolve())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing image: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = build_manifest(args, output)
    if not args.export_only:
        run_blender(args, manifest_path, output)
    print(f"Output: {output}")


def parse_blender_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal Blender render mode.")
    parser.add_argument("--blender-render", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--debug-frames", action="store_true")
    return parser.parse_args(argv_after_double_dash())


def load_blender_modules() -> None:
    global bpy, Quaternion, Vector
    import bpy as _bpy
    from mathutils import Quaternion as _Quaternion
    from mathutils import Vector as _Vector

    bpy = _bpy
    Quaternion = _Quaternion
    Vector = _Vector


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_principled(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.4,
    metallic: float = 0.0,
    alpha: float | None = None,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
    specular_ior_level: float | None = None,
):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color if alpha is None else (color[0], color[1], color[2], alpha)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        def set_input(input_name: str, value) -> None:
            if input_name in bsdf.inputs:
                bsdf.inputs[input_name].default_value = value

        set_input("Base Color", color)
        set_input("Roughness", roughness)
        set_input("Metallic", metallic)
        if specular_ior_level is not None:
            set_input("Specular IOR Level", specular_ior_level)
        if alpha is not None:
            set_input("Alpha", alpha)
            mat.blend_method = "BLEND"
            mat.use_screen_refraction = True
        if emission is not None:
            set_input("Emission Color", emission)
            set_input("Emission Strength", emission_strength)
    return mat


def set_bsdf_input(bsdf, name: str, value) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = value


def add_color_style_node(mat, saturation: float, value: float, fallback_color: tuple[float, float, float, float]) -> None:
    if mat.get("paper_franka_styled"):
        return
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None or "Base Color" not in bsdf.inputs:
        return
    base_input = bsdf.inputs["Base Color"]
    if base_input.is_linked:
        source_socket = base_input.links[0].from_socket
        mat.node_tree.links.remove(base_input.links[0])
        hsv = mat.node_tree.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = saturation
        hsv.inputs["Value"].default_value = value
        mat.node_tree.links.new(source_socket, hsv.inputs["Color"])
        mat.node_tree.links.new(hsv.outputs["Color"], base_input)
    else:
        base_input.default_value = fallback_color
    mat["paper_franka_styled"] = True


def make_pbr_material(
    name: str,
    base_color_path: str,
    roughness_path: str | None = None,
    normal_path: str | None = None,
    roughness: float = 0.48,
    normal_strength: float = 0.32,
    saturation: float = 1.0,
    value: float = 1.0,
    use_roughness_texture: bool = True,
    specular_ior_level: float = 0.42,
):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    base = nodes.new("ShaderNodeTexImage")
    base.image = bpy.data.images.load(base_color_path, check_existing=True)
    base.extension = "REPEAT"
    color_output = base.outputs["Color"]
    if saturation != 1.0 or value != 1.0:
        hsv = nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = saturation
        hsv.inputs["Value"].default_value = value
        links.new(color_output, hsv.inputs["Color"])
        color_output = hsv.outputs["Color"]
    links.new(color_output, bsdf.inputs["Base Color"])

    set_bsdf_input(bsdf, "Roughness", roughness)
    set_bsdf_input(bsdf, "Metallic", 0.0)
    set_bsdf_input(bsdf, "Specular IOR Level", specular_ior_level)

    if use_roughness_texture and roughness_path and Path(roughness_path).exists():
        rough = nodes.new("ShaderNodeTexImage")
        rough.image = bpy.data.images.load(roughness_path, check_existing=True)
        rough.image.colorspace_settings.name = "Non-Color"
        links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    if normal_path and Path(normal_path).exists():
        normal_tex = nodes.new("ShaderNodeTexImage")
        normal_tex.image = bpy.data.images.load(normal_path, check_existing=True)
        normal_tex.image.colorspace_settings.name = "Non-Color"
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = normal_strength
        links.new(normal_tex.outputs["Color"], normal.inputs["Color"])
        links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_materials(manifest: dict) -> dict:
    texture_paths = manifest.get("texture_paths", {})
    style = manifest.get("style", {})
    wood_roughness = style.get("wood_roughness", 0.72)
    wood_saturation = style.get("wood_saturation", 1.24)
    wood_value = style.get("wood_value", 0.55)
    return {
        "oak_wood": make_pbr_material(
            "oak_wood",
            texture_paths["wood_base"],
            texture_paths.get("wood_roughness"),
            texture_paths.get("wood_normal_gl"),
            roughness=wood_roughness,
            normal_strength=0.14,
            saturation=wood_saturation,
            value=wood_value,
            use_roughness_texture=False,
            specular_ior_level=0.18,
        ),
        "oak_handle_atlas": make_pbr_material(
            "oak_handle_atlas",
            texture_paths["wood_handle_atlas"],
            texture_paths.get("wood_roughness"),
            texture_paths.get("wood_normal_gl"),
            roughness=max(wood_roughness, 0.76),
            normal_strength=0.12,
            saturation=max(wood_saturation - 0.04, 1.0),
            value=wood_value + 0.03,
            use_roughness_texture=False,
            specular_ior_level=0.16,
        ),
        "franka_white": make_principled("franka_white", (0.86, 0.86, 0.82, 1.0), 0.38, 0.0),
        "black_rubber": make_principled("black_rubber", (0.012, 0.011, 0.010, 1.0), 0.84, 0.0, specular_ior_level=0.16),
        "gray_plastic": make_principled("gray_plastic", (0.055, 0.058, 0.064, 1.0), 0.70, 0.0, specular_ior_level=0.22),
        "bulb_emission": make_principled(
            "bulb_emission",
            (1.0, 0.82, 0.35, 1.0),
            0.25,
            0.0,
            emission=(1.0, 0.80, 0.30, 1.0),
            emission_strength=2.6,
        ),
        "led_emission": make_principled(
            "led_emission",
            (0.05, 0.95, 0.10, 1.0),
            0.35,
            0.0,
            emission=(0.05, 1.0, 0.12, 1.0),
            emission_strength=1.2,
        ),
        "axis_x": make_principled("axis_x", (1.0, 0.05, 0.05, 1.0), 0.5, 0.0),
        "axis_y": make_principled("axis_y", (0.05, 0.85, 0.05, 1.0), 0.5, 0.0),
        "axis_z": make_principled("axis_z", (0.05, 0.25, 1.0, 1.0), 0.5, 0.0),
        "debug_text": make_principled("debug_text", (0.02, 0.02, 0.02, 1.0), 0.5, 0.0),
    }


def assign_material(obj, shape: dict, materials: dict, manifest: dict) -> None:
    body = shape["body_name"].lower()
    link = shape["link_name"].lower()
    mesh = (shape.get("mesh_path") or "").lower()
    robot_value = manifest.get("style", {}).get("robot_value", 0.82)

    if "plane" in body or "plane" in link:
        return
    if "playtable" in body and link in {"base_link", "plank_link"}:
        obj.data.materials.clear()
        obj.data.materials.append(materials["oak_wood"])
    elif "playtable" in body and link in {"drawer_link", "slide_link"}:
        obj.data.materials.clear()
        obj.data.materials.append(materials["oak_handle_atlas"])
    elif "light_link" in link:
        obj.data.materials.clear()
        obj.data.materials.append(materials["bulb_emission"])
    elif "led_link" in link:
        obj.data.materials.clear()
        obj.data.materials.append(materials["led_emission"])
    elif "button" in link or "handle" in mesh:
        obj.data.materials.clear()
        obj.data.materials.append(materials["black_rubber"])
    elif "switch" in link:
        obj.data.materials.clear()
        obj.data.materials.append(materials["gray_plastic"])
    elif "panda" in body:
        if not obj.data.materials:
            obj.data.materials.append(materials["franka_white"])
        for mat in obj.data.materials:
            if mat is None:
                continue
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                add_color_style_node(mat, saturation=0.82, value=robot_value, fallback_color=(0.78, 0.79, 0.78, 1.0))
                set_bsdf_input(bsdf, "Roughness", 0.43)
                set_bsdf_input(bsdf, "Metallic", 0.0)
                set_bsdf_input(bsdf, "Specular IOR Level", 0.50)
    elif shape["geometry_type"] != GEOM_MESH:
        obj.data.materials.clear()
        obj.data.materials.append(make_principled(f"{obj.name}_rgba", tuple(shape["rgba"]), 0.45, 0.0))


def import_mesh(shape: dict) -> list:
    mesh_path = Path(shape["mesh_path"])
    before = set(bpy.context.scene.objects)
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="Y", up_axis="Z")
        else:
            bpy.ops.import_scene.obj(filepath=str(mesh_path), axis_forward="Y", axis_up="Z")
    elif suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(mesh_path), forward_axis="Y", up_axis="Z")
        else:
            bpy.ops.import_mesh.stl(filepath=str(mesh_path), axis_forward="Y", axis_up="Z")
    else:
        return []
    return [obj for obj in bpy.context.scene.objects if obj not in before]


def create_primitive(shape: dict) -> list:
    dims = shape["scale"]
    geom = shape["geometry_type"]
    if geom == GEOM_BOX:
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.object
        obj.dimensions = dims
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return [obj]
    if geom == GEOM_SPHERE:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=dims[0] if dims else 0.02)
        return [bpy.context.object]
    if geom == GEOM_CYLINDER:
        radius = dims[0] if len(dims) > 0 else 0.03
        depth = dims[1] if len(dims) > 1 else 0.03
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth)
        return [bpy.context.object]
    return []


def apply_transform(obj, shape: dict) -> None:
    obj.location = Vector(shape["position"])
    q = shape["orientation_xyzw"]
    obj.rotation_euler = Quaternion((q[3], q[0], q[1], q[2])).to_euler()
    if shape["geometry_type"] == GEOM_MESH:
        obj.scale = shape["scale"]
    obj.name = f"{shape['body_name']}_{shape['link_name']}_{shape['visual_index']}"


def import_scene_objects(manifest: dict, materials: dict) -> list:
    objects = []
    for shape in manifest["visual_shapes"]:
        body = shape["body_name"].lower()
        link = shape["link_name"].lower()
        if "plane" in body or "plane" in link:
            continue
        if "gripper_cam" in body or "gripper_cam" in link:
            continue
        imported = import_mesh(shape) if shape["geometry_type"] == GEOM_MESH and shape.get("mesh_path") else create_primitive(shape)
        for obj in imported:
            apply_transform(obj, shape)
            if hasattr(obj.data, "materials"):
                assign_material(obj, shape, materials, manifest)
            objects.append(obj)
    return objects


def create_curve(name: str, points: list[list[float]], mat, bevel_depth: float):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 3
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 5
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, xyz in zip(spline.points, points):
        point.co = (xyz[0], xyz[1], xyz[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    curve.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def blend_color(c0: tuple[float, float, float, float], c1: tuple[float, float, float, float], t: float) -> tuple[float, float, float, float]:
    return tuple((1.0 - t) * c0[i] + t * c1[i] for i in range(4))


def punch_color(color: tuple[float, float, float, float], gain: float = 1.18) -> tuple[float, float, float, float]:
    return (min(color[0] * gain, 1.0), min(color[1] * gain, 1.0), min(color[2] * gain, 1.0), color[3])


def trajectory_material(manifest: dict, name: str, color: tuple[float, float, float, float], alpha: float = 1.0):
    return make_principled(
        name,
        (color[0], color[1], color[2], alpha),
        0.72,
        0.0,
        alpha=None if alpha >= 1.0 else alpha,
        emission=color,
        emission_strength=manifest.get("style", {}).get("trajectory_emission", 1.15),
        specular_ior_level=0.0,
    )


def add_trajectory(manifest: dict) -> None:
    style = manifest.get("style", {})
    default_radius = style.get("trajectory_radius", 0.0054)
    default_halo_radius = style.get("trajectory_halo_radius", 0.0068)
    halo_alpha = style.get("trajectory_halo_alpha", 0.13)
    fade_steps = int(style.get("transition_steps", 12))
    trajectories = manifest.get("trajectories") or [
        {
            "name": "eef_trajectory",
            "points": manifest["eef_xyz"],
            "events": manifest.get("events", []),
            "color": [0.95, 0.16, 0.42, 1.0],
        }
    ]

    def add_phase_curve(
        name: str,
        points: list[list[float]],
        start: int,
        end: int,
        color: tuple[float, float, float, float],
        radius: float,
        halo_radius: float,
    ) -> None:
        start = max(0, min(start, len(points) - 1))
        end = max(0, min(end, len(points) - 1))
        if end - start < 1:
            return
        color = tuple(color)
        halo_color = (0.82 + 0.18 * color[0], 0.82 + 0.18 * color[1], 0.82 + 0.18 * color[2], halo_alpha)
        if halo_alpha > 0.0:
            create_curve(f"{name}_halo", points[start : end + 1], trajectory_material(manifest, f"{name}_halo_mat", halo_color, alpha=halo_alpha), halo_radius)
        create_curve(name, points[start : end + 1], trajectory_material(manifest, f"{name}_mat", color, alpha=1.0), radius)

    for traj_idx, trajectory in enumerate(trajectories):
        points = trajectory["points"]
        events = sorted(trajectory.get("events", []), key=lambda event: event["step"])
        radius = float(trajectory.get("radius", default_radius))
        halo_radius = float(trajectory.get("halo_radius", default_halo_radius))
        base_color = tuple(trajectory.get("color", [0.95, 0.16, 0.42, 1.0]))
        prefix = f"trajectory_{traj_idx:02d}_{trajectory.get('name', 'trace')}"
        if not events:
            create_curve(prefix, points, trajectory_material(manifest, f"{prefix}_mat", base_color), radius)
            continue

        for idx, event in enumerate(events):
            event_step = int(event["step"])
            color = tuple(event.get("color", base_color))
            phase_start = 0 if idx == 0 else min(int(events[idx - 1]["step"]) + max(fade_steps, 0), event_step)
            add_phase_curve(f"{prefix}_phase_{idx:02d}_{event['name']}", points, phase_start, event_step, color, radius, halo_radius)

            if idx + 1 >= len(events) or fade_steps <= 0:
                continue
            next_step = int(events[idx + 1]["step"])
            transition_end = min(event_step + fade_steps, next_step)
            next_color = tuple(events[idx + 1].get("color", base_color))
            span = max(1, transition_end - event_step)
            for step in range(event_step, transition_end):
                t = (step - event_step + 1) / span
                seg_color = blend_color(color, next_color, t)
                mat = trajectory_material(manifest, f"{prefix}_transition_{idx:02d}_{step:03d}_mat", seg_color, alpha=1.0)
                create_curve(f"{prefix}_transition_{idx:02d}_{step:03d}", points[step : step + 2], mat, radius)


def add_event_markers(manifest: dict) -> None:
    style = manifest.get("style", {})
    radius = manifest.get("style", {}).get("event_radius", 0.018)
    trajectories = manifest.get("trajectories") or [{"events": manifest.get("events", [])}]
    for trajectory in trajectories:
        for event in trajectory.get("events", []):
            color = tuple(event.get("color", [1.0, 0.5, 0.2, 1.0]))
            mat = make_principled(
                f"event_{event['name']}",
                color,
                0.70,
                0.0,
                emission=color,
                emission_strength=style.get("event_emission", 0.7),
                specular_ior_level=0.0,
            )
            bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=20, radius=radius, location=event["position"])
            sphere = bpy.context.object
            sphere.name = f"event_marker_{event['name']}_{event['step']}"
            sphere.data.materials.append(mat)


def add_safety_boxes(manifest: dict) -> None:
    for box in manifest.get("safety_boxes", []):
        color = tuple(box.get("color", [1.0, 0.1, 0.1, 1.0]))
        mat = make_principled(
            f"{box.get('name', 'safety_box')}_material",
            color,
            0.45,
            0.0,
            alpha=float(box.get("alpha", 0.24)),
            emission=color,
            emission_strength=0.08,
            specular_ior_level=0.0,
        )
        x_min, x_max = float(box["x_min"]), float(box["x_max"])
        y_min, y_max = float(box["y_min"]), float(box["y_max"])
        z_min, z_max = float(box.get("z_min", 0.44)), float(box.get("z_max", 0.63))
        location = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)
        dimensions = (x_max - x_min, y_max - y_min, z_max - z_min)
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        cube = bpy.context.object
        cube.name = box.get("name", "safety_box")
        cube.dimensions = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        cube.data.materials.append(mat)


def add_lighting(manifest: dict) -> None:
    style = manifest.get("style", {})
    bpy.ops.object.light_add(type="AREA", location=(-0.25, -0.65, 1.65))
    key = bpy.context.object
    key.name = "large_softbox"
    key.data.energy = style.get("key_light_energy", 185.0)
    key.data.size = 3.8

    for shape in manifest["visual_shapes"]:
        link = shape["link_name"].lower()
        if "light_link" in link:
            bpy.ops.object.light_add(type="POINT", location=shape["position"])
            light = bpy.context.object
            light.name = "bulb_point_light"
            light.data.energy = style.get("bulb_light_energy", 8.0)
            light.data.shadow_soft_size = 0.11
        if "led_link" in link:
            bpy.ops.object.light_add(type="POINT", location=shape["position"])
            light = bpy.context.object
            light.name = "led_point_light"
            light.data.color = (0.1, 1.0, 0.1)
            light.data.energy = style.get("led_light_energy", 3.0)
            light.data.shadow_soft_size = 0.055


def create_axis_curve(name: str, start, end, mat) -> None:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0012
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (start.x, start.y, start.z, 1.0)
    spline.points[1].co = (end.x, end.y, end.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    curve.materials.append(mat)
    bpy.context.collection.objects.link(obj)


def add_debug_frames(manifest: dict, materials: dict) -> None:
    length = 0.035
    for shape in manifest["visual_shapes"]:
        if "plane" in shape["body_name"].lower():
            continue
        origin = Vector(shape["position"])
        q = shape["orientation_xyzw"]
        rot = Quaternion((q[3], q[0], q[1], q[2])).to_matrix()
        create_axis_curve(f"debug_x_{shape['body_name']}_{shape['link_name']}_{shape['visual_index']}", origin, origin + rot @ Vector((length, 0, 0)), materials["axis_x"])
        create_axis_curve(f"debug_y_{shape['body_name']}_{shape['link_name']}_{shape['visual_index']}", origin, origin + rot @ Vector((0, length, 0)), materials["axis_y"])
        create_axis_curve(f"debug_z_{shape['body_name']}_{shape['link_name']}_{shape['visual_index']}", origin, origin + rot @ Vector((0, 0, length)), materials["axis_z"])
        bpy.ops.object.text_add(location=origin + Vector((0, 0, length * 1.15)), rotation=(1.2, 0.0, -0.5))
        text = bpy.context.object
        text.name = f"debug_label_{shape['body_name']}_{shape['link_name']}_{shape['visual_index']}"
        text.data.body = f"{shape['body_name']}:{shape['link_name']}"
        text.data.size = 0.018
        text.data.align_x = "CENTER"
        text.data.materials.append(materials["debug_text"])


def look_at_camera(camera, eye, target) -> None:
    camera.location = eye
    direction = target - eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(manifest: dict) -> None:
    cfg = manifest["camera"]
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.location = Vector(cfg["look_from"])
    if "rotation_euler_deg" in cfg:
        camera.rotation_euler = tuple(math.radians(value) for value in cfg["rotation_euler_deg"])
    else:
        look_at_camera(camera, Vector(cfg["look_from"]), Vector(cfg["look_at"]))
    camera.data.lens_unit = "FOV"
    camera.data.angle = math.radians(cfg["fov"])
    camera.data.clip_start = cfg["nearval"]
    camera.data.clip_end = cfg["farval"]
    bpy.context.scene.camera = camera


def configure_render(manifest: dict, output: Path, samples: int, resolution_scale: float) -> None:
    style = manifest.get("style", {})
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("white_world") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    nodes.clear()
    world_output = nodes.new("ShaderNodeOutputWorld")
    light_path = nodes.new("ShaderNodeLightPath")
    camera_bg = nodes.new("ShaderNodeBackground")
    scene_bg = nodes.new("ShaderNodeBackground")
    mix = nodes.new("ShaderNodeMixShader")
    camera_bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    camera_bg.inputs["Strength"].default_value = style.get("camera_bg_strength", 10.0)
    scene_bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scene_bg.inputs["Strength"].default_value = style.get("world_strength", 0.38)
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(scene_bg.outputs["Background"], mix.inputs[1])
    links.new(camera_bg.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], world_output.inputs["Surface"])
    scene.render.resolution_x = int(manifest["camera"]["width"] * resolution_scale)
    scene.render.resolution_y = int(manifest["camera"]["height"] * resolution_scale)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = style.get("exposure", -0.35)
    scene.view_settings.gamma = 1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)


def pack_blender_assets() -> None:
    """Make the saved .blend usable without external texture paths."""
    import bpy

    try:
        bpy.ops.file.pack_all()
    except Exception as exc:
        print(f"Warning: bpy.ops.file.pack_all() failed: {exc}")

    for image in bpy.data.images:
        if image.packed_file is not None or not image.filepath:
            continue
        try:
            image.pack()
        except Exception as exc:
            print(f"Warning: could not pack image {image.name}: {exc}")


def blender_render_main() -> None:
    load_blender_modules()
    args = parse_blender_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()

    clear_scene()
    materials = make_materials(manifest)
    import_scene_objects(manifest, materials)
    add_safety_boxes(manifest)
    add_trajectory(manifest)
    add_event_markers(manifest)
    if args.debug_frames:
        add_debug_frames(manifest, materials)
    add_lighting(manifest)
    add_camera(manifest)
    configure_render(manifest, output, args.samples, args.resolution_scale)
    pack_blender_assets()
    bpy.ops.wm.save_as_mainfile(filepath=str(output.with_suffix(".blend")))
    bpy.ops.render.render(write_still=True)
    print(output)


if __name__ == "__main__":
    if is_blender_render_mode():
        blender_render_main()
    else:
        driver_main()
