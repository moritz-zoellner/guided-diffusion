from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
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
SAVED_BLEND_DIR = REPO_ROOT / "outputs/paper_plots/editable_blender_panels"
SAVED_BLEND_PANELS = {
    "long-horizon": "long_horizon.blend",
    "complex-chained": "long_horizon.blend",
    "conditional": "conditional.blend",
    "complex-conditional": "conditional.blend",
    "safety": "safety_constraint.blend",
    "complex-region": "safety_constraint.blend",
    "behavior-prior": "behavior_prior.blend",
    "base-diverse": "behavior_prior.blend",
}
OUR_BLUE_HEX = "#275fca"
UNSAFE_RED_HEX = "#b85f5a"
FLOWER_GREEN_HEX = "#4e8b68"
BASE_PRIOR_COLOR_HEX = ["#A37B63", "#5B6C7D", "#7D8A63", "#6D7E95"]
TASK_COLOR_HEX = [OUR_BLUE_HEX]
ALIGN_START_PREFIX_POINTS = 14
ALIGN_START_MIN_DISTANCE = 0.004
ALIGN_START_MAX_DISTANCE = 0.16
SCENE_SLIDE_INDEX = 0
SCENE_DRAWER_INDEX = 1
SCENE_BUTTON_INDEX = 2
SCENE_SWITCH_INDEX = 3
SCENE_LIGHTBULB_INDEX = 4
SCENE_LED_INDEX = 5
DRAWER_RENDER_CLOSE_THRESHOLD = 0.125
DRAWER_RENDER_CLOSE_OFFSET = 0.075
SAFETY_SWITCH_DIR = REPO_ROOT / "outputs/calvin/paper_stls/F_switch_G_safety/20260506_125354"
COMPLEX_REGION_DIR = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_complex_N10/region"
COMPLEX_CHAINED_DIR = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_chained_N10_restored/chained"
COMPLEX_CONDITIONAL_DIR = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/hint2_complex_N10/conditional"
FLOWER_COMPLEX_DIR = REPO_ROOT / "outputs/calvin_paper/complex-behaviors/baselines/flower/flower_complex_N10_same_specs"
FLOWER_CHAINED_ROLLOUT_DIR = FLOWER_COMPLEX_DIR / "chained/rollout_002_seed_002"
FLOWER_CONDITIONAL_ROLLOUT_DIR = FLOWER_COMPLEX_DIR / "conditional/rollout_000_seed_000"
FLOWER_REGION_DIR = FLOWER_COMPLEX_DIR / "region"
FLOWER_ROLLOUT_DIR = REPO_ROOT / (
    "outputs/calvin/baselines/flower/our_env_rollouts/"
    "flower_vla_scene_blocks_hidden_rollouts1_horizon300_seed0_07-05-26__00-15-13/rollout_000"
)
ORDERED_STAGE_DIR = REPO_ROOT / (
    "outputs/calvin/paper_stls/F_drawer_after_button_switch/ordered_stage_20260507_191217/"
    "ordered_button_on_and_switch_on_then_drawer_open_start02_x0p10_ym0p20_seed_000"
)
CYCLIC_ROLLOUT_DIR = REPO_ROOT / (
    "outputs/calvin/paper_stls/G_cyclic_drawer_switch/20260521_011654/"
    "cycle_drawer_open_then_switch_on_then_button_pressed_then_drawer_closed_then_switch_off_then_button_pressed_seed_001"
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
        3,
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
LABEL_IDX = {
    "switch_off": 1,
    "button_pressed": 4,
    "drawer_closed": 6,
}

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


def method_color() -> list[float]:
    return hex_to_rgba(OUR_BLUE_HEX)


def unsafe_color() -> list[float]:
    return hex_to_rgba(UNSAFE_RED_HEX)


def flower_color() -> list[float]:
    return hex_to_rgba(FLOWER_GREEN_HEX)


def base_prior_color(index: int) -> list[float]:
    return hex_to_rgba(BASE_PRIOR_COLOR_HEX[index % len(BASE_PRIOR_COLOR_HEX)])


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
            "Run this with pixi/python, not directly with blender. "
            "Use --blend or --blend-panel to render an existing saved .blend without CALVIN/PyBullet inputs."
        )
    )
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument(
        "--figure-preset",
        choices=[
            "single",
            "safety-switch",
            "flower",
            "base-diverse",
            "ordered-stage",
            "cyclic",
            "complex-region",
            "complex-chained",
            "complex-conditional",
        ],
        default="single",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Optional exact output PNG path. Must not already exist.")
    parser.add_argument(
        "--blender",
        default=default_blender_path(),
        help='Blender executable path or command prefix, e.g. "flatpak run org.blender.Blender".',
    )
    parser.add_argument(
        "--blend",
        type=Path,
        default=None,
        help="Render an existing saved .blend file instead of rebuilding the scene from CALVIN rollout traces.",
    )
    parser.add_argument(
        "--blend-panel",
        choices=sorted(SAVED_BLEND_PANELS),
        default=None,
        help="Render one of the downloaded editable Blender panels without CALVIN/PyBullet inputs.",
    )
    parser.add_argument(
        "--list-blends",
        action="store_true",
        help="List saved .blend panel files found in the expected download locations and exit.",
    )
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


def found_saved_blends() -> dict[str, Path]:
    found = {}
    for panel, filename in SAVED_BLEND_PANELS.items():
        if panel in found:
            continue
        candidate = SAVED_BLEND_DIR / filename
        if candidate.exists():
            found[panel] = candidate
    return found


def print_saved_blends() -> None:
    found = found_saved_blends()
    if not found:
        print(f"No saved .blend panels found in: {SAVED_BLEND_DIR}")
        return
    print("Saved .blend panels:")
    for panel in sorted(found):
        print(f"  {panel}: {found[panel]}")


def resolve_saved_blend(args: argparse.Namespace) -> Path:
    if args.blend is not None and args.blend_panel is not None:
        raise ValueError("Use either --blend or --blend-panel, not both.")
    if args.blend is not None:
        blend_path = args.blend.expanduser().resolve()
        if not blend_path.exists():
            raise FileNotFoundError(f"Saved Blender file not found: {blend_path}")
        return blend_path
    if args.blend_panel is None:
        raise ValueError("No saved Blender file requested.")
    filename = SAVED_BLEND_PANELS[args.blend_panel]
    blend_path = SAVED_BLEND_DIR / filename
    if blend_path.exists():
        return blend_path.resolve()
    raise FileNotFoundError(f"Could not find .blend panel '{args.blend_panel}': {blend_path}")


def resolve_blender_command(blender: str) -> list[str]:
    parts = shlex.split(str(blender))
    if not parts:
        raise ValueError("Empty Blender executable command.")

    blender_path = Path(parts[0]).expanduser()
    if blender_path.is_absolute() or blender_path.parent != Path("."):
        if blender_path.exists():
            return [str(blender_path.resolve()), *parts[1:]]
        raise FileNotFoundError(
            "Blender executable not found: "
            f"{blender_path}\nInstall Blender, extract it under tools/blender-4.2.0-linux-x64, "
            "or pass --blender /path/to/blender."
        )
    resolved = shutil.which(parts[0])
    if resolved is not None:
        return [resolved, *parts[1:]]
    raise FileNotFoundError(
        f"Blender executable '{parts[0]}' was not found on PATH, and the repo-local Blender binary is missing at "
        f"{LOCAL_BLENDER}. Install Blender or pass --blender /path/to/blender. "
        'Flatpak installs can be passed as --blender "flatpak run org.blender.Blender".'
    )


def render_saved_blend(args: argparse.Namespace, blend_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    expression_lines = [
        "import bpy",
        f"output = {str(output)!r}",
        "scene = bpy.context.scene",
        "scene.render.engine = 'CYCLES'",
        f"scene.cycles.samples = {int(args.samples)}",
        "scene.cycles.use_denoising = True",
        "scene.render.filepath = output",
    ]
    if args.resolution is not None:
        expression_lines.extend(
            [
                f"scene.render.resolution_x = {int(args.resolution[0])}",
                f"scene.render.resolution_y = {int(args.resolution[1])}",
                "scene.render.resolution_percentage = 100",
            ]
        )
    elif args.resolution_scale != 1.0:
        expression_lines.append(f"scene.render.resolution_percentage = {max(1, int(round(args.resolution_scale * 100.0)))}")
    expression_lines.append("bpy.ops.render.render(write_still=True)")
    blender_cmd = resolve_blender_command(args.blender)
    cmd = [
        *blender_cmd,
        "-b",
        str(blend_path),
        "--python-expr",
        "\n".join(expression_lines),
    ]
    print("Rendering saved Blender file:")
    print(" ".join(blender_cmd + ["-b", str(blend_path), "--python-expr", "<render expression>"]), flush=True)
    subprocess.run(cmd, check=True)


def format_command_for_error(cmd) -> str:
    if not isinstance(cmd, (list, tuple)):
        return str(cmd)
    parts = [str(part) for part in cmd]
    if "--python-expr" in parts:
        idx = parts.index("--python-expr")
        parts = parts[: idx + 1] + ["<render expression>"]
    return " ".join(parts)


def validate_calvin_driver_inputs(args: argparse.Namespace) -> None:
    if ENV_CHECKPOINT.exists():
        return
    found = found_saved_blends()
    hint = ""
    if found:
        first_panel = sorted(found)[0]
        hint = (
            "\nSaved .blend panels are available, so you can bypass CALVIN/PyBullet with e.g.:\n"
            f"  python figures/render_calvin_trajectory_figure.py --blend-panel {first_panel} --output outputs/{first_panel}.png"
        )
    raise FileNotFoundError(
        "Cannot rebuild a CALVIN scene because the environment checkpoint is missing:\n"
        f"  {ENV_CHECKPOINT}\n"
        "Rebuilding from rollout traces also requires pybullet, robomimic, and the referenced outputs/calvin run data."
        f"{hint}"
    )


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


def events_from_raw_events(raw_events: list[dict], eef_xyz, colors: list[list[float]] | None = None) -> list[dict]:
    events = []
    for idx, event in enumerate(raw_events):
        step = min(int(event["step"]), len(eef_xyz) - 1)
        color = colors[idx] if colors is not None and idx < len(colors) else method_color()
        events.append(
            {
                "name": str(event["target_name"]),
                "step": step,
                "position": eef_xyz[step].astype(float).tolist(),
                "color": color,
            }
        )
    return events


def summary_target_events_for_trace(trace_path: Path, eef_xyz, colors: list[list[float]] | None = None) -> list[dict] | None:
    rollout_summary = trace_path.parent / "rollout_summary.json"
    if rollout_summary.exists():
        summary = json.loads(rollout_summary.read_text(encoding="utf-8"))
        raw_events = summary.get("target_events", [])
        if raw_events:
            return events_from_raw_events(raw_events, eef_xyz, colors=colors)

    for filename in ("task_summary.json", "summary.json"):
        summary_path = trace_path.parent.parent / filename
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for rollout in summary.get("rollouts", []):
            if Path(rollout.get("trace", "")).resolve() != trace_path.resolve():
                continue
            return events_from_raw_events(rollout.get("target_events", []), eef_xyz, colors=colors)
    return None


def event_markers(scene_states, eef_xyz, rollout_dir: Path) -> list[dict]:
    summary_events = summary_target_events(rollout_dir, eef_xyz)
    if summary_events is not None:
        return summary_events

    events = []
    candidates = [
        ("button_on", first_crossing(scene_states[:, 5], 0.5), method_color()),
        ("drawer_open", first_crossing(scene_states[:, 1], 0.08), method_color()),
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


def label_rising_edges(trace, label_idx: int) -> list[int]:
    import numpy as np

    labels = np.asarray(trace["labels_over_time"], dtype=np.int32)
    values = labels[:, label_idx]
    return [int(idx) for idx in (np.flatnonzero((values[1:] == 1) & (values[:-1] == 0)) + 1)]


def halfway_between_first_two_label_edges(trace, label_idx: int) -> int | None:
    edges = label_rising_edges(trace, label_idx)
    if len(edges) < 2:
        return None
    return int(round((edges[0] + edges[1]) * 0.5))


def first_label_rising_edge(trace, label_idx: int) -> int | None:
    edges = label_rising_edges(trace, label_idx)
    return edges[0] if edges else None


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


def trajectory_from_rollout(
    name: str,
    rollout_dir: Path,
    color: list[float],
    horizon: int | None = None,
    include_events: bool = True,
    event_colors: list[list[float]] | None = None,
    radius_scale: float | None = None,
) -> tuple[dict, object]:
    trace_path = rollout_trace_path(rollout_dir)
    if horizon is None:
        horizon = rollout_horizon(rollout_dir)
    trace, points = load_trace_points(trace_path, horizon=horizon)
    events = []
    if include_events:
        events = summary_target_events_for_trace(
            trace_path,
            points,
            colors=event_colors if event_colors is not None else [color] * 16,
        ) or []
    trajectory = {
        "name": name,
        "trace_path": str(trace_path.resolve()),
        "points": points.astype(float).tolist(),
        "color": color,
        "events": events,
    }
    if radius_scale is not None:
        trajectory["radius_scale"] = float(radius_scale)
        trajectory["halo_radius_scale"] = float(radius_scale)
    return trajectory, trace


def first_target_point(trajectories: list[dict]):
    import numpy as np

    for trajectory in trajectories:
        events = sorted(trajectory.get("events", []), key=lambda event: int(event["step"]))
        if events:
            return np.asarray(events[0]["position"], dtype=float)
    return None


def smooth_start_prefix(anchor, start, points, count: int) -> list[list[float]]:
    import numpy as np

    p0 = np.asarray(anchor, dtype=float)
    p3 = np.asarray(start, dtype=float)
    delta = p3 - p0
    dist = float(np.linalg.norm(delta[:2]))
    if dist <= 1e-8 or count <= 0:
        return []

    if len(points) > 2:
        lookahead = np.asarray(points[min(5, len(points) - 1)], dtype=float) - p3
    else:
        lookahead = delta
    delta_xy = delta[:2]
    lookahead_xy = lookahead[:2]
    if np.linalg.norm(lookahead_xy) < 1e-8 or float(np.dot(lookahead_xy, delta_xy)) < 0.0:
        end_tangent_xy = delta_xy
    else:
        end_tangent_xy = lookahead_xy / np.linalg.norm(lookahead_xy) * dist

    c0_xy = p0[:2]
    c1_xy = p0[:2] + 0.35 * delta_xy
    c2_xy = p3[:2] - 0.25 * end_tangent_xy
    c3_xy = p3[:2]

    prefix = []
    for k in range(count):
        t = k / float(count)
        one_minus_t = 1.0 - t
        xy = (
            (one_minus_t ** 3) * c0_xy
            + 3.0 * (one_minus_t ** 2) * t * c1_xy
            + 3.0 * one_minus_t * (t ** 2) * c2_xy
            + (t ** 3) * c3_xy
        )
        z = (1.0 - t) * p0[2] + t * p3[2]
        prefix.append([float(xy[0]), float(xy[1]), float(z)])
    return prefix


def align_trajectory_starts(plot_spec: dict) -> dict:
    import numpy as np

    if not plot_spec.get("align_start_anchor", False):
        return plot_spec
    trajectories = plot_spec.get("trajectories", [])
    target = first_target_point(trajectories)
    if target is None or not trajectories:
        return plot_spec

    starts = [np.asarray(trajectory["points"][0], dtype=float) for trajectory in trajectories]
    distances_to_target = [float(np.linalg.norm(start - target)) for start in starts]
    anchor_idx = int(np.argmax(distances_to_target))
    anchor = starts[anchor_idx]
    anchor_trace_path = trajectories[anchor_idx].get("trace_path")

    for trajectory, start in zip(trajectories, starts):
        gap = float(np.linalg.norm(start - anchor))
        if gap < ALIGN_START_MIN_DISTANCE or gap > ALIGN_START_MAX_DISTANCE:
            continue
        prefix = smooth_start_prefix(
            anchor,
            start,
            trajectory["points"],
            count=ALIGN_START_PREFIX_POINTS,
        )
        if not prefix:
            continue
        trajectory["points"] = prefix + trajectory["points"]
        for event in trajectory.get("events", []):
            event["step"] = int(event["step"]) + len(prefix)

    if anchor_trace_path:
        plot_spec["robot_pose_trace_path"] = anchor_trace_path
        plot_spec["robot_state_index"] = 0
    plot_spec["start_anchor_trajectory"] = trajectories[anchor_idx].get("name")
    plot_spec["start_anchor_point"] = anchor.astype(float).tolist()
    return plot_spec


def close_drawer_for_render(scene_state):
    import numpy as np

    scene = np.asarray(scene_state, dtype=np.float32).copy()
    if float(scene[SCENE_DRAWER_INDEX]) <= DRAWER_RENDER_CLOSE_THRESHOLD:
        scene[SCENE_DRAWER_INDEX] = max(
            0.0,
            float(scene[SCENE_DRAWER_INDEX]) - DRAWER_RENDER_CLOSE_OFFSET,
        )
    return scene


def mean_final_scene_value(trace_paths: list[Path], index: int, default: float) -> float:
    import numpy as np

    if not trace_paths:
        return float(default)
    values = []
    for path in trace_paths:
        trace = np.load(path, allow_pickle=True)
        values.append(float(trace["scene_states"][-1, index]))
    return float(np.asarray(values, dtype=np.float32).mean())


def consolidated_base_prior_final_scene(trace_paths_by_label: dict[str, list[Path]]):
    import numpy as np

    all_paths = [path for paths in trace_paths_by_label.values() for path in paths]
    if not all_paths:
        return None
    traces = [np.load(path, allow_pickle=True) for path in all_paths]
    scene = np.asarray(traces[0]["scene_states"][-1], dtype=np.float32).copy()
    scene[SCENE_SLIDE_INDEX] = mean_final_scene_value(
        trace_paths_by_label.get("door_left", []),
        SCENE_SLIDE_INDEX,
        scene[SCENE_SLIDE_INDEX],
    )
    scene[SCENE_DRAWER_INDEX] = mean_final_scene_value(
        trace_paths_by_label.get("drawer", []),
        SCENE_DRAWER_INDEX,
        scene[SCENE_DRAWER_INDEX],
    )
    scene[SCENE_BUTTON_INDEX] = mean_final_scene_value(
        trace_paths_by_label.get("button", []),
        SCENE_BUTTON_INDEX,
        scene[SCENE_BUTTON_INDEX],
    )
    scene[SCENE_SWITCH_INDEX] = mean_final_scene_value(
        trace_paths_by_label.get("switch", []),
        SCENE_SWITCH_INDEX,
        scene[SCENE_SWITCH_INDEX],
    )
    scene[SCENE_LIGHTBULB_INDEX] = 1.0
    scene[SCENE_LED_INDEX] = 1.0
    return scene


def closed_drawer_lights_on_scene(trace):
    import numpy as np

    scene_states = np.asarray(trace["scene_states"], dtype=np.float32)
    scene = scene_states[0].copy()
    scene[SCENE_DRAWER_INDEX] = 0.0
    both_lights_on = np.flatnonzero(
        (scene_states[:, SCENE_LIGHTBULB_INDEX] > 0.5)
        & (scene_states[:, SCENE_LED_INDEX] > 0.5)
    )
    if len(both_lights_on):
        source = scene_states[int(both_lights_on[0])]
        scene[SCENE_BUTTON_INDEX] = source[SCENE_BUTTON_INDEX]
        scene[SCENE_SWITCH_INDEX] = source[SCENE_SWITCH_INDEX]
    scene[SCENE_LIGHTBULB_INDEX] = 1.0
    scene[SCENE_LED_INDEX] = 1.0
    return scene


def rollout_trace_path(rollout_dir: Path) -> Path:
    return rollout_dir.expanduser().resolve() / "rollout_trace.npz"


def safety_box_from_summary(
    summary_dir: Path,
    trajectories: list[dict] | None = None,
    include_margin: bool = True,
    show_edges: bool = True,
) -> list[dict]:
    import numpy as np

    box = None
    for filename in ("task_summary.json", "summary.json"):
        summary_path = summary_dir / filename
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        box = summary.get("safety_box")
        if box:
            break
        for rollout in summary.get("rollouts", []):
            box = rollout.get("safety_metrics", {}).get("safety_box")
            if box:
                break
        if box:
            break
    if box is None:
        for rollout_summary in sorted(summary_dir.glob("rollout_*_seed_*/rollout_summary.json")):
            summary = json.loads(rollout_summary.read_text(encoding="utf-8"))
            box = summary.get("safety_metrics", {}).get("safety_box")
            if box:
                break
    if not box:
        return []
    margin = float(box.get("margin", 0.0)) if include_margin else 0.0
    x_min = float(box["x_min"]) - margin
    x_max = float(box["x_max"]) + margin
    y_min = float(box["y_min"]) - margin
    y_max = float(box["y_max"]) + margin
    z_min = 0.44
    z_max = 0.70
    if trajectories:
        inside_z = []
        for trajectory in trajectories:
            points = np.asarray(trajectory.get("points", []), dtype=float)
            if points.ndim != 2 or points.shape[1] < 3:
                continue
            mask = (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
            )
            if np.any(mask):
                inside_z.append(points[mask, 2])
        if inside_z:
            z_max = max(z_min + 0.045, float(np.max(np.concatenate(inside_z))) + 0.025)
    return [
        {
            "name": "safety_box",
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "color": unsafe_color(),
            "alpha": 0.42,
            "edge_alpha": 0.82,
            "show_edges": show_edges,
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


def rollout_dir_for_seed(task_dir: Path, seed: int) -> Path:
    matches = sorted(task_dir.glob(f"rollout_*_seed_{seed:03d}"))
    if not matches:
        raise FileNotFoundError(f"No rollout with seed {seed:03d} in {task_dir}")
    return matches[0]


def rollout_horizon(rollout_dir: Path) -> int | None:
    summary_path = rollout_dir / "rollout_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step = summary.get("termination_step")
    return int(step) if step is not None else None


def method_trajectory_from_rollout(name: str, rollout_dir: Path) -> tuple[dict, object]:
    color = method_color()
    return trajectory_from_rollout(
        name,
        rollout_dir,
        color,
        event_colors=[color] * 16,
    )


def build_plot_spec(args: argparse.Namespace) -> dict:
    import numpy as np

    if args.figure_preset == "safety-switch":
        rollout_dirs = sorted(SAFETY_SWITCH_DIR.glob("F_switch_G_safety_scale_10_seed_*"))
        trajectories = []
        primary_trace_path = rollout_trace_path(rollout_dirs[0])
        for rollout_dir in rollout_dirs:
            trace_path = rollout_trace_path(rollout_dir)
            traj, _ = trajectory_from_trace(rollout_dir.name, trace_path, method_color())
            trajectories.append(traj)
        return {
            "primary_trace_path": primary_trace_path,
            "trajectories": trajectories,
            "safety_boxes": safety_box_from_summary(SAFETY_SWITCH_DIR),
        }

    if args.figure_preset == "complex-region":
        rollout_dirs = [rollout_dir_for_seed(COMPLEX_REGION_DIR, seed) for seed in (2, 4)]
        flower_rollout_dirs = [rollout_dir_for_seed(FLOWER_REGION_DIR, seed) for seed in (2, 4)]
        trajectories = []
        for rollout_dir in rollout_dirs:
            traj, _ = method_trajectory_from_rollout(rollout_dir.name, rollout_dir)
            trajectories.append(traj)
        for rollout_dir in flower_rollout_dirs:
            traj, _ = trajectory_from_rollout(
                f"flower_{rollout_dir.name}",
                rollout_dir,
                flower_color(),
                include_events=False,
                radius_scale=0.92,
            )
            trajectories.append(traj)
        return align_trajectory_starts({
            "primary_trace_path": rollout_trace_path(rollout_dirs[0]),
            "trajectories": trajectories,
            "safety_boxes": safety_box_from_summary(
                COMPLEX_REGION_DIR,
                trajectories=trajectories,
                include_margin=False,
                show_edges=False,
            ),
            "scene_state_index": -1,
            "robot_state_index": 0,
            "align_start_anchor": True,
        })

    if args.figure_preset == "complex-chained":
        rollout_dir = rollout_dir_for_seed(COMPLEX_CHAINED_DIR, 0)
        traj, _ = method_trajectory_from_rollout("complex_chained_seed_000", rollout_dir)
        flower_trace = np.load(rollout_trace_path(FLOWER_CHAINED_ROLLOUT_DIR), allow_pickle=True)
        flower_horizon = halfway_between_first_two_label_edges(flower_trace, LABEL_IDX["button_pressed"])
        flower_traj, _ = trajectory_from_rollout(
            "flower_chained_seed_002",
            FLOWER_CHAINED_ROLLOUT_DIR,
            flower_color(),
            horizon=flower_horizon,
            include_events=False,
            radius_scale=0.92,
        )
        return align_trajectory_starts({
            "primary_trace_path": rollout_trace_path(rollout_dir),
            "trajectories": [traj, flower_traj],
            "safety_boxes": [],
            "scene_state_index": -1,
            "robot_state_index": 0,
            "align_start_anchor": True,
        })

    if args.figure_preset == "complex-conditional":
        rollout_dir = rollout_dir_for_seed(COMPLEX_CONDITIONAL_DIR, 0)
        traj, _ = method_trajectory_from_rollout("complex_conditional_seed_000", rollout_dir)
        flower_trace = np.load(rollout_trace_path(FLOWER_CONDITIONAL_ROLLOUT_DIR), allow_pickle=True)
        flower_horizon = first_label_rising_edge(flower_trace, LABEL_IDX["drawer_closed"])
        flower_traj, _ = trajectory_from_rollout(
            "flower_conditional_seed_000",
            FLOWER_CONDITIONAL_ROLLOUT_DIR,
            flower_color(),
            horizon=flower_horizon,
            include_events=False,
            radius_scale=0.92,
        )
        return align_trajectory_starts({
            "primary_trace_path": rollout_trace_path(rollout_dir),
            "trajectories": [traj, flower_traj],
            "safety_boxes": [],
            "scene_state_index": -1,
            "robot_state_index": 0,
            "align_start_anchor": True,
        })

    if args.figure_preset == "flower":
        trace_path = rollout_trace_path(FLOWER_ROLLOUT_DIR)
        traj, _ = trajectory_from_trace("flower_vla_h150", trace_path, task_color(0, 5), horizon=150)
        return {"primary_trace_path": trace_path, "trajectories": [traj], "safety_boxes": []}

    if args.figure_preset == "base-diverse":
        trajectories = []
        primary_trace_path = None
        selected_trace_paths_by_label = {label: [] for label, *_ in BASE_DIVERSE_BATCHES}
        for label, batch_dir, color_idx, take_lowest in BASE_DIVERSE_BATCHES:
            for rollout in selected_batch_rollouts(batch_dir, take_lowest_steps=take_lowest, n=5):
                trace_path = Path(rollout["trace"]).resolve()
                primary_trace_path = primary_trace_path or trace_path
                selected_trace_paths_by_label[label].append(trace_path)
                name = f"{label}_seed_{rollout.get('seed')}_step_{rollout.get('termination_step')}"
                traj, _ = trajectory_from_trace(name, trace_path, base_prior_color(color_idx))
                trajectories.append(traj)
        scene_override = consolidated_base_prior_final_scene(selected_trace_paths_by_label)
        return {
            "primary_trace_path": primary_trace_path,
            "trajectories": trajectories,
            "safety_boxes": [],
            "scene_state_override": None if scene_override is None else scene_override.astype(float).tolist(),
            "apply_drawer_close_adjust": False,
        }

    if args.figure_preset == "ordered-stage":
        trace_path = rollout_trace_path(ORDERED_STAGE_DIR)
        trace, points = load_trace_points(trace_path)
        colors = [method_color()] * 16
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

    if args.figure_preset == "cyclic":
        trace_path = rollout_trace_path(CYCLIC_ROLLOUT_DIR)
        trace, points = load_trace_points(trace_path)
        color = method_color()
        events = summary_target_events_for_trace(trace_path, points, colors=[color] * 64) or []
        return {
            "primary_trace_path": trace_path,
            "trajectories": [
                {
                    "name": "cyclic_drawer_switch_button",
                    "trace_path": str(trace_path),
                    "points": points.astype(float).tolist(),
                    "color": color,
                    "events": events,
                }
            ],
            "safety_boxes": [],
            "scene_state_index": 0,
            "scene_state_override": closed_drawer_lights_on_scene(trace).astype(float).tolist(),
            "robot_state_index": 0,
            "apply_drawer_close_adjust": False,
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
                "color": method_color(),
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
    robot_pose_trace_path = Path(plot_spec.get("robot_pose_trace_path", primary_trace_path)).resolve()
    robot_pose_trace = primary_trace if robot_pose_trace_path == primary_trace_path else np.load(robot_pose_trace_path, allow_pickle=True)

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(ENV_CHECKPOINT))
    env = None
    try:
        env, _ = CRU.load_fresh_env_from_checkpoint(
            ckpt_dict,
            seed=int(np.asarray(primary_trace["rollout_seed"]).item()),
            suppress_output=True,
        )
        scene_state_index = int(plot_spec.get("scene_state_index", -1))
        robot_state_index = int(plot_spec.get("robot_state_index", 0))
        if plot_spec.get("scene_state_override") is not None:
            render_scene_state = np.asarray(plot_spec["scene_state_override"], dtype=np.float32)
        else:
            render_scene_state = np.asarray(primary_trace["scene_states"][scene_state_index], dtype=np.float32)
        if plot_spec.get("apply_drawer_close_adjust", True):
            render_scene_state = close_drawer_for_render(render_scene_state)
        CRU.reset_env_to_scene_robot(
            env,
            render_scene_state,
            robot_pose_trace["robot_states"][robot_state_index],
        )
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
        "robot_pose_trace_path": str(robot_pose_trace_path),
        "env_checkpoint": str(ENV_CHECKPOINT),
        "view": args.view,
        "camera": build_camera_config(args),
        "scene_state_index": int(plot_spec.get("scene_state_index", -1)),
        "scene_state_override": plot_spec.get("scene_state_override"),
        "render_scene_state": render_scene_state.astype(float).tolist(),
        "robot_state_index": int(plot_spec.get("robot_state_index", 0)),
        "start_anchor_trajectory": plot_spec.get("start_anchor_trajectory"),
        "start_anchor_point": plot_spec.get("start_anchor_point"),
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
    blender_cmd = resolve_blender_command(args.blender)
    cmd = [
        *blender_cmd,
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
    if args.list_blends:
        print_saved_blends()
        return
    output = args.output.expanduser().resolve() if args.output is not None else next_versioned_output(args.output_dir.expanduser().resolve())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing image: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.blend is not None or args.blend_panel is not None:
        blend_path = resolve_saved_blend(args)
        render_saved_blend(args, blend_path, output)
        print(f"Output: {output}")
        return
    validate_calvin_driver_inputs(args)
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
        if body in {"block_red", "block_blue", "block_pink"}:
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
        radius_scale = float(trajectory.get("radius_scale", 1.0))
        halo_radius_scale = float(trajectory.get("halo_radius_scale", radius_scale))
        radius = float(trajectory.get("radius", default_radius * radius_scale))
        halo_radius = float(trajectory.get("halo_radius", default_halo_radius * halo_radius_scale))
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

        if not bool(box.get("show_edges", True)):
            continue

        edge_mat = make_principled(
            f"{box.get('name', 'safety_box')}_edge_material",
            color,
            0.35,
            0.0,
            alpha=float(box.get("edge_alpha", 0.82)),
            emission=color,
            emission_strength=0.16,
            specular_ior_level=0.0,
        )
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        edge_cube = bpy.context.object
        edge_cube.name = f"{box.get('name', 'safety_box')}_edge"
        edge_cube.dimensions = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        edge_cube.data.materials.append(edge_mat)
        wire = edge_cube.modifiers.new(name="visible_edges", type="WIREFRAME")
        wire.thickness = float(box.get("edge_thickness", 0.006))
        wire.use_even_offset = True


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
        try:
            driver_main()
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            print(
                f"Error: command failed with exit code {exc.returncode}: {format_command_for_error(exc.cmd)}",
                file=sys.stderr,
            )
            sys.exit(exc.returncode)
