from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import bpy


OUR_BLUE = "#174A99"
FLOWER_GREEN = "#376D52"
BASE_PRIOR_COLORS = {
    "button": "#7B61FF",
    "drawer": "#00A6D6",
    "switch": "#00A88A",
    "door_left": "#5B8DEF",
}


def argv_after_double_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tweak CALVIN Blender trajectory tube styling in-place.")
    parser.add_argument("--panel", choices=["behavior_prior", "conditional", "safety", "cyclic"], required=True)
    parser.add_argument("--cyclic-keep-events", type=int, default=24)
    return parser.parse_args(argv_after_double_dash())


def srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def hex_to_rgba(hex_color: str) -> tuple[float, float, float, float]:
    hex_color = hex_color.strip().lstrip("#")
    return tuple(srgb_to_linear(int(hex_color[i : i + 2], 16) / 255.0) for i in (0, 2, 4)) + (1.0,)


def set_socket(node, name: str, value) -> None:
    socket = node.inputs.get(name)
    if socket is None:
        return
    socket.default_value = value


def tune_material(
    material,
    color: tuple[float, float, float, float],
    *,
    emission_strength: float,
    roughness: float = 0.95,
    alpha: float = 1.0,
    specular_ior_level: float = 0.12,
) -> None:
    if material is None:
        return
    material.use_nodes = True
    material.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        set_socket(node, "Base Color", color)
        set_socket(node, "Alpha", alpha)
        set_socket(node, "Metallic", 0.0)
        set_socket(node, "Roughness", roughness)
        set_socket(node, "Diffuse Roughness", 0.85)
        set_socket(node, "Specular IOR Level", specular_ior_level)
        set_socket(node, "Coat Weight", 0.0)
        set_socket(node, "Emission Color", color)
        set_socket(node, "Emission Strength", emission_strength)


def materials_for_object(obj):
    data = getattr(obj, "data", None)
    if data is None:
        return []
    return list(getattr(data, "materials", []))


def recolor_object(
    obj,
    hex_color: str,
    *,
    emission: float,
    roughness: float = 0.95,
    specular: float = 0.12,
) -> None:
    color = hex_to_rgba(hex_color)
    for material in materials_for_object(obj):
        tune_material(material, color, emission_strength=emission, roughness=roughness, specular_ior_level=specular)


def set_curve_radius(obj, radius: float) -> None:
    data = getattr(obj, "data", None)
    if data is not None and hasattr(data, "bevel_depth"):
        data.bevel_depth = radius
        data.bevel_resolution = max(int(getattr(data, "bevel_resolution", 0)), 4)


def trajectory_objects():
    return [obj for obj in bpy.data.objects if obj.name.startswith("trajectory_")]


def event_objects():
    return [obj for obj in bpy.data.objects if obj.name.startswith("event_marker_")]


def behavior_prior_color(name: str) -> str:
    for key, color in BASE_PRIOR_COLORS.items():
        if f"_{key}_" in name or name.startswith(key):
            return color
    return "#0072B2"


def tweak_behavior_prior() -> None:
    for obj in trajectory_objects():
        recolor_object(obj, behavior_prior_color(obj.name), emission=0.07, roughness=0.94, specular=0.08)
        set_curve_radius(obj, 0.0027)


def tweak_conditional() -> None:
    for obj in trajectory_objects():
        color = FLOWER_GREEN if "flower" in obj.name else OUR_BLUE
        recolor_object(obj, color, emission=0.16, roughness=0.84, specular=0.24)
        set_curve_radius(obj, 0.0044)
    for obj in event_objects():
        recolor_object(obj, OUR_BLUE, emission=0.23, roughness=0.84, specular=0.20)


def tweak_safety() -> None:
    for obj in trajectory_objects():
        color = FLOWER_GREEN if "flower" in obj.name else OUR_BLUE
        recolor_object(obj, color, emission=0.14, roughness=0.86, specular=0.20)
        set_curve_radius(obj, 0.0037)
    for obj in event_objects():
        recolor_object(obj, OUR_BLUE, emission=0.20, roughness=0.86, specular=0.18)


def parse_phase_idx(name: str) -> int | None:
    match = re.search(r"_phase_(\d+)_", name)
    return int(match.group(1)) if match else None


def parse_event_step(name: str) -> int | None:
    match = re.search(r"_(\d+)(?:\.\d+)?$", name)
    return int(match.group(1)) if match else None


def delete_objects(objects: list) -> None:
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def tweak_cyclic(keep_events: int) -> None:
    keep_events = max(1, keep_events)
    event_steps = sorted(step for obj in event_objects() if (step := parse_event_step(obj.name)) is not None)
    max_step = event_steps[min(keep_events, len(event_steps)) - 1] if event_steps else None
    delete_objects([obj for obj in event_objects() if max_step is not None and (parse_event_step(obj.name) or 10**9) > max_step])

    delete_candidates = []
    for obj in trajectory_objects():
        phase_idx = parse_phase_idx(obj.name)
        if phase_idx is not None and phase_idx >= keep_events:
            delete_candidates.append(obj)
            continue
        recolor_object(obj, OUR_BLUE, emission=0.22, roughness=0.82, specular=0.28)
        set_curve_radius(obj, 0.0024)
    delete_objects(delete_candidates)


def main() -> None:
    args = parse_args()
    if args.panel == "behavior_prior":
        tweak_behavior_prior()
    elif args.panel == "conditional":
        tweak_conditional()
    elif args.panel == "safety":
        tweak_safety()
    elif args.panel == "cyclic":
        tweak_cyclic(args.cyclic_keep_events)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print(f"Saved tube tweaks for {args.panel}: {Path(bpy.data.filepath)}")


if __name__ == "__main__":
    main()
