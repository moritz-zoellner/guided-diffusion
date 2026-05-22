"""Shared setup and evaluation helpers for the complex CALVIN STL experiments."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from calvin_experiments import calvin_rollout_utils as CRU
from calvin_experiments.label_calvin_world_model import LABEL_NAMES, LABEL_THRESHOLDS, label_scene_states_for_names
from calvin_experiments.run_dynaguide_articulated_automaton import (
    filter_reset_poses,
    load_json,
    repo_path,
    resolve_existing_path,
)


DEFAULT_SCENE_CONFIG = Path("calvin_experiments/configs/robust_direction_initial.json")
LIVENESS_SELECTION_SCENE_CONFIG = Path("calvin_experiments/configs/complex_liveness_initial.json")
DEFAULT_RESET_POSE_DIR = Path("calvin_experiments/configs/dynaguide_articulated_objects/reset_poses")
DEFAULT_RESET_POSE_FILES = (
    DEFAULT_RESET_POSE_DIR / "initial_calvin_robot_states_constrained.json",
    DEFAULT_RESET_POSE_DIR / "initial_calvin_robot_states_midpoint.json",
    DEFAULT_RESET_POSE_DIR / "initial_calvin_robot_states_right_side_midpoint.json",
)
DEFAULT_COMPLEX_RESET_ROBOT_X_MIN = 0.0
DEFAULT_COMPLEX_RESET_ROBOT_X_MAX = 0.2
DEFAULT_COMPLEX_RESET_ROBOT_Y_MIN = -0.25
DEFAULT_COMPLEX_RESET_ROBOT_Y_MAX = -0.15
VIDEO_FPS = 30
GRIPPER_WIDTH_RAW_ROBOT_IDX = 6
ANGLE_TASK_GOAL_DEG = 20.0
RANDOM_SAFETY_BOX_X_MIN_RANGE = (0.15, 0.25)
RANDOM_SAFETY_BOX_Y_MIN_RANGE = (-0.15, -0.10)
RANDOM_RZZ_ANGLE_DEG_RANGE = (0.0, 20.0)
RANDOM_RZZ_TOLERANCE = 0.015


@dataclass(frozen=True)
class SafetyBox:
    x_min: float = 0.225
    x_max: float = 0.275
    y_min: float = -0.125
    y_max: float = -0.075
    margin: float = 0.02

    def normalized(self) -> "SafetyBox":
        return SafetyBox(
            min(self.x_min, self.x_max),
            max(self.x_min, self.x_max),
            min(self.y_min, self.y_max),
            max(self.y_min, self.y_max),
            float(max(self.margin, 0.0)),
        )


@dataclass(frozen=True)
class GripperOpenSpec:
    min_width: float = 0.06
    margin: float = 0.02

    def normalized(self) -> "GripperOpenSpec":
        return GripperOpenSpec(float(self.min_width), float(max(self.margin, 0.0)))


@dataclass(frozen=True)
class RzzSpec:
    angle_deg: float = ANGLE_TASK_GOAL_DEG
    axis_sign: float = -1.0
    tolerance: float = 0.02
    smooth_min_tau: float = 0.02
    tolerance_deg: Optional[float] = None

    @property
    def target(self) -> float:
        return float(self.axis_sign * np.cos(np.deg2rad(self.angle_deg)))


@dataclass(frozen=True)
class ComplexSTLSpec:
    name: str
    mode: str
    formula: str
    category: str = "liveness"
    target_names: tuple[str, ...] = ()
    stage_target_names: tuple[tuple[str, ...], ...] = ()
    first_option_names: tuple[str, ...] = ()
    middle_target_name: Optional[str] = None
    cycle_target_names: tuple[str, ...] = ()
    safety_kind: Optional[str] = None
    scene_config: Optional[Path] = None
    default_horizon: int = 400
    default_n_candidates: int = 16
    prompt: str = ""
    target_timeout_steps: int = 300
    max_target_events: int = 0
    safety_guidance_scale: Optional[float] = None
    gripper_guidance_scale: Optional[float] = None
    gradient_steps: Optional[int] = None
    step_size: Optional[float] = None
    action_reg: Optional[float] = None
    smooth_min_tau: Optional[float] = None
    gripper_guidance_mode: str = "smooth_min_all_actions"
    rzz_spec: RzzSpec = field(default_factory=RzzSpec)
    rzz_init_mode: str = "none"
    rzz_warmup_max_steps: int = 20
    rzz_warmup_tolerance: Optional[float] = None
    restack_after_warmup: bool = True

    @property
    def flattened_targets(self) -> tuple[str, ...]:
        if self.mode in {"selection_then_target", "branch_remaining"}:
            return tuple(self.first_option_names) + tuple(
                [self.middle_target_name] if self.middle_target_name is not None else []
            )
        if self.mode == "cyclic":
            return tuple(self.cycle_target_names)
        if self.stage_target_names:
            return tuple(name for stage in self.stage_target_names for name in stage)
        return tuple(self.target_names)

    @property
    def required_subgoal_count(self) -> int:
        if self.mode == "selection_then_target":
            return 2
        if self.mode == "branch_remaining":
            return len(self.first_option_names) + (1 if self.middle_target_name else 0)
        if self.mode == "cyclic":
            return int(self.max_target_events or len(self.cycle_target_names))
        if self.mode == "or":
            return 1
        return len(self.flattened_targets)


def format_goal_angle_deg(angle_deg: float) -> str:
    return f"{float(angle_deg):g}"


def angle_task_safety_term(angle_deg: float = ANGLE_TASK_GOAL_DEG) -> str:
    return f"tcp_tilt_{format_goal_angle_deg(angle_deg)}deg"


def rzz_tolerance_from_angle_tolerance_deg(angle_deg: float, axis_sign: float, tolerance_deg: float) -> float:
    center = float(axis_sign) * np.cos(np.deg2rad(float(angle_deg)))
    low_deg = max(0.0, float(angle_deg) - float(tolerance_deg))
    high_deg = min(180.0, float(angle_deg) + float(tolerance_deg))
    edge_values = np.asarray(
        [
            float(axis_sign) * np.cos(np.deg2rad(low_deg)),
            float(axis_sign) * np.cos(np.deg2rad(high_deg)),
        ],
        dtype=np.float32,
    )
    return float(np.max(np.abs(edge_values - center)))


def make_angle_stl_spec(angle_deg: float = ANGLE_TASK_GOAL_DEG) -> ComplexSTLSpec:
    angle_text = format_goal_angle_deg(angle_deg)
    return ComplexSTLSpec(
        name="angle",
        mode="target",
        category="safety",
        formula=f"F door_right AND G {angle_task_safety_term(angle_deg)}",
        target_names=("door_right",),
        safety_kind="tcp_rzz_angle",
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=400,
        default_n_candidates=16,
        safety_guidance_scale=0.3,
        gradient_steps=10,
        step_size=0.03,
        action_reg=0.01,
        smooth_min_tau=0.02,
        rzz_spec=RzzSpec(angle_deg=float(angle_deg), axis_sign=-1.0, tolerance=0.02, smooth_min_tau=0.02),
        rzz_init_mode="warmup",
        rzz_warmup_max_steps=20,
        rzz_warmup_tolerance=0.002,
        restack_after_warmup=True,
        prompt=f"move the sliding door right while keeping the TCP tilted near {angle_text} degrees",
    )


def _stable_task_seed(seed: int, task_name: str, salt: int) -> int:
    task_hash = sum((idx + 1) * ord(char) for idx, char in enumerate(str(task_name)))
    return int((int(seed) + 1) * 1009 + task_hash + int(salt)) % (2**32)


def sampled_safety_box_for_rollout(
    base_box: SafetyBox,
    *,
    seed: int,
    task_name: str,
    enabled: bool = True,
) -> tuple[SafetyBox, Dict[str, Any]]:
    """Sample a same-size avoid box by moving its bottom-left corner."""

    base = base_box.normalized()
    if not enabled:
        return base, {
            "enabled": False,
            "kind": "eef_avoid_box",
            "safety_box": asdict(base),
        }
    rng = np.random.default_rng(_stable_task_seed(seed, task_name, salt=17))
    width = float(base.x_max - base.x_min)
    height = float(base.y_max - base.y_min)
    x_min = float(rng.uniform(*RANDOM_SAFETY_BOX_X_MIN_RANGE))
    y_min = float(rng.uniform(*RANDOM_SAFETY_BOX_Y_MIN_RANGE))
    box = SafetyBox(
        x_min=x_min,
        x_max=x_min + width,
        y_min=y_min,
        y_max=y_min + height,
        margin=base.margin,
    ).normalized()
    return box, {
        "enabled": True,
        "kind": "eef_avoid_box",
        "bottom_left_x_range": list(RANDOM_SAFETY_BOX_X_MIN_RANGE),
        "bottom_left_y_range": list(RANDOM_SAFETY_BOX_Y_MIN_RANGE),
        "kept_size": {"width": width, "height": height},
        "safety_box": asdict(box),
    }


def sampled_rzz_spec_for_rollout(
    base_spec: RzzSpec,
    *,
    seed: int,
    task_name: str,
    enabled: bool = True,
    angle_deg_range: Optional[Sequence[float]] = None,
    tolerance: Optional[float] = None,
    tolerance_deg: Optional[float] = None,
) -> tuple[RzzSpec, Dict[str, Any]]:
    """Sample a TCP/gripper tilt target angle for the Rzz safety task."""

    angle_range = tuple(float(value) for value in (angle_deg_range or RANDOM_RZZ_ANGLE_DEG_RANGE))
    if len(angle_range) != 2:
        raise ValueError(f"angle_deg_range must have two values, got {angle_deg_range}")
    angle_range = (min(angle_range), max(angle_range))
    tolerance_value = float(RANDOM_RZZ_TOLERANCE if tolerance is None else tolerance)
    if not enabled:
        if tolerance_deg is not None:
            tolerance_value = rzz_tolerance_from_angle_tolerance_deg(
                base_spec.angle_deg,
                base_spec.axis_sign,
                float(tolerance_deg),
            )
        base_spec = replace(
            base_spec,
            tolerance=tolerance_value,
            tolerance_deg=None if tolerance_deg is None else float(tolerance_deg),
        )
        return base_spec, {
            "enabled": False,
            "kind": "tcp_rzz_angle",
            "angle_deg_range": list(angle_range),
            "tolerance_deg": None if tolerance_deg is None else float(tolerance_deg),
            "tolerance_rzz": float(base_spec.tolerance),
            "rzz_spec": asdict(base_spec),
            "target_rzz": float(base_spec.target),
        }
    rng = np.random.default_rng(_stable_task_seed(seed, task_name, salt=31))
    angle_deg = float(rng.uniform(*angle_range))
    if tolerance_deg is not None:
        tolerance_value = rzz_tolerance_from_angle_tolerance_deg(
            angle_deg,
            base_spec.axis_sign,
            float(tolerance_deg),
        )
    rzz_spec = replace(
        base_spec,
        angle_deg=angle_deg,
        tolerance=float(tolerance_value),
        tolerance_deg=None if tolerance_deg is None else float(tolerance_deg),
    )
    return rzz_spec, {
        "enabled": True,
        "kind": "tcp_rzz_angle",
        "angle_deg_range": list(angle_range),
        "tolerance_deg": None if tolerance_deg is None else float(tolerance_deg),
        "tolerance_rzz": float(rzz_spec.tolerance),
        "rzz_spec": asdict(rzz_spec),
        "target_rzz": float(rzz_spec.target),
    }


def spec_with_rzz_spec(spec: ComplexSTLSpec, rzz_spec: RzzSpec) -> ComplexSTLSpec:
    angle_text = format_goal_angle_deg(rzz_spec.angle_deg)
    return replace(
        spec,
        formula=f"F door_right AND G {angle_task_safety_term(rzz_spec.angle_deg)}",
        prompt=f"move the sliding door right while keeping the TCP tilted near {angle_text} degrees",
        rzz_spec=rzz_spec,
    )


def randomized_safety_context_for_rollout(
    spec: ComplexSTLSpec,
    base_safety_box: SafetyBox,
    *,
    seed: int,
    enabled: bool = True,
    rzz_angle_deg_range: Optional[Sequence[float]] = None,
    rzz_tolerance: Optional[float] = None,
    rzz_tolerance_deg: Optional[float] = None,
) -> tuple[ComplexSTLSpec, SafetyBox, Dict[str, Any]]:
    """Return rollout-specific safety geometry / Rzz target and log metadata."""

    normalized_box = base_safety_box.normalized()
    metadata: Dict[str, Any] = {
        "enabled": bool(enabled),
        "task": spec.name,
        "safety_kind": spec.safety_kind,
    }
    rollout_spec = spec
    rollout_box = normalized_box
    if spec.safety_kind == "eef_avoid_box":
        rollout_box, box_meta = sampled_safety_box_for_rollout(
            normalized_box,
            seed=seed,
            task_name=spec.name,
            enabled=enabled,
        )
        metadata["safety_box_randomization"] = box_meta
    elif spec.safety_kind in {"tcp_rzz_30deg", "tcp_rzz_angle"}:
        rzz_spec, rzz_meta = sampled_rzz_spec_for_rollout(
            spec.rzz_spec,
            seed=seed,
            task_name=spec.name,
            enabled=enabled,
            angle_deg_range=rzz_angle_deg_range,
            tolerance=rzz_tolerance,
            tolerance_deg=rzz_tolerance_deg,
        )
        rollout_spec = spec_with_rzz_spec(spec, rzz_spec)
        metadata["rzz_goal_randomization"] = rzz_meta
    return rollout_spec, rollout_box, metadata


def safety_randomization_plan(
    base_safety_box: SafetyBox,
    *,
    enabled: bool = True,
    rzz_angle_deg_range: Optional[Sequence[float]] = None,
    rzz_tolerance: Optional[float] = None,
    rzz_tolerance_deg: Optional[float] = None,
) -> Dict[str, Any]:
    box = base_safety_box.normalized()
    angle_range = tuple(float(value) for value in (rzz_angle_deg_range or RANDOM_RZZ_ANGLE_DEG_RANGE))
    return {
        "enabled": bool(enabled),
        "avoid_box_bottom_left_x_range": list(RANDOM_SAFETY_BOX_X_MIN_RANGE),
        "avoid_box_bottom_left_y_range": list(RANDOM_SAFETY_BOX_Y_MIN_RANGE),
        "avoid_box_size": {
            "width": float(box.x_max - box.x_min),
            "height": float(box.y_max - box.y_min),
        },
        "rzz_angle_deg_range": [min(angle_range), max(angle_range)],
        "rzz_tolerance": float(RANDOM_RZZ_TOLERANCE if rzz_tolerance is None else rzz_tolerance),
        "rzz_tolerance_deg": None if rzz_tolerance_deg is None else float(rzz_tolerance_deg),
    }


COMPLEX_STL_SPECS: Dict[str, ComplexSTLSpec] = {
    "single_drawer_open": ComplexSTLSpec(
        name="single_drawer_open",
        mode="target",
        category="liveness",
        formula="F drawer_open",
        target_names=("drawer_open",),
        scene_config=Path("calvin_experiments/configs/blocks_hidden.json"),
        default_horizon=220,
        default_n_candidates=16,
        prompt="open the drawer",
    ),
    "selection": ComplexSTLSpec(
        name="selection",
        mode="or",
        category="liveness",
        formula="F(door_left OR switch_off OR button_pressed)",
        target_names=("door_left", "switch_off", "button_pressed"),
        scene_config=LIVENESS_SELECTION_SCENE_CONFIG,
        default_horizon=300,
        default_n_candidates=16,
        prompt="move the sliding door left, turn off the lightbulb with the switch, or press the button",
    ),
    "unordered": ComplexSTLSpec(
        name="unordered",
        mode="and",
        category="liveness",
        formula="F door_left AND F switch_off AND F button_pressed",
        target_names=("door_left", "switch_off", "button_pressed"),
        scene_config=LIVENESS_SELECTION_SCENE_CONFIG,
        default_horizon=500,
        default_n_candidates=16,
        prompt="move the sliding door left, turn off the lightbulb with the switch, and press the button in any order",
    ),
    "conditional": ComplexSTLSpec(
        name="conditional",
        mode="ordered_stage",
        category="liveness",
        formula="F drawer_closed AND (!drawer_closed U (button_pressed AND switch_off))",
        stage_target_names=(("button_pressed", "switch_off"), ("drawer_closed",)),
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=450,
        default_n_candidates=16,
        prompt="close the drawer, but press the button and turn off the lightbulb with the switch before that",
    ),
    "chained": ComplexSTLSpec(
        name="chained",
        mode="chain",
        category="liveness",
        formula="F(button_pressed ; door_right ; switch_off ; drawer_closed)",
        target_names=("button_pressed", "door_right", "switch_off", "drawer_closed"),
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=700,
        default_n_candidates=16,
        prompt="press the button, then move the sliding door right, then turn off the lightbulb with the switch, then close the drawer",
    ),
    "branched": ComplexSTLSpec(
        name="branched",
        mode="branch_remaining",
        category="liveness",
        formula=(
            "F(button_pressed ; drawer_closed ; switch_off) OR "
            "F(switch_off ; drawer_closed ; button_pressed)"
        ),
        first_option_names=("button_pressed", "switch_off"),
        middle_target_name="drawer_closed",
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=500,
        default_n_candidates=32,
        prompt="press the button or turn off the lightbulb with the switch, then close the drawer, then do the remaining option from the first selection",
    ),
    "cyclic": ComplexSTLSpec(
        name="cyclic",
        mode="cyclic",
        category="liveness",
        formula="G F(drawer_open ; switch_on ; drawer_closed ; switch_off), finite cyclic approximation",
        cycle_target_names=("drawer_open", "switch_on", "drawer_closed", "switch_off"),
        scene_config=Path("calvin_experiments/configs/blocks_hidden.json"),
        default_horizon=2400,
        default_n_candidates=32,
        target_timeout_steps=300,
        max_target_events=8,
        prompt="repeatedly open the drawer, turn the switch on, close the drawer, and turn the switch off",
    ),
    "region": ComplexSTLSpec(
        name="region",
        mode="target",
        category="safety",
        formula="F switch_off AND G avoid_unsafe_square",
        target_names=("switch_off",),
        safety_kind="eef_avoid_box",
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=220,
        default_n_candidates=16,
        safety_guidance_scale=0.5,
        gradient_steps=10,
        step_size=0.03,
        action_reg=0.05,
        smooth_min_tau=0.02,
        prompt="turn off the lightbulb with the switch while keeping the robot arm out of the unsafe square",
    ),
    "angle": make_angle_stl_spec(),
    "gripper": ComplexSTLSpec(
        name="gripper",
        mode="target",
        category="safety",
        formula="F drawer_closed AND G gripper_open",
        target_names=("drawer_closed",),
        safety_kind="gripper_open",
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=200,
        default_n_candidates=32,
        gripper_guidance_scale=100.0,
        gradient_steps=50,
        step_size=0.10,
        action_reg=0.001,
        smooth_min_tau=0.01,
        gripper_guidance_mode="world_model_gripper_value_only",
        prompt="close the drawer while keeping the gripper open",
    ),
    # Backwards-compatible aliases for older local commands and plots.
    "F_a_or_F_b": ComplexSTLSpec(
        name="F_a_or_F_b",
        mode="or",
        category="legacy",
        formula="F button_pressed OR F switch_off",
        target_names=("button_pressed", "switch_off"),
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=250,
        default_n_candidates=16,
        prompt="press the button or turn off the lightbulb with the switch",
    ),
    "F_switch_G_safety": ComplexSTLSpec(
        name="F_switch_G_safety",
        mode="target",
        category="legacy",
        formula="F switch_off AND G avoid_unsafe_square",
        target_names=("switch_off",),
        safety_kind="eef_avoid_box",
        scene_config=DEFAULT_SCENE_CONFIG,
        default_horizon=220,
        default_n_candidates=16,
        prompt="turn off the lightbulb with the switch while keeping the robot arm out of the unsafe square",
    ),
}
COMPLEX_STL_SPECS.update(
    {
        # Backwards-compatible aliases for old local commands.
        "selection_button_or_switch_then_drawer": COMPLEX_STL_SPECS["selection"],
        "unordered_button_and_switch_then_drawer": COMPLEX_STL_SPECS["conditional"],
        "chained_drawer_switch_button_door": COMPLEX_STL_SPECS["chained"],
        "branched_button_or_switch_drawer_remaining": COMPLEX_STL_SPECS["branched"],
        "cyclic_drawer_switch": COMPLEX_STL_SPECS["cyclic"],
        "safety_region_switch": COMPLEX_STL_SPECS["region"],
        "safety_rzz_door": COMPLEX_STL_SPECS["angle"],
        "safety_gripper_drawer": COMPLEX_STL_SPECS["gripper"],
    }
)
TASK_ORDER = (
    "selection",
    "unordered",
    "conditional",
    "chained",
    "branched",
    "cyclic",
    "region",
    "angle",
    "gripper",
)


def labels_for_scene(scene: Sequence[float]) -> np.ndarray:
    return label_scene_states_for_names(
        np.asarray(scene, dtype=np.float32)[None, :],
        LABEL_NAMES,
        LABEL_THRESHOLDS,
    )[0].astype(np.float32)


class SpecMonitor:
    def __init__(self, spec: ComplexSTLSpec):
        self.spec = spec
        self.events: list[Dict[str, Any]] = []
        self.violations: list[Dict[str, Any]] = []
        self.done = False
        self.pos = 0
        self.stage_pos = 0
        self.achieved = set()
        self.stage_achieved = [set() for _ in spec.stage_target_names]
        self.violation_keys = set()

    def idx(self, name: str) -> int:
        if name not in LABEL_NAMES:
            raise ValueError(f"Unknown label {name}; labels={LABEL_NAMES}")
        return LABEL_NAMES.index(name)

    def achieved_names(self) -> list[str]:
        names = [event["target_name"] for event in self.events]
        out = []
        seen = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def sync(self, label: np.ndarray, step: int) -> bool:
        if self.spec.mode == "or":
            return self._sync_or(label, step)
        if self.spec.mode == "selection_then_target":
            return self._sync_selection_then_target(label, step)
        if self.spec.mode == "and":
            return self._sync_and(label, step)
        if self.spec.mode == "chain":
            return self._sync_chain(label, step)
        if self.spec.mode == "branch_remaining":
            return self._sync_branch_remaining(label, step)
        if self.spec.mode == "cyclic":
            return self._sync_cyclic(label, step)
        if self.spec.mode == "ordered_stage":
            return self._sync_ordered_stage(label, step)
        if self.spec.mode == "target":
            return self._sync_target(label, step)
        raise ValueError(f"Unsupported mode: {self.spec.mode}")

    def _sync_or(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        for name in self.spec.target_names:
            idx = self.idx(name)
            if float(label[idx]) > 0.5:
                self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
                self.done = True
                return True
        return False

    def _sync_selection_then_target(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        advanced = False
        if self.pos == 0:
            for name in self.spec.first_option_names:
                idx = self.idx(name)
                if float(label[idx]) > 0.5:
                    self.events.append({"step": int(step), "role": "first_or", "target_idx": int(idx), "target_name": name})
                    self.pos = 1
                    advanced = True
                    break
        if self.pos == 1 and self.spec.middle_target_name is not None:
            idx = self.idx(self.spec.middle_target_name)
            if float(label[idx]) > 0.5:
                self.events.append({"step": int(step), "role": "then", "target_idx": int(idx), "target_name": self.spec.middle_target_name})
                self.pos = 2
                self.done = True
                advanced = True
        return advanced

    def _sync_and(self, label: np.ndarray, step: int) -> bool:
        advanced = False
        for name in self.spec.target_names:
            idx = self.idx(name)
            if idx not in self.achieved and float(label[idx]) > 0.5:
                self.achieved.add(idx)
                self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
                advanced = True
        self.done = len(self.achieved) == len(self.spec.target_names)
        return advanced

    def _sync_chain(self, label: np.ndarray, step: int) -> bool:
        advanced = False
        while self.pos < len(self.spec.target_names):
            name = self.spec.target_names[self.pos]
            idx = self.idx(name)
            if float(label[idx]) <= 0.5:
                break
            self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
            self.pos += 1
            advanced = True
        self.done = self.pos >= len(self.spec.target_names)
        return advanced

    def _sync_branch_remaining(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        advanced = False
        if self.pos == 0:
            for name in self.spec.first_option_names:
                idx = self.idx(name)
                if float(label[idx]) > 0.5:
                    self.achieved.add(idx)
                    self.events.append({"step": int(step), "role": "first_or", "target_idx": int(idx), "target_name": name})
                    self.pos = 1
                    advanced = True
                    break
        if self.pos == 1 and self.spec.middle_target_name is not None:
            idx = self.idx(self.spec.middle_target_name)
            if float(label[idx]) > 0.5:
                self.events.append({"step": int(step), "role": "middle", "target_idx": int(idx), "target_name": self.spec.middle_target_name})
                self.pos = 2
                advanced = True
        if self.pos == 2:
            remaining = [name for name in self.spec.first_option_names if self.idx(name) not in self.achieved]
            if not remaining:
                self.done = True
                return advanced
            idx = self.idx(remaining[0])
            if float(label[idx]) > 0.5:
                self.events.append({"step": int(step), "role": "remaining_first", "target_idx": int(idx), "target_name": remaining[0]})
                self.done = True
                advanced = True
        return advanced

    def _sync_cyclic(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        advanced = False
        max_events = int(self.spec.max_target_events or len(self.spec.cycle_target_names))
        while len(self.events) < max_events:
            name = self.spec.cycle_target_names[self.pos % len(self.spec.cycle_target_names)]
            idx = self.idx(name)
            if float(label[idx]) <= 0.5:
                break
            self.events.append(
                {
                    "step": int(step),
                    "event_idx": int(len(self.events)),
                    "cycle_idx": int(self.pos // len(self.spec.cycle_target_names)),
                    "cycle_pos": int(self.pos % len(self.spec.cycle_target_names)),
                    "target_idx": int(idx),
                    "target_name": name,
                }
            )
            self.pos += 1
            advanced = True
        self.done = len(self.events) >= max_events
        return advanced

    def _record_future_stage_violations(self, label: np.ndarray, step: int) -> None:
        for future_stage_idx in range(max(1, self.stage_pos + 1), len(self.spec.stage_target_names)):
            for name in self.spec.stage_target_names[future_stage_idx]:
                idx = self.idx(name)
                key = (future_stage_idx, idx)
                if key not in self.violation_keys and float(label[idx]) > 0.5:
                    self.violation_keys.add(key)
                    self.violations.append(
                        {
                            "step": int(step),
                            "stage_idx": int(future_stage_idx),
                            "target_idx": int(idx),
                            "target_name": name,
                            "message": "future-stage target became true before prior stages completed",
                        }
                    )

    def _sync_ordered_stage(self, label: np.ndarray, step: int) -> bool:
        advanced = False
        if self.stage_pos < len(self.spec.stage_target_names):
            self._record_future_stage_violations(label, step)
        while self.stage_pos < len(self.spec.stage_target_names):
            stage = self.spec.stage_target_names[self.stage_pos]
            for name in stage:
                idx = self.idx(name)
                if idx not in self.stage_achieved[self.stage_pos] and float(label[idx]) > 0.5:
                    self.stage_achieved[self.stage_pos].add(idx)
                    self.events.append(
                        {
                            "step": int(step),
                            "stage_idx": int(self.stage_pos),
                            "target_idx": int(idx),
                            "target_name": name,
                        }
                    )
                    advanced = True
            if len(self.stage_achieved[self.stage_pos]) == len(stage):
                self.stage_pos += 1
                advanced = True
                continue
            break
        self.done = self.stage_pos >= len(self.spec.stage_target_names)
        return advanced

    def _sync_target(self, label: np.ndarray, step: int) -> bool:
        if self.done:
            return False
        name = self.spec.target_names[0]
        idx = self.idx(name)
        if float(label[idx]) > 0.5:
            self.events.append({"step": int(step), "target_idx": int(idx), "target_name": name})
            self.done = True
            return True
        return False


def unique_run_dir(output_root: Path, run_name: str) -> Path:
    candidate = output_root / run_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        suffixed = output_root / f"{run_name}_{suffix:02d}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def resolve_reset_pose_paths(raw_paths: Sequence[Path | str]) -> list[Path]:
    return [
        resolve_existing_path(path, base_dir=repo_path(DEFAULT_RESET_POSE_DIR))
        for path in raw_paths
    ]


def load_reset_pose_pool(paths: Sequence[Path]) -> tuple[list[np.ndarray], Dict[str, Any]]:
    poses: list[np.ndarray] = []
    sources = []
    for path in paths:
        payload = load_json(path)
        states = [np.asarray(item, dtype=np.float32) for item in payload["robot_states"]]
        poses.extend(states)
        sources.append({"path": str(path), "count": len(states)})
    return poses, {"sources": sources, "total_count": len(poses)}


def filter_complex_reset_poses(
    poses: Sequence[np.ndarray],
    *,
    robot_x_min: Optional[float],
    robot_x_max: Optional[float],
    robot_y_min: Optional[float],
    robot_y_max: Optional[float],
    switch_clearance: Optional[float],
) -> tuple[list[np.ndarray], Dict[str, Any]]:
    base_filtered, filter_meta = filter_reset_poses(
        list(poses),
        robot_y_min=robot_y_min,
        robot_y_max=robot_y_max,
        switch_clearance=switch_clearance,
    )
    filtered = list(base_filtered or [])
    if robot_x_min is not None:
        filtered = [pose for pose in filtered if float(pose[0]) >= float(robot_x_min)]
    after_robot_x_min = len(filtered)
    if robot_x_max is not None:
        filtered = [pose for pose in filtered if float(pose[0]) <= float(robot_x_max)]
    after_robot_x_max = len(filtered)
    if not filtered:
        raise ValueError(
            "Complex-STL reset-pose XY frame removed every pose "
            f"(x_min={robot_x_min}, x_max={robot_x_max}, y_min={robot_y_min}, y_max={robot_y_max}, "
            f"switch_clearance={switch_clearance})"
        )
    frame_enabled = any(
        value is not None
        for value in (robot_x_min, robot_x_max, robot_y_min, robot_y_max, switch_clearance)
    )
    return filtered, {
        **filter_meta,
        "enabled": bool(frame_enabled),
        "after_robot_x_min": int(after_robot_x_min),
        "after_robot_x_max": int(after_robot_x_max),
        "filtered_count": int(len(filtered)),
        "robot_x_min": None if robot_x_min is None else float(robot_x_min),
        "robot_x_max": None if robot_x_max is None else float(robot_x_max),
        "xy_frame": {
            "x_min": None if robot_x_min is None else float(robot_x_min),
            "x_max": None if robot_x_max is None else float(robot_x_max),
            "y_min": None if robot_y_min is None else float(robot_y_min),
            "y_max": None if robot_y_max is None else float(robot_y_max),
        },
    }


def make_fixed_scene_robot(
    base_env_state: Dict[str, np.ndarray],
    scene_config_path: Path,
    reset_poses: Sequence[np.ndarray],
    fixed_reset_pose_index: Optional[int],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], bool]:
    fixed_scene, fixed_robot, scene_cfg = CRU.fixed_scene_robot_from_config(base_env_state, scene_config_path)
    if reset_poses:
        if fixed_reset_pose_index is None:
            selected_pose = random.choice(list(reset_poses))
        else:
            selected_pose = list(reset_poses)[int(fixed_reset_pose_index) % len(reset_poses)]
        fixed_robot = np.asarray(selected_pose, dtype=np.float32).copy()
        return fixed_scene, fixed_robot, scene_cfg, True
    return fixed_scene, fixed_robot, scene_cfg, False


def signed_distance_to_box_np(xy: np.ndarray, safety_box: SafetyBox) -> np.ndarray:
    box = safety_box.normalized()
    xy = np.asarray(xy, dtype=np.float32)
    center = np.asarray([(box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2], dtype=np.float32)
    half = np.asarray([(box.x_max - box.x_min) / 2, (box.y_max - box.y_min) / 2], dtype=np.float32)
    q = np.abs(xy - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside


def path_enters_safety_box(eef_xy: np.ndarray, safety_box: SafetyBox) -> bool:
    return bool(np.any(signed_distance_to_box_np(eef_xy, safety_box) <= 0.0))


def rzz_from_euler_xyz_np(euler_xyz: np.ndarray) -> np.ndarray:
    euler_xyz = np.asarray(euler_xyz, dtype=np.float32)
    return np.cos(euler_xyz[..., 1]) * np.cos(euler_xyz[..., 0])


def rzz_to_tilt_angle_deg_np(rzz: np.ndarray, rzz_spec: RzzSpec) -> np.ndarray:
    signed_cos = float(rzz_spec.axis_sign) * np.asarray(rzz, dtype=np.float32)
    return np.rad2deg(np.arccos(np.clip(signed_cos, -1.0, 1.0)))


def rzz_angle_error_deg_np(rzz: np.ndarray, rzz_spec: RzzSpec) -> np.ndarray:
    return np.abs(rzz_to_tilt_angle_deg_np(rzz, rzz_spec) - float(rzz_spec.angle_deg))


def rzz_tolerance_band_deg_np(rzz_spec: RzzSpec) -> tuple[float, float]:
    if rzz_spec.tolerance_deg is not None:
        return (
            float(max(0.0, float(rzz_spec.angle_deg) - float(rzz_spec.tolerance_deg))),
            float(min(180.0, float(rzz_spec.angle_deg) + float(rzz_spec.tolerance_deg))),
        )
    angles = rzz_to_tilt_angle_deg_np(
        np.asarray([rzz_spec.target - rzz_spec.tolerance, rzz_spec.target + rzz_spec.tolerance], dtype=np.float32),
        rzz_spec,
    )
    return float(np.min(angles)), float(np.max(angles))


def save_trace_without_video(rollout: Dict[str, Any], output_dir: Path, rollout_tag: str) -> None:
    rollout_dir = output_dir / rollout_tag
    rollout_dir.mkdir(parents=True, exist_ok=True)
    trace_path = rollout_dir / "rollout_trace.npz"
    scene_snapshot_path = rollout_dir / "scene_snapshot.json"
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
        "rollout_seed": np.asarray(rollout["seed"], dtype=np.int32),
        "scene_config": np.asarray(rollout["scene_config"]),
        "initial_label": np.asarray(rollout["initial_label"], dtype=np.int32),
        "final_label": np.asarray(rollout["final_label"], dtype=np.int32),
        "labels_over_time": np.asarray(rollout["labels_over_time"], dtype=np.int32),
        "pre_settle_label": np.asarray(rollout["pre_settle_label"], dtype=np.int32),
        "settle_scene_states": np.asarray(rollout["settle_scene_states"], dtype=np.float32),
        "settle_robot_states": np.asarray(rollout["settle_robot_states"], dtype=np.float32),
        "settle_action": np.asarray(rollout["settle_action"], dtype=np.float32),
    }
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
    CRU.save_scene_snapshot(rollout["scene_snapshot"], scene_snapshot_path)
    rollout["video"] = None
    rollout["trace"] = trace_path
    rollout["scene_snapshot_path"] = scene_snapshot_path
    rollout["rollout_dir"] = rollout_dir


def save_rollout_diagnostics(rollout: Dict[str, Any]) -> None:
    rollout_dir = Path(rollout["rollout_dir"])
    diagnostics_path = rollout_dir / "diagnostics.npz"
    diagnostics = {
        "labels_over_time": np.asarray(rollout["labels_over_time"], dtype=np.int32),
        "gripper_width": np.asarray(rollout.get("gripper_width", []), dtype=np.float32),
    }
    if "tcp_rzz" in rollout:
        diagnostics["tcp_rzz"] = np.asarray(rollout["tcp_rzz"], dtype=np.float32)
    if "tcp_tilt_angle_deg" in rollout:
        diagnostics["tcp_tilt_angle_deg"] = np.asarray(rollout["tcp_tilt_angle_deg"], dtype=np.float32)
    if rollout.get("safety_distances") is not None:
        diagnostics["safety_distances"] = np.asarray(rollout["safety_distances"], dtype=np.float32)
    np.savez_compressed(diagnostics_path, **diagnostics)
    rollout["diagnostics"] = diagnostics_path


def rollout_summary_payload(rollout: Dict[str, Any], policy: Optional[str] = None) -> Dict[str, Any]:
    return {
        "task": rollout["task"],
        "formula": rollout["formula"],
        "seed": int(rollout["seed"]),
        "policy": policy or rollout["policy"],
        "scene_config": rollout["scene_config"],
        "liveness_satisfied": bool(rollout["liveness_satisfied"]),
        "safety_satisfied": rollout["safety_satisfied"],
        "stl_satisfied": bool(rollout["stl_satisfied"]),
        "success": bool(rollout["success"]),
        "subgoal_completion_rate": float(rollout["subgoal_completion_rate"]),
        "completed_subgoals": int(rollout["completed_subgoals"]),
        "total_subgoals": int(rollout["total_subgoals"]),
        "target_events": rollout["target_events"],
        "order_violation": bool(rollout.get("order_violation", False)),
        "order_violations": rollout.get("order_violations", []),
        "safety_kind": rollout.get("safety_kind"),
        "safety_metrics": rollout.get("safety_metrics", {}),
        "safety_box": rollout.get("safety_box"),
        "rzz_spec": rollout.get("rzz_spec"),
        "safety_randomization": rollout.get("safety_randomization", {"enabled": False}),
        "behavior": rollout["behavior"],
        "first_behavior": rollout["first_behavior"],
        "first_behavior_step": int(rollout["first_behavior_step"]),
        "behavior_step": int(rollout["behavior_step"]),
        "termination_step": int(rollout["termination_step"]),
        "termination_reason": rollout["termination_reason"],
        "env_done_step": int(rollout["env_done_step"]),
        "return": float(rollout["return"]),
        "initial_label": rollout["initial_label"],
        "final_label": rollout["final_label"],
        "video": None if rollout.get("video") is None else str(rollout["video"]),
        "trace": str(rollout["trace"]),
        "diagnostics": str(rollout["diagnostics"]),
        "topdown_plot": str(Path(rollout["rollout_dir"]) / "rollout_xy.png"),
        "reset_robot_from_pose_file": bool(rollout["reset_robot_from_pose_file"]),
        "fixed_reset_pose_index": rollout.get("fixed_reset_pose_index"),
        "reset_pose_filter": rollout["reset_pose_filter"],
        "settle_steps": int(rollout["settle_steps"]),
        "settle_action": rollout["settle_action"],
        "pre_settle_label": rollout["pre_settle_label"],
        "settle_metrics": rollout["settle_metrics"],
        "rzz_warmup": rollout.get("rzz_warmup", {"enabled": False}),
        "reset_rzz": rollout.get("reset_rzz"),
        "reset_tcp_tilt_angle_deg": rollout.get("reset_tcp_tilt_angle_deg"),
        "records": rollout["records"],
    }


def evaluate_subgoals(spec: ComplexSTLSpec, events: Sequence[Dict[str, Any]]) -> tuple[int, int, float]:
    total = spec.required_subgoal_count
    if spec.mode == "or":
        completed = 1 if events else 0
    elif spec.mode == "cyclic":
        completed = min(len(events), total)
    else:
        target_names = set(spec.flattened_targets)
        completed = len({event["target_name"] for event in events if event["target_name"] in target_names})
    rate = float(completed / total) if total else 0.0
    return int(completed), int(total), rate


def evaluate_safety(
    spec: ComplexSTLSpec,
    robot_states: np.ndarray,
    eef_xy: np.ndarray,
    safety_box: SafetyBox,
    gripper_spec: GripperOpenSpec,
) -> tuple[Optional[bool], Dict[str, Any], Optional[np.ndarray]]:
    if spec.safety_kind == "eef_avoid_box":
        distances = signed_distance_to_box_np(eef_xy, safety_box)
        violation = bool(np.any(distances <= 0.0))
        metrics = {
            "kind": "eef_avoid_box",
            "safety_box": asdict(safety_box.normalized()),
            "violation": violation,
            "min_signed_distance": float(np.min(distances)),
        }
        return (not violation), metrics, distances
    if spec.safety_kind == "gripper_open":
        widths = np.asarray(robot_states, dtype=np.float32)[:, GRIPPER_WIDTH_RAW_ROBOT_IDX]
        spec_norm = gripper_spec.normalized()
        violation = bool(np.any(widths < spec_norm.min_width))
        metrics = {
            "kind": "gripper_open",
            "gripper_spec": asdict(spec_norm),
            "violation": violation,
            "min_gripper_width": float(np.min(widths)),
        }
        return (not violation), metrics, None
    if spec.safety_kind in {"tcp_rzz_30deg", "tcp_rzz_angle"}:
        robot_states = np.asarray(robot_states, dtype=np.float32)
        rzz_spec = spec.rzz_spec
        rzz = rzz_from_euler_xyz_np(robot_states[:, 3:6])
        abs_error = np.abs(rzz - float(rzz_spec.target))
        angle_deg = rzz_to_tilt_angle_deg_np(rzz, rzz_spec)
        angle_error = np.abs(angle_deg - float(rzz_spec.angle_deg))
        if rzz_spec.tolerance_deg is not None:
            violation = bool(np.any(angle_error > float(rzz_spec.tolerance_deg)))
        else:
            violation = bool(np.any(abs_error > float(rzz_spec.tolerance)))
        rzz_spec_payload = asdict(rzz_spec)
        rzz_spec_payload.update(
            {
                "target": float(rzz_spec.target),
                "angle_tolerance_band_deg": list(rzz_tolerance_band_deg_np(rzz_spec)),
            }
        )
        metrics = {
            "kind": spec.safety_kind,
            "rzz_spec": rzz_spec_payload,
            "violation": violation,
            "max_abs_rzz_error": float(np.max(abs_error)),
            "mean_abs_rzz_error": float(np.mean(abs_error)),
            "max_abs_angle_error_deg": float(np.max(angle_error)),
            "mean_abs_angle_error_deg": float(np.mean(angle_error)),
            "min_angle_deg": float(np.min(angle_deg)),
            "max_angle_deg": float(np.max(angle_deg)),
        }
        return (not violation), metrics, abs_error
    return None, {}, None


def _angle_trace_from_rollout_or_disk(
    rollout: Optional[Dict[str, Any]],
    rollout_dir: Path,
    rzz_spec: RzzSpec,
) -> Optional[np.ndarray]:
    if rollout is not None and rollout.get("tcp_tilt_angle_deg") is not None:
        values = np.asarray(rollout["tcp_tilt_angle_deg"], dtype=np.float32)
        if values.size:
            return values
    trace_path = rollout_dir / "rollout_trace.npz"
    if not trace_path.exists():
        return None
    with np.load(trace_path, allow_pickle=True) as trace:
        if "tcp_tilt_angle_deg" in trace:
            values = np.asarray(trace["tcp_tilt_angle_deg"], dtype=np.float32)
            return values if values.size else None
        if "robot_states" in trace:
            robot_states = np.asarray(trace["robot_states"], dtype=np.float32)
            if robot_states.ndim == 2 and robot_states.shape[1] >= 6:
                rzz = rzz_from_euler_xyz_np(robot_states[:, 3:6])
                return np.asarray(rzz_to_tilt_angle_deg_np(rzz, rzz_spec), dtype=np.float32)
    return None


def rzz_spec_from_rollout(rollout: Dict[str, Any], fallback: Optional[RzzSpec] = None) -> RzzSpec:
    base = fallback or RzzSpec()
    payload = rollout.get("rzz_spec")
    if not isinstance(payload, dict):
        payload = rollout.get("safety_metrics", {}).get("rzz_spec")
    if not isinstance(payload, dict):
        return base
    return RzzSpec(
        angle_deg=float(payload.get("angle_deg", base.angle_deg)),
        axis_sign=float(payload.get("axis_sign", base.axis_sign)),
        tolerance=float(payload.get("tolerance", base.tolerance)),
        smooth_min_tau=float(payload.get("smooth_min_tau", base.smooth_min_tau)),
        tolerance_deg=None if payload.get("tolerance_deg") is None else float(payload.get("tolerance_deg")),
    )


def write_rollout_rzz_angle_diagnostic_plot(rollout: Dict[str, Any]) -> Optional[Path]:
    """Write one TCP-tilt diagnostic plot inside an angle-task rollout folder."""

    if rollout.get("safety_kind") not in {"tcp_rzz_30deg", "tcp_rzz_angle"}:
        return None
    rollout_dir_raw = rollout.get("rollout_dir")
    if rollout_dir_raw is None:
        return None
    rollout_dir = Path(rollout_dir_raw)
    rzz_spec = rzz_spec_from_rollout(rollout)
    values = _angle_trace_from_rollout_or_disk(rollout, rollout_dir, rzz_spec)
    if values is None or not values.size:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    band_low, band_high = rzz_tolerance_band_deg_np(rzz_spec)
    steps = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.axhspan(band_low, band_high, color="#dbeafe", alpha=0.9, label=f"goal band ({band_low:.1f}-{band_high:.1f} deg)")
    ax.axhline(float(rzz_spec.angle_deg), color="#1d4ed8", linewidth=1.2, linestyle="--", label=f"goal {rzz_spec.angle_deg:.1f} deg")
    ax.plot(steps, values, color="#111827", linewidth=1.8, label="rollout")
    ax.set_title(f"{rollout.get('task', 'angle')}: TCP tilt angle, seed {rollout.get('seed', 'unknown')}")
    ax.set_xlabel("policy step")
    ax.set_ylabel("TCP tilt angle (deg)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    y_min = float(np.nanmin([np.nanmin(values), band_low, float(rzz_spec.angle_deg)]))
    y_max = float(np.nanmax([np.nanmax(values), band_high, float(rzz_spec.angle_deg)]))
    pad = max(1.0, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    fig.tight_layout()
    out_path = rollout_dir / "angle_over_time.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = rollout_dir / "angle_over_time_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "tcp_tilt_angle_deg", "goal_angle_deg", "band_low_deg", "band_high_deg"])
        for step, value in enumerate(values):
            writer.writerow([step, f"{float(value):.6f}", f"{float(rzz_spec.angle_deg):.6f}", f"{band_low:.6f}", f"{band_high:.6f}"])
    return out_path


def write_rzz_angle_diagnostic_plot(
    task_dir: Path,
    spec: ComplexSTLSpec,
    rollouts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """Plot per-rollout TCP tilt traces with the target angle band."""

    if spec.safety_kind not in {"tcp_rzz_30deg", "tcp_rzz_angle"}:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    task_dir = Path(task_dir)
    rollout_by_dir = {}
    if rollouts is not None:
        rollout_by_dir = {
            Path(rollout["rollout_dir"]).resolve(): rollout
            for rollout in rollouts
            if rollout.get("rollout_dir") is not None
        }
        rollout_dirs = [Path(rollout["rollout_dir"]) for rollout in rollouts if rollout.get("rollout_dir") is not None]
    else:
        rollout_dirs = sorted(path for path in task_dir.glob("rollout_*") if path.is_dir())

    traces: list[tuple[str, np.ndarray, RzzSpec]] = []
    for rollout_dir in rollout_dirs:
        rollout = rollout_by_dir.get(Path(rollout_dir).resolve())
        rollout_rzz_spec = spec.rzz_spec
        if rollout is not None and rollout.get("rzz_spec") is not None:
            rzz_payload = rollout["rzz_spec"]
            rollout_rzz_spec = RzzSpec(
                angle_deg=float(rzz_payload.get("angle_deg", spec.rzz_spec.angle_deg)),
                axis_sign=float(rzz_payload.get("axis_sign", spec.rzz_spec.axis_sign)),
                tolerance=float(rzz_payload.get("tolerance", spec.rzz_spec.tolerance)),
                smooth_min_tau=float(rzz_payload.get("smooth_min_tau", spec.rzz_spec.smooth_min_tau)),
                tolerance_deg=None if rzz_payload.get("tolerance_deg") is None else float(rzz_payload.get("tolerance_deg")),
            )
        values = _angle_trace_from_rollout_or_disk(rollout, Path(rollout_dir), rollout_rzz_spec)
        if values is not None and values.size:
            traces.append((Path(rollout_dir).name, values, rollout_rzz_spec))

    if not traces:
        return None

    targets = np.asarray([float(rzz_spec.angle_deg) for _, _, rzz_spec in traces], dtype=np.float32)
    bands = np.asarray([rzz_tolerance_band_deg_np(rzz_spec) for _, _, rzz_spec in traces], dtype=np.float32)
    target = float(spec.rzz_spec.angle_deg)
    band_low = float(np.min(bands[:, 0]))
    band_high = float(np.max(bands[:, 1]))
    max_len = max(len(values) for _, values, _ in traces)
    stacked = np.full((len(traces), max_len), np.nan, dtype=np.float32)
    for idx, (_, values, _) in enumerate(traces):
        stacked[idx, : len(values)] = values
    steps = np.arange(max_len)
    mean_trace = np.nanmean(stacked, axis=0)
    min_trace = np.nanmin(stacked, axis=0)
    max_trace = np.nanmax(stacked, axis=0)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    variable_targets = bool(np.max(targets) - np.min(targets) > 1e-4)
    band_label = (
        f"sampled goal band envelope ({band_low:.1f}-{band_high:.1f} deg)"
        if variable_targets
        else f"goal band ({band_low:.1f}-{band_high:.1f} deg)"
    )
    ax.axhspan(band_low, band_high, color="#dbeafe", alpha=0.85, label=band_label)
    if variable_targets:
        for target_i in targets:
            ax.axhline(float(target_i), color="#1d4ed8", linewidth=0.6, linestyle="--", alpha=0.25)
        ax.axhline(float(np.mean(targets)), color="#1d4ed8", linewidth=1.2, linestyle="--", label="mean sampled goal")
    else:
        ax.axhline(target, color="#1d4ed8", linewidth=1.2, linestyle="--", label=f"goal {target:.0f} deg")
    for name, values, _ in traces:
        ax.plot(np.arange(len(values)), values, color="#94a3b8", linewidth=0.8, alpha=0.5)
    ax.fill_between(steps, min_trace, max_trace, color="#cbd5e1", alpha=0.35, label="rollout min/max")
    ax.plot(steps, mean_trace, color="#111827", linewidth=1.8, label="mean")
    ax.set_title(f"{spec.name}: TCP tilt angle over time")
    ax.set_xlabel("policy step")
    ax.set_ylabel("TCP tilt angle (deg)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    y_min = float(np.nanmin([np.nanmin(stacked), band_low, float(np.min(targets))]))
    y_max = float(np.nanmax([np.nanmax(stacked), band_high, float(np.max(targets))]))
    pad = max(1.0, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    fig.tight_layout()
    out_path = task_dir / "angle_over_time.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = task_dir / "angle_over_time_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rollout", "step", "tcp_tilt_angle_deg", "goal_angle_deg", "band_low_deg", "band_high_deg"])
        for name, values, rzz_spec in traces:
            target_i = float(rzz_spec.angle_deg)
            band_low_i, band_high_i = rzz_tolerance_band_deg_np(rzz_spec)
            for step, value in enumerate(values):
                writer.writerow([name, step, f"{float(value):.6f}", f"{target_i:.6f}", f"{band_low_i:.6f}", f"{band_high_i:.6f}"])
    return out_path


def task_summary(spec: ComplexSTLSpec, rollouts: Sequence[Dict[str, Any]], n_candidates: int, horizon: int) -> Dict[str, Any]:
    behavior_counts = Counter(rollout["behavior"] for rollout in rollouts)
    first_behavior_counts = Counter(rollout["first_behavior"] for rollout in rollouts)
    event_patterns = Counter(tuple(event["target_name"] for event in rollout["target_events"]) for rollout in rollouts)
    liveness_count = sum(1 for rollout in rollouts if rollout["liveness_satisfied"])
    stl_count = sum(1 for rollout in rollouts if rollout["stl_satisfied"])
    safety_values = [rollout["safety_satisfied"] for rollout in rollouts if rollout["safety_satisfied"] is not None]
    safety_count = sum(1 for value in safety_values if value)
    order_violation_count = sum(1 for rollout in rollouts if rollout.get("order_violation", False))
    return {
        "task": spec.name,
        "formula": spec.formula,
        "mode": spec.mode,
        "safety_kind": spec.safety_kind,
        "n_rollouts": len(rollouts),
        "n_candidates": int(n_candidates),
        "horizon": int(horizon),
        "liveness_satisfied_count": int(liveness_count),
        "liveness_satisfaction_rate": float(liveness_count / len(rollouts)) if rollouts else 0.0,
        "safety_satisfied_count": None if not safety_values else int(safety_count),
        "safety_satisfaction_rate": None if not safety_values else float(safety_count / len(safety_values)),
        "stl_satisfied_count": int(stl_count),
        "stl_satisfaction_rate": float(stl_count / len(rollouts)) if rollouts else 0.0,
        "subgoal_completion_rate": float(np.mean([rollout["subgoal_completion_rate"] for rollout in rollouts])) if rollouts else 0.0,
        "order_violation_count": int(order_violation_count),
        "order_violation_rate": float(order_violation_count / len(rollouts)) if rollouts else 0.0,
        "avg_termination_step": float(np.mean([rollout["termination_step"] for rollout in rollouts])) if rollouts else 0.0,
        "behavior_counts": dict(behavior_counts),
        "first_behavior_counts": dict(first_behavior_counts),
        "event_count_patterns": {
            " -> ".join(pattern) if pattern else "none": count
            for pattern, count in event_patterns.items()
        },
        "rollouts": [
            {
                "seed": rollout["seed"],
                "liveness_satisfied": bool(rollout["liveness_satisfied"]),
                "safety_satisfied": rollout["safety_satisfied"],
                "stl_satisfied": bool(rollout["stl_satisfied"]),
                "subgoal_completion_rate": float(rollout["subgoal_completion_rate"]),
                "completed_subgoals": int(rollout["completed_subgoals"]),
                "total_subgoals": int(rollout["total_subgoals"]),
                "target_events": rollout["target_events"],
                "order_violation": bool(rollout.get("order_violation", False)),
                "safety_metrics": rollout.get("safety_metrics", {}),
                "safety_box": rollout.get("safety_box"),
                "rzz_spec": rollout.get("rzz_spec"),
                "safety_randomization": rollout.get("safety_randomization", {"enabled": False}),
                "behavior": rollout["behavior"],
                "first_behavior": rollout["first_behavior"],
                "termination_step": rollout["termination_step"],
                "termination_reason": rollout["termination_reason"],
                "env_done_step": rollout["env_done_step"],
                "initial_label": rollout["initial_label"],
                "final_label": rollout["final_label"],
                "video": None if rollout.get("video") is None else str(rollout["video"]),
                "trace": str(rollout.get("trace")),
                "diagnostics": str(rollout.get("diagnostics")),
                "topdown_plot": str(Path(rollout.get("rollout_dir", "")) / "rollout_xy.png"),
            }
            for rollout in rollouts
        ],
    }


def write_summary_tables(run_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "mode",
        "formula",
        "n_rollouts",
        "n_candidates",
        "horizon",
        "liveness_satisfaction_rate",
        "safety_satisfaction_rate",
        "subgoal_completion_rate",
        "stl_satisfaction_rate",
        "order_violation_rate",
        "avg_termination_step",
        "event_count_patterns",
    ]
    rows = []
    for item in summaries:
        safety_rate = item["safety_satisfaction_rate"]
        rows.append(
            {
                "task": item["task"],
                "mode": item["mode"],
                "formula": item["formula"],
                "n_rollouts": item["n_rollouts"],
                "n_candidates": item["n_candidates"],
                "horizon": item["horizon"],
                "liveness_satisfaction_rate": f"{item['liveness_satisfaction_rate']:.4f}",
                "safety_satisfaction_rate": "" if safety_rate is None else f"{safety_rate:.4f}",
                "subgoal_completion_rate": f"{item['subgoal_completion_rate']:.4f}",
                "stl_satisfaction_rate": f"{item['stl_satisfaction_rate']:.4f}",
                "order_violation_rate": f"{item['order_violation_rate']:.4f}",
                "avg_termination_step": f"{item['avg_termination_step']:.2f}",
                "event_count_patterns": json.dumps(item["event_count_patterns"], sort_keys=True),
            }
        )
    with (run_dir / "summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "summary_table.md").open("w") as f:
        f.write("| task | mode | liveness | safety | subgoal | STL | order violations | n | horizon | events |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['task']} | {row['mode']} | {row['liveness_satisfaction_rate']} | "
                f"{row['safety_satisfaction_rate']} | {row['subgoal_completion_rate']} | "
                f"{row['stl_satisfaction_rate']} | {row['order_violation_rate']} | "
                f"{row['n_rollouts']} | {row['horizon']} | `{row['event_count_patterns']}` |\n"
            )
    write_summary_table_image(run_dir, rows)


def _rate_cell(value: str) -> str:
    if value is None or value == "":
        return ""
    return f"{100.0 * float(value):.0f}%"


def _count_events(value: str) -> str:
    try:
        events = json.loads(value)
    except Exception:
        return value
    compact = []
    for name, count in sorted(events.items(), key=lambda item: (-int(item[1]), item[0])):
        if len(name) > 42:
            name = name[:39] + "..."
        compact.append(f"{name}: {count}")
    return "\n".join(compact[:3])


def _task_label(task: str) -> str:
    labels = {
        "selection": "selection",
        "unordered": "unordered",
        "conditional": "conditional",
        "chained": "chained",
        "branched": "branched",
        "cyclic": "cyclic",
        "region": "region",
        "angle": "angle",
        "gripper": "gripper",
        "single_drawer_open": "single",
        "selection_button_or_switch_then_drawer": "selection",
        "unordered_button_and_switch_then_drawer": "conditional",
        "chained_drawer_switch_button_door": "chained",
        "branched_button_or_switch_drawer_remaining": "branched",
        "cyclic_drawer_switch": "cyclic",
        "safety_region_switch": "region",
        "safety_rzz_door": "angle",
        "safety_gripper_drawer": "gripper",
    }
    return labels.get(task, task)


def _color_for_rate(text: str, *, invert: bool = False) -> str:
    if not text:
        return "#f8fafc"
    value = float(text.rstrip("%")) / 100.0
    if invert:
        value = 1.0 - value
    if value >= 0.8:
        return "#dcfce7"
    if value >= 0.5:
        return "#fef9c3"
    return "#fee2e2"


def write_summary_table_image(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt

    core_headers = ["Task", "Mode", "Live", "Subgoal", "STL", "Order\nviol.", "Avg\nsteps", "Top events"]
    core_rows = [
        [
            _task_label(row["task"]),
            row["mode"],
            _rate_cell(row["liveness_satisfaction_rate"]),
            _rate_cell(row["subgoal_completion_rate"]),
            _rate_cell(row["stl_satisfaction_rate"]),
            _rate_cell(row["order_violation_rate"]),
            row["avg_termination_step"],
            _count_events(row["event_count_patterns"]),
        ]
        for row in rows
    ]
    safety_rows = [
        [_task_label(row["task"]), _rate_cell(row["safety_satisfaction_rate"])]
        for row in rows
        if row["safety_satisfaction_rate"] != ""
    ]

    fig_height = 1.4 + 0.58 * len(core_rows) + (0.9 + 0.28 * len(safety_rows) if safety_rows else 0.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    ax.axis("off")
    title = f"Complex STL Summary | {rows[0]['n_rollouts']} rollouts"
    ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")

    core_table = ax.table(
        cellText=core_rows,
        colLabels=core_headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.09, 0.07, 0.08, 0.07, 0.08, 0.07, 0.38],
        bbox=[0.0, 0.32 if safety_rows else 0.0, 1.0, 0.65 if safety_rows else 0.92],
    )
    core_table.auto_set_font_size(False)
    core_table.set_fontsize(9)
    for (r, c), cell in core_table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
            if c in (2, 3, 4):
                cell.set_facecolor(_color_for_rate(core_rows[r - 1][c]))
            if c == 5:
                cell.set_facecolor(_color_for_rate(core_rows[r - 1][c], invert=True))
            if c in (0, 7):
                cell.get_text().set_ha("left")

    if safety_rows:
        ax.text(0.0, 0.22, "Safety metrics (only tasks where safety is defined)", transform=ax.transAxes, fontsize=11, fontweight="bold")
        safety_table = ax.table(
            cellText=safety_rows,
            colLabels=["Task", "Safety"],
            cellLoc="center",
            colLoc="center",
            colWidths=[0.3, 0.12],
            bbox=[0.0, 0.0, 0.42, 0.20],
        )
        safety_table.auto_set_font_size(False)
        safety_table.set_fontsize(9)
        for (r, c), cell in safety_table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#334155")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
                if c == 1:
                    cell.set_facecolor(_color_for_rate(safety_rows[r - 1][1]))
                if c == 0:
                    cell.get_text().set_ha("left")

    out_path = run_dir / "summary_table.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
