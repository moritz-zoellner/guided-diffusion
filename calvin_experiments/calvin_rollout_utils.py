"""DynaGuide-style CALVIN rollout reset and behavior labeling helpers.

These utilities intentionally work on CALVIN low-dimensional `scene_obs` and
`robot_obs` arrays, so they can be reused for base-policy rollouts, STL world
model labeling, and post-hoc trace analysis without depending on DynaGuide.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


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


@dataclass(frozen=True)
class BehaviorEvent:
    """First behavior detected in a rollout relative to its start state."""

    name: str
    step: int


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
        "drawer_open": bool(state[SCENE_INDEX["drawer"]] > 0.12),
        "drawer_closed": bool(state[SCENE_INDEX["drawer"]] < 0.04),
        "button_on": bool(state[SCENE_INDEX["green_light"]] > 0.5),
        "button_off": bool(state[SCENE_INDEX["green_light"]] <= 0.5),
        "switch_on": bool(state[SCENE_INDEX["lightbulb"]] > 0.5),
        "switch_off": bool(state[SCENE_INDEX["lightbulb"]] <= 0.5),
    }
