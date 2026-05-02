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


def classify_behavior(start_state: Sequence[float], state: Sequence[float]) -> str:
    """Classify a rollout segment using DynaGuide's CALVIN behavior thresholds."""

    start_state = np.asarray(start_state)
    state = np.asarray(state)

    if state[SCENE_INDEX["green_light"]] > 0.8 and start_state[SCENE_INDEX["green_light"]] < 0.2:
        return "button_on"
    if state[SCENE_INDEX["green_light"]] < 0.2 and start_state[SCENE_INDEX["green_light"]] > 0.8:
        return "button_off"
    if state[SCENE_INDEX["switch"]] < 0.01 and start_state[SCENE_INDEX["switch"]] > 0.07:
        return "switch_off"
    if state[SCENE_INDEX["switch"]] > 0.07 and start_state[SCENE_INDEX["switch"]] < 0.01:
        return "switch_on"
    if state[SCENE_INDEX["drawer"]] < 0.05 and start_state[SCENE_INDEX["drawer"]] > 0.10:
        return "drawer_close"
    if state[SCENE_INDEX["drawer"]] > 0.10 and start_state[SCENE_INDEX["drawer"]] < 0.05:
        return "drawer_open"
    if state[SCENE_INDEX["sliding_door"]] < 0.05 and start_state[SCENE_INDEX["sliding_door"]] > 0.25:
        return "door_right"
    if state[SCENE_INDEX["sliding_door"]] > 0.25 and start_state[SCENE_INDEX["sliding_door"]] < 0.05:
        return "door_left"
    if np.linalg.norm(state[BLOCK_POS_SLICES["red"]] - start_state[BLOCK_POS_SLICES["red"]]) > 0.01:
        return "red_displace"
    if np.linalg.norm(state[BLOCK_POS_SLICES["blue"]] - start_state[BLOCK_POS_SLICES["blue"]]) > 0.01:
        return "blue_displace"
    if np.linalg.norm(state[BLOCK_POS_SLICES["pink"]] - start_state[BLOCK_POS_SLICES["pink"]]) > 0.01:
        return "pink_displace"
    return "other"


def check_state_difference(
    start_state: Sequence[float],
    state: Sequence[float],
    robot_pos: Sequence[float],
    binaries: Iterable[bool],
    for_display: bool = False,
) -> bool:
    """DynaGuide rollout stopper.

    Returns True once an articulated object changes direction relative to the
    binary start state, or when a block near the robot has moved.
    """

    start_state = np.asarray(start_state)
    state = np.asarray(state)
    robot_pos = np.asarray(robot_pos)

    for binary, idx, limits in zip(binaries, ADJUSTABLE_INDEX, ADJUSTABLE_LIMITS):
        midpoint = (limits[1] - limits[0]) / 2
        near_low = limits[0] + 0.25 * (limits[1] - limits[0])
        near_high = limits[0] + 0.75 * (limits[1] - limits[0])

        if not for_display:
            if binary and state[idx] < midpoint:
                return True
            if not binary and state[idx] > midpoint:
                return True
        else:
            if binary and state[idx] < near_low:
                return True
            if not binary and state[idx] > near_high:
                return True

    block_threshold = 0.03 if for_display else 0.001
    for color, pos_slice in BLOCK_POS_SLICES.items():
        del color
        if np.linalg.norm(robot_pos - state[pos_slice]) < 0.06:
            if np.linalg.norm(state[pos_slice.start : pos_slice.start + 2] - start_state[pos_slice.start : pos_slice.start + 2]) > block_threshold:
                return True
    return False


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
            return BehaviorEvent(classify_behavior(start_state, scene_states[step]), step)
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
