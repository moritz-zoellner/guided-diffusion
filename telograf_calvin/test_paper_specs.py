"""Lightweight unit tests for CALVIN paper-STL utilities."""

from __future__ import annotations

import unittest

import numpy as np

from telograf_calvin.paper_specs import (
    FIXED_SAFETY_BOX,
    LABEL_NAMES,
    SCENE_OBS_INDICES,
    SafetyBox,
    evaluate_spec_sequence,
    label_margins,
    make_spec,
    rising_edges,
)


def blank_scene(length: int) -> np.ndarray:
    scene = np.zeros((length, 24), dtype=np.float32)
    scene[:, SCENE_OBS_INDICES["drawer"]] = 0.0
    scene[:, SCENE_OBS_INDICES["slide"]] = 0.0
    scene[:, SCENE_OBS_INDICES["button"]] = 0.0
    scene[:, SCENE_OBS_INDICES["lightbulb"]] = 0.0
    scene[:, SCENE_OBS_INDICES["led"]] = 0.0
    return scene


class PaperSpecTests(unittest.TestCase):
    def test_label_edges(self):
        scene = blank_scene(8)
        scene[3:, SCENE_OBS_INDICES["lightbulb"]] = 1.0
        margins = label_margins(scene)
        self.assertGreater(margins[-1, LABEL_NAMES.index("switch_on")], 0.0)
        labels = margins >= 0.0
        edges = rising_edges(labels)
        self.assertEqual(edges["switch_on"].tolist(), [3])

    def test_or_and_ordered(self):
        scene = blank_scene(12)
        scene[2:, SCENE_OBS_INDICES["lightbulb"]] = 1.0
        scene[6:, SCENE_OBS_INDICES["led"]] = 1.0
        robot = np.zeros((12, 15), dtype=np.float32)
        ok, _ = evaluate_spec_sequence(make_spec("or", "or", labels=["switch_on", "button_on"]), robot, scene)
        self.assertTrue(ok)
        ok, _ = evaluate_spec_sequence(make_spec("and", "and", labels=["switch_on", "button_on"]), robot, scene)
        self.assertTrue(ok)
        ok, _ = evaluate_spec_sequence(
            make_spec("ordered", "ordered", labels=["switch_on", "button_on"]), robot, scene
        )
        self.assertTrue(ok)
        ok, _ = evaluate_spec_sequence(
            make_spec("ordered_bad", "ordered", labels=["button_on", "switch_on"]), robot, scene
        )
        self.assertFalse(ok)

    def test_safety_box(self):
        scene = blank_scene(8)
        scene[3:, SCENE_OBS_INDICES["lightbulb"]] = 1.0
        robot = np.zeros((8, 15), dtype=np.float32)
        robot[:, 0] = -0.1
        robot[:, 1] = -0.2
        ok, _ = evaluate_spec_sequence(
            make_spec(
                "safe",
                "safety_box",
                goal="switch_on",
                safety_box=FIXED_SAFETY_BOX.__dict__,
            ),
            robot,
            scene,
        )
        self.assertTrue(ok)
        robot[:, :2] = [0.24, -0.10]
        ok, _ = evaluate_spec_sequence(
            make_spec(
                "unsafe",
                "safety_box",
                goal="switch_on",
                safety_box=SafetyBox().__dict__,
            ),
            robot,
            scene,
        )
        self.assertFalse(ok)

    def test_gripper_open(self):
        scene = blank_scene(8)
        scene[4:, SCENE_OBS_INDICES["drawer"]] = 0.2
        robot = np.zeros((8, 15), dtype=np.float32)
        robot[:, 6] = 0.07
        ok, _ = evaluate_spec_sequence(
            make_spec("gripper", "gripper_open", goal="drawer_open", min_width=0.06), robot, scene
        )
        self.assertTrue(ok)
        robot[2, 6] = 0.01
        ok, _ = evaluate_spec_sequence(
            make_spec("gripper_bad", "gripper_open", goal="drawer_open", min_width=0.06), robot, scene
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

