#!/usr/bin/env python
"""Unit tests for Toy Squares TeLoGraF spec helpers."""

from __future__ import annotations

import numpy as np

from telograf_toy_squares.toy_specs import (
    ACTION_DIM,
    STATE_DIM,
    evaluate_spec_sequence,
    label_states,
    padded_action_window,
    padded_state_window,
    spec_to_vector,
)


def make_state(agent_xy, blue=(0.2, 0.2), red=(0.8, 0.2), green=(0.8, 0.8), yellow=(0.2, 0.8)):
    return np.asarray([*agent_xy, *blue, *red, *green, *yellow], dtype=np.float32)


def test_labels():
    states = np.stack([make_state((0.21, 0.22)), make_state((0.79, 0.19))], axis=0)
    labels = label_states(states, radius=0.05)
    assert labels.shape == (2, 4)
    assert labels[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert labels[1].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_eventual_or_and_sequence():
    traj = np.stack(
        [
            make_state((0.5, 0.5)),
            make_state((0.21, 0.2)),
            make_state((0.5, 0.5)),
            make_state((0.8, 0.21)),
            make_state((0.79, 0.8)),
        ],
        axis=0,
    )
    ok, score = evaluate_spec_sequence({"type": "eventual", "labels": ["blue"], "radius": 0.05}, traj)
    assert ok and score > 0.0
    ok, score = evaluate_spec_sequence({"type": "or", "labels": ["yellow", "green"], "radius": 0.05}, traj)
    assert ok and score > 0.0
    ok, score = evaluate_spec_sequence({"type": "and", "labels": ["blue", "red"], "radius": 0.05}, traj)
    assert ok and score > 0.0
    ok, score = evaluate_spec_sequence({"type": "sequence", "labels": ["blue", "red", "green"], "radius": 0.05}, traj)
    assert ok and score > 0.0
    ok, _ = evaluate_spec_sequence({"type": "sequence", "labels": ["green", "red"], "radius": 0.05}, traj)
    assert not ok


def test_padding_and_spec_vector():
    states = np.stack([make_state((0.0, 0.0)), make_state((0.2, 0.2))], axis=0)
    actions = np.asarray([[0.1, 0.1]], dtype=np.float32)
    state_window = padded_state_window(states, start=0, horizon=4)
    action_window = padded_action_window(actions, start=0, horizon=4)
    assert state_window.shape == (5, STATE_DIM)
    assert action_window.shape == (4, ACTION_DIM)
    assert np.allclose(state_window[-1], states[-1])
    assert np.allclose(action_window[-1], actions[-1])
    vec = spec_to_vector({"type": "sequence", "labels": ["blue", "yellow", "green"], "radius": 0.2})
    assert vec.ndim == 1
    assert vec.dtype == np.float32
    assert float(vec.sum()) > 0.0


def main() -> None:
    test_labels()
    test_eventual_or_and_sequence()
    test_padding_and_spec_vector()
    print("toy spec tests passed")


if __name__ == "__main__":
    main()
