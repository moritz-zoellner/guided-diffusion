import argparse
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import corallab_stl.torch as stl


LABEL_NAMES = ["switch_on", "button_pressed", "drawer_open"]

SCENE_OBS_INDICES = {
    "slide": 0,
    "drawer": 1,
    "button": 2,
    "switch": 3,
    "lightbulb": 4,
    "led": 5,
}

LABEL_THRESHOLDS = {
    "switch_on": 0.055,       # midpoint of base__switch joint limits [0, 0.11]
    "button_pressed": 0.0125, # midpoint of base__button joint limits [0, 0.025]
    "drawer_open": 0.12,      # CALVIN open_drawer task threshold
}


def _demo_sort_key(name):
    return int(re.search(r"\d+", name).group())


def get_demo_keys(hdf5_path, mask=None):
    with h5py.File(hdf5_path, "r") as f:
        if mask is None:
            keys = list(f["data"].keys())
        else:
            keys = [k.decode("utf-8") for k in np.asarray(f[f"mask/{mask}"])]
    return sorted(keys, key=_demo_sort_key)


def label_scene_states(scene_states):
    """Return binary [switch_on, button_pressed, drawer_open] labels."""
    scene_states_t = torch.as_tensor(scene_states, dtype=torch.float32)

    state = stl.Var("state", dim=24)
    switch_on = stl.Predicate(
        state,
        lambda s, _: s[SCENE_OBS_INDICES["switch"]] - LABEL_THRESHOLDS["switch_on"],
        0.0,
    )
    button_pressed = stl.Predicate(
        state,
        lambda s, _: s[SCENE_OBS_INDICES["button"]] - LABEL_THRESHOLDS["button_pressed"],
        0.0,
    )
    drawer_open = stl.Predicate(
        state,
        lambda s, _: s[SCENE_OBS_INDICES["drawer"]] - LABEL_THRESHOLDS["drawer_open"],
        0.0,
    )
    predicates = [switch_on, button_pressed, drawer_open]
    labels = torch.stack([p({"state": scene_states_t}) >= 0.0 for p in predicates], axis=-1)
    return labels.cpu().numpy().astype(np.float32)


def next_changed_labels(labels, immediate_next_labels):
    """
    Match the toy workflow: for each timestep, target the next future label
    configuration after the current plateau, not merely the immediate t+1 label.
    """
    if len(labels) != len(immediate_next_labels):
        raise ValueError("labels and immediate_next_labels must have the same length")
    if len(labels) == 0:
        return labels.copy()
    if len(labels) == 1:
        return immediate_next_labels.copy()

    out = np.empty_like(labels)
    previous = immediate_next_labels[-2].copy()
    carry = immediate_next_labels[-1].copy()

    for idx in range(len(labels) - 1, -1, -1):
        current = labels[idx]
        if np.array_equal(current, previous):
            out[idx] = carry
        else:
            out[idx] = previous
            carry = previous.copy()
        previous = current.copy()

    return out


def build_labelled_calvin_trajectories(hdf5_path, mask=None, max_demos=None):
    demo_keys = get_demo_keys(hdf5_path, mask=mask)
    if max_demos is not None:
        demo_keys = demo_keys[:max_demos]

    trajectories = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            scene_states = demo["obs/states"][:].astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)

            if len(scene_states) != len(actions):
                raise ValueError(
                    f"{demo_key}: expected obs/states and actions to align, "
                    f"got {len(scene_states)} states and {len(actions)} actions"
                )
            if len(scene_states) < 2:
                continue

            labels_all = label_scene_states(scene_states)
            labels = labels_all[:-1]
            immediate_next_labels = labels_all[1:]
            next_labels = next_changed_labels(labels, immediate_next_labels)

            states = scene_states[:-1]
            next_states = scene_states[1:]
            actions = actions[:-1]

            trajectories.append(
                {
                    "path": str(hdf5_path),
                    "demo_id": demo_key,
                    "states": states,
                    "actions": actions,
                    "next_states": next_states,
                    "deltas": next_states - states,
                    "labels": labels,
                    "next_labels": next_labels,
                }
            )

    return trajectories


def _as_tuple(label):
    return tuple(int(x) for x in np.asarray(label).tolist())


def _runs(labels):
    if len(labels) == 0:
        return []

    runs = []
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if not np.array_equal(labels[idx], current):
            runs.append((start, idx, current.copy()))
            start = idx
            current = labels[idx]
    runs.append((start, len(labels), current.copy()))
    return runs


def sanity_report(trajectories, max_runs_to_print=8):
    labels = np.concatenate([traj["labels"] for traj in trajectories], axis=0)
    next_labels = np.concatenate([traj["next_labels"] for traj in trajectories], axis=0)
    states = np.concatenate([traj["states"][:, :6] for traj in trajectories], axis=0)

    print("\nCALVIN label dataset")
    print(f"  trajectories: {len(trajectories)}")
    print(f"  transitions:  {len(labels)}")
    print(f"  label names:  {LABEL_NAMES}")
    print(f"  thresholds:   {LABEL_THRESHOLDS}")
    print("  label backend: corallab_stl")

    print("\nScene obs ranges")
    for name, idx in SCENE_OBS_INDICES.items():
        values = states[:, idx]
        print(
            f"  {idx:>2} {name:<9} "
            f"min={values.min(): .4f} p50={np.percentile(values, 50): .4f} "
            f"p99={np.percentile(values, 99): .4f} max={values.max(): .4f}"
        )

    print("\nLabel counts")
    for title, array in [("labels", labels), ("next_labels", next_labels)]:
        counts = Counter(_as_tuple(row) for row in array)
        print(f"  {title}:")
        for label, count in sorted(counts.items()):
            print(f"    {label}: {count}")

    all_zero = (0, 0, 0)
    if set(Counter(_as_tuple(row) for row in labels)) == {all_zero}:
        print("\nWARNING: every current label is (0, 0, 0). Check thresholds or scene_obs ordering.")
    if set(Counter(_as_tuple(row) for row in next_labels)) == {all_zero}:
        print("\nWARNING: every next label is (0, 0, 0). Check thresholds or future-label logic.")

    run_lengths = []
    progression_errors = 0
    for traj in trajectories:
        runs = _runs(traj["labels"])
        run_lengths.extend(end - start for start, end, _ in runs)
        for run_idx, (start, _, _) in enumerate(runs[:-1]):
            expected = runs[run_idx + 1][2]
            if not np.array_equal(traj["next_labels"][start], expected):
                progression_errors += 1

    if run_lengths:
        print("\nRun stability")
        print(
            f"  runs: {len(run_lengths)} | "
            f"median length: {np.median(run_lengths):.1f} | "
            f"p10 length: {np.percentile(run_lengths, 10):.1f} | "
            f"min length: {np.min(run_lengths)}"
        )
        if np.median(run_lengths) <= 2:
            print("  WARNING: labels are changing very frequently; inspect thresholds.")
    if progression_errors:
        print(f"  WARNING: {progression_errors} runs did not point to the next label run.")
    else:
        print("  next_labels match the next label run at every checked transition.")

    interesting = max(trajectories, key=lambda traj: len(_runs(traj["labels"])))
    print(f"\nExample progression from {interesting['demo_id']}")
    for start, end, label in _runs(interesting["labels"])[:max_runs_to_print]:
        target_counts = Counter(_as_tuple(row) for row in interesting["next_labels"][start:end])
        target, count = target_counts.most_common(1)[0]
        print(
            f"  t={start:>5}:{end:<5} label={_as_tuple(label)} "
            f"most_common_next={target} ({count}/{end - start})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("calvin/dataset/calvin_D_training.hdf5"),
        help="CALVIN robomimic-style HDF5 dataset",
    )
    parser.add_argument("--mask", default=None, help="optional HDF5 mask key, e.g. train or valid")
    parser.add_argument("--max-demos", type=int, default=None, help="optional quick-debug limit")
    args = parser.parse_args()

    trajectories = build_labelled_calvin_trajectories(args.dataset, mask=args.mask, max_demos=args.max_demos)
    sanity_report(trajectories)


if __name__ == "__main__":
    main()
