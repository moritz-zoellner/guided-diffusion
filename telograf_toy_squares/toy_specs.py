#!/usr/bin/env python
"""Toy Squares formulas, labels, and HDF5 helpers for TeLoGraF baselines."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np


ACTION_DIM = 2
STATE_DIM = 10
DATA_DIM = ACTION_DIM + STATE_DIM
LABEL_NAMES = ("blue", "red", "green", "yellow")
LABEL_TO_BLOCK_SLICE = {
    "blue": slice(2, 4),
    "red": slice(4, 6),
    "green": slice(6, 8),
    "yellow": slice(8, 10),
}
SPEC_TYPES = ("eventual", "or", "and", "sequence")
DEFAULT_RADIUS = 0.2
DEFAULT_CHAIN_BASE = ("blue", "yellow", "green", "red")
MAX_SPEC_LABELS = 5


@dataclass
class ToyDemo:
    key: str
    split: str
    states: np.ndarray
    actions: np.ndarray
    state_seq: np.ndarray
    labels: np.ndarray
    target_label: str | None

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def demo_sort_key(name: str) -> int:
    match = re.search(r"\d+", str(name))
    return int(match.group()) if match else 0


def sorted_demo_keys(h5: h5py.File, max_demos: int | None = None) -> List[str]:
    keys = sorted(list(h5["data"].keys()), key=demo_sort_key)
    if max_demos is not None:
        keys = keys[: int(max_demos)]
    return keys


def _decode_mask(values: np.ndarray) -> List[str]:
    out = []
    for item in values:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def hdf5_demo_splits(h5: h5py.File, val_ratio: float = 0.1, seed: int = 7) -> Dict[str, str]:
    keys = sorted_demo_keys(h5)
    splits = {key: "train" for key in keys}
    if "mask" in h5:
        for split_name in ("train", "valid", "val"):
            if split_name in h5["mask"]:
                canonical = "valid" if split_name == "val" else split_name
                for key in _decode_mask(np.asarray(h5[f"mask/{split_name}"])):
                    splits[key] = canonical
        return splits

    rng = np.random.default_rng(seed)
    n_valid = max(1, int(round(len(keys) * float(val_ratio)))) if len(keys) > 1 else 0
    valid_indices = set(int(i) for i in rng.permutation(len(keys))[:n_valid])
    for idx, key in enumerate(keys):
        splits[key] = "valid" if idx in valid_indices else "train"
    return splits


def flat_obs(agent_pos: np.ndarray, block_states: np.ndarray) -> np.ndarray:
    agent_pos = np.asarray(agent_pos, dtype=np.float32)
    block_states = np.asarray(block_states, dtype=np.float32)
    return np.concatenate([agent_pos, block_states], axis=-1).astype(np.float32)


def label_states(states: np.ndarray, radius: float = DEFAULT_RADIUS) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    agent = states[:, 0:2]
    labels = []
    for name in LABEL_NAMES:
        block = states[:, LABEL_TO_BLOCK_SLICE[name]]
        labels.append(np.linalg.norm(agent - block, axis=-1) <= float(radius))
    return np.stack(labels, axis=-1).astype(np.float32)


def label_edges(labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float32)
    edges = {}
    prev = np.zeros(labels.shape[1], dtype=bool)
    current = labels.astype(bool)
    rising = np.logical_and(current, np.logical_not(np.vstack([prev[None], current[:-1]])))
    for idx, name in enumerate(LABEL_NAMES):
        edges[name] = np.flatnonzero(rising[:, idx]).astype(np.int64)
    return edges


def load_demo_arrays(h5: h5py.File, key: str, split: str, radius: float = DEFAULT_RADIUS) -> ToyDemo:
    group = h5[f"data/{key}"]
    obs = group["obs"]
    next_obs = group["next_obs"]
    states = flat_obs(np.asarray(obs["agent_pos"], dtype=np.float32), np.asarray(obs["states"], dtype=np.float32))
    next_states = flat_obs(
        np.asarray(next_obs["agent_pos"], dtype=np.float32),
        np.asarray(next_obs["states"], dtype=np.float32),
    )
    actions = np.asarray(group["actions"], dtype=np.float32)
    if not (len(states) == len(next_states) == len(actions)):
        raise ValueError(
            f"{key}: expected aligned obs, next_obs, and actions; "
            f"got {len(states)}, {len(next_states)}, {len(actions)}"
        )
    state_seq = np.concatenate([states, next_states[-1:]], axis=0).astype(np.float32)
    labels = label_states(state_seq, radius=radius)
    target_label = None
    if "label" in group and len(group["label"]):
        idx = int(np.asarray(group["label"])[0])
        if 0 <= idx < len(LABEL_NAMES):
            target_label = LABEL_NAMES[idx]
    return ToyDemo(
        key=str(key),
        split=str(split),
        states=states,
        actions=actions,
        state_seq=state_seq,
        labels=labels,
        target_label=target_label,
    )


def robustness_by_label(states: np.ndarray, radius: float = DEFAULT_RADIUS) -> Dict[str, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    agent = states[:, 0:2]
    out = {}
    for name in LABEL_NAMES:
        block = states[:, LABEL_TO_BLOCK_SLICE[name]]
        out[name] = float(radius) - np.linalg.norm(agent - block, axis=-1)
    return out


def first_event_time(labels: np.ndarray, label: str) -> int | None:
    idx = LABEL_NAMES.index(label)
    hits = np.flatnonzero(np.asarray(labels)[:, idx] > 0.5)
    return int(hits[0]) if len(hits) else None


def ordered_event_times(labels: np.ndarray, spec_labels: Sequence[str]) -> Dict[str, int] | None:
    start = 0
    out = {}
    labels_np = np.asarray(labels)
    for label in spec_labels:
        idx = LABEL_NAMES.index(label)
        hits = np.flatnonzero(labels_np[start:, idx] > 0.5)
        if not len(hits):
            return None
        t = int(start + hits[0])
        out[label] = t
        start = t + 1
    return out


def evaluate_spec_sequence(
    spec: Mapping,
    states: np.ndarray,
    radius: float | None = None,
) -> Tuple[bool, float]:
    radius = float(spec.get("radius", DEFAULT_RADIUS) if radius is None else radius)
    r = robustness_by_label(states, radius=radius)
    kind = str(spec["type"])
    labels = list(spec["labels"])

    if kind == "eventual":
        score = float(np.max(r[labels[0]]))
    elif kind == "or":
        score = float(max(np.max(r[labels[0]]), np.max(r[labels[1]])))
    elif kind == "and":
        score = float(min(np.max(r[labels[0]]), np.max(r[labels[1]])))
    elif kind == "sequence":
        score_arr = r[labels[0]]
        for label in labels[1:]:
            shifted_prefix = np.concatenate(
                [np.asarray([-np.inf], dtype=np.float32), np.maximum.accumulate(score_arr)[:-1]]
            )
            score_arr = np.minimum(shifted_prefix, r[label])
        score = float(np.max(score_arr))
    else:
        raise ValueError(f"Unknown spec type: {kind}")
    return score > 0.0, score


def toy_paper_specs(
    chain_base: Sequence[str] = DEFAULT_CHAIN_BASE,
    max_chain_horizon: int = 5,
    radius: float = DEFAULT_RADIUS,
) -> List[Dict]:
    specs: List[Dict] = []
    for label in LABEL_NAMES:
        specs.append(
            {
                "id": f"eventual_{label}",
                "type": "eventual",
                "labels": [label],
                "formula": f"F reach_{label}",
                "radius": float(radius),
            }
        )
    specs.extend(
        [
            {
                "id": "paper_or_blue_red",
                "type": "or",
                "labels": ["blue", "red"],
                "formula": "F reach_blue OR F reach_red",
                "radius": float(radius),
            },
            {
                "id": "paper_and_blue_yellow",
                "type": "and",
                "labels": ["blue", "yellow"],
                "formula": "F reach_blue AND F reach_yellow",
                "radius": float(radius),
            },
        ]
    )
    base = tuple(str(label) for label in chain_base)
    for horizon in range(2, int(max_chain_horizon) + 1):
        labels = [base[i % len(base)] for i in range(horizon)]
        specs.append(
            {
                "id": f"paper_chain_prefix_{horizon}",
                "type": "sequence",
                "labels": labels,
                "formula": " -> ".join(f"reach_{label}" for label in labels),
                "radius": float(radius),
            }
        )
    return specs


def deoverlap_windows(windows: Sequence[Mapping], horizon: int) -> List[Dict]:
    ordered = sorted(
        (dict(window) for window in windows),
        key=lambda item: (str(item.get("demo_key", "")), int(item.get("start", 0)), str(item.get("spec_id", ""))),
    )
    last_end_by_demo: Dict[str, int] = {}
    kept = []
    for window in ordered:
        demo_key = str(window.get("demo_key", ""))
        start = int(window.get("start", 0))
        if start < last_end_by_demo.get(demo_key, -1):
            continue
        kept.append(window)
        last_end_by_demo[demo_key] = start + int(horizon)
    return kept


def iter_spec_windows(
    spec: Mapping,
    demo: ToyDemo,
    horizon: int,
    pre_event_steps: int,
    allow_padding: bool = True,
) -> Iterable[Dict]:
    spec_labels = list(spec["labels"])
    kind = str(spec["type"])

    if kind == "eventual":
        t = first_event_time(demo.labels, spec_labels[0])
        if t is None:
            return
        event_times = {spec_labels[0]: t}
        event_t = t
    elif kind == "or":
        candidates = [(label, first_event_time(demo.labels, label)) for label in spec_labels]
        candidates = [(label, t) for label, t in candidates if t is not None]
        if not candidates:
            return
        label, event_t = min(candidates, key=lambda item: int(item[1]))
        event_times = {label: int(event_t)}
    elif kind == "and":
        times = {label: first_event_time(demo.labels, label) for label in spec_labels}
        if any(t is None for t in times.values()):
            return
        event_times = {label: int(t) for label, t in times.items() if t is not None}
        event_t = min(event_times.values())
    elif kind == "sequence":
        times = ordered_event_times(demo.labels, spec_labels)
        if times is None:
            return
        event_times = times
        event_t = min(times.values())
    else:
        raise ValueError(f"Unknown spec type: {kind}")

    start = max(0, int(event_t) - int(pre_event_steps))
    max_start = max(0, demo.length - int(horizon))
    if not allow_padding:
        if start > max_start:
            return
        padded_steps = 0
    else:
        padded_steps = max(0, start + int(horizon) - demo.length)
    if not allow_padding:
        start = min(start, max_start)

    end = start + int(horizon)
    states = padded_state_window(demo.state_seq, start, horizon)
    ok, score = evaluate_spec_sequence(spec, states)
    if not ok:
        return
    yield {
        "spec": dict(spec),
        "spec_id": str(spec["id"]),
        "demo_key": demo.key,
        "split": demo.split,
        "start": int(start),
        "end": int(end),
        "event_times": event_times,
        "score": float(score),
        "padded_steps": int(padded_steps),
        "demo_length": int(demo.length),
        "target_label": demo.target_label,
    }


def padded_state_window(state_seq: np.ndarray, start: int, horizon: int) -> np.ndarray:
    state_seq = np.asarray(state_seq, dtype=np.float32)
    end = int(start) + int(horizon) + 1
    window = state_seq[int(start) : min(end, len(state_seq))]
    if len(window) <= 0:
        window = state_seq[-1:]
    if len(window) < int(horizon) + 1:
        pad = np.repeat(window[-1:], int(horizon) + 1 - len(window), axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window.astype(np.float32)


def padded_action_window(actions: np.ndarray, start: int, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    end = int(start) + int(horizon)
    window = actions[int(start) : min(end, len(actions))]
    if len(window) <= 0:
        window = actions[-1:] if len(actions) else np.zeros((1, ACTION_DIM), dtype=np.float32)
    if len(window) < int(horizon):
        pad = np.repeat(window[-1:], int(horizon) - len(window), axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window.astype(np.float32)


def spec_to_vector(spec: Mapping) -> np.ndarray:
    vec = []
    kind = str(spec["type"])
    vec.extend([1.0 if kind == item else 0.0 for item in SPEC_TYPES])
    labels = list(spec.get("labels", []))
    vec.append(float(len(labels)) / float(MAX_SPEC_LABELS))
    for pos in range(MAX_SPEC_LABELS):
        label = labels[pos] if pos < len(labels) else None
        vec.extend([1.0 if label == name else 0.0 for name in LABEL_NAMES])
    vec.append(float(spec.get("radius", DEFAULT_RADIUS)))
    return np.asarray(vec, dtype=np.float32)


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)
