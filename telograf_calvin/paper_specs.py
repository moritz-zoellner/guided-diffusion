"""Shared CALVIN paper-STL utilities for diagnostics, mining, and training.

The utilities here intentionally operate only on low-dimensional CALVIN state:
`obs/proprio`, `obs/states`, and `actions` from the robomimic-style HDF5.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calvin_experiments.label_calvin_world_model import (  # noqa: E402
    LABEL_NAMES,
    LABEL_THRESHOLDS,
    SCENE_OBS_INDICES,
    label_scene_states,
)


STATE_DIM = 39
PROPRIO_DIM = 15
SCENE_DIM = 24
ACTION_DIM = 7

PAPER_CHAIN = [
    "button_on",
    "drawer_open",
    "switch_on",
    "button_pressed",
    "door_left",
    "drawer_closed",
]
PAPER_AND_OR = ["switch_on", "button_on"]
STAGED_FIRST = ["button_on", "switch_on"]
STAGED_FINAL = "drawer_open"
GRIPPER_OPEN_MIN_WIDTH = 0.06
GRIPPER_OPEN_MARGIN = 0.02

SPEC_TYPES = [
    "eventual",
    "or",
    "and",
    "ordered",
    "chain_prefix",
    "staged_before",
    "safety_box",
    "gripper_open",
]
SPEC_TYPE_TO_ID = {name: idx for idx, name in enumerate(SPEC_TYPES)}


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
            float(self.margin),
        )

    def contains(self, xy: np.ndarray) -> np.ndarray:
        box = self.normalized()
        xy = np.asarray(xy, dtype=np.float32)
        return (
            (xy[..., 0] >= box.x_min)
            & (xy[..., 0] <= box.x_max)
            & (xy[..., 1] >= box.y_min)
            & (xy[..., 1] <= box.y_max)
        )

    def signed_distance(self, xy: np.ndarray) -> np.ndarray:
        """Signed distance to the rectangle, positive outside and negative inside."""

        box = self.normalized()
        xy = np.asarray(xy, dtype=np.float32)
        center = np.asarray(
            [(box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0],
            dtype=np.float32,
        )
        half = np.asarray(
            [(box.x_max - box.x_min) / 2.0, (box.y_max - box.y_min) / 2.0],
            dtype=np.float32,
        )
        q = np.abs(xy - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
        return outside + inside


FIXED_SAFETY_BOX = SafetyBox()


@dataclass
class DemoArrays:
    key: str
    split: str
    robot: np.ndarray
    scene: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    labels: np.ndarray
    margins: np.ndarray
    edges: Dict[str, np.ndarray]
    eef_xy: np.ndarray
    gripper_width: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.states))


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def decode_mask_entries(values: np.ndarray) -> List[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def hdf5_demo_splits(h5: h5py.File) -> Dict[str, str]:
    keys = list(h5["data"].keys())
    splits = {key: "unknown" for key in keys}
    if "mask" not in h5:
        return splits
    for split in h5["mask"].keys():
        for key in decode_mask_entries(np.asarray(h5["mask"][split])):
            splits[key] = split
    return splits


def sorted_demo_keys(h5: h5py.File, max_demos: Optional[int] = None) -> List[str]:
    def key_fn(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    keys = sorted(h5["data"].keys(), key=key_fn)
    if max_demos is not None:
        keys = keys[: int(max_demos)]
    return keys


def load_demo_arrays(h5: h5py.File, key: str, split: str) -> DemoArrays:
    demo = h5["data"][key]
    robot = demo["obs/proprio"][:].astype(np.float32)
    scene = demo["obs/states"][:].astype(np.float32)
    actions = np.clip(demo["actions"][:], -1.0, 1.0).astype(np.float32)
    states = np.concatenate([robot, scene], axis=-1).astype(np.float32)
    labels = label_scene_states(scene).astype(bool)
    margins = label_margins(scene)
    edges = rising_edges(labels)
    return DemoArrays(
        key=key,
        split=split,
        robot=robot,
        scene=scene,
        actions=actions,
        states=states,
        labels=labels,
        margins=margins,
        edges=edges,
        eef_xy=robot[:, :2].astype(np.float32),
        gripper_width=robot[:, 6].astype(np.float32),
    )


def label_margins(scene_states: np.ndarray) -> np.ndarray:
    scene = np.asarray(scene_states, dtype=np.float32)
    values = {
        "switch_on": scene[:, SCENE_OBS_INDICES["lightbulb"]] - LABEL_THRESHOLDS["switch_light"],
        "switch_off": LABEL_THRESHOLDS["switch_light"] - scene[:, SCENE_OBS_INDICES["lightbulb"]],
        "button_on": scene[:, SCENE_OBS_INDICES["led"]] - LABEL_THRESHOLDS["button_light"],
        "button_off": LABEL_THRESHOLDS["button_light"] - scene[:, SCENE_OBS_INDICES["led"]],
        "button_pressed": scene[:, SCENE_OBS_INDICES["button"]] - LABEL_THRESHOLDS["button_pressed"],
        "drawer_open": scene[:, SCENE_OBS_INDICES["drawer"]] - LABEL_THRESHOLDS["drawer"],
        "drawer_closed": LABEL_THRESHOLDS["drawer"] - scene[:, SCENE_OBS_INDICES["drawer"]],
        "door_left": scene[:, SCENE_OBS_INDICES["slide"]] - LABEL_THRESHOLDS["door"],
        "door_right": LABEL_THRESHOLDS["door"] - scene[:, SCENE_OBS_INDICES["slide"]],
    }
    return np.stack([values[name] for name in LABEL_NAMES], axis=-1).astype(np.float32)


def rising_edges(labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=bool)
    edges = {}
    for idx, name in enumerate(LABEL_NAMES):
        if len(labels) <= 1:
            edges[name] = np.empty((0,), dtype=np.int64)
        else:
            edges[name] = (np.flatnonzero((~labels[:-1, idx]) & labels[1:, idx]) + 1).astype(np.int64)
    return edges


def spec_formula(spec: Mapping) -> str:
    typ = spec["type"]
    if typ == "eventual":
        return f"F {spec['labels'][0]}"
    if typ == "or":
        return " || ".join(f"F {name}" for name in spec["labels"])
    if typ == "and":
        return " && ".join(f"F {name}" for name in spec["labels"])
    if typ in {"ordered", "chain_prefix"}:
        return " -> ".join(spec["labels"])
    if typ == "staged_before":
        first = " AND ".join(spec["stage1"])
        return f"F {spec['final']} AND (!{spec['final']} U ({first}))"
    if typ == "safety_box":
        return f"F {spec['goal']} AND G avoid_box"
    if typ == "gripper_open":
        return f"F {spec['goal']} AND G gripper_open"
    raise ValueError(f"Unknown spec type {typ!r}")


def make_spec(spec_id: str, typ: str, **kwargs) -> Dict:
    spec = {"id": spec_id, "type": typ, **kwargs}
    spec["formula"] = spec_formula(spec)
    return spec


def core_paper_specs(include_gripper: bool = True) -> List[Dict]:
    specs = [
        make_spec("paper_or_switch_on_button_on", "or", labels=list(PAPER_AND_OR)),
        make_spec("paper_and_switch_on_button_on", "and", labels=list(PAPER_AND_OR)),
        make_spec(
            "paper_staged_button_switch_before_drawer",
            "staged_before",
            stage1=list(STAGED_FIRST),
            final=STAGED_FINAL,
        ),
        make_spec(
            "paper_safety_switch_on_avoid_box",
            "safety_box",
            goal="switch_on",
            safety_box=asdict(FIXED_SAFETY_BOX),
        ),
    ]
    for prefix_len in range(2, len(PAPER_CHAIN) + 1):
        specs.append(
            make_spec(
                f"paper_chain_prefix_{prefix_len}",
                "chain_prefix",
                labels=list(PAPER_CHAIN[:prefix_len]),
            )
        )
    if include_gripper:
        specs.append(
            make_spec(
                "paper_gripper_drawer_open",
                "gripper_open",
                goal="drawer_open",
                min_width=GRIPPER_OPEN_MIN_WIDTH,
                margin=GRIPPER_OPEN_MARGIN,
            )
        )
    return specs


def diagnostic_specs(include_triples: bool = True, include_gripper: bool = True) -> List[Dict]:
    specs = list(core_paper_specs(include_gripper=include_gripper))
    existing = {spec["id"] for spec in specs}

    for label in LABEL_NAMES:
        spec = make_spec(f"eventual_{label}", "eventual", labels=[label])
        if spec["id"] not in existing:
            specs.append(spec)
            existing.add(spec["id"])

    for left in LABEL_NAMES:
        for right in LABEL_NAMES:
            if left == right:
                continue
            spec = make_spec(f"ordered_pair_{left}_then_{right}", "ordered", labels=[left, right])
            if spec["id"] not in existing:
                specs.append(spec)
                existing.add(spec["id"])

    if include_triples:
        for first in LABEL_NAMES:
            for second in LABEL_NAMES:
                for third in LABEL_NAMES:
                    if len({first, second, third}) != 3:
                        continue
                    spec = make_spec(
                        f"ordered_triple_{first}_then_{second}_then_{third}",
                        "ordered",
                        labels=[first, second, third],
                    )
                    specs.append(spec)

    return specs


def _label_idx(label: str) -> int:
    return LABEL_NAMES.index(label)


def _valid_start_for_events(events: Sequence[int], horizon: int, length: int, pre_event_steps: int) -> Optional[int]:
    if not events:
        return None
    first = int(events[0])
    last = int(events[-1])
    if last - first > horizon:
        return None
    max_start = length - horizon - 1
    if max_start < 0:
        return None
    preferred = max(0, first - int(pre_event_steps))
    start = max(preferred, last - horizon)
    start = min(start, max_start)
    if start >= first:
        return None
    if last > start + horizon:
        return None
    return int(start)


def _score_goal_margins(demo: DemoArrays, labels: Sequence[str], start: int, end: int) -> float:
    scores = []
    for label in labels:
        idx = _label_idx(label)
        scores.append(float(np.max(demo.margins[start : end + 1, idx])))
    return float(min(scores)) if scores else 0.0


def _candidate(
    spec: Mapping,
    demo: DemoArrays,
    start: int,
    horizon: int,
    event_times: Mapping[str, int],
    score: float,
    extra: Optional[Mapping] = None,
) -> Dict:
    out = {
        "spec_id": spec["id"],
        "spec": dict(spec),
        "demo_key": demo.key,
        "split": demo.split,
        "start": int(start),
        "end": int(start + horizon),
        "event_times": {str(k): int(v) for k, v in event_times.items()},
        "score": float(score),
    }
    if extra:
        out.update(dict(extra))
    return out


def iter_eventual_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    label = spec["labels"][0]
    for edge in demo.edges[label]:
        start = _valid_start_for_events([int(edge)], horizon, demo.length, pre_event_steps)
        if start is None:
            continue
        score = _score_goal_margins(demo, [label], start, start + horizon)
        yield _candidate(spec, demo, start, horizon, {label: int(edge - start)}, score)


def _next_edge_after(edges: np.ndarray, t: int, upper: int) -> Optional[int]:
    pos = int(np.searchsorted(edges, t + 1, side="left"))
    if pos >= len(edges):
        return None
    value = int(edges[pos])
    if value > upper:
        return None
    return value


def iter_ordered_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    labels = list(spec["labels"])
    if not labels:
        return
    for first_edge in demo.edges[labels[0]]:
        events = [int(first_edge)]
        upper = int(first_edge) + horizon
        ok = True
        for label in labels[1:]:
            nxt = _next_edge_after(demo.edges[label], events[-1], upper)
            if nxt is None:
                ok = False
                break
            events.append(nxt)
        if not ok:
            continue
        start = _valid_start_for_events(events, horizon, demo.length, pre_event_steps)
        if start is None:
            continue
        score = _score_goal_margins(demo, labels, start, start + horizon)
        yield _candidate(
            spec,
            demo,
            start,
            horizon,
            {label: event - start for label, event in zip(labels, events)},
            score,
        )


def iter_unordered_pair_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    left, right = spec["labels"]
    seen = set()
    for left_event in demo.edges[left]:
        right_edges = demo.edges[right]
        lo = int(np.searchsorted(right_edges, int(left_event) - horizon, side="left"))
        hi = int(np.searchsorted(right_edges, int(left_event) + horizon, side="right"))
        for right_event in right_edges[lo:hi]:
            events_abs = sorted([int(left_event), int(right_event)])
            key = tuple(events_abs)
            if key in seen:
                continue
            seen.add(key)
            start = _valid_start_for_events(events_abs, horizon, demo.length, pre_event_steps)
            if start is None:
                continue
            score = _score_goal_margins(demo, [left, right], start, start + horizon)
            yield _candidate(
                spec,
                demo,
                start,
                horizon,
                {left: int(left_event - start), right: int(right_event - start)},
                score,
            )


def iter_or_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    seen = set()
    for label in spec["labels"]:
        for item in iter_eventual_windows(make_spec(spec["id"], "eventual", labels=[label]), demo, horizon, pre_event_steps):
            start = int(item["start"])
            if start in seen:
                continue
            seen.add(start)
            score = _score_goal_margins(demo, spec["labels"], start, start + horizon)
            yield _candidate(spec, demo, start, horizon, item["event_times"], score)


def iter_staged_before_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    first_labels = list(spec["stage1"])
    final_label = spec["final"]
    left, right = first_labels
    seen = set()
    for left_event in demo.edges[left]:
        right_edges = demo.edges[right]
        lo = int(np.searchsorted(right_edges, int(left_event) - horizon, side="left"))
        hi = int(np.searchsorted(right_edges, int(left_event) + horizon, side="right"))
        for right_event in right_edges[lo:hi]:
            stage_done = max(int(left_event), int(right_event))
            final_event = _next_edge_after(demo.edges[final_label], stage_done, min(stage_done + horizon, demo.length - 1))
            if final_event is None:
                continue
            events_abs = sorted([int(left_event), int(right_event)]) + [int(final_event)]
            if int(final_event) - events_abs[0] > horizon:
                continue
            key = (int(left_event), int(right_event), int(final_event))
            if key in seen:
                continue
            seen.add(key)
            start = _valid_start_for_events(events_abs, horizon, demo.length, pre_event_steps)
            if start is None:
                continue
            final_idx = _label_idx(final_label)
            stage_end_rel = stage_done - start
            if np.any(demo.labels[start : start + stage_end_rel + 1, final_idx]):
                continue
            score = _score_goal_margins(demo, first_labels + [final_label], start, start + horizon)
            yield _candidate(
                spec,
                demo,
                start,
                horizon,
                {
                    left: int(left_event - start),
                    right: int(right_event - start),
                    final_label: int(final_event - start),
                },
                score,
            )


def iter_safety_box_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    box = SafetyBox(**spec.get("safety_box", asdict(FIXED_SAFETY_BOX)))
    goal = spec["goal"]
    for edge in demo.edges[goal]:
        start = _valid_start_for_events([int(edge)], horizon, demo.length, pre_event_steps)
        if start is None:
            continue
        xy = demo.eef_xy[start : start + horizon + 1]
        signed_dist = box.signed_distance(xy)
        min_dist = float(np.min(signed_dist))
        if min_dist <= 0.0:
            continue
        goal_score = _score_goal_margins(demo, [goal], start, start + horizon)
        score = float(min(goal_score, min(min_dist, float(box.margin))))
        yield _candidate(
            spec,
            demo,
            start,
            horizon,
            {goal: int(edge - start)},
            score,
            {"min_signed_distance": min_dist},
        )


def iter_gripper_open_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int) -> Iterator[Dict]:
    goal = spec["goal"]
    min_width = float(spec.get("min_width", GRIPPER_OPEN_MIN_WIDTH))
    for edge in demo.edges[goal]:
        start = _valid_start_for_events([int(edge)], horizon, demo.length, pre_event_steps)
        if start is None:
            continue
        width_margin = demo.gripper_width[start : start + horizon + 1] - min_width
        min_margin = float(np.min(width_margin))
        if min_margin <= 0.0:
            continue
        goal_score = _score_goal_margins(demo, [goal], start, start + horizon)
        score = float(min(goal_score, min_margin))
        yield _candidate(
            spec,
            demo,
            start,
            horizon,
            {goal: int(edge - start)},
            score,
            {"min_gripper_margin": min_margin},
        )


def iter_spec_windows(spec: Mapping, demo: DemoArrays, horizon: int, pre_event_steps: int = 16) -> Iterator[Dict]:
    typ = spec["type"]
    if typ == "eventual":
        yield from iter_eventual_windows(spec, demo, horizon, pre_event_steps)
    elif typ == "or":
        yield from iter_or_windows(spec, demo, horizon, pre_event_steps)
    elif typ == "and":
        yield from iter_unordered_pair_windows(spec, demo, horizon, pre_event_steps)
    elif typ in {"ordered", "chain_prefix"}:
        yield from iter_ordered_windows(spec, demo, horizon, pre_event_steps)
    elif typ == "staged_before":
        yield from iter_staged_before_windows(spec, demo, horizon, pre_event_steps)
    elif typ == "safety_box":
        yield from iter_safety_box_windows(spec, demo, horizon, pre_event_steps)
    elif typ == "gripper_open":
        yield from iter_gripper_open_windows(spec, demo, horizon, pre_event_steps)
    else:
        raise ValueError(f"Unknown spec type {typ!r}")


def deoverlap_windows(windows: Iterable[Mapping], horizon: int) -> List[Dict]:
    sorted_windows = sorted(windows, key=lambda item: (item["demo_key"], item["start"], item["end"]))
    kept = []
    last_end_by_demo: Dict[str, int] = {}
    min_gap = max(1, int(horizon))
    for item in sorted_windows:
        demo_key = str(item["demo_key"])
        start = int(item["start"])
        last_end = last_end_by_demo.get(demo_key, -10**9)
        if start < last_end:
            continue
        kept.append(dict(item))
        last_end_by_demo[demo_key] = start + min_gap
    return kept


def evaluate_spec_sequence(spec: Mapping, robot: np.ndarray, scene: np.ndarray) -> Tuple[bool, float]:
    labels = label_scene_states(scene).astype(bool)
    margins = label_margins(scene)
    edges = rising_edges(labels)

    def first_true(label: str, after: int = -1) -> Optional[int]:
        idx = _label_idx(label)
        hits = np.flatnonzero(labels[max(0, after + 1) :, idx])
        if len(hits) == 0:
            return None
        return int(hits[0] + max(0, after + 1))

    def first_edge(label: str, after: int = -1) -> Optional[int]:
        label_edges = edges[label]
        pos = int(np.searchsorted(label_edges, after + 1, side="left"))
        if pos >= len(label_edges):
            return None
        return int(label_edges[pos])

    typ = spec["type"]
    if typ == "eventual":
        idx = _label_idx(spec["labels"][0])
        score = float(np.max(margins[:, idx]))
        return score >= 0.0, score
    if typ == "or":
        score = max(float(np.max(margins[:, _label_idx(label)])) for label in spec["labels"])
        return score >= 0.0, score
    if typ == "and":
        score = min(float(np.max(margins[:, _label_idx(label)])) for label in spec["labels"])
        return score >= 0.0, score
    if typ in {"ordered", "chain_prefix"}:
        t = -1
        scores = []
        for label in spec["labels"]:
            t = first_edge(label, after=t)
            if t is None:
                return False, -1.0
            scores.append(float(margins[t, _label_idx(label)]))
        return True, float(min(scores))
    if typ == "staged_before":
        final_idx = _label_idx(spec["final"])
        times = []
        for label in spec["stage1"]:
            t = first_edge(label)
            if t is None:
                return False, -1.0
            times.append(t)
        stage_done = max(times)
        if np.any(labels[: stage_done + 1, final_idx]):
            return False, -1.0
        final_t = first_edge(spec["final"], after=stage_done)
        if final_t is None:
            return False, -1.0
        score = min(
            [float(margins[t, _label_idx(label)]) for t, label in zip(times, spec["stage1"])]
            + [float(margins[final_t, final_idx])]
        )
        return True, score
    if typ == "safety_box":
        goal_idx = _label_idx(spec["goal"])
        goal_score = float(np.max(margins[:, goal_idx]))
        box = SafetyBox(**spec.get("safety_box", asdict(FIXED_SAFETY_BOX)))
        min_dist = float(np.min(box.signed_distance(robot[:, :2])))
        score = float(min(goal_score, min_dist))
        return score >= 0.0, score
    if typ == "gripper_open":
        goal_idx = _label_idx(spec["goal"])
        goal_score = float(np.max(margins[:, goal_idx]))
        min_width = float(spec.get("min_width", GRIPPER_OPEN_MIN_WIDTH))
        min_margin = float(np.min(robot[:, 6] - min_width))
        score = float(min(goal_score, min_margin))
        return score >= 0.0, score
    raise ValueError(f"Unknown spec type {typ!r}")


def spec_to_vector(spec: Mapping) -> np.ndarray:
    vec: List[float] = []
    type_vec = [0.0] * len(SPEC_TYPES)
    type_vec[SPEC_TYPE_TO_ID[spec["type"]]] = 1.0
    vec.extend(type_vec)

    label_vec = [0.0] * len(LABEL_NAMES)
    for key in ("labels", "stage1"):
        for label in spec.get(key, []):
            label_vec[_label_idx(label)] = 1.0
    for key in ("goal", "final"):
        if key in spec:
            label_vec[_label_idx(spec[key])] = 1.0
    vec.extend(label_vec)

    box = SafetyBox(**spec.get("safety_box", asdict(SafetyBox(0.0, 0.0, 0.0, 0.0, 0.0))))
    vec.extend([box.x_min, box.x_max, box.y_min, box.y_max, box.margin])
    vec.extend(
        [
            float(spec.get("min_width", 0.0)),
            float(spec.get("margin", 0.0)),
            float(len(spec.get("labels", spec.get("stage1", [])))),
        ]
    )
    return np.asarray(vec, dtype=np.float32)


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
