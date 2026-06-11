#!/usr/bin/env python3
"""Generate low-dimensional ToySquares TeLoGraF data for length-5 chains."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym  # noqa: E402
import robomimic.envs  # noqa: F401,E402

from telograf_toy_squares.run_telograf_toy_squares_rollouts import obs_to_state  # noqa: E402
from telograf_toy_squares.toy_specs import DEFAULT_RADIUS, LABEL_NAMES, evaluate_spec_sequence, label_states  # noqa: E402


SCRATCH_ROOT = Path("/scratch/gilbreth/zoellner/guided-diffusion")
DEFAULT_OUTPUT_DIR = SCRATCH_ROOT / "outputs/telograf/toy_squares/data/long_chains_len5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-rollouts", type=int, default=10000)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[128, 196])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps-per-goal", type=int, default=42)
    parser.add_argument("--min-steps-per-goal", type=int, default=8)
    parser.add_argument("--action-noise-scale", type=float, default=0.015)
    parser.add_argument("--waypoint-prob", type=float, default=0.35)
    parser.add_argument("--max-attempts-factor", type=int, default=20)
    parser.add_argument("--preview-videos", type=int, default=0)
    parser.add_argument("--video-size", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=10)
    return parser.parse_args()


def make_env():
    return gym.make("TouchCube")


def all_no_repeat_chains(labels: Sequence[str], length: int) -> List[Tuple[str, ...]]:
    chains = []
    for chain in itertools.product(labels, repeat=int(length)):
        if all(chain[i] != chain[i - 1] for i in range(1, len(chain))):
            chains.append(tuple(chain))
    return chains


def make_sequence_spec(chain: Sequence[str], radius: float) -> Dict:
    labels = [str(label) for label in chain]
    return {
        "id": f"sequence_{'_'.join(labels)}",
        "type": "sequence",
        "labels": labels,
        "formula": " -> ".join(f"reach_{label}" for label in labels),
        "radius": float(radius),
    }


def reset_env(env, rng: np.random.Generator):
    # Gym's legacy reset path uses global numpy/random state internally.
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)
    return env.reset()


def sample_waypoint(start: np.ndarray, target: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    midpoint = 0.5 * (start + target)
    for _ in range(128):
        candidate = np.clip(midpoint + rng.normal(0.0, 0.35, size=2), -0.96, 0.96)
        if np.min(np.linalg.norm(blocks - candidate[None, :], axis=-1)) > 0.18:
            return candidate.astype(np.float32)
    return midpoint.astype(np.float32)


def interpolate_actions(
    start: np.ndarray,
    target: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
    min_steps: int,
    max_steps: int,
    waypoint_prob: float,
    noise_scale: float,
) -> List[np.ndarray]:
    dist = float(np.linalg.norm(target - start))
    steps = int(np.clip(np.ceil(dist * 58.0), int(min_steps), int(max_steps)))
    points = [start.astype(np.float32)]
    if rng.random() < float(waypoint_prob):
        points.append(sample_waypoint(start, target, blocks, rng))
    points.append(target.astype(np.float32))

    segment_lengths = [max(1e-6, float(np.linalg.norm(points[i + 1] - points[i]))) for i in range(len(points) - 1)]
    total_length = sum(segment_lengths)
    actions: List[np.ndarray] = []
    for i, seg_len in enumerate(segment_lengths):
        seg_steps = max(2, int(round(steps * seg_len / total_length)))
        for alpha in np.linspace(0.0, 1.0, seg_steps, endpoint=False)[1:]:
            action = (1.0 - alpha) * points[i] + alpha * points[i + 1]
            if noise_scale > 0:
                action = action + rng.normal(0.0, noise_scale, size=2)
            actions.append(np.clip(action, -1.0, 1.0).astype(np.float32))
    actions.append(np.clip(target, -1.0, 1.0).astype(np.float32))
    return actions


def ordered_stages(states: np.ndarray, chain: Sequence[str], radius: float) -> int:
    labels = label_states(states, radius=radius)
    pos = 0
    for row in labels:
        if pos < len(chain) and row[LABEL_NAMES.index(chain[pos])] > 0.5:
            pos += 1
    return int(pos)


def render_state_frame(state: np.ndarray, size: int = 256) -> np.ndarray:
    colors = {
        "blue": np.array([42, 126, 255], dtype=np.uint8),
        "red": np.array([230, 57, 70], dtype=np.uint8),
        "green": np.array([46, 204, 113], dtype=np.uint8),
        "yellow": np.array([255, 214, 10], dtype=np.uint8),
    }
    frame = np.full((size, size, 3), 245, dtype=np.uint8)
    grid = np.linspace(0, size - 1, 5).astype(int)
    frame[grid, :, :] = 225
    frame[:, grid, :] = 225

    def xy_to_px(xy):
        x = int(np.clip((float(xy[0]) + 1.0) * 0.5 * (size - 1), 0, size - 1))
        y = int(np.clip((1.0 - (float(xy[1]) + 1.0) * 0.5) * (size - 1), 0, size - 1))
        return x, y

    blocks = state[2:10].reshape(4, 2)
    block_r = max(5, size // 28)
    for label, xy in zip(LABEL_NAMES, blocks):
        x, y = xy_to_px(xy)
        frame[max(0, y - block_r) : min(size, y + block_r + 1), max(0, x - block_r) : min(size, x + block_r + 1)] = colors[label]

    agent_x, agent_y = xy_to_px(state[0:2])
    rr = max(4, size // 34)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - agent_x) ** 2 + (yy - agent_y) ** 2 <= rr**2
    frame[mask] = np.array([20, 20, 20], dtype=np.uint8)
    return frame


def rollout_chain(env, chain: Sequence[str], args: argparse.Namespace, rng: np.random.Generator, record_video: bool):
    obs = reset_env(env, rng)
    initial = obs_to_state(obs)
    agent = initial[0:2].copy()
    blocks = initial[2:10].reshape(4, 2).copy()
    states = [initial.copy()]
    actions: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    reaches = []
    if record_video:
        frames.append(render_state_frame(states[-1], args.video_size))

    for stage, label in enumerate(chain):
        block_idx = LABEL_NAMES.index(label)
        target = blocks[block_idx].copy()
        dist = float(np.linalg.norm(target - agent))
        n_steps = int(np.clip(np.ceil(dist * 18.0), int(args.min_steps_per_goal), int(args.max_steps_per_goal)))
        for alpha in np.linspace(0.0, 1.0, n_steps + 1, endpoint=True)[1:]:
            action = ((1.0 - alpha) * agent + alpha * target).astype(np.float32)
            if args.action_noise_scale > 0 and alpha < 0.98:
                action = action + rng.normal(0.0, args.action_noise_scale, size=2).astype(np.float32)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            state = np.concatenate([action, blocks.reshape(-1)], axis=0).astype(np.float32)
            actions.append(action.astype(np.float32))
            states.append(state.astype(np.float32))
            if record_video:
                frames.append(render_state_frame(state, args.video_size))
            labels_now = label_states(np.asarray(states[-1:], dtype=np.float32), radius=args.radius)[0]
            if labels_now[block_idx] > 0.5:
                reaches.append({"stage": int(stage), "label": label, "t": int(len(actions))})
                break
        agent = states[-1][0:2].copy()

    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)
    spec = make_sequence_spec(chain, args.radius)
    ok, score = evaluate_spec_sequence(spec, states_np)
    return {
        "states": states_np,
        "actions": actions_np,
        "spec": spec,
        "chain": list(chain),
        "success": bool(ok),
        "score": float(score),
        "stages": ordered_stages(states_np, chain, args.radius),
        "reaches": reaches,
        "frames": frames,
    }


def pad_record(raw: Dict, horizon: int, split: str) -> Dict:
    states = np.asarray(raw["states"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    raw_steps = int(len(actions))
    if len(actions) > horizon:
        actions = actions[:horizon]
        states = states[: horizon + 1]
    if len(actions) < horizon:
        pad_n = int(horizon - len(actions))
        last_action = states[-1, 0:2].astype(np.float32)
        actions = np.concatenate([actions, np.repeat(last_action[None, :], pad_n, axis=0)], axis=0)
    if states.shape[0] < horizon + 1:
        states = np.concatenate([states, np.repeat(states[-1:], horizon + 1 - states.shape[0], axis=0)], axis=0)
    states = states[: horizon + 1].astype(np.float32)
    actions = actions[:horizon].astype(np.float32)
    spec = dict(raw["spec"])
    return {
        "spec_id": spec["id"],
        "spec": spec,
        "state": states[0].astype(np.float32),
        "trajs": states,
        "us": actions,
        "split": split,
        "chain": list(raw["chain"]),
        "success": bool(raw["success"]),
        "score": float(raw["score"]),
        "stages": int(raw["stages"]),
        "padded_steps": int(max(0, horizon - raw_steps)),
        "raw_steps": int(raw_steps),
    }


def write_dataset(path: Path, records: List[Dict], meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=np.asarray(records, dtype=object), meta=json.dumps(meta))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "preview_videos"
    if args.preview_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    chains = all_no_repeat_chains(LABEL_NAMES, args.chain_length)
    ordered_chains = list(chains)
    rng.shuffle(ordered_chains)

    env = make_env()
    raw_rollouts: List[Dict] = []
    attempts = 0
    max_attempts = max(args.n_rollouts, args.n_rollouts * int(args.max_attempts_factor))
    min_dataset_horizon = min(int(h) for h in args.horizons)
    while len(raw_rollouts) < args.n_rollouts and attempts < max_attempts:
        chain = ordered_chains[len(raw_rollouts) % len(ordered_chains)]
        record_video = len(raw_rollouts) < int(args.preview_videos)
        attempts += 1
        raw = rollout_chain(env, chain, args, rng, record_video=record_video)
        if not raw["success"]:
            continue
        if len(raw["actions"]) > min_dataset_horizon:
            continue
        raw_rollouts.append(raw)
        idx = len(raw_rollouts) - 1
        if record_video:
            imageio.mimsave(video_dir / f"preview_{idx:03d}_{'_'.join(chain)}.mp4", raw["frames"], fps=int(args.video_fps))
        if len(raw_rollouts) % 100 == 0 or len(raw_rollouts) <= 4:
            print(f"saved={len(raw_rollouts)}/{args.n_rollouts} attempts={attempts} chain={'->'.join(chain)} steps={len(raw['actions'])}", flush=True)

    if len(raw_rollouts) < args.n_rollouts:
        raise RuntimeError(f"Only collected {len(raw_rollouts)} successful rollouts after {attempts} attempts")

    chain_counts = {"->".join(raw["chain"]): 0 for raw in raw_rollouts}
    for raw in raw_rollouts:
        chain_counts["->".join(raw["chain"])] = chain_counts.get("->".join(raw["chain"]), 0) + 1
    possible_chains = len(chains)
    samples_per_chain_floor = args.n_rollouts // possible_chains
    samples_per_chain_remainder = args.n_rollouts % possible_chains

    for horizon in sorted(set(int(h) for h in args.horizons)):
        records = []
        for idx, raw in enumerate(raw_rollouts):
            split = "valid" if (idx % max(1, int(round(1.0 / args.val_ratio))) == 0) else "train"
            records.append(pad_record(raw, horizon, split))
        write_dataset(
            args.output_dir / f"data_h{horizon}.npz",
            records,
            {
                "dataset_type": "toy_squares_long_chain_point_to_point",
                "horizon": int(horizon),
                "n_rollouts": int(args.n_rollouts),
                "chain_length": int(args.chain_length),
                "label_names": list(LABEL_NAMES),
                "possible_chains": int(possible_chains),
                "samples_per_chain_floor": int(samples_per_chain_floor),
                "samples_per_chain_remainder": int(samples_per_chain_remainder),
                "radius": float(args.radius),
                "seed": int(args.seed),
                "val_ratio": float(args.val_ratio),
                "chain_counts": chain_counts,
            },
        )

    stats = {
        "output_dir": str(args.output_dir),
        "n_rollouts": int(args.n_rollouts),
        "attempts": int(attempts),
        "success_collection_rate": float(args.n_rollouts / max(1, attempts)),
        "chain_length": int(args.chain_length),
        "possible_chains": int(possible_chains),
        "samples_per_chain_floor": int(samples_per_chain_floor),
        "samples_per_chain_remainder": int(samples_per_chain_remainder),
        "horizons": sorted(set(int(h) for h in args.horizons)),
        "max_allowed_raw_steps": int(min_dataset_horizon),
        "mean_raw_steps": float(np.mean([len(raw["actions"]) for raw in raw_rollouts])),
        "max_raw_steps": int(max(len(raw["actions"]) for raw in raw_rollouts)),
        "min_raw_steps": int(min(len(raw["actions"]) for raw in raw_rollouts)),
        "preview_video_dir": str(video_dir) if args.preview_videos else None,
    }
    (args.output_dir / "generation_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
