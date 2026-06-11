#!/usr/bin/env python3
"""Generate real TouchCube rollouts for ToySquares length-chain TeLoGraF data."""

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
from toy_squares.collect_scripted_data_pymunk import ReachingPolicyKey, squeeze_obs_dict  # noqa: E402


SCRATCH_ROOT = Path("/scratch/gilbreth/zoellner/guided-diffusion")
DEFAULT_OUTPUT_DIR = SCRATCH_ROOT / "outputs/telograf/toy_squares/data/real_chains_len5_h128_10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-rollouts", type=int, default=10000)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-steps-per-goal", type=int, default=34)
    parser.add_argument("--policy-noise-scale", type=float, default=0.03)
    parser.add_argument("--policy-max-waypoints", type=int, default=1)
    parser.add_argument(
        "--policy-step-stride",
        type=int,
        default=2,
        help="Subsample the scripted curve after planning. This keeps the same path but fits chains into H128.",
    )
    parser.add_argument("--curve-stride", type=int, default=None, help="Alias for --policy-step-stride.")
    parser.add_argument("--max-attempts-factor", type=int, default=60)
    parser.add_argument("--preview-videos", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=10)
    return parser.parse_args()


def make_env():
    return gym.make("TouchCube")


def all_no_repeat_chains(labels: Sequence[str], length: int) -> List[Tuple[str, ...]]:
    return [
        tuple(chain)
        for chain in itertools.product(labels, repeat=int(length))
        if all(chain[i] != chain[i - 1] for i in range(1, len(chain)))
    ]


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
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)
    return env.reset()


def policy_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        out[key] = arr[..., None] if arr.ndim == 1 else arr
    return out


def decimate_policy_curve(policy: ReachingPolicyKey, stride: int) -> None:
    stride = max(1, int(stride))
    curve = np.asarray(policy.curve, dtype=np.float32)
    if curve.ndim != 2 or curve.shape[0] <= 2 or stride <= 1:
        policy.curve = curve
        return
    keep = list(range(0, curve.shape[0], stride))
    if keep[-1] != curve.shape[0] - 1:
        keep.append(curve.shape[0] - 1)
    policy.curve = curve[keep].astype(np.float32)


def ordered_stages(states: np.ndarray, chain: Sequence[str], radius: float) -> int:
    labels = label_states(states, radius=radius)
    pos = 0
    for row in labels:
        if pos < len(chain) and row[LABEL_NAMES.index(chain[pos])] > 0.5:
            pos += 1
    return int(pos)


def rollout_chain(env, chain: Sequence[str], args: argparse.Namespace, rng: np.random.Generator, record_video: bool):
    obs = reset_env(env, rng)
    states = [obs_to_state(obs)]
    actions: List[np.ndarray] = []
    contacts: List[int] = []
    frames: List[np.ndarray] = []
    reaches = []
    reached_targets = set()
    policy = ReachingPolicyKey(
        num_cubes=len(LABEL_NAMES),
        noise=args.policy_noise_scale > 0,
        noise_scale=float(args.policy_noise_scale),
        max_waypoints=int(args.policy_max_waypoints),
    )
    if record_video:
        frames.append(env.render(mode="rgb_array"))

    wrong_contact = False
    for stage, label in enumerate(chain):
        target_idx = int(LABEL_NAMES.index(label))
        policy.start_episode(target_idx, squeeze_obs_dict(obs))
        decimate_policy_curve(policy, int(args.policy_step_stride))

        stage_reached = False
        for _ in range(int(args.max_steps_per_goal)):
            action = np.asarray(policy(ob=policy_obs(obs)), dtype=np.float32).reshape(-1)[:2]
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            next_obs, reward, done, info = env.step(action)
            contact = int(info.get("cube_contacted", -1))

            actions.append(action)
            contacts.append(contact)
            states.append(obs_to_state(next_obs))
            if record_video:
                frames.append(env.render(mode="rgb_array"))

            if contact == target_idx:
                reaches.append({"stage": int(stage), "label": str(label), "t": int(len(actions) - 1)})
                reached_targets.add(target_idx)
                obs = next_obs
                stage_reached = True
                break
            if contact >= 0 and contact not in reached_targets:
                wrong_contact = True
                obs = next_obs
                break

            obs = next_obs
            if len(actions) >= int(args.horizon):
                break

        if wrong_contact or not stage_reached or len(actions) >= int(args.horizon):
            break

    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)
    spec = make_sequence_spec(chain, args.radius)
    ok, score = evaluate_spec_sequence(spec, states_np)
    stages = ordered_stages(states_np, chain, args.radius)
    success = bool(ok and stages == len(chain) and len(reaches) == len(chain) and not wrong_contact)
    return {
        "states": states_np,
        "actions": actions_np,
        "contacts": np.asarray(contacts, dtype=np.int64),
        "spec": spec,
        "chain": list(chain),
        "success": success,
        "score": float(score),
        "stages": int(stages),
        "contact_stages": int(len(reaches)),
        "wrong_contact": bool(wrong_contact),
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
        hold_action = states[-1, 0:2].astype(np.float32)
        actions = np.concatenate([actions, np.repeat(hold_action[None, :], pad_n, axis=0)], axis=0)
    if states.shape[0] < horizon + 1:
        states = np.concatenate([states, np.repeat(states[-1:], horizon + 1 - states.shape[0], axis=0)], axis=0)
    states = states[: horizon + 1].astype(np.float32)
    actions = actions[:horizon].astype(np.float32)
    spec = dict(raw["spec"])
    return {
        "env": "toy_squares",
        "spec_id": spec["id"],
        "spec": spec,
        "formula": str(spec.get("formula", "")),
        "state": states[0].astype(np.float32),
        "trajs": states,
        "us": actions,
        "actions": actions,
        "obs": states,
        "split": split,
        "chain": list(raw["chain"]),
        "success": bool(raw["success"]),
        "score": np.asarray([float(raw["score"])], dtype=np.float32),
        "stages": int(raw["stages"]),
        "contact_stages": int(raw["contact_stages"]),
        "contacts": np.asarray(raw["contacts"], dtype=np.int64),
        "padded_steps": int(max(0, horizon - raw_steps)),
        "raw_steps": int(raw_steps),
        "extra": {
            "reaches": list(raw["reaches"]),
            "wrong_contact": bool(raw["wrong_contact"]),
        },
    }


def write_dataset(path: Path, records: List[Dict], meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=np.asarray(records, dtype=object), meta=json.dumps(meta))


def main() -> None:
    args = parse_args()
    if args.curve_stride is not None:
        args.policy_step_stride = int(args.curve_stride)
    horizons = sorted(set(int(h) for h in (args.horizons if args.horizons is not None else [args.horizon])))
    args.horizon = min(horizons)
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
    while len(raw_rollouts) < args.n_rollouts and attempts < max_attempts:
        chain = ordered_chains[len(raw_rollouts) % len(ordered_chains)]
        record_video = len(raw_rollouts) < int(args.preview_videos)
        attempts += 1
        raw = rollout_chain(env, chain, args, rng, record_video=record_video)
        if not raw["success"] or len(raw["actions"]) > int(args.horizon):
            if attempts <= 10 or attempts % 100 == 0:
                print(
                    "skip "
                    f"attempt={attempts} saved={len(raw_rollouts)} chain={'->'.join(chain)} "
                    f"success={raw['success']} stages={raw['stages']} contacts={raw['contact_stages']} "
                    f"steps={len(raw['actions'])} wrong_contact={raw['wrong_contact']}",
                    flush=True,
                )
            continue
        raw_rollouts.append(raw)
        idx = len(raw_rollouts) - 1
        if record_video:
            imageio.mimsave(video_dir / f"preview_{idx:03d}_{'_'.join(chain)}.mp4", raw["frames"], fps=int(args.video_fps))
        if len(raw_rollouts) % 100 == 0 or len(raw_rollouts) <= 4:
            print(
                f"saved={len(raw_rollouts)}/{args.n_rollouts} attempts={attempts} "
                f"chain={'->'.join(chain)} steps={len(raw['actions'])}",
                flush=True,
            )

    if len(raw_rollouts) < args.n_rollouts:
        raise RuntimeError(f"Only collected {len(raw_rollouts)} successful real rollouts after {attempts} attempts")

    chain_counts = {"->".join(raw["chain"]): 0 for raw in raw_rollouts}
    for raw in raw_rollouts:
        chain_counts["->".join(raw["chain"])] = chain_counts.get("->".join(raw["chain"]), 0) + 1
    possible_chains = len(chains)
    samples_per_chain_floor = args.n_rollouts // possible_chains
    samples_per_chain_remainder = args.n_rollouts % possible_chains

    valid_period = max(1, int(round(1.0 / args.val_ratio)))
    for horizon in horizons:
        records = []
        for idx, raw in enumerate(raw_rollouts):
            split = "valid" if idx % valid_period == 0 else "train"
            records.append(pad_record(raw, int(horizon), split))

        meta = {
            "dataset_type": "toy_squares_long_chain_real_env",
            "horizon": int(horizon),
            "collection_horizon": int(args.horizon),
            "n_rollouts": int(args.n_rollouts),
            "chain_length": int(args.chain_length),
            "label_names": list(LABEL_NAMES),
            "possible_chains": int(possible_chains),
            "samples_per_chain_floor": int(samples_per_chain_floor),
            "samples_per_chain_remainder": int(samples_per_chain_remainder),
            "radius": float(args.radius),
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "policy_noise_scale": float(args.policy_noise_scale),
            "policy_max_waypoints": int(args.policy_max_waypoints),
            "policy_step_stride": int(args.policy_step_stride),
            "max_steps_per_goal": int(args.max_steps_per_goal),
            "chain_counts": chain_counts,
        }
        write_dataset(args.output_dir / f"data_h{int(horizon)}.npz", records, meta)

    stats = {
        "output_dir": str(args.output_dir),
        "data": [str(args.output_dir / f"data_h{int(horizon)}.npz") for horizon in horizons],
        "n_rollouts": int(args.n_rollouts),
        "attempts": int(attempts),
        "success_collection_rate": float(args.n_rollouts / max(1, attempts)),
        "chain_length": int(args.chain_length),
        "possible_chains": int(possible_chains),
        "samples_per_chain_floor": int(samples_per_chain_floor),
        "samples_per_chain_remainder": int(samples_per_chain_remainder),
        "horizon": int(args.horizon),
        "horizons": horizons,
        "mean_raw_steps": float(np.mean([len(raw["actions"]) for raw in raw_rollouts])),
        "max_raw_steps": int(max(len(raw["actions"]) for raw in raw_rollouts)),
        "min_raw_steps": int(min(len(raw["actions"]) for raw in raw_rollouts)),
        "preview_video_dir": str(video_dir) if args.preview_videos else None,
    }
    (args.output_dir / "generation_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
