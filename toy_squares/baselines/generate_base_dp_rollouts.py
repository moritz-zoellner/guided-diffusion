from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from toy_squares.baselines.paper_horizon_test import (
    OursRunner,
    PaperTestConfig,
    flat_state_from_low_level,
    jsonable,
    labels_from_state,
    obs_snapshot,
    plot_method_overlay,
    reseed,
    rollout_setup_state,
    save_rollout_artifacts,
)


########
#
# UNGUIDED DIFFUSION POLICY PAPER TRACES
# This script records short rollouts from the trained base Diffusion Policy
# without any automaton or STL/LTL guidance. The paper plotting notebook uses
# these traces to show the environment layout and what the learned base policy
# tends to do before symbolic steering is applied.
#
#######


def block_hits_from_state(state: np.ndarray, radius: float) -> Dict[str, bool]:
    return {name: robustness > 0.0 for name, robustness in labels_from_state(state, radius).items()}


def run_base_dp_rollout(runner: OursRunner, config: PaperTestConfig, rollout_idx: int, run_dir: Path) -> Dict:
    seed = int(config.seed_start) + int(rollout_idx)
    reseed(seed)
    setup_state = rollout_setup_state(config)
    reseed(seed)
    obs = runner.env.reset_to(setup_state)
    runner.policy.start_episode()

    low_level_obs = [obs_snapshot(obs)]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    records: List[Dict] = []
    first_hits: Dict[str, int] = {}
    action_queue: List[np.ndarray] = []

    old_debug = getattr(runner.policy.policy, "debug_guidance_actions", False)
    runner.policy.policy.debug_guidance_actions = False
    try:
        for t in range(int(config.env_horizon)):
            if not action_queue:
                obs_tensor = runner.policy._prepare_observation(obs)
                with torch.no_grad():
                    chunk_n = runner.policy.policy._get_action_trajectory(obs_dict=obs_tensor).detach()
                chunk_raw = runner.unnormalize_action_sequence(chunk_n).detach().cpu().numpy()[0]
                action_queue.extend(np.asarray(chunk_raw, dtype=np.float32))

            action = np.asarray(action_queue.pop(0), dtype=np.float32)
            obs, reward, _done, info = runner.env.step(action)
            low_level_obs.append(obs_snapshot(obs))
            actions.append(action.copy())
            rewards.append(float(reward))

            state = flat_state_from_low_level(obs_snapshot(obs))
            hits = block_hits_from_state(state, config.radius)
            for block_name, hit in hits.items():
                if hit and block_name not in first_hits:
                    first_hits[block_name] = int(t + 1)
            records.append(
                {
                    "t": int(t + 1),
                    "hits": hits,
                    "first_contact": int(info.get("cube_contacted", -1)),
                    "robustness_by_block": labels_from_state(state, config.radius),
                }
            )
    finally:
        runner.policy.policy.debug_guidance_actions = old_debug

    result = {
        "method": "base_dp_unguided",
        "chain": [],
        "complete": False,
        "stages": int(len(first_hits)),
        "steps": int(len(actions)),
        "return": float(np.sum(rewards)),
        "env_seed": int(seed),
        "rollout_seed": int(seed),
        "first_hits": first_hits,
        "records": records,
    }
    save_rollout_artifacts(run_dir, setup_state, low_level_obs, actions, rewards, result)
    return result


def summarize(results: List[Dict], output_dir: Path) -> Dict:
    summary = {
        "method": "base_dp_unguided",
        "n": int(len(results)),
        "mean_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "hit_counts": {
            name: int(sum(name in r.get("first_hits", {}) for r in results))
            for name in ["blue", "red", "green", "yellow"]
        },
        "records": results,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate short unguided base-DP traces for paper plots.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=12)
    parser.add_argument("--env_horizon", type=int, default=80)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--setup_variant", type=str, default="compact_deterministic")
    parser.add_argument("--compact_block_scale", type=float, default=0.55)
    parser.add_argument("--deterministic_setup", action="store_true", default=True)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--dp_checkpoint", type=str, default=PaperTestConfig.dp_checkpoint)
    parser.add_argument("--automaton_run_dir", type=str, default=PaperTestConfig.automaton_run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PaperTestConfig(
        output_dir=str(output_dir),
        dp_checkpoint=args.dp_checkpoint,
        automaton_run_dir=args.automaton_run_dir,
        n_rollouts=int(args.n_rollouts),
        env_horizon=int(args.env_horizon),
        seed_start=int(args.seed_start),
        deterministic_setup=bool(args.deterministic_setup),
        setup_variant=str(args.setup_variant),
        compact_block_scale=float(args.compact_block_scale),
        radius=float(args.radius),
        methods="ours",
    )
    with open(output_dir / "base_dp_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    runner = OursRunner(config)
    results = []
    for idx in tqdm(range(int(config.n_rollouts)), desc="base DP traces", dynamic_ncols=True):
        run_dir = output_dir / f"rollout_{idx:03d}"
        summary_path = run_dir / "rollout_summary.json"
        if summary_path.exists():
            result = json.loads(summary_path.read_text())
        else:
            result = run_base_dp_rollout(runner, config, idx, run_dir)
        result["run_dir"] = str(run_dir)
        results.append(result)

    summarize(results, output_dir)
    if results:
        plot_method_overlay(output_dir, "base_dp_unguided", [], results)


if __name__ == "__main__":
    main()
