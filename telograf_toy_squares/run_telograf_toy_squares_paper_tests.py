#!/usr/bin/env python3
"""Paper-style ToySquares horizon sweep for the trained TeLoGraF H64 policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/scratch/gilbreth/zoellner/guided-diffusion/outputs/telograf/.cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/scratch/gilbreth/zoellner/guided-diffusion/outputs/telograf/.cache")

import imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/scratch/gilbreth/zoellner/guided-diffusion")
DEFAULT_CHECKPOINT = (
    SCRATCH_ROOT
    / "outputs/telograf/toy_squares/runs/toy_squares_reach_h64_full_10k_telograf/checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    SCRATCH_ROOT
    / "outputs/telograf/toy_squares_rollouts/paper_tests/telograf_h64_paper_test"
)

for path in [REPO_ROOT, REPO_ROOT / "robomimic", REPO_ROOT / "TeLoGraF" / "code"]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gym  # noqa: E402
import robomimic.envs  # noqa: F401,E402

from telograf_toy_squares.run_telograf_toy_squares_rollouts import (  # noqa: E402
    TelografToySquaresPolicy,
    obs_to_state,
    paper_early_decision_setup_state,
    render_frame,
)
from telograf_toy_squares.toy_specs import DEFAULT_RADIUS, evaluate_spec_sequence, label_states, spec_to_vector  # noqa: E402
from toy_squares.baselines.paper_horizon_test import (  # noqa: E402
    chain_for_horizon,
    chain_tag,
    flat_state_from_low_level,
    jsonable,
    labels_from_state,
    obs_snapshot,
    plot_aggregate,
    plot_method_overlay,
    plot_single_rollout,
    reached_label_from_state,
    save_rollout_artifacts,
    summarize_results,
    upsert_summary,
)


@dataclass
class TelografPaperConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    n_rollouts: int = 20
    env_horizon: int = 64
    max_ltl_horizon: int = 5
    seed_start: int = 0
    radius: float = DEFAULT_RADIUS
    chain_base: str = "blue,yellow,green,red"
    setup_variant: str = "compact_deterministic"
    compact_block_scale: float = 0.55
    method_name: str = "telograf_h64_open_loop"
    append_existing_aggregate: bool = True
    overwrite_existing_rollouts: bool = False
    cpu: bool = True
    record_video: bool = True
    video_size: int = 256
    video_fps: int = 10


def reseed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_env():
    return gym.make("TouchCube")


def reset_to_setup(env, setup_state: np.ndarray):
    env.reset()
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset_to"):
        return env.unwrapped.reset_to(np.asarray(setup_state, dtype=np.float32).copy())
    if hasattr(env, "reset_to"):
        return env.reset_to(np.asarray(setup_state, dtype=np.float32).copy())
    return env.reset(np.asarray(setup_state, dtype=np.float32).copy())


def make_chain_spec(chain: Sequence[str], radius: float) -> Dict:
    labels = [str(label) for label in chain]
    if len(labels) == 1:
        return {
            "id": f"eventual_{labels[0]}",
            "type": "eventual",
            "labels": labels,
            "formula": f"F reach_{labels[0]}",
            "radius": float(radius),
        }
    return {
        "id": f"sequence_{'_'.join(labels)}",
        "type": "sequence",
        "labels": labels,
        "formula": " -> ".join(f"reach_{label}" for label in labels),
        "radius": float(radius),
    }


class TelografPaperRunner:
    def __init__(self, config: TelografPaperConfig):
        self.config = config
        device = torch.device("cpu" if config.cpu or not torch.cuda.is_available() else "cuda")
        self.device = device
        self.policy = TelografToySquaresPolicy(Path(config.checkpoint), device)
        self.env = make_env()
        self.setup_state = paper_early_decision_setup_state(
            deterministic=True,
            setup_variant=config.setup_variant,
            compact_block_scale=float(config.compact_block_scale),
        )

    def sample_for_chain(self, state: np.ndarray, chain: Sequence[str]) -> tuple[np.ndarray, np.ndarray, Dict]:
        spec = make_chain_spec(chain, self.config.radius)
        spec_vec = spec_to_vector(spec)
        if len(spec_vec) != self.policy.spec_dim:
            raise ValueError(f"Spec dim mismatch: expected {self.policy.spec_dim}, got {len(spec_vec)}")
        state_norm = (state - self.policy.stats["state_mean"]) / self.policy.stats["state_std"]
        cond_np = np.concatenate([state_norm, spec_vec.astype(np.float32)], axis=0).astype(np.float32)
        cond = torch.from_numpy(cond_np[None, :]).to(self.device)
        with torch.no_grad():
            sample = self.policy.diffuser.conditional_sample(
                cond,
                args=SimpleNamespace(flow_pattern=13),
            ).trajectories[0]
        transitions_norm = sample.detach().cpu().numpy().astype(np.float32)
        transitions = transitions_norm * self.policy.stats["transition_std"] + self.policy.stats["transition_mean"]
        actions = transitions[:, self.policy.state_dim : self.policy.state_dim + self.policy.action_dim].astype(np.float32)
        return actions, transitions.astype(np.float32), spec

    def rollout(self, chain: Sequence[str], env_seed: int, rollout_seed: int, run_dir: Path) -> Dict:
        reseed(env_seed)
        setup_state = self.setup_state.copy()
        reseed(rollout_seed)
        obs = reset_to_setup(self.env, setup_state)
        initial_state = obs_to_state(obs)
        planned_actions, planned_transitions, spec = self.sample_for_chain(initial_state, chain)
        planned_ok, planned_score = evaluate_spec_sequence(spec, planned_transitions[:, : self.policy.state_dim])

        low_level_obs = [obs_snapshot(obs)]
        states = [initial_state.copy()]
        actions, rewards, frames, reaches = [], [], [], []
        if self.config.record_video:
            frames.append(render_frame(self.env, self.config.video_size))
        chain_pos = 0
        steps = 0
        records = [
            {
                "t": 0,
                "stage": 0,
                "target": chain[0],
                "remaining_chain": list(chain),
                "imagined_telograf_success": bool(planned_ok),
                "imagined_telograf_score": float(planned_score),
                "robustness_by_block": labels_from_state(initial_state, self.config.radius),
            }
        ]

        for action in planned_actions[: int(self.config.env_horizon)]:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            obs, reward, done, info = self.env.step(action)
            current_state = flat_state_from_low_level(obs_snapshot(obs))
            states.append(current_state.copy())
            low_level_obs.append(obs_snapshot(obs))
            actions.append(action.copy())
            rewards.append(float(reward))
            steps += 1
            if self.config.record_video:
                frames.append(render_frame(self.env, self.config.video_size))

            if chain_pos < len(chain):
                reached, robustness = reached_label_from_state(current_state, chain[chain_pos], self.config.radius)
                if reached:
                    reaches.append(
                        {
                            "t": int(steps),
                            "stage": int(chain_pos),
                            "label": chain[chain_pos],
                            "robustness": float(robustness),
                        }
                    )
                    chain_pos += 1
                    if chain_pos >= len(chain):
                        break
            if bool(done) or float(reward) < 0.0:
                break

        actual_states = np.asarray(states, dtype=np.float32)
        actual_ok, actual_score = evaluate_spec_sequence(spec, actual_states)
        complete = chain_pos >= len(chain)
        final_state = actual_states[-1]
        result = {
            "method": self.config.method_name,
            "chain": list(chain),
            "complete": bool(complete),
            "stages": int(chain_pos),
            "steps": int(steps),
            "return": float(np.sum(rewards)),
            "env_seed": int(env_seed),
            "rollout_seed": int(rollout_seed),
            "imagined_telograf_success": bool(planned_ok),
            "imagined_telograf_score": float(planned_score),
            "actual_spec_success": bool(actual_ok),
            "actual_spec_score": float(actual_score),
            "reaches": reaches,
            "records": records,
            "final_robustness_by_block": labels_from_state(final_state, self.config.radius),
        }
        save_rollout_artifacts(run_dir, setup_state, low_level_obs, actions, rewards, result, predicted=planned_transitions)
        np.savez_compressed(
            run_dir / "telograf_trace.npz",
            planned_actions=planned_actions.astype(np.float32),
            planned_transitions=planned_transitions.astype(np.float32),
            actual_states=actual_states.astype(np.float32),
            actual_labels=label_states(actual_states, radius=self.config.radius).astype(np.float32),
        )
        if self.config.record_video and frames:
            imageio.mimsave(run_dir / "rollout.mp4", frames, fps=int(self.config.video_fps))
        plot_single_rollout(run_dir, result, title=f"TeLoGraF H={len(chain)} {' -> '.join(chain)}")
        return result


def run_method_for_horizon(runner: TelografPaperRunner, chain: Sequence[str], horizon_dir: Path, config: TelografPaperConfig) -> Dict:
    method = config.method_name
    method_dir = horizon_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    with open(method_dir / "method_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    results = []
    for idx in range(int(config.n_rollouts)):
        seed = int(config.seed_start) + idx
        run_dir = method_dir / f"rollout_{idx:03d}"
        summary_path = run_dir / "rollout_summary.json"
        if summary_path.exists() and not config.overwrite_existing_rollouts:
            result = json.loads(summary_path.read_text())
        else:
            print(f"horizon={len(chain)} rollout={idx}/{config.n_rollouts} chain={'->'.join(chain)}", flush=True)
            result = runner.rollout(chain, env_seed=seed, rollout_seed=seed, run_dir=run_dir)
        result["run_dir"] = str(run_dir)
        results.append(result)
    overlay_path = plot_method_overlay(method_dir, method, chain, results)
    summary = summarize_results(results, chain, method)
    summary["imagined_telograf_success_rate"] = float(np.mean([r.get("imagined_telograf_success", False) for r in results]))
    summary["actual_spec_success_rate"] = float(np.mean([r.get("actual_spec_success", False) for r in results]))
    summary["overlay_path"] = str(overlay_path)
    with open(method_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2)
    return summary


def run_paper_test(config: TelografPaperConfig) -> List[Dict]:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "paper_test_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    runner = TelografPaperRunner(config)
    aggregate_path = output_dir / "aggregate_summary.json"
    all_summaries: List[Dict] = []
    if config.append_existing_aggregate and aggregate_path.exists():
        all_summaries = json.loads(aggregate_path.read_text())

    for horizon in range(1, int(config.max_ltl_horizon) + 1):
        chain = chain_for_horizon(horizon, config.chain_base)
        horizon_dir = output_dir / f"horizon_{horizon:02d}_{chain_tag(chain)}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        with open(horizon_dir / "sequence.json", "w", encoding="utf-8") as f:
            json.dump({"ltl_horizon": horizon, "chain": chain}, f, indent=2)
        summary = run_method_for_horizon(runner, chain, horizon_dir, config)
        all_summaries = upsert_summary(all_summaries, summary)
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(jsonable(all_summaries), f, indent=2)
        plot_aggregate(output_dir, all_summaries)
    return all_summaries


def parse_args() -> TelografPaperConfig:
    defaults = TelografPaperConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field, value in asdict(defaults).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", dest=field, action="store_true")
            parser.add_argument(f"--no-{field}", dest=field, action="store_false")
            parser.set_defaults(**{field: value})
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    args = parser.parse_args()
    return TelografPaperConfig(**vars(args))


def main() -> None:
    config = parse_args()
    summaries = run_paper_test(config)
    print(json.dumps(jsonable(summaries), indent=2))


if __name__ == "__main__":
    main()
