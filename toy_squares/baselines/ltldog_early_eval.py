from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toy_squares.baselines.ltldog_toy import FormulaSpec, ToyLTLRobustness, make_env, obs_from_env_dict  # noqa: E402
from toy_squares.baselines.ltldog_train import LABEL_COLORS, LABEL_TO_BLOCK_SLICE, LTLDogPlanner, load_ltldog_planner  # noqa: E402
from toy_squares.toy_squares_utils import _draw_blocks as draw_toy_blocks  # noqa: E402
from toy_squares.toy_squares_utils import _flip_y as flip_toy_y  # noqa: E402
from toy_squares.toy_squares_utils import early_decision_cube_setup  # noqa: E402


########
#
# EARLY-DECISION EVAL CONFIGURATION
# This script is intentionally narrow: it validates the easy formulas
# F at_blue/red/green/yellow in the same early-decision reset workflow used by
# automaton_guidance.ipynb. Each formula gets N seeded rollouts and one overlay
# plot with a time/progress color gradient.
#
#######


EASY_FORMULAS = [
    FormulaSpec("F_blue", "eventually", ("blue",), "eventually visit blue"),
    FormulaSpec("F_red", "eventually", ("red",), "eventually visit red"),
    FormulaSpec("F_green", "eventually", ("green",), "eventually visit green"),
    FormulaSpec("F_yellow", "eventually", ("yellow",), "eventually visit yellow"),
]


@dataclass
class EarlyEvalConfig:
    checkpoint: str = "outputs/toy_squares_rollouts/baseline_ltldog/training/h128_full_diffuser_train3000_cpu_full/best.pt"
    output_dir: str = "outputs/toy_squares_rollouts/baseline_ltldog/rollouts/fix_02_early_jiggle_easy_targets_exact_ps_scale0p01_g5_exec8_n10_blue_red_yellow"
    formulas: str = "F_blue,F_red,F_green,F_yellow"
    n_rollouts: int = 10
    env_horizon: int = 128
    execute_steps: int = 8
    batch_size: int = 4
    guidance_scale: float = 0.01
    n_guide_steps: int = 5
    t_stopgrad: int = 2
    guidance_mode: str = "ps"
    scale_grad_by_std: bool = False
    guidance_threshold: float = 0.0
    diffusion_steps: int = 100
    seed_start: int = 0
    deterministic_setup: bool = False
    device: str = "auto"
    radius: float = 0.2
    tau: float = 0.05


########
#
# NOTEBOOK-MATCHING SEED AND OBSERVATION HELPERS
# The notebook seeds Python/NumPy/Torch before building the early setup, then
# reseeds before rollout sampling. We do the same here so the setup jiggle and
# the Diffuser sampling randomness are controlled separately.
#
#######


def reseed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def obs_snapshot(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: to_numpy(obs[key]).copy() for key in ("states", "agent_pos") if key in obs}


def stack_or_object(values: Sequence[np.ndarray]) -> np.ndarray:
    values = list(values)
    if not values:
        return np.array([])
    try:
        return np.stack(values)
    except Exception:
        return np.asarray(values, dtype=object)


def latest_low_level_vector(value) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=np.float32)
    if array.ndim >= 2:
        array = array[-1]
    return array.reshape(-1)


########
#
# EXACT EVENTUALLY DIAGNOSTICS
# The smooth robustness is still used for gradients inside LTLDoG-S. Reporting
# and early stopping use exact distance-to-block robustness so a plot that
# visually touches blue also counts as F_blue when it crosses the 0.2 radius.
#
#######


def target_distance_trace(observations: np.ndarray, label: str) -> np.ndarray:
    agent = observations[:, 0:2]
    block = observations[:, LABEL_TO_BLOCK_SLICE[label]]
    return np.linalg.norm(agent - block, axis=-1)


def reached_label(observations: np.ndarray, label: str, radius: float) -> Tuple[float, bool, int]:
    distances = target_distance_trace(observations, label)
    best_idx = int(np.argmin(distances))
    robustness = float(radius - distances[best_idx])
    return robustness, bool(robustness > 0.0), best_idx


########
#
# SINGLE ROLLOUT
# LTLDoG samples a full H-step guided trajectory, executes only the first
# execute_steps actions, observes the real environment state, and replans. This
# is the receding-horizon correction loop we want to validate on easy goals.
#
#######


def run_one_rollout(
    planner: LTLDogPlanner,
    formula: FormulaSpec,
    env_seed: int,
    rollout_seed: int,
    config: EarlyEvalConfig,
    formula_dir: Path,
    rollout_idx: int,
) -> Dict:
    target_label = formula.labels[0]
    robustness = ToyLTLRobustness(formula, radius=config.radius, tau=config.tau)

    reseed(env_seed)
    setup_state = to_numpy(early_decision_cube_setup(deterministic=config.deterministic_setup)).copy()
    reseed(rollout_seed)

    env = make_env()
    env.reset()
    obs = env.unwrapped.reset_to(setup_state)
    current_obs = obs_from_env_dict(obs)

    low_level_obs = [obs_snapshot(obs)]
    flat_obs = [current_obs.copy()]
    actions, rewards, contacts, records = [], [], [], []
    predicted_first = None
    predicted_values = []

    for t in range(int(config.env_horizon)):
        if t % int(config.execute_steps) != 0:
            continue

        samples, values = planner.sample_plan(
            current_obs,
            formula,
            batch_size=config.batch_size,
            guidance_scale=config.guidance_scale,
            n_guide_steps=config.n_guide_steps,
            t_stopgrad=config.t_stopgrad,
            guidance_mode=config.guidance_mode,
            scale_grad_by_std=config.scale_grad_by_std,
            guidance_threshold=config.guidance_threshold,
            radius=config.radius,
            tau=config.tau,
        )
        sample = samples[0]
        predicted_values.append(float(values[0]))
        if predicted_first is None:
            predicted_first = sample.copy()

        pred_obs = sample[:, 2:]
        pred_robustness, pred_satisfied = robustness.hard_satisfaction(pred_obs)
        current_robustness, current_satisfied, _ = reached_label(np.asarray(flat_obs, dtype=np.float32), target_label, config.radius)
        records.append(
            {
                "t": int(t),
                "target": target_label,
                "guide_value": float(values[0]),
                "predicted_robustness": float(pred_robustness),
                "predicted_satisfied": bool(pred_satisfied),
                "actual_so_far_robustness": float(current_robustness),
                "actual_so_far_satisfied": bool(current_satisfied),
            }
        )

        for action in sample[: int(config.execute_steps), :2]:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            obs, reward, _done, info = env.step(action)
            current_obs = obs_from_env_dict(obs)
            low_level_obs.append(obs_snapshot(obs))
            flat_obs.append(current_obs.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            contacts.append(int(info.get("cube_contacted", -1)))

            actual_robustness, actual_satisfied, best_step = reached_label(
                np.asarray(flat_obs, dtype=np.float32),
                target_label,
                config.radius,
            )
            if actual_satisfied or len(actions) >= int(config.env_horizon):
                break
        if actual_satisfied or len(actions) >= int(config.env_horizon):
            break

    flat_obs_arr = np.asarray(flat_obs, dtype=np.float32)
    actual_robustness, actual_satisfied, best_step = reached_label(flat_obs_arr, target_label, config.radius)
    predicted_arr = predicted_first.astype(np.float32) if predicted_first is not None else np.zeros((0, 12), dtype=np.float32)
    pred_obs = predicted_arr[:, 2:] if len(predicted_arr) else np.zeros((0, 10), dtype=np.float32)
    predicted_robustness, predicted_satisfied = robustness.hard_satisfaction(pred_obs) if len(pred_obs) else (float("nan"), False)

    run_dir = formula_dir / f"rollout_{rollout_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "setup_state.npy", setup_state)
    np.savez_compressed(
        run_dir / "low_level_obs.npz",
        states=stack_or_object([obs.get("states") for obs in low_level_obs]),
        agent_pos=stack_or_object([obs.get("agent_pos") for obs in low_level_obs]),
    )
    np.savez_compressed(
        run_dir / "ltldog_trace.npz",
        observations=flat_obs_arr,
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        contacts=np.asarray(contacts, dtype=np.int32),
        predicted=predicted_arr,
    )

    result = {
        "formula": asdict(formula),
        "rollout_idx": int(rollout_idx),
        "env_seed": int(env_seed),
        "rollout_seed": int(rollout_seed),
        "run_dir": str(run_dir),
        "steps": int(len(actions)),
        "return": float(np.sum(rewards)),
        "contacts": contacts,
        "first_contact": int(next((c for c in contacts if c >= 0), -1)),
        "best_step": int(best_step),
        "actual_robustness": float(actual_robustness),
        "actual_satisfied": bool(actual_satisfied),
        "predicted_guidance_value": float(predicted_values[0]) if predicted_values else None,
        "predicted_robustness": float(predicted_robustness),
        "predicted_satisfied": bool(predicted_satisfied),
        "records": records,
    }
    with open(run_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


########
#
# NOTEBOOK-STYLE OVERLAY PLOTS
# Each formula gets a single plot with all 10 rollouts. The path color follows
# turbo from start to finish, matching the chunk-progress visual language used
# in automaton_guidance.ipynb.
#
#######


def setup_blocks_and_agent(setup_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = setup_state[2:10].reshape(4, 2) / 256.0 - 1.0
    agent = setup_state[:2] / 256.0 - 1.0
    angles = np.asarray(setup_state[10:14], dtype=float) if setup_state.shape[0] >= 14 else np.zeros(4)
    return flip_toy_y(blocks), flip_toy_y(agent), -angles


def plot_formula_overlay(formula_dir: Path, formula: FormulaSpec, results: List[Dict], config: EarlyEvalConfig) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=180, constrained_layout=True)
    cmap = plt.get_cmap("turbo")
    palette = [LABEL_COLORS["blue"], LABEL_COLORS["red"], LABEL_COLORS["green"], LABEL_COLORS["yellow"]]

    for result in results:
        setup = np.load(Path(result["run_dir"]) / "setup_state.npy")
        blocks, _, angles = setup_blocks_and_agent(setup)
        draw_toy_blocks(ax, blocks, angles, palette, radius=0.11, alpha=0.14, zorder=3)

    for result in results:
        trace = np.load(Path(result["run_dir"]) / "ltldog_trace.npz")
        obs = trace["observations"]
        trajectory = flip_toy_y(obs[:, 0:2])
        if len(trajectory) > 1:
            segments = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
            line = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0.0, 1.0), linewidths=1.65, alpha=0.62, zorder=2)
            line.set_array(np.linspace(0.0, 1.0, len(segments)))
            ax.add_collection(line)
        final_color = "#111111" if result["actual_satisfied"] else "#b00020"
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], s=12, marker="o", c=final_color, alpha=0.78, edgecolors="none", zorder=6)

    first_setup = np.load(Path(results[0]["run_dir"]) / "setup_state.npy")
    blocks, start_agent, angles = setup_blocks_and_agent(first_setup)
    draw_toy_blocks(ax, blocks, angles, palette, radius=0.11, alpha=0.95, zorder=4)
    ax.scatter([start_agent[0]], [start_agent[1]], s=210, marker="o", c="#2f2f2f", alpha=0.92, edgecolors="white", linewidths=0.4, zorder=5)

    success_rate = float(np.mean([r["actual_satisfied"] for r in results])) if results else 0.0
    avg_steps = float(np.mean([r["steps"] for r in results])) if results else 0.0
    ax.set_title(f"LTLDoG Early {formula.name}: {success_rate:.2f} success, {avg_steps:.1f} steps", fontsize=11, fontweight="bold")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    frame = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#9a9a9a", linewidth=2.0, zorder=10)
    ax.add_patch(frame)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("rollout progress", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plot_path = formula_dir / "early_overlay.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


########
#
# BATCH EVALUATION
# We run the four easy formulas separately and write per-formula JSON summaries,
# per-rollout raw traces, and an all-formula summary for quick comparison.
#
#######


def select_formulas(selection: str) -> List[FormulaSpec]:
    names = [item.strip() for item in selection.split(",") if item.strip()]
    by_name = {formula.name: formula for formula in EASY_FORMULAS}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown easy formula(s): {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


def summarize(results: List[Dict]) -> Dict:
    return {
        "n": int(len(results)),
        "satisfaction_rate": float(np.mean([r["actual_satisfied"] for r in results])) if results else 0.0,
        "predicted_satisfaction_rate": float(np.mean([r["predicted_satisfied"] for r in results])) if results else 0.0,
        "mean_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "mean_actual_robustness": float(np.mean([r["actual_robustness"] for r in results])) if results else float("nan"),
        "mean_predicted_robustness": float(np.mean([r["predicted_robustness"] for r in results])) if results else float("nan"),
        "contact_rate": float(np.mean([r["first_contact"] >= 0 for r in results])) if results else 0.0,
    }


def run_eval(config: EarlyEvalConfig) -> Dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "early_eval_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    planner = load_ltldog_planner(config.checkpoint, device=config.device, diffusion_steps=config.diffusion_steps)
    formulas = select_formulas(config.formulas)
    all_summaries = {}

    for formula in formulas:
        formula_dir = output_dir / formula.name
        formula_dir.mkdir(parents=True, exist_ok=True)
        results = []
        iterator = tqdm(range(int(config.n_rollouts)), desc=formula.name, dynamic_ncols=True)
        for idx in iterator:
            seed = int(config.seed_start) + idx
            result = run_one_rollout(
                planner,
                formula,
                env_seed=seed,
                rollout_seed=seed,
                config=config,
                formula_dir=formula_dir,
                rollout_idx=idx,
            )
            results.append(result)
            iterator.set_postfix(success=float(np.mean([r["actual_satisfied"] for r in results])))

        summary = summarize(results)
        plot_path = plot_formula_overlay(formula_dir, formula, results, config)
        payload = {"summary": summary, "plot_path": str(plot_path), "rollouts": results}
        with open(formula_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        all_summaries[formula.name] = summary

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    return all_summaries


########
#
# CLI
#
#######


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    action = "store_false" if default else "store_true"
    parser.add_argument(f"--{name}", action=action, default=default)


def parse_args() -> EarlyEvalConfig:
    defaults = EarlyEvalConfig()
    parser = argparse.ArgumentParser(description="Evaluate LTLDoG on early-decision Toy Squares easy goals")
    for field, value in asdict(defaults).items():
        if isinstance(value, bool):
            add_bool_flag(parser, field, value)
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    return EarlyEvalConfig(**vars(parser.parse_args()))


def main() -> None:
    summaries = run_eval(parse_args())
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
