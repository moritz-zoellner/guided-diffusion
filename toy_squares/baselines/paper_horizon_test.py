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
ROBOMIMIC_ROOT = REPO_ROOT / "robomimic"
if str(ROBOMIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOMIMIC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robomimic.utils.python_utils as PyUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402

from toy_squares.baselines.ltldog_toy import ACTION_DIM, FormulaSpec, make_env, obs_from_env_dict  # noqa: E402
from toy_squares.baselines.ltldog_train import LTLDogPlanner, load_ltldog_planner  # noqa: E402
from toy_squares.toy_squares_utils import _draw_blocks as draw_toy_blocks  # noqa: E402
from toy_squares.toy_squares_utils import _flip_y as flip_toy_y  # noqa: E402
from toy_squares.toy_squares_utils import early_decision_cube_setup, load_automaton_model_for_eval  # noqa: E402


########
#
# PAPER HORIZON TEST CONFIG
# This file runs matched early-decision rollouts for our post-sample guided
# diffusion policy and the LTLDoG-S baseline. The output tree is:
# paper_test/horizon_XX_<chain>/{ours,ltldog}/...
#
# The comparison deliberately preserves each method's natural control surface:
#   - Ours is an 8-action Diffusion Policy, so it samples/guides one executable
#     action chunk at a time and replans after each reached symbolic stage.
#   - LTLDoG is a full-trajectory Diffuser baseline, so it samples/guides one
#     H-step state-action trajectory and executes the generated actions once.
#
# The main paper result folder uses the defaults in PaperTestConfig below. If
# you rerun the script without overrides, it targets main_result_LTLDOG and will
# reuse existing rollout folders unless overwrite_existing_rollouts is enabled.
#
#######


STATE_BLOCK_NAMES = ["blue", "red", "green", "yellow"]
LABEL_NAMES = ["at_green", "at_blue", "at_red", "at_yellow"]
LABEL_TO_STATE_BLOCK_IDX = [2, 0, 1, 3]
STATE_BLOCK_TO_LABEL_IDX = [1, 2, 0, 3]
LABEL_NAME_TO_IDX = {"green": 0, "blue": 1, "red": 2, "yellow": 3}
LABEL_IDX_TO_NAME = {idx: name for name, idx in LABEL_NAME_TO_IDX.items()}
BLOCK_COLORS = ["#0a84ff", "#ff3b30", "#34c759", "#ffd60a"]


@dataclass
class PaperTestConfig:
    # Output root for the final paper-style horizon scaling experiment.
    output_dir: str = "outputs/toy_squares_rollouts/baseline_ltldog/rollouts/paper_test/main_result_LTLDOG"

    # Our trained 8-action Diffusion Policy checkpoint and learned automaton
    # world model. OursRunner uses both: DP proposes action chunks, automaton
    # gradients steer each chunk toward the current symbolic subgoal.
    dp_checkpoint: str = "/home/moritz/data/diffusion_runs/toy_squares_dp_n500/20260422190540/models/model_epoch_320_best_validation_0.005075077305082232.pth"
    automaton_run_dir: str = "outputs/automaton_world_model/training-run_2026-04-28_17-52-27"

    # LTLDoG full-trajectory Diffuser checkpoint. It predicts H-step
    # state-action trajectories directly and is guided with differentiable LTL
    # robustness during reverse diffusion.
    ltldog_checkpoint: str = "outputs/toy_squares_rollouts/baseline_ltldog/training/h128_full_diffuser_train3000_cpu_full/best.pt"
    n_rollouts: int = 20
    env_horizon: int = 128
    max_ltl_horizon: int = 5
    seed_start: int = 0
    deterministic_setup: bool = True
    methods: str = "ours,ltldog"
    append_existing_aggregate: bool = True
    overwrite_existing_rollouts: bool = False
    stop_ltldog_success_at: float = -1.0
    stop_after_horizon: int = 2
    radius: float = 0.2
    ours_guidance_steps: int = 10
    ours_step_size: float = 3e-4

    # Final LTLDoG tuning we settled on for the compact deterministic paper
    # result: posterior-sampling guidance, low smooth-min tau, moderate scale,
    # and enough guide steps to make imagined satisfaction nontrivial.
    ltldog_batch_size: int = 4
    ltldog_guidance_scale: float = 0.10
    ltldog_n_guide_steps: int = 20
    ltldog_t_stopgrad: int = 2
    ltldog_diffusion_steps: int = 100
    ltldog_guidance_mode: str = "ps"
    ltldog_scale_grad_by_std: bool = False
    ltldog_tau: float = 0.001
    ltldog_freeze_static_blocks: bool = False
    ltldog_method_name: str = "ltldog_h128_compact_full_exec_tau0p001_scale0p10_g20"
    chain_base: str = "blue,yellow,green,red"
    setup_variant: str = "compact_deterministic"
    compact_block_scale: float = 0.55


def parse_chain_base(chain_base: str) -> List[str]:
    base = [item.strip().lower() for item in str(chain_base).split(",") if item.strip()]
    if not base:
        raise ValueError("chain_base must contain at least one block label")
    unknown = [item for item in base if item not in LABEL_NAME_TO_IDX]
    if unknown:
        raise ValueError(f"Unknown labels in chain_base: {unknown}. Valid labels: {sorted(LABEL_NAME_TO_IDX)}")
    return base


def chain_for_horizon(horizon: int, chain_base: str = "blue,red,yellow,green") -> List[str]:
    base = parse_chain_base(chain_base)
    if horizon <= len(base):
        return base[:horizon]
    return [base[i % len(base)] for i in range(horizon)]


def chain_tag(chain: Sequence[str]) -> str:
    return "_".join(chain)


def rollout_setup_state(config: PaperTestConfig) -> np.ndarray:
    # All paper rollouts should start from the same early-decision layout logic.
    # The compact variants scale block coordinates toward the board center,
    # making the horizon-prefix task visually cleaner without changing the
    # state representation or action interface.
    setup_state = to_numpy(early_decision_cube_setup(deterministic=config.deterministic_setup)).copy()
    variant = str(config.setup_variant).strip().lower()
    if variant in {"early", "early_decision"}:
        return setup_state
    if variant in {"compact", "compact_early", "compact_deterministic"}:
        positions = setup_state[:10].reshape(5, 2).astype(np.float32)
        center = np.array([256.0, 256.0], dtype=np.float32)
        positions[1:] = center + float(config.compact_block_scale) * (positions[1:] - center)
        setup_state[:10] = positions.reshape(-1)
        if config.deterministic_setup and setup_state.shape[0] >= 14:
            setup_state[10:14] = 0.0
        return setup_state
    raise ValueError(f"Unknown setup_variant={config.setup_variant!r}")


########
#
# SHARED RUNTIME HELPERS
# These keep seeding, JSON serialization, low-level observation snapshots, and
# exact block-reaching checks identical across both methods.
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


def latest_vector(value) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=np.float32)
    if array.ndim >= 2:
        array = array[-1]
    return array.reshape(-1)


def flat_state_from_low_level(obs: Dict[str, np.ndarray]) -> np.ndarray:
    state = np.concatenate([latest_vector(obs["agent_pos"]), latest_vector(obs["states"])]).astype(np.float32)
    if state.shape != (10,):
        raise ValueError(f"Expected 10D state, got {state.shape}")
    return state


def reached_label_from_state(state: np.ndarray, label_name: str, radius: float) -> Tuple[bool, float]:
    # This is the hard satisfaction primitive used for rollout success. The
    # same radius appears in LTLDoG robustness, automaton labels, and plots.
    block_idx = STATE_BLOCK_NAMES.index(label_name)
    agent = state[0:2]
    block = state[2 + 2 * block_idx : 4 + 2 * block_idx]
    robustness = float(radius - np.linalg.norm(agent - block))
    return robustness > 0.0, robustness


def labels_from_state(state: np.ndarray, radius: float) -> Dict[str, float]:
    return {name: reached_label_from_state(state, name, radius)[1] for name in STATE_BLOCK_NAMES}


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


########
#
# OUR POST-SAMPLE GUIDED DIFFUSION POLICY
# This is the notebook method: sample an 8-action normalized DP chunk, optimize
# that executable action chunk with automaton gradients, unnormalize, execute,
# and replan at the next stage.
#
#######


class OursRunner:
    def __init__(self, config: PaperTestConfig):
        # OursRunner owns the robomimic DP policy, the TouchCube environment,
        # and the learned automaton model used for gradient guidance.
        self.config = config
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=config.dp_checkpoint,
            device=self.device,
            verbose=False,
        )
        self.env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=True, verbose=False)
        self.env.reset()
        self.automaton_model, self.eval_stats, _, self.eval_meta = load_automaton_model_for_eval(
            model_or_run_path=str(config.automaton_run_dir),
            predictor_kind="learned",
            device=self.device,
            load_val_trajectories=False,
        )
        self.state_mean = torch.tensor(self.eval_stats["states_mean"], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.state_std = torch.tensor(self.eval_stats["states_std"], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.action_mean = torch.tensor(self.eval_stats["actions_mean"], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.action_std = torch.tensor(self.eval_stats["actions_std"], device=self.device, dtype=torch.float32).unsqueeze(0)

    def _latest_obs_tensor(self, obs_dict, key: str) -> torch.Tensor:
        value = obs_dict[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.float()
        if value.ndim == 3:
            return value[:, -1, :]
        if value.ndim == 2:
            return value if value.shape[0] == 1 else value[-1:, :]
        if value.ndim == 1:
            return value.unsqueeze(0)
        raise ValueError(f"Unexpected obs shape for {key}: {tuple(value.shape)}")

    def _automaton_state_from_obs(self, obs_dict) -> torch.Tensor:
        return torch.cat([self._latest_obs_tensor(obs_dict, "agent_pos"), self._latest_obs_tensor(obs_dict, "states")], dim=-1)

    def _automaton_label_from_state(self, state: torch.Tensor) -> torch.Tensor:
        # The automaton model expects label order:
        # [at_green, at_blue, at_red, at_yellow].
        # The env state stores blocks as [blue, red, green, yellow], so this
        # function both detects contact and reorders to the automaton convention.
        agent = state[:, 0:2]
        cubes = state[:, 2:10].reshape(state.shape[0], 4, 2)
        distances = torch.linalg.norm(agent[:, None, :] - cubes, dim=-1)
        contact = (self.config.radius - distances >= 0.0).float()
        has_contact = contact.bool().any(dim=-1)
        active_distances = torch.where(contact.bool(), distances, torch.full_like(distances, float("inf")))
        nearest_active = active_distances.argmin(dim=-1)
        contact_onehot = torch.zeros_like(contact)
        contact_onehot.scatter_(1, nearest_active[:, None], has_contact.to(contact.dtype)[:, None])
        contact = contact_onehot
        return torch.stack([contact[:, 2], contact[:, 0], contact[:, 1], contact[:, 3]], dim=-1)

    def _torch_action_normalization_parts(self, dtype: torch.dtype):
        if self.policy.action_normalization_stats is None:
            return None
        action_keys = self.policy.policy.global_config.train.action_keys
        offsets, scales = [], []
        for key in action_keys:
            offset = torch.as_tensor(self.policy.action_normalization_stats[key]["offset"].reshape(-1), device=self.device, dtype=dtype)
            scale = torch.as_tensor(self.policy.action_normalization_stats[key]["scale"].reshape(-1), device=self.device, dtype=dtype)
            offsets.append(offset)
            scales.append(scale)
        return torch.cat(offsets, dim=0), torch.cat(scales, dim=0)

    def unnormalize_action_sequence(self, action_sequence: torch.Tensor) -> torch.Tensor:
        # Robomimic DP samples normalized actions. The env expects raw absolute
        # agent-position actions in [-1, 1], so every generated chunk must pass
        # through the checkpoint's action normalization stats.
        parts = self._torch_action_normalization_parts(dtype=action_sequence.dtype)
        if parts is None:
            return action_sequence
        offset, scale = parts
        offset = offset.to(device=action_sequence.device, dtype=action_sequence.dtype)
        scale = scale.to(device=action_sequence.device, dtype=action_sequence.dtype)
        return action_sequence * scale.view(1, 1, -1) + offset.view(1, 1, -1)

    def score_action_chunk(self, obs_tensor, raw_action_chunk: torch.Tensor) -> np.ndarray:
        state = self._automaton_state_from_obs(obs_tensor).to(device=self.device, dtype=torch.float32)
        label = self._automaton_label_from_state(state).to(device=self.device, dtype=torch.float32)
        action_flat = raw_action_chunk.reshape(raw_action_chunk.shape[0], -1).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            logits = self.automaton_model(
                (state - self.state_mean) / self.state_std,
                (action_flat - self.action_mean) / self.action_std,
                label,
            )
            return torch.sigmoid(logits).detach().cpu().numpy()

    def direct_chunk_grad(self, obs_tensor, chunk_n: torch.Tensor, target_idx: int) -> torch.Tensor:
        # Gradient target for our method: increase the automaton logit for the
        # current symbolic stage after executing this 8-action chunk.
        state = self._automaton_state_from_obs(obs_tensor).to(device=chunk_n.device, dtype=chunk_n.dtype)
        label = self._automaton_label_from_state(state).to(device=chunk_n.device, dtype=chunk_n.dtype)
        chunk_raw = self.unnormalize_action_sequence(chunk_n)
        action_flat = chunk_raw.reshape(chunk_raw.shape[0], -1)
        logits = self.automaton_model(
            (state - self.state_mean.to(chunk_n.dtype)) / self.state_std.to(chunk_n.dtype),
            (action_flat - self.action_mean.to(chunk_n.dtype)) / self.action_std.to(chunk_n.dtype),
            label,
        )
        objective = logits[:, int(target_idx)].mean()
        return torch.autograd.grad(objective, chunk_n, retain_graph=False, create_graph=False)[0]

    def optimize_chunk(self, obs_tensor, base_chunk_n: torch.Tensor, target_idx: int) -> torch.Tensor:
        # Lightweight post-sample guidance: repeatedly step the normalized
        # action chunk in automaton-gradient direction and keep it in the DP
        # action support via clipping.
        chunk_n = base_chunk_n.detach().clone()
        for _ in range(int(self.config.ours_guidance_steps)):
            x = chunk_n.detach().requires_grad_(True)
            grad = self.direct_chunk_grad(obs_tensor, x, target_idx)
            chunk_n = torch.clamp(x + float(self.config.ours_step_size) * grad.detach(), -1.0, 1.0).detach()
        return chunk_n

    def rollout(self, chain: Sequence[str], env_seed: int, rollout_seed: int, run_dir: Path) -> Dict:
        # Ours executes a symbolic chain stage-by-stage. Once the current block
        # is reached, the pending action queue is cleared and the next DP chunk
        # is optimized for the next symbolic target.
        label_chain = [LABEL_NAME_TO_IDX[name] for name in chain]
        old_debug = getattr(self.policy.policy, "debug_guidance_actions", False)
        self.policy.policy.debug_guidance_actions = False
        reseed(env_seed)
        setup_state = rollout_setup_state(self.config)
        reseed(rollout_seed)
        obs = self.env.reset_to(setup_state)
        self.policy.start_episode()

        records, reaches, action_queue = [], [], []
        low_level_obs = [obs_snapshot(obs)]
        actions, rewards = [], []
        chain_pos = 0
        steps = 0

        for t in range(int(self.config.env_horizon)):
            if not action_queue:
                # DP proposes a short action chunk; the automaton model scores
                # the whole chunk and supplies gradients for the current target.
                target_idx = label_chain[chain_pos]
                obs_tensor = self.policy._prepare_observation(obs)
                with torch.no_grad():
                    base_chunk_n = self.policy.policy._get_action_trajectory(obs_dict=obs_tensor).detach()
                guided_chunk_n = self.optimize_chunk(obs_tensor, base_chunk_n, target_idx)
                guided_chunk_raw = self.unnormalize_action_sequence(guided_chunk_n).detach()
                probs = self.score_action_chunk(obs_tensor, guided_chunk_raw)[0]
                delta = guided_chunk_n - base_chunk_n
                current_state = flat_state_from_low_level(obs_snapshot(obs))
                records.append(
                    {
                        "t": int(t),
                        "stage": int(chain_pos),
                        "target": LABEL_IDX_TO_NAME[target_idx],
                        "target_label_idx": int(target_idx),
                        "target_score": float(probs[target_idx]),
                        "pred_probs": probs.tolist(),
                        "max_da": float(delta.abs().max().detach().cpu()),
                        "clamp_frac": float(((guided_chunk_n.abs() >= 0.999).float().mean()).detach().cpu()),
                        "robustness_by_block": labels_from_state(current_state, self.config.radius),
                    }
                )
                action_queue.extend(guided_chunk_raw.detach().cpu().numpy()[0])

            action = np.asarray(action_queue.pop(0), dtype=np.float32)
            obs, reward, _done, _info = self.env.step(action)
            low_level_obs.append(obs_snapshot(obs))
            actions.append(action.copy())
            rewards.append(float(reward))
            steps = t + 1
            current_state = flat_state_from_low_level(obs_snapshot(obs))
            target_name = chain[chain_pos]
            reached, robustness = reached_label_from_state(current_state, target_name, self.config.radius)
            if reached:
                # Stage progress is based on real executed env state, not the
                # automaton prediction. This is the "actual" success criterion.
                reaches.append({"t": int(steps), "stage": int(chain_pos), "label": target_name, "robustness": float(robustness)})
                chain_pos += 1
                action_queue.clear()
                if chain_pos >= len(chain):
                    break

        complete = chain_pos >= len(chain)
        final_state = flat_state_from_low_level(obs_snapshot(obs))
        result = {
            "method": "ours",
            "chain": list(chain),
            "label_chain": label_chain,
            "complete": bool(complete),
            "stages": int(chain_pos),
            "steps": int(steps),
            "return": float(np.sum(rewards)),
            "env_seed": int(env_seed),
            "rollout_seed": int(rollout_seed),
            "reaches": reaches,
            "records": records,
            "final_robustness_by_block": labels_from_state(final_state, self.config.radius),
        }
        save_rollout_artifacts(run_dir, setup_state, low_level_obs, actions, rewards, result)
        plot_single_rollout(run_dir, result, title=f"Ours H={len(chain)} {' -> '.join(chain)}")
        self.policy.policy.debug_guidance_actions = old_debug
        return result


########
#
# LTLDoG-S BASELINE
# This is the paper-faithful baseline path we use in the final experiments:
# sample one full H-step state-action trajectory with LTLDoG-S posterior
# guidance, then execute the generated action sequence once in TouchCube.
# There is deliberately no receding-horizon replanning in this runner.
#
#######


class LTLDogRunner:
    def __init__(self, config: PaperTestConfig):
        self.config = config
        self.planner: LTLDogPlanner = load_ltldog_planner(
            config.ltldog_checkpoint,
            device="auto",
            diffusion_steps=config.ltldog_diffusion_steps,
        )

    def suffix_formula(self, chain: Sequence[str], chain_pos: int) -> FormulaSpec:
        # Kept as a small helper so single-target and ordered-prefix formulas
        # are represented consistently. In the final full-exec path chain_pos is
        # always zero because LTLDoG does not replan after partial progress.
        suffix = tuple(chain[chain_pos:])
        if len(suffix) == 1:
            return FormulaSpec(f"F_{suffix[0]}", "eventually", suffix, f"eventually visit {suffix[0]}")
        return FormulaSpec(f"seq_{'_'.join(suffix)}", "sequence", suffix, " then ".join(suffix))

    def rollout(self, chain: Sequence[str], env_seed: int, rollout_seed: int, run_dir: Path) -> Dict:
        # LTLDoG full-exec rollout:
        #   1. Reset the env to the paper layout.
        #   2. Sample one guided full-H trajectory for the complete formula.
        #   3. Execute predicted actions open-loop in the real env.
        #   4. Judge progress using the real env state sequence.
        reseed(env_seed)
        setup_state = rollout_setup_state(self.config)
        reseed(rollout_seed)
        env = make_env()
        env.reset()
        obs = env.unwrapped.reset_to(setup_state)
        current_obs = obs_from_env_dict(obs)

        low_level_obs = [obs_snapshot(obs)]
        actions, rewards, contacts, records, reaches = [], [], [], [], []
        chain_pos = 0
        steps = 0
        formula = self.suffix_formula(chain, 0)
        samples, values = self.planner.sample_plan(
            current_obs,
            formula,
            batch_size=self.config.ltldog_batch_size,
            guidance_scale=self.config.ltldog_guidance_scale,
            n_guide_steps=self.config.ltldog_n_guide_steps,
            t_stopgrad=self.config.ltldog_t_stopgrad,
            guidance_mode=self.config.ltldog_guidance_mode,
            scale_grad_by_std=self.config.ltldog_scale_grad_by_std,
            radius=self.config.radius,
            tau=self.config.ltldog_tau,
            freeze_static_blocks=self.config.ltldog_freeze_static_blocks,
        )
        predicted_first = samples[0].copy()
        # Store the guide metadata before execution. Diagnostics later compare
        # this imagined trajectory against the real executed one.
        current_state = flat_state_from_low_level(obs_snapshot(obs))
        records.append(
            {
                "t": int(steps),
                "stage": int(chain_pos),
                "target": chain[chain_pos],
                "remaining_chain": list(chain),
                "guide_value": float(values[0]),
                "execution_mode": "full",
                "tau": float(self.config.ltldog_tau),
                "freeze_static_blocks": bool(self.config.ltldog_freeze_static_blocks),
                "action_source": "actions",
                "robustness_by_block": labels_from_state(current_state, self.config.radius),
            }
        )

        action_sequence = predicted_first[: int(self.config.env_horizon), :ACTION_DIM]
        for action in action_sequence:
            # The generated action channel is already in env action coordinates;
            # clipping only protects against rare diffusion overshoot.
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            obs, reward, _done, info = env.step(action)
            current_obs = obs_from_env_dict(obs)
            low_level_obs.append(obs_snapshot(obs))
            actions.append(action.copy())
            rewards.append(float(reward))
            contacts.append(int(info.get("cube_contacted", -1)))
            steps += 1
            current_state = flat_state_from_low_level(obs_snapshot(obs))
            if chain_pos < len(chain):
                # Even though LTLDoG does not replan, we still track how many
                # ordered stages the actual executed trajectory completed.
                reached, robustness = reached_label_from_state(current_state, chain[chain_pos], self.config.radius)
                if reached:
                    reaches.append({"t": int(steps), "stage": int(chain_pos), "label": chain[chain_pos], "robustness": float(robustness)})
                    chain_pos += 1
                    if chain_pos >= len(chain):
                        break
            if steps >= int(self.config.env_horizon) or float(reward) < 0.0:
                break

        complete = chain_pos >= len(chain)
        final_state = flat_state_from_low_level(obs_snapshot(obs))
        result = {
            "method": self.config.ltldog_method_name,
            "ltldog_base_method": "ltldog",
            "ltldog_execution_mode": "full",
            "ltldog_action_source": "actions",
            "ltldog_tau": float(self.config.ltldog_tau),
            "ltldog_freeze_static_blocks": bool(self.config.ltldog_freeze_static_blocks),
            "chain": list(chain),
            "complete": bool(complete),
            "stages": int(chain_pos),
            "steps": int(steps),
            "return": float(np.sum(rewards)),
            "env_seed": int(env_seed),
            "rollout_seed": int(rollout_seed),
            "reaches": reaches,
            "records": records,
            "first_contact": int(next((c for c in contacts if c >= 0), -1)),
            "final_robustness_by_block": labels_from_state(final_state, self.config.radius),
        }
        save_rollout_artifacts(run_dir, setup_state, low_level_obs, actions, rewards, result, predicted=predicted_first)
        plot_single_rollout(run_dir, result, title=f"{self.config.ltldog_method_name} H={len(chain)} {' -> '.join(chain)}")
        return result


########
#
# SAVING AND PLOTTING
# Every individual rollout gets raw traces, a JSON summary, and a PNG. Each
# method/horizon also gets an overlay plot, and the paper_test root gets
# aggregate success/steps curves.
#
#######


def save_rollout_artifacts(
    run_dir: Path,
    setup_state: np.ndarray,
    low_level_obs: Sequence[Dict[str, np.ndarray]],
    actions: Sequence[np.ndarray],
    rewards: Sequence[float],
    result: Dict,
    predicted: np.ndarray | None = None,
) -> None:
    # Every rollout directory is self-contained:
    #   setup_state.npy      raw env reset vector
    #   low_level_obs.npz    actual env states observed during execution
    #   trace.npz            actions, rewards, and optionally LTLDoG prediction
    #   rollout_summary.json human-readable success/progress metadata
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "setup_state.npy", setup_state)
    np.savez_compressed(
        run_dir / "low_level_obs.npz",
        states=stack_or_object([obs.get("states") for obs in low_level_obs]),
        agent_pos=stack_or_object([obs.get("agent_pos") for obs in low_level_obs]),
    )
    np.savez_compressed(
        run_dir / "trace.npz",
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        predicted=np.asarray(predicted, dtype=np.float32) if predicted is not None else np.zeros((0, 12), dtype=np.float32),
    )
    with open(run_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)


def setup_blocks(setup_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = setup_state[2:10].reshape(4, 2) / 256.0 - 1.0
    agent = setup_state[:2] / 256.0 - 1.0
    angles = np.asarray(setup_state[10:14], dtype=float) if setup_state.shape[0] >= 14 else np.zeros(4)
    return flip_toy_y(blocks), flip_toy_y(agent), -angles


def load_trajectory(run_dir: Path) -> np.ndarray:
    # Plotting uses the actual executed agent path, not predicted LTLDoG states.
    with np.load(run_dir / "low_level_obs.npz", allow_pickle=True) as data:
        agent_pos = np.asarray(data["agent_pos"])
    if agent_pos.dtype == object:
        trajectory = np.stack([latest_vector(item) for item in agent_pos], axis=0)
    else:
        trajectory = np.stack([latest_vector(item) for item in agent_pos], axis=0)
    return flip_toy_y(trajectory)


def plot_path_on_axis(ax, trajectory: np.ndarray, complete: bool, alpha: float = 0.72, linewidth: float = 1.55) -> None:
    cmap = plt.get_cmap("turbo")
    if len(trajectory) > 1:
        segments = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
        line = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0.0, 1.0), linewidths=linewidth, alpha=alpha, zorder=2)
        line.set_array(np.linspace(0.0, 1.0, len(segments)))
        ax.add_collection(line)
    final_color = "#111111" if complete else "#b00020"
    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], s=14, c=final_color, edgecolors="none", zorder=6)


def style_env_axis(ax, setup_state: np.ndarray, title: str) -> None:
    blocks, agent, angles = setup_blocks(setup_state)
    draw_toy_blocks(ax, blocks, angles, BLOCK_COLORS, radius=0.11, alpha=0.95, zorder=4)
    ax.scatter([agent[0]], [agent[1]], s=180, marker="o", c="#2f2f2f", alpha=0.92, edgecolors="white", linewidths=0.4, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    frame = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#9a9a9a", linewidth=1.8, zorder=10)
    ax.add_patch(frame)


def plot_single_rollout(run_dir: Path, result: Dict, title: str) -> None:
    setup_state = np.load(run_dir / "setup_state.npy")
    trajectory = load_trajectory(run_dir)
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=170, constrained_layout=True)
    plot_path_on_axis(ax, trajectory, bool(result["complete"]), alpha=0.9, linewidth=2.0)
    reached = " -> ".join([item["label"] for item in result["reaches"]]) if result["reaches"] else "none"
    subtitle = f"{title}\ncomplete={int(result['complete'])}, stages={result['stages']}/{len(result['chain'])}, steps={result['steps']}, reached={reached}"
    style_env_axis(ax, setup_state, subtitle)
    fig.savefig(run_dir / "rollout.png", bbox_inches="tight")
    plt.close(fig)


def plot_method_overlay(method_dir: Path, method: str, chain: Sequence[str], results: Sequence[Dict]) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=180, constrained_layout=True)
    for result in results:
        run_dir = Path(result["run_dir"])
        setup_state = np.load(run_dir / "setup_state.npy")
        blocks, _, angles = setup_blocks(setup_state)
        draw_toy_blocks(ax, blocks, angles, BLOCK_COLORS, radius=0.11, alpha=0.11, zorder=3)
    for result in results:
        trajectory = load_trajectory(Path(result["run_dir"]))
        plot_path_on_axis(ax, trajectory, bool(result["complete"]), alpha=0.62, linewidth=1.45)
    first_setup = np.load(Path(results[0]["run_dir"]) / "setup_state.npy")
    success = float(np.mean([r["complete"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))
    style_env_axis(ax, first_setup, f"{method} H={len(chain)} {' -> '.join(chain)}\n{success:.2f} success, {mean_steps:.1f} steps")
    plot_path = method_dir / "overlay_rollouts.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def summarize_results(results: Sequence[Dict], chain: Sequence[str], method: str) -> Dict:
    return {
        "method": method,
        "chain": list(chain),
        "ltl_horizon": int(len(chain)),
        "n": int(len(results)),
        "success_rate": float(np.mean([r["complete"] for r in results])) if results else 0.0,
        "mean_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "median_steps": float(np.median([r["steps"] for r in results])) if results else 0.0,
        "mean_stages": float(np.mean([r["stages"] for r in results])) if results else 0.0,
        "records": [
            {
                "rollout_idx": int(idx),
                "run_dir": r["run_dir"],
                "complete": bool(r["complete"]),
                "stages": int(r["stages"]),
                "steps": int(r["steps"]),
                "reaches": r["reaches"],
            }
            for idx, r in enumerate(results)
        ],
    }


def upsert_summary(summaries: Sequence[Dict], new_summary: Dict) -> List[Dict]:
    new_key = (new_summary["method"], int(new_summary["ltl_horizon"]), tuple(new_summary["chain"]))
    kept = [
        summary
        for summary in summaries
        if (summary["method"], int(summary["ltl_horizon"]), tuple(summary["chain"])) != new_key
    ]
    kept.append(new_summary)
    return kept


def plot_aggregate(output_dir: Path, summaries: Sequence[Dict]) -> None:
    if not summaries:
        return
    methods = sorted({s["method"] for s in summaries})
    horizons = sorted({int(s["ltl_horizon"]) for s in summaries})
    by_key = {(s["method"], int(s["ltl_horizon"])): s for s in summaries}
    colors = {
        "ours": "#1f77b4",
        "ltldog": "#d62728",
        "ltldog_full_exec_tau0p05": "#ff7f0e",
        "ltldog_full_exec_tau0p001": "#9467bd",
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.4), dpi=180, constrained_layout=True)
    for method in methods:
        xs = [h for h in horizons if (method, h) in by_key]
        ys = [by_key[(method, h)]["success_rate"] for h in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.3, label=method, color=colors.get(method))
    ax.set_xlabel("LTL sequence length")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "aggregate_success_rate.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.4), dpi=180, constrained_layout=True)
    for method in methods:
        xs = [h for h in horizons if (method, h) in by_key]
        ys = [by_key[(method, h)]["mean_steps"] for h in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.3, label=method, color=colors.get(method))
    ax.set_xlabel("LTL sequence length")
    ax.set_ylabel("mean steps")
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "aggregate_mean_steps.png", bbox_inches="tight")
    plt.close(fig)


########
#
# MAIN EVALUATION LOOP
#
#######


def run_method_for_horizon(runner, method: str, chain: Sequence[str], horizon_dir: Path, config: PaperTestConfig) -> Dict:
    # Idempotent by default. This is important because LTLDoG full-exec sampling
    # is slow: adding H4/H5 later should reuse already-completed H1-H3 rollouts.
    method_dir = horizon_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    method_config = asdict(config)
    method_config["output_method"] = method
    with open(method_dir / "method_config.json", "w", encoding="utf-8") as f:
        json.dump(method_config, f, indent=2)
    results = []
    iterator = tqdm(range(int(config.n_rollouts)), desc=f"{method} H={len(chain)}", dynamic_ncols=True)
    for idx in iterator:
        seed = int(config.seed_start) + idx
        run_dir = method_dir / f"rollout_{idx:03d}"
        summary_path = run_dir / "rollout_summary.json"
        if summary_path.exists() and not config.overwrite_existing_rollouts:
            result = json.loads(summary_path.read_text())
        else:
            result = runner.rollout(chain, env_seed=seed, rollout_seed=seed, run_dir=run_dir)
        result["run_dir"] = str(run_dir)
        results.append(result)
        iterator.set_postfix(success=float(np.mean([r["complete"] for r in results])))
    overlay_path = plot_method_overlay(method_dir, method, chain, results)
    summary = summarize_results(results, chain, method)
    summary["overlay_path"] = str(overlay_path)
    with open(method_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2)
    return summary


def run_paper_test(config: PaperTestConfig) -> List[Dict]:
    # Main orchestration: instantiate each method once, then sweep prefix
    # horizons. Summaries are upserted so reruns can extend or refresh one
    # horizon without duplicating aggregate rows.
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "paper_test_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    selected_methods = [item.strip() for item in config.methods.split(",") if item.strip()]
    runners = {}
    if "ours" in selected_methods:
        runners["ours"] = OursRunner(config)
    if "ltldog" in selected_methods:
        runners["ltldog"] = LTLDogRunner(config)

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

        for method in selected_methods:
            output_method = config.ltldog_method_name if method == "ltldog" else method
            summary = run_method_for_horizon(runners[method], output_method, chain, horizon_dir, config)
            all_summaries = upsert_summary(all_summaries, summary)

        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(jsonable(all_summaries), f, indent=2)
        plot_aggregate(output_dir, all_summaries)

        stop_method = config.ltldog_method_name if "ltldog" in selected_methods else "ltldog"
        ltldog_summary = next((s for s in all_summaries if s["method"] == stop_method and s["ltl_horizon"] == horizon), None)
        if (
            ltldog_summary is not None
            and horizon >= int(config.stop_after_horizon)
            and float(ltldog_summary["success_rate"]) <= float(config.stop_ltldog_success_at)
        ):
            break

    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(all_summaries), f, indent=2)
    plot_aggregate(output_dir, all_summaries)
    return all_summaries


def parse_args() -> PaperTestConfig:
    defaults = PaperTestConfig()
    parser = argparse.ArgumentParser(description="Paper LTL horizon scaling test: ours vs LTLDoG")
    for field, value in asdict(defaults).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", dest=field, action="store_true")
            parser.add_argument(f"--no-{field}", dest=field, action="store_false")
            parser.set_defaults(**{field: value})
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    return PaperTestConfig(**vars(parser.parse_args()))


def main() -> None:
    summaries = run_paper_test(parse_args())
    print(json.dumps(jsonable(summaries), indent=2))


if __name__ == "__main__":
    main()
