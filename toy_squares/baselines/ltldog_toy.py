from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOMIMIC_ROOT = REPO_ROOT / "robomimic"
if str(ROBOMIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOMIMIC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


########
#
# GLOBAL TRAJECTORY LAYOUT
# The Diffuser models full state-action trajectories. Each transition is
# [action_x, action_y, agent_xy, blue_xy, red_xy, green_xy, yellow_xy].
# The LTL/STL labels are derived from distances between the agent and blocks.
#
#######

ACTION_DIM = 2
OBS_DIM = 10
TRANSITION_DIM = ACTION_DIM + OBS_DIM
LABEL_NAMES = ("blue", "red", "green", "yellow")
LABEL_TO_BLOCK_SLICE = {
    "blue": slice(2, 4),
    "red": slice(4, 6),
    "green": slice(6, 8),
    "yellow": slice(8, 10),
}
LABEL_COLORS = {
    "blue": "#0074D9",
    "red": "#FF4136",
    "green": "#2ECC40",
    "yellow": "#FFDC00",
}


########
#
# RUN CONFIGURATION
# TrainConfig defines the offline full-trajectory diffusion model. EvalConfig
# defines receding-horizon LTLDoG rollouts: sample H-step plans, execute the
# first execute_steps actions in the real env, observe, and replan.
#
#######

@dataclass
class TrainConfig:
    data_path: str = "/home/shared/data/toy_squares/train/data.hdf5"
    output_dir: str = "outputs/toy_squares_rollouts/baseline_ltldog/training/h128_full_diffuser"
    horizon: int = 128
    batch_size: int = 64
    train_steps: int = 20000
    val_batches: int = 25
    lr: float = 2e-4
    ema_decay: float = 0.995
    channels: int = 128
    depth: int = 6
    diffusion_steps: int = 100
    save_every: int = 1000
    val_every: int = 500
    seed: int = 7
    device: str = "auto"
    num_workers: int = 2
    max_demos: int = 0
    min_start_gap: int = 1


@dataclass
class EvalConfig:
    checkpoint: str
    output_dir: str = "outputs/toy_squares_rollouts/baseline_ltldog/rollouts/formula_suite_dataset_resets_ps_scale0p01_g5_exec8"
    data_path: str = "/home/shared/data/toy_squares/train/data.hdf5"
    formulas: str = "default"
    n_rollouts: int = 12
    env_horizon: int = 150
    execute_steps: int = 8
    batch_size: int = 4
    guidance_scale: float = 0.01
    n_guide_steps: int = 5
    t_stopgrad: int = 2
    guidance_mode: str = "ps"
    scale_grad_by_std: bool = False
    guidance_threshold: float = 0.0
    diffusion_steps: int = 100
    seed: int = 11
    device: str = "auto"
    render_video: bool = False
    reset_mode: str = "dataset"
    formula_limit: int = 0
    stop_on_contact: bool = False
    stop_on_satisfaction: bool = True


########
#
# SMALL RUNTIME AND ENV HELPERS
# These functions keep device/seed setup deterministic and convert between the
# robomimic TouchCube observation dicts and the flat observation vectors used by
# the trajectory model.
#
#######

def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_from_env_dict(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs["agent_pos"], dtype=np.float32), np.asarray(obs["states"], dtype=np.float32)],
        axis=-1,
    )


def raw_reset_state_from_obs(obs: np.ndarray) -> np.ndarray:
    positions = ((np.asarray(obs, dtype=np.float32) + 1.0) * 0.5 * 512.0).reshape(5, 2)
    rotations = np.zeros(4, dtype=np.float32)
    return np.concatenate([positions.reshape(-1), rotations], axis=0)


########
#
# DATASET: FULL TRAJECTORIES FOR DIFFUSER
# This dataset reads demonstration episodes from HDF5 and returns fixed-length
# H-step windows. Short suffix windows are padded so the model always trains on
# full trajectories with the first observation clamped as the conditioning state.
#
#######

class ToySquaresTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        horizon: int,
        max_demos: int = 0,
        min_start_gap: int = 1,
    ):
        self.data_path = str(data_path)
        self.horizon = int(horizon)
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.indices: List[Tuple[int, int]] = []
        self.demo_keys: List[str] = []

        with h5py.File(self.data_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
            if max_demos and max_demos > 0:
                keys = keys[: int(max_demos)]
            for demo_key in keys:
                group = f["data"][demo_key]
                actions = np.asarray(group["actions"], dtype=np.float32)
                obs = np.concatenate(
                    [
                        np.asarray(group["obs/agent_pos"], dtype=np.float32),
                        np.asarray(group["obs/states"], dtype=np.float32),
                    ],
                    axis=-1,
                )
                if len(actions) < 2:
                    continue
                self.demo_keys.append(demo_key)
                episode_idx = len(self.episodes)
                self.episodes.append({"actions": actions, "obs": obs})
                for start in range(0, len(actions), max(1, int(min_start_gap))):
                    self.indices.append((episode_idx, start))

        if not self.indices:
            raise ValueError(f"No usable trajectories found in {self.data_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, start = self.indices[idx]
        episode = self.episodes[episode_idx]
        actions = episode["actions"]
        obs = episode["obs"]
        end = start + self.horizon

        act_window = actions[start:min(end, len(actions))]
        obs_window = obs[start:min(end, len(obs))]
        valid_len = len(act_window)
        if valid_len < self.horizon:
            pad = self.horizon - valid_len
            act_pad = np.repeat(act_window[-1:][..., :], pad, axis=0)
            obs_pad = np.repeat(obs_window[-1:][..., :], pad, axis=0)
            act_window = np.concatenate([act_window, act_pad], axis=0)
            obs_window = np.concatenate([obs_window, obs_pad], axis=0)

        trajectory = np.concatenate([act_window, obs_window], axis=-1)
        trajectory = np.clip(trajectory, -1.0, 1.0).astype(np.float32)
        condition = trajectory[0, ACTION_DIM:].astype(np.float32)
        return {
            "trajectory": torch.from_numpy(trajectory),
            "condition": torch.from_numpy(condition),
            "valid_len": torch.tensor(valid_len, dtype=torch.long),
        }

    def sample_initial_obs(self, n: int, seed: int) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            ep = self.episodes[int(rng.integers(0, len(self.episodes)))]
            out.append(ep["obs"][0].astype(np.float32))
        return out


########
#
# TEMPORAL DENOISING MODEL
# This is the neural network inside the Diffuser. Given a noisy H-step
# state-action trajectory, diffusion timestep, and current observation, it
# predicts the Gaussian noise that was added to the trajectory.
#
#######

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
        )
        self.cond_proj = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        return x + h


class TemporalDenoiser(nn.Module):
    def __init__(self, transition_dim: int, obs_dim: int, channels: int = 128, depth: int = 6):
        super().__init__()
        emb_dim = channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.obs_mlp = nn.Sequential(nn.Linear(obs_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.in_conv = nn.Conv1d(transition_dim, channels, 5, padding=2)
        self.blocks = nn.ModuleList([ResidualTemporalBlock(channels, emb_dim) for _ in range(depth)])
        self.out = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv1d(channels, transition_dim, 5, padding=2))

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x.transpose(1, 2))
        cond = self.time_mlp(t) + self.obs_mlp(condition)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).transpose(1, 2)


########
#
# DIFFUSION PROCESS AND LTLDoG-S GUIDANCE
# GaussianTrajectoryDiffusion implements training noise prediction, reverse
# denoising, and the LTLDoG-S-style gradient insertion step. When a guide is
# passed to sample(), every reverse step can backprop differentiable robustness
# through the predicted clean trajectory and nudge the noisy sample.
#
#######

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(values: torch.Tensor, t: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianTrajectoryDiffusion(nn.Module):
    def __init__(self, model: TemporalDenoiser, horizon: int, timesteps: int):
        super().__init__()
        self.model = model
        self.horizon = int(horizon)
        self.transition_dim = TRANSITION_DIM
        self.timesteps = int(timesteps)

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def apply_conditioning(self, x: torch.Tensor, condition: torch.Tensor, freeze_static_blocks: bool = False) -> torch.Tensor:
        x = x.clone()
        x[:, 0, ACTION_DIM:] = condition
        if freeze_static_blocks:
            # Toy Squares block positions are fixed within an episode. Freezing
            # these coordinates prevents guidance from satisfying formulas by
            # hallucinating block motion that the real environment cannot execute.
            x[:, :, ACTION_DIM + 2 : ACTION_DIM + OBS_DIM] = condition[:, None, 2:OBS_DIM]
        return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        logvar = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, logvar

    def loss(self, x_start: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.apply_conditioning(self.q_sample(x_start, t, noise), condition)
        noise_pred = self.model(x_noisy, t, condition)
        noise_pred = self.apply_conditioning(noise_pred, torch.zeros_like(condition))
        noise = self.apply_conditioning(noise, torch.zeros_like(condition))
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, batch_size: int, guide=None, sample_kwargs=None) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_kwargs = {} if sample_kwargs is None else dict(sample_kwargs)
        guidance_mode = sample_kwargs.pop("guidance_mode", "ps")
        freeze_static_blocks = bool(sample_kwargs.pop("freeze_static_blocks", False))
        device = condition.device
        x = torch.randn(batch_size, self.horizon, self.transition_dim, device=device)
        x = self.apply_conditioning(x, condition, freeze_static_blocks=freeze_static_blocks)
        values = torch.zeros(batch_size, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if guide is not None:
                # LTLDoG-S guidance path. The default "ps" mode matches the
                # reference repo's posterior-sampling update; "pre_posterior"
                # keeps the older simpler gradient-insertion ablation.
                if guidance_mode == "ps":
                    x, values = self.ps_guided_step(
                        x,
                        condition,
                        t,
                        guide,
                        freeze_static_blocks=freeze_static_blocks,
                        **sample_kwargs,
                    )
                elif guidance_mode == "pre_posterior":
                    x, values = self.guided_step(
                        x,
                        condition,
                        t,
                        guide,
                        freeze_static_blocks=freeze_static_blocks,
                        **sample_kwargs,
                    )
                else:
                    raise ValueError(f"Unknown guidance_mode: {guidance_mode}")
            else:
                eps = self.model(x, t, condition)
                x0 = self.predict_start_from_noise(x, t, eps).clamp(-1.0, 1.0)
                x0 = self.apply_conditioning(x0, condition, freeze_static_blocks=freeze_static_blocks)
                mean, logvar = self.q_posterior(x0, x, t)
                noise = torch.randn_like(x)
                noise[t == 0] = 0
                x = mean + torch.exp(0.5 * logvar) * noise
                x = self.apply_conditioning(x, condition, freeze_static_blocks=freeze_static_blocks)

        # This ordering is only a best-of-batch convenience after guided
        # denoising. The baseline here is not pure sample-and-rank; with
        # batch_size=1 this line has no practical effect.
        order = torch.argsort(values, descending=True)
        return x[order], values[order]

    def guided_step(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        guide,
        scale: float = 0.1,
        n_guide_steps: int = 2,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = True,
        freeze_static_blocks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_var = extract(self.posterior_variance, t, x.shape)
        values = torch.zeros(x.shape[0], device=x.device)
        if int(t[0].item()) >= int(t_stopgrad):
            for _ in range(int(n_guide_steps)):
                with torch.enable_grad():
                    # Core LTLDoG-S insertion: estimate x0, score it with the
                    # differentiable temporal-logic robustness, then move x_t
                    # in the direction that increases that robustness.
                    x_grad = x.detach().requires_grad_(True)
                    eps = self.model(x_grad, t, condition)
                    x0 = self.predict_start_from_noise(x_grad, t, eps).clamp(-1.0, 1.0)
                    x0 = self.apply_conditioning(x0, condition, freeze_static_blocks=freeze_static_blocks)
                    values = guide(x0)
                    grad = torch.autograd.grad(values.sum(), x_grad)[0]
                if scale_grad_by_std:
                    grad = model_var * grad
                x = self.apply_conditioning(x + float(scale) * grad.detach(), condition, freeze_static_blocks=freeze_static_blocks)

        with torch.no_grad():
            eps = self.model(x, t, condition)
            x0 = self.predict_start_from_noise(x, t, eps).clamp(-1.0, 1.0)
            x0 = self.apply_conditioning(x0, condition, freeze_static_blocks=freeze_static_blocks)
            values = guide(x0)
            mean, logvar = self.q_posterior(x0, x, t)
            noise = torch.randn_like(x)
            noise[t == 0] = 0
            x = mean + torch.exp(0.5 * logvar) * noise
            x = self.apply_conditioning(x, condition, freeze_static_blocks=freeze_static_blocks)
        return x, values

    def ps_guided_step(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        guide,
        scale: float = 0.1,
        n_guide_steps: int = 2,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = False,
        threshold: float = 0.0,
        freeze_static_blocks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_var = extract(self.posterior_variance, t, x.shape)
        with torch.no_grad():
            eps = self.model(x, t, condition)
            x0 = self.predict_start_from_noise(x, t, eps).clamp(-1.0, 1.0)
            x0 = self.apply_conditioning(x0, condition, freeze_static_blocks=freeze_static_blocks)
            mean, logvar = self.q_posterior(x0, x, t)
            noise = torch.randn_like(x)
            noise[t == 0] = 0
            x_next = mean + torch.exp(0.5 * logvar) * noise
            x_next = self.apply_conditioning(x_next, condition, freeze_static_blocks=freeze_static_blocks)
            values = self.guide_values_from_noisy(
                x_next,
                condition,
                torch.clamp(t - 1, min=0),
                guide,
                freeze_static_blocks=freeze_static_blocks,
            )
            if torch.all(values > float(threshold)):
                return x_next, values

        if int(t[0].item()) >= int(t_stopgrad):
            with torch.enable_grad():
                # Reference LTLDoG-S posterior sampling: estimate x0 from the
                # current noisy sample, backprop robustness through that x0,
                # then use the gradient to correct the posterior x_{t-1}.
                x_prev = x.detach().requires_grad_(True)
                eps = self.model(x_prev, t, condition)
                x0_hat = self.predict_start_from_noise(x_prev, t, eps).clamp(-1.0, 1.0)
                x0_hat = self.apply_conditioning(x0_hat, condition, freeze_static_blocks=freeze_static_blocks)
                values = guide(x0_hat)
                grad = torch.autograd.grad(values.sum(), x_prev)[0]

            if scale_grad_by_std:
                grad = model_var * grad
            active = (values <= float(threshold)).to(grad.dtype).reshape(-1, 1, 1)
            grad = grad * active

            for _ in range(int(n_guide_steps)):
                if torch.all(values > float(threshold)):
                    break
                x_next = self.apply_conditioning(
                    x_next + float(scale) * grad.detach(),
                    condition,
                    freeze_static_blocks=freeze_static_blocks,
                )
                with torch.no_grad():
                    values = self.guide_values_from_noisy(
                        x_next,
                        condition,
                        torch.clamp(t - 1, min=0),
                        guide,
                        freeze_static_blocks=freeze_static_blocks,
                    )
                    active = (values <= float(threshold)).to(grad.dtype).reshape(-1, 1, 1)
                    grad = grad * active

        return x_next.detach(), values.detach()

    def guide_values_from_noisy(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        guide,
        freeze_static_blocks: bool = False,
    ) -> torch.Tensor:
        eps = self.model(x, t, condition)
        x0 = self.predict_start_from_noise(x, t, eps).clamp(-1.0, 1.0)
        x0 = self.apply_conditioning(x0, condition, freeze_static_blocks=freeze_static_blocks)
        return guide(x0)


########
#
# TRAINING STABILIZER
# The EMA copy is what we evaluate and save as best.pt/latest.pt. It smooths
# noisy optimization updates and usually gives cleaner sampled trajectories than
# the raw instantaneous model weights.
#
#######

class EMAModel:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


########
#
# DIFFERENTIABLE TEMPORAL-LOGIC ROBUSTNESS
# These smooth max/min operators make eventually, ordered visit, and
# preceded-by formulas differentiable. The resulting scalar robustness is the
# guide used by LTLDoG-S during reverse diffusion.
#
#######

def smooth_max(x: torch.Tensor, dim=None, tau: float = 0.05) -> torch.Tensor:
    if dim is None:
        return tau * torch.logsumexp(x / tau, dim=tuple(range(x.ndim)))
    return tau * torch.logsumexp(x / tau, dim=dim)


def smooth_min(x: torch.Tensor, dim=None, tau: float = 0.05) -> torch.Tensor:
    return -smooth_max(-x, dim=dim, tau=tau)


def smooth_or(a: torch.Tensor, b: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    return smooth_max(torch.stack([a, b], dim=0), dim=0, tau=tau)


def smooth_and(a: torch.Tensor, b: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    return smooth_min(torch.stack([a, b], dim=0), dim=0, tau=tau)


def cumulative_smooth_max(x: torch.Tensor, tau: float) -> torch.Tensor:
    values = []
    acc = x[:, 0]
    values.append(acc)
    for t in range(1, x.shape[1]):
        acc = smooth_or(acc, x[:, t], tau=tau)
        values.append(acc)
    return torch.stack(values, dim=1)


def cumulative_smooth_min(x: torch.Tensor, tau: float) -> torch.Tensor:
    values = []
    acc = x[:, 0]
    values.append(acc)
    for t in range(1, x.shape[1]):
        acc = smooth_and(acc, x[:, t], tau=tau)
        values.append(acc)
    return torch.stack(values, dim=1)


@dataclass
class FormulaSpec:
    name: str
    kind: str
    labels: Tuple[str, ...]
    description: str


DEFAULT_FORMULAS = [
    FormulaSpec("F_green", "eventually", ("green",), "eventually visit green"),
    FormulaSpec("F_blue", "eventually", ("blue",), "eventually visit blue"),
    FormulaSpec("F_red", "eventually", ("red",), "eventually visit red"),
    FormulaSpec("F_yellow", "eventually", ("yellow",), "eventually visit yellow"),
    FormulaSpec("seq_green_blue_red", "sequence", ("green", "blue", "red"), "visit green, then blue, then red"),
    FormulaSpec(
        "seq_yellow_green_blue",
        "sequence",
        ("yellow", "green", "blue"),
        "visit yellow, then green, then blue",
    ),
    FormulaSpec(
        "green_after_yellow_or_red",
        "preceded_by_any",
        ("green", "yellow", "red"),
        "visit green, but only after yellow or red",
    ),
]


class ToyLTLRobustness:
    def __init__(self, formula: FormulaSpec, radius: float = 0.2, tau: float = 0.05):
        self.formula = formula
        self.radius = float(radius)
        self.tau = float(tau)

    def label_robustness(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        agent = observations[..., 0:2]
        out = {}
        for label, block_slice in LABEL_TO_BLOCK_SLICE.items():
            block = observations[..., block_slice]
            out[label] = self.radius - torch.linalg.norm(agent - block, dim=-1)
        return out

    def __call__(self, trajectories: torch.Tensor) -> torch.Tensor:
        observations = trajectories[..., ACTION_DIM:]
        r = self.label_robustness(observations)
        kind = self.formula.kind
        labels = self.formula.labels
        if kind == "eventually":
            return smooth_max(r[labels[0]], dim=1, tau=self.tau)
        if kind == "sequence":
            return self.sequence([r[label] for label in labels])
        if kind == "preceded_by_any":
            target, pre_a, pre_b = labels
            pre = smooth_or(r[pre_a], r[pre_b], tau=self.tau)
            no_target_prefix = cumulative_smooth_min(-r[target], tau=self.tau)
            event = smooth_and(pre, no_target_prefix, tau=self.tau)
            shifted_prefix = F.pad(cumulative_smooth_max(event, tau=self.tau)[:, :-1], (1, 0), value=-10.0)
            target_after = smooth_and(shifted_prefix, r[target], tau=self.tau)
            return smooth_max(target_after, dim=1, tau=self.tau)
        raise ValueError(f"Unknown formula kind: {kind}")

    def sequence(self, rs: Sequence[torch.Tensor]) -> torch.Tensor:
        score = rs[0]
        for r_next in rs[1:]:
            prefix = cumulative_smooth_max(score, tau=self.tau)
            shifted_prefix = F.pad(prefix[:, :-1], (1, 0), value=-10.0)
            score = smooth_and(shifted_prefix, r_next, tau=self.tau)
        return smooth_max(score, dim=1, tau=self.tau)

    def hard_satisfaction(self, observations_np: np.ndarray) -> Tuple[float, bool]:
        obs = np.asarray(observations_np, dtype=np.float32)
        agent = obs[:, 0:2]
        r = {}
        for label, block_slice in LABEL_TO_BLOCK_SLICE.items():
            block = obs[:, block_slice]
            r[label] = self.radius - np.linalg.norm(agent - block, axis=-1)

        kind = self.formula.kind
        labels = self.formula.labels
        if kind == "eventually":
            value = float(np.max(r[labels[0]]))
        elif kind == "sequence":
            score = r[labels[0]]
            for label in labels[1:]:
                shifted_prefix = np.concatenate([np.asarray([-np.inf], dtype=np.float32), np.maximum.accumulate(score)[:-1]])
                score = np.minimum(shifted_prefix, r[label])
            value = float(np.max(score))
        elif kind == "preceded_by_any":
            target, pre_a, pre_b = labels
            pre = np.maximum(r[pre_a], r[pre_b])
            no_target_prefix = np.minimum.accumulate(-r[target])
            event = np.minimum(pre, no_target_prefix)
            shifted_prefix = np.concatenate([np.asarray([-np.inf], dtype=np.float32), np.maximum.accumulate(event)[:-1]])
            target_after = np.minimum(shifted_prefix, r[target])
            value = float(np.max(target_after))
        else:
            raise ValueError(f"Unknown formula kind: {kind}")
        return value, value > 0.0


def parse_formula_selection(selection: str, limit: int = 0) -> List[FormulaSpec]:
    if selection == "default":
        formulas = list(DEFAULT_FORMULAS)
    else:
        names = {name.strip() for name in selection.split(",") if name.strip()}
        formulas = [formula for formula in DEFAULT_FORMULAS if formula.name in names]
        if len(formulas) != len(names):
            found = {formula.name for formula in formulas}
            raise ValueError(f"Unknown formula names: {sorted(names - found)}")
    if limit and limit > 0:
        formulas = formulas[: int(limit)]
    return formulas


########
#
# CHECKPOINT AND MODEL CONSTRUCTION
# These helpers keep the train/eval code using the same architecture and load
# EMA weights for rollout evaluation.
#
#######

def build_model_from_config(config: Dict) -> GaussianTrajectoryDiffusion:
    model = TemporalDenoiser(
        transition_dim=TRANSITION_DIM,
        obs_dim=OBS_DIM,
        channels=int(config["channels"]),
        depth=int(config["depth"]),
    )
    return GaussianTrajectoryDiffusion(model, horizon=int(config["horizon"]), timesteps=int(config["diffusion_steps"]))


def save_checkpoint(path: Path, diffusion: GaussianTrajectoryDiffusion, ema: EMAModel, config: TrainConfig, step: int, metrics: Dict) -> None:
    payload = {
        "model": diffusion.model.state_dict(),
        "ema": ema.shadow,
        "config": asdict(config),
        "step": int(step),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[GaussianTrajectoryDiffusion, Dict]:
    payload = torch.load(path, map_location=device)
    config = dict(payload["config"])
    diffusion = build_model_from_config(config).to(device)
    diffusion.model.load_state_dict(payload.get("ema", payload["model"]), strict=True)
    diffusion.eval()
    return diffusion, payload


########
#
# TRAINING ENTRY POINT
# This optimizes the Diffuser on demonstration trajectory windows with standard
# denoising score matching. No LTL formula is used during training; the logic
# enters only at sampling time through the guide.
#
#######

def train(args: argparse.Namespace) -> None:
    config = TrainConfig(**{k: v for k, v in vars(args).items() if k != "func"})
    seed_everything(config.seed)
    device = get_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    dataset = ToySquaresTrajectoryDataset(
        config.data_path,
        horizon=config.horizon,
        max_demos=config.max_demos,
        min_start_gap=config.min_start_gap,
    )
    val_len = max(1, int(0.05 * len(dataset)))
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)

    diffusion = build_model_from_config(asdict(config)).to(device)
    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=config.lr, weight_decay=1e-5)
    ema = EMAModel(diffusion.model, config.ema_decay)
    metrics = {"train_loss": [], "val_loss": []}
    loader_iter = iter(train_loader)
    best_val = float("inf")

    progress = tqdm(range(1, config.train_steps + 1), desc="train", dynamic_ncols=True)
    for step in progress:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        trajectory = batch["trajectory"].to(device)
        condition = batch["condition"].to(device)
        loss = diffusion.loss(trajectory, condition)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
        optimizer.step()
        ema.update(diffusion.model)
        metrics["train_loss"].append([step, float(loss.detach().cpu())])
        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", best_val=f"{best_val:.4f}")

        if step % config.val_every == 0 or step == config.train_steps:
            backup = {k: v.detach().clone() for k, v in diffusion.model.state_dict().items()}
            ema.copy_to(diffusion.model)
            diffusion.eval()
            val_losses = []
            with torch.no_grad():
                for idx, val_batch in enumerate(val_loader):
                    if idx >= config.val_batches:
                        break
                    val_loss = diffusion.loss(val_batch["trajectory"].to(device), val_batch["condition"].to(device))
                    val_losses.append(float(val_loss.detach().cpu()))
            diffusion.model.load_state_dict(backup, strict=True)
            diffusion.train()
            val_loss_mean = float(np.mean(val_losses)) if val_losses else float("nan")
            metrics["val_loss"].append([step, val_loss_mean])
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            if val_loss_mean < best_val:
                best_val = val_loss_mean
                save_checkpoint(output_dir / "best.pt", diffusion, ema, config, step, {"best_val": best_val})
            save_checkpoint(output_dir / "latest.pt", diffusion, ema, config, step, {"best_val": best_val})

        if step % config.save_every == 0:
            save_checkpoint(output_dir / f"step_{step}.pt", diffusion, ema, config, step, {"best_val": best_val})

    plot_training_curves(output_dir / "metrics.json", output_dir / "training_curve.png")


########
#
# TRAINING ARTIFACTS
# The curve plot is a quick sanity check for underfitting/overfitting: training
# loss should fall, and validation loss should track it without blowing up.
#
#######

def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text())
    plt.figure(figsize=(7, 4), dpi=160)
    if metrics.get("train_loss"):
        arr = np.asarray(metrics["train_loss"], dtype=np.float32)
        stride = max(1, len(arr) // 1000)
        plt.plot(arr[::stride, 0], arr[::stride, 1], label="train", alpha=0.7)
    if metrics.get("val_loss"):
        arr = np.asarray(metrics["val_loss"], dtype=np.float32)
        plt.plot(arr[:, 0], arr[:, 1], label="val", marker="o", markersize=2)
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


########
#
# ENVIRONMENT ROLLOUT AND RECEDING-HORIZON EVALUATION
# This is where generated plans become actual env behavior. Each iteration
# samples an LTL-guided full trajectory, executes only the first chunk, then
# replans from the observed state. actual_satisfied is computed from this real
# env trace; predicted_satisfied is only the first generated plan's satisfaction.
#
#######

def make_env():
    import gym
    import robomimic.envs  # noqa: F401

    return gym.make("TouchCube")


def rollout_formula(
    diffusion: GaussianTrajectoryDiffusion,
    formula: FormulaSpec,
    initial_obs: np.ndarray,
    config: EvalConfig,
    device: torch.device,
    rollout_idx: int,
    formula_dir: Path,
) -> Dict:
    env = make_env()
    env.reset()
    obs = env.unwrapped.reset_to(raw_reset_state_from_obs(initial_obs))
    current_obs = obs_from_env_dict(obs)
    robustness = ToyLTLRobustness(formula)
    frames = []
    actual_obs = [current_obs.copy()]
    actions = []
    rewards = []
    contacts = []
    predicted_first = None
    predicted_value = None

    if config.render_video:
        frames.append(env.render(mode="rgb_array", height=256, width=256))

    for step in range(config.env_horizon):
        cond = torch.as_tensor(current_obs[None], dtype=torch.float32, device=device).repeat(config.batch_size, 1)
        samples, values = diffusion.sample(
            cond,
            batch_size=config.batch_size,
            guide=robustness,
            sample_kwargs={
                "scale": config.guidance_scale,
                "n_guide_steps": config.n_guide_steps,
                "t_stopgrad": config.t_stopgrad,
                "guidance_mode": config.guidance_mode,
                "scale_grad_by_std": config.scale_grad_by_std,
                "threshold": config.guidance_threshold,
            },
        )
        sample = samples[0].detach().cpu().numpy()
        if predicted_first is None:
            predicted_first = sample.copy()
            predicted_value = float(values[0].detach().cpu())
        chunk = sample[: config.execute_steps, :ACTION_DIM]
        for action in chunk:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            next_obs, reward, done, info = env.step(action)
            current_obs = obs_from_env_dict(next_obs)
            actual_obs.append(current_obs.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            contacts.append(int(info.get("cube_contacted", -1)))
            if config.render_video:
                frames.append(env.render(mode="rgb_array", height=256, width=256))
            if config.stop_on_satisfaction:
                _, is_sat_now = robustness.hard_satisfaction(np.asarray(actual_obs, dtype=np.float32))
                if is_sat_now:
                    break
            if reward < 0:
                break
            if config.stop_on_contact and done:
                break
            if len(actions) >= config.env_horizon:
                break
        if len(actions) >= config.env_horizon:
            break
        if rewards and rewards[-1] < 0:
            break
        if config.stop_on_satisfaction:
            _, is_sat_now = robustness.hard_satisfaction(np.asarray(actual_obs, dtype=np.float32))
            if is_sat_now:
                break
        if config.stop_on_contact and contacts and contacts[-1] >= 0:
            break

    actual_obs_arr = np.asarray(actual_obs, dtype=np.float32)
    actual_value, actual_sat = robustness.hard_satisfaction(actual_obs_arr)
    pred_obs = predicted_first[:, ACTION_DIM:] if predicted_first is not None else np.zeros((0, OBS_DIM), dtype=np.float32)
    pred_value_hard, pred_sat = robustness.hard_satisfaction(pred_obs) if len(pred_obs) else (float("nan"), False)
    result = {
        "formula": asdict(formula),
        "rollout_idx": int(rollout_idx),
        "steps": int(len(actions)),
        "total_reward": float(np.sum(rewards)),
        "contacts": contacts,
        "first_contact": int(next((c for c in contacts if c >= 0), -1)),
        "predicted_guidance_value": predicted_value,
        "predicted_robustness": pred_value_hard,
        "predicted_satisfied": bool(pred_sat),
        "actual_robustness": actual_value,
        "actual_satisfied": bool(actual_sat),
    }
    np.savez_compressed(
        formula_dir / f"rollout_{rollout_idx:03d}.npz",
        observations=actual_obs_arr,
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        contacts=np.asarray(contacts, dtype=np.int32),
        predicted=predicted_first.astype(np.float32) if predicted_first is not None else np.zeros((0, TRANSITION_DIM), dtype=np.float32),
    )
    if config.render_video and frames:
        imageio.mimsave(formula_dir / f"rollout_{rollout_idx:03d}.mp4", frames, fps=10)
    return result


########
#
# ROLLOUT PLOTS AND SUMMARY METRICS
# Plots show the actual executed agent path against fixed block locations.
# summary.json reports actual rollout success separately from predicted plan
# satisfaction so receding-horizon correction is visible in the results.
#
#######

def plot_formula_rollouts(formula_dir: Path, formula: FormulaSpec, results: List[Dict], max_rollouts: int = 12) -> None:
    n = min(max_rollouts, len(results))
    if n == 0:
        return
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), dpi=160, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for idx, result in enumerate(results[:n]):
        ax = axes.flat[idx]
        data = np.load(formula_dir / f"rollout_{idx:03d}.npz")
        obs = data["observations"]
        start = obs[0]
        blocks = {name: start[block_slice] for name, block_slice in LABEL_TO_BLOCK_SLICE.items()}
        for name, xy in blocks.items():
            ax.scatter([xy[0]], [xy[1]], s=180, marker="s", c=LABEL_COLORS[name], alpha=0.82, edgecolor="black", linewidth=0.5)
        ax.plot(obs[:, 0], obs[:, 1], color="#111111", linewidth=1.5, alpha=0.82)
        ax.scatter([obs[0, 0]], [obs[0, 1]], s=70, c="#888888", edgecolor="white", linewidth=0.8)
        ax.scatter([obs[-1, 0]], [obs[-1, 1]], s=70, c="#000000", edgecolor="white", linewidth=0.8)
        title = f"{idx}: r={result['actual_robustness']:.3f} sat={int(result['actual_satisfied'])}"
        if result["first_contact"] >= 0:
            title += f" hit={LABEL_NAMES[result['first_contact']]}"
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.18)
    fig.suptitle(f"{formula.name}: {formula.description}", fontsize=12)
    fig.tight_layout()
    fig.savefig(formula_dir / "rollouts.png")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    config = EvalConfig(**{k: v for k, v in vars(args).items() if k != "func"})
    seed_everything(config.seed)
    device = get_device(config.device)
    diffusion, payload = load_checkpoint(config.checkpoint, device)
    diffusion.timesteps = int(config.diffusion_steps)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_config.json").write_text(json.dumps(asdict(config), indent=2))

    dataset = ToySquaresTrajectoryDataset(config.data_path, horizon=int(payload["config"]["horizon"]), max_demos=0, min_start_gap=64)
    initial_obs_list = dataset.sample_initial_obs(config.n_rollouts, seed=config.seed + 100) if config.reset_mode == "dataset" else []
    formulas = parse_formula_selection(config.formulas, config.formula_limit)
    all_results = {}

    for formula in formulas:
        formula_dir = output_dir / formula.name
        formula_dir.mkdir(parents=True, exist_ok=True)
        results = []
        iterator = tqdm(range(config.n_rollouts), desc=formula.name, dynamic_ncols=True)
        for idx in iterator:
            if config.reset_mode == "dataset":
                initial_obs = initial_obs_list[idx]
            else:
                env = make_env()
                initial_obs = obs_from_env_dict(env.reset())
            result = rollout_formula(diffusion, formula, initial_obs, config, device, idx, formula_dir)
            results.append(result)
            iterator.set_postfix(sat=np.mean([r["actual_satisfied"] for r in results]))
        summary = summarize_results(results)
        payload_formula = {"summary": summary, "rollouts": results}
        (formula_dir / "results.json").write_text(json.dumps(payload_formula, indent=2))
        plot_formula_rollouts(formula_dir, formula, results)
        all_results[formula.name] = summary

    (output_dir / "summary.json").write_text(json.dumps(all_results, indent=2))


def summarize_results(results: List[Dict]) -> Dict:
    if not results:
        return {}
    return {
        "n": len(results),
        "satisfaction_rate": float(np.mean([r["actual_satisfied"] for r in results])),
        "predicted_satisfaction_rate": float(np.mean([r["predicted_satisfied"] for r in results])),
        "mean_actual_robustness": float(np.mean([r["actual_robustness"] for r in results])),
        "mean_predicted_robustness": float(np.mean([r["predicted_robustness"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "contact_rate": float(np.mean([r["first_contact"] >= 0 for r in results])),
    }


########
#
# COMMAND LINE INTERFACE
# The file exposes two subcommands: train for fitting the Diffuser, and eval
# for LTLDoG-S guided posterior sampling plus real TouchCube rollouts.
#
#######

def add_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train")
    defaults = TrainConfig()
    for field, value in asdict(defaults).items():
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", action="store_true" if not value else "store_false")
        else:
            parser.add_argument(f"--{field}", type=arg_type, default=value)
    parser.set_defaults(func=train)


def add_eval_parser(subparsers) -> None:
    parser = subparsers.add_parser("eval")
    defaults = EvalConfig(checkpoint="")
    for field, value in asdict(defaults).items():
        arg_type = type(value)
        if field == "checkpoint":
            parser.add_argument(f"--{field}", required=True)
        elif isinstance(value, bool):
            parser.add_argument(f"--{field}", action="store_true" if not value else "store_false")
        else:
            parser.add_argument(f"--{field}", type=arg_type, default=value)
    parser.set_defaults(func=evaluate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy Squares full-trajectory Diffuser + LTLDoG-S baseline")
    subparsers = parser.add_subparsers(required=True)
    add_train_parser(subparsers)
    add_eval_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
