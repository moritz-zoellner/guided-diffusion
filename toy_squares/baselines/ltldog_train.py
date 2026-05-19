from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toy_squares.baselines.ltldog_toy import (  # noqa: E402
    FormulaSpec,
    GaussianTrajectoryDiffusion,
    ToyLTLRobustness,
    add_train_parser,
    get_device,
    load_checkpoint,
)


########
#
# PUBLIC LTLDoG TRAINING AND LOADING API
# This file is the stable entry point for training and loading the Toy Squares
# LTLDoG Diffuser. The model architecture still lives in ltldog_toy.py for now,
# but users should import load_ltldog_planner() from here for experiments.
#
# Mental model:
#   1. ltldog_toy.py defines the neural Diffuser and robustness math.
#   2. This file wraps a trained checkpoint into a tiny LTLDogPlanner object.
#   3. Rollout scripts call planner.sample_plan(obs, formula, ...).
#   4. sample_plan returns full H-step trajectories sorted by guided
#      robustness, plus the corresponding robustness values.
#
# Keeping this wrapper small is intentional: rollout code should not have to
# know about EMA weights, torch devices, checkpoint payload structure, or how
# the differentiable LTL robustness object is constructed.
#
#######


@dataclass
class LTLDogPlanner:
    # The loaded diffusion module. It contains both the TemporalDenoiser and
    # the Gaussian reverse-process buffers (betas, alphas, posterior variance).
    diffusion: GaussianTrajectoryDiffusion

    # Raw checkpoint dictionary. We keep it around so experiments can inspect
    # the training horizon and architecture without reopening the checkpoint.
    checkpoint_payload: Dict

    # Torch device used for all sampling calls. The public sample_plan API
    # accepts / returns numpy arrays, so callers never need to manage this.
    device: torch.device

    @property
    def train_config(self) -> Dict:
        # Stored by ltldog_toy.train(); includes horizon, channels, depth, etc.
        return dict(self.checkpoint_payload["config"])

    @property
    def horizon(self) -> int:
        return int(self.train_config["horizon"])

    @property
    def diffusion_steps(self) -> int:
        return int(self.diffusion.timesteps)

    def make_robustness(self, formula: FormulaSpec, radius: float = 0.2, tau: float = 0.05) -> ToyLTLRobustness:
        # ToyLTLRobustness is differentiable. During sampling, gradients of this
        # scalar robustness are inserted into the reverse diffusion process.
        return ToyLTLRobustness(formula, radius=radius, tau=tau)

    @torch.no_grad()
    def sample_plan(
        self,
        observation: np.ndarray,
        formula: FormulaSpec,
        batch_size: int = 4,
        guidance_scale: float = 0.01,
        n_guide_steps: int = 5,
        t_stopgrad: int = 2,
        guidance_mode: str = "ps",
        scale_grad_by_std: bool = False,
        guidance_threshold: float = 0.0,
        radius: float = 0.2,
        tau: float = 0.05,
        freeze_static_blocks: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Convert the current env observation into the conditioning vector used
        # by the Diffuser. We repeat it across the sampling batch because each
        # candidate plan starts from the same observed state but different noise.
        robustness = self.make_robustness(formula, radius=radius, tau=tau)
        condition = torch.as_tensor(observation[None], dtype=torch.float32, device=self.device).repeat(int(batch_size), 1)

        # diffusion.sample() performs posterior-sampling guidance internally.
        # In the final paper baseline, the first returned sample is then
        # executed open-loop for the full horizon by paper_horizon_test.py.
        samples, values = self.diffusion.sample(
            condition,
            batch_size=int(batch_size),
            guide=robustness,
            sample_kwargs={
                "scale": float(guidance_scale),
                "n_guide_steps": int(n_guide_steps),
                "t_stopgrad": int(t_stopgrad),
                "guidance_mode": str(guidance_mode),
                "scale_grad_by_std": bool(scale_grad_by_std),
                "threshold": float(guidance_threshold),
                "freeze_static_blocks": bool(freeze_static_blocks),
            },
        )
        return samples.detach().cpu().numpy(), values.detach().cpu().numpy()


########
#
# CHECKPOINT LOADING
# load_ltldog_planner() returns a small object with sample_plan(), so evaluation
# code does not need to know about checkpoint payloads, EMA weights, or device
# placement. diffusion_steps can be overridden for faster/slower sampling tests.
#
#######


def load_ltldog_planner(
    checkpoint: str | Path,
    device: str | torch.device = "auto",
    diffusion_steps: int | None = None,
) -> LTLDogPlanner:
    # load_checkpoint handles checkpoint payload format and EMA weights. This
    # wrapper only resolves the requested device and optionally overrides the
    # number of reverse diffusion steps for evaluation-time ablations.
    resolved_device = get_device(device) if isinstance(device, str) else device
    diffusion, payload = load_checkpoint(str(checkpoint), resolved_device)
    if diffusion_steps is not None:
        diffusion.timesteps = int(diffusion_steps)
    diffusion.eval()
    return LTLDogPlanner(diffusion=diffusion, checkpoint_payload=payload, device=resolved_device)


########
#
# TRAINING CLI
# The train subcommand delegates to the original training implementation, while
# this module gives downstream eval scripts a cleaner import surface.
#
#######


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/load Toy Squares LTLDoG Diffuser")
    subparsers = parser.add_subparsers(required=True)
    add_train_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
