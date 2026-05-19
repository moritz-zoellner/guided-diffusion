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
    ACTION_DIM,
    DEFAULT_FORMULAS,
    LABEL_COLORS,
    LABEL_NAMES,
    LABEL_TO_BLOCK_SLICE,
    OBS_DIM,
    TRANSITION_DIM,
    EvalConfig,
    FormulaSpec,
    GaussianTrajectoryDiffusion,
    ToyLTLRobustness,
    TrainConfig,
    add_train_parser,
    build_model_from_config,
    get_device,
    load_checkpoint,
    seed_everything,
    train,
)


########
#
# PUBLIC LTLDoG TRAINING AND LOADING API
# This file is the stable entry point for training and loading the Toy Squares
# LTLDoG Diffuser. The model architecture still lives in ltldog_toy.py for now,
# but users should import load_ltldog_planner() from here for experiments.
#
#######


@dataclass
class LTLDogPlanner:
    diffusion: GaussianTrajectoryDiffusion
    checkpoint_payload: Dict
    device: torch.device

    @property
    def train_config(self) -> Dict:
        return dict(self.checkpoint_payload["config"])

    @property
    def horizon(self) -> int:
        return int(self.train_config["horizon"])

    @property
    def diffusion_steps(self) -> int:
        return int(self.diffusion.timesteps)

    def make_robustness(self, formula: FormulaSpec, radius: float = 0.2, tau: float = 0.05) -> ToyLTLRobustness:
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
        robustness = self.make_robustness(formula, radius=radius, tau=tau)
        condition = torch.as_tensor(observation[None], dtype=torch.float32, device=self.device).repeat(int(batch_size), 1)
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
