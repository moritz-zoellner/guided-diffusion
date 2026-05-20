#!/usr/bin/env python
"""Train a TeLoGraF-style CALVIN low-dimensional flow policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TELOGRAF_CODE = REPO_ROOT / "TeLoGraF" / "code"
if str(TELOGRAF_CODE) not in sys.path:
    sys.path.insert(0, str(TELOGRAF_CODE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from z_diffuser import GaussianFlow, TemporalUnet  # noqa: E402

    TELOGRAF_BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional TeLoGraF deps.
    GaussianFlow = None
    TemporalUnet = None
    TELOGRAF_BACKEND_IMPORT_ERROR = repr(exc)

from telograf_calvin.paper_specs import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    evaluate_spec_sequence,
    spec_to_vector,
    write_json,
)


DATA_DIM = STATE_DIM + ACTION_DIM


class LocalTemporalFlowNet(nn.Module):
    """Small flow-matching fallback for CALVIN envs missing TeLoGraF deps."""

    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        denom = max(1.0, float(torch.max(t).detach().cpu().item() + 1.0))
        t_norm = t.float().view(-1, 1) / denom
        ctx = self.cond_mlp(cond) + self.time_mlp(t_norm.to(cond.device))
        ctx = ctx[:, None, :].expand(-1, x.shape[1], -1)
        return self.net(torch.cat([x, ctx], dim=-1))


class LocalGaussianFlow(nn.Module):
    def __init__(self, model: nn.Module, horizon: int, transition_dim: int, n_timesteps: int):
        super().__init__()
        self.model = model
        self.horizon = int(horizon)
        self.transition_dim = int(transition_dim)
        self.n_timesteps = int(n_timesteps)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        tt = ((t + 1).float() / self.n_timesteps)[:, None, None]
        return tt * noise + (1.0 - tt) * x_start

    @torch.no_grad()
    def conditional_sample(self, cond: torch.Tensor, args=None):
        x = torch.randn((len(cond), self.horizon, self.transition_dim), device=cond.device)
        t = torch.full((len(cond),), self.n_timesteps - 1, device=cond.device, dtype=torch.long)
        x = x + self.model(x, cond, t)
        return SimpleNamespace(trajectories=x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Generated TeLoGraF-CALVIN data.npz")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/telograf/runs"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-timesteps", type=int, default=100)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One fast epoch over a small subset.")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument(
        "--require-telograf-backend",
        action="store_true",
        help="Fail instead of using the local fallback if TeLoGraF z_diffuser cannot be imported.",
    )
    return parser.parse_args()


def load_npz(path: Path, max_records: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    payload = np.load(path, allow_pickle=True)
    records = list(payload["data"])
    if max_records is not None:
        records = records[: int(max_records)]
    meta = {}
    if "meta" in payload:
        meta = json.loads(str(payload["meta"].item()))
    return records, meta


def transition_from_record(record: Mapping) -> np.ndarray:
    traj = np.asarray(record["trajs"], dtype=np.float32)
    actions = np.asarray(record["us"], dtype=np.float32)
    return np.concatenate([traj[:-1], actions], axis=-1).astype(np.float32)


def compute_stats(records: List[Mapping]) -> Dict[str, np.ndarray]:
    transitions = np.stack([transition_from_record(record) for record in records], axis=0)
    flat = transitions.reshape(-1, transitions.shape[-1])
    states = np.stack([np.asarray(record["state"], dtype=np.float32) for record in records], axis=0)
    return {
        "transition_mean": flat.mean(axis=0).astype(np.float32),
        "transition_std": (flat.std(axis=0) + 1e-6).astype(np.float32),
        "state_mean": states.mean(axis=0).astype(np.float32),
        "state_std": (states.std(axis=0) + 1e-6).astype(np.float32),
    }


class CalvinFlowDataset(Dataset):
    def __init__(self, records: List[Mapping], stats: Mapping[str, np.ndarray]):
        self.records = records
        self.stats = stats
        self.spec_dim = len(spec_to_vector(records[0]["spec"])) if records else 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        transition = transition_from_record(record)
        transition = (transition - self.stats["transition_mean"]) / self.stats["transition_std"]
        init_state = np.asarray(record["state"], dtype=np.float32)
        init_state = (init_state - self.stats["state_mean"]) / self.stats["state_std"]
        spec_vec = spec_to_vector(record["spec"])
        cond = np.concatenate([init_state, spec_vec], axis=0).astype(np.float32)
        return {
            "transition": torch.from_numpy(transition),
            "cond": torch.from_numpy(cond),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def split_records(records: List[Mapping]) -> Tuple[List[Mapping], List[Mapping]]:
    train = [record for record in records if str(record.get("split")) == "train"]
    valid = [record for record in records if str(record.get("split")) == "valid"]
    if not valid and len(train) > 10:
        valid = train[-max(1, len(train) // 10) :]
        train = train[: -len(valid)]
    if not train:
        raise RuntimeError("No train records found.")
    if not valid:
        valid = train[: min(len(train), 16)]
    return train, valid


def denorm_transition(x: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    return x * stats["transition_std"] + stats["transition_mean"]


@torch.no_grad()
def evaluate_ground_truth(records: List[Mapping]) -> Dict:
    by_spec = {}
    total_ok = 0
    scores = []
    for record in records:
        traj = np.asarray(record["trajs"], dtype=np.float32)
        robot = traj[:, :15]
        scene = traj[:, 15:]
        ok, score = evaluate_spec_sequence(record["spec"], robot, scene)
        total_ok += int(ok)
        scores.append(float(score))
        spec_id = record["spec_id"]
        by_spec.setdefault(spec_id, [0, 0])
        by_spec[spec_id][0] += int(ok)
        by_spec[spec_id][1] += 1
    return {
        "count": int(len(records)),
        "satisfaction_rate": float(total_ok / max(1, len(records))),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "by_spec": {spec: float(ok / max(1, total)) for spec, (ok, total) in by_spec.items()},
    }


@torch.no_grad()
def evaluate_generated(
    diffuser: GaussianFlow,
    dataset: CalvinFlowDataset,
    records: List[Mapping],
    stats: Mapping[str, np.ndarray],
    device: torch.device,
    max_samples: int,
    n_timesteps: int,
) -> Dict:
    n = min(len(records), int(max_samples))
    if n <= 0:
        return {"count": 0, "satisfaction_rate": 0.0, "mean_score": 0.0, "by_spec": {}}
    cond = torch.stack([dataset[i]["cond"] for i in range(n)], dim=0).to(device)
    sample_args = SimpleNamespace(flow_pattern=13)
    generated = diffuser.conditional_sample(cond, args=sample_args).trajectories.detach().cpu().numpy()
    generated = denorm_transition(generated, stats)
    by_spec = {}
    ok_count = 0
    scores = []
    for i in range(n):
        record = records[i]
        states = generated[i, :, :STATE_DIM]
        robot = states[:, :15]
        scene = states[:, 15:]
        ok, score = evaluate_spec_sequence(record["spec"], robot, scene)
        ok_count += int(ok)
        scores.append(float(score))
        by_spec.setdefault(record["spec_id"], [0, 0])
        by_spec[record["spec_id"]][0] += int(ok)
        by_spec[record["spec_id"]][1] += 1
    return {
        "count": int(n),
        "satisfaction_rate": float(ok_count / max(1, n)),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "by_spec": {spec: float(ok / max(1, total)) for spec, (ok, total) in by_spec.items()},
    }


def plot_xy_samples(path: Path, records: List[Mapping], title: str, max_items: int = 16) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for record in records[:max_items]:
        traj = np.asarray(record["trajs"], dtype=np.float32)
        xy = traj[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], alpha=0.45, linewidth=1.3)
        ax.scatter(xy[0, 0], xy[0, 1], s=12, color="#111827")
    ax.set_xlabel("EEF x")
    ax.set_ylabel("EEF y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_records = args.max_records or 64
        args.batch_size = min(args.batch_size, 16)
        args.eval_samples = min(args.eval_samples, 16)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    run_name = args.name or f"calvin_flow_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records, meta = load_npz(args.data, max_records=args.max_records)
    train_records, valid_records = split_records(records)
    stats = compute_stats(train_records)
    train_dataset = CalvinFlowDataset(train_records, stats)
    valid_dataset = CalvinFlowDataset(valid_records, stats)

    horizon = int(np.asarray(train_records[0]["us"]).shape[0])
    cond_dim = STATE_DIM + train_dataset.spec_dim

    if GaussianFlow is not None and TemporalUnet is not None:
        backend = "telograf_z_diffuser"
        model = TemporalUnet(
            horizon=horizon,
            transition_dim=DATA_DIM,
            cond_dim=cond_dim,
            dim=args.dim,
            dim_mults=tuple(args.dim_mults),
            attention=True,
        ).to(device)
        diffuser = GaussianFlow(
            model,
            horizon=horizon,
            observation_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_timesteps=args.n_timesteps,
        ).to(device)
    else:
        backend = "local_flow_fallback"
        if args.require_telograf_backend:
            raise RuntimeError(
                "TeLoGraF z_diffuser backend is required but unavailable: "
                f"{TELOGRAF_BACKEND_IMPORT_ERROR}"
            )
        print(f"TeLoGraF z_diffuser unavailable, using local flow fallback: {TELOGRAF_BACKEND_IMPORT_ERROR}")
        model = LocalTemporalFlowNet(DATA_DIM, cond_dim, max(args.dim, 16)).to(device)
        diffuser = LocalGaussianFlow(model, horizon=horizon, transition_dim=DATA_DIM, n_timesteps=args.n_timesteps).to(device)
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=args.lr)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    def make_checkpoint_payload() -> Dict:
        return {
            "model": diffuser.state_dict(),
            "stats": {k: v for k, v in stats.items()},
            "meta": meta,
            "args": vars(args),
            "horizon": horizon,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "condition_dim": cond_dim,
            "spec_dim": train_dataset.spec_dim,
            "backend": backend,
            "telograf_backend_import_error": TELOGRAF_BACKEND_IMPORT_ERROR,
        }

    logs = []
    for epoch in range(args.epochs):
        diffuser.train()
        losses = []
        for batch in loader:
            x = batch["transition"].to(device)
            cond = batch["cond"].to(device)
            batch_size = x.shape[0]
            t = torch.randint(0, args.n_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x)
            x_noisy = diffuser.q_sample(x_start=x, t=t, noise=noise)
            pred = diffuser.model(x_noisy, cond, t)
            target = x - noise
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffuser.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log = {"epoch": int(epoch + 1), "train_loss": float(np.mean(losses))}
        logs.append(log)
        print(f"epoch {epoch + 1:04d}/{args.epochs:04d} train_loss={log['train_loss']:.6f}")
        if args.save_every and (epoch + 1) % int(args.save_every) == 0:
            torch.save(make_checkpoint_payload(), run_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")

    gt_eval = evaluate_ground_truth(valid_records)
    gen_eval = None
    if not args.skip_sampling:
        diffuser.eval()
        gen_eval = evaluate_generated(
            diffuser,
            valid_dataset,
            valid_records,
            stats,
            device,
            max_samples=args.eval_samples,
            n_timesteps=args.n_timesteps,
        )

    torch.save(make_checkpoint_payload(), run_dir / "checkpoint.pt")
    np.savez(
        run_dir / "normalization_stats.npz",
        **{k: v for k, v in stats.items()},
    )
    write_json(
        run_dir / "metrics.json",
        {
            "logs": logs,
            "ground_truth_valid": gt_eval,
            "generated_valid": gen_eval,
            "num_train": len(train_records),
            "num_valid": len(valid_records),
            "data": str(args.data),
            "backend": backend,
            "telograf_backend_import_error": TELOGRAF_BACKEND_IMPORT_ERROR,
        },
    )
    plot_xy_samples(run_dir / "valid_ground_truth_xy.png", valid_records, "Valid ground-truth mined windows")

    print(f"\nwrote run to {run_dir}")
    print(f"ground-truth valid satisfaction={gt_eval['satisfaction_rate']:.3f}")
    if gen_eval is not None:
        print(f"generated valid satisfaction={gen_eval['satisfaction_rate']:.3f}")


if __name__ == "__main__":
    main()
