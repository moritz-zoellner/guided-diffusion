import os
import json
import math
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------

STATE_KEYS = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_vel",
]


def demo_num(demo_id: str) -> int:
    return int(demo_id.split("_")[-1])


def build_state_vector_from_hdf(demo_grp, t: int) -> np.ndarray:
    obs_grp = demo_grp["obs"]
    parts = [obs_grp[k][t].reshape(-1) for k in STATE_KEYS]
    return np.concatenate(parts, axis=0).astype(np.float32)  # (39,)


# -----------------------------
# Dataset
# -----------------------------

class RzzChunkDataset(Dataset):
    """
    Returns: (s_t, a_chunk_flat, y) where y = rzz[t+K]
      s_t: (39,)
      a_chunk_flat: (K*7,)
      y: scalar
    """
    def __init__(self, demo_hdf_path: str, rzz_hdf_path: str, K: int, demo_ids: List[str]):
        self.demo_hdf_path = demo_hdf_path
        self.rzz_hdf_path = rzz_hdf_path
        self.K = K
        self.demo_ids = list(demo_ids)

        # build flat index of (demo_id, t)
        with h5py.File(self.demo_hdf_path, "r") as f_demo:
            self.index: List[Tuple[str, int]] = []
            for demo_id in self.demo_ids:
                T = f_demo["data"][demo_id]["actions"].shape[0]
                for t in range(0, T - self.K):
                    self.index.append((demo_id, t))

        self._f_demo = None
        self._f_rzz = None

    def _lazy_open(self):
        if self._f_demo is None:
            self._f_demo = h5py.File(self.demo_hdf_path, "r")
        if self._f_rzz is None:
            self._f_rzz = h5py.File(self.rzz_hdf_path, "r")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        self._lazy_open()
        demo_id, t = self.index[idx]

        demo = self._f_demo["data"][demo_id]
        s_t = build_state_vector_from_hdf(demo, t)  # (39,)

        a = demo["actions"][t:t + self.K].astype(np.float32)  # (K,7)
        a = a.reshape(-1)  # (K*7,)

        y = self._f_rzz["data"][demo_id]["rzz"][t + self.K].astype(np.float32)

        return (
            torch.from_numpy(s_t),
            torch.from_numpy(a),
            torch.tensor(y, dtype=torch.float32),
        )

    def close(self):
        if self._f_demo is not None:
            self._f_demo.close()
            self._f_demo = None
        if self._f_rzz is not None:
            self._f_rzz.close()
            self._f_rzz = None


# -----------------------------
# Model + Normalization
# -----------------------------

class Normalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)


class RzzMLP(nn.Module):
    def __init__(self, state_dim: int, act_flat_dim: int):
        super().__init__()
        d_in = state_dim + act_flat_dim

        self.in_ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.out = nn.Linear(256, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        x = self.in_ln(x)
        x = F.silu(self.ln1(self.fc1(x)))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        y = torch.tanh(self.out(x)).squeeze(-1)  # keep bounded [-1,1]
        return y


@torch.no_grad()
def estimate_mean_std(dl: DataLoader, num_batches: int = 200, device: str = "cpu"):
    s_list, a_list = [], []
    for i, (s, a, y) in enumerate(dl):
        s_list.append(s)
        a_list.append(a)
        if i + 1 >= num_batches:
            break
    S = torch.cat(s_list, dim=0).to(device)
    A = torch.cat(a_list, dim=0).to(device)
    return S.mean(0), S.std(0), A.mean(0), A.std(0)


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).mean() / (x.std() * y.std() + 1e-8))


# -----------------------------
# Train
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_hdf", type=str, required=True)
    ap.add_argument("--rzz_hdf", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # list demos
    with h5py.File(args.demo_hdf, "r") as f:
        demo_ids = sorted(list(f["data"].keys()), key=demo_num)

    # split by demo id
    n = len(demo_ids)
    n_val = max(1, int(round(n * args.val_frac)))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    val_set = set([demo_ids[i] for i in perm[:n_val]])
    train_ids = [d for d in demo_ids if d not in val_set]
    val_ids = [d for d in demo_ids if d in val_set]

    print(f"demos: total={n} train={len(train_ids)} val={len(val_ids)}  K={args.K}")

    train_ds = RzzChunkDataset(args.demo_hdf, args.rzz_hdf, K=args.K, demo_ids=train_ids)
    val_ds   = RzzChunkDataset(args.demo_hdf, args.rzz_hdf, K=args.K, demo_ids=val_ids)

    # loaders
    train_dl_stats = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    s_mean, s_std, a_mean, a_std = estimate_mean_std(train_dl_stats, num_batches=200, device="cpu")
    s_norm = Normalizer(s_mean, s_std).to(device)
    a_norm = Normalizer(a_mean, a_std).to(device)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=False,
    )

    state_dim = int(s_mean.numel())
    act_dim = int(a_mean.numel())

    model = RzzMLP(state_dim=state_dim, act_flat_dim=act_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_losses = []
        tr_corrs = []
        for s, a, y in train_dl:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            s = s_norm(s)
            a = a_norm(a)

            pred = model(s, a)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr_losses.append(float(loss.detach().cpu()))
            tr_corrs.append(corrcoef(pred.detach().cpu(), y.detach().cpu()))

        # ---- val ----
        model.eval()
        va_losses = []
        va_corrs = []
        with torch.no_grad():
            for s, a, y in val_dl:
                s = s.to(device, non_blocking=True)
                a = a.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                s = s_norm(s)
                a = a_norm(a)

                pred = model(s, a)
                loss = F.mse_loss(pred, y)
                va_losses.append(float(loss.detach().cpu()))
                va_corrs.append(corrcoef(pred.detach().cpu(), y.detach().cpu()))

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        tr_c = float(np.mean(tr_corrs))
        va_c = float(np.mean(va_corrs))
        print(f"epoch {epoch:03d} | train mse {tr_loss:.6f} corr {tr_c:.3f} | val mse {va_loss:.6f} corr {va_c:.3f}")

        # save best
        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "s_mean": s_mean,
                "s_std": s_std,
                "a_mean": a_mean,
                "a_std": a_std,
                "K": args.K,
                "STATE_KEYS": STATE_KEYS,
                "train_demo_ids": train_ids,
                "val_demo_ids": val_ids,
            }
            torch.save(ckpt, args.out)
            print("saved:", args.out)

    # close hdf5 handles in main process
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()