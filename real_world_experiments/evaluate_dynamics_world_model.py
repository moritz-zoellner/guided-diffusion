"""Evaluate a real-world low-level dynamics model.

This mirrors the CALVIN dynamics evaluator, but uses the compact Cheez-It state:

    [eef_pos, eef_rot6d, gripper_width, cheezit_pos, cheezit_rot6d]

The main diagnostic is a one-step component CDF comparing the learned model to a
zero-delta predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from train_dynamics_world_model import (
        build_dynamics_trajectories,
        flatten_trajectories,
        load_dynamics_model_for_eval,
    )
except ModuleNotFoundError:
    from real_world_experiments.train_dynamics_world_model import (
        build_dynamics_trajectories,
        flatten_trajectories,
        load_dynamics_model_for_eval,
    )


DEFAULT_COMPONENT_DIMS = {
    "eef_pos": 3,
    "eef_rot6d": 6,
    "gripper_width": 1,
    "gripper_binary": 1,
    "cheezit_pos": 3,
    "cheezit_rot6d": 6,
}


def resolve_repo_root(start: Path) -> Path:
    start = Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "real_world_experiments" / "train_dynamics_world_model.py").exists():
            return path
    raise FileNotFoundError(f"Could not resolve repo root from {start}")


def normalize_rot6d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    r1 = values[..., 0:3]
    r2 = values[..., 3:6]
    r1 = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)
    r2 = r2 - np.sum(r1 * r2, axis=-1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-8)
    values[..., 0:3] = r1
    values[..., 3:6] = r2
    return values


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = normalize_rot6d(values)
    r1 = values[..., 0:3]
    r2 = values[..., 3:6]
    r3 = np.cross(r1, r2, axis=-1)
    return np.stack([r1, r2, r3], axis=-1)


def rot6d_geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = rot6d_to_matrix(a)
    rb = rot6d_to_matrix(b)
    r_delta = np.matmul(np.swapaxes(ra, -1, -2), rb)
    trace = np.trace(r_delta, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)).astype(np.float32)


def state_slices(state_keys: list[str]) -> dict[str, slice]:
    cursor = 0
    slices = {}
    for key in state_keys:
        if key not in DEFAULT_COMPONENT_DIMS:
            raise ValueError(
                f"Unknown state key {key!r}; add its dimensionality to DEFAULT_COMPONENT_DIMS."
            )
        dim = DEFAULT_COMPONENT_DIMS[key]
        slices[key] = slice(cursor, cursor + dim)
        cursor += dim
    return slices


def project_state_rot6d(state: np.ndarray, slices: dict[str, slice]) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    for key, slc in slices.items():
        if key.endswith("_rot6d"):
            state[..., slc] = normalize_rot6d(state[..., slc])
    return state


def component_specs(state_keys: list[str]) -> dict[str, tuple[str, slice, str]]:
    slices = state_slices(state_keys)
    specs = {}
    for key, slc in slices.items():
        if key.endswith("_pos"):
            specs[key] = ("dist", slc, "position distance [m]")
        elif key.endswith("_rot6d"):
            specs[key.replace("_rot6d", "_rot")] = ("rot_deg", slc, "rotation error [deg]")
        elif key in {"gripper_width", "gripper_binary"}:
            specs[key] = ("abs", slc, "absolute scalar error")
        else:
            specs[key] = ("l2", slc, "component L2 error")
    return specs


def predict_next_states_batched(
    states: np.ndarray,
    actions: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    device: str,
    batch_size: int,
    project_rot6d: bool,
    slices: dict[str, slice],
) -> np.ndarray:
    states_n = ((states - stats["state_mean"]) / stats["state_std"]).astype(np.float32)
    actions_n = ((actions - stats["action_mean"]) / stats["action_std"]).astype(np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states_n), batch_size):
            end = start + batch_size
            delta_n = model(
                torch.from_numpy(states_n[start:end]).to(device),
                torch.from_numpy(actions_n[start:end]).to(device),
            ).cpu().numpy()
            delta = delta_n * stats["delta_std"] + stats["delta_mean"]
            next_state = states[start:end] + delta
            if project_rot6d:
                next_state = project_state_rot6d(next_state, slices)
            preds.append(next_state.astype(np.float32))
    return np.concatenate(preds, axis=0)


def predict_next_state(
    state: np.ndarray,
    action: np.ndarray,
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    device: str,
    project_rot6d: bool,
    slices: dict[str, slice],
) -> np.ndarray:
    return predict_next_states_batched(
        state[None],
        action[None],
        model,
        stats,
        device,
        batch_size=1,
        project_rot6d=project_rot6d,
        slices=slices,
    )[0]


def component_error(true_next: np.ndarray, pred_next: np.ndarray, metric: str, slc: slice) -> np.ndarray:
    true_comp = true_next[:, slc]
    pred_comp = pred_next[:, slc]
    if metric in {"dist", "l2"}:
        return np.linalg.norm(pred_comp - true_comp, axis=-1).astype(np.float32)
    if metric == "abs":
        return np.abs(pred_comp - true_comp).reshape(len(true_next), -1).mean(axis=-1).astype(np.float32)
    if metric == "rot_deg":
        return rot6d_geodesic_deg(pred_comp, true_comp)
    raise ValueError(f"Unknown component metric: {metric}")


def one_step_component_errors(
    flat: dict[str, np.ndarray],
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    device: str,
    batch_size: int,
    state_keys: list[str],
    project_rot6d: bool,
):
    slices = state_slices(state_keys)
    pred_next = predict_next_states_batched(
        flat["states"],
        flat["actions"],
        model,
        stats,
        device,
        batch_size,
        project_rot6d=project_rot6d,
        slices=slices,
    )
    zero_next = flat["states"]
    true_next = flat["next_states"]

    errors = {"learned": {}, "zero_delta": {}}
    descriptors = {}
    for name, (metric, slc, description) in component_specs(state_keys).items():
        errors["learned"][name] = component_error(true_next, pred_next, metric, slc)
        errors["zero_delta"][name] = component_error(true_next, zero_next, metric, slc)
        descriptors[name] = {"metric": metric, "description": description}
    return errors, descriptors


def summarize_error_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def save_component_cdf_plots(errors: dict, descriptors: dict, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        component: {
            model_name: summarize_error_values(errors[model_name][component])
            for model_name in ("learned", "zero_delta")
        }
        for component in descriptors
    }

    components = list(descriptors)
    n_cols = 3
    n_rows = int(np.ceil(len(components) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    colors = {"learned": "#2563eb", "zero_delta": "#f97316"}

    for idx, component in enumerate(components):
        ax = axes[idx]
        for model_name in ("learned", "zero_delta"):
            values = np.sort(errors[model_name][component])
            y = np.linspace(0.0, 100.0, len(values), endpoint=True)
            ax.plot(values, y, label=model_name, color=colors[model_name], linewidth=1.8)
        p99 = max(summary[component]["learned"]["p99"], summary[component]["zero_delta"]["p99"])
        if p99 > 0:
            ax.set_xlim(left=0.0, right=p99 * 1.05)
        ax.set_title(component)
        ax.set_xlabel(descriptors[component]["description"])
        ax.set_ylabel("% transitions <= error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(components):]:
        ax.axis("off")

    fig.tight_layout()
    plot_path = output_dir / f"{prefix}_component_error_cdfs.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = output_dir / f"{prefix}_component_error_cdf_summary.json"
    summary_path.write_text(json.dumps({"descriptors": descriptors, "summary": summary}, indent=2))
    return plot_path, summary_path, summary


def summarize_final_errors(final_rows: list[dict], state_keys: list[str]) -> dict:
    true = np.stack([row["true_final"] for row in final_rows], axis=0)
    learned = np.stack([row["learned_final"] for row in final_rows], axis=0)
    zero = np.stack([row["zero_final"] for row in final_rows], axis=0)
    specs = component_specs(state_keys)
    out = {}
    for name, pred in (("learned", learned), ("zero_delta", zero)):
        errors = pred - true
        out[name] = {
            "rmse_per_value": float(np.sqrt(np.mean(np.square(errors)))),
            "mae_per_value": float(np.mean(np.abs(errors))),
            "l2_mean": float(np.mean(np.linalg.norm(errors, axis=-1))),
            "l2_median": float(np.median(np.linalg.norm(errors, axis=-1))),
            "l2_max": float(np.max(np.linalg.norm(errors, axis=-1))),
            "components": {},
        }
        for comp, (metric, slc, description) in specs.items():
            out[name]["components"][comp] = summarize_error_values(component_error(true, pred, metric, slc))
            out[name]["components"][comp]["description"] = description
    return out


def finite_status(final_rows: list[dict]) -> dict:
    statuses = {}
    for key in ("true_final", "learned_final", "zero_final"):
        values = np.stack([row[key] for row in final_rows], axis=0)
        statuses[key] = {
            "all_finite": bool(np.isfinite(values).all()),
            "finite_fraction": float(np.isfinite(values).mean()),
            "max_abs_finite": float(np.max(np.abs(values[np.isfinite(values)]))) if np.isfinite(values).any() else None,
        }
    return statuses


def save_summary_plot(summary: dict, path: Path):
    names = ["learned", "zero_delta"]
    rmse = [summary[name]["rmse_per_value"] for name in names]
    l2 = [summary[name]["l2_mean"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rmse, color=["#2563eb", "#f97316"])
    axes[0].set_title("Final-state RMSE per value")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(names, l2, color=["#2563eb", "#f97316"])
    axes[1].set_title("Final-state L2 mean")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_component_plot(summary: dict, path: Path):
    components = list(summary["learned"]["components"])
    learned = [summary["learned"]["components"][key]["mean"] for key in components]
    zero = [summary["zero_delta"]["components"][key]["mean"] for key in components]
    x = np.arange(len(components))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(components) * 1.2), 5))
    ax.bar(x - width / 2, learned, width, label="learned", color="#2563eb")
    ax.bar(x + width / 2, zero, width, label="zero_delta", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=35, ha="right")
    ax.set_ylabel("Final-state component mean error")
    ax.set_title("Validation final-state component errors")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_or_run_path", type=Path)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument(
        "--rollout-mode",
        choices=["teacher_forced_final", "open_loop"],
        default="teacher_forced_final",
    )
    parser.add_argument("--project-rot6d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cdf-batch-size", type=int, default=65536)
    parser.add_argument("--skip-cdf", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path.cwd())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, checkpoint = load_dynamics_model_for_eval(
        args.model_or_run_path, device=device, checkpoint_name=args.checkpoint_name
    )
    checkpoint_path = args.model_or_run_path / args.checkpoint_name if args.model_or_run_path.is_dir() else args.model_or_run_path
    run_dir = checkpoint_path.parent
    output_dir = args.output_dir or (run_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((run_dir / "data_provenance.json").read_text())
    dataset_path = Path(provenance.get("dataset", checkpoint["training_config"]["dataset"]))
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    state_keys = checkpoint.get("training_config", {}).get(
        "state_keys", provenance.get("state_keys")
    )
    if state_keys is None:
        state_keys = checkpoint.get("training_config", {}).get(
            "state_components", ["eef_pos", "eef_rot6d", "gripper_width", "cheezit_pos", "cheezit_rot6d"]
        )
    state_keys = list(state_keys)
    action_key = checkpoint.get("training_config", {}).get("action_key", "actions")
    slices = state_slices(state_keys)

    if args.split == "train":
        demo_refs = provenance["train_demos"]
    elif args.split == "val":
        demo_refs = provenance["val_demos"]
    else:
        demo_refs = provenance["train_demos"] + provenance["val_demos"]

    demo_keys = [demo["demo_id"] for demo in demo_refs]
    trajectories = build_dynamics_trajectories(dataset_path, demo_keys, state_keys, action_key)
    flat = flatten_trajectories(trajectories)

    final_rows = []
    for traj in trajectories:
        if args.rollout_mode == "teacher_forced_final":
            learned_state = predict_next_state(
                traj["states"][-1],
                traj["actions"][-1],
                model,
                stats,
                device,
                project_rot6d=args.project_rot6d,
                slices=slices,
            )
            zero_state = traj["states"][-1].astype(np.float32)
        else:
            learned_state = traj["states"][0].copy()
            if args.project_rot6d:
                learned_state = project_state_rot6d(learned_state, slices)
            for action in traj["actions"]:
                learned_state = predict_next_state(
                    learned_state,
                    action,
                    model,
                    stats,
                    device,
                    project_rot6d=args.project_rot6d,
                    slices=slices,
                )
            zero_state = traj["states"][0].astype(np.float32)

        final_rows.append(
            {
                "demo_id": traj["demo_id"],
                "num_steps": int(len(traj["actions"])),
                "true_final": traj["next_states"][-1].astype(np.float32),
                "learned_final": learned_state.astype(np.float32),
                "zero_final": zero_state,
            }
        )

    finite = finite_status(final_rows)
    summary = summarize_final_errors(final_rows, state_keys)
    improvement = 1.0 - summary["learned"]["rmse_per_value"] ** 2 / max(
        summary["zero_delta"]["rmse_per_value"] ** 2, 1e-12
    )

    prefix = args.checkpoint_name.replace(".pt", "")
    report_path = output_dir / f"{prefix}_{args.split}_{args.rollout_mode}_final_state_report.json"
    summary_plot_path = output_dir / f"{prefix}_{args.split}_{args.rollout_mode}_final_state_summary.png"
    component_plot_path = output_dir / f"{prefix}_{args.split}_{args.rollout_mode}_final_state_components.png"
    save_summary_plot(summary, summary_plot_path)
    save_component_plot(summary, component_plot_path)

    cdf_plot_path = None
    cdf_summary_path = None
    cdf_summary = None
    if not args.skip_cdf:
        one_step_errors, descriptors = one_step_component_errors(
            flat,
            model,
            stats,
            device,
            batch_size=args.cdf_batch_size,
            state_keys=state_keys,
            project_rot6d=args.project_rot6d,
        )
        cdf_plot_path, cdf_summary_path, cdf_summary = save_component_cdf_plots(
            one_step_errors,
            descriptors,
            output_dir,
            f"{prefix}_{args.split}_one_step",
        )

    report = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": None if checkpoint.get("val_loss") is None else float(checkpoint["val_loss"]),
        "dataset": str(dataset_path),
        "split": args.split,
        "state_keys": state_keys,
        "project_rot6d": bool(args.project_rot6d),
        "rollout_mode": args.rollout_mode,
        "finite_status": finite,
        "num_demos": int(len(trajectories)),
        "num_transitions": int(flat["states"].shape[0]),
        "mse_improvement_over_zero_delta": float(improvement),
        "summary": summary,
        "cdf_summary": cdf_summary,
        "plots": {
            "summary": str(summary_plot_path),
            "components": str(component_plot_path),
            "component_cdfs": None if cdf_plot_path is None else str(cdf_plot_path),
        },
        "final_rows": [{"demo_id": row["demo_id"], "num_steps": row["num_steps"]} for row in final_rows],
    }
    report_path.write_text(json.dumps(report, indent=2))

    np.savez_compressed(
        output_dir / f"{prefix}_{args.split}_{args.rollout_mode}_final_states.npz",
        true_final=np.stack([row["true_final"] for row in final_rows], axis=0),
        learned_final=np.stack([row["learned_final"] for row in final_rows], axis=0),
        zero_final=np.stack([row["zero_final"] for row in final_rows], axis=0),
        demo_ids=np.asarray([row["demo_id"] for row in final_rows]),
    )

    print(f"run_dir: {run_dir}")
    print(f"split: {args.split}")
    print(f"state_keys: {state_keys}")
    print(f"project_rot6d: {args.project_rot6d}")
    print(f"rollout_mode: {args.rollout_mode}")
    print(f"learned final RMSE/value: {summary['learned']['rmse_per_value']:.6f}")
    print(f"zero final RMSE/value:    {summary['zero_delta']['rmse_per_value']:.6f}")
    print(f"MSE improvement over zero final-state baseline: {100.0 * improvement:.2f}%")
    print(f"report: {report_path}")
    print(f"summary plot: {summary_plot_path}")
    print(f"component plot: {component_plot_path}")
    if cdf_plot_path is not None:
        print(f"component CDF plot: {cdf_plot_path}")
        print(f"component CDF summary: {cdf_summary_path}")


if __name__ == "__main__":
    main()
