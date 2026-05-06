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
        build_calvin_dynamics_trajectories,
        flatten_trajectories,
        load_dynamics_model_for_eval,
    )
except ModuleNotFoundError:
    from calvin_experiments.train_dynamics_world_model import (
        build_calvin_dynamics_trajectories,
        flatten_trajectories,
        load_dynamics_model_for_eval,
    )


ROT6D_SLICES = (slice(3, 9), slice(27, 33), slice(36, 42), slice(45, 51))


def resolve_repo_root(start):
    start = Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "calvin_experiments" / "train_dynamics_world_model.py").exists():
            return path
    raise FileNotFoundError(f"Could not resolve repo root from {start}")


def normalize_rot6d(values):
    values = np.asarray(values, dtype=np.float32).copy()
    r1 = values[..., 0:3]
    r2 = values[..., 3:6]
    r1 = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)
    r2 = r2 - np.sum(r1 * r2, axis=-1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-8)
    values[..., 0:3] = r1
    values[..., 3:6] = r2
    return values


def project_state_rot6d(state):
    state = np.asarray(state, dtype=np.float32).copy()
    if state.shape[-1] != 51:
        return state
    for rot_slice in ROT6D_SLICES:
        state[..., rot_slice] = normalize_rot6d(state[..., rot_slice])
    return state


def rot6d_to_matrix(values):
    values = normalize_rot6d(values)
    r1 = values[..., 0:3]
    r2 = values[..., 3:6]
    r3 = np.cross(r1, r2, axis=-1)
    return np.stack([r1, r2, r3], axis=-1)


def rot6d_geodesic_deg(a, b):
    ra = rot6d_to_matrix(a)
    rb = rot6d_to_matrix(b)
    r_delta = np.matmul(np.swapaxes(ra, -1, -2), rb)
    trace = np.trace(r_delta, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)).astype(np.float32)


def predict_next_state(state, action, model, stats, device, project_rot6d=False):
    state_n = ((state - stats["state_mean"]) / stats["state_std"]).astype(np.float32)
    action_n = ((action - stats["action_mean"]) / stats["action_std"]).astype(np.float32)
    with torch.no_grad():
        delta_n = model(
            torch.from_numpy(state_n).to(device).unsqueeze(0),
            torch.from_numpy(action_n).to(device).unsqueeze(0),
        )[0].cpu().numpy()
    delta = delta_n * stats["delta_std"] + stats["delta_mean"]
    next_state = state + delta
    if project_rot6d:
        next_state = project_state_rot6d(next_state)
    return next_state.astype(np.float32)


def predict_next_states_batched(states, actions, model, stats, device, batch_size, project_rot6d=False):
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
                next_state = project_state_rot6d(next_state)
            preds.append(next_state.astype(np.float32))
    return np.concatenate(preds, axis=0)


def component_slices(state_dim):
    if state_dim == 51:
        return {
            "tcp_pos": slice(0, 3),
            "tcp_rot6d": slice(3, 9),
            "proprio_rest": slice(9, 18),
            "scene_scalars": slice(18, 24),
            "block_red_pos": slice(24, 27),
            "block_red_rot6d": slice(27, 33),
            "block_blue_pos": slice(33, 36),
            "block_blue_rot6d": slice(36, 42),
            "block_pink_pos": slice(42, 45),
            "block_pink_rot6d": slice(45, 51),
        }
    if state_dim == 39:
        return {
            "tcp_pos": slice(0, 3),
            "tcp_euler": slice(3, 6),
            "proprio_rest": slice(6, 15),
            "scene_scalars": slice(15, 21),
            "block_red_pos": slice(21, 24),
            "block_red_euler": slice(24, 27),
            "block_blue_pos": slice(27, 30),
            "block_blue_euler": slice(30, 33),
            "block_pink_pos": slice(33, 36),
            "block_pink_euler": slice(36, 39),
        }
    return {"full_state": slice(0, state_dim)}


def metric_components(state_dim):
    if state_dim == 51:
        return {
            "tcp_pos": ("dist", slice(0, 3), "position distance"),
            "tcp_rot": ("rot_deg", slice(3, 9), "rotation error (deg)"),
            "gripper_width": ("abs", slice(9, 10), "absolute scalar error"),
            "arm_joints": ("l2", slice(10, 17), "joint-space L2 error"),
            "gripper_action": ("abs", slice(17, 18), "absolute scalar error"),
            "scene_scalars": ("l2", slice(18, 24), "scene-scalar L2 error"),
            "block_red_pos": ("dist", slice(24, 27), "position distance"),
            "block_red_rot": ("rot_deg", slice(27, 33), "rotation error (deg)"),
            "block_blue_pos": ("dist", slice(33, 36), "position distance"),
            "block_blue_rot": ("rot_deg", slice(36, 42), "rotation error (deg)"),
            "block_pink_pos": ("dist", slice(42, 45), "position distance"),
            "block_pink_rot": ("rot_deg", slice(45, 51), "rotation error (deg)"),
        }
    if state_dim == 39:
        return {
            "tcp_pos": ("dist", slice(0, 3), "position distance"),
            "tcp_euler": ("l2", slice(3, 6), "Euler L2 error"),
            "gripper_width": ("abs", slice(6, 7), "absolute scalar error"),
            "arm_joints": ("l2", slice(7, 14), "joint-space L2 error"),
            "gripper_action": ("abs", slice(14, 15), "absolute scalar error"),
            "scene_scalars": ("l2", slice(15, 21), "scene-scalar L2 error"),
            "block_red_pos": ("dist", slice(21, 24), "position distance"),
            "block_red_euler": ("l2", slice(24, 27), "Euler L2 error"),
            "block_blue_pos": ("dist", slice(27, 30), "position distance"),
            "block_blue_euler": ("l2", slice(30, 33), "Euler L2 error"),
            "block_pink_pos": ("dist", slice(33, 36), "position distance"),
            "block_pink_euler": ("l2", slice(36, 39), "Euler L2 error"),
        }
    return {"full_state": ("l2", slice(0, state_dim), "state L2 error")}


def component_error(true_next, pred_next, metric, slc):
    true_comp = true_next[:, slc]
    pred_comp = pred_next[:, slc]
    if metric in {"dist", "l2"}:
        return np.linalg.norm(pred_comp - true_comp, axis=-1).astype(np.float32)
    if metric == "abs":
        return np.abs(pred_comp - true_comp).reshape(len(true_next), -1).mean(axis=-1).astype(np.float32)
    if metric == "rot_deg":
        return rot6d_geodesic_deg(pred_comp, true_comp)
    raise ValueError(f"Unknown component metric: {metric}")


def one_step_component_errors(flat, model, stats, device, batch_size, project_rot6d=False):
    pred_next = predict_next_states_batched(
        flat["states"], flat["actions"], model, stats, device, batch_size, project_rot6d=project_rot6d
    )
    zero_next = flat["states"]
    true_next = flat["next_states"]

    errors = {"learned": {}, "zero_delta": {}}
    descriptors = {}
    for name, (metric, slc, description) in metric_components(flat["states"].shape[-1]).items():
        errors["learned"][name] = component_error(true_next, pred_next, metric, slc)
        errors["zero_delta"][name] = component_error(true_next, zero_next, metric, slc)
        descriptors[name] = {"metric": metric, "description": description}
    return errors, descriptors


def summarize_error_values(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def save_component_cdf_plots(errors, descriptors, output_dir, prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for component in descriptors:
        summary[component] = {
            model_name: summarize_error_values(errors[model_name][component])
            for model_name in ("learned", "zero_delta")
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
    return plot_path, summary_path


def summarize_final_errors(final_rows, state_dim):
    learned_errors = np.stack([row["learned_final"] - row["true_final"] for row in final_rows], axis=0)
    zero_errors = np.stack([row["zero_final"] - row["true_final"] for row in final_rows], axis=0)

    summary = {}
    for name, errors in [("learned", learned_errors), ("zero_delta", zero_errors)]:
        summary[name] = {
            "rmse_per_value": float(np.sqrt(np.mean(np.square(errors)))),
            "mae_per_value": float(np.mean(np.abs(errors))),
            "l2_mean": float(np.mean(np.linalg.norm(errors, axis=-1))),
            "l2_median": float(np.median(np.linalg.norm(errors, axis=-1))),
            "l2_max": float(np.max(np.linalg.norm(errors, axis=-1))),
            "components": {},
        }
        for key, slc in component_slices(state_dim).items():
            comp = errors[:, slc]
            summary[name]["components"][key] = {
                "rmse_per_value": float(np.sqrt(np.mean(np.square(comp)))),
                "mae_per_value": float(np.mean(np.abs(comp))),
            }
    return summary


def finite_status(final_rows):
    statuses = {}
    for key in ("true_final", "learned_final", "zero_final"):
        values = np.stack([row[key] for row in final_rows], axis=0)
        statuses[key] = {
            "all_finite": bool(np.isfinite(values).all()),
            "finite_fraction": float(np.isfinite(values).mean()),
            "max_abs_finite": float(np.max(np.abs(values[np.isfinite(values)]))) if np.isfinite(values).any() else None,
        }
    return statuses


def save_summary_plot(summary, path):
    names = ["learned", "zero_delta"]
    rmse = [summary[name]["rmse_per_value"] for name in names]
    l2 = [summary[name]["l2_mean"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rmse, color=["#2563eb", "#f97316"])
    axes[0].set_title("Final-state RMSE per value")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(names, l2, color=["#2563eb", "#f97316"])
    axes[1].set_title("Final-state L2 mean")
    axes[1].set_ylabel("L2")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_component_plot(summary, path):
    components = list(summary["learned"]["components"].keys())
    learned = [summary["learned"]["components"][key]["rmse_per_value"] for key in components]
    zero = [summary["zero_delta"]["components"][key]["rmse_per_value"] for key in components]
    x = np.arange(len(components))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(components) * 1.1), 5))
    ax.bar(x - width / 2, learned, width, label="learned", color="#2563eb")
    ax.bar(x + width / 2, zero, width, label="zero_delta", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=35, ha="right")
    ax.set_ylabel("Final-state RMSE per value")
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
        help=(
            "teacher_forced_final predicts only the final held-out transition in each demo from true state/action; "
            "open_loop rolls from the demo initial state through the full action sequence."
        ),
    )
    parser.add_argument("--project-rot6d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cdf-batch-size", type=int, default=65536)
    parser.add_argument("--skip-cdf", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path.cwd())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, checkpoint, meta = load_dynamics_model_for_eval(
        args.model_or_run_path, device=device, checkpoint_name=args.checkpoint_name
    )
    run_dir = Path(meta["run_dir"])
    output_dir = args.output_dir or (run_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((run_dir / "data_provenance.json").read_text())
    dataset_path = Path(provenance["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    state_representation = checkpoint.get("training_config", {}).get(
        "state_representation", provenance.get("state_representation", "raw")
    )

    if args.split == "train":
        demo_refs = provenance["train_demos"]
    elif args.split == "val":
        demo_refs = provenance["val_demos"]
    else:
        demo_refs = provenance["train_demos"] + provenance["val_demos"]

    demo_keys = [demo["demo_id"] for demo in demo_refs]
    trajectories = build_calvin_dynamics_trajectories(
        dataset_path, demo_keys, state_representation=state_representation
    )
    flat = flatten_trajectories(trajectories)
    state_dim = int(flat["states"].shape[-1])
    project_rot6d = bool(args.project_rot6d and state_representation == "rot6d")

    final_rows = []
    for traj in trajectories:
        if args.rollout_mode == "teacher_forced_final":
            learned_state = predict_next_state(
                traj["states"][-1], traj["actions"][-1], model, stats, device, project_rot6d=project_rot6d
            )
            zero_state = traj["states"][-1].astype(np.float32)
        else:
            learned_state = traj["states"][0].copy()
            if project_rot6d:
                learned_state = project_state_rot6d(learned_state)
            for action in traj["actions"]:
                learned_state = predict_next_state(
                    learned_state, action, model, stats, device, project_rot6d=project_rot6d
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
    summary = summarize_final_errors(final_rows, state_dim)
    improvement = 1.0 - summary["learned"]["rmse_per_value"] ** 2 / max(
        summary["zero_delta"]["rmse_per_value"] ** 2, 1e-12
    )
    report = {
        "run_dir": str(run_dir),
        "checkpoint_path": meta["checkpoint_path"],
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": None if checkpoint.get("val_loss") is None else float(checkpoint["val_loss"]),
        "dataset": str(dataset_path),
        "split": args.split,
        "state_representation": state_representation,
        "project_rot6d": project_rot6d,
        "rollout_mode": args.rollout_mode,
        "finite_status": finite,
        "num_demos": int(len(trajectories)),
        "num_transitions": int(flat["states"].shape[0]),
        "mse_improvement_over_zero_delta": float(improvement),
        "summary": summary,
        "final_rows": [
            {"demo_id": row["demo_id"], "num_steps": row["num_steps"]} for row in final_rows
        ],
    }

    report_path = output_dir / f"{args.checkpoint_name.replace('.pt', '')}_{args.split}_final_state_report.json"
    summary_plot_path = output_dir / f"{args.checkpoint_name.replace('.pt', '')}_{args.split}_final_state_summary.png"
    component_plot_path = output_dir / f"{args.checkpoint_name.replace('.pt', '')}_{args.split}_final_state_components.png"
    report_path.write_text(json.dumps(report, indent=2))
    save_summary_plot(summary, summary_plot_path)
    save_component_plot(summary, component_plot_path)

    cdf_plot_path = None
    cdf_summary_path = None
    if not args.skip_cdf:
        one_step_errors, descriptors = one_step_component_errors(
            flat,
            model,
            stats,
            device,
            batch_size=args.cdf_batch_size,
            project_rot6d=project_rot6d,
        )
        cdf_plot_path, cdf_summary_path = save_component_cdf_plots(
            one_step_errors,
            descriptors,
            output_dir,
            f"{args.checkpoint_name.replace('.pt', '')}_{args.split}_one_step",
        )

    np.savez_compressed(
        output_dir / f"{args.checkpoint_name.replace('.pt', '')}_{args.split}_final_states.npz",
        true_final=np.stack([row["true_final"] for row in final_rows], axis=0),
        learned_final=np.stack([row["learned_final"] for row in final_rows], axis=0),
        zero_final=np.stack([row["zero_final"] for row in final_rows], axis=0),
        demo_ids=np.asarray([row["demo_id"] for row in final_rows]),
    )

    print(f"run_dir: {run_dir}")
    print(f"split: {args.split}")
    print(f"state_representation: {state_representation}")
    print(f"project_rot6d: {project_rot6d}")
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
