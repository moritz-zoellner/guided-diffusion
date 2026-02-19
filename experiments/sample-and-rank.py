import argparse
import json
import os
from datetime import datetime
import time
import imageio
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import env_utils as EnvUtils
from robomimic.utils import torch_utils as TorchUtils

from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def get_metrics(env, body_name='Can_main'): #'payload_root' for transport
    sim = env.env.env.sim
    bid = sim.model.body_name2id(body_name)
    return sim.data.body_xmat[bid].reshape(3,3)[2,2], sim.data.body_xpos[bid].copy()

def run_rejection_sampling(actions_list, env, eval_env, goal=1):

    scores = np.zeros(len(actions_list))
    state0 = env.get_state()
    for i, actions in enumerate(actions_list):
        eval_env.reset_to(state0)
        chunk = actions[0,1:]
        vals = []
        for t in range(chunk.shape[0]):
            _obs, _r, done, _info = eval_env.step(chunk[t].cpu().numpy())
            rzz, _ = get_metrics(eval_env)
            vals.append(rzz)
            if done:
                vals.append(1) # final sequence likes goal state but Rzz is as important 
                break
        scores[i] = np.min(vals)
    
    return np.argmin(np.abs(scores - goal)), scores

def rollout(policy, env, eval_env, horizon, num_samples=8, goal=1, video_writer=None, video_skip=5, camera_names=['agentview'], render=False, init_state=None, use_rejection_sampling=True):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.
    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
    """

    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    
    env.reset()
    if init_state is not None:
        obs = env.reset_to(init_state)
    else:
        state_dict = env.get_state()
        obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.

    Rzz = []
    Pos = []

    try:
        for step_i in tqdm(range(horizon)):

            if use_rejection_sampling:
                obs_torch = policy._prepare_observation(obs)

                # INJECTION OF REJECTION SAMPLING
                recompute_interval = 8
                if step_i % recompute_interval == 0:
                    actions_list = list()
                    sample_start_time = time.perf_counter()
                    for i in range(num_samples):
                        with torch.no_grad():
                            actions = policy.policy.get_full_action(obs_torch)
                            actions_list.append(actions)
                    sample_end_time = time.perf_counter()
                    print(
                        f"sample_actions_time_s={sample_end_time - sample_start_time:.4f} "
                        f"(num_samples={num_samples})"
                    )
                    rs_start_time = time.perf_counter()
                    selection, scores_list = run_rejection_sampling(actions_list, env, eval_env, goal=goal)
                    rs_end_time = time.perf_counter()
                    print(f"run_rejection_sampling_time_s={rs_end_time - rs_start_time:.4f}")
                    #selection, scores_list = 0, []
                    selected_chunk = actions_list[selection][0].detach().cpu() # this takes the batch away and turns it into np
                    policy.policy.set_full_action(selected_chunk[1:9]) # forcing the policy to adopt this action, but only Ta, not the Tp 
            
            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)
            rzz, pos = get_metrics(env)
            Rzz.append(rzz)
            Pos.append(pos.tolist())
            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(
        Return=total_reward, 
        Horizon=(step_i + 1), 
        Success_Rate=float(success),
        Rzz_List=Rzz,
        Pos_List=Pos
    )

    return stats

def run_diffusion(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    run_id = args.name if args.name is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_path = os.path.join(args.output_path, run_id)
    stats_path = os.path.join(run_output_path, "rollout_stats.json")
    os.makedirs(run_output_path, exist_ok=True)

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_path=args.ckpt_path,
        render=(args.record_video == "y"),
        render_offscreen=(args.record_video == "y"),
    )
    eval_env, _ = FileUtils.env_from_checkpoint(
        ckpt_path=args.ckpt_path,
        render=False,
        render_offscreen=False,
    )

    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt_path,
        device=device,
        verbose=True,
    )
    video_dir = None
    if args.record_video == "y":
        video_dir = os.path.join(run_output_path, "video")
        os.makedirs(video_dir, exist_ok=True)

    all_stats = []
    print("Starting Rollouts")
    init_state = None
    if args.fix_init_state == "y":
        env.reset()
        init_state = env.get_state()
    baseline_cutoff = args.n_rollouts // 2
    for rollout_i in range(args.n_rollouts):
        rollout_num = rollout_i + 1
        use_rejection_sampling = True
        if args.get_baseline == "y" and rollout_i < baseline_cutoff:
            use_rejection_sampling = False
        video_writer = None
        make_video = (
            args.record_video == "y"
            and rollout_num % args.n_step_rollout_video == 0
        )
        if make_video:
            video_path = os.path.join(video_dir, f"rollout_{rollout_num}.mp4")
            video_writer = imageio.get_writer(video_path, fps=20)

        rollout_start_time = time.perf_counter()
        stats = rollout(
            policy=policy,
            env=env,
            eval_env=eval_env,
            num_samples=args.num_samples,
            goal=args.goal,
            horizon=args.horizon,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
            init_state=init_state,
            use_rejection_sampling=use_rejection_sampling,
        )
        rollout_end_time = time.perf_counter()
        stats["Mode"] = "baseline" if not use_rejection_sampling else "sample_and_rank"
        all_stats.append(stats)
        # Persist stats after each rollout so partial results are available.
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)
        if video_writer is not None:
            video_writer.close()
        rollout_seconds = rollout_end_time - rollout_start_time
        print(
            f"[rollout {rollout_num}/{args.n_rollouts}] "
            f"time_s={rollout_seconds:.2f} "
            f"mode={stats['Mode']} "
            f"video_made={make_video} "
            f"stats: { {key: (len(value) if key in ['Rzz_List', 'Pos_List'] else value) for key, value in stats.items()} }"
        )



    if video_dir is not None:
        print(
            f"Completed {args.n_rollouts} rollouts. Outputs: stats={stats_path}, "
            f"videos={video_dir}, run_dir={run_output_path}"
        )
    else:
        print(
            f"Completed {args.n_rollouts} rollouts. Outputs: stats={stats_path}, run_dir={run_output_path}"
        )


CKPT_PATH = "./models/model_epoch_1100_low_dim_v15_success_0.7.pth"  # <-- change
OUTPUT_PATH = "./outputs/sr_rollout"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--goal", type=int, default=1)
    parser.add_argument("--Ta", type=int, default=8)
    parser.add_argument("--T_eval", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--n_step_rollout_video", type=int, default=1)
    parser.add_argument("--record_video", type=str, choices=["y", "n"], default="y")
    parser.add_argument("--video_skip", type=int, default=1)
    parser.add_argument("--camera_names", nargs="+", default=["frontview"])
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--fix_init_state", type=str, choices=["y", "n"], default="n")
    parser.add_argument("--get_baseline", type=str, choices=["y", "n"], default="n")
    args = parser.parse_args()
    run_diffusion(args)


if __name__ == "__main__":
    main()
