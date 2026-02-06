import argparse
import json
import os
import imageio
import numpy as np
import torch

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import env_utils as EnvUtils
from robomimic.utils import torch_utils as TorchUtils

from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def rollout(policy, env, horizon, video_writer=None, video_skip=5, camera_names=['agentview'], render=False):
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
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    
    payload_pos_seq = np.zeros((horizon,3), dtype=np.float32)
    payload_body_seq = np.zeros((horizon,3,3), dtype=np.float32)

    try:
        for step_i in range(horizon):

            print(step_i)
            print(obs.keys())
            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            sim = env.env.env.sim            
            R = sim.data.body_xmat[bid].reshape(3,3)
            payload_pos_seq[step_i] = sim.data.body_xpos[bid]
            payload_body_seq[step_i] = R


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

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success), Pos_Seq=payload_pos_seq, Upz_Seq=payload_body_seq)

    return stats


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def run_diffusion(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    os.makedirs(args.output_path, exist_ok=True)

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_path=args.ckpt_path,
        render=(args.record_video == "y"),
        render_offscreen=(args.record_video == "y"),
    )

    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt_path,
        device=device,
        verbose=True,
    )

    video_writer = None
    if args.record_video == "y":
        video_dir = os.path.join(args.output_path, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, args.video_name)
        video_writer = imageio.get_writer(video_path, fps=20)

    all_stats = []
    all_trajs = []
    for _ in range(args.n_rollouts):
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=args.horizon,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
        )
        all_stats.append(stats)
        all_trajs.append(traj)

    if video_writer is not None:
        video_writer.close()

    stats_path = os.path.join(args.output_path, "rollout_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(all_stats), f, indent=2)


CKPT_PATH = "./models/model_epoch_1100_low_dim_v15_success_0.7.pth"  # <-- change
OUTPUT_PATH = "./outputs/sr_rollout"
VIDEO_SKIP = 30
CAMERA_NAMES = ["agentview"]
VIDEO_NAME = "rollout.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--record_video", type=str, choices=["y", "n"], default="y")
    parser.add_argument("--video_skip", type=int, default=VIDEO_SKIP)
    parser.add_argument("--video_name", type=str, default=VIDEO_NAME)
    parser.add_argument("--camera_names", nargs="+", default=CAMERA_NAMES)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    args = parser.parse_args()
    run_diffusion(args)


if __name__ == "__main__":
    main()
