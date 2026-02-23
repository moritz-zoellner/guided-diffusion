#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import os 

def get_rzz_from_env(env, body_name: str) -> float:
    sim = env.env.sim
    bid = sim.model.body_name2id(body_name)
    return float(sim.data.body_xmat[bid].reshape(3, 3)[2, 2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--body", type=str, default="Can_main")
    args = parser.parse_args()

    env_meta = FileUtils.get_env_metadata_from_dataset(args.input)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt_path,
        device=device,
        verbose=True,
    )

    env.reset()
    state_template = env.get_state()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.input, "r") as f_in, h5py.File(args.output, "w") as f_out:
        f_out.attrs["source_dataset"] = args.input
        f_out.attrs["body_name"] = args.body
        f_out.attrs["env_args_json"] = f_in["data"].attrs.get("env_args", "")

        demo_ids = list(f_in["data"].keys())
        g_out = f_out.create_group("data")

        for demo_id in tqdm(demo_ids, desc="demos"):
            states = f_in["data"][demo_id]["states"][:]  # (T, 71)
            T = states.shape[0]
            rzz = np.empty((T,), dtype=np.float32)

            for t in tqdm(range(T)):
                sd = state_template.copy()
                sd["states"] = states[t]
                env.reset_to(sd)
                rzz[t] = get_rzz_from_env(env, args.body)

            d_out = g_out.create_group(demo_id)
            d_out.create_dataset("rzz", data=rzz, dtype=np.float32)
            d_out.attrs["T"] = T

    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()