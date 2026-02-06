import os
import numpy as np
import torch
import imageio
import json

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import env_utils as EnvUtils

from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from copy import deepcopy

CKPT_PATH = "./models/model_epoch_1100_low_dim_v15_success_0.7.pth"  # <-- change
OUT_DIR = "video"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load checkpoint dict (robomimic-style)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg = ckpt["config"]
cfg = json.loads(cfg)
cfg["observation"]["modalities"]

# Important: init obs modalities exactly as training expected
ObsUtils.initialize_obs_utils_with_obs_specs(cfg["observation"]["modalities"])

env, _ = FileUtils.env_from_checkpoint(
    ckpt_path=CKPT_PATH,
    render=True,
    render_offscreen=True,
)

env.reset()


while True:
    a = np.zeros(14)
    a[4] = 0.2

    a[12] = 0.05
    obs, *_ = env.step(a)

    env.render(mode="human", camera_name='agentview')