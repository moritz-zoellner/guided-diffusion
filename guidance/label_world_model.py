import h5py
import numpy as np
import os
import random
import torch
import torch.nn as nn
import imageio
import json
import re

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import env_utils as EnvUtils

from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from copy import deepcopy
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from .world_model_utils import (
    DynamicsMLP,
    quat_to_6d,
    xyzw_to_wxyz_batch,
    build_state_from_obs_dict,
    deconstruct_state,
    load_model_for_eval,
    plot_rzz_world_model_branches,
    predict_next_state_from_raw,
    reconstruct_rotation_matrix_from_6d,
    rollout_policy_for_rzz_analysis,
    rzz_from_state_29d,
)

import corallab_stl.torch as stl
from corallab_stl.automata import get_spot_formula_and_aps


def main():
    x_var = stl.Var("x", dim=1)
    y_var = stl.Var("y", dim=1)

    pos_x = stl.Predicate(x_var, lambda x, _: x[0], 0.0)
    pos_y = stl.Predicate(y_var, lambda y, _: y[0], 0.0)

    # phi = stl.UntimedAlways(stl.And(pos_x, pos_y))
    phi = stl.UntimedAlways(pos_x)

    x = torch.linspace(2, -2, 64).unsqueeze(-1)
    y = torch.linspace(0, 4, 64).unsqueeze(-1)

    rho = phi({ "x": x, "y": y })

    spot_form, aps, spot_aps = get_spot_formula_and_aps(phi)

    breakpoint()


if __name__ == "__main__":
    main()
