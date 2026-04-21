#!/usr/bin/env bash
set -euo pipefail

# LAPTOP
# PYTHON_BIN=${PYTHON_BIN:-/home/mzoellner/miniconda3/envs/guided-diffusion/bin/python}
# REPO_ROOT=${REPO_ROOT:-/home/mzoellner/Projects/research/guided-diffusion}

# GILBRETH
PYTHON_BIN=${PYTHON_BIN:-/home/zoellner/.conda/envs/guided_diffusion/bin/python}
REPO_ROOT=${REPO_ROOT:-/home/zoellner/src/guided-diffusion}

TRAIN_OUT=${TRAIN_OUT:-$REPO_ROOT/data/toy_squares/train}
TRAIN_ROLLOUTS=${TRAIN_ROLLOUTS:-10}
HORIZON=${HORIZON:-150}

mkdir -p "$TRAIN_OUT"

"$PYTHON_BIN" -u "$REPO_ROOT/toy_squares/collect_scripted_data_pymunk.py" \
  --video_path "$TRAIN_OUT/train_preview.gif" \
  --dataset_path "$TRAIN_OUT/data.hdf5" \
  --dataset_obs \
  --json_path "$TRAIN_OUT/stats.json" \
  --horizon "$HORIZON" \
  --n_rollouts "$TRAIN_ROLLOUTS" \
  --env_config "$REPO_ROOT/toy_squares/touchcubes.json" \
  --output_folder "$TRAIN_OUT" \
  --keep_only_successful \
  --camera_names image \
  --video_skip 5 \
  --repeat_environment

"$PYTHON_BIN" -u "$REPO_ROOT/robomimic/robomimic/scripts/split_train_val.py" \
  --dataset "$TRAIN_OUT/data.hdf5" \
  --ratio 0.1

echo "Dataset created at: $TRAIN_OUT/data.hdf5"
