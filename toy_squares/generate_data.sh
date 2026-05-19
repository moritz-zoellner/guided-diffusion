#!/usr/bin/env bash
set -euo pipefail

# LAPTOP
# PYTHON_BIN=${PYTHON_BIN:-/home/mzoellner/miniconda3/envs/guided-diffusion/bin/python}
# REPO_ROOT=${REPO_ROOT:-/home/mzoellner/Projects/research/guided-diffusion}

# GILBRETH
#PYTHON_BIN=${PYTHON_BIN:-/home/zoellner/.conda/envs/guided_diffusion/bin/python}
#REPO_ROOT=${REPO_ROOT:-/home/zoellner/src/guided-diffusion}
#TRAIN_OUT=${TRAIN_OUT:-$REPO_ROOT/data/toy_squares/train}

# CORALLAB
PYTHON_BIN=${PYTHON_BIN:-/home/moritz/src/guided-diffusion/.pixi/envs/default/bin/python}
REPO_ROOT=${REPO_ROOT:-/home/moritz/src/guided-diffusion}
TRAIN_OUT=${TRAIN_OUT:-/home/shared/data/toy_squares/train}
#TRAIN_OUT=${TRAIN_OUT:-$REPO_ROOT/data/toy_squares/train}


TRAIN_ROLLOUTS=${TRAIN_ROLLOUTS:-10000}
HORIZON=${HORIZON:-150}
TARGET_SEQUENCE_LENGTH=${TARGET_SEQUENCE_LENGTH:-1}
MIN_STEPS_PER_TARGET=${MIN_STEPS_PER_TARGET:-8}
POLICY_NOISE_SCALE=${POLICY_NOISE_SCALE:-0.03}
POLICY_MAX_WAYPOINTS=${POLICY_MAX_WAYPOINTS:-1}
DATASET_OBS_KEYS=${DATASET_OBS_KEYS:-}
VIDEO_PATH=${VIDEO_PATH-$TRAIN_OUT/train_preview.gif}

mkdir -p "$TRAIN_OUT"

EXTRA_ARGS=()
if [[ -n "$DATASET_OBS_KEYS" ]]; then
  EXTRA_ARGS+=(--dataset_obs_keys "$DATASET_OBS_KEYS")
fi
if [[ -n "$VIDEO_PATH" ]]; then
  EXTRA_ARGS+=(--video_path "$VIDEO_PATH")
fi

"$PYTHON_BIN" -u "$REPO_ROOT/toy_squares/collect_scripted_data_pymunk.py" \
  --dataset_path "$TRAIN_OUT/data.hdf5" \
  --dataset_obs \
  --json_path "$TRAIN_OUT/stats.json" \
  --horizon "$HORIZON" \
  --n_rollouts "$TRAIN_ROLLOUTS" \
  --target_sequence_length "$TARGET_SEQUENCE_LENGTH" \
  --min_steps_per_target "$MIN_STEPS_PER_TARGET" \
  --policy_noise_scale "$POLICY_NOISE_SCALE" \
  --policy_max_waypoints "$POLICY_MAX_WAYPOINTS" \
  --env_config "$REPO_ROOT/toy_squares/touchcubes.json" \
  --output_folder "$TRAIN_OUT" \
  --keep_only_successful \
  --camera_names image \
  --video_skip 5 \
  --repeat_environment \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" -u "$REPO_ROOT/robomimic/robomimic/scripts/split_train_val.py" \
  --dataset "$TRAIN_OUT/data.hdf5" \
  --ratio 0.1

echo "Dataset created at: $TRAIN_OUT/data.hdf5"
