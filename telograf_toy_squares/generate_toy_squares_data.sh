#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/mzoellner/Projects/research/guided-diffusion}
PYTHON_BIN=${PYTHON_BIN:-/home/mzoellner/miniconda3/envs/guided-diffusion/bin/python}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/telograf/toy_squares/raw/full_10k}
N_ROLLOUTS=${N_ROLLOUTS:-10000}
HORIZON=${HORIZON:-150}
POLICY_NOISE_SCALE=${POLICY_NOISE_SCALE:-0.03}
POLICY_MAX_WAYPOINTS=${POLICY_MAX_WAYPOINTS:-1}
VAL_RATIO=${VAL_RATIO:-0.1}

mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

"${PYTHON_BIN}" -u "${REPO_ROOT}/toy_squares/collect_scripted_data_pymunk.py" \
  --dataset_path "${OUT_DIR}/data.hdf5" \
  --dataset_obs \
  --dataset_obs_keys agent_pos states \
  --json_path "${OUT_DIR}/stats.json" \
  --horizon "${HORIZON}" \
  --n_rollouts "${N_ROLLOUTS}" \
  --policy_noise_scale "${POLICY_NOISE_SCALE}" \
  --policy_max_waypoints "${POLICY_MAX_WAYPOINTS}" \
  --env_config "${REPO_ROOT}/toy_squares/touchcubes.json" \
  --output_folder "${OUT_DIR}" \
  --keep_only_successful \
  --repeat_environment \
  --seed 7

"${PYTHON_BIN}" -u "${REPO_ROOT}/robomimic/robomimic/scripts/split_train_val.py" \
  --dataset "${OUT_DIR}/data.hdf5" \
  --ratio "${VAL_RATIO}"

echo "Toy Squares dataset created at ${OUT_DIR}/data.hdf5"
