#!/usr/bin/env bash
set -euo pipefail

# Submit toy_squares data generation first
GEN_JOB_ID=$(sbatch --parsable hpc_jobs/generate_toy_squares_data.sub)
echo "Submitted generate job: ${GEN_JOB_ID}"

# Submit diffusion training only if generation succeeds
TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:${GEN_JOB_ID} hpc_jobs/train_diffusion.sub)
echo "Submitted train job: ${TRAIN_JOB_ID} (afterok:${GEN_JOB_ID})"
