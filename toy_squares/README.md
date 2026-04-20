# Toy Squares (Base DP Only)

This folder contains the minimal setup for:
1. TouchCube toy environment with 4 blocks.
2. Scripted data collection (same style as DynaGuide toy pipeline).
3. Base diffusion policy training config and command.
4. Notebook for base policy evaluation on Early/Late Decision (100 rollouts each).

## Files
- `collect_scripted_data_pymunk.py`: scripted data generation to HDF5.
- `touchcubes.json`: env + observation metadata used by collection script.
- `diffusion_policy_toy_squares.json`: DP training config for robomimic train script.
- `generate_data.sh`: dataset generation + train/valid split helper.
- `train_dp.sh`: base DP training command helper.
- `eval_base_policy_early_late.ipynb`: notebook for 100x evaluation on Early/Late Decision.

## Data Generation
Run (full-scale example, can take time on CPU):

```bash
bash toy_squares/generate_data.sh
```

Environment variables (optional):
- `TRAIN_ROLLOUTS` (default `100000`)
- `HORIZON` (default `150`)
- `TRAIN_OUT` (default `data/toy_squares/train`)

## Base DP Training
Run:

```bash
bash toy_squares/train_dp.sh
```

Equivalent direct command:

```bash
python robomimic/robomimic/scripts/train.py \
  --config toy_squares/diffusion_policy_toy_squares.json \
  --name toy_squares_dp
```

## Slurm Job
A ready-to-run job file is available at:
- `hpc_jobs/train_toy_squares_dp.sub`

Submit with:

```bash
sbatch hpc_jobs/train_toy_squares_dp.sub
```

## Notebook Evaluation
Open:
- `toy_squares/eval_base_policy_early_late.ipynb`

Then:
1. Set `DP_CKPT_PATH` in the checkpoint-loading cell.
2. Run the 100x Early/Late evaluation cell.
3. Run the fixed-action sanity cell to save `toy_squares/sanity_fixed_action.gif`.
