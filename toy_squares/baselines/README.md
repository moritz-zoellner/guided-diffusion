# Toy Squares LTLDoG Baseline Scripts

This folder is organized around the final paper comparison: our automaton-guided
Diffusion Policy versus a paper-faithful full-trajectory LTLDoG baseline.

## Core Files

- `ltldog_toy.py` contains the full-trajectory Diffuser model, LTL/STL
  robustness functions, checkpoint loading, and the training CLI.
- `ltldog_train.py` is the stable loading API. Import `load_ltldog_planner()`
  from here in rollout scripts.
- `paper_horizon_test.py` runs the main horizon-scaling experiment. LTLDoG
  samples one full trajectory and executes the generated action sequence once;
  there is no receding-horizon LTLDoG controller in this script.
- `ltldog_rollout_diagnostics.py` compares LTLDoG imagined satisfaction with
  actual execution, including state prediction drift.
- `generate_base_dp_rollouts.py` records unguided base Diffusion Policy traces
  used by the paper plotting notebook.
- `paperplots.ipynb` loads the final `main_result_LTLDOG` folder and generates
  the paper-facing plots.

## Final Result Folder

The current paper plot notebook points at:

`outputs/toy_squares_rollouts/baseline_ltldog/rollouts/paper_test/main_result_LTLDOG`

The experiment uses the compact deterministic early-decision layout, a 128-step
execution budget, and the `blue -> yellow -> green -> red -> blue` prefix chain.
