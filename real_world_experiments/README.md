# Real-World Cheez-It Pouring Experiments

This folder mirrors the CALVIN/robomimic pipeline for the real robot setup.

Expected converted HDF5 layout:

```text
data/demo_i/actions
data/demo_i/obs/eef_pos
data/demo_i/obs/eef_rot6d
data/demo_i/obs/gripper_width
data/demo_i/obs/cheezit_pos
data/demo_i/obs/cheezit_rot6d
mask/train, mask/valid
```

If the converter stores `*_quat` instead of `*_rot6d`, the trainers can convert
on read when the requested state key ends in `_rot6d`. Quaternions are assumed
to be `wxyz`.

Train low-level dynamics:

```bash
python real_world_experiments/train_dynamics_world_model.py \
  --dataset data/real_world/cheezit_pouring.hdf5
```

Train high-level automaton model:

```bash
python real_world_experiments/train_automaton_world_model.py \
  --dataset data/real_world/cheezit_pouring.hdf5 \
  --label-config real_world_experiments/label_config_example.json
```

Current labels are geometric placeholders:

```text
pour_left_bowl  = cheezit over left bowl  and Rzz below threshold
pour_right_bowl = cheezit over right bowl and Rzz below threshold
cheezits_released = gripper open and cheezit near table height
```

The bowl centers, radius, Rzz thresholds, and release thresholds should be
updated after inspecting the first converted trajectories.

