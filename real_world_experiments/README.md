# Real-World Cheez-It Pouring Experiments

This folder mirrors the CALVIN/robomimic pipeline for the real robot setup.

Expected converted HDF5 layout:

```text
data/demo_i/actions
data/demo_i/obs/eef_pos
data/demo_i/obs/eef_rot6d
data/demo_i/obs/gripper_width
data/demo_i/obs/gripper_binary
data/demo_i/obs/cheezit_pos
data/demo_i/obs/cheezit_rot6d
data/demo_i/obs/robot0_eef_pos
data/demo_i/obs/robot0_eef_quat
data/demo_i/obs/robot0_gripper_qpos
data/demo_i/obs/object
mask/train, mask/valid
```

The robomimic keys use `xyzw` quaternions, matching robosuite. The `*_wxyz`
aliases preserve the source ROS data order. If a future converter stores
`*_quat` instead of `*_rot6d`, the trainers can convert on read when the
requested state key ends in `_rot6d`; those fallback quaternions are assumed to
be `wxyz`.
kay

For the current ROS pickle episodes, `actions` is synthesized from consecutive
measured EEF states:

```text
[dx, dy, dz, drot_x, drot_y, drot_z, next_gripper_binary]
```

The rotation delta is a world-frame rotation vector with convention
`q_delta = q_next * conjugate(q_current)`, so the controller can reconstruct
`q_next = q_delta * q_current`. The final gripper dimension is the next binary
gripper target using a dataset-global midpoint (`-1` below midpoint, `+1`
above). The original logged 3D position-only command is preserved as
`data/demo_i/actions_logged_xyz` for diagnostics.

The diffusion-policy config uses `gripper_binary` as its low-dimensional
gripper observation. `gripper_width` remains the raw gripper reading for label
thresholding and later world-model experiments.

Convert the collected ROS pickle episodes without modifying the originals:

```bash
python real_world_experiments/real_world_data.py \
  --output data/real_world/cheezit_pouring.hdf5
```

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
