"""
Create robomimic train / valid masks for a CALVIN HDF5 dataset.

This is a thin wrapper around robomimic's standard split script so the CALVIN
workflow can use the same entry point as DynaGuide.
"""

import argparse

import h5py
import numpy as np


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    with h5py.File(hdf5_path, "a") as f:
        if "mask" not in f:
            f.create_group("mask")
        key_path = f"mask/{key_name}"
        if key_path in f:
            del f[key_path]
        f[key_path] = np.array(demo_keys, dtype="S")
        return [f[f"data/{demo_key}"].attrs["num_samples"] for demo_key in demo_keys]


def split_train_val_from_hdf5(hdf5_path, val_ratio=0.1, filter_key=None):
    with h5py.File(hdf5_path, "r") as f:
        if filter_key is not None:
            print(f"using filter key: {filter_key}")
            demos = sorted([elem.decode("utf-8") for elem in np.array(f[f"mask/{filter_key}"])])
        else:
            demos = sorted(list(f["data"].keys()))

    num_demos = len(demos)
    num_val = int(val_ratio * num_demos)
    mask = np.zeros(num_demos)
    mask[:num_val] = 1.0
    np.random.shuffle(mask)
    mask = mask.astype(int)
    train_inds = (1 - mask).nonzero()[0]
    valid_inds = mask.nonzero()[0]
    train_keys = [demos[i] for i in train_inds]
    valid_keys = [demos[i] for i in valid_inds]
    print(f"{num_val} validation demonstrations out of {num_demos} total demonstrations.")

    train_name = "train" if filter_key is None else f"{filter_key}_train"
    valid_name = "valid" if filter_key is None else f"{filter_key}_valid"
    train_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=train_keys, key_name=train_name)
    valid_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=valid_keys, key_name=valid_name)

    print(f"Total number of train samples: {np.sum(train_lengths)}")
    print(f"Average number of train samples {np.mean(train_lengths)}")
    print(f"Total number of valid samples: {np.sum(valid_lengths)}")
    print(f"Average number of valid samples {np.mean(valid_lengths) if valid_lengths else 0.0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to hdf5 dataset")
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="optional existing mask key to split instead of all demos",
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="validation ratio")
    args = parser.parse_args()

    np.random.seed(0)
    split_train_val_from_hdf5(args.dataset, val_ratio=args.ratio, filter_key=args.filter_key)
