#!/usr/bin/env bash
set -e

# clone if missing
[ -d robosuite ] || git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
[ -d robomimic ] || git clone https://github.com/ARISE-Initiative/robomimic.git robomimic

# pin versions (example)
cd robosuite && git fetch --tags && git checkout v1.5.1 && cd ..
cd robomimic && git fetch --tags && git checkout v0.5.0 && cd ..

# install editable
python -m pip install -e ./robosuite
python -m pip install -e ./robomimic

