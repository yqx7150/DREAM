#!/usr/bin/env bash
# Stage 1: train the compact-prior extraction network (CPEN) of DREAM-HM.
#
#   bash scripts/train_S1.sh [options/train_DREAMHM_S1.yml]
#
# Environment variables: NUM_GPU (default 1), PORT (default 29500)
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-options/train_DREAMHM_S1.yml}
NUM_GPU=${NUM_GPU:-1}
PORT=${PORT:-29500}

torchrun --nproc_per_node="${NUM_GPU}" --master_port="${PORT}" \
    dream_hm/train.py -opt "${CONFIG}" --launcher pytorch