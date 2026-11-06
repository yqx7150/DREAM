#!/usr/bin/env bash
# Stage 2: train the diffusion stage + the transformer stage of DREAM-HM.
#
#   bash scripts/train_S2.sh [options/train_DREAMHM_S2.yml]
#
# Before running, set `pretrain_network_g` and `pretrain_network_S1` in the
# yml to the stage-1 checkpoint, e.g.
#   experiments_sinogram/DREAMHM_S1/models/net_g_latest.pth
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-options/train_DREAMHM_S2.yml}
NUM_GPU=${NUM_GPU:-1}
PORT=${PORT:-29500}

torchrun --nproc_per_node="${NUM_GPU}" --master_port="${PORT}" \
    dream_hm/train.py -opt "${CONFIG}" --launcher pytorch