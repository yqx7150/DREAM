#!/usr/bin/env bash
# Test / reconstruction: write the reconstructed sinogram of every sample to
#   results_sinogram/<name>/visualization/<file>.mat   (key: sino_net1)
#
#   bash scripts/test.sh [options/test_DREAMHM_S2.yml]
#
# Then convert the sinograms into PET images with
#   matlab -batch "cd reconstruction; recon_from_mat('../results_sinogram/DREAMHM_S2_test/visualization', '../results_sinogram/DREAMHM_S2_test/images')"
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-options/test_DREAMHM_S2.yml}

python dream_hm/test.py -opt "${CONFIG}"