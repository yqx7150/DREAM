# flake8: noqa
"""Entry point of the training pipeline.

``python dream_hm/train.py -opt options/train_DREAMHM_S1.yml --launcher pytorch``
"""
import os
import sys
import warnings
from os import path as osp

ROOT_PATH = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.insert(0, ROOT_PATH)

# output folders are named experiments_{data_type} / results_{data_type};
# PET sinogram reconstruction is the default mode of this repository
os.environ.setdefault('data_type', 'sinogram')

import dream_hm.archs  # noqa: E402,F401
import dream_hm.data  # noqa: E402,F401
import dream_hm.losses  # noqa: E402,F401
import dream_hm.models  # noqa: E402,F401
from dream_hm.train_pipeline import train_pipeline  # noqa: E402

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    train_pipeline(ROOT_PATH)
