# flake8: noqa
"""Entry point of the test / reconstruction pipeline.

``python dream_hm/test.py -opt options/test_DREAMHM_S2.yml``
"""
import os
import sys
from os import path as osp

ROOT_PATH = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.insert(0, ROOT_PATH)

os.environ.setdefault('data_type', 'sinogram')

import dream_hm.archs  # noqa: E402,F401
import dream_hm.data  # noqa: E402,F401
import dream_hm.models  # noqa: E402,F401
from basicsr.test import test_pipeline  # noqa: E402

if __name__ == '__main__':
    test_pipeline(ROOT_PATH)
