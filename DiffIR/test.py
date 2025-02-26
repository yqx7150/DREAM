# flake8: noqa
import os.path as osp
import sys
sys.path.insert(0, osp.abspath(osp.join(__file__, osp.pardir, osp.pardir)))
from basicsr.test import test_pipeline
import DiffIR.data.dataset_GTLQ
import DiffIR.archs
import DiffIR.data
import DiffIR.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
