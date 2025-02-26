# flake8: noqa
import os
import os.path as osp
# root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# print(os.getcwd(),osp.abspath(osp.join(__file__, osp.pardir, osp.pardir)))
import sys
sys.path.insert(0, osp.abspath(osp.join(__file__, osp.pardir, osp.pardir)))

from DiffIR.train_pipeline import train_pipeline

import DiffIR.archs
import DiffIR.data
import DiffIR.models
import DiffIR.losses
import warnings


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # a = os.environ.get('data_type', "none")
    # print(a, type(a))
    # assert 0
    train_pipeline(root_path)
