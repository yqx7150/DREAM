#!/usr/bin/env bash

export data_type=sinogram
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4393 DiffIR/train.py -opt options/train_DiffIRS1_sino.yml --launcher pytorch 

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4394 DiffIR/train.py -opt options/train_DiffIRS1_sino_nomin.yml --launcher pytorch 