import os
import random
import sys

import cv2
# import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass
import random
from basicsr.utils.registry import DATASET_REGISTRY
import torch.nn.functional as F
from DiffIR.data.transforms import paired_random_crop, random_augmentation
from scipy.io import loadmat

@DATASET_REGISTRY.register()
class LQGTDataset_sino_nomin_mask(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        # self.LR_sizex, self.GT_sizex = opt["LR_sizex"], opt["GT_sizex"]
        # self.LR_sizey, self.GT_sizey = opt["LR_sizey"], opt["GT_sizey"]
        # self.phase = phase

        self.img_id = None
        # self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0
        # print(opt["dataroot_GT"],opt["LR_size"])
        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            # self.LR_paths = util.get_image_paths(
            #     opt["data_type"], opt["dataroot_LQ"]
            # )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]
    '''
    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    '''
    def nor(self, data):
        size = data.shape
        assert len(size) == 2
        if isinstance(data, torch.Tensor):
            minValue = torch.min(data)
            maxValue = torch.max(data)
            out = (data - minValue) / (maxValue - minValue)
        else:
            minValue = np.min(data)
            maxValue = np.max(data)
            out = (data - minValue) / (maxValue - minValue)
        return {'nor':out, 'max':maxValue, 'min':minValue}

    def mask(self, img, white_rate):
        cell_pix_x = 16

        cell_num_x = 256 / cell_pix_x
        # assert cell_num_x
        white_rate = white_rate
        black_rate = 1 - white_rate
        black_cell_num = round((cell_num_x*cell_num_x * black_rate) / 1)  # 1:黑率

        x_idx = np.random.randint(0, cell_num_x, black_cell_num)

        y_idx = np.random.randint(0, cell_num_x, black_cell_num)

        mask_cell = np.ones((256,256))

        for i in range(black_cell_num):
            mask_cell[y_idx[i] * cell_pix_x:y_idx[i] * cell_pix_x + cell_pix_x, 
                    x_idx[i] * cell_pix_x:x_idx[i] * cell_pix_x + cell_pix_x] = 0
        
        if len(img.shape) == 3:# 应该不用管，可以广播
            mask_cell = np.repeat(mask_cell[None, ...], img.shape[0], 0)
        return {'data':np.multiply(img, mask_cell), 'location':[x_idx, y_idx]}


    def __getitem__(self, index):
        
        # print(index, self.img_id)
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        # scale = self.opt["scale"] if self.opt["scale"] else 1
        # GT_size = self.opt["GT_size"]
        # LR_size = self.opt["LR_size"]
        

        # get GT image
        GT_path = self.GT_paths[index]
        # LR_path = self.LR_paths[index]


        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None



        need_num = 3
        if self.opt["phase"] == 'train':
            data_sino = loadmat(GT_path)

            GT_img = np.reshape(data_sino['gt'], [256, 192], order = 'F')
            GT_img = GT_img / np.max(GT_img)
            GT_img = np.pad(GT_img,((0,0),(32, 32)), mode='constant')
            GT_img = np.repeat(GT_img[None, ...], 3, 0)


            LQ_img = np.reshape(data_sino['lq'], [256, 192], order = 'F')
            LQ_img = LQ_img / np.max(LQ_img)
            LQ_img = np.pad(LQ_img,((0,0),(32, 32)), mode='constant')
            LQ_img = np.repeat(LQ_img[None, ...], 3, 0)
        
            mask1 = self.mask(np.stack((GT_img[0,:,:], LQ_img[0,:,:]),0), 0.9)['data']
            mask2 = self.mask(np.stack((GT_img[1,:,:], LQ_img[1,:,:]),0), 0.9)['data']

            GT = np.stack((mask1[0,:,:],mask2[0,:,:],GT_img[2,:,:]), 0)
            LQ = np.stack((mask1[1,:,:],mask2[1,:,:],LQ_img[2,:,:]), 0)



            lct = {}
            




            # GT_img, LQ_img = random_augmentation(GT_img.transpose(1,2,0), LQ_img.transpose(1,2,0))
            # GT_img = GT_img.transpose(2,0,1)
            # LQ_img = LQ_img.transpose(2,0,1)
        

        elif self.opt["phase"] == 'val':
            data_sino = loadmat(GT_path)

            GT_img = np.reshape(data_sino['gt'], [256, 192], order = 'F')
            GT_img = GT_img / np.max(GT_img)
            GT_img = np.pad(GT_img,((0,0),(32, 32)), mode='constant')
            GT_img = np.repeat(GT_img[None, ...], 3, 0)


            LQ_img = np.reshape(data_sino['lq'], [256, 192], order = 'F')
            LQ_img = LQ_img / np.max(LQ_img)
            LQ_img = np.pad(LQ_img,((0,0),(32, 32)), mode='constant')
            LQ_img = np.repeat(LQ_img[None, ...], 3, 0)


            data1 = self.mask(np.stack((GT_img[0,:,:], LQ_img[0,:,:]),0), 0.9)
            data2 = self.mask(np.stack((GT_img[1,:,:], LQ_img[1,:,:]),0), 0.9)


            mask1 = data1['data']
            mask2 = data2['data']

            GT = np.stack((mask1[0,:,:],mask2[0,:,:],GT_img[2,:,:]), 0)
            LQ = np.stack((mask1[1,:,:],mask2[1,:,:],LQ_img[2,:,:]), 0)

            lct = {'0':data1['location'], '1':data2['location']}






        elif self.opt["phase"] == 'test':
            data_sino = loadmat(GT_path)

            GT_img = np.reshape(data_sino['gt'], [256, 192], order = 'F')
            # GT_img = GT_img / np.max(GT_img)
            GT_img = np.pad(GT_img,((0,0),(32, 32)), mode='constant')
            GT_img = np.repeat(GT_img[None, ...], 3, 0)


            LQ_img = np.reshape(data_sino['lq'], [256, 192], order = 'F')
            # LQ_img = LQ_img / np.max(LQ_img)
            LQ_img = np.pad(LQ_img,((0,0),(32, 32)), mode='constant')
            LQ_img = np.repeat(LQ_img[None, ...], 3, 0)


            data1 = self.mask(np.stack((GT_img[0,:,:], LQ_img[0,:,:]),0), 0.9)
            data2 = self.mask(np.stack((GT_img[1,:,:], LQ_img[1,:,:]),0), 0.9)


            mask1 = data1['data']
            mask2 = data2['data']

            GT = np.stack((mask1[0,:,:],mask2[0,:,:],GT_img[2,:,:]), 0)
            LQ = np.stack((mask1[1,:,:],mask2[1,:,:],LQ_img[2,:,:]), 0)

            lct = {'0':data1['location'], '1':data2['location']}

        
        

        GT_img = torch.from_numpy(
            np.ascontiguousarray(np.array(GT, np.float32)))
        
        LQ_img = torch.from_numpy(
            np.ascontiguousarray(np.array(LQ, np.float32))
        )
        # print(GT_img.shape)

        if LR_path is None:
            LR_path = GT_path

        return {"lq": LQ_img, "gt": GT_img, "lq_path": LR_path, "gt_path": GT_path, 'lct':lct}

    def __len__(self):
        return len(self.GT_paths)
