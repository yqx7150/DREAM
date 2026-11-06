"""Random-mask dataset for DREAM-RM (the stochastic counterpart of DREAM-HM).

``num_masks`` sinograms are covered with ``block_size x block_size`` random
blocks that hide ``mask_ratio`` of the spatial area; the untouched sinogram is
kept as the last channel to preserve data fidelity.  The resulting
``num_masks + 1`` channels are stacked into a 3-D sinogram block, as described
in Section II of the paper.
"""

import numpy as np
import torch
import torch.utils.data as data

from basicsr.utils import scandir
from basicsr.utils.registry import DATASET_REGISTRY

from .heatmask_dataset import load_sino


def random_block_mask(shape, block_size=16, mask_ratio=0.1, rng=None):
    """Return a binary mask (1 = keep) and the coordinates of the masked blocks."""
    rng = np.random if rng is None else rng
    height, width = shape
    cell_num_y, cell_num_x = height // block_size, width // block_size
    num_blocks = int(round(cell_num_y * cell_num_x * mask_ratio))

    mask = np.ones((height, width), dtype=np.float32)
    y_idx = rng.randint(0, cell_num_y, num_blocks)
    x_idx = rng.randint(0, cell_num_x, num_blocks)
    for y, x in zip(y_idx, x_idx):
        y0, x0 = y * block_size, x * block_size
        mask[y0:y0 + block_size, x0:x0 + block_size] = 0.0
    return mask, [np.asarray(x_idx), np.asarray(y_idx)]


@DATASET_REGISTRY.register()
class SinoRandomMaskDataset(data.Dataset):
    """DREAM-RM dataset (``num_masks`` random-masked channels + 1 clean channel).

    Args:
        opt (dict): dataset options. In addition to the keys understood by
            :class:`~dream_hm.data.heatmask_dataset.SinoHeatMaskDataset`, this
            dataset uses ``num_masks`` (default ``2``), ``block_size``
            (default ``16``) and ``mask_ratio`` (default ``0.1``).
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.sino_h = int(opt.get('sino_h', 256))
        self.sino_w = int(opt.get('sino_w', 192))
        self.pad_w = int(opt.get('pad_w', 32))
        self.num_masks = int(opt.get('num_masks', 2))
        self.block_size = int(opt.get('block_size', 16))
        self.mask_ratio = float(opt.get('mask_ratio', 0.1))
        self.normalize = bool(opt.get('normalize', self.phase != 'test'))

        root = opt.get('dataroot_GT') or opt.get('dataroot_gt')
        assert root is not None, 'dataroot_GT is required for SinoRandomMaskDataset'
        self.paths = sorted(scandir(root, suffix='.mat', recursive=False, full_path=True))
        assert len(self.paths) > 0, f'No .mat sinogram found in {root}'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        gt, lq = load_sino(path, self.sino_h, self.sino_w, self.pad_w, self.normalize)

        gt_channels, lq_channels, locations = [], [], {}
        for i in range(self.num_masks):
            mask, where = random_block_mask(gt.shape, self.block_size, self.mask_ratio)
            gt_channels.append(gt * mask)
            lq_channels.append(lq * mask)
            locations[str(i)] = where
        # the last channel is left untouched to preserve data fidelity
        gt_channels.append(gt)
        lq_channels.append(lq)

        return {
            'lq': torch.from_numpy(np.stack(lq_channels, axis=0).astype(np.float32)),
            'gt': torch.from_numpy(np.stack(gt_channels, axis=0).astype(np.float32)),
            'lq_path': path,
            'gt_path': path,
            'lct': locations,
        }