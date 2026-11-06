r"""HCMM (hot-cold mutual mask) dataset for DREAM-HM.

Each sample is a paired PET sinogram stored in a ``.mat`` file with two
variables:

``gt``
    noise-free (full-count) sinogram, flattened column-major.
``lq``
    noisy (low-count) sinogram, flattened column-major.

The sinogram is reshaped to ``sino_h x sino_w`` (256 x 192 by default), zero
padded to a square patch (256 x 256) and decoupled into two channels with the
hot-cold mutual mask (HCMM) strategy:

.. math::

    M_{hot} = \mathbb{1}[S \ge \tau], \quad M_{cold} = 1 - M_{hot}

    H = S \odot M_{hot}, \quad C = S \odot M_{cold}

The threshold :math:`\tau` is the ``100 * (1 - hot_ratio)`` percentile of the
*noise-free* sinogram, so that the LQ/GT pair always shares exactly the same
partition.  Channel order in the returned tensors is ``[cold, hot]``, which is
the order used by the released checkpoints.
"""

import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat

from basicsr.utils import scandir
from basicsr.utils.registry import DATASET_REGISTRY


def estimate_tau(sino, hot_ratio=0.1):
    """Return the intensity threshold that keeps ``hot_ratio`` of the pixels hot.

    The threshold is the ``100 * (1 - hot_ratio)``-th percentile of the
    sinogram, evaluated on the sorted values (``part`` instead of a
    quantile interpolation) so that ``sino >= tau`` selects exactly
    ``int(numel(sino) * hot_ratio)`` pixels.
    """
    if not 0.0 < hot_ratio < 1.0:
        raise ValueError(f'hot_ratio must be in (0, 1), got {hot_ratio}')
    x = np.asarray(sino, dtype=np.float64).reshape(-1)
    num_hot = max(1, int(x.size * hot_ratio))
    k = x.size - num_hot
    return float(np.partition(x, k)[k])


def hcmm_split(sino, tau):
    """Split one sinogram into ``(cold, hot)`` channels given a threshold."""
    mask_hot = (sino >= tau).astype(np.float32)
    mask_cold = 1.0 - mask_hot
    return sino * mask_cold, sino * mask_hot


def load_sino(path, sino_h=256, sino_w=192, pad_w=32, normalize=True):
    """Load one ``.mat`` sinogram pair and return ``(gt, lq)`` as 2-D arrays.

    The stored vectors are column-major (MATLAB / F order) and are zero padded
    symmetrically in the detector dimension to reach ``sino_h x (sino_w + 2 * pad_w)``.
    """
    data_sino = loadmat(path)
    gt = np.reshape(np.asarray(data_sino['gt'], dtype=np.float32), (sino_h, sino_w), order='F')
    lq = np.reshape(np.asarray(data_sino['lq'], dtype=np.float32), (sino_h, sino_w), order='F')
    if normalize:
        gt = gt / np.max(gt)
        lq = lq / np.max(lq)
    pad = ((0, 0), (pad_w, pad_w))
    gt = np.pad(gt, pad, mode='constant', constant_values=0)
    lq = np.pad(lq, pad, mode='constant', constant_values=0)
    return gt, lq


@DATASET_REGISTRY.register()
class SinoHeatMaskDataset(data.Dataset):
    """DREAM-HM dataset (HCMM dual-channel sinogram blocks).

    Args:
        opt (dict): dataset options. Recognised keys:

            - ``dataroot_GT``: folder that contains the ``.mat`` sinograms.
            - ``phase``: ``train`` / ``val`` / ``test`` (injected by the pipeline).
            - ``hot_ratio``: fraction of *pixels of the padded block* assigned
              to the hot channel (default ``0.1``, i.e. ``tau`` is the 90th
              percentile of the 256 x 256 block).  The zero padding of the
              detector axis makes this ~13% of the 192-column sinogram, exactly
              as in the reference implementation.
            - ``sino_h``, ``sino_w``, ``pad_w``: sinogram geometry.
            - ``clean_channel``: append the unmasked sinogram as a third,
              data-fidelity channel (default ``True``, as used for the released
              checkpoints); set to ``False`` for the two-channel block of
              Eq. (4) in the paper.
            - ``normalize``: divide by the per-sinogram maximum; defaults to
              ``True`` for train/val and ``False`` for test (the test pipeline
              rescales with the peak of the input block before saving).
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.sino_h = int(opt.get('sino_h', 256))
        self.sino_w = int(opt.get('sino_w', 192))
        self.pad_w = int(opt.get('pad_w', 32))
        self.hot_ratio = float(opt.get('hot_ratio', 0.1))
        self.normalize = bool(opt.get('normalize', self.phase != 'test'))
        # the released checkpoints use ``[cold, hot, full]`` (3 channels); set
        # ``clean_channel: false`` for the strict two-channel block of Eq. (4)
        self.clean_channel = bool(opt.get('clean_channel', True))

        root = opt.get('dataroot_GT') or opt.get('dataroot_gt')
        assert root is not None, 'dataroot_GT is required for SinoHeatMaskDataset'
        self.paths = sorted(scandir(root, suffix='.mat', recursive=False, full_path=True))
        assert len(self.paths) > 0, f'No .mat sinogram found in {root}'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        gt, lq = load_sino(path, self.sino_h, self.sino_w, self.pad_w, self.normalize)

        # the threshold is estimated on the noise-free sinogram, the resulting
        # hot/cold partition is shared by the LQ/GT pair
        tau = estimate_tau(gt, self.hot_ratio)
        cold_gt, hot_gt = hcmm_split(gt, tau)
        cold_lq, hot_lq = hcmm_split(lq, tau)

        gt_channels = [cold_gt, hot_gt]
        lq_channels = [cold_lq, hot_lq]
        if self.clean_channel:
            # data-fidelity channel: the (unmasked) sinogram itself
            gt_channels.append(gt)
            lq_channels.append(lq)

        gt_block = np.stack(gt_channels, axis=0).astype(np.float32)
        lq_block = np.stack(lq_channels, axis=0).astype(np.float32)

        return {
            'lq': torch.from_numpy(lq_block),
            'gt': torch.from_numpy(gt_block),
            'lq_path': path,
            'gt_path': path,
            'tau': tau,
        }