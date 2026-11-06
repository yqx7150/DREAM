"""Inspect the HCMM decomposition of one sinogram.

Example
-------
    python tools/visualize_hcmm.py --mat datasets/sinogram/test/4202.mat --out hcmm.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from dream_hm.data.heatmask_dataset import estimate_tau, hcmm_split, load_sino


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat', required=True, help='a sinogram .mat file (keys: gt, lq)')
    parser.add_argument('--out', default='hcmm.png', help='output figure')
    parser.add_argument('--hot-ratio', type=float, default=0.1, help='fraction of hot pixels (default 0.1)')
    parser.add_argument('--sino-h', type=int, default=256)
    parser.add_argument('--sino-w', type=int, default=192)
    parser.add_argument('--pad-w', type=int, default=32)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gt, lq = load_sino(args.mat, args.sino_h, args.sino_w, args.pad_w, normalize=True)
    tau = estimate_tau(gt, args.hot_ratio)
    cold, hot = hcmm_split(gt, tau)
    mask_hot = (gt >= tau).astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        (gt, f'input sinogram (GT)\ntau = {tau:.4f}, hot ratio = {args.hot_ratio:.0%}'),
        (mask_hot, 'hot mask  M_hot = 1[S >= tau]'),
        (hot, 'hot channel  H = S * M_hot'),
        (lq, 'input sinogram (LQ)'),
        (1.0 - mask_hot, 'cold mask  M_cold = 1 - M_hot'),
        (cold, 'cold channel  C = S * M_cold'),
    ]
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'hot ratio {mask_hot.mean():.3f}, tau {tau:.4f} -> {args.out}')


if __name__ == '__main__':
    main()