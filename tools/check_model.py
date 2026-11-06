"""Instantiate both DREAM-HM networks and run a dummy forward pass.

    python tools/check_model.py --channels 3 --size 64

Useful to verify that the environment is correctly set up (no data or GPU
required) and to read the parameter count of the models.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch  # noqa: E402

from dream_hm.archs.DREAMHM_S1_arch import DREAMHM_S1  # noqa: E402
from dream_hm.archs.DREAMHM_S2_arch import DREAMHM_S2  # noqa: E402

ARCH = dict(dim=48, num_blocks=[3, 5, 6, 6], num_refinement_blocks=4, heads=[1, 2, 4, 8],
            ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')


def n_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.rand(1, args.channels, args.size, args.size, device=device)
    gt = torch.rand_like(x)

    s1 = DREAMHM_S1(n_encoder_res=5, inp_channels=args.channels, out_channels=args.channels, **ARCH).to(device).eval()
    with torch.no_grad():
        y1 = s1(x, gt)
    print(f'DREAMHM_S1: {n_params(s1):.2f} M params, output {tuple(y1.shape)}')

    s2 = DREAMHM_S2(n_encoder_res=5, inp_channels=args.channels, out_channels=args.channels,
                    n_denoise_res=1, linear_start=0.1, linear_end=0.99, timesteps=4, **ARCH).to(device).eval()
    with torch.no_grad():
        y2 = s2(x)
    print(f'DREAMHM_S2: {n_params(s2):.2f} M params, output {tuple(y2.shape)}')
    print('forward pass ok')


if __name__ == '__main__':
    main()