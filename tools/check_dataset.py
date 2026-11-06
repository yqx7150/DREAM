"""Sanity check of a DREAM-HM sinogram dataset.

    python tools/check_dataset.py --root datasets/sinogram/train

The dataset must contain one ``.mat`` file per slice with the variables ``gt``
and ``lq`` (each a flattened ``256 x 192`` sinogram, column-major).
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--sino-h', type=int, default=256)
    parser.add_argument('--sino-w', type=int, default=192)
    parser.add_argument('--limit', type=int, default=0, help='check only the first N files (0 = all)')
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.root) if f.endswith('.mat'))
    if not files:
        raise SystemExit(f'no .mat file in {args.root}')
    if args.limit:
        files = files[:args.limit]

    expected = args.sino_h * args.sino_w
    bad = 0
    for name in files:
        mat = loadmat(os.path.join(args.root, name))
        if 'gt' not in mat or 'lq' not in mat:
            print(f'[FAIL] {name}: needs both `gt` and `lq`')
            bad += 1
            continue
        for key in ('gt', 'lq'):
            arr = np.asarray(mat[key]).reshape(-1)
            if arr.size != expected:
                print(f'[FAIL] {name}: `{key}` has {arr.size} values, expected {expected}')
                bad += 1
            elif arr.max() <= 0:
                print(f'[FAIL] {name}: `{key}` is all zeros')
                bad += 1

    print(f'checked {len(files)} files, {len(files) - bad} ok, {bad} problems')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()