"""Convert the reconstructed ``.mat`` files into PNG images.

    python tools/mat2png.py --root results_sinogram/DREAMHM_S2_test/images --key rec
"""

import argparse
import os

import cv2
import numpy as np
from scipy.io import loadmat


def to_uint8(img):
    img = np.squeeze(img).astype(np.float32)
    if np.iscomplexobj(img):
        img = np.abs(img)
    lo, hi = float(img.min()), float(img.max())
    if hi <= 1.0:
        img = img * 255.0
    elif hi - lo > 0:
        img = (img - lo) / (hi - lo) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='folder with the .mat files')
    parser.add_argument('--out', default=None, help='output folder (default: <root>/png)')
    parser.add_argument('--key', default='rec', help='mat variable to export (default: rec)')
    parser.add_argument('--size', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='optional resize (height width)')
    args = parser.parse_args()

    out = args.out or os.path.join(args.root, 'png')
    os.makedirs(out, exist_ok=True)

    files = sorted(f for f in os.listdir(args.root) if f.endswith('.mat'))
    n = 0
    for name in files:
        mat = loadmat(os.path.join(args.root, name))
        if args.key not in mat:
            continue
        img = to_uint8(mat[args.key])
        if args.size:
            img = cv2.resize(img, (args.size[1], args.size[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out, os.path.splitext(name)[0] + '.png'), img)
        n += 1
    print(f'{n} images -> {out}')


if __name__ == '__main__':
    main()