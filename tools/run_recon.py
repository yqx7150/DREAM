"""Run the MATLAB reconstruction of DREAM sinograms from Python.

    python tools/run_recon.py --sino-dir results_sinogram/DREAMHM_S2_test/visualization \
                             --out-dir  results_sinogram/DREAMHM_S2_test/images \
                             --method osem --nblock 12 --niter 6

Requires MATLAB on the PATH (``matlab``) and the KER_v0.3 toolbox bundled in
``third_party/``.
"""

import argparse
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sino-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--method', default='osem', choices=['osem', 'mlem'])
    parser.add_argument('--nblock', type=int, default=12)
    parser.add_argument('--niter', type=int, default=None)
    parser.add_argument('--fwhm', type=float, default=7)
    parser.add_argument('--matlab', default=os.environ.get('MATLAB', 'matlab'))
    args = parser.parse_args()

    matlab = args.matlab or shutil.which('matlab')
    if not matlab:
        sys.exit('MATLAB not found: pass --matlab /path/to/matlab or set $MATLAB')

    sino_dir = os.path.abspath(args.sino_dir)
    out_dir = os.path.abspath(args.out_dir)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    niter = '' if args.niter is None else f", 'niter', {args.niter}"
    cmd = (
        f"cd('{root}'); addpath(fullfile('{root}', 'reconstruction')); "
        f"recon_from_mat('{sino_dir}', '{out_dir}', 'method', '{args.method}', "
        f"'nblock', {args.nblock}, 'fwhm', {args.fwhm}{niter});"
    )
    print('running:', cmd)
    subprocess.run([matlab, '-batch', cmd], check=True)


if __name__ == '__main__':
    main()