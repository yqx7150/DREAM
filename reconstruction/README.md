# PET image reconstruction (MLEM / OSEM)

`dream_hm/test.py` writes one reconstructed sinogram per test slice:

```
results_sinogram/DREAMHM_S2_test/visualization/0001.mat   # variable: sino_net1
```

The script in this folder turns those sinograms into PET images with the
ordered-subset expectation-maximisation (OSEM) algorithm of the KER_v0.3
toolbox, which is bundled under `third_party/KER_v0.3`.

## Usage

```matlab
cd reconstruction
recon_from_mat('../results_sinogram/DREAMHM_S2_test/visualization', ...
               '../results_sinogram/DREAMHM_S2_test/images');
```

or from a shell:

```bash
matlab -batch "cd reconstruction; recon_from_mat('../results_sinogram/DREAMHM_S2_test/visualization', '../results_sinogram/DREAMHM_S2_test/images')"
```

Name/value options:

| option    | default | meaning                                                    |
|-----------|---------|------------------------------------------------------------|
| `method`  | `osem`  | `osem` (12 subsets) or plain `mlem` (all angles per update) |
| `nblock`  | 12      | number of subsets for OSEM                                  |
| `niter`   | 6 / 60  | iterations (`6 x 12 = 72` subset updates by default)         |
| `fwhm`    | 7       | Gaussian post-smoothing in pixels, `0` disables it           |
| `ni`, `ri`| ones / zeros | normalisation and additive (randoms + scatter) sinograms |

Each output is `<name>_rec.mat` (variable `rec`, 256 x 256) and `<name>_rec.png`.

## Notes

* The reconstruction geometry is the one used in the paper: a 256 x 192
  parallel-beam sinogram with a 32 cm field of view and a 256 x 256 image.
* `ni` and `ri` should be supplied for real acquisitions (attenuation
  correction and randoms/scatter estimates). With the defaults the script
  performs the same reconstruction as the paper's experiments but without
  attenuation / randoms modelling.
* The reference images are produced by running the same script on the
  noise-free sinograms.