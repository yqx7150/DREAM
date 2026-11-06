# Third-party components

| folder | origin | license |
|---|---|---|
| `KER_v0.3/` | KER + IRT toolbox by J. Fessler (University of Michigan), used for the system matrix and the MLEM/OSEM reconstruction | see `KER_v0.3/readme.txt` and `KER_v0.3/DISCLAIMER.m` |
| `../basicsr/` | [BasicSR](https://github.com/XPixelGroup/BasicSR) | Apache-2.0 |
| `../ldm/` | [latent-diffusion](https://github.com/CompVis/latent-diffusion) (DDPM/DDIM helpers) | MIT |

`basicsr/` is a vendored copy; `basicsr/models/sr_model.py`,
`basicsr/utils/options.py` and `basicsr/metrics/metric_util.py` were adapted for
sinogram reconstruction. If you redistribute this repository, keep the original
license notices of these projects.

The `KER_v0.3` toolbox is included to make the reconstruction step reproducible.
It contains prebuilt `mex` binaries for some platforms; on other systems run the
build scripts in `KER_v0.3/irt/mex/` (`make_all.m`) or install the toolbox from
its original distribution.

The `KER_v0.3/demo/` folder keeps only the upstream phantom demos. Scripts and sample
data derived from clinical acquisitions are **not** distributed with this repository.
