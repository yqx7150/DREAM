# DREAM

**Diffusion Transformer Meets Heat Masks — PET sinogram reconstruction with physics-guided hot–cold masks.**

This repository is the official implementation of **DREAM**
(*Diffusion tRansformer mEets heAt Masks*). The `main` branch hosts the
heat-mask variant **DREAM-HM**; the random-mask variant **DREAM-RM** is kept as
a reproducible baseline (`options/rm/`). The paper is described in

> **Diffusion Transformer Meets Heat Masks: An Advanced PET Reconstruction Framework**
> Bin Huang, Yanhan Chen, Binzhong He, Zhili Liu, Xinyue Wang, Binxuan Li, Qiegen Liu
> *IEEE Transactions on Image Processing*

DREAM integrates masking into **both** the sinogram domain and the latent space:
a high-dimensional stacking of masked sinograms expands the solution space, a
compact prior extracted from that block drives a diffusion process, and a
diffusion-transformer reconstructs the sinogram that is finally turned into a
PET image by ordered-subset EM.

* **DREAM-RM** (random mask): stochastic block masks, keeps one unmasked channel.
* **DREAM-HM** (heat mask, this work): the sinogram is decoupled into a
  **high-count hot channel** and a **low-count cold channel** by a global
  intensity threshold, and the two channels are coupled by the
  **hot–cold mutual mask (HCMM)** strategy.

What is in this release:

* `dream_hm/` — the model package (renamed from `DiffIR/`), with
  `DREAMHM_S1` / `DREAMHM_S2` and the two training stages;
* `dream_hm/data/heatmask_dataset.py` — the HCMM dual-channel block
  (`tau` = P90 of the padded 256 x 256 block) and
  `randommask_dataset.py` — the DREAM-RM baseline;
* `options/` (DREAM-HM) and `options/rm/` (DREAM-RM) configs, `scripts/`,
  `tools/`, `docs/`, `reconstruction/` (MATLAB OSEM / MLEM);
* `basicsr/`, `ldm/` and `third_party/KER_v0.3` are vendored, so no external
  BasicSR installation is required;
* the code of the previous DREAM-RM-only release is still in the git history
  (commits up to `6a1e35a`).

---

## Paper figures

All figures are taken from the TIP manuscript (*Diffusion Transformer Meets Heat
Masks: An Advanced PET Reconstruction Framework*).

**Fig. 1 — DREAM framework.** Comparison of the traditional method and the
proposed DREAM framework. DREAM-RM employs the random mask strategy with
progressive hierarchical masking to construct 3-D sinogram blocks, whereas
DREAM-HM replaces the random masks by the physics-guided **HCMM** strategy and
explicitly decouples the sinogram into high-count (hot) and low-count (cold)
channels.

![Fig. 1 — DREAM framework](DREAM_data/figs/fig1%20framework.png)

**Fig. 2 — Training procedure.** The framework mainly consists of mask
generation modules, a mask-driven latent space integrating a diffusion stage,
and a Transformer U-Net. Two parallel strategies for prior generation are
shown: the DREAM-HM path (top, HCMM + progressive hierarchical masking →
HCCP) and the DREAM-RM path (bottom, random masks → SMCP). The extracted prior
guides the transformer stage to reconstruct the result from the noisy
sinograms.

![Fig. 2 — Training procedure](DREAM_data/figs/fig2%20training%20procedure.png)

**Fig. 3 — Reconstruction procedure.** Both data-preparation strategies are
supported (DREAM-RM: `N` channels, DREAM-HM: two hot/cold channels). The UCP is
extracted and refined through the reverse diffusion process, then guides the
transformer stage; the reconstructed channels are fused into a single sinogram
and converted into a PET image by MLEM.

![Fig. 3 — Reconstruction procedure](DREAM_data/figs/fig3%20reconstruction%20procedure.png)

**Fig. 4 — Reconstructed PET images.** (a)-(i) comparison methods and DREAM;
(j) ground truth. The second row shows the residuals w.r.t. the ground truth.

![Fig. 4 — Reconstructed PET images](DREAM_data/figs/fig4%20pet%20images.jpg)

**Fig. 5 — Reconstructed PET sinograms.** (a)-(i) comparison methods and DREAM;
(j) ground-truth sinogram. The second row shows the residuals.

![Fig. 5 — Reconstructed PET sinograms](DREAM_data/figs/fig5%20sinograms.jpg)

<details>
<summary><b>More figures (Fig. 6-16): line profiles, low-dose, 3-D and ablation studies</b></summary>

**Fig. 6 — Image line profiles.** Comparison of PET image line profiles between
the ground truth and the nine reconstruction methods.

![Fig. 6](DREAM_data/figs/fig6%20line%20profiles%20image.png)

**Fig. 7 — Sinogram line profiles.** Comparison between the ground truth, the
low-quality input and eight reconstruction methods.

![Fig. 7](DREAM_data/figs/fig7%20line%20profiles%20sinogram.png)

**Fig. 8 — Low-dose reconstruction.** LD PET reconstruction at 1 %, 5 %, 10 %
and 20 % of the clinical dose, in the image domain (A)-(I) and in the sinogram
domain (a)-(i); the second and fourth rows are the absolute residuals.

![Fig. 8](DREAM_data/figs/fig8%20dose%20levels.jpg)

**Fig. 9 — 3-D simulation.** Sampling geometry and sampled 3-D PET data.

![Fig. 9](DREAM_data/figs/fig9%203d%20sampling.png)

**Fig. 10 — 3-D reconstruction.** 3-D PET reconstruction results of DREAM-HM.

![Fig. 10](DREAM_data/figs/fig10%203d%20results.png)

**Fig. 11 — Number of diffusion steps.** PSNR / SSIM / RMSE of the reconstructed
sinogram as a function of the diffusion steps.

![Fig. 11](DREAM_data/figs/fig11%20diffusion%20steps.png)

**Fig. 12 — Mask algorithms.** Boxplots of PSNR / SSIM / RMSE for different
masking strategies.

![Fig. 12](DREAM_data/figs/fig12%20mask%20algorithms.png)

**Fig. 13 — Mask levels.** Boxplots of PSNR / SSIM / RMSE for different mask
levels.

![Fig. 13](DREAM_data/figs/fig13%20mask%20levels.png)

**Fig. 14 — Channel numbers.** Boxplots of PSNR / SSIM / RMSE for different
numbers of stacked channels.

![Fig. 14](DREAM_data/figs/fig14%20channel%20numbers.png)

**Fig. 15 — Threshold `tau`.** Reconstruction results of DREAM-HM for different
`tau`; the second row shows the residuals.

![Fig. 15](DREAM_data/figs/fig15%20tau.png)

**Fig. 16 — Hot/cold SNR analysis.** (a) Violin plots of the channel-level SNR
(hot channel: the top 10 % high-count measurements; cold channel: the remaining
90 %). (b) Relative noise versus the ground-truth signal-intensity percentile.

![Fig. 16](DREAM_data/figs/fig16%20snr.png)

</details>

## Highlights

| Component | What it does | Where |
|---|---|---|
| **HCMM** | decouples the sinogram into hot / cold channels with threshold `tau = P90` | [`dream_hm/data/heatmask_dataset.py`](dream_hm/data/heatmask_dataset.py) |
| **Progressive hierarchical masking** | the training patch grows `64 -> 128 -> 192 -> 256`, so the network first learns local texture and then global structure | [`dream_hm/train_pipeline.py`](dream_hm/train_pipeline.py), `mini_batch_sizes` / `gt_sizes` |
| **Compact prior (HCP + CCP -> HCCP)** | a pixel-unshuffle CPEN encodes the 3-D block into a 256-d unified compact prior (UCP) | [`dream_hm/archs/DREAMHM_S1_arch.py`](dream_hm/archs/DREAMHM_S1_arch.py) |
| **Diffusion stage** | the UCP lives in a 4-step DDPM; the diffusion loss only trains the prior | [`ldm/ddpm.py`](ldm/ddpm.py), [`dream_hm/archs/DREAMHM_S2_arch.py`](dream_hm/archs/DREAMHM_S2_arch.py) |
| **UCP-guided transformer** | modulation (`x * k_v1 + k_v2`) inside the transposed attention and the gated FFN | [`dream_hm/archs/DREAMHM_S2_arch.py`](dream_hm/archs/DREAMHM_S2_arch.py) |
| **Fusion + OSEM** | channels are fused back to one sinogram, then OSEM/MLEM reconstructs the image | [`basicsr/models/sr_model.py`](basicsr/models/sr_model.py), [`reconstruction/recon_from_mat.m`](reconstruction/recon_from_mat.m) |

---

## Repository layout

```
DREAM-HM_code/
├── dream_hm/                     # the model
│   ├── archs/
│   │   ├── DREAMHM_S1_arch.py    # CPEN + stage-1 transformer (prior extraction)
│   │   ├── DREAMHM_S2_arch.py    # CPEN + denoise MLP + diffusion + transformer
│   │   ├── attention.py          # QKV attention used by the diffusion stage
│   │   └── common.py             # Conv / ResBlock helpers
│   ├── data/
│   │   ├── heatmask_dataset.py   # DREAM-HM : HCMM dual-channel blocks
│   │   └── randommask_dataset.py # DREAM-RM : random block masks
│   ├── losses/kd_loss.py         # KD loss between stage-1 and stage-2 priors
│   ├── models/                   # DREAMHM_S1Model / DREAMHM_S2Model (train loops)
│   ├── train_pipeline.py         # progressive hierarchical masking + EMA
│   ├── train.py, test.py         # entry points
├── basicsr/                      # vendored BasicSR (sinogram-adapted SRModel)
├── ldm/                          # DDPM / DDIM / beta schedules
├── options/                      # DREAM-HM configs (DREAM-RM under options/rm/)
├── scripts/                      # train_S1.sh / train_S2.sh / test.sh
├── reconstruction/               # MATLAB OSEM / MLEM (uses third_party/KER_v0.3)
├── tools/                        # HCMM visualisation, mat->png, recon wrapper, model check
├── third_party/KER_v0.3/         # KER / IRT toolbox (MATLAB, J. Fessler)
└── DREAM_data/figs/              # figures of the paper (Fig. 1-16)
```

---

## Installation

```bash
conda create -n dream-hm python=3.9 -y
conda activate dream-hm
pip install torch torchvision          # tested with torch 2.4.1 + cu121
pip install -r requirements.txt
```

`basicsr`, `ldm` and `third_party/KER_v0.3` are vendored, nothing else has to be
installed. All commands are run **from the repository root**:

```bash
cd DREAM-HM_code
```

Check that the networks build (no data, no GPU):

```bash
python tools/check_model.py          # prints the parameter counts of S1 and S2
# DREAMHM_S1: 26.74 M params, output (1, 3, 64, 64)
# DREAMHM_S2: 26.91 M params, output (1, 3, 64, 64)
```

Pretrained checkpoints are not shipped with the repository; download them into
[`weights/`](weights/README.md) and point the configs at them.

The MATLAB part needs MATLAB R2021a or newer (no toolbox required); the
reconstruction scripts add `third_party/KER_v0.3` to the path themselves.

---

## Training

DREAM-HM is trained in two stages.

```bash
# stage 1: compact-prior extraction network
bash scripts/train_S1.sh                       # options/train_DREAMHM_S1.yml

# stage 2: diffusion stage + transformer stage
#   -> first set pretrain_network_g / pretrain_network_S1 in the yml to
#      experiments_sinogram/DREAMHM_S1/models/net_g_latest.pth
bash scripts/train_S2.sh                       # options/train_DREAMHM_S2.yml
```

Key settings (identical to the paper):

| item | value |
|---|---|
| transformer levels | 4, heads `[1,2,4,8]`, channels `[48,96,192,384]`, blocks `[3,5,6,6]` |
| CPEN channels | 64 |
| diffusion steps | `T = 4`, `linear_start 0.1`, `linear_end 0.99` |
| optimiser | Adam, lr `2e-4`, `betas (0.9, 0.99)` |
| iterations | 400k total, the prior is fine-tuned for the first 100k |
| progressive patch | `64 -> 128 -> 192 -> 256` at iters `120k / 100k / 100k / 80k` |
| batch per GPU | `[8, 4, 2, 2]` (drop the batch size if you run out of memory) |

Checkpoints are written to `experiments_sinogram/<name>/models/`, TensorBoard
logs to `tb_logger/<name>/`. See [`docs/TRAIN.md`](docs/TRAIN.md) for resuming,
multi-GPU notes and hyper-parameter tuning.

---

## Testing and reconstruction

```bash
# 1) reconstruct the sinograms -> results_sinogram/DREAMHM_S2_test/visualization/*.mat
#    (set pretrain_network_g / pretrain_network_S1 in options/test_DREAMHM_S2.yml first)
bash scripts/test.sh

# 2) sinogram -> PET image with OSEM (MATLAB)
python tools/run_recon.py \
    --sino-dir results_sinogram/DREAMHM_S2_test/visualization \
    --out-dir  results_sinogram/DREAMHM_S2_test/images \
    --method osem --nblock 12 --niter 6
```

or from MATLAB directly:

```matlab
cd reconstruction
recon_from_mat('../results_sinogram/DREAMHM_S2_test/visualization', ...
               '../results_sinogram/DREAMHM_S2_test/images', 'method', 'osem');
```

The test log reports PSNR / SSIM / MSE of the fused sinogram and the average
reconstruction time per slice; the reconstructed images are stored as both
`<name>_rec.mat` and `<name>_rec.png`. Use `--method mlem` (all angles per
update) for the plain MLEM variant of Eq. (13).

Convert reconstructed images to PNG with

```bash
python tools/mat2png.py --root results_sinogram/DREAMHM_S2_test/images --key rec
```

---

## Relation between the paper and the code

| Paper | Code |
|---|---|
| HCMM, Eqs. (3)–(4) | `dream_hm/data/heatmask_dataset.py` (`estimate_tau`, `hcmm_split`) |
| threshold `tau`, hot ratio 10 % | dataset option `hot_ratio: 0.1`: `tau = P90` of the padded 256 x 256 block, i.e. ~13 % of the 192-column sinogram |
| progressive hierarchical masking, Eq. (5) | `train_pipeline.py` + `mini_batch_sizes` / `gt_sizes` |
| HCP / CCP -> HCCP / UCP | `archs/DREAMHM_S1_arch.py::CPEN` |
| diffusion stage, Eqs. (9)–(10) | `ldm/ddpm.py`, `archs/DREAMHM_S2_arch.py::denoise` |
| UCP-guided attention / FFN, Eqs. (6)–(8) | `archs/DREAMHM_S2_arch.py::Attention`, `FeedForward` |
| reverse diffusion, Eq. (11) | `models/DREAMHM_S2_model.py::test` |
| weighted fusion, Eq. (12) | `basicsr/models/sr_model.py::_fuse_channels` |
| MLEM / OSEM update, Eq. (13) | `reconstruction/recon_from_mat.m` |

Two implementation notes:

1. **Channel order.** The dual-channel block is stored as `[cold, hot]`, i.e.
   channel 0 is `S * M_cold` and channel 1 is `S * M_hot`. The threshold is
   estimated from the *noise-free* sinogram so that the LQ/GT pair always
   shares exactly the same partition.
2. **Data-fidelity channel.** The reference checkpoints were trained with a
   three-channel block `[cold, hot, full]`: the extra unmasked sinogram keeps
   the observed data in the network and is used to fill the masked positions
   before fusion, which is the `N + 1` channel construction of Fig. 1. Set
   `clean_channel: false` (and `inp_channels` / `out_channels: 2`) to obtain the
   strictly two-channel block of Eq. (4).

---

## DREAM-RM baseline

The random-mask variant uses exactly the same networks and training schedule,
only the dataset changes (16 x 16 blocks, 10 % of the area, two masked channels
plus the unmasked one):

```bash
bash scripts/train_S1.sh options/rm/train_DREAMRM_S1.yml
bash scripts/train_S2.sh options/rm/train_DREAMRM_S2.yml
bash scripts/test.sh     options/rm/test_DREAMRM_S2.yml
```

## Results

Quantitative comparison on the test set (Table I of the paper):

| Method | PSNR ↑ | SSIM ↑ | RMSE ↓ |
|---|---|---|---|
| MLEM | 21.12 | 0.801 | 0.0910 |
| DDPM-PET | 29.55 | 0.911 | 0.0352 |
| NCSN++ | 30.41 | 0.936 | 0.0334 |
| IR-SDE | 33.22 | 0.950 | 0.0231 |
| CoreDiff | 32.41 | 0.936 | 0.0251 |
| CPM-IRM | 32.56 | 0.952 | 0.0265 |
| DREAM-RM | 34.57 | 0.955 | 0.0191 |
| **DREAM-HM** | **36.32** | **0.978** | **0.0162** |

---

## Other Related Projects

* ALL-PET: A Low-resource and Low-shot PET Foundation Model in Projection Domain  [<font size=5>**[Paper]**</font>](https://github.com/yqx7150/RAYSOLUTION_PETdata/blob/main/Paper/ALL_PET_Finalx.pdf)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/ALL-PET)

* Diffusion Transformer Model with Compact Prior for Low-dose PET Reconstruction [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2407.00944)     [<font size=5>**[Code]**</font>](https://github.com/yqx7150/dtm)

* RED: Residual Estimation Diffusion for Low-Dose PET Sinogram Reconstruction  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1361841525001057)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/RED)    

* Double-Constraint Diffusion Model with Nuclear Regularization for Ultra-low-dose PET Reconstruction  [<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2509.00395)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DCDM)    
    
* Raysolution_PET_Data [<font size=5>**[Data]**</font>](https://github.com/yqx7150/Raysolution_PET_Data)   
     
* Temporal Image Sequence Separation in Dual-tracer Dynamic PET with an Invertible Network  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10542421)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DTS-INN)        
    
* Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10980366)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/st-DTPM)    

* PET Tracer Separation using Conditional Diffusion Transformer with Multi-latent Space Learning [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2506.16934#:~:text=In%20this%20study%2C%20a%20multi-latent%20space%20guided%20texture,model%20%28MS-CDT%29%20is%20proposed%20for%20PET%20tracer%20separation.)

* Synthetic CT Generation via Variant Invertible Network for Brain PET Attenuation Correction [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10666843) [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)     
    
* A Prior-Guided Joint Diffusion Model in Projection Domain for PET Tracer Conversion [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2506.16733) [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PJDM)    

* Positron Emission Tomography Tracer Conversion via Variable Augmented Invertible Network [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s12204-025-2844-2)    

## Citation

```bibtex
@article{huang2026dream,
  title   = {Diffusion Transformer Meets Heat Masks: An Advanced {PET} Reconstruction Framework},
  author  = {Huang, Bin and Chen, Yanhan and He, Binzhong and Liu, Zhili and Wang, Xinyue and Li, Binxuan and Liu, Qiegen},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```

## Acknowledgements

This code builds on [BasicSR](https://github.com/XPixelGroup/BasicSR),
[DiffIR](https://github.com/Zj-BinXia/DiffIR), [SwinIR](https://github.com/JingyunLiang/SwinIR),
[latent-diffusion](https://github.com/CompVis/latent-diffusion) and the
KER / IRT toolbox of J. Fessler. We thank the authors for releasing their code.

## License

The code of this repository is released under the MIT license (see `LICENSE`).
The vendored third-party components keep their original licenses — see
[`third_party/README.md`](third_party/README.md).