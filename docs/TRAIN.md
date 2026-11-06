# Training notes

## Two stages

| stage | entry point | config | what is trained |
|---|---|---|---|
| 1 | `scripts/train_S1.sh` | `options/train_DREAMHM_S1.yml` | the compact-prior extraction network (CPEN) with an L1 loss on the 3-D sinogram block |
| 2 | `scripts/train_S2.sh` | `options/train_DREAMHM_S2.yml` | the diffusion stage (KD loss against the stage-1 prior) and then the UCP-guided transformer stage (L1 + KD) |

Stage 2 starts from the stage-1 checkpoint: its generator is initialised with
`pretrain_network_g` (`strict_load_g: false`, so only the matching tensors are
loaded) and the frozen prior network is loaded from `pretrain_network_S1`.

```yaml
path:
  pretrain_network_g: experiments_sinogram/DREAMHM_S1/models/net_g_latest.pth
  pretrain_network_S1: experiments_sinogram/DREAMHM_S1/models/net_g_latest.pth
  strict_load_g: false
  ignore_resume_networks: network_S1
```

## Progressive hierarchical masking

`dream_hm/train_pipeline.py` reads `iters`, `mini_batch_sizes`, `gt_sizes` and
`gt_size` from the dataset section and grows the patch size during training:

| iterations | patch | batch / GPU |
|---|---|---|
| 1 – 120,000 | 64 x 64 | 8 |
| 120,001 – 220,000 | 128 x 128 | 4 |
| 220,001 – 320,000 | 192 x 192 | 2 |
| 320,001 – 400,000 | 256 x 256 | 2 |

The patch is cropped from the padded `256 x 256` block that the dataset
returns, so `gt_size` must be the largest patch and equal to
`sino_h x (sino_w + 2 * pad_w)`.

## Losses and schedule

* stage 1: L1 on the reconstructed block, `CosineAnnealingRestartCyclicLR`
  (`periods [120000, 280000]`), 400 k iterations.
* stage 2: for the first `encoder_iter = 100000` iterations only the prior
  network is updated with the KD loss (`temperature 0.15`); afterwards the
  transformer is trained with `L1 + KD`, `MultiStepLR` (`milestones [300000]`,
  `gamma 0.5`), 400 k iterations in total, EMA decay `0.999`.

## Resuming

`resume_state` restores weights **and** optimiser / scheduler / iteration
count, `pretrain_network_g` only loads weights. Use the former to continue an
interrupted run:

```yaml
path:
  resume_state: experiments_sinogram/DREAMHM_S2/training_states/latest.state
```

Use the latter to fine-tune from a checkpoint with a fresh schedule:

```yaml
path:
  resume_state: ~
  pretrain_network_g: experiments_sinogram/DREAMHM_S2/models/net_g_300000.pth
```

When resuming, keep `name`, the network definition and the batch settings
unchanged, otherwise the state cannot be restored.

## Multi-GPU and memory

* `num_gpu: auto` uses every visible GPU; the scripts use `torchrun` with
  `NUM_GPU` processes (`NUM_GPU=4 bash scripts/train_S2.sh`).
* A single-GPU run without `torchrun` also works:
  `python dream_hm/train.py -opt options/train_DREAMHM_S2.yml` (the model falls
  back to the non-distributed network and the plain dataloader).
* If you run out of memory, scale `batch_size_per_gpu` and
  `mini_batch_sizes` down by the same factor (e.g. `[4,2,1,1]`); the
  progressive patch sizes can stay unchanged.
* `num_worker_per_gpu` (default 8) is the dataloader parallelism; each worker
  loads and masks one `.mat` file at a time.

## Outputs

```
experiments_sinogram/<name>/
├── models/net_g_<iter>.pth        # params + params_ema
├── training_states/<iter>.state   # optimiser / scheduler / iter
├── <name>.yml                     # a copy of the config used
└── train_<name>_<time>.log
tb_logger/<name>/                  # TensorBoard
results_sinogram/<name>/visualization/   # test: reconstructed sinograms
```

## Troubleshooting

| symptom | fix |
|---|---|
| `yaml` cannot find the config | run from the repository root, paths in the yml are relative to it |
| `No .mat sinogram found in ...` | check `dataroot_GT`; files must end with `.mat` |
| `CUDA out of memory` | reduce `batch_size_per_gpu` / `mini_batch_sizes` |
| `KeyError: 'sino_net1'` in MATLAB | the test stage did not finish / the folder is not `.../visualization` |
| TensorBoard import error | `pip install tensorboard` or set `logger.use_tb_logger: false` |