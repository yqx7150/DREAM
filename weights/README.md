# Pretrained weights

The checkpoints of the paper are **not** stored in this repository (they are
~100 MB each). Download them and place them here, or use your own trained
models.

| model | file | where it is used |
|---|---|---|
| DREAM-HM stage 1 | `DREAMHM_S1.pth` | `pretrain_network_g` / `pretrain_network_S1` of the stage-2 config |
| DREAM-HM stage 2 | `DREAMHM_S2.pth` | `pretrain_network_g` of `options/test_DREAMHM_S2.yml` |

Then point the configs at the files:

```yaml
path:
  pretrain_network_g: weights/DREAMHM_S2.pth
  pretrain_network_S1: weights/DREAMHM_S1.pth
  param_key_g: params_ema
  strict_load_g: false
```

`param_key_g: params_ema` selects the EMA weights (recommended); use
`params` to evaluate the raw training weights instead.