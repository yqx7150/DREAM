# DREAM

**扩散 Transformer 遇上热掩码 —— 用物理引导的“热-冷”掩码重建 PET 正弦图。**

本仓库是 **DREAM**（*Diffusion tRansformer mEets heAt Masks*）的官方实现。`main` 分支以热掩码版本 **DREAM-HM** 为主，随机掩码版本 **DREAM-RM** 作为可复现的对比方法保留（`options/rm/`）。上一版（仅 DREAM-RM）的代码仍在 git 历史中（截至 `6a1e35a`）。对应论文

> **Diffusion Transformer Meets Heat Masks: An Advanced PET Reconstruction Framework**
> Bin Huang, Yanhan Chen, Binzhong He, Zhili Liu, Xinyue Wang, Binxuan Li, Qiegen Liu
> *IEEE Transactions on Image Processing*

DREAM 把掩码机制同时引入**正弦图域**与**隐空间**：用高维堆叠扩展解空间，从三维数据块中提取紧凑先验（compact prior）驱动扩散过程，再由扩散 Transformer 重建正弦图，最后用有序子集期望最大化（OSEM/MLEM）得到 PET 图像。

* **DREAM-RM**（随机掩码）：随机块掩码，保留一个未被掩码的通道用于数据保真。
* **DREAM-HM**（热掩码，本文方法）：按全局强度阈值把正弦图解耦成**高计数热通道**与**低计数冷通道**，并用**冷热互掩码（HCMM）**策略让二者互补耦合。

---

## 论文插图

以下插图均取自 TIP 论文定稿（*Diffusion Transformer Meets Heat Masks: An Advanced PET Reconstruction Framework*）。

**图 1 — DREAM 框架。** 传统方法与 DREAM 框架对比：DREAM-RM 用随机掩码 + 渐进分层掩码构造三维正弦图数据块；DREAM-HM 改用物理引导的 **HCMM** 策略，把正弦图显式解耦成高计数（热）与低计数（冷）两个通道。

![图 1 — DREAM 框架](DREAM_data/figs/fig1%20framework.png)

**图 2 — 训练流程。** 框架由掩码生成、含扩散阶段的掩码驱动隐空间、以及 Transformer U-Net 组成。图中给出两条并行的先验生成路径：上方为 DREAM-HM（HCMM + 渐进分层掩码 → HCCP），下方为 DREAM-RM（随机掩码 → SMCP），提取出的先验引导 Transformer 阶段重建。

![图 2 — 训练流程](DREAM_data/figs/fig2%20training%20procedure.png)

**图 3 — 重建流程。** 同时支持两种数据准备策略（DREAM-RM 为 `N` 通道，DREAM-HM 为热/冷双通道）。UCP 经反向扩散细化后引导 Transformer 阶段，各通道重建结果融合为单一正弦图，最后由 MLEM 得到 PET 图像。

![图 3 — 重建流程](DREAM_data/figs/fig3%20reconstruction%20procedure.png)

**图 4 — PET 图像重建结果。** (a)-(i) 为各对比方法与 DREAM，(j) 为 GT；第二行为相对 GT 的残差。

![图 4 — PET 图像重建结果](DREAM_data/figs/fig4%20pet%20images.jpg)

**图 5 — PET 正弦图重建结果。** (a)-(i) 为各对比方法与 DREAM，(j) 为 GT 正弦图；第二行为残差。

![图 5 — PET 正弦图重建结果](DREAM_data/figs/fig5%20sinograms.jpg)

<details>
<summary><b>更多插图（图 6-16）：剖面线、低剂量、三维仿真与消融实验</b></summary>

**图 6 — 图像剖面线。** GT 与九种重建方法的 PET 图像剖面线对比。

![图 6](DREAM_data/figs/fig6%20line%20profiles%20image.png)

**图 7 — 正弦图剖面线。** GT、低质量输入与八种重建方法的正弦图剖面线对比。

![图 7](DREAM_data/figs/fig7%20line%20profiles%20sinogram.png)

**图 8 — 低剂量重建。** 临床剂量 1%、5%、10%、20% 下的低剂量 PET 重建，图像域 (A)-(I) 与正弦图域 (a)-(i)；第二、四行为绝对残差。

![图 8](DREAM_data/figs/fig8%20dose%20levels.jpg)

**图 9 — 三维仿真。** 三维 PET 数据采样几何与采样结果。

![图 9](DREAM_data/figs/fig9%203d%20sampling.png)

**图 10 — 三维重建结果。** DREAM-HM 的三维 PET 重建结果。

![图 10](DREAM_data/figs/fig10%203d%20results.png)

**图 11 — 扩散步数。** 不同扩散步数下重建正弦图的 PSNR / SSIM / RMSE。

![图 11](DREAM_data/figs/fig11%20diffusion%20steps.png)

**图 12 — 掩码策略。** 不同掩码策略的 PSNR / SSIM / RMSE 箱线图。

![图 12](DREAM_data/figs/fig12%20mask%20algorithms.png)

**图 13 — 掩码层级。** 不同掩码层级的 PSNR / SSIM / RMSE 箱线图。

![图 13](DREAM_data/figs/fig13%20mask%20levels.png)

**图 14 — 通道数。** 不同堆叠通道数的 PSNR / SSIM / RMSE 箱线图。

![图 14](DREAM_data/figs/fig14%20channel%20numbers.png)

**图 15 — 阈值 `tau`。** DREAM-HM 在不同 `tau` 下的重建结果，第二行为残差。

![图 15](DREAM_data/figs/fig15%20tau.png)

**图 16 — 冷热通道 SNR 分析。** (a) 通道级 SNR 小提琴图（热通道为前 10% 高计数测量，冷通道为其余 90%）；(b) 相对噪声随 GT 信号强度百分位的变化。

![图 16](DREAM_data/figs/fig16%20snr.png)

</details>

## 主要模块

| 模块 | 作用 | 位置 |
|---|---|---|
| **HCMM** | 用阈值 `tau = P90` 把正弦图拆成热/冷两个通道 | [`dream_hm/data/heatmask_dataset.py`](dream_hm/data/heatmask_dataset.py) |
| **渐进分层掩码** | 训练块从 `64` 增长到 `128/192/256`，先学局部纹理再学全局结构 | [`dream_hm/train_pipeline.py`](dream_hm/train_pipeline.py) |
| **紧凑先验（HCP + CCP → HCCP）** | PixelUnshuffle + CPEN 把三维块编码成 256 维统一紧凑先验（UCP） | [`dream_hm/archs/DREAMHM_S1_arch.py`](dream_hm/archs/DREAMHM_S1_arch.py) |
| **扩散阶段** | UCP 在 4 步 DDPM 中训练，扩散损失只更新先验 | [`ldm/ddpm.py`](ldm/ddpm.py)、[`dream_hm/archs/DREAMHM_S2_arch.py`](dream_hm/archs/DREAMHM_S2_arch.py) |
| **UCP 引导的 Transformer** | 在转置注意力与门控 FFN 中做仿射调制（`x * k_v1 + k_v2`） | [`dream_hm/archs/DREAMHM_S2_arch.py`](dream_hm/archs/DREAMHM_S2_arch.py) |
| **通道融合 + OSEM** | 多通道加权融合回二维正弦图，再用 OSEM/MLEM 重建图像 | [`basicsr/models/sr_model.py`](basicsr/models/sr_model.py)、[`reconstruction/recon_from_mat.m`](reconstruction/recon_from_mat.m) |

---

## 目录结构

```
DREAM-HM_code/
├── dream_hm/                     # 模型主体
│   ├── archs/
│   │   ├── DREAMHM_S1_arch.py    # CPEN + 第一阶段 Transformer（先验提取）
│   │   ├── DREAMHM_S2_arch.py    # CPEN + denoise MLP + 扩散 + Transformer
│   │   ├── attention.py          # 扩散阶段使用的 QKV attention
│   │   └── common.py             # Conv / ResBlock
│   ├── data/
│   │   ├── heatmask_dataset.py   # DREAM-HM：HCMM 双通道数据块
│   │   └── randommask_dataset.py # DREAM-RM：随机块掩码数据块
│   ├── losses/kd_loss.py         # 一、二阶段先验之间的 KD 损失
│   ├── models/                   # DREAMHM_S1Model / DREAMHM_S2Model
│   ├── train_pipeline.py         # 渐进分层掩码 + EMA
│   ├── train.py, test.py         # 入口
├── basicsr/                      # 内置 BasicSR（已适配正弦图）
├── ldm/                          # DDPM / DDIM / beta schedule
├── options/                      # DREAM-HM 配置（DREAM-RM 在 options/rm/）
├── scripts/                      # train_S1.sh / train_S2.sh / test.sh
├── reconstruction/               # MATLAB OSEM / MLEM（调用 third_party/KER_v0.3）
├── tools/                        # HCMM 可视化、mat→png、重建封装、模型自检
├── third_party/KER_v0.3/         # KER / IRT 工具箱（MATLAB）
└── DREAM_data/figs/              # 论文插图（图 1-16）
```

---

## 环境安装

```bash
conda create -n dream-hm python=3.9 -y
conda activate dream-hm
pip install torch torchvision          # 论文使用 torch 2.4.1 + cu121
pip install -r requirements.txt
```

`basicsr`、`ldm`、`third_party/KER_v0.3` 均已内置，无需另外安装。所有命令都在仓库根目录执行：

```bash
cd DREAM-HM_code
```

可先做一次网络构建自检（不需要数据、不需要 GPU）：

```bash
python tools/check_model.py          # 打印 S1 / S2 的参数量
# DREAMHM_S1: 26.74 M params, output (1, 3, 64, 64)
# DREAMHM_S2: 26.91 M params, output (1, 3, 64, 64)
```

预训练权重不随仓库发布，请下载后放到 [`weights/`](weights/README.md)，并在配置里指向对应文件。

MATLAB 部分需要 R2021a 及以上版本（不需要额外工具箱），脚本会自动把 `third_party/KER_v0.3` 加入路径。

---

## 训练

DREAM-HM 分两阶段训练：

```bash
# 第一阶段：紧凑先验提取网络
bash scripts/train_S1.sh                      # options/train_DREAMHM_S1.yml

# 第二阶段：扩散阶段 + Transformer 阶段
#   先把 yml 里的 pretrain_network_g / pretrain_network_S1 指向
#   experiments_sinogram/DREAMHM_S1/models/net_g_latest.pth
bash scripts/train_S2.sh                      # options/train_DREAMHM_S2.yml
```

关键参数（与论文一致）：

| 项目 | 取值 |
|---|---|
| Transformer | 4 级，heads `[1,2,4,8]`，通道 `[48,96,192,384]`，块数 `[3,5,6,6]` |
| CPEN 通道数 | 64 |
| 扩散步数 | `T = 4`，`linear_start 0.1`，`linear_end 0.99` |
| 优化器 | Adam，lr `2e-4`，`betas (0.9, 0.99)` |
| 迭代次数 | 共 400k，其中前 100k 只微调先验网络 |
| 渐进块大小 | `64 → 128 → 192 → 256`，对应迭代 `120k / 100k / 100k / 80k` |
| 批大小 | `[8, 4, 2, 2]`（显存不足时同比例调小） |

权重保存在 `experiments_sinogram/<name>/models/`，TensorBoard 日志在 `tb_logger/<name>/`。断点续训、多卡训练等说明见 [`docs/TRAIN.md`](docs/TRAIN.md)。

---

## 测试与重建

```bash
# 1) 重建正弦图 -> results_sinogram/DREAMHM_S2_test/visualization/*.mat
#    先修改 options/test_DREAMHM_S2.yml 中的 pretrain_network_g / pretrain_network_S1
bash scripts/test.sh

# 2) 正弦图 -> PET 图像（OSEM，MATLAB）
python tools/run_recon.py \
    --sino-dir results_sinogram/DREAMHM_S2_test/visualization \
    --out-dir  results_sinogram/DREAMHM_S2_test/images \
    --method osem --nblock 12 --niter 6
```

也可以直接在 MATLAB 里：

```matlab
cd reconstruction
recon_from_mat('../results_sinogram/DREAMHM_S2_test/visualization', ...
               '../results_sinogram/DREAMHM_S2_test/images', 'method', 'osem');
```

测试日志会给出融合后正弦图的 PSNR / SSIM / MSE 以及每层的平均重建时间；重建图像同时保存为 `<name>_rec.mat` 与 `<name>_rec.png`。把 `--method` 换成 `mlem` 即为公式 (13) 的 MLEM 版本。

---

## 论文与代码的对应关系

| 论文 | 代码 |
|---|---|
| HCMM，公式 (3)(4) | `dream_hm/data/heatmask_dataset.py`（`estimate_tau`、`hcmm_split`） |
| 阈值 `tau`、热区比例 10% | 数据配置项 `hot_ratio: 0.1`：取补零后 256×256 块的 P90，对应 192 列正弦图上约 13% |
| 渐进分层掩码，公式 (5) | `train_pipeline.py` 与配置里的 `mini_batch_sizes` / `gt_sizes` |
| HCP / CCP → HCCP / UCP | `archs/DREAMHM_S1_arch.py::CPEN` |
| 扩散阶段，公式 (9)(10) | `ldm/ddpm.py`、`archs/DREAMHM_S2_arch.py::denoise` |
| UCP 引导的注意力/FFN，公式 (6)-(8) | `archs/DREAMHM_S2_arch.py::Attention`、`FeedForward` |
| 反向扩散，公式 (11) | `models/DREAMHM_S2_model.py::test` |
| 加权融合，公式 (12) | `basicsr/models/sr_model.py::_fuse_channels` |
| MLEM / OSEM 迭代，公式 (13) | `reconstruction/recon_from_mat.m` |

两点实现说明：

1. **通道顺序**：双通道数据块存成 `[cold, hot]`，即通道 0 为 `S * M_cold`、通道 1 为 `S * M_hot`。阈值在**无噪**正弦图上估计，保证 LQ/GT 使用完全相同的划分。
2. **数据保真通道**：随仓库发布的权重是用三通道块 `[cold, hot, full]` 训练的——多出的未掩码正弦图用于保留观测数据，并在融合前回填被掩码的位置，即图 1 中的 `N + 1` 通道结构。把 `clean_channel` 设为 `false`（同时把 `inp_channels` / `out_channels` 改为 `2`）即得到公式 (4) 的严格双通道版本。

---

## DREAM-RM 对比方法

随机掩码版本使用完全相同的网络与训练流程，只更换数据管线（16×16 块、覆盖 10% 面积、两个掩码通道加一个未掩码通道）：

```bash
bash scripts/train_S1.sh options/rm/train_DREAMRM_S1.yml
bash scripts/train_S2.sh options/rm/train_DREAMRM_S2.yml
bash scripts/test.sh     options/rm/test_DREAMRM_S2.yml
```

## 实验结果

测试集上的定量比较（论文 Table I）：

| 方法 | PSNR ↑ | SSIM ↑ | RMSE ↓ |
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

## 引用

```bibtex
@article{huang2026dream,
  title   = {Diffusion Transformer Meets Heat Masks: An Advanced {PET} Reconstruction Framework},
  author  = {Huang, Bin and Chen, Yanhan and He, Binzhong and Liu, Zhili and Wang, Xinyue and Li, Binxuan and Liu, Qiegen},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```

## 致谢

本代码基于 [BasicSR](https://github.com/XPixelGroup/BasicSR)、[DiffIR](https://github.com/Zj-BinXia/DiffIR)、[SwinIR](https://github.com/JingyunLiang/SwinIR)、[latent-diffusion](https://github.com/CompVis/latent-diffusion) 以及 J. Fessler 的 KER / IRT 工具箱，感谢原作者开源。

## 许可证

本仓库代码采用 MIT 许可证（见 `LICENSE`）。内置的第三方组件遵循其原始许可证，详见 [`third_party/README.md`](third_party/README.md)。