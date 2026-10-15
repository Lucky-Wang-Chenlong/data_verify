# data_verify

通过 finetune 来检验**自己生成的 3D 检测数据是否正确**。

做法：以 [WildDet3D](https://github.com/allenai/WildDet3D) 发布的 checkpoint 为起点，把标准数据（Omni3D 子集）和你的数据按一定比例混合后 finetune，然后看目标类别有没有变好、标准测试集有没有退化。本仓库只放自己的代码，不改 WildDet3D 的源码。

```
1. 安装环境          ./setup_env.sh
2. 下载标准数据集     scripts/download_data.sh [DATA_DIR]
3. 准备自己的数据集   按 dv/datasets/README.md 的格式放进一个目录，用 scripts/validate.sh 校验
4. 启动训练          scripts/train.sh --custom <数据集目录> --ratio <混合比例>
5. 查看输出          --out 目录：checkpoints/、metrics/（loss 等）、results/（AP）
```

---

## 1. 安装环境

```bash
mkdir -p <workspace> && cd <workspace>     # 工作区：WildDet3D、环境、数据和输出都放在这里
git clone <本仓库地址> data_verify
cd data_verify
./setup_env.sh                             # 日志写到 <workspace>/envs/setup_env.log
```

之后所有命令都在 `data_verify/` 目录下执行。

这一步会做三件事：
- 在本仓库旁边克隆 WildDet3D，包括它的 submodule，并把 MoGe 固定到兼容的版本；
- 用 micromamba 建环境 `../envs/wilddet3d`，环境里自带 CUDA 12.1、torch 2.5.1 和 vis4d 1.0.0；
- 下载发布的 checkpoint 和 LingBot-Depth 权重。

SAM3 的权重不用下载，发布的 checkpoint 里已经包含全部权重。

下载缓存（micromamba、pip、HuggingFace、torch）默认放在 `<workspace>/.cache/`。如果环境里已经设置了 `MAMBA_ROOT_PREFIX`、`PIP_CACHE_DIR`、`HF_HOME` 或 `TORCH_HOME`，就沿用已有的值。PATH 里已经有 `micromamba` 时会直接使用，否则下载到 `<workspace>/.cache/bin/`。

硬件和环境要求：Linux 和 NVIDIA GPU，系统不需要预装 CUDA。`vis4d_cuda_ops` 默认按 H100 编译（`TORCH_CUDA_ARCH_LIST=9.0`）；其他型号的 GPU 要在运行前设置这个变量，例如 A100 用 `TORCH_CUDA_ARCH_LIST=8.0 ./setup_env.sh`。

## 2. 下载标准数据集

```bash
scripts/download_data.sh                      # 默认下载到 ../data
scripts/download_data.sh /path/to/data        # 也可以指定目录
```

下载过程中最多占用约 25 GB，完成后约 13 GB。已经完成的步骤会自动跳过，中断后重跑会接着下载。

| 内容 | 路径（相对 DATA_DIR） | 用途 |
|---|---|---|
| SUNRGBD train + val（可用 4,547 张） | `SUNRGBD.hdf5`、`omni3d/annotations/SUNRGBD_{train,val}.json` | 训练 |
| ARKitScenes train 子集（seed 0 抽 5k 张，可用 4,146 张） | `ARKitScenes.hdf5`、`ARKitScenes_depth.hdf5`、`omni3d/annotations/ARKitScenes_train_sub5k.json` | 训练 |
| SUNRGBD test（5,050 张） | `SUNRGBD.hdf5`、`omni3d/annotations/SUNRGBD_test.json` | 遗忘检查、论文复现 |
| toy 自定义数据集：SUNRGBD_val 里的 chair（训练 161 张，测试 53 张） | `custom/toy/` | 格式示例、冒烟测试 |

脚本最后会创建软链接 `WildDet3D/data -> DATA_DIR`。WildDet3D 上游代码按 `data/` 这个固定的相对路径读取 Omni3D，所以所有脚本都靠这个链接找到数据。如果以后要把数据搬到别处，重新执行一次 `scripts/download_data.sh <新目录>` 就会更新这个链接。

## 3. 准备自己的数据集

数据格式见 **[dv/datasets/README.md](dv/datasets/README.md)**。目录结构如下：

```
/path/to/mycat/
├── annotations/MyCat_train.json
├── annotations/MyCat_test.json
├── images/...
└── depth/...
```

先校验数据。校验结果里所有 ERROR 都要改掉，然后看一下渲染出来的 GT 框是否贴合物体：

```bash
scripts/validate.sh --custom /path/to/mycat   # 渲染图在 ../outputs/data_verify/vis/<split>/
```

再做 overfit 检查：只用你的训练集训练，并在训练集上评估，用时约 20 分钟。AP 应该明显、持续地上升；如果 AP 一直不动，说明标注之间不一致：

```bash
scripts/train.sh --custom /path/to/mycat --overfit
```

## 4. 启动训练

```bash
scripts/train.sh --custom /path/to/mycat --ratio 0.2 --out ../outputs/data_verify/mycat_r0.2
```

`--ratio` 是每个 batch 中来自你的数据的比例，其余部分来自标准数据。

| `--ratio` | 含义 |
|---|---|
| `0` | 只用标准数据，作为对照组（A1） |
| `0.2` | 80% 标准数据 + 20% 你的数据（A2，默认值） |
| `1` | 只用你的数据 |

不同 ratio 的 run 之间，每个 epoch 的样本数、epoch 数、学习率调度和 seed 都相同，所以结果的差别只来自数据。

**常用参数**（完整列表见 `scripts/train.sh -h`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gpus N` | 8 | GPU 数量 |
| `--epochs N` | 3 | 在 2/3 处和最后各把学习率乘 0.1 |
| `--samples N` | 8000 | 每个 epoch 的总样本数。8 卡时为 250 步/epoch，约 15 分钟 |
| `--lr X` | 1e-4 | 预训练部分用 0.1 倍 |
| `--train-split` / `--test-split` | 自动识别 | 用于 `annotations/` 下有多个候选文件时 |
| `--depth-scale S` | 1000 | 16-bit PNG depth 每米对应的数值（1000 表示毫米） |
| `--no-val-omni3d` | | 每个 epoch 不在 SUNRGBD_test 上验证，每轮省 3–4 分钟 |
| `--no-eval` | | 训练完不跑最终评估 |

训练结束后，脚本会自动用 box 和 text 两种 prompt 评估 `last.ckpt`，并导出各项指标。

**一次跑完整组对比**：zero-shot 基线，加上每个 ratio 的训练和评估，最后汇总成一张表：

```bash
nohup scripts/compare.sh --custom /path/to/mycat --ratios "0 0.2" > compare.log 2>&1 &
```

**单独评估某个 checkpoint**：

```bash
scripts/eval.sh --ckpt released --custom /path/to/mycat --prompt box   # zero-shot（A0）
scripts/eval.sh --ckpt <run>/checkpoints/last.ckpt --custom /path/to/mycat --prompt text
```

`--prompt box` 用 GT 2D 框作为 prompt，只考察 3D 框估计，最直接地反映 3D 标注的质量；`--prompt text` 只给类别名，模型需要自己先把物体找出来。加 `--depth` 表示把 GT depth 也输入给模型。

## 5. 输出

每次运行的所有产物都放在 `--out` 目录下。默认是 `../outputs/data_verify/<数据集名>_ratio<R>`，可以用环境变量 `DV_OUT` 修改这个父目录。

```
<out>/
├── checkpoints/
│   ├── last.ckpt               最终权重（8.7 GB）
│   └── epoch=N-step=M.ckpt     最近一个 epoch 的权重
├── metrics/
│   ├── train.csv               每 10 步一行：总 loss、各项 loss 和学习率
│   └── val.csv                 每个 epoch 一行：AP_<测试集>、AP15/25/50、ATE/ASE/AOE、ODS 等
├── results/
│   ├── summary.md              最终 AP 表（box / text × 每个测试集）
│   ├── summary.json            所有最终评估指标
│   ├── box/                    box prompt 评估：eval.log、results.json、log_*.txt（含逐类 AP）
│   └── text/                   text prompt 评估
├── logs/train.log              训练的完整输出
├── log_*.txt                   vis4d 自己的日志（含每个 epoch 的逐类 AP 表）
├── config_*.yaml               本次运行的完整配置
└── events.out.tfevents.*       tensorboard 日志（tensorboard --logdir <out>）
```

`compare.sh` 的输出目录里，每个 run 各占一个子目录，另外还有一个汇总表 `compare.md`：

```
| run        | prompt | MyCat_test | SUNRGBD_test |
| zeroshot_box | ...  |            |              |   ← A0
| ratio0     | box    |            |              |   ← A1（对照组）
| ratio0.2   | box    |            |              |   ← A2
```

**判断标准**：如果目标测试集（`AP_MyCat_test`）上 ratio0.2 明显高于 ratio0，同时 `AP_SUNRGBD_test` 两者持平，说明数据是对的，模型能从中学到东西。

想重新导出指标或者对比任意几个 run，可以直接用：

```bash
source scripts/env.sh
python $DV/tools/export_metrics.py run <out>
python $DV/tools/export_metrics.py compare <run1> <run2> ... --to table.md
```

---

## 已验证的结果（发布的 checkpoint，8×H100）

**论文复现**：用仓库里的官方 eval 配置（`scripts/repro_omni3d.sh`）在 SUNRGBD_test 上评估，单位为 AP（%）：

| 模式 | 论文 | 官方配置（bf16 / fp32） | 本仓库 eval 配置 |
|---|---|---|---|
| text | 38.9 | 38.67 / – | **38.67** |
| box | 43.1 | 41.40 / 40.85 | **41.39** |
| text + depth | 51.1 | 53.21 / 53.15 | – |
| box + depth | 58.7 | 59.84 / – | – |

和论文的差距在 −1.7 到 +2.1 之间，有正有负，也不是计算精度造成的。本仓库的 eval 配置和官方配置的结果一致。

**toy 数据集**：
- zero-shot：box AP 0.836，text AP 0.760。这两个数偏高，因为发布的模型很可能见过这些图。正式实验时应该选模型没见过、或者 zero-shot 表现差的类别。
- overfit：10 个 epoch，AP 从 0.57 升到 0.81。这组数是在修正 GT 内参之前测的，绝对值偏低，但上升趋势有效。

---

## 目录结构

```
<workspace>/             # 工作区，可以给多个 codebase 共用
├── WildDet3D/          上游仓库，未改动（ckpt/、pretrained/、data -> DATA_DIR）
├── envs/wilddet3d/     环境
├── data/               标准数据（默认的 DATA_DIR）
├── outputs/data_verify/  本仓库的默认输出目录
└── data_verify/        ← 本仓库
    ├── setup_env.sh
    ├── scripts/
    │   ├── env.sh          被其他脚本 source：设置路径和 PYTHONPATH，然后 cd 进 WildDet3D/
    │   ├── download_data.sh、validate.sh、train.sh、eval.sh、compare.sh
    │   └── repro_omni3d.sh 复现论文的 Omni3D 数字
    ├── dv/
    │   ├── datasets/       自定义数据集类，以及数据格式说明 README.md
    │   ├── configs/        common.py（全部超参和构建函数）、finetune.py、evaluate.py、repro_omni3d.py
    │   └── moge_compat.py  MoGe 接口兼容层
    └── tools/              数据校验、指标导出、toy 数据生成、各类检查
```

**运行机制**：

- 每个脚本先 source `scripts/env.sh`，它会：
  - 把 `data_verify/` 和 `WildDet3D/` 加入 `PYTHONPATH`；
  - `cd` 进 `WildDet3D/`，因为上游代码按相对路径读取 `data/` 和 `pretrained/`。
- 然后脚本调用 vis4d：训练用 `vis4d fit --config dv/configs/finetune.py`，评估用 `vis4d test --config dv/configs/evaluate.py`。
- 每次运行的具体设置都由脚本通过 `DV_*` 环境变量传给配置文件。
- 命令行里的相对路径会在 `cd` 之前转成绝对路径，所以不用担心目录切换的影响。

## 和上游的差异与注意事项

所有默认超参都在 `dv/configs/common.py` 的开头。

- **推理用 GT 内参**：上游的训练配置设置了 `use_predicted_intrinsics=True`。本仓库在评估和验证时都改用数据集提供的 GT 内参，和官方 eval 配置一致；不改的话 SUNRGBD 的 AP 会低 1.6–2。
- **warmup**：上游 warmup 是 1000 步，比这里的整个训练还长，学习率永远升不到设定值。这里改成 `min(100, 1 个 epoch 的步数)`。
- **验证和日志频率**：验证改为每个 epoch 一次（上游每半个 epoch 一次），loss 改为每 10 步记录一次。
- **MoGe**：
  - 固定到 `925b8ed`，因为 MoGe V3 改了 loss 函数的签名。
  - WildDet3D 用的是作者自己改过、没有公开的 MoGe，多一个 `importance_sampling` 参数。`dv/moge_compat.py` 补上了这个参数：设为 False 时，锚点在有效像素上均匀采样。
- **依赖和 CUDA**：`transformers` 锁定在 5 以下，因为 5.x 需要 torch 2.6 以上。环境里自带一份 CUDA 12.1 toolkit，用来针对 torch cu121 编译 `vis4d_cuda_ops`，所以和系统的 CUDA 版本无关。
- **运行开销**：每个 checkpoint 有 8.7 GB，每个 epoch 会保存两份（最近一个 epoch 和 `last.ckpt`），要留够磁盘空间。网络文件系统上每次保存约需 40 秒。CPU 负载很高时，启动（import 和建模型）可能需要 10 分钟左右。
- **后台运行**：用 `setsid nohup ... &` 启动，这样关掉终端也不会中断。
