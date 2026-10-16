# data_verify

English | [中文](README_zh.md)

Check whether a **self-generated 3D detection dataset is correct** by finetuning on it.

Start from the released [WildDet3D](https://github.com/allenai/WildDet3D) checkpoint and finetune on standard data (an Omni3D subset) mixed with your data at a chosen ratio. Then check two things: whether the target category improves, and whether the standard test set holds steady. This repo only contains its own code. WildDet3D's source is used unmodified.

```
1. Install the environment      ./setup_env.sh
2. Download standard data       scripts/download_data.sh [DATA_DIR]
3. Prepare your own dataset     format: dv/datasets/README.md; check it with scripts/validate.sh
4. Train                        scripts/train.sh --custom <dataset dir> --ratio <mix ratio>
5. Read the outputs             in --out: checkpoints/, metrics/ (losses etc.), results/ (AP)
```

---

## 1. Install the environment

```bash
mkdir -p <workspace> && cd <workspace>     # holds WildDet3D, the env, data and outputs
git clone <repo-url> data_verify
cd data_verify
./setup_env.sh                             # log: <workspace>/envs/setup_env.log
```

Run every later command from `data_verify/`.

This step does three things:
- clones WildDet3D (with submodules) next to this repo and pins MoGe to a compatible commit;
- builds a micromamba env in `../envs/wilddet3d` with its own CUDA 12.1, torch 2.5.1 and vis4d 1.0.0;
- downloads the released checkpoint and the LingBot-Depth weights.

You don't need the SAM3 weights. The released checkpoint already contains every weight.

Download caches (micromamba, pip, HuggingFace, torch) go to `<workspace>/.cache/` by default. If `MAMBA_ROOT_PREFIX`, `PIP_CACHE_DIR`, `HF_HOME` or `TORCH_HOME` are already set, those values are kept. A `micromamba` already on `PATH` is used. Otherwise one is downloaded to `<workspace>/.cache/bin/`.

Requirements: Linux and an NVIDIA GPU. No system CUDA is needed. `vis4d_cuda_ops` is compiled for H100 by default (`TORCH_CUDA_ARCH_LIST=9.0`). For other GPUs, set the variable first, e.g. `TORCH_CUDA_ARCH_LIST=8.0 ./setup_env.sh` for A100.

## 2. Download standard data

```bash
scripts/download_data.sh                      # default: ../data
scripts/download_data.sh /path/to/data        # or any directory
```

The download needs up to ~25 GB of disk while it runs and ~13 GB after it finishes. Finished steps are skipped, so you can re-run it to resume.

| Content | Path (relative to DATA_DIR) | Used for |
|---|---|---|
| SUNRGBD train + val (4,547 usable images) | `SUNRGBD.hdf5`, `omni3d/annotations/SUNRGBD_{train,val}.json` | training |
| ARKitScenes train subset (5k sampled with seed 0, 4,146 usable) | `ARKitScenes.hdf5`, `ARKitScenes_depth.hdf5`, `omni3d/annotations/ARKitScenes_train_sub5k.json` | training |
| SUNRGBD test (5,050 images) | `SUNRGBD.hdf5`, `omni3d/annotations/SUNRGBD_test.json` | forgetting check, paper reproduction |
| Toy custom set: chair from SUNRGBD_val (161 train / 53 test images) | `custom/toy/` | format example, smoke tests |

At the end the script creates the symlink `WildDet3D/data -> DATA_DIR`. Upstream WildDet3D reads Omni3D from the fixed relative path `data/`, so every script finds the data through this link. To move the data later, run `scripts/download_data.sh <new dir>` again and it will update the link.

## 3. Prepare your own dataset

The format is described in **[dv/datasets/README.md](dv/datasets/README.md)**. The layout is:

```
/path/to/mycat/
├── annotations/MyCat_train.json
├── annotations/MyCat_test.json
├── images/...
└── depth/...
```

First validate the data. Fix every ERROR, then check that the rendered GT boxes fit the objects:

```bash
scripts/validate.sh --custom /path/to/mycat   # renders: ../outputs/data_verify/vis/<split>/
```

Then run the overfit check: train on your train split only and evaluate on that same split (~20 min). AP should rise clearly and steadily. If it stays flat, the labels are inconsistent with each other.

```bash
scripts/train.sh --custom /path/to/mycat --overfit
```

## 4. Train

```bash
scripts/train.sh --custom /path/to/mycat --ratio 0.2 --out ../outputs/data_verify/mycat_r0.2
```

`--ratio` is the fraction of each batch drawn from your data. The rest comes from the standard data.

| `--ratio` | Meaning |
|---|---|
| `0` | standard data only: the control (A1) |
| `0.2` | 80% standard + 20% yours: the treatment (A2, default) |
| `1` | your data only |

Runs with different ratios share the samples per epoch, the epoch count, the LR schedule and the seed. Any difference in the results therefore comes from the data.

**Common options** (full list: `scripts/train.sh -h`):

| Option | Default | Description |
|---|---|---|
| `--gpus N` | 8 | number of GPUs |
| `--epochs N` | 3 | LR ×0.1 after 2/3 of the epochs and again after the last one |
| `--samples N` | 8000 | global samples per epoch; on 8 GPUs that is 250 steps/epoch, ~15 min |
| `--lr X` | 1e-4 | pretrained parts use 0.1× |
| `--train-split` / `--test-split` | auto-detected | needed when `annotations/` has several candidates |
| `--depth-scale S` | 1000 | 16-bit PNG depth units per meter (1000 = millimeters) |
| `--no-val-omni3d` | | skip SUNRGBD_test validation each epoch (saves 3–4 min per epoch) |
| `--no-eval` | | skip the final evaluation |

When training finishes, the script evaluates `last.ckpt` with both box and text prompts and exports all metrics.

**Full comparison in one go**: the zero-shot baseline, then one training run per ratio, each evaluated, summarized in one table:

```bash
nohup scripts/compare.sh --custom /path/to/mycat --ratios "0 0.2" > compare.log 2>&1 &
```

**Evaluate a single checkpoint**:

```bash
scripts/eval.sh --ckpt released --custom /path/to/mycat --prompt box   # zero-shot (A0)
scripts/eval.sh --ckpt <run>/checkpoints/last.ckpt --custom /path/to/mycat --prompt text
```

`--prompt box` gives the model the GT 2D box as the prompt. It measures only 3D box estimation, which reflects the quality of your 3D labels most directly. `--prompt text` gives only the category name, so the model has to find the objects itself. Add `--depth` to also feed GT depth to the model.

## 5. Outputs

Everything from one run goes into its `--out` directory. The default is `../outputs/data_verify/<dataset>_ratio<R>`. Set `DV_OUT` to change the parent directory.

```
<out>/
├── checkpoints/
│   ├── last.ckpt               final weights (8.7 GB)
│   └── epoch=N-step=M.ckpt     weights of the latest epoch
├── metrics/
│   ├── train.csv               one row per 10 steps: total loss, each loss term, LR
│   └── val.csv                 one row per epoch: AP_<test set>, AP15/25/50, ATE/ASE/AOE, ODS, ...
├── results/
│   ├── summary.md              final AP table (box / text × each test set)
│   ├── summary.json            every final evaluation metric
│   ├── box/                    box-prompt eval: eval.log, results.json, log_*.txt (per-class AP)
│   └── text/                   text-prompt eval
├── logs/train.log              full training output
├── log_*.txt                   vis4d's own log (per-class AP table every epoch)
├── config_*.yaml               the full config of this run
└── events.out.tfevents.*       tensorboard (tensorboard --logdir <out>)
```

`compare.sh` puts each run in its own subdirectory and adds a summary table, `compare.md`:

```
| run          | prompt | MyCat_test | SUNRGBD_test |
| zeroshot_box | box    |            |              |   ← A0
| ratio0       | box    |            |              |   ← A1 (control)
| ratio0.2     | box    |            |              |   ← A2
```

**How to read it**: your data is correct and learnable if ratio0.2 is clearly above ratio0 on the target test set (`AP_MyCat_test`), while `AP_SUNRGBD_test` stays about the same for both.

To re-export metrics or compare any set of runs:

```bash
source scripts/env.sh
python $DV/tools/export_metrics.py run <out>
python $DV/tools/export_metrics.py compare <run1> <run2> ... --to table.md
```

---

## Verified results (released checkpoint, 8×H100)

**Paper reproduction**: the repo's official eval configs (`scripts/repro_omni3d.sh`) on SUNRGBD_test, AP (%):

| Mode | Paper | Official config (bf16 / fp32) | This repo's eval config |
|---|---|---|---|
| text | 38.9 | 38.67 / – | **38.67** |
| box | 43.1 | 41.40 / 40.85 | **41.39** |
| text + depth | 51.1 | 53.21 / 53.15 | – |
| box + depth | 58.7 | 59.84 / – | – |

The gaps to the paper range from −1.7 to +2.1 and go both ways. They are not caused by numeric precision. This repo's eval config matches the official one.

**Toy set**:
- Zero-shot: box AP 0.836, text AP 0.760. These numbers are high because the released model has likely seen these images. For a real experiment, pick a category the model hasn't seen or does poorly on zero-shot.
- Overfit: over 10 epochs, AP rises from 0.57 to 0.81. These numbers were measured before the GT-intrinsics fix (see below), so the absolute values are low, but the trend holds.

---

## Layout

```
<workspace>/               # can be shared by several codebases
├── WildDet3D/             upstream repo, unmodified (ckpt/, pretrained/, data -> DATA_DIR)
├── envs/wilddet3d/        environment
├── data/                  standard data (default DATA_DIR)
├── outputs/data_verify/   default output directory of this repo
└── data_verify/           ← this repo
    ├── setup_env.sh
    ├── scripts/
    │   ├── env.sh          sourced by the other scripts: paths and PYTHONPATH, then cd into WildDet3D/
    │   ├── download_data.sh, validate.sh, train.sh, eval.sh, compare.sh
    │   └── repro_omni3d.sh reproduces the paper's Omni3D numbers
    ├── dv/
    │   ├── datasets/       custom dataset class, plus the data format spec README.md
    │   ├── configs/        common.py (all hyperparameters and builders), finetune.py, evaluate.py, repro_omni3d.py
    │   └── moge_compat.py  MoGe API compatibility shim
    └── tools/              dataset validator, metric exporter, toy-set maker, checks
```

**How it runs**:

- Every script first sources `scripts/env.sh`, which:
  - puts `data_verify/` and `WildDet3D/` on `PYTHONPATH`;
  - `cd`s into `WildDet3D/`, because the upstream code reads `data/` and `pretrained/` by relative path.
- The script then calls vis4d: `vis4d fit --config dv/configs/finetune.py` for training, `vis4d test --config dv/configs/evaluate.py` for evaluation.
- The per-run settings reach the config files as `DV_*` environment variables set by the script.
- Relative paths on the command line are made absolute before the `cd`, so the directory change doesn't affect them.

## Differences from upstream, and notes

All default hyperparameters are at the top of `dv/configs/common.py`.

- **GT intrinsics at inference**: upstream's training configs set `use_predicted_intrinsics=True`. Here evaluation and validation use the dataset's GT intrinsics, as the official eval configs do. Without this, SUNRGBD AP is 1.6–2 points lower.
- **Warmup**: upstream warms up for 1000 steps, which is longer than these whole runs, so the LR would never reach its set value. Here it is `min(100, steps in one epoch)`.
- **Validation and logging frequency**: validation runs once per epoch (upstream: every half epoch). Losses are logged every 10 steps.
- **MoGe**:
  - Pinned to `925b8ed`, because MoGe V3 changed the loss function signatures.
  - WildDet3D was trained with the authors' own unreleased MoGe patch, which adds an `importance_sampling` argument. `dv/moge_compat.py` adds that argument. With `False`, patch anchors are sampled uniformly over valid pixels.
- **Dependencies and CUDA**: `transformers` is pinned below 5, since 5.x needs torch ≥ 2.6. The env ships its own CUDA 12.1 toolkit to compile `vis4d_cuda_ops` against torch cu121, so the system CUDA version doesn't matter.
- **Cost**: each checkpoint is 8.7 GB, and each epoch keeps two (the latest epoch and `last.ckpt`), so leave enough disk space. Saving takes ~40 s on a network file system. Under heavy CPU load, startup (imports and model build) can take ~10 min.
- **Running in the background**: start runs with `setsid nohup ... &` so they survive closing the terminal.
