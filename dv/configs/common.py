"""Shared builders for the data-verification finetuning experiments.

Experiment design (see README.md):
    A0  zero-shot  : released checkpoint, no training     (eval configs only)
    A1  control    : finetune on standard data only
    A2  treatment  : finetune on standard data + custom data

A1 and A2 use the same number of samples per epoch, epochs, LR schedule and
seed, so any difference on the target category comes from the custom data.

Per-run settings come from environment variables set by scripts/train.sh and
scripts/eval.sh, so one config file serves every dataset and run:
    DV_CUSTOM_ROOT, DV_CUSTOM_TRAIN, DV_CUSTOM_TEST, DV_CUSTOM_DEPTH_SCALE
    DV_OUTPUT_DIR   exact output directory of the run
The entry points (finetune.py, evaluate.py) read the remaining knobs.
"""

from __future__ import annotations

import os

from torch.optim.lr_scheduler import LinearLR

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io import FileBackend
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import CallbackConnector
from vis4d.zoo.base import get_default_callbacks_cfg, get_default_cfg
from vis4d.zoo.base import get_inference_dataloaders_cfg, get_lr_scheduler_cfg

from configs.base.base_connector import CONN_OMNI3D_DET3D_EVAL
from configs.base.callback import (
    get_omni3d_evaluator_cfg,
    get_visualizer_callback_cfg,
)
from configs.base.connector import get_wilddet3d_data_connector_cfg
from configs.base.data import (
    get_wilddet3d_data_cfg,
    wilddet3d_5mode_mask_pt_collate_fn,
    wilddet3d_test_collate_fn,
)
from dv.datasets.config import (
    get_custom3d_test_cfg,
    get_custom3d_train_cfg,
)
from configs.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg
from dv.datasets.custom3d import load_custom3d_class_map
from wilddet3d.data.samplers import build_train_dataloader_with_ratios
from wilddet3d.eval.detect3d import Detect3DEvaluator
from wilddet3d.eval.open import OpenDetect3DEvaluator

from dv import moge_compat

moge_compat.install()  # runs in every DDP rank (each one loads this config)

# ============================================================
# Data locations
# ============================================================
SAM3_IMAGE_SHAPE = (1008, 1008)

# Exact output directory of a run (checkpoints, logs, tensorboard, eval).
# Set by scripts/train.sh / eval.sh; unset -> vis4d-workspace/<exp>/<time>.
OUTPUT_DIR = os.environ.get("DV_OUTPUT_DIR", "")

OMNI3D_ROOT = "data/omni3d"
OMNI3D_TRAIN = ("SUNRGBD_train", "SUNRGBD_val", "ARKitScenes_train_sub5k")
OMNI3D_TEST = ("SUNRGBD_test",)

CUSTOM_ROOT = os.environ.get("DV_CUSTOM_ROOT", "data/custom/toy")
CUSTOM_TRAIN = os.environ.get("DV_CUSTOM_TRAIN", "Toy_train")
CUSTOM_TEST = os.environ.get("DV_CUSTOM_TEST", "Toy_test")
CUSTOM_DEPTH_SCALE = float(os.environ.get("DV_CUSTOM_DEPTH_SCALE", "1000"))

# ============================================================
# Finetuning recipe (Stage 3 settings, shortened)
# ============================================================
SAMPLES_PER_GPU = 4
WORKERS_PER_GPU = 4
BASE_LR = 1e-4
NUM_EPOCHS = 3
STEP_1, STEP_2 = 2, 3  # LR x0.1 after epoch 2
# Linear warmup from 0.1x. Upstream uses 1000 iters, which is longer than
# these whole runs (3 x 250 steps), so the LR would never reach BASE_LR.
WARMUP_ITERS = 100
# Upstream logs every 50 steps, more than an epoch of the overfit run.
LOG_EVERY_N_STEPS = 10
# Fixed epoch length shared by A1 and A2 (global samples, all GPUs).
SAMPLES_PER_EPOCH = 8000
SEED = 42  # fixed so A1 / A2 differ only in data
BACKBONE_FREEZE_BLOCKS = 28
LINGBOT_ENCODER_FREEZE_BLOCKS = 21


def _default_cfg(exp_name: str) -> ExperimentConfig:
    """vis4d default config whose output_dir is exactly OUTPUT_DIR.

    vis4d writes to <work_dir>/<experiment_name>/<version> (tensorboard too),
    so split OUTPUT_DIR into those parts and drop the timestamped version.
    """
    if not OUTPUT_DIR:
        return get_default_cfg(exp_name=exp_name)
    out = os.path.abspath(OUTPUT_DIR)
    config = get_default_cfg(
        exp_name=os.path.basename(out), work_dir=os.path.dirname(out)
    )
    config.version = ""
    return config


def _backend(root: str) -> ConfigDict:
    """HDF5 if the dataset was packed (images.hdf5 present), else files."""
    if os.path.exists(os.path.join(root, "images.hdf5")):
        return class_config(HDF5Backend)
    return class_config(FileBackend)


def _sam3_structure() -> ConfigDict:
    """SAM3 built without pretrained weights.

    All SAM3 weights come from the WildDet3D checkpoint passed via --ckpt,
    so the gated facebook/sam3 download is not needed.
    """
    from sam3.model_builder import build_sam3_image_model

    return class_config(
        build_sam3_image_model,
        checkpoint_path=None,
        load_from_HF=False,
        device="cpu",
        eval_mode=False,
        enable_segmentation=False,
    )


def _model_cfg(params: ConfigDict, oracle_eval: bool = False,
               use_depth_input_test: bool = False) -> tuple[ConfigDict, ConfigDict]:
    return get_wilddet3d_cfg(
        params,
        sam3_model=_sam3_structure(),
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=LINGBOT_ENCODER_FREEZE_BLOCKS,
        backbone_freeze_blocks=BACKBONE_FREEZE_BLOCKS,
        canonical_rotation=True,
        # Decode 3D boxes with the dataset's GT intrinsics, as the official eval
        # configs do. (Training configs set use_predicted_intrinsics=True; it
        # only affects inference decoding and costs ~3 AP on SUNRGBD.)
        use_predicted_intrinsics=False,
        eval_3d_conf_weight=0.5,
        oracle_eval=oracle_eval,
        use_depth_input_test=use_depth_input_test,
    )


def _optim_cfg(params: ConfigDict, warmup_iters: int) -> list[ConfigDict]:
    """Upstream optimizer and step schedule with a configurable warmup."""
    optimizers = get_wilddet3d_optim_cfg(params, warmup=False)
    if warmup_iters > 0:
        warmup = get_lr_scheduler_cfg(
            class_config(LinearLR, start_factor=0.1, total_iters=warmup_iters),
            end=warmup_iters,
            epoch_based=False,
        )
        optimizers[0].lr_schedulers = [warmup, *optimizers[0].lr_schedulers]
    return optimizers


def _custom_evaluator_cfg(iou_type: str = "bbox") -> ConfigDict:
    annotation = os.path.join(
        CUSTOM_ROOT, "annotations", f"{CUSTOM_TEST}.json"
    )
    class_map = load_custom3d_class_map(annotation)
    return class_config(
        Detect3DEvaluator,
        det_map={n: i for i, n in enumerate(sorted(class_map))},
        cat_map=class_map,
        annotation=annotation,
        iou_type=iou_type,
        num_columns=4,
    )


def _test_datasets(
    with_custom: bool, with_omni3d: bool, with_depth: bool = False
) -> list[ConfigDict]:
    datasets = []
    if with_omni3d:
        datasets.append(
            get_omni3d_test_cfg(
                data_root=OMNI3D_ROOT,
                test_datasets=OMNI3D_TEST,
                data_backend=class_config(HDF5Backend),
                shape=SAM3_IMAGE_SHAPE,
                with_depth=with_depth,
            )
        )
    if with_custom:
        datasets.append(
            get_custom3d_test_cfg(
                data_root=CUSTOM_ROOT,
                test_dataset=CUSTOM_TEST,
                data_backend=_backend(CUSTOM_ROOT),
                with_depth=with_depth,
                shape=SAM3_IMAGE_SHAPE,
                depth_scale=CUSTOM_DEPTH_SCALE,
            )
        )
    return datasets


def _callbacks(
    output_dir, with_custom: bool, with_omni3d: bool
) -> list[ConfigDict]:
    """One evaluator that routes predictions by dataset name."""
    callbacks = get_default_callbacks_cfg(refresh_rate=LOG_EVERY_N_STEPS)
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                OpenDetect3DEvaluator,
                datasets=[CUSTOM_TEST] if with_custom else [],
                evaluators=[_custom_evaluator_cfg()] if with_custom else [],
                omni3d_evaluator=(
                    get_omni3d_evaluator_cfg(
                        data_root=OMNI3D_ROOT,
                        omni3d50=True,
                        test_datasets=list(OMNI3D_TEST),
                    )
                    if with_omni3d
                    else None
                ),
            ),
            metrics_to_eval=["3D"],
            save_predictions=True,
            output_dir=output_dir,
            save_prefix="detection",
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_OMNI3D_DET3D_EVAL
            ),
        )
    )
    callbacks.extend(get_visualizer_callback_cfg(output_dir))
    return callbacks


def build_finetune_config(
    exp_name: str,
    custom_proportion: float,
    num_epochs: int = NUM_EPOCHS,
    samples_per_epoch: int = SAMPLES_PER_EPOCH,
    validate_custom: bool = True,
    validate_omni3d: bool = True,
    step_1: int = STEP_1,
    step_2: int = STEP_2,
    warmup_iters: int = WARMUP_ITERS,
    base_lr: float = BASE_LR,
) -> ExperimentConfig:
    """Finetune from the released checkpoint (pass it with --ckpt).

    Args:
        exp_name: Run name (output dir when DV_OUTPUT_DIR is unset).
        custom_proportion: Fraction of each batch drawn from the custom
            train set. 0 = control run (custom data never loaded),
            1 = custom data only (Omni3D never loaded).
        num_epochs: Number of epochs.
        samples_per_epoch: Global samples per epoch (same for A1 / A2).
        validate_custom: Also validate on the custom test set. Turn off for
            runs where the custom test set does not exist yet.
        validate_omni3d: Also validate on Omni3D (forgetting check).
        step_1, step_2: Epochs after which the LR drops x0.1 (each).
        warmup_iters: Linear LR warmup length in steps (0 = none).
        base_lr: Base learning rate (pretrained parts use 0.1x).
    """
    config = _default_cfg(exp_name)
    config.use_checkpoint = True
    config.seed = SEED

    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=num_epochs,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        base_lr=base_lr,
        step_1=step_1,
        step_2=step_2,
        nms=True,
        nms_iou_threshold=0.6,
        score_threshold=0.05,
    )
    config.params = params

    config.model, box_coder = _model_cfg(params)

    # -------------------- Train data --------------------
    train_sets, proportions = [], []
    if custom_proportion < 1:
        train_sets.append(
            get_omni3d_train_cfg(
                data_root=OMNI3D_ROOT,
                train_datasets=OMNI3D_TRAIN,
                data_backend=class_config(HDF5Backend),
                shape=SAM3_IMAGE_SHAPE,
            )
        )
        proportions.append(1.0 - custom_proportion)
    if custom_proportion > 0:
        train_sets.append(
            get_custom3d_train_cfg(
                data_root=CUSTOM_ROOT,
                train_dataset=CUSTOM_TRAIN,
                data_backend=_backend(CUSTOM_ROOT),
                shape=SAM3_IMAGE_SHAPE,
                depth_scale=CUSTOM_DEPTH_SCALE,
            )
        )
        proportions.append(custom_proportion)

    data = DataConfig()
    data.train_dataloader = class_config(
        build_train_dataloader_with_ratios,
        dataset=class_config(DataPipe, datasets=train_sets),
        target_proportions=proportions,
        num_samples=samples_per_epoch,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        batchprocess_fn=class_config(
            compose, transforms=[class_config(ToTensor)]
        ),
        collate_fn=wilddet3d_5mode_mask_pt_collate_fn,
    )

    # -------------------- Val data (text prompts) --------------------
    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=class_config(
            DataPipe,
            datasets=_test_datasets(
                with_custom=validate_custom, with_omni3d=validate_omni3d
            ),
        ),
        batchprocess_cfg=class_config(
            compose, transforms=[class_config(ToTensor)]
        ),
        samples_per_gpu=1,
        workers_per_gpu=WORKERS_PER_GPU,
        collate_fn=wilddet3d_test_collate_fn,
    )
    config.data = data

    config.loss = get_wilddet3d_loss_cfg(
        params,
        box_coder=box_coder,
        use_3d_conf=True,
        use_ignore_suppress=True,
        presence_loss_weight=5.0,
    )
    config.optimizers = _optim_cfg(params, warmup_iters)
    config.callbacks = _callbacks(
        config.output_dir,
        with_custom=validate_custom,
        with_omni3d=validate_omni3d,
    )
    (
        config.train_data_connector,
        config.test_data_connector,
    ) = get_wilddet3d_data_connector_cfg()
    config.log_every_n_steps = LOG_EVERY_N_STEPS
    pl_trainer = get_pl_cfg(config, params)
    # Upstream validates every half epoch; once per epoch is enough here
    # (the Omni3D test set alone is 5050 images).
    pl_trainer.val_check_interval = 1.0
    config.pl_trainer = pl_trainer
    return config.value_mode()


def build_eval_config(
    exp_name: str,
    prompt: str,
    with_depth: bool = False,
    with_custom: bool = True,
    with_omni3d: bool = True,
) -> ExperimentConfig:
    """Evaluate a checkpoint (pass it with --ckpt) via `vis4d test`.

    Args:
        exp_name: Run name (output dir when DV_OUTPUT_DIR is unset).
        prompt: "text" (detect by category name) or "box" (GT 2D box as
            prompt; isolates 3D box quality from 2D recognition).
        with_depth: Feed GT depth to the model at test time.
        with_custom: Evaluate the custom test set.
        with_omni3d: Evaluate the Omni3D test set(s) (forgetting check).
    """
    assert prompt in ("text", "box"), prompt
    oracle = prompt == "box"

    config = _default_cfg(exp_name)
    config.use_checkpoint = True
    config.seed = SEED

    params = get_wilddet3d_hyperparams_cfg(
        samples_per_gpu=SAMPLES_PER_GPU, workers_per_gpu=WORKERS_PER_GPU
    )
    config.params = params

    test_pipe = class_config(
        DataPipe,
        datasets=_test_datasets(with_custom, with_omni3d, with_depth),
    )
    config.data = get_wilddet3d_data_cfg(
        train_datasets=test_pipe,
        test_datasets=[test_pipe],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
        oracle_eval=oracle,
    )

    config.model, box_coder = _model_cfg(
        params, oracle_eval=oracle, use_depth_input_test=with_depth
    )
    config.loss = get_wilddet3d_loss_cfg(params, box_coder)
    config.optimizers = get_wilddet3d_optim_cfg(params)
    config.callbacks = _callbacks(config.output_dir, with_custom, with_omni3d)
    (
        config.train_data_connector,
        config.test_data_connector,
    ) = get_wilddet3d_data_connector_cfg()
    config.pl_trainer = get_pl_cfg(config, params)
    return config.value_mode()
