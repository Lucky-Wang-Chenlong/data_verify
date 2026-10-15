"""Finetune entry point (vis4d fit). Use scripts/train.sh, which sets:

    DV_RATIO              fraction of each batch from the custom train set
                          (0 = standard data only, i.e. the A1 control)
    DV_EPOCHS             epochs (LR x0.1 after 2/3 of them and after the last)
    DV_SAMPLES_PER_EPOCH  global samples per epoch (same for every ratio)
    DV_LR                 base learning rate
    DV_GPUS               used only to size the warmup (<= 1 epoch)
    DV_VAL_OMNI3D         1 = also validate on SUNRGBD_test (forgetting check)
    DV_EXP_NAME           run name
"""

from __future__ import annotations

import math
import os

from dv.configs.common import (
    BASE_LR,
    NUM_EPOCHS,
    SAMPLES_PER_EPOCH,
    SAMPLES_PER_GPU,
    WARMUP_ITERS,
    build_finetune_config,
)


def _env(key, default, cast=str):
    value = os.environ.get(key, "")
    return cast(value) if value else default


def get_config():
    ratio = _env("DV_RATIO", 0.2, float)
    epochs = _env("DV_EPOCHS", NUM_EPOCHS, int)
    samples = _env("DV_SAMPLES_PER_EPOCH", SAMPLES_PER_EPOCH, int)
    steps_per_epoch = max(1, samples // (_env("DV_GPUS", 8, int) * SAMPLES_PER_GPU))
    return build_finetune_config(
        exp_name=_env("DV_EXP_NAME", f"ratio_{ratio:g}"),
        custom_proportion=ratio,
        num_epochs=epochs,
        samples_per_epoch=samples,
        validate_omni3d=_env("DV_VAL_OMNI3D", 1, int) == 1,
        step_1=math.ceil(2 * epochs / 3),
        step_2=epochs,
        warmup_iters=min(WARMUP_ITERS, steps_per_epoch),
        base_lr=_env("DV_LR", BASE_LR, float),
    )
