"""Evaluation entry point (vis4d test). Use scripts/eval.sh, which sets:

    DV_PROMPT       box (GT 2D box as prompt) | text (category name)
    DV_WITH_DEPTH   1 = feed GT depth to the model
    DV_EVAL_CUSTOM  1 = evaluate the custom test set
    DV_EVAL_OMNI3D  1 = evaluate SUNRGBD_test
    DV_EXP_NAME     run name
"""

from __future__ import annotations

import os

from dv.configs.common import build_eval_config


def get_config():
    prompt = os.environ.get("DV_PROMPT", "box")
    return build_eval_config(
        os.environ.get("DV_EXP_NAME", f"eval_{prompt}"),
        prompt=prompt,
        with_depth=os.environ.get("DV_WITH_DEPTH", "0") == "1",
        with_custom=os.environ.get("DV_EVAL_CUSTOM", "1") == "1",
        with_omni3d=os.environ.get("DV_EVAL_OMNI3D", "1") == "1",
    )
