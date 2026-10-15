"""Reproduce the paper's Omni3D numbers with the *official* eval configs.

Wraps ``configs/eval/omni3d/<mode>.py`` from the repo unchanged, except:
    - test sets: only the Omni3D subsets we have locally (default SUNRGBD);
    - train set (built but unused by ``vis4d test``): SUNRGBD_train only;
    - SAM3: structure only (weights come from --ckpt, as in our configs);
    - outputs: DV_OUTPUT_DIR (set per mode by scripts/repro_omni3d.sh).

Select the mode with DV_REPRO_MODE (text | text_with_depth | box_prompt |
box_prompt_with_depth) and the subsets with DV_REPRO_SETS (comma list).
Use scripts/repro_omni3d.sh rather than calling this directly.
"""

from __future__ import annotations

import importlib
import os

from configs.base.dataset import omni3d as omni3d_ds

from dv.configs.common import _default_cfg, _sam3_structure

MODE = os.environ.get("DV_REPRO_MODE", "text")
TEST_SETS = tuple(os.environ.get("DV_REPRO_SETS", "SUNRGBD_test").split(","))


def get_config():
    official = importlib.import_module(f"configs.eval.omni3d.{MODE}")

    get_wilddet3d_cfg = official.get_wilddet3d_cfg

    def wilddet3d_cfg(*args, sam3_checkpoint=None, **kwargs):
        return get_wilddet3d_cfg(*args, sam3_model=_sam3_structure(), **kwargs)

    def default_cfg(exp_name, work_dir=None):
        return _default_cfg(f"repro_{MODE}")

    def train_cfg(*args, **kwargs):
        kwargs["train_datasets"] = ("SUNRGBD_train",)
        return omni3d_ds.get_omni3d_train_cfg(*args, **kwargs)

    official.OMNI3D_TEST_DATASETS = TEST_SETS
    official.get_wilddet3d_cfg = wilddet3d_cfg
    official.get_default_cfg = default_cfg
    official.get_omni3d_train_cfg = train_cfg
    return official.get_config()
