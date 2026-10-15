"""Custom (user-generated) Omni3D-format dataset config."""

from __future__ import annotations

import os

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from dv.datasets.custom3d import (
    Custom3DDataset,
    load_custom3d_class_map,
)

from configs.base.dataset.transform import get_test_transforms_cfg, get_train_transforms_cfg


def get_custom3d_dataset_cfg(
    data_root: str,
    dataset_name: str,
    data_backend: None | ConfigDict = None,
    remove_empty: bool = False,
    with_depth: bool = False,
    depth_scale: float = 1000.0,
    max_depth: float = 20.0,
    cache_as_binary: bool = True,
    truncation_thres: float = 0.33333333,
    visibility_thres: float = 0.33333333,
    min_height_thres: float = 0.0625,
    max_height_thres: float = 1.50,
) -> ConfigDict:
    """Get dataset config for one custom split.

    Args:
        data_root: Dataset directory containing annotations/, images/, depth/.
        dataset_name: Annotation file stem, e.g. "MyCat_train".
        data_backend: HDF5Backend or FileBackend config (None = FileBackend).
        remove_empty: Drop images without valid annotations.
        with_depth: Load depth maps (required for training).
        depth_scale: PNG depth divisor (1000 = millimeters).
        max_depth: Depth beyond this (meters) is zeroed.
        cache_as_binary: Cache the parsed annotations as a pickle.
        truncation_thres: Truncation threshold for is_ignore.
        visibility_thres: Visibility threshold for is_ignore.
        min_height_thres: Min box height ratio for is_ignore.
        max_height_thres: Max box height ratio for is_ignore.
    """
    annotation_path = os.path.join(
        data_root, "annotations", f"{dataset_name}.json"
    )
    return class_config(
        Custom3DDataset,
        data_root=data_root,
        dataset_name=dataset_name,
        class_map=load_custom3d_class_map(annotation_path),
        data_backend=data_backend,
        data_prefix=data_root,
        remove_empty=remove_empty,
        with_depth=with_depth,
        depth_scale=depth_scale,
        max_depth=max_depth,
        cache_as_binary=cache_as_binary,
        cached_file_path=os.path.join(
            data_root, "cache", f"{dataset_name}.pkl"
        ),
        truncation_thres=truncation_thres,
        visibility_thres=visibility_thres,
        min_height_thres=min_height_thres,
        max_height_thres=max_height_thres,
    )


def get_custom3d_train_cfg(
    data_root: str,
    train_dataset: str,
    data_backend: None | ConfigDict = None,
    shape: tuple[int, int] = (1008, 1008),
    depth_scale: float = 1000.0,
    max_depth: float = 20.0,
    cache_as_binary: bool = True,
    truncation_thres: float = 0.50,
    visibility_thres: float = 0.1,
    min_height_thres: float = 0.0,
    max_height_thres: float = 1.50,
) -> ConfigDict:
    """Get the train config (relaxed filtering, like the other datasets)."""
    dataset_cfg = get_custom3d_dataset_cfg(
        data_root=data_root,
        dataset_name=train_dataset,
        data_backend=data_backend,
        remove_empty=True,
        with_depth=True,
        depth_scale=depth_scale,
        max_depth=max_depth,
        cache_as_binary=cache_as_binary,
        truncation_thres=truncation_thres,
        visibility_thres=visibility_thres,
        min_height_thres=min_height_thres,
        max_height_thres=max_height_thres,
    )
    return class_config(
        DataPipe,
        datasets=dataset_cfg,
        preprocess_fn=get_train_transforms_cfg(shape=shape),
    )


def get_custom3d_test_cfg(
    data_root: str,
    test_dataset: str,
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    shape: tuple[int, int] = (1008, 1008),
    depth_scale: float = 1000.0,
    max_depth: float = 20.0,
    cache_as_binary: bool = True,
) -> ConfigDict:
    """Get the test config (default Omni3D filtering thresholds)."""
    dataset_cfg = get_custom3d_dataset_cfg(
        data_root=data_root,
        dataset_name=test_dataset,
        data_backend=data_backend,
        with_depth=with_depth,
        depth_scale=depth_scale,
        max_depth=max_depth,
        cache_as_binary=cache_as_binary,
    )
    return class_config(
        DataPipe,
        datasets=dataset_cfg,
        preprocess_fn=get_test_transforms_cfg(
            shape=shape, with_depth=with_depth
        ),
    )
