"""Generic Omni3D-format dataset for user-generated (e.g. synthetic) data.

Layout (one directory per dataset, all paths relative to ``data_root``)::

    <data_root>/
        annotations/<name>.json     # Omni3D-style COCO JSON
        images/...                  # or images.hdf5 (HDF5Backend)
        depth/...                   # or depth.hdf5 (HDF5Backend)

Each image entry carries its own ``depth_path`` (relative to ``data_root``),
so there is no hidden filename convention. Depth is either a 16-bit PNG with
``depth = value / depth_scale`` (default millimeters) or a float32 ``.npy`` in
meters. See ``dv/datasets/README.md`` for the full field spec.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
from vis4d.common.typing import ArgsType, DictStrAny

from wilddet3d.data.datasets.coco3d import COCO3DDataset


def load_custom3d_class_map(annotation_path: str) -> dict[str, int]:
    """Category name -> id, read from the annotation JSON."""
    with open(annotation_path) as f:
        categories = json.load(f)["categories"]
    return {cat["name"]: cat["id"] for cat in categories}


class Custom3DDataset(COCO3DDataset):
    """Omni3D-format dataset with explicit per-image depth paths."""

    def __init__(
        self,
        class_map: dict[str, int],
        max_depth: float = 20.0,
        depth_scale: float = 1000.0,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        Args:
            class_map: Category name -> id from the annotation JSON. Every
                category listed in the JSON is used as a text prompt, so
                categories without annotations act as negative prompts.
            max_depth: Depth values beyond this (meters) are zeroed.
            depth_scale: PNG depth divisor (1000 = millimeters).
        """
        super().__init__(
            class_map=class_map,
            det_map={n: i for i, n in enumerate(sorted(class_map))},
            max_depth=max_depth,
            depth_scale=depth_scale,
            **kwargs,
        )

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Depth path is stored explicitly on each image entry."""
        if "depth_path" not in img:
            return None
        return os.path.join(self.data_prefix, img["depth_path"])

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Load PNG (scaled) or NPY (meters) depth."""
        if sample["depth_filename"].endswith(".npy"):
            depth_bytes = self.data_backend.get(sample["depth_filename"])
            return np.load(io.BytesIO(depth_bytes)).astype(np.float32)
        return super().get_depth_map(sample)
