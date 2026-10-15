"""Instantiate the Omni3D train/test datasets used for finetuning and load samples."""
import sys
from vis4d.config import instantiate_classes
from vis4d.config import class_config
from vis4d.data.io.hdf5 import HDF5Backend
from configs.base.dataset.omni3d import get_omni3d_dataset_cfg

for name in ["SUNRGBD_train", "SUNRGBD_val", "ARKitScenes_train_sub5k", "SUNRGBD_test"]:
    is_train = "test" not in name
    cfg = get_omni3d_dataset_cfg(
        data_root="data/omni3d", datasets=[name],
        data_backend=class_config(HDF5Backend),
        remove_empty=is_train, with_depth=True, cache_as_binary=True,
    )[0]
    ds = instantiate_classes(cfg)
    s = ds[0]
    print(f"== {name}: {len(ds)} samples | img {s['images'].shape} | "
          f"depth {s['depth_maps'].shape} max={s['depth_maps'].max():.2f} | "
          f"boxes3d {s['boxes3d'].shape} | K {s['intrinsics'].tolist()[0]}", flush=True)
