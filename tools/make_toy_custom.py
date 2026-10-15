"""Make a small toy dataset in the custom format from Omni3D SUNRGBD_val.

It exercises the full custom-data pipeline (files on disk, explicit depth
paths, uint16 mm depth PNGs) before real data exists, and doubles as a
worked example of the format described in dv/datasets/README.md.

Usage (after `source scripts/env.sh`, which cd's into WildDet3D/):
    python $DV/tools/make_toy_custom.py --category chair \
        --out data/custom/toy --prefix Toy
"""

from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np
from vis4d.config import class_config, instantiate_classes
from vis4d.data.io.hdf5 import HDF5Backend

from configs.base.dataset.omni3d import get_omni3d_dataset_cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="SUNRGBD_val")
    p.add_argument("--category", default="chair")
    p.add_argument("--out", default="data/custom/toy")
    p.add_argument("--prefix", default="Toy")
    p.add_argument("--test_frac", type=float, default=0.25)
    p.add_argument("--negatives", nargs="*", default=["table", "sofa"],
                   help="extra categories listed without annotations "
                        "(used as negative text prompts)")
    args = p.parse_args()

    with open(f"data/omni3d/annotations/{args.src}.json") as f:
        src = json.load(f)
    anns_by_img: dict[int, list] = {}
    for a in src["annotations"]:
        R = np.asarray(a["R_cam"])
        if a["valid3D"] and np.abs(R @ R.T - np.eye(3)).max() > 1e-2:
            continue  # a few SUNRGBD boxes have broken rotations
        if a["category_name"] == args.category:
            anns_by_img.setdefault(a["image_id"], []).append(a)

    # Depth via the repo's own SUNRGBD loader (handles its depth encoding).
    ds = instantiate_classes(
        get_omni3d_dataset_cfg(
            data_root="data/omni3d", datasets=[args.src],
            data_backend=class_config(HDF5Backend), with_depth=True,
            cache_as_binary=True,
        )[0]
    )
    idx_by_id = {s["img_id"]: i for i, s in enumerate(ds.samples)}
    img_ids = sorted(i for i in anns_by_img if i in idx_by_id)
    random.Random(0).shuffle(img_ids)
    n_test = int(len(img_ids) * args.test_frac)
    splits = {"test": img_ids[:n_test], "train": img_ids[n_test:]}

    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    categories = [{"id": 0, "name": args.category}] + [
        {"id": i + 1, "name": n} for i, n in enumerate(args.negatives)
    ]
    src_images = {im["id"]: im for im in src["images"]}

    for split, ids in splits.items():
        images, annotations = [], []
        for new_id, old_id in enumerate(ids):
            sample = ds[idx_by_id[old_id]]
            rgb = sample["images"][0].astype(np.uint8)
            depth_m = sample["depth_maps"]
            stem = f"{split}_{new_id:06d}"
            img_rel = f"images/{stem}.jpg"
            depth_rel = f"depth/{stem}.png"
            for rel in (img_rel, depth_rel):
                os.makedirs(os.path.dirname(os.path.join(args.out, rel)),
                            exist_ok=True)
            cv2.imwrite(os.path.join(args.out, img_rel),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.out, depth_rel),
                        np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))

            im = src_images[old_id]
            images.append({
                "id": new_id, "file_path": img_rel, "depth_path": depth_rel,
                "width": im["width"], "height": im["height"], "K": im["K"],
            })
            for a in anns_by_img[old_id]:
                annotations.append({
                    "id": len(annotations), "image_id": new_id,
                    "category_id": 0, "category_name": args.category,
                    **{k: a[k] for k in (
                        "bbox2D_proj", "bbox2D_tight", "bbox2D_trunc",
                        "bbox3D_cam", "center_cam", "dimensions", "R_cam",
                        "behind_camera", "valid3D", "truncation",
                        "visibility", "lidar_pts", "segmentation_pts",
                        "depth_error",
                    )},
                })
        out = {
            "info": {"name": f"{args.prefix}_{split}"},
            "categories": categories, "images": images,
            "annotations": annotations,
        }
        path = os.path.join(args.out, "annotations",
                            f"{args.prefix}_{split}.json")
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"{path}: {len(images)} images, {len(annotations)} boxes")


if __name__ == "__main__":
    main()
