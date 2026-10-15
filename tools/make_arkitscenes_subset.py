"""Build a deterministic N-image subset of Omni3D ARKitScenes_train.

Only images that have a depth map in ARKitScenes_depth.hdf5 are eligible
(samples without depth are dropped by the training dataloader anyway).
Writes <out_name>.json next to the source JSON and fetches just those images
from the remote Omni3D zip via HTTP range requests.

Usage (after `source scripts/env.sh`, which cd's into WildDet3D/):
    python $DV/tools/make_arkitscenes_subset.py --data_root data --num 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import h5py
from remotezip import RemoteZip
from tqdm import tqdm

ZIP_URL = "https://dl.fbaipublicfiles.com/omni3d_data/ARKitScenes_images.zip"
ZIP_PREFIX = "datasets/"  # zip member = datasets/ARKitScenes/Training/...


def depth_key(file_path: str) -> str:
    # ARKitScenes/Training/<vid>/<name>.jpg -> Training/<vid>/<name>.png
    _, split, vid, name = file_path.split("/")
    return f"{split}/{vid}/{name.replace('jpg', 'png')}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--num", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src", default="ARKitScenes_train")
    p.add_argument("--out_name", default="ARKitScenes_train_sub5k")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    ann_dir = os.path.join(args.data_root, "omni3d", "annotations")
    with open(os.path.join(ann_dir, f"{args.src}.json")) as f:
        data = json.load(f)

    with h5py.File(
        os.path.join(args.data_root, "ARKitScenes_depth.hdf5"), "r"
    ) as h5:
        eligible = [
            im for im in data["images"] if depth_key(im["file_path"]) in h5
        ]
    print(f"{len(eligible)}/{len(data['images'])} images have depth")

    eligible.sort(key=lambda im: im["id"])
    images = random.Random(args.seed).sample(eligible, args.num)
    img_ids = {im["id"] for im in images}
    anns = [a for a in data["annotations"] if a["image_id"] in img_ids]
    subset = {k: v for k, v in data.items() if k not in ("images", "annotations")}
    subset.update(images=images, annotations=anns)
    out_json = os.path.join(ann_dir, f"{args.out_name}.json")
    with open(out_json, "w") as f:
        json.dump(subset, f)
    print(f"wrote {out_json}: {len(images)} images, {len(anns)} annotations")

    todo = [
        im["file_path"]
        for im in images
        if not os.path.exists(os.path.join(args.data_root, im["file_path"]))
    ]

    def fetch(chunk: list[str]) -> None:
        with RemoteZip(ZIP_URL) as z:
            for fp in chunk:
                dst = os.path.join(args.data_root, fp)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst + ".tmp", "wb") as f:
                    f.write(z.read(ZIP_PREFIX + fp))
                os.replace(dst + ".tmp", dst)
                pbar.update(1)

    chunks = [todo[i :: args.workers] for i in range(args.workers)]
    with tqdm(total=len(todo), desc="fetch images") as pbar:
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(fetch, chunks))


if __name__ == "__main__":
    main()
