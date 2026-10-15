"""Pull a few train/test batches from a data_verify config (CPU only).

Usage (after `source scripts/env.sh`, which cd's into WildDet3D/):
    dv_custom data/custom/toy        # point the configs at a dataset
    DV_RATIO=0.2 python $DV/tools/check_dataloader.py dv.configs.finetune --train 20
    python $DV/tools/check_dataloader.py dv.configs.evaluate --train 0
"""
import argparse
import collections
import importlib

from vis4d.config import instantiate_classes

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--train", type=int, default=20, help="#train batches (0=skip)")
args = p.parse_args()

cfg = importlib.import_module(args.config).get_config()

if args.train > 0:
    train_dl = instantiate_classes(cfg.data.train_dataloader)
    print(f"train: {len(train_dl)} batches/epoch/rank")
    src = collections.Counter()
    for i, batch in enumerate(train_dl):
        src.update(batch["dataset_name"])
        if i == args.train - 1:
            break
    print("train samples by dataset:", dict(src))

for dl in instantiate_classes(cfg.data.test_dataloader):
    print(f"test: {len(dl)} batches over {len(dl.dataset)} samples")
    batch = next(iter(dl))
    print("test first batch type:", type(batch).__name__)
