"""Build the model from a data_verify config and check checkpoint key coverage."""
import importlib, sys, torch
from vis4d.config import instantiate_classes

cfg = importlib.import_module(sys.argv[1]).get_config()
model = instantiate_classes(cfg.model)
sd = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
own = model.state_dict()
shape_mismatch = [k for k in sd if k in own and own[k].shape != sd[k].shape]
print(f"model params: {len(own)} | ckpt: {len(sd)}")
print(f"missing: {len(res.missing_keys)} {res.missing_keys[:10]}")
print(f"unexpected: {len(res.unexpected_keys)} {res.unexpected_keys[:10]}")
print(f"shape mismatch: {len(shape_mismatch)} {shape_mismatch[:10]}")
