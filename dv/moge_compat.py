"""Make public MoGe match the loss API that WildDet3D's LingBot backend calls.

WildDet3D was trained with a patched MoGe whose ``affine_invariant_local_loss``
takes ``importance_sampling``; public MoGe never had that flag. With
``importance_sampling=False`` (what WildDet3D passes) patch anchors are drawn
uniformly over valid pixels, i.e. MoGe-1 sampling (see the note in MoGe's
``compute_anchor_sampling_weight``). This also skips the costly per-pixel
neighbour test.

Requires MoGe at the last pre-V3 commit (925b8ed, pinned by setup_env.sh):
V3 changed the global/local loss signatures (explicit ``mask`` argument).
"""

from __future__ import annotations

import inspect


def _uniform_anchor_weight(points, mask, radius_2d, radius_3d, num_test=64):
    weight = mask.float()
    return weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)


def install() -> None:
    """Add ``importance_sampling`` to MoGe's local loss (idempotent)."""
    import moge.train.losses as losses

    if "mask" in inspect.signature(
        losses.affine_invariant_global_loss
    ).parameters:
        raise ImportError(
            "MoGe V3+ found; WildDet3D needs the pre-V3 loss API. Run "
            "`git -C WildDet3D/third_party/moge checkout 925b8ed`."
        )
    original = losses.affine_invariant_local_loss
    if "importance_sampling" in inspect.signature(original).parameters:
        return

    def affine_invariant_local_loss(
        *args, importance_sampling: bool = True, **kwargs
    ):
        if importance_sampling:
            return original(*args, **kwargs)
        # The original resolves compute_anchor_sampling_weight from module
        # globals at call time, so swap it for the duration of the call.
        saved = losses.compute_anchor_sampling_weight
        losses.compute_anchor_sampling_weight = _uniform_anchor_weight
        try:
            return original(*args, **kwargs)
        finally:
            losses.compute_anchor_sampling_weight = saved

    losses.affine_invariant_local_loss = affine_invariant_local_loss

