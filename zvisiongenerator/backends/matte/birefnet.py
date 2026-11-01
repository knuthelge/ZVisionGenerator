"""Load BiRefNet from the vendored model code and a pinned snapshot, with the deformable convolutions swapped."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = ["load_birefnet"]

_WEIGHTS = "model.safetensors"


def load_birefnet(snapshot: Path, device: str, dtype: torch.dtype) -> nn.Module:
    """Build BiRefNet, swap its deformable convolutions, load the pinned weights and return it on *device* in eval mode.

    The model is constructed from the vendored classes, never through the Hub's ``trust_remote_code`` path or an Auto
    class: the config's ``model_type`` is a Segformer name, so Auto resolution would pick the wrong class.

    Args:
        snapshot: Folder holding ``model.safetensors``.
        device: Torch device string to move the model to.
        dtype: Torch dtype to cast the model to.

    Returns:
        The loaded model.
    """
    from birefnet.birefnet import BiRefNet
    from birefnet.BiRefNet_config import BiRefNetConfig
    from safetensors.torch import load_file

    from zvisiongenerator.backends.matte.deform_conv import GridSampleDeformableConv2d, replace_deformable_convs

    model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
    swapped = replace_deformable_convs(model, GridSampleDeformableConv2d)
    if swapped == 0:
        warnings.warn("BiRefNet has no deformable convolutions to replace; the vendored model code may have changed.", stacklevel=2)
    model.load_state_dict(load_file(Path(snapshot) / _WEIGHTS), strict=True)
    return model.to(device, dtype).eval()
