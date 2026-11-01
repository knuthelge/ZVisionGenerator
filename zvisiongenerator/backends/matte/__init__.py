"""Background removal on a torch device: BiRefNet for the mask, ViTMatte for the edge."""

from __future__ import annotations

from zvisiongenerator.backends.matte.chain import TorchMatteChain

__all__ = ["TorchMatteChain"]
