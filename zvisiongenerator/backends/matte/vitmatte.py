"""Load ViTMatte and its image processor from a pinned snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

__all__ = ["load_vitmatte"]


def load_vitmatte(snapshot: Path, device: str) -> tuple[VitMatteImageProcessor, VitMatteForImageMatting]:
    """Return ViTMatte's image processor and its model, in float32 on *device* in eval mode.

    Args:
        snapshot: Folder holding ``config.json``, ``preprocessor_config.json`` and ``model.safetensors``.
        device: Torch device string to move the model to.
    """
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    processor = VitMatteImageProcessor.from_pretrained(str(Path(snapshot)))
    model = VitMatteForImageMatting.from_pretrained(str(Path(snapshot)))
    return processor, model.to(device).eval()
