"""SeedVR2Upscaler Protocol — contract for the platform engine behind the plain image upscale."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class SeedVR2Upscaler(Protocol):
    """Single-step diffusion upscaler that enlarges an existing image without a prompt.

    Required interface:
    - load_model(): load a SeedVR2 model from a path with optional quantization
    - save_model(): write a loaded (quantized) model's weights for reuse
    - upscale_image(): enlarge one image file to a target size
    """

    def load_model(self, model: str, *, quantize: int | None, model_path: str) -> Any:
        """Load the SeedVR2 variant *model* ("seedvr2-3b" or "seedvr2-7b") from *model_path*."""
        ...

    def save_model(self, model: Any, path: Path) -> None:
        """Write a loaded model's weights to the directory *path*."""
        ...

    def upscale_image(self, model: Any, image_path: str, *, width: int, height: int, seed: int, softness: float) -> Image.Image:
        """Return *image_path* enlarged to *width* × *height*, pre-blurred by *softness* (0-1)."""
        ...
