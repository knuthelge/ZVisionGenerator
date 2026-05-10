"""ImageBackend Protocol — explicit contract for platform-specific image inference engines.

Protocol requires name, load_model, text_to_image, image_to_image.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from PIL import Image

from zvisiongenerator.utils.image_model_detect import ImageModelInfo


@runtime_checkable
class ImageBackend(Protocol):
    """Platform-specific image inference engine.

    Required interface:
    - name: str identifier ("mflux" or "diffusers")
    - load_model(): load model from path with optional quantization/LoRA
    - text_to_image(): generate image from text prompt
    - image_to_image(): refine existing image with text prompt
    """

    name: str  # "mflux" or "diffusers"

    def load_model(
        self,
        model_path: str,
        quantize: int | None = None,
        precision: str = "bfloat16",
        lora_paths: list[str] | None = None,
        lora_weights: list[float] | None = None,
    ) -> tuple[Any, ImageModelInfo]:
        """Load a model. Returns (model_handle, model_info)."""
        ...

    def text_to_image(
        self,
        model: Any,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        skip_signal: Any | None = None,
        step_callback: Any | None = None,
    ) -> Image.Image | None:
        """Generate image from text. Returns None if skipped."""
        ...

    def image_to_image(
        self,
        model: Any,
        image: Image.Image,
        prompt: str,
        strength: float,
        steps: int,
        seed: int,
        guidance: float,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        skip_signal: Any | None = None,
        step_callback: Any | None = None,
    ) -> Image.Image | None:
        """Refine image. Returns None if skipped."""
        ...
