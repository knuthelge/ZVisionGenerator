"""ImageBackend Protocol — explicit contract for platform-specific image inference engines.

Protocol requires name, load_model, text_to_image, image_to_image, stored_quant_format, quantizes_from_files,
save_quantized, write_quantized_files.
"""

from __future__ import annotations

from collections.abc import Callable
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
    - stored_quant_format(): tag of saved quantized weights at a level, or None when unsupported
    - quantizes_from_files(): whether a level's stored quant is written from the source files
    - save_quantized(): write a loaded quantized model's weights for reuse
    - write_quantized_files(): write a stored quant straight from the source files
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
        steps_explicit: bool = False,
        guidance_explicit: bool = False,
        first_sigma: float | None = None,
    ) -> Image.Image | None:
        """Generate image from text. Returns None if skipped.

        steps_explicit/guidance_explicit mark whether the user explicitly set
        steps/guidance (rather than a resolved config default). They are consumed
        only by families with tuned default schedules (e.g. ideogram4); other
        families ignore them.

        first_sigma overrides Ideogram 4's first-step sigma for this run; other
        families ignore it.
        """
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

    def stored_quant_format(self, bits: int) -> str | None:
        """Return the format tag of this backend's stored quants at *bits*, or ``None`` when unsupported.

        A stored quant whose recorded tag differs is stale and is quantized again.
        """
        ...

    def quantizes_from_files(self, bits: int) -> bool:
        """Return whether the stored quant at *bits* is written by :meth:`write_quantized_files`.

        ``False`` means it is saved from a model loaded at *bits* with :meth:`save_quantized`.
        """
        ...

    def save_quantized(self, model: Any, path: str) -> None:
        """Write a loaded, quantized, LoRA-free model's weights to directory *path*."""
        ...

    def write_quantized_files(self, source: str, path: str, bits: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Write the stored quant of the local model *source* at *bits* into directory *path*, without loading it.

        Stops early, leaving an incomplete folder, once *cancelled* returns true.
        """
        ...
