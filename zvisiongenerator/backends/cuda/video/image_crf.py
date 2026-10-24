"""Re-compress an image-to-video conditioning image at the model's H.264 CRF.

LTX-2.5 was trained on conditioning frames that went through H.264 at CRF 18, so the image is re-compressed the same
way before it conditions the first frame. This reuses diffusers' own port of that step instead of re-implementing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["MAX_CRF", "recompress"]

MAX_CRF = 51  # H.264's highest constant-rate-factor value.


def recompress(image: Image.Image, crf: int) -> Image.Image:
    """Return *image* re-compressed through H.264 at *crf*, as an RGB image.

    A *crf* of 0 skips re-compression and returns *image* unchanged. The H.264 encoder needs even dimensions, so an
    odd width or height loses its last column or row, exactly as in diffusers.

    Args:
        image: The conditioning image; converted to RGB before encoding.
        crf: The constant rate factor, 0 to 51. Higher values compress harder.

    Returns:
        The re-compressed RGB image, or *image* itself when *crf* is 0.

    Raises:
        ValueError: If *crf* is not an integer in 0..51.
    """
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= MAX_CRF:
        raise ValueError(f"crf must be an integer from 0 to {MAX_CRF}, got {crf!r}.")
    if crf == 0:
        return image

    import numpy as np
    from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf
    from PIL import Image as PILImage

    rgb = np.asarray(image.convert("RGB"))
    return PILImage.fromarray(apply_image_conditioning_crf(rgb, crf))
