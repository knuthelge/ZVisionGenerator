"""Apply contrast-adaptive sharpening (CAS) to PIL images."""

from __future__ import annotations

import numpy as np
from PIL import Image


def contrast_adaptive_sharpening(image: Image.Image, amount: float = 0.8) -> Image.Image:
    """Apply AMD Contrast Adaptive Sharpening (CAS) to a PIL Image.

    Numpy port of the ComfyUI_essentials / AMD FidelityFX CAS algorithm.
    Each color channel is sharpened independently.

    Args:
        image: Input PIL Image (RGB).
        amount: Sharpening strength from 0.0 (off) to 1.0 (max). Default 0.8.

    Returns:
        Sharpened PIL Image.
    """
    if amount <= 0:
        return image

    image = image.convert("RGB")

    # Convert to float32 in [0, 1], shape [H, W, C]
    img = np.asarray(image, dtype=np.float32) / 255.0

    # Pad by 1 pixel on spatial dims (reflect to avoid edge artifacts)
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="reflect")

    # Extract 3x3 neighborhood
    # a b c
    # d e f
    # g h i
    a = padded[:-2, :-2]  # top-left
    b = padded[:-2, 1:-1]  # top
    c = padded[:-2, 2:]  # top-right
    d = padded[1:-1, :-2]  # left
    e = padded[1:-1, 1:-1]  # center
    f = padded[1:-1, 2:]  # right
    g = padded[2:, :-2]  # bottom-left
    h = padded[2:, 1:-1]  # bottom
    i = padded[2:, 2:]  # bottom-right

    # Cross min/max (clamped to valid range)
    cross = np.stack([b, d, e, f, h])
    mn = np.clip(cross.min(axis=0), 0, None)
    mx = np.clip(cross.max(axis=0), None, 1)

    # Diagonal min/max (clamped to valid range)
    diag = np.stack([a, c, g, i])
    mn2 = np.clip(diag.min(axis=0), 0, None)
    mx2 = np.clip(diag.max(axis=0), None, 1)

    # Combine cross and diagonal contributions
    mx = mx + mx2
    mn = mn + mn2

    # Adaptive amplitude: high-contrast areas get less sharpening
    epsilon = 1e-5
    inv_mx = 1.0 / (mx + epsilon)
    amp = inv_mx * np.minimum(mn, 2.0 - mx)
    amp = np.sqrt(np.clip(amp, 0, None))

    # Weight: interpolate between -1/8 (amount=0) and -1/5 (amount=1)
    w = -amp * (amount * (1.0 / 5.0 - 1.0 / 8.0) + 1.0 / 8.0)

    # Apply cross sharpening filter: ((b+d+f+h)*w + e) / (1 + 4*w)
    output = ((b + d + f + h) * w + e) / (1.0 + 4.0 * w)

    # Clamp and convert back to uint8
    output = np.clip(output, 0, 1)
    output = (output * 255.0 + 0.5).astype(np.uint8)

    return Image.fromarray(output)
