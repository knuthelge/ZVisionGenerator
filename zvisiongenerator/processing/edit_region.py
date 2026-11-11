"""Crop planning, the Extend canvas and the paste-back: the pure image side of mask editing.

The model redraws a crop around the painted area, at about one megapixel (a whole-picture edit is drawn at the picture's own size); the result is pasted back through an alpha
that is 1 inside the paint, fades to 0 over ``PAINT_FADE`` px outward into the picture around the paint, and is 0
beyond, so everything further from the paint than that stays exactly as it was. An Extend draws the source scaled to
about one megapixel with a new strip of whole 16 px cells beside it; the strip is pasted whole and its join with the
source is hard: the model has already made it continuous, and its colour is matched to the source at the join.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]

# The size the model works at, in megapixels, and its pixel grid.
TARGET_MP = 1.0
ALIGN = 16
# The crop's margin around the painted box: a fraction of the box's side (``sqrt(area)``), at least this many pixels.
PAD_FRAC = 0.35
MIN_PAD = 64
# The crop is at least this wide and tall, and at most twice as long as it is short.
MIN_CROP_SIDE = 384
# A mask covering more than this fraction of the canvas takes the whole canvas.
BIG_MASK_FRACTION = 0.5
# Width in pixels of the fade outward from painted source pixels into the unpainted picture.
PAINT_FADE = 12
# The pixel width of one FLUX.2 Klein latent cell: the grid the paint is folded onto (``soft_cell_mask`` reads it).
LATENT_CELL_PIXELS = 16
# The depth in pixels of the band of source beside an Extend's strip that its colour is matched on.
JOIN_BAND = 64

_EXTEND_PADDING = {
    "top": lambda amount: ((amount, 0), (0, 0)),
    "bottom": lambda amount: ((0, amount), (0, 0)),
    "left": lambda amount: ((0, 0), (amount, 0)),
    "right": lambda amount: ((0, 0), (0, amount)),
}


@dataclass(frozen=True)
class Plan:
    """Where the model works.

    Attributes:
        box: The crop ``(x0, y0, x1, y1)`` in canvas pixels.
        target: The ``(width, height)`` the crop is drawn at.
    """

    box: tuple[int, int, int, int]
    target: tuple[int, int]


def _round_align(value: float) -> int:
    return max(ALIGN, int(round(value / ALIGN)) * ALIGN)


def target_size(width: int, height: int) -> tuple[int, int]:
    """Return the model size for a *width* x *height* crop: ``TARGET_MP`` megapixels at the crop's aspect, aligned to 16 px."""
    scale = (TARGET_MP * 1_000_000 / (width * height)) ** 0.5
    return _round_align(width * scale), _round_align(height * scale)


def grid_size(width: int, height: int) -> tuple[int, int]:
    """Return *width* x *height* with each side rounded to the nearest multiple of 16 px (at least 16): the picture's own size on the model's grid."""
    return _round_align(width), _round_align(height)


def extend_draw_size(size: tuple[int, int], side: str, amount: int) -> tuple[tuple[int, int], int]:
    """Return the ``(width, height)`` an Extend draws the source of *size* at, and the width of the new strip at that scale.

    The source is drawn at :func:`target_size`, about one megapixel, and never enlarged: a source smaller than that is
    drawn at its own size, its sides snapped down to 16 px. The strip is the *amount* at the source's scale along *side*,
    rounded to whole 16 px cells, so the source and the strip sit on whole latent cells. The drawn canvas is the model's
    reference, and both libraries shrink one over 1024 x 1024 px.
    """
    width, height = size
    if width * height < TARGET_MP * 1_000_000:
        return (max(ALIGN, width // ALIGN * ALIGN), max(ALIGN, height // ALIGN * ALIGN)), _round_align(amount)
    drawn = target_size(width, height)
    scale = drawn[0] / width if side in ("left", "right") else drawn[1] / height
    return drawn, _round_align(amount * scale)


def edit_output_size(width: int, height: int, *, extend: tuple[str, int] | None) -> tuple[int, int]:
    """Return the ``(width, height)`` an edit of a *width* x *height* picture saves.

    An Extend adds its amount on its side (the result is pasted onto the extended canvas); every other edit keeps the
    picture's size (a whole-picture edit is drawn at :func:`grid_size` and scaled back).
    """
    if extend is not None:
        side, amount = extend
        return (width + amount, height) if side in ("left", "right") else (width, height + amount)
    return (width, height)


def _fit_axis(low: int, high: int, limit: int) -> tuple[int, int]:
    """Snap ``[low, high)`` outward to the 16 px grid, then slide it inside ``[0, limit]`` without shrinking it."""
    low, high = low // ALIGN * ALIGN, -(-high // ALIGN) * ALIGN
    if high - low >= limit:
        return 0, limit
    if low < 0:
        low, high = 0, high - low
    if high > limit:
        low, high = low - (high - limit), limit
    return low, high


def plan_region(paint: np.ndarray) -> Plan:
    """Plan the crop around the painted pixels of *paint* (bool, height x width).

    The box is the painted bounding box plus ``max(MIN_PAD, PAD_FRAC * sqrt(its area))`` on every side, grown to an
    aspect between 1:2 and 2:1 and to at least ``MIN_CROP_SIDE``, snapped outward to the 16 px grid and kept inside the
    canvas. A mask covering more than ``BIG_MASK_FRACTION`` of the canvas takes the whole canvas.
    """
    height, width = paint.shape
    if paint.mean() > BIG_MASK_FRACTION:
        return Plan((0, 0, width, height), target_size(width, height))
    ys, xs = np.nonzero(paint)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad = max(MIN_PAD, round(PAD_FRAC * ((x1 - x0) * (y1 - y0)) ** 0.5))
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    box_w, box_h = max(x1 - x0, MIN_CROP_SIDE), max(y1 - y0, MIN_CROP_SIDE)
    box_w, box_h = max(box_w, -(-box_h // 2)), max(box_h, -(-box_w // 2))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = round(cx - box_w / 2), round(cx + box_w / 2)
    y0, y1 = round(cy - box_h / 2), round(cy + box_h / 2)
    x0, x1 = _fit_axis(x0, x1, width)
    y0, y1 = _fit_axis(y0, y1, height)
    return Plan((x0, y0, x1, y1), target_size(x1 - x0, y1 - y0))


def extend_canvas(image: Image.Image, side: str, amount: int) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
    """Return the source on a canvas *amount* px larger on *side*, the paint mask and where the source sits.

    The new area repeats the source's edge pixels. The model redraws it whole, but both backends VAE-encode this canvas
    and pin the clean latents of the source cells beside the seam, so the fill shapes those latents: an edge-continuing
    fill keeps the seam from reading as a hard edge. The paint mask is that new area, and the offset ``(left, top)`` is
    the source's position on the canvas. The source pixels are untouched. *side* is one of top, right, bottom or left (``parse_extend`` has checked it).
    """
    pad = _EXTEND_PADDING[side](amount)
    source = np.asarray(image.convert("RGB"))
    canvas = np.pad(source, (*pad, (0, 0)), mode="edge")
    paint = np.ones(canvas.shape[:2], dtype=bool)
    top, left = pad[0][0], pad[1][0]
    paint[top : top + source.shape[0], left : left + source.shape[1]] = False
    return Image.fromarray(canvas), paint, (left, top)


def blurred_canvas(source: Image.Image, side: str, strip: int) -> Image.Image:
    """Return *source* on a canvas *strip* px larger on *side*, the new area a heavily blurred reflection of the source.

    This is the model's reference for an Extend: it has the frame of the drawn canvas, so the source sits where it will
    be drawn, and the blur leaves the strip with colour and tone but nothing to copy. The blur is Gaussian at sigma
    ``max(24, strip / 4)``. The source pixels are untouched.
    """
    pad = _EXTEND_PADDING[side](strip)
    pixels = np.asarray(source.convert("RGB"))
    from scipy import ndimage

    sigma = max(24.0, strip / 4)
    padded = np.pad(pixels, (*pad, (0, 0)), mode="reflect").astype(np.float32)
    # A blur this wide looks the same at a quarter of the resolution, and far faster there.
    small = ndimage.gaussian_filter(padded[::4, ::4], sigma=(sigma / 4, sigma / 4, 0), mode="reflect")
    blurred = ndimage.zoom(small, (4, 4, 1), order=1)[: padded.shape[0], : padded.shape[1]]
    canvas = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    top, left = pad[0][0], pad[1][0]
    canvas[top : top + pixels.shape[0], left : left + pixels.shape[1]] = pixels
    return Image.fromarray(canvas)


# sRGB <-> CIE Lab (D65), for matching the strip's colour to the source's in a space where lightness and hue are apart.
_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    xyz = (linear @ _RGB_TO_XYZ.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27))
    linear = (xyz * _WHITE) @ np.linalg.inv(_RGB_TO_XYZ).T
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint8)


def match_join(generated: np.ndarray, model_view: np.ndarray, canvas: np.ndarray, strip: np.ndarray) -> np.ndarray:
    """Return *generated* (the model's drawn canvas, RGB) with its *strip* colour-matched to the source across the join.

    The blend pins the source region to the real source at the end, but the model drew the strip against its own version
    of that region, which drifts a little in colour; the hard paste then shows the drift as a shade line at the join.
    *model_view* is that version (the last step's latents before the pin, decoded): on a band of ``JOIN_BAND`` px of
    source next to the strip, a per-channel affine in Lab maps the model's band onto the real one in *canvas*, and the same
    map is applied to the strip. The slope is limited to 0.8 to 1.25, and a flat band (almost no variance) takes slope 1.

    Args:
        generated: The drawn canvas as the model returned it, ``(height, width, 3)`` uint8.
        model_view: The model's own view of the whole drawn canvas, same shape.
        canvas: The drawn canvas as it was given to the model (the real source beside the strip's fill).
        strip: Bool mask of the new strip.
    """
    from scipy import ndimage

    band = ~strip & (ndimage.distance_transform_edt(~strip) <= JOIN_BAND)
    model_lab, real_lab, made_lab = _to_lab(model_view), _to_lab(canvas), _to_lab(generated)
    for channel in range(3):
        x, y = model_lab[..., channel][band], real_lab[..., channel][band]
        variance = x.var()
        slope = float(np.clip(np.cov(x, y)[0, 1] / variance, 0.8, 1.25)) if variance > 4.0 else 1.0
        made_lab[..., channel][strip] = slope * made_lab[..., channel][strip] + (y.mean() - slope * x.mean())
    matched = generated.copy()
    matched[strip] = _from_lab(made_lab)[strip]
    return matched


def fade_alpha(painted: np.ndarray, fade: float) -> np.ndarray:
    """Return the paste alpha of *painted* (bool): 1 inside, then falling outward from 1 at its edge to 0 at *fade* px (smoothstep).

    The distance is measured from pixel centres, half a pixel to the edge, so the pixel beside the paint is almost 1. The
    array's own border is not an edge: a fully painted array is 1 everywhere.
    """
    if painted.all():
        return np.ones(painted.shape, dtype=np.float32)
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(~painted)
    ramp = np.clip((fade - distance) / (fade - 0.5), 0.0, 1.0)
    return np.where(painted, 1.0, ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)


def soft_cell_mask(paint: np.ndarray) -> np.ndarray:
    """Return *paint* (bool, height x width) as a float grid: the painted fraction of each latent cell (``LATENT_CELL_PIXELS``).

    Raises:
        ValueError: *paint*'s sides are not multiples of the cell.
    """
    cell = LATENT_CELL_PIXELS
    height, width = paint.shape
    if height % cell or width % cell:
        raise ValueError(f"A {width}x{height} mask is not a whole number of {cell} px cells.")
    return paint.reshape(height // cell, cell, width // cell, cell).mean(axis=(1, 3)).astype(np.float32)


def paste_paint(canvas: Image.Image, generated: Image.Image, plan: Plan, paint: np.ndarray, fade: float) -> Image.Image:
    """Paste *generated* (the model's crop, any size) over the crop box of *canvas* through the *paint* and a fade around it.

    Inside the paint the model's pixels replace the canvas whole. Around painted pixels the result fades back to the canvas
    over *fade* px (:func:`fade_alpha`), and every pixel further away keeps its exact value, as does everything outside
    ``plan.box``. *paint* is canvas-sized (bool); the crop box holds all of it. Only the painted area and the fade around
    it are resampled and written.
    """
    base = np.array(canvas.convert("RGB"))
    x0, y0, x1, y1 = plan.box
    rows, columns = np.flatnonzero(paint.any(axis=1)), np.flatnonzero(paint.any(axis=0))
    margin = math.ceil(fade)
    wx0, wx1 = max(x0, int(columns[0]) - margin), min(x1, int(columns[-1]) + 1 + margin)
    wy0, wy1 = max(y0, int(rows[0]) - margin), min(y1, int(rows[-1]) + 1 + margin)
    scale_x, scale_y = generated.width / (x1 - x0), generated.height / (y1 - y0)
    box = ((wx0 - x0) * scale_x, (wy0 - y0) * scale_y, (wx1 - x0) * scale_x, (wy1 - y0) * scale_y)
    fresh = np.asarray(generated.convert("RGB").resize((wx1 - wx0, wy1 - wy0), Image.Resampling.LANCZOS, box=box)).astype(np.float32)
    weight = fade_alpha(paint[wy0:wy1, wx0:wx1], fade)[..., None]
    current = base[wy0:wy1, wx0:wx1].astype(np.float32)
    base[wy0:wy1, wx0:wx1] = np.clip(current * (1.0 - weight) + fresh * weight, 0, 255).round().astype(np.uint8)
    return Image.fromarray(base)


def paste_strip(canvas: Image.Image, generated: Image.Image, source_box: Box, drawn_source_box: Box, drawn_size: tuple[int, int]) -> Image.Image:
    """Paste the new strip of an Extend from *generated* (the model's drawn canvas, any size) onto *canvas*.

    The strip is the part of *canvas* outside *source_box* (the source's ``(left, top, right, bottom)`` on it), replaced
    whole: the model has already made its join with the source continuous, so it does not fade. The source stays exact.
    The model drew the canvas at *drawn_size*, with the source at *drawn_source_box*, at another scale than *canvas*; the
    strip is scaled from its own part of the drawing.
    """
    base = np.array(canvas.convert("RGB"))
    generated = generated.convert("RGB")
    spans, drawn_spans = [], []
    for axis in (0, 1):
        length, drawn_length = canvas.size[axis], drawn_size[axis]
        low, high = source_box[axis], source_box[axis + 2]
        if low > 0:
            spans.append((0, low))
            drawn_spans.append((0, drawn_source_box[axis]))
        elif high < length:
            spans.append((high, length))
            drawn_spans.append((drawn_source_box[axis + 2], drawn_length))
        else:
            spans.append((0, length))
            drawn_spans.append((0, drawn_length))
    (x0, x1), (y0, y1) = spans
    scale_x, scale_y = generated.width / drawn_size[0], generated.height / drawn_size[1]
    (dx0, dx1), (dy0, dy1) = drawn_spans
    box = (dx0 * scale_x, dy0 * scale_y, min(generated.width, dx1 * scale_x), min(generated.height, dy1 * scale_y))
    base[y0:y1, x0:x1] = np.asarray(generated.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=box))
    return Image.fromarray(base)
