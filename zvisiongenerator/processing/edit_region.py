"""Crop planning, the Extend canvas and the paste-back: the pure image side of mask editing.

The model redraws a crop around the painted area, at about one megapixel (an Extend of a large picture, up to two); the result is pasted back through an alpha
that is 1 inside the paint, fades to 0 over ``PAINT_FADE`` px outward into the picture around painted source pixels, and
is 0 beyond, so everything further from the paint than that stays exactly as it was. An Extend's new area is 1 and its
join with the source is hard: the model has already made it continuous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]

# The size the model works at, in megapixels, and its pixel grid. An Extend draws its whole canvas at its own size up to
# ``EXTEND_TARGET_MP``: a 1 MP draw enlarges the new strip of a large picture several times over and looks blobby; 3 MP
# and more invents odd content.
TARGET_MP = 1.0
EXTEND_TARGET_MP = 2.0
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
# The families that can redraw a painted area (``supports_mask_edit`` in the model metadata), each with the pixel width of
# one latent cell: the grid the paint is folded onto. Both backends read it.
EDIT_CELL_PIXELS = {"zimage": 8, "flux2_klein": 16, "krea2": 16}

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


def extend_megapixels(width: int, height: int) -> float:
    """Return the megapixels an Extend draws a *width* x *height* canvas at: its own size up to ``EXTEND_TARGET_MP``, and ``TARGET_MP`` for a canvas of that or less."""
    return max(TARGET_MP, min(width * height / 1_000_000, EXTEND_TARGET_MP))


def target_size(width: int, height: int, megapixels: float = TARGET_MP) -> tuple[int, int]:
    """Return the model size for a *width* x *height* crop: *megapixels* at the crop's aspect, aligned to 16 px."""
    scale = (megapixels * 1_000_000 / (width * height)) ** 0.5
    return _round_align(width * scale), _round_align(height * scale)


def edit_output_size(width: int, height: int, *, painted: bool, extend: tuple[str, int] | None) -> tuple[int, int]:
    """Return the ``(width, height)`` an edit of a *width* x *height* picture saves.

    An Extend adds its amount on its side (the result is pasted onto the extended canvas); a painted edit keeps the
    picture's size; an edit with neither is the model's own picture, at :func:`target_size`.
    """
    if extend is not None:
        side, amount = extend
        return (width + amount, height) if side in ("left", "right") else (width, height + amount)
    return (width, height) if painted else target_size(width, height)


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


def plan_region(paint: np.ndarray, *, whole: bool = False, whole_megapixels: float = TARGET_MP) -> Plan:
    """Plan the crop around the painted pixels of *paint* (bool, height x width).

    The box is the painted bounding box plus ``max(MIN_PAD, PAD_FRAC * sqrt(its area))`` on every side, grown to an
    aspect between 1:2 and 2:1 and to at least ``MIN_CROP_SIDE``, snapped outward to the 16 px grid and kept inside the
    canvas. A mask covering more than ``BIG_MASK_FRACTION`` of the canvas, or ``whole=True``, takes the whole canvas, drawn at *whole_megapixels*.
    """
    height, width = paint.shape
    if whole or paint.mean() > BIG_MASK_FRACTION:
        return Plan((0, 0, width, height), target_size(width, height, whole_megapixels))
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

    The new area repeats the source's edge pixels; the model redraws it whole, so the fill only has to be neutral. The
    paint mask is that new area, and the offset ``(left, top)`` is the source's position on the canvas. The source
    pixels are untouched. *side* is one of top, right, bottom or left (``parse_extend`` has checked it).
    """
    pad = _EXTEND_PADDING[side](amount)
    source = np.asarray(image.convert("RGB"))
    canvas = np.pad(source, (*pad, (0, 0)), mode="edge")
    paint = np.ones(canvas.shape[:2], dtype=bool)
    top, left = pad[0][0], pad[1][0]
    paint[top : top + source.shape[0], left : left + source.shape[1]] = False
    return Image.fromarray(canvas), paint, (left, top)


def combined_paint(source_paint: np.ndarray, extend_paint: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """Return *source_paint* laid onto the extended canvas at *offset* ``(left, top)`` and unioned with *extend_paint*."""
    left, top = offset
    height, width = source_paint.shape
    padded = np.zeros(extend_paint.shape, dtype=bool)
    padded[top : top + height, left : left + width] = source_paint
    return padded | extend_paint


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


def soft_cell_mask(paint: np.ndarray, cell: int) -> np.ndarray:
    """Return *paint* (bool, height x width) as a float grid: the painted fraction of each ``cell``-pixel cell.

    Raises:
        ValueError: *paint*'s sides are not multiples of *cell*.
    """
    height, width = paint.shape
    if height % cell or width % cell:
        raise ValueError(f"A {width}x{height} mask is not a whole number of {cell} px cells.")
    return paint.reshape(height // cell, cell, width // cell, cell).mean(axis=(1, 3)).astype(np.float32)


def _intersect(a: Box, b: Box) -> Box | None:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _around(box: Box, inner: Box | None) -> list[Box]:
    """Return the parts of *box* outside *inner* as up to four rectangles; *inner* ``None`` leaves all of *box*."""
    if inner is None:
        return [box]
    x0, y0, x1, y1 = box
    ix0, iy0, ix1, iy1 = inner
    parts = [(x0, y0, x1, iy0), (x0, iy1, x1, y1), (x0, iy0, ix0, iy1), (ix1, iy0, x1, iy1)]
    return [part for part in parts if part[0] < part[2] and part[1] < part[3]]


def _painted_bounds(paint: np.ndarray, area: Box) -> Box | None:
    """Return the bounding box, in canvas pixels, of the painted pixels of *paint* inside *area*, or ``None`` when there are none."""
    x0, y0, x1, y1 = area
    part = paint[y0:y1, x0:x1]
    rows, columns = np.flatnonzero(part.any(axis=1)), np.flatnonzero(part.any(axis=0))
    if not rows.size:
        return None
    return x0 + int(columns[0]), y0 + int(rows[0]), x0 + int(columns[-1]) + 1, y0 + int(rows[-1]) + 1


def _fade_region(paint: np.ndarray, bounds: Box, area: Box, fade: float) -> tuple[Box, np.ndarray]:
    """Return the region within *fade* px of *bounds* inside *area*, and :func:`fade_alpha` of *paint* cut to it.

    The distance is computed on *area* cut to *bounds* widened by *fade*, so the cut changes nothing.
    """
    x0, y0, x1, y1 = bounds
    ax0, ay0, ax1, ay1 = area
    margin = math.ceil(fade)
    window = (max(ax0, x0 - margin), max(ay0, y0 - margin), min(ax1, x1 + margin), min(ay1, y1 + margin))
    alpha = fade_alpha(paint[window[1] : window[3], window[0] : window[2]], fade)
    return window, alpha


def _generated_part(generated: Image.Image, plan: Plan, region: Box) -> np.ndarray:
    """Return the part of *generated* (the model's crop, any size) that lies under *region*, resampled to the region's size."""
    bx0, by0, bx1, by1 = plan.box
    scale_x, scale_y = generated.width / (bx1 - bx0), generated.height / (by1 - by0)
    x0, y0, x1, y1 = region
    source = (
        (x0 - bx0) * scale_x,
        (y0 - by0) * scale_y,
        min(generated.width, (x1 - bx0) * scale_x),
        min(generated.height, (y1 - by0) * scale_y),
    )
    return np.asarray(generated.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=source))


def _paste(base: np.ndarray, generated: Image.Image, plan: Plan, region: Box, alpha: np.ndarray | None) -> None:
    """Write the model's pixels into *base* over *region*, through *alpha* (same shape as the region), or whole when it is ``None``."""
    x0, y0, x1, y1 = region
    fresh = _generated_part(generated, plan, region)
    if alpha is None:
        base[y0:y1, x0:x1] = fresh
        return
    current = base[y0:y1, x0:x1]
    weight = alpha[..., None]
    base[y0:y1, x0:x1] = np.clip(current.astype(np.float32) * (1.0 - weight) + fresh.astype(np.float32) * weight, 0, 255).round().astype(np.uint8)


def composite(
    canvas: Image.Image,
    generated: Image.Image,
    plan: Plan,
    paint: np.ndarray,
    fade: float,
    source_box: Box | None = None,
) -> Image.Image:
    """Paste *generated* (the model's crop, any size) over *canvas* through the paint and a fade around it.

    Inside the paint the model's pixels replace the canvas whole. Around painted source pixels the result fades back to the
    canvas over *fade* px (:func:`fade_alpha`), and every pixel further away keeps its exact value, as does everything
    outside ``plan.box``. *source_box* is where the source sits on an extended canvas ``(left, top, right, bottom)``:
    everything outside it is the Extend's new area, which is replaced whole and does not fade into the source, since the
    model has already made that join continuous. Only the painted pixels and the fade around them are read or written.
    """
    base = np.array(canvas.convert("RGB"))
    generated = generated.convert("RGB")
    height, width = paint.shape
    source = source_box if source_box is not None else (0, 0, width, height)
    for region in _around(plan.box, _intersect(source, plan.box)):
        _paste(base, generated, plan, region, None)
    bounds = _painted_bounds(paint, source)
    if bounds is not None:
        window, alpha = _fade_region(paint, bounds, source, fade)
        region = _intersect(window, plan.box)
        if region is not None:
            x0, y0, x1, y1 = region
            _paste(base, generated, plan, region, alpha[y0 - window[1] : y1 - window[1], x0 - window[0] : x1 - window[0]])
    return Image.fromarray(base)
