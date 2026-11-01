"""Name a ``#RRGGBB`` colour with the nearest plain colour word an image model understands."""

from __future__ import annotations

import math
import re

__all__ = ["COLOR_NAMES", "color_name", "parse_hex"]

_HEX_RE = re.compile(r"^#?([0-9A-Fa-f]{6})$")
_CHROMA_WEIGHT = 2.0

# Plain names image models know, with a typical swatch for each. Matching is by OKLab distance, so the set needs
# enough spread (light, mid and dark of the main hues, plus neutrals) that every colour lands near a fitting word.
# The vivid names (electric blue, neon green, fuchsia ...) and the less common ones (cerulean, periwinkle, dusty rose ...)
# come from the xkcd colour survey, with its swatches: the names people most often gave those colours.
# Every name carries a colour word: a bare object name ("eggplant", "wine") makes the image model draw the object.
# Hue and chroma count double against lightness: a dark blue-green is "petrol blue", not "charcoal grey".
# The basic presets of the system colour pickers (pure yellow, navy) need an anchor of their own, or they drift to a
# vivid neighbour ("chartreuse", "ultramarine").
COLOR_NAMES: dict[str, str] = {
    "black": "#111111",
    "charcoal grey": "#36454F",
    "slate grey": "#6B7B8C",
    "grey": "#8E8E8E",
    "silver": "#C0C0C0",
    "white": "#F7F7F5",
    "ivory": "#FFF8E7",
    "cream": "#F3E5C0",
    "beige": "#D9C6A5",
    "sand": "#C2A878",
    "tan": "#B58A5A",
    "brown": "#6B4226",
    "chocolate brown": "#3E2416",
    "rust": "#A64B2A",
    "terracotta": "#C8643B",
    "burnt orange": "#CC5500",
    "orange": "#F28C28",
    "bright orange": "#FF5B00",
    "amber": "#FFB000",
    "peach": "#FFC4A3",
    "coral": "#F27561",
    "salmon pink": "#F4A6A0",
    "blush pink": "#F2C9CF",
    "pink": "#F48FB1",
    "hot pink": "#FF3E9A",
    "fuchsia": "#ED0DD9",
    "magenta": "#D2189B",
    "red": "#D32F2F",
    "crimson": "#9B111E",
    "dusty rose": "#C0737A",
    "burgundy": "#6D1A2B",
    "wine red": "#7B0323",
    "mustard yellow": "#D6A520",
    "gold": "#D4AF37",
    "yellow": "#FFD92F",
    "bright yellow": "#FFFF14",
    "pale yellow": "#FCEFA1",
    "khaki": "#AAA662",
    "olive green": "#6B7032",
    "sage green": "#9CAF88",
    "lime green": "#9ACD32",
    "chartreuse": "#C1F80A",
    "neon green": "#0CFF0C",
    "green": "#2E9B48",
    "emerald green": "#0E7C4A",
    "jade green": "#2BAF6A",
    "forest green": "#1F4D2B",
    "mint green": "#B5E8C8",
    "seafoam green": "#7AF9AB",
    "petrol blue": "#1F4257",
    "deep teal": "#1B4D5C",
    "teal": "#0F7C7C",
    "turquoise": "#2EC4B6",
    "cyan": "#00BCD4",
    "bright cyan": "#41FDFE",
    "azure": "#069AF3",
    "cerulean": "#0485D1",
    "sky blue": "#8CC8EC",
    "powder blue": "#C6DDF0",
    "steel blue": "#5A7D9A",
    "cornflower blue": "#5170D7",
    "blue": "#1E5AA8",
    "cobalt blue": "#0047AB",
    "electric blue": "#0652FF",
    "ultramarine": "#2000B1",
    "navy blue": "#1B2A4A",
    "deep navy blue": "#000080",
    "midnight blue": "#020035",
    "lavender": "#B9A6E0",
    "periwinkle": "#8E82FE",
    "lilac": "#C8A2C8",
    "orchid": "#C875C4",
    "indigo": "#3F2A7A",
    "violet": "#7F3FBF",
    "electric purple": "#AA23FF",
    "bright purple": "#BE03FD",
    "purple": "#6A1B9A",
    "plum": "#5E2750",
    "deep purple": "#36013F",
}


def parse_hex(text: str) -> str:
    """Return *text* as an uppercase ``#RRGGBB`` colour; the ``#`` is optional.

    Raises:
        ValueError: When *text* is not a six-digit hex colour.
    """
    match = _HEX_RE.match(text.strip())
    if not match:
        raise ValueError(f"'{text}' is not a colour; use a six-digit hex code such as #1B3A4B.")
    return f"#{match.group(1).upper()}"


def color_name(hex_color: str) -> str:
    """Return the plain colour name nearest to *hex_color* (``#RRGGBB``)."""
    target = _oklab(parse_hex(hex_color))
    return min(COLOR_NAMES, key=lambda name: math.dist(_weighted(target), _weighted(_OKLAB_NAMES[name])))


def _weighted(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return OKLab coordinates with the hue and chroma axes scaled by ``_CHROMA_WEIGHT``."""
    return lab[0], lab[1] * _CHROMA_WEIGHT, lab[2] * _CHROMA_WEIGHT


def _oklab(hex_color: str) -> tuple[float, float, float]:
    """Return the OKLab coordinates of a ``#RRGGBB`` colour."""
    r, g, b = (_linear(int(hex_color[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _linear(channel: float) -> float:
    """Return an sRGB channel (0 to 1) in linear light."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


_OKLAB_NAMES = {name: _oklab(value) for name, value in COLOR_NAMES.items()}
