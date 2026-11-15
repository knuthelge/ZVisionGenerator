"""Validate an Adjust request from the viewer and save the adjusted picture, as a copy or over the original."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.processing.adjust import MIX_NAMES, SETTING_MAX, SETTING_MIN, LightSettings, adjust_light
from zvisiongenerator.processing.reframe import CropBox, Reframe, reframe
from zvisiongenerator.utils.atomic_write import write_bytes_atomic, write_new_file
from zvisiongenerator.utils.filename import derived_stem
from zvisiongenerator.utils.input_images import to_srgb
from zvisiongenerator.utils.provenance import EXIF_IMAGE_DESCRIPTION, embed_png_config, image_prompt_text, read_png_config, with_copied_from

SAVE_COPY = "copy"
SAVE_REPLACE = "replace"
_SAVE_CHOICES = (SAVE_COPY, SAVE_REPLACE)
_ROTATIONS = (0, 90, 180, 270)
_LIGHT_FIELDS = ("exposure", "contrast", "highlights", "shadows", "saturation", "vibrance", "warmth", *(f"mix_{name}" for name in MIX_NAMES))
_CROP_FIELDS = ("x", "y", "width", "height")
_COPY_TAG = "adjusted"
NOTHING_TO_SAVE = "Nothing to save: no adjustment is set."
ONLY_PNG_REPLACED = "Only PNG files can be replaced; use Save as copy."
# Single-channel integer and float modes carry more than 8 bits, so the 8-bit adjustment would quietly lose them.
_REFUSED_MODES = ("I", "F")
_REFUSED_MODE_PREFIX = "I;16"


@dataclass(frozen=True)
class AdjustRequest:
    """A validated Adjust request: which picture, how to save it, and the framing and colour to apply."""

    asset_id: str
    save: str
    frame: Reframe
    light: LightSettings


def adjust_json_request(body: object) -> AdjustRequest:
    """Validate the JSON *body* of an Adjust request.

    Raises:
        ValueError: If the body is not an object, a field is missing or out of range, or nothing differs from neutral.
    """
    if not isinstance(body, Mapping):
        raise ValueError("Request body must be a JSON object.")
    asset_id = body.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("Field 'asset_id' must be a non-empty string.")
    save = body.get("save")
    if save not in _SAVE_CHOICES:
        raise ValueError('Field \'save\' must be "copy" or "replace".')
    rotate = body.get("rotate", 0)
    if isinstance(rotate, bool) or rotate not in _ROTATIONS:
        raise ValueError("Field 'rotate' must be 0, 90, 180 or 270.")
    flip = body.get("flip", False)
    if not isinstance(flip, bool):
        raise ValueError("Field 'flip' must be true or false.")
    frame = Reframe(rotate=rotate, flip=flip, crop=_crop_field(body.get("crop")))
    light = LightSettings(**{name: _slider_field(body, name) for name in _LIGHT_FIELDS})
    if frame == Reframe() and light.is_neutral:
        raise ValueError(NOTHING_TO_SAVE)
    return AdjustRequest(asset_id=asset_id, save=save, frame=frame, light=light)


def _slider_field(body: Mapping[str, Any], name: str) -> int:
    value = body.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or not SETTING_MIN <= value <= SETTING_MAX:
        raise ValueError(f"Field '{name}' must be an integer from {SETTING_MIN} to {SETTING_MAX}.")
    return value


def _crop_field(value: object) -> CropBox | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Field 'crop' must be null or an object with x, y, width and height.")
    parts: dict[str, int] = {}
    for name in _CROP_FIELDS:
        part = value.get(name)
        least = 0 if name in ("x", "y") else 1
        if isinstance(part, bool) or not isinstance(part, int) or part < least:
            raise ValueError(f"Field 'crop.{name}' must be an integer of {least} or more.")
        parts[name] = part
    return CropBox(**parts)


def save_adjusted(source: Path, request: AdjustRequest) -> Path:
    """Apply *request* to the picture at *source* and save it as ``<name>_adjusted.png`` beside it, or over it.

    Copy and replace both write a PNG. A copy records where it came from in its settings; a replace keeps the
    settings as they were. The source is closed before anything is written, because Windows cannot replace an open file.

    Args:
        source: An existing image file.
        request: The validated request.

    Returns:
        The path that was written.

    Raises:
        ValueError: If the file cannot be replaced or adjusted, or the crop is outside the picture.
        FileNotFoundError: If *source* does not exist.
        OSError: If the result cannot be written.
    """
    if request.save == SAVE_REPLACE and source.suffix.lower() != ".png":
        raise ValueError(ONLY_PNG_REPLACED)
    working, prompt, config = _read_source(source)
    exif = working.getexif()
    framed = reframe(working, request.frame)
    adjusted = framed if request.light.is_neutral else _adjust_colour(framed, request.light)
    if request.save == SAVE_REPLACE:
        data = _encode_png(adjusted, exif, prompt, config)
        write_bytes_atomic(source, data)
        return source
    config_out = None if config is None else with_copied_from(config, source)
    data = _encode_png(adjusted, exif, prompt, config_out)
    stem = derived_stem(source, _COPY_TAG)
    return write_new_file(source.parent / f"{stem}.png", data, stem=stem)


def _read_source(source: Path) -> tuple[Image.Image, str | None, dict[str, Any] | None]:
    """Return the upright sRGB RGB or RGBA working copy of *source*, its prompt and its recorded settings, with the file closed."""
    name = source.name
    try:
        with Image.open(source) as opened:
            if opened.mode in _REFUSED_MODES or opened.mode.startswith(_REFUSED_MODE_PREFIX):
                raise ValueError(f"{name} can't be adjusted: its pixel format ({opened.mode}) isn't supported.")
            prompt = image_prompt_text(opened)
            config = read_png_config(source) if source.suffix.lower() == ".png" else None
            working = _normalized(to_srgb(ImageOps.exif_transpose(opened)))
            working.load()
            if working is opened:
                working = working.copy()
    except FileNotFoundError:
        raise
    except Image.UnidentifiedImageError as exc:
        raise ValueError(f"{name} can't be adjusted: the file isn't a readable image.") from exc
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{name} can't be adjusted: {exc}") from exc
    return working, prompt, config


def _normalized(image: Image.Image) -> Image.Image:
    """Return *image* as RGBA when it has transparency and as RGB otherwise."""
    if image.mode == "La":
        image = image.convert("LA")
    return image.convert("RGBA" if image.has_transparency_data else "RGB")


def _adjust_colour(image: Image.Image, light: LightSettings) -> Image.Image:
    """Return *image* with *light* applied to its colour; alpha is carried over unchanged."""
    pixels = np.asarray(image)
    colour = adjust_light(np.ascontiguousarray(pixels[..., :3]), light)
    if image.mode == "RGBA":
        colour = np.concatenate([colour, pixels[..., 3:]], axis=2)
    return Image.fromarray(colour, image.mode)


def _encode_png(image: Image.Image, exif: Image.Exif, prompt: str | None, config: dict[str, Any] | None) -> bytes:
    """Encode *image* as PNG bytes carrying *exif*, the prompt (text chunk and EXIF description) and the recorded settings."""
    info = PngInfo()
    if prompt is not None:
        info.add_text("Description", prompt)
        exif[EXIF_IMAGE_DESCRIPTION] = prompt
    if config is not None:
        embed_png_config(info, config)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=info, exif=exif.tobytes())
    return buffer.getvalue()
