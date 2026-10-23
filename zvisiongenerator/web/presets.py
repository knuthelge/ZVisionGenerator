"""Store the Web UI's settings presets: named working setups saved as YAML files.

A preset is one file, ``<data dir>/presets/<mode>/<slug>.yaml`` with ``mode`` ``image`` or ``video``, holding the
keys ``name``, ``model`` and ``settings``; the folder decides the mode, and names are unique within a mode. The ``settings`` keys are the Web form field names; a ``null`` setting
means "not set". Missing presets raise ``FileNotFoundError`` and taken names ``FileExistsError``, so the routes can
map them to 404 and 409.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
import threading
from typing import Any
import unicodedata

import yaml

from zvisiongenerator.utils.atomic_write import write_text_atomic
from zvisiongenerator.utils.paths import get_ziv_data_dir

__all__ = ["SettingsPreset", "create_preset", "delete_preset", "list_presets", "rename_preset", "update_preset"]

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_SLUG_LENGTH = 64

_IMAGE_KEYS: dict[str, type] = {
    "quantize": int,
    "lora": str,
    "ratio": str,
    "size": str,
    "width": int,
    "height": int,
    "steps": int,
    "guidance": float,
    "image_strength": float,
    "runs": int,
    "scheduler": str,
    "first_sigma": float,
    "sharpen_enabled": bool,
    "sharpen_amount": float,
    "contrast_enabled": bool,
    "contrast_amount": float,
    "saturation_enabled": bool,
    "saturation_amount": float,
    "upscale": int,
    "upscale_denoise": float,
    "upscale_steps": int,
    "upscale_guidance": float,
    "upscale_sharpen": bool,
    "review": bool,
}
_VIDEO_KEYS: dict[str, type] = {
    "lora": str,
    "ratio": str,
    "size": str,
    "width": int,
    "height": int,
    "runs": int,
    "steps": int,
    "frames": int,
    "audio": bool,
    "low_memory": bool,
    "video_upscale_factor": int,
    "review": bool,
}
_KEYS = {"image": _IMAGE_KEYS, "video": _VIDEO_KEYS}

_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class SettingsPreset:
    """A saved working setup: the settings of one mode and, optionally, the model they were made for."""

    slug: str
    name: str
    mode: str
    model: str | None
    settings: dict[str, str | int | float | bool | None]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape the Web API sends."""
        return {"slug": self.slug, "name": self.name, "mode": self.mode, "model": self.model, "settings": dict(self.settings)}


def list_presets() -> list[SettingsPreset]:
    """Return every valid preset of both modes, sorted by name; invalid files are skipped."""
    presets: list[SettingsPreset] = []
    for mode in _KEYS:
        for path in sorted((_presets_dir() / mode).glob("*.yaml")):
            try:
                presets.append(_read_preset(path, mode))
            except OSError, ValueError, yaml.YAMLError:
                continue
    return sorted(presets, key=lambda preset: (preset.name.casefold(), preset.slug))


def create_preset(data: object) -> SettingsPreset:
    """Save the preset *data* (``{name, mode, model, settings}``) under the slug of its name in its mode's folder.

    Raises:
        FileExistsError: When a preset of that mode already has that slug.
        ValueError: When *data* is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError("A preset must be a mapping with the keys name, mode, model and settings.")
    preset = _parse_preset(data, data.get("mode"))
    path = _preset_path(preset.mode, preset.slug)
    with _WRITE_LOCK:
        if path.exists():
            raise FileExistsError(f"A preset named {preset.name!r} already exists.")
        _write(path, preset)
    return preset


def update_preset(mode: str, slug: str, data: object) -> SettingsPreset:
    """Overwrite the preset *slug* of *mode* with *data*, whose name must keep the same slug; a ``mode`` in *data* is ignored.

    Raises:
        FileNotFoundError: When there is no such preset.
        ValueError: When *data* is invalid or its name has another slug.
    """
    path = _preset_path(mode, slug)
    preset = _parse_preset(data, mode)
    if preset.slug != slug:
        raise ValueError(f"The name {preset.name!r} does not match the preset {slug!r}; use rename.")
    with _WRITE_LOCK:
        if not path.is_file():
            raise FileNotFoundError(f"No {mode} preset {slug!r}.")
        _write(path, preset)
    return preset


def rename_preset(mode: str, slug: str, new_name: object) -> SettingsPreset:
    """Give the preset *slug* of *mode* a new name; its file moves when the slug changes.

    Raises:
        FileNotFoundError: When there is no such preset.
        FileExistsError: When the new slug belongs to another preset of that mode.
        ValueError: When the name is invalid.
    """
    name = _clean_name(new_name)
    new_slug = _slugify(name)
    old_path, new_path = _preset_path(mode, slug), _preset_path(mode, new_slug)
    with _WRITE_LOCK:
        if not old_path.is_file():
            raise FileNotFoundError(f"No {mode} preset {slug!r}.")
        if new_slug != slug and new_path.exists():
            raise FileExistsError(f"A preset named {name!r} already exists.")
        renamed = replace(_read_preset(old_path, mode), slug=new_slug, name=name)
        _write(new_path, renamed)
        if new_slug != slug:
            old_path.unlink()
    return renamed


def delete_preset(mode: str, slug: str) -> None:
    """Delete the preset *slug* of *mode*.

    Raises:
        FileNotFoundError: When there is no such preset.
    """
    with _WRITE_LOCK:
        _preset_path(mode, slug).unlink()


def _presets_dir() -> Path:
    return get_ziv_data_dir() / "presets"


def _preset_path(mode: object, slug: str) -> Path:
    """Return the file of the preset *slug* of *mode*; raise ``ValueError`` for an unknown mode or a non-slug."""
    if mode not in _KEYS:
        raise ValueError("A preset's mode must be 'image' or 'video'.")
    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"Invalid preset slug: {slug!r}.")
    return _presets_dir() / mode / f"{slug}.yaml"


def _slugify(name: str) -> str:
    """Return the file slug of *name*: ASCII letters and digits in lowercase, other runs becoming one dash."""
    ascii_text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text)[:_MAX_SLUG_LENGTH].strip("-")
    if not slug:
        raise ValueError(f"A preset name needs at least one letter or digit: {name!r}.")
    return slug


def _clean_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A preset name must be non-empty text.")
    return name.strip()


def _parse_preset(data: object, mode: object, *, slug: str | None = None) -> SettingsPreset:
    """Build a preset of *mode* from *data*, ignoring unknown settings; every key of the mode is present, missing ones as ``None``."""
    if mode not in _KEYS:
        raise ValueError("A preset's mode must be 'image' or 'video'.")
    if not isinstance(data, dict):
        raise ValueError("A preset must be a mapping with the keys name, model and settings.")
    name = _clean_name(data.get("name"))
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("A preset's model must be text or null.")
    raw = data.get("settings") or {}
    if not isinstance(raw, dict):
        raise ValueError("A preset's settings must be a mapping.")
    settings = {key: _typed(key, raw.get(key), expected) for key, expected in _KEYS[mode].items()}
    return SettingsPreset(slug=slug or _slugify(name), name=name, mode=mode, model=model or None, settings=settings)


def _typed(key: str, value: object, expected: type) -> str | int | float | bool | None:
    """Return *value* when it is null or of the type of setting *key* (a float setting takes whole numbers too)."""
    if value is None:
        return None
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if type(value) is not expected:
        raise ValueError(f"The setting {key!r} must be of type {expected.__name__}, not {value!r}.")
    return value


def _read_preset(path: Path, mode: str) -> SettingsPreset:
    return _parse_preset(yaml.safe_load(path.read_text(encoding="utf-8")), mode, slug=path.stem)


def _write(path: Path, preset: SettingsPreset) -> None:
    document = {"name": preset.name, "model": preset.model, "settings": preset.settings}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
