"""Stored quants: quantized copies of installed image models, kept next to the source for reuse.

A model installed as ``<models dir>/<name>`` gets its quantized copy at ``<models dir>/<name>@q<bits>``.
The copy holds the backend's saved weights plus the detection files ``detect_image_model`` reads, and a
manifest recording what it was made from, so a changed source or backend format makes it stale.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.model_files import model_weight_files

__all__ = [
    "MANIFEST_NAME",
    "build_manifest",
    "copy_detection_files",
    "discard_partial",
    "is_current",
    "parse_stored_quant_name",
    "partial_dir",
    "promote_partial",
    "source_fingerprint",
    "stored_quant_dir",
    "stored_quant_name",
    "write_manifest",
]

MANIFEST_NAME = "ziv-quant.json"
_MANIFEST_VERSION = 1
_NAME_PATTERN = re.compile(r"^(?P<base>.+)@q(?P<bits>4|8)$")
# Files detect_image_model needs (family, distillation, Klein size) that backends do not save with the weights.
_DETECTION_FILES = ("model_index.json", "transformer/config.json")
_DETECTION_DIRS = ("scheduler",)


def stored_quant_name(name: str, bits: int) -> str:
    """Return the folder name of *name*'s stored quant at *bits*, e.g. ``snofs@q8``."""
    return f"{name}@q{bits}"


def parse_stored_quant_name(name: str) -> tuple[str, int] | None:
    """Return ``(base name, bits)`` when *name* is a stored-quant folder name, else ``None``."""
    match = _NAME_PATTERN.match(name)
    return (match["base"], int(match["bits"])) if match else None


def stored_quant_dir(source_dir: Path, bits: int) -> Path:
    """Return where *source_dir*'s stored quant at *bits* lives (a sibling folder)."""
    return source_dir.with_name(stored_quant_name(source_dir.name, bits))


def source_fingerprint(source_dir: Path) -> dict[str, int]:
    """Return the total size and newest modification time of the weight files the loader reads."""
    stats = [path.stat() for path in model_weight_files(source_dir)]
    return {
        "source_bytes": sum(stat.st_size for stat in stats),
        "source_mtime_ns": max((stat.st_mtime_ns for stat in stats), default=0),
    }


def build_manifest(source_dir: Path, bits: int, backend_format: str) -> dict[str, Any]:
    """Return the manifest describing a stored quant made from *source_dir*."""
    return {
        "version": _MANIFEST_VERSION,
        "source": str(source_dir),
        "bits": bits,
        "backend_format": backend_format,
        **source_fingerprint(source_dir),
    }


def is_current(stored_dir: Path, source_dir: Path, bits: int, backend_format: str) -> bool:
    """Return whether *stored_dir* holds a complete stored quant matching the source, bits and format.

    The source path is informational only, so renaming the models directory keeps copies valid.
    """
    try:
        manifest = json.loads((stored_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return False
    expected = {"version": _MANIFEST_VERSION, "bits": bits, "backend_format": backend_format, **source_fingerprint(source_dir)}
    return isinstance(manifest, dict) and all(manifest.get(key) == value for key, value in expected.items())


def write_manifest(stored_dir: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* into *stored_dir*; written last, it marks the copy complete."""
    (stored_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def copy_detection_files(source_dir: Path, stored_dir: Path) -> None:
    """Copy the detection files that exist in *source_dir* into *stored_dir* (following symlinks)."""
    for relative in _DETECTION_FILES:
        source = source_dir / relative
        if source.is_file():
            target = stored_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for relative in _DETECTION_DIRS:
        source = source_dir / relative
        if source.is_dir():
            shutil.copytree(source, stored_dir / relative, dirs_exist_ok=True)


def partial_dir(target: Path) -> Path:
    """Return a unique hidden sibling of *target* to write into before it is complete."""
    return target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.partial")


def promote_partial(partial: Path, target: Path) -> None:
    """Replace *target* with the completed *partial* folder."""
    if target.exists():
        shutil.rmtree(target)
    partial.rename(target)


def discard_partial(partial: Path) -> None:
    """Remove an incomplete *partial* folder, if any."""
    shutil.rmtree(partial, ignore_errors=True)
