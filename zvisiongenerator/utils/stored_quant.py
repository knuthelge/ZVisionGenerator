"""Stored quants: quantized copies of installed image models, kept next to the source for reuse.

A model installed as ``<models dir>/<name>`` gets its quantized copy at ``<models dir>/<name>@q<bits>``.
The copy holds the backend's saved weights plus the detection files ``detect_image_model`` reads, and a
manifest recording what it was made from, so a changed source or backend format makes it stale.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.model_files import model_weight_files

__all__ = [
    "MANIFEST_NAME",
    "STORED_QUANT_BITS",
    "build_manifest",
    "copy_detection_files",
    "discard_partial",
    "flush_to_disk",
    "fsync_file",
    "is_current",
    "parse_stored_quant_name",
    "partial_dir",
    "promote_partial",
    "source_fingerprint",
    "stored_quant_dir",
    "stored_quant_dirs_for",
    "stored_quant_name",
    "stored_quant_bits",
    "sweep_stale_partials",
    "write_manifest",
]

MANIFEST_NAME = "ziv-quant.json"
STORED_QUANT_BITS = (4, 8)
_MANIFEST_VERSION = 1
_NAME_PATTERN = re.compile(r"^(?P<base>.+)@q(?P<bits>4|8)$")
# Files detect_image_model needs (family, distillation, Klein size) that backends may not save with the weights.
# The source's model_index.json always wins; a component config a backend wrote (e.g. with its quantization
# settings) is kept.
_INDEX_FILE = "model_index.json"
_COMPONENT_DETECTION_FILES = ("transformer/config.json",)
_DETECTION_DIRS = ("scheduler",)
# A running save writes to its partial folder at least every few minutes; one with nothing written for an hour
# was left by an interrupted process.
_STALE_PARTIAL_SECONDS = 60 * 60


def stored_quant_name(name: str, bits: int) -> str:
    """Return the folder name of *name*'s stored quant at *bits*, e.g. ``atlas@q8``."""
    return f"{name}@q{bits}"


def parse_stored_quant_name(name: str) -> tuple[str, int] | None:
    """Return ``(base name, bits)`` when *name* is a stored-quant folder name, else ``None``."""
    match = _NAME_PATTERN.match(name)
    return (match["base"], int(match["bits"])) if match else None


def stored_quant_dir(source_dir: Path, bits: int) -> Path:
    """Return where *source_dir*'s stored quant at *bits* lives (a sibling folder)."""
    return source_dir.with_name(stored_quant_name(source_dir.name, bits))


def stored_quant_dirs_for(models_dir: Path, name: str) -> tuple[Path, ...]:
    """Return the existing stored quants of the model picked as *name* (installed model or alias)."""
    if parse_stored_quant_name(name) is not None:
        return ()
    return tuple(path for bits in STORED_QUANT_BITS if (path := models_dir / stored_quant_name(name, bits)).is_dir())


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
    manifest = _read_manifest(stored_dir)
    expected = {"version": _MANIFEST_VERSION, "bits": bits, "backend_format": backend_format, **source_fingerprint(source_dir)}
    return manifest is not None and all(manifest.get(key) == value for key, value in expected.items())


def stored_quant_bits(stored_dir: Path) -> int | None:
    """Return the bits recorded in *stored_dir*'s manifest, or ``None`` when it is not a complete stored quant."""
    bits = (_read_manifest(stored_dir) or {}).get("bits")
    return bits if bits in STORED_QUANT_BITS else None


def _read_manifest(stored_dir: Path) -> dict[str, Any] | None:
    """Return *stored_dir*'s manifest, or ``None`` when it is missing or unreadable (an incomplete copy)."""
    try:
        manifest = json.loads((stored_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return manifest if isinstance(manifest, dict) else None


def flush_to_disk(stored_dir: Path) -> None:
    """Write every file in *stored_dir* through to disk.

    A new copy is several gigabytes of page cache the kernel cannot release until it is written back, so it is
    flushed before the model loads. Flushing also makes the copy durable before its manifest marks it complete.
    """
    for path in stored_dir.rglob("*"):
        # Hard links to the source need no flush.
        if path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1:
            fsync_file(path)


def fsync_file(path: Path) -> None:
    """Write *path* through to disk if this platform can; a file it cannot sync stays in the page cache.

    Windows only syncs handles opened for writing, so a file copied with read-only permissions is opened
    read-only and, there, cannot be synced. Flushing is a memory and durability aid, never a reason to discard
    a finished copy.
    """
    try:
        try:
            handle = open(path, "r+b")
        except PermissionError:
            handle = open(path, "rb")
        with handle:
            os.fsync(handle.fileno())
    except OSError:
        pass  # e.g. another process holding the file on Windows


def write_manifest(stored_dir: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* into *stored_dir*; written last, it marks the copy complete."""
    (stored_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def copy_detection_files(source_dir: Path, stored_dir: Path) -> None:
    """Copy the detection files that exist in *source_dir* into *stored_dir* (following symlinks).

    ``model_index.json`` is always replaced by the source's; component configs and folders the backend already
    wrote are kept.
    """
    for relative in (_INDEX_FILE, *_COMPONENT_DETECTION_FILES):
        source = source_dir / relative
        target = stored_dir / relative
        if source.is_file() and (relative == _INDEX_FILE or not target.exists()):
            target.parent.mkdir(parents=True, exist_ok=True)
            # Unlink first: the target may be a hard link to the source's own file.
            target.unlink(missing_ok=True)
            shutil.copyfile(source, target)
    for relative in _DETECTION_DIRS:
        source = source_dir / relative
        if source.is_dir() and not (stored_dir / relative).exists():
            shutil.copytree(source, stored_dir / relative)


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


def sweep_stale_partials(target: Path, *, older_than_seconds: float = _STALE_PARTIAL_SECONDS) -> None:
    """Remove *target*'s partial folders with nothing written for *older_than_seconds* (left by a killed or crashed save)."""
    cutoff = time.time() - older_than_seconds
    try:
        candidates = list(target.parent.glob(f".{target.name}.*.partial"))
    except OSError:
        return
    for partial in candidates:
        try:
            if _last_write(partial) < cutoff:
                discard_partial(partial)
        except OSError:
            continue


def _last_write(folder: Path) -> float:
    """Return the newest modification time of *folder* and everything in it."""
    return max([folder.stat().st_mtime, *(path.stat().st_mtime for path in folder.rglob("*"))])
