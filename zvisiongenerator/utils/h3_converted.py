"""The manifest of a converted MiniMax H3 model folder (``ziv model model --model-type h3``).

A converted folder holds its own transformer in the platform's storage (``nf4-folded``: the folded 4-bit copy of
Windows/Linux; ``mlx-int8``: the INT8 transformer the macOS port reads) and takes every other component from the
stock repository. The manifest records how it was made: the storage, whether the turbo LoRA was merged or the
checkpoint was already step-distilled, the fold keys the copy serves (Windows/Linux), and the source. It is written
last, so a folder with a readable manifest is complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["CONVERTED_H3_KIND", "CONVERTED_H3_MANIFEST", "CONVERTED_H3_STORAGES", "read_converted_h3", "write_converted_h3"]

CONVERTED_H3_MANIFEST = "ziv-h3.json"
CONVERTED_H3_KIND = "h3"
# Each storage of a converted H3 folder, with the bits it holds the transformer at.
CONVERTED_H3_STORAGES = {"nf4-folded": 4, "mlx-int8": 8}
_LAYOUT = 1
_TURBO = ("merged", "distilled")


def read_converted_h3(model_dir: Path) -> dict[str, Any] | None:
    """Return the manifest of the converted H3 folder *model_dir*, or ``None`` when it is not one (or not complete).

    A manifest from a newer layout than this version reads is treated as missing.
    """
    try:
        manifest = json.loads((model_dir / CONVERTED_H3_MANIFEST).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(manifest, dict) or manifest.get("kind") != CONVERTED_H3_KIND or manifest.get("layout") != _LAYOUT:
        return None
    if manifest.get("storage") not in CONVERTED_H3_STORAGES or manifest.get("turbo") not in _TURBO:
        return None
    return manifest


def write_converted_h3(model_dir: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* into *model_dir*, adding the kind and layout version."""
    content = {"kind": CONVERTED_H3_KIND, "layout": _LAYOUT, **manifest}
    (model_dir / CONVERTED_H3_MANIFEST).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
