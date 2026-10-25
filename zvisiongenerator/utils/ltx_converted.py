"""The manifest of a converted LTX model folder (``ziv model model --model-type ltx-2.3`` or ``ltx-2.5``).

A converted folder holds its transformer at one quantize level, and the manifest records its kind (the converter
model type, which names the LTX version), that level and what the loader and the memory estimates need: the storage
format, and the transformer's element counts from the source. It is written last, so a folder with a readable
manifest is complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["CONVERTED_LTX_LAYOUT", "CONVERTED_LTX_MANIFEST", "LTX_FAMILY_MODEL_TYPE", "LTX_MODEL_TYPES", "read_converted_ltx", "write_converted_ltx"]

CONVERTED_LTX_MANIFEST = "ziv-ltx.json"
# The converter model type of each LTX version: also the kind of a folder converted from it. The one list of LTX versions.
LTX_MODEL_TYPES = {"ltx-2.3": "2.3", "ltx-2.5": "2.5"}
# The model type of the config's LTX family block; every other version has a ``versions`` block naming its model type.
LTX_FAMILY_MODEL_TYPE = "ltx-2.3"
CONVERTED_LTX_LAYOUT = 1


def read_converted_ltx(model_dir: Path) -> dict[str, Any] | None:
    """Return the manifest of the converted LTX folder *model_dir*, or ``None`` when it is not one (or not complete).

    A manifest from a newer layout than this version reads is treated as missing.
    """
    try:
        manifest = json.loads((model_dir / CONVERTED_LTX_MANIFEST).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(manifest, dict) or manifest.get("kind") not in LTX_MODEL_TYPES or manifest.get("layout") != CONVERTED_LTX_LAYOUT:
        return None
    if manifest.get("bits") not in (4, 8) or not all(isinstance(manifest.get(key), int) for key in ("block_matrices", "transformer_rest")):
        return None
    return manifest


def write_converted_ltx(model_dir: Path, manifest: dict[str, Any], *, version: str = "2.3") -> None:
    """Write *manifest* into *model_dir*, adding the kind of LTX-*version* and the layout version."""
    kind = next(model_type for model_type, found in LTX_MODEL_TYPES.items() if found == version)
    content = {"kind": kind, "layout": CONVERTED_LTX_LAYOUT, **manifest}
    (model_dir / CONVERTED_LTX_MANIFEST).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
