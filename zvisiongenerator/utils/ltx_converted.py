"""The manifest of a converted LTX-2.3 model folder (``ziv model model --model-type ltx-2.3``).

A converted folder holds its transformer at one quantize level, and the manifest records that level and what the
loader and the memory estimates need: the storage format, and the transformer's element counts from the source.
It is written last, so a folder with a readable manifest is complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["CONVERTED_LTX_KIND", "CONVERTED_LTX_LAYOUT", "CONVERTED_LTX_MANIFEST", "read_converted_ltx", "write_converted_ltx"]

CONVERTED_LTX_MANIFEST = "ziv-ltx.json"
CONVERTED_LTX_KIND = "ltx-2.3"
CONVERTED_LTX_LAYOUT = 1


def read_converted_ltx(model_dir: Path) -> dict[str, Any] | None:
    """Return the manifest of the converted LTX folder *model_dir*, or ``None`` when it is not one (or not complete).

    A manifest from a newer layout than this version reads is treated as missing.
    """
    try:
        manifest = json.loads((model_dir / CONVERTED_LTX_MANIFEST).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(manifest, dict) or manifest.get("kind") != CONVERTED_LTX_KIND or manifest.get("layout") != CONVERTED_LTX_LAYOUT:
        return None
    if manifest.get("bits") not in (4, 8) or not all(isinstance(manifest.get(key), int) for key in ("block_matrices", "transformer_rest")):
        return None
    return manifest


def write_converted_ltx(model_dir: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* into *model_dir*, adding the kind and layout version."""
    content = {"kind": CONVERTED_LTX_KIND, "layout": CONVERTED_LTX_LAYOUT, **manifest}
    (model_dir / CONVERTED_LTX_MANIFEST).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
