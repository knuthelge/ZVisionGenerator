"""Read the pinned files of the background removal models from the packaged model metadata."""

from __future__ import annotations

import sys

from zvisiongenerator.downloads.declaration import model_targets, role_targets
from zvisiongenerator.downloads.target import HubFiles

__all__ = ["matte_pins"]


def matte_pins() -> tuple[HubFiles, HubFiles]:
    """Return the pinned files of the two background removal repos: BiRefNet for the mask, then ViTMatte for the edge.

    They are the ``birefnet`` and ``vitmatte`` targets of the ``matte`` entry of the model metadata, which declares the same
    downloads on every platform.
    """
    pairs = model_targets("matte", sys.platform)
    birefnet, vitmatte = (role_targets(pairs, role)[0] for role in ("birefnet", "vitmatte"))
    return HubFiles(birefnet.repo_id, birefnet.revision, birefnet.allow_patterns), HubFiles(vitmatte.repo_id, vitmatte.revision, vitmatte.allow_patterns)
