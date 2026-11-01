"""Pin the two Hub repositories the background removal chain loads: BiRefNet for the mask and ViTMatte for the edge.

The BiRefNet revision is also the revision of the vendored model code in ``packages/birefnet``
(``BIREFNET_REVISION`` in the ``Makefile``), so the weights are the ones the code was written for. The ViTMatte
revision is pinned on its own.
"""

from __future__ import annotations

from zvisiongenerator.utils.h3_preset import HubFiles

__all__ = ["BIREFNET", "MATTE_DOWNLOAD_SIZE_LABEL", "MATTE_LABEL", "MATTE_ROW_NAME", "VITMATTE"]

BIREFNET = HubFiles("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", ("model.safetensors",))
VITMATTE = HubFiles("hustvl/vitmatte-small-composition-1k", "6a58ad7646403c1df626fbd746900aec7361ea1d", ("config.json", "model.safetensors", "preprocessor_config.json"))

MATTE_ROW_NAME = "background-removal"
MATTE_LABEL = "Background removal"
# The one-time download of both repos, read from a dry-run plan against an empty cache (444.5 MB + 103.3 MB).
MATTE_DOWNLOAD_SIZE_LABEL = "548 MB"
