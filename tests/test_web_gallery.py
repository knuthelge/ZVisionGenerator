"""Tests for gallery metadata loading from embedded config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.utils.provenance import embed_png_config
from zvisiongenerator.web.gallery import (
    GalleryAsset,
    delete_gallery_assets,
    list_gallery_assets,
)


def _make_png_with_config(path: Path, payload: dict) -> None:
    """Write a small PNG with an embedded zvisiongenerator.config chunk."""
    img = Image.new("RGB", (64, 64), color="blue")
    info = PngInfo()
    embed_png_config(info, payload)
    img.save(path, pnginfo=info)


def _make_plain_png(path: Path) -> None:
    """Write a small PNG with no embedded metadata."""
    img = Image.new("RGB", (32, 32), color="red")
    img.save(path)


def _find_asset(assets: list[GalleryAsset], name: str) -> GalleryAsset:
    for asset in assets:
        if asset.name == name:
            return asset
    raise AssertionError(f"Asset {name!r} not found in gallery listing")


# ---------------------------------------------------------------------------
# Embedded PNG config is the primary metadata source
# ---------------------------------------------------------------------------


def test_embedded_png_config_drives_gallery_detail(tmp_path):
    """Gallery metadata for a PNG with embedded config should reflect that config exactly."""
    config = {
        "schema": "zvisiongenerator.config.v1",
        "workflow": "txt2img",
        "prompt": "A golden sunset over rolling hills",
        "model": "zit",
        "seed": 4242,
        "steps": 10,
        "guidance": 3.5,
        "width": 1024,
        "height": 768,
        "ratio": "4:3",
        "size": "l",
        "frame_count": None,
        "image_path": None,
        "lora": None,
    }
    img_path = tmp_path / "output.png"
    _make_png_with_config(img_path, config)

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "output.png")

    assert asset.prompt == "A golden sunset over rolling hills"
    assert asset.model_label == "zit"
    assert asset.seed == 4242
    assert asset.steps == 10
    assert asset.guidance == 3.5
    assert asset.width == 1024
    assert asset.height == 768
    assert asset.ratio == "4:3"
    assert asset.size == "l"
    assert asset.workflow == "txt2img"
    assert asset.reference_image_path is None
    assert asset.lora is None


def test_embedded_png_config_missing_model_shows_unavailable(tmp_path):
    """When embedded config omits the model field, model_label should be Unavailable."""
    config = {
        "schema": "zvisiongenerator.config.v1",
        "workflow": "txt2img",
        "prompt": "Sunset",
        "seed": 1,
        "steps": 4,
        "guidance": 2.0,
        "width": 512,
        "height": 512,
    }
    img_path = tmp_path / "nomodel.png"
    _make_png_with_config(img_path, config)

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "nomodel.png")

    assert asset.model_label == "Unavailable"


def test_embedded_png_config_with_img2img_reference(tmp_path):
    """Embedded img2img config should surface the reference image path."""
    config = {
        "schema": "zvisiongenerator.config.v1",
        "workflow": "img2img",
        "prompt": "Oil painting style",
        "model": "zit",
        "seed": 99,
        "steps": 8,
        "guidance": 1.5,
        "width": 512,
        "height": 512,
        "image_path": "/home/user/ref.png",
    }
    img_path = tmp_path / "styled.png"
    _make_png_with_config(img_path, config)

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "styled.png")

    assert asset.workflow == "img2img"
    assert asset.reference_image_path == "/home/user/ref.png"


# ---------------------------------------------------------------------------
# Adjacent JSON files are ignored for reusable metadata
# ---------------------------------------------------------------------------


def test_embedded_config_ignores_adjacent_json_sidecar(tmp_path):
    """When embedded config and adjacent JSON conflict, embedded config fields should be used."""
    embedded_config = {
        "schema": "zvisiongenerator.config.v1",
        "workflow": "txt2img",
        "prompt": "Embedded prompt wins",
        "model": "embedded-model",
        "seed": 111,
        "steps": 6,
        "guidance": 2.0,
        "width": 512,
        "height": 512,
    }
    sidecar_data = {
        "schema": "zvisiongenerator.asset-provenance.v1",
        "workflow": "img2img",
        "prompt": "Adjacent JSON prompt ignored",
        "model_name": "legacy-model",
        "seed": 999,
        "steps": 20,
        "guidance": 7.0,
        "width": 256,
        "height": 256,
    }
    img_path = tmp_path / "both.png"
    _make_png_with_config(img_path, embedded_config)
    img_path.with_suffix(".json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "both.png")

    assert asset.prompt == "Embedded prompt wins"
    assert asset.model_label == "embedded-model"
    assert asset.seed == 111
    assert asset.steps == 6
    assert asset.guidance == 2.0
    assert asset.width == 512
    assert asset.height == 512
    assert asset.workflow == "txt2img"


def test_plain_png_ignores_adjacent_json_sidecar(tmp_path):
    """A PNG without embedded config should not pick up reusable metadata from adjacent JSON."""
    sidecar_data = {
        "schema": "zvisiongenerator.asset-provenance.v1",
        "workflow": "txt2img",
        "prompt": "Ignored sidecar prompt",
        "model_name": "ignored-model",
        "seed": 55,
        "steps": 12,
        "guidance": 3.0,
        "width": 800,
        "height": 600,
        "ratio": "4:3",
        "size": "m",
    }
    img_path = tmp_path / "legacy.png"
    _make_plain_png(img_path)
    img_path.with_suffix(".json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "legacy.png")

    assert asset.prompt == "legacy"
    assert asset.model_label == "Unavailable"
    assert asset.seed is None
    assert asset.steps is None
    assert asset.guidance is None
    assert asset.width == 32
    assert asset.height == 32
    assert asset.workflow is None
    assert asset.ratio is None
    assert asset.size is None


def test_plain_png_ignores_asset_ext_json_sidecar(tmp_path):
    """The asset.ext.json sidecar name variant should not provide reusable metadata."""
    sidecar_data = {
        "prompt": "Ignored dot-json sidecar",
        "model_name": "ignored-dot-json-model",
        "seed": 77,
        "steps": 5,
        "width": 256,
        "height": 256,
    }
    img_path = tmp_path / "legacy2.png"
    _make_plain_png(img_path)
    img_path.with_name(f"{img_path.name}.json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "legacy2.png")

    assert asset.prompt == "legacy2"
    assert asset.model_label == "Unavailable"
    assert asset.seed is None
    assert asset.steps is None


# ---------------------------------------------------------------------------
# Missing metadata falls back to filename / PIL image info
# ---------------------------------------------------------------------------


def test_missing_metadata_uses_filename_and_pil_for_display_only_fields(tmp_path):
    """Assets with no embedded config should still have display labels and media dimensions."""
    img_path = tmp_path / "mystery_image.png"
    _make_plain_png(img_path)

    assets = list_gallery_assets(str(tmp_path))
    asset = _find_asset(assets, "mystery_image.png")

    assert asset.prompt == "mystery image"
    assert asset.model_label == "Unavailable"
    assert asset.width == 32
    assert asset.height == 32
    assert asset.seed is None
    assert asset.steps is None
    assert asset.guidance is None
    assert asset.workflow is None


# ---------------------------------------------------------------------------
# Embedded video config via mocked read_mp4_config
# ---------------------------------------------------------------------------


def test_embedded_mp4_config_drives_gallery_metadata(tmp_path):
    """Gallery metadata for an MP4 should use read_mp4_config payload when present."""
    embedded_config = {
        "schema": "zvisiongenerator.config.v1",
        "workflow": "txt2vid",
        "prompt": "Neon city flythrough",
        "model": "ltx-8",
        "seed": 321,
        "steps": 8,
        "guidance": None,
        "width": 704,
        "height": 448,
        "ratio": None,
        "size": None,
        "frame_count": 49,
        "image_path": None,
        "lora": None,
    }
    mp4_path = tmp_path / "clip.mp4"
    mp4_path.write_bytes(b"\x00" * 8)  # Minimal placeholder; read_mp4_config is mocked.

    with patch("zvisiongenerator.web.gallery.read_mp4_config", return_value=embedded_config):
        assets = list_gallery_assets(str(tmp_path))

    asset = _find_asset(assets, "clip.mp4")

    assert asset.prompt == "Neon city flythrough"
    assert asset.model_label == "ltx-8"
    assert asset.seed == 321
    assert asset.steps == 8
    assert asset.frame_count == 49
    assert asset.width == 704
    assert asset.height == 448
    assert asset.workflow == "txt2vid"


def test_embedded_mp4_config_ignores_mp4_sidecar(tmp_path):
    """Embedded MP4 config should be used while adjacent JSON is ignored."""
    embedded_config = {
        "schema": "zvisiongenerator.config.v1",
        "workflow": "txt2vid",
        "prompt": "Embedded video prompt",
        "model": "ltx-8",
        "seed": 1,
        "steps": 8,
        "width": 704,
        "height": 448,
        "frame_count": 49,
    }
    sidecar_data = {
        "prompt": "Ignored video sidecar",
        "model_name": "old-model",
        "seed": 9999,
        "steps": 50,
    }
    mp4_path = tmp_path / "dual.mp4"
    mp4_path.write_bytes(b"\x00" * 8)
    mp4_path.with_suffix(".json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    with patch("zvisiongenerator.web.gallery.read_mp4_config", return_value=embedded_config):
        assets = list_gallery_assets(str(tmp_path))

    asset = _find_asset(assets, "dual.mp4")

    assert asset.prompt == "Embedded video prompt"
    assert asset.model_label == "ltx-8"
    assert asset.seed == 1


def test_mp4_without_embedded_config_ignores_sidecar(tmp_path):
    """An MP4 with no embedded config should not use adjacent JSON metadata."""
    sidecar_data = {
        "prompt": "Ignored fallback video sidecar",
        "model_name": "ignored-sidecar-model",
        "seed": 77,
        "steps": 10,
        "frame_count": 25,
        "width": 512,
        "height": 320,
    }
    mp4_path = tmp_path / "nosidebar.mp4"
    mp4_path.write_bytes(b"\x00" * 8)
    mp4_path.with_suffix(".json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    with patch("zvisiongenerator.web.gallery.read_mp4_config", return_value=None):
        assets = list_gallery_assets(str(tmp_path))

    asset = _find_asset(assets, "nosidebar.mp4")

    assert asset.prompt == "nosidebar"
    assert asset.model_label == "Unavailable"
    assert asset.seed is None
    assert asset.steps is None
    assert asset.frame_count is None


# ---------------------------------------------------------------------------
# Delete removes only selected media assets
# ---------------------------------------------------------------------------


def test_delete_preserves_adjacent_json_files(tmp_path):
    """Deleting a gallery asset should leave adjacent JSON files untouched."""
    img_path = tmp_path / "deleteme.png"
    _make_plain_png(img_path)
    sidecar_a = img_path.with_suffix(".json")
    sidecar_b = img_path.with_name(f"{img_path.name}.json")
    sidecar_a.write_text("{}", encoding="utf-8")
    sidecar_b.write_text("{}", encoding="utf-8")

    delete_gallery_assets(str(tmp_path), ["deleteme.png"])

    assert not img_path.exists()
    assert sidecar_a.exists()
    assert sidecar_b.exists()


def test_delete_media_asset_succeeds_without_adjacent_json(tmp_path):
    """Deleting an asset that has no adjacent JSON should succeed without error."""
    img_path = tmp_path / "nosidecar.png"
    _make_plain_png(img_path)

    delete_gallery_assets(str(tmp_path), ["nosidecar.png"])

    assert not img_path.exists()


# ---------------------------------------------------------------------------
# Error resilience: embedded config read failure does not consult sidecars
# ---------------------------------------------------------------------------


def test_embedded_config_read_error_lists_plain_asset_without_sidecar_metadata(tmp_path):
    """When read_png_config raises, adjacent JSON should still be ignored."""
    sidecar_data = {"prompt": "Ignored adjacent JSON on read error", "model_name": "ignored-model", "seed": 88}
    img_path = tmp_path / "broken_meta.png"
    _make_plain_png(img_path)
    img_path.with_suffix(".json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    with patch("zvisiongenerator.web.gallery.read_png_config", side_effect=Exception("corrupt metadata")):
        assets = list_gallery_assets(str(tmp_path))

    asset = _find_asset(assets, "broken_meta.png")

    assert asset.prompt == "broken meta"
    assert asset.model_label == "Unavailable"
    assert asset.seed is None
