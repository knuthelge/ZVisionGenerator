"""Tests for gallery metadata loading from embedded config."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.utils.provenance import embed_png_config
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.gallery import (
    GalleryAsset,
    delete_gallery_assets,
    gallery_asset_for_output_path,
    gallery_asset_to_json,
    list_gallery_assets,
    resolve_output_asset_path,
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


# ---------------------------------------------------------------------------
# Scene in the gallery JSON and Reuse params
# ---------------------------------------------------------------------------


def _reuse_web_config() -> SimpleNamespace:
    return SimpleNamespace(
        image_model_options=("zit",),
        video_model_options=("ltx-8",),
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
    )


def _reuse_params(payload: dict) -> dict[str, str]:
    query = urlsplit(payload["reuse_workspace_url"].removeprefix("#")).query
    return {key: values[0] for key, values in parse_qs(query, keep_blank_values=True).items()}


def _video_reuse(tmp_path, config_overrides: dict) -> tuple[dict, dict[str, str]]:
    """Return a video asset's gallery JSON and parsed Reuse query params for an embedded config with *config_overrides*."""
    config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "model": "ltx-8", "seed": 1, "steps": 8, "frame_count": 49, **config_overrides}
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 8)
    with patch("zvisiongenerator.web.gallery.read_mp4_config", return_value=config):
        asset = _find_asset(list_gallery_assets(str(tmp_path)), "clip.mp4")
    payload = gallery_asset_to_json(asset, _reuse_web_config())
    return payload, _reuse_params(payload)


@pytest.mark.parametrize("scene", [None, "A castle at dusk."])
def test_reuse_of_a_video_fills_the_recorded_scene_and_prompt_only(tmp_path, scene):
    payload, params = _video_reuse(tmp_path, {"prompt": "A knight rides in.", **({"scene": scene} if scene else {})})

    assert (params.get("scene"), params["prompt"]) == (scene, "A knight rides in.")
    assert not {"enhanced_scene", "enhanced_prompt"} & set(params)
    assert (payload["scene"], payload["prompt"]) == (scene, "A knight rides in.")


def test_reuse_of_an_enhanced_video_offers_inputs_and_the_generated_pair_without_scene_in_the_prompt(tmp_path):
    """The user's inputs go to Your text, the generated pair to Enhanced, and no prompt value carries scene text."""
    payload, params = _video_reuse(
        tmp_path,
        {
            "prompt": "A knight in plate armour rides across the drawbridge as the gates open.",
            "scene": "A grey stone castle at dusk, cold light, wide static shot.",
            "scene_input": "A castle at dusk.",
            "prompt_input": "A knight rides in.",
        },
    )

    assert params["scene"] == "A castle at dusk."
    assert params["prompt"] == "A knight rides in."
    assert params["enhanced_scene"] == "A grey stone castle at dusk, cold light, wide static shot."
    assert params["enhanced_prompt"] == "A knight in plate armour rides across the drawbridge as the gates open."
    assert "castle" not in params["enhanced_prompt"]
    assert payload["prompt"] == params["enhanced_prompt"]
    assert payload["scene"] == params["enhanced_scene"]


def test_reuse_of_an_enhanced_video_without_a_user_scene_omits_the_scene_input(tmp_path):
    _payload, params = _video_reuse(
        tmp_path,
        {"prompt": "A knight rides in, armour clanking.", "scene": "A castle at dusk.", "prompt_input": "A knight rides in."},
    )

    assert "scene" not in params
    assert params["prompt"] == "A knight rides in."
    assert (params["enhanced_scene"], params["enhanced_prompt"]) == ("A castle at dusk.", "A knight rides in, armour clanking.")


def test_reuse_of_an_action_only_video_carries_the_users_scene_and_only_the_enhanced_action(tmp_path):
    """The scene was not rewritten (it equals its input), so only the action is offered as Enhanced."""
    _payload, params = _video_reuse(
        tmp_path,
        {"prompt": "A knight draws his sword and raises it.", "scene": "A castle at dusk.", "scene_input": "A castle at dusk.", "prompt_input": "He draws his sword."},
    )

    assert (params["scene"], params["prompt"]) == ("A castle at dusk.", "He draws his sword.")
    assert params["enhanced_prompt"] == "A knight draws his sword and raises it."
    assert "enhanced_scene" not in params


def test_reuse_offers_no_enhanced_text_for_a_part_that_equals_its_input(tmp_path):
    _payload, params = _video_reuse(
        tmp_path,
        {"prompt": "He draws his sword.", "scene": "A grey castle at dusk.", "scene_input": "A castle at dusk.", "prompt_input": "He draws his sword."},
    )

    assert params["enhanced_scene"] == "A grey castle at dusk."
    assert "enhanced_prompt" not in params


def test_reuse_of_a_scene_only_video_carries_no_prompt_and_not_the_file_name(tmp_path):
    payload, params = _video_reuse(tmp_path, {"scene": "A castle at dusk."})

    assert params["scene"] == "A castle at dusk."
    assert "prompt" not in params
    assert payload["prompt"] == "A castle at dusk."


def test_reuse_of_an_image_keeps_only_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)
    _make_png_with_config(
        tmp_path / "pic.png",
        {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "prompt": "A castle.", "model": "zit", "seed": 1, "steps": 8},
    )

    asset = _find_asset(list_gallery_assets(str(tmp_path)), "pic.png")
    params = _reuse_params(gallery_asset_to_json(asset, _reuse_web_config()))

    assert params["prompt"] == "A castle."
    assert not {"scene", "enhanced_scene", "enhanced_prompt"} & set(params)


# ---------------------------------------------------------------------------
# Extend: capability block, source and Reuse
# ---------------------------------------------------------------------------


def _extend_config(source: Path | None, **overrides) -> dict:
    return {
        "workflow": "extend",
        "scene": "A woman in a red raincoat on a rainy street.",
        "prompt": "She opens a black umbrella and walks away.",
        "source_video": str(source) if source is not None else None,
        "extend_frames": 48,
        "frame_count": 97,
        "width": 512,
        "height": 256,
        **overrides,
    }


def test_extend_info_is_present_on_videos_and_absent_on_images(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)
    video, _params = _video_reuse(tmp_path, {"prompt": "A knight rides in."})
    _make_png_with_config(tmp_path / "pic.png", {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "prompt": "A castle.", "model": "zit"})
    image = gallery_asset_to_json(_find_asset(list_gallery_assets(str(tmp_path)), "pic.png"), _reuse_web_config())

    assert video["extend"] == {"available": True, "reason": None}
    assert image["extend"] is None


def test_extend_asset_carries_source_and_new_frames(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")

    payload, _params = _video_reuse(tmp_path, _extend_config(source))

    assert payload["source_video"] == str(source)
    assert payload["extend_frames"] == 48
    assert payload["workflow"] == "extend"
    assert payload["frame_count"] == 97


def test_other_videos_carry_no_source(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)

    payload, _params = _video_reuse(tmp_path, {"prompt": "A knight rides in."})

    assert (payload["source_video"], payload["extend_frames"]) == (None, None)


@pytest.mark.parametrize(("lora", "expected"), [(None, ""), ("/loras/style.safetensors:0.8", "/loras/style.safetensors:0.8")])
def test_reuse_always_names_the_recorded_loras_even_when_there_are_none(tmp_path, lora, expected):
    """A blank ``lora`` tells the Workspace to clear the LoRAs already chosen, so the reused job matches its source."""
    _payload, params = _video_reuse(tmp_path, {"prompt": "A knight rides in.", "lora": lora})

    assert params["lora"] == expected


def test_reuse_of_an_extend_output_returns_to_extend_with_source_length_and_text(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")

    payload, params = _video_reuse(tmp_path, _extend_config(source, lora="/loras/style.safetensors:0.8", guidance=3.0, ratio="16:9", size="m"))

    assert params == {
        "workflow": "extend",
        "source_video": str(source),
        "extend_frames": "48",
        "scene": "A woman in a red raincoat on a rainy street.",
        "prompt": "She opens a black umbrella and walks away.",
        "model": "ltx-8",
        "lora": "/loras/style.safetensors:0.8",
        "steps": "8",
        "seed": "1",
    }
    assert payload["reuse_state"]["resolved_workflow"] == "extend"
    assert payload["reuse_state"]["workflow_available"] is True
    assert payload["reuse_state"]["fallback_reasons"] == []


def test_reuse_of_an_enhanced_extend_output_offers_inputs_and_the_generated_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    config = _extend_config(source, prompt="She opens a black umbrella, rain drumming on it, and walks off.", prompt_input="She walks away.")

    _payload, params = _video_reuse(tmp_path, config)

    assert params["prompt"] == "She walks away."
    assert params["enhanced_prompt"] == "She opens a black umbrella, rain drumming on it, and walks off."
    assert params["enhanced_scene"] == "A woman in a red raincoat on a rainy street."
    assert params["scene"] == ""
    assert not {"frames", "width", "height", "ratio", "size", "image_path", "guidance"} & set(params)


def test_a_video_with_settings_but_no_text_shows_no_file_name_as_its_prompt(tmp_path):
    payload, params = _video_reuse(tmp_path, {})

    assert payload["prompt"] == ""
    assert not {"scene", "prompt"} & set(params)


def test_a_video_without_embedded_settings_still_shows_its_file_name(tmp_path):
    (tmp_path / "walk_in_rain.mp4").write_bytes(b"\x00" * 8)
    with patch("zvisiongenerator.web.gallery.read_mp4_config", return_value=None):
        asset = _find_asset(list_gallery_assets(str(tmp_path)), "walk_in_rain.mp4")

    payload = gallery_asset_to_json(asset, _reuse_web_config())

    assert payload["prompt"] == "walk in rain"
    assert asset.has_reusable_config is False
    assert set(_reuse_params(payload)) == {"workflow"}


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("case", "config"),
    [
        (
            "enhanced_source_moved",
            {"scene": "A wet street at night.", "prompt": "She waves slowly.", "scene_input": "My street.", "prompt_input": "She waves."},
        ),
        ("blank_text", {"scene": "", "prompt": ""}),
    ],
)
def test_extend_reuse_params_match_the_shared_fixture_the_frontend_loads(tmp_path, monkeypatch, case, config):
    """``frontend/src/lib/state/extendFlow.test.ts`` feeds this same file into ``loadFromUrl``; change both together."""
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    expected = json.loads((_FIXTURES / "extend_reuse_params.json").read_text(encoding="utf-8"))[case]

    payload, params = _video_reuse(tmp_path, _extend_config(Path("/clips/walk.mp4"), **config))

    assert params == expected
    if case == "blank_text":
        assert payload["prompt"] == ""  # the file name is not shown as the prompt


def test_reuse_of_an_extend_output_whose_source_moved_stays_extend_and_flags_it(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)

    payload, params = _video_reuse(tmp_path, _extend_config(tmp_path / "moved.mp4"))

    assert params["workflow"] == "extend"
    assert params["missing_source_video"] == "true"
    assert payload["reuse_state"]["resolved_workflow"] == "extend"
    assert payload["reuse_state"]["workflow_available"] is False
    assert payload["reuse_state"]["fallback_reasons"] == ["missing_source_video"]


def test_reuse_of_an_extend_output_without_the_capability_falls_back_to_text_to_video(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: False)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")

    payload, params = _video_reuse(tmp_path, _extend_config(source))

    assert params["workflow"] == "txt2vid"
    assert not {"source_video", "extend_frames", "missing_source_video"} & set(params)
    assert payload["reuse_state"]["fallback_reasons"] == ["extend_unavailable"]
    assert payload["extend"] is None


@pytest.mark.parametrize("directory", [".web_uploads", ".splice-x9y8z7"])
def test_staging_and_scratch_directories_never_list_as_gallery_assets(tmp_path, directory):
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 8)
    (tmp_path / directory).mkdir()
    (tmp_path / directory / "scratch.mp4").write_bytes(b"\x00" * 8)
    (tmp_path / directory / "scratch.png").write_bytes(b"\x00" * 8)
    (tmp_path / "kept").mkdir()
    (tmp_path / "kept" / "nested.mp4").write_bytes(b"\x00" * 8)

    ids = sorted(asset.id for asset in list_gallery_assets(str(tmp_path)))

    assert ids == ["clip.mp4", "kept/nested.mp4"]
    assert gallery_asset_for_output_path(str(tmp_path), str(tmp_path / directory / "scratch.mp4")) is None
    assert resolve_output_asset_path(tmp_path, f"{directory}/scratch.mp4") is None


def test_other_dot_directories_keep_listing_their_assets(tmp_path):
    (tmp_path / ".archive").mkdir()
    (tmp_path / ".archive" / "old.mp4").write_bytes(b"\x00" * 8)

    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == [".archive/old.mp4"]


def test_output_media_url_is_only_given_for_served_files_under_the_root(tmp_path):
    from zvisiongenerator.web.gallery import output_media_url

    served = tmp_path / "sub dir" / "clip.mp4"
    served.parent.mkdir()
    served.write_bytes(b"\x00")
    (tmp_path / "other.m4v").write_bytes(b"\x00")
    staging = tmp_path / ".web_uploads" / "clip.mp4"
    staging.parent.mkdir()
    staging.write_bytes(b"\x00")
    outside = tmp_path.parent / "outside.mp4"

    assert output_media_url(tmp_path, served) == "/media/sub%20dir/clip.mp4"
    assert output_media_url(tmp_path, tmp_path / "other.m4v") is None
    assert output_media_url(tmp_path, staging) is None
    assert output_media_url(tmp_path, outside) is None


def test_a_video_that_cannot_be_extended_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)

    def extend_info(name: str, *, upscaled: bool, vanished: bool) -> dict:
        directory = tmp_path / name
        directory.mkdir()
        config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "model": "ltx-8", "prompt": "A knight rides in."}
        if upscaled:
            config["generation"] = {"upscale": {"factor": 2, "steps": 8}}
        (directory / "clip.mp4").write_bytes(b"\x00" * 8)
        with patch("zvisiongenerator.web.gallery.read_mp4_config", return_value=config):
            asset = _find_asset(list_gallery_assets(str(directory)), "clip.mp4")
        if vanished:
            (directory / "clip.mp4").unlink()
        return gallery_asset_to_json(asset, _reuse_web_config())["extend"]

    upscaled = extend_info("upscaled", upscaled=True, vanished=False)
    vanished = extend_info("vanished", upscaled=False, vanished=True)
    both = extend_info("both", upscaled=True, vanished=True)

    assert all(info["available"] is False and info["reason"] for info in (upscaled, vanished, both))
    assert upscaled["reason"] == both["reason"] != vanished["reason"]  # an upscaled video gives that reason even when its file is gone too
