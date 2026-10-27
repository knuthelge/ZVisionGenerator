"""Tests for gallery metadata loading from embedded config."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.upscale import WORK_DIR_PREFIX
from zvisiongenerator.utils.provenance import build_video_upscale_payload, embed_png_config
from zvisiongenerator.web import gallery
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.gallery import (
    EXTEND_UPSCALED_REASON,
    GalleryAsset,
    _scan_output_root,
    delete_gallery_assets,
    gallery_asset_for_output_path,
    gallery_asset_to_json,
    list_gallery_assets,
    normalize_asset_id,
    resolve_output_asset_path,
)
from zvisiongenerator.web.gallery_index import GALLERY_INDEX_VERSION, GalleryIndex


def _make_png_with_config(path: Path, payload: dict) -> None:
    """Write a small PNG with an embedded zvisiongenerator.config chunk."""
    img = Image.new("RGB", (64, 64), color="blue")
    info = PngInfo()
    embed_png_config(info, payload)
    img.save(path, pnginfo=info)


def _video_probe_returning(config: dict | None, *, size: tuple[int, int] | None = None):
    """Return a patch that makes the gallery's one ffprobe call report *config* as the embedded payload and *size* as the video stream."""
    tags = {"zvisiongenerator.config": json.dumps(config)} if config is not None else {}
    streams = [{"codec_type": "video", "codec_name": "h264", "width": size[0], "height": size[1], "r_frame_rate": "24/1", "avg_frame_rate": "24/1"}] if size else []
    return patch("zvisiongenerator.web.gallery.ffprobe_json", return_value={"streams": streams, "format": {"tags": tags}})


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
# Missing metadata falls back to filename / PIL image info
# ---------------------------------------------------------------------------


def test_missing_metadata_falls_back_to_the_filename_and_pil_and_an_unavailable_model(tmp_path):
    """Assets with no embedded config, or a config without a model, still have display labels and media dimensions."""
    _make_plain_png(tmp_path / "mystery_image.png")
    _make_png_with_config(tmp_path / "nomodel.png", {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "prompt": "Sunset", "seed": 1})

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
    assert _find_asset(assets, "nomodel.png").model_label == "Unavailable"


# ---------------------------------------------------------------------------
# Embedded video config via a mocked ffprobe
# ---------------------------------------------------------------------------


def test_embedded_mp4_config_drives_gallery_metadata(tmp_path):
    """Gallery metadata for an MP4 should use the embedded config payload when present."""
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
    mp4_path.write_bytes(b"\x00" * 8)  # Minimal placeholder; ffprobe is mocked.

    with _video_probe_returning(embedded_config):
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


# ---------------------------------------------------------------------------
# Scene in the gallery JSON and Reuse params
# ---------------------------------------------------------------------------


def _reuse_web_config() -> SimpleNamespace:
    return SimpleNamespace(
        image_model_options=("zit",),
        video_model_options=("ltx-8",),
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
        seedvr2=None,
    )


def _reuse_params(payload: dict) -> dict[str, str]:
    query = urlsplit(payload["reuse_workspace_url"].removeprefix("#")).query
    return {key: values[0] for key, values in parse_qs(query, keep_blank_values=True).items()}


def _video_reuse(tmp_path, config_overrides: dict) -> tuple[dict, dict[str, str]]:
    """Return a video asset's gallery JSON and parsed Reuse query params for an embedded config with *config_overrides*."""
    config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "model": "ltx-8", "seed": 1, "steps": 8, "frame_count": 49, **config_overrides}
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 8)
    with _video_probe_returning(config):
        asset = _find_asset(list_gallery_assets(str(tmp_path)), "clip.mp4")
    payload = gallery_asset_to_json(asset, _reuse_web_config())
    return payload, _reuse_params(payload)


@pytest.mark.parametrize(
    ("overrides", "params_include", "params_absent", "payload_include"),
    [
        pytest.param(
            {"prompt": "A knight rides in."},
            {"prompt": "A knight rides in."},
            {"scene", "enhanced_scene", "enhanced_prompt"},
            {"scene": None, "prompt": "A knight rides in."},
            id="prompt-only",
        ),
        pytest.param(
            {"prompt": "A knight rides in.", "scene": "A castle at dusk."},
            {"scene": "A castle at dusk.", "prompt": "A knight rides in."},
            {"enhanced_scene", "enhanced_prompt"},
            {"scene": "A castle at dusk.", "prompt": "A knight rides in."},
            id="scene-and-prompt",
        ),
        pytest.param(
            {
                "prompt": "A knight in plate armour rides across the drawbridge as the gates open.",
                "scene": "A grey stone castle at dusk, cold light, wide static shot.",
                "scene_input": "A castle at dusk.",
                "prompt_input": "A knight rides in.",
            },
            {
                "scene": "A castle at dusk.",
                "prompt": "A knight rides in.",
                "enhanced_scene": "A grey stone castle at dusk, cold light, wide static shot.",
                "enhanced_prompt": "A knight in plate armour rides across the drawbridge as the gates open.",
            },
            set(),
            {
                "scene": "A grey stone castle at dusk, cold light, wide static shot.",
                "prompt": "A knight in plate armour rides across the drawbridge as the gates open.",
            },
            id="enhanced-pair",
        ),
        pytest.param(
            {"prompt": "A knight rides in, armour clanking.", "scene": "A castle at dusk.", "prompt_input": "A knight rides in."},
            {"prompt": "A knight rides in.", "enhanced_scene": "A castle at dusk.", "enhanced_prompt": "A knight rides in, armour clanking."},
            {"scene"},
            {},
            id="enhanced-without-a-user-scene",
        ),
        pytest.param(
            {"prompt": "A knight draws his sword and raises it.", "scene": "A castle at dusk.", "scene_input": "A castle at dusk.", "prompt_input": "He draws his sword."},
            {"scene": "A castle at dusk.", "prompt": "He draws his sword.", "enhanced_prompt": "A knight draws his sword and raises it."},
            {"enhanced_scene"},
            {},
            id="action-only-enhanced",
        ),
        pytest.param(
            {"prompt": "He draws his sword.", "scene": "A grey castle at dusk.", "scene_input": "A castle at dusk.", "prompt_input": "He draws his sword."},
            {"enhanced_scene": "A grey castle at dusk."},
            {"enhanced_prompt"},
            {},
            id="part-equal-to-its-input",
        ),
        pytest.param(
            {"scene": "A castle at dusk."},
            {"scene": "A castle at dusk."},
            {"prompt"},
            {"prompt": "A castle at dusk."},
            id="scene-only-never-the-file-name",
        ),
    ],
)
def test_reuse_of_a_video_fills_the_recorded_inputs_and_the_generated_pair(tmp_path, overrides, params_include, params_absent, payload_include):
    """The user's inputs go to Your text, the generated pair to Enhanced only where it differs from its input."""
    payload, params = _video_reuse(tmp_path, overrides)

    assert {key: params[key] for key in params_include} == params_include
    assert not params_absent & set(params)
    assert {key: payload[key] for key in payload_include} == payload_include


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


def test_reuse_of_a_json_caption_image_fills_the_json_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)
    caption = '{"high_level_description":"A {red} fox."}'
    _make_png_with_config(tmp_path / "pic.png", {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "prompt": caption, "model": "ideo", "seed": 1, "steps": 20})

    params = _reuse_params(gallery_asset_to_json(_find_asset(list_gallery_assets(str(tmp_path)), "pic.png"), _reuse_web_config()))

    assert params["json_prompt"] == caption
    assert "prompt" not in params


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
    with _video_probe_returning(None):
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


@pytest.mark.parametrize("directory", [".web_uploads", ".splice-x9y8z7", f"{WORK_DIR_PREFIX}x"])
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
        with _video_probe_returning(config):
            asset = _find_asset(list_gallery_assets(str(directory)), "clip.mp4")
        if vanished:
            (directory / "clip.mp4").unlink()
        return gallery_asset_to_json(asset, _reuse_web_config())["extend"]

    upscaled = extend_info("upscaled", upscaled=True, vanished=False)
    vanished = extend_info("vanished", upscaled=False, vanished=True)
    both = extend_info("both", upscaled=True, vanished=True)

    assert all(info["available"] is False and info["reason"] for info in (upscaled, vanished, both))
    assert upscaled["reason"] == both["reason"] != vanished["reason"]  # an upscaled video gives that reason even when its file is gone too


# ---------------------------------------------------------------------------
# How a prompt was enhanced: the viewer summary and Reuse
# ---------------------------------------------------------------------------

_ENHANCE = {"settings": {"style": "cinematic", "mood": "melancholic", "details": [], "length": "longer"}, "model": "org/fast@r1", "tier": "fast", "format": "text"}


def _image_reuse(tmp_path, config_overrides: dict) -> tuple[dict, dict[str, str]]:
    config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "model": "zit", "seed": 1, "steps": 8, "prompt": "A red fox curls up in deep snow.", **config_overrides}
    _make_png_with_config(tmp_path / "fox.png", config)
    payload = gallery_asset_to_json(_find_asset(list_gallery_assets(str(tmp_path)), "fox.png"), _reuse_web_config())
    return payload, _reuse_params(payload)


def test_an_enhanced_image_shows_how_and_reuses_both_texts_and_the_options(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)
    payload, params = _image_reuse(tmp_path, {"prompt_input": "a fox in snow", "enhance": _ENHANCE})
    assert payload["enhance"]["summary"] == "Fast · Cinematic, Melancholic, Longer"
    assert payload["enhance"]["prompt_input"] == "a fox in snow"
    assert (params["prompt"], params["enhanced_prompt"]) == ("a fox in snow", "A red fox curls up in deep snow.")
    assert json.loads(params["enhance"])["settings"]["style"] == "cinematic"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"generation": {"quantize": 8}}, "8"),
        ({}, ""),
        # A stored-quant id falls back to the default model, which the level was not recorded with.
        ({"model": "zit@q4", "generation": {"quantize": 4}}, ""),
    ],
)
def test_reuse_of_an_image_names_its_quantize_level_even_when_there_is_none(tmp_path, monkeypatch, overrides, expected):
    """A blank ``quantize`` tells the Workspace to clear the level already chosen, so the reused job matches its source."""
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)
    _payload, params = _image_reuse(tmp_path, overrides)

    assert params["quantize"] == expected


def test_a_first_and_last_frame_clip_reuses_both_images_and_lists_the_last_frame(tmp_path):
    payload, params = _video_reuse(tmp_path, {"workflow": "img2vid", "image_path": "/in/start.png", "end_image_path": "/in/end.png"})

    assert params["workflow"] == "img2vid"
    assert (params["image_path"], params["end_image_path"]) == ("/in/start.png", "/in/end.png")
    assert payload["end_image_path"] == "/in/end.png"


def test_reuse_drops_the_last_frame_with_its_missing_first_frame(tmp_path):
    payload, params = _video_reuse(tmp_path, {"workflow": "img2vid", "end_image_path": "/in/end.png"})

    assert params["workflow"] == "txt2vid"
    assert "image_path" not in params and "end_image_path" not in params
    assert payload["end_image_path"] == "/in/end.png"


def test_a_clip_without_a_last_frame_reuses_none(tmp_path):
    payload, params = _video_reuse(tmp_path, {"workflow": "img2vid", "image_path": "/in/start.png"})

    assert payload["end_image_path"] is None
    assert "end_image_path" not in params


def test_reuse_of_a_video_leaves_quantize_to_the_model_alias(tmp_path):
    _payload, params = _video_reuse(tmp_path, {"prompt": "A knight rides in.", "generation": {"quantize": 8}})

    assert "quantize" not in params


def test_a_plain_image_has_no_enhance_record(tmp_path, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)
    payload, params = _image_reuse(tmp_path, {})
    assert payload["enhance"] is None
    assert params["prompt"] == "A red fox curls up in deep snow." and "enhanced_prompt" not in params and "enhance" not in params


def test_an_enhanced_video_reuses_its_options(tmp_path):
    payload, params = _video_reuse(tmp_path, {"prompt": "A fox sprints.", "prompt_input": "a fox runs", "enhance": {**_ENHANCE, "settings": {"style": "cinematic", "camera": "orbit"}}})
    assert json.loads(params["enhance"])["settings"]["camera"] == "orbit"


# ---------------------------------------------------------------------------
# Gallery index: persisted facts, reconciliation and per-request fields
# ---------------------------------------------------------------------------

_CONFIG = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "prompt": "A castle.", "model": "zit", "seed": 3, "steps": 8}


def _backdate(*paths: Path) -> None:
    """Age files past the index's settle window so their facts are stored."""
    stamp = time.time() - 60
    for path in paths:
        os.utime(path, (stamp, stamp))


def _aged_png(path: Path, payload: dict | None = None) -> Path:
    """Write an aged PNG, with an embedded config when *payload* is given."""
    if payload is None:
        _make_plain_png(path)
    else:
        _make_png_with_config(path, payload)
    _backdate(path)
    return path


def _stored_rel_paths() -> list[str]:
    """Return the asset IDs the gallery index has stored, read straight from the database file."""
    path = app_db.app_db_path()
    if not path.exists():
        return []
    with closing(sqlite3.connect(path)) as conn:
        return sorted(row[0] for row in conn.execute(f"SELECT rel_path FROM gallery_assets_v{GALLERY_INDEX_VERSION}"))


def _restart_index(monkeypatch) -> None:
    """Swap in a new index on the same database file, as a server restart would."""
    monkeypatch.setattr("zvisiongenerator.web.gallery._GALLERY_INDEX", GalleryIndex(app_db.app_db_path))


def _use_unusable_index(monkeypatch, tmp_path: Path) -> None:
    """Swap in an index whose database path is a directory, so every database operation fails."""
    broken = tmp_path / "db-is-a-directory"
    broken.mkdir()
    monkeypatch.setattr("zvisiongenerator.web.gallery._GALLERY_INDEX", GalleryIndex(lambda: broken))


@pytest.fixture
def reads(monkeypatch) -> list[str]:
    """Record the name of every file whose facts are read from disk, in order."""
    real = gallery._read_asset_facts
    names: list[str] = []

    def counting(path, kind):
        names.append(path.name)
        return real(path, kind)

    monkeypatch.setattr("zvisiongenerator.web.gallery._read_asset_facts", counting)
    return names


def test_app_db_path_is_isolated_per_test(_isolated_app_db):
    assert app_db.app_db_path().parent == _isolated_app_db()
    assert _isolated_app_db() != Path.home() / ".ziv"


def test_a_changed_file_is_read_again(tmp_path, reads):
    png = _aged_png(tmp_path / "a.png", _CONFIG)
    list_gallery_assets(str(tmp_path))

    _aged_png(png, {**_CONFIG, "prompt": "A tower."})
    assets = list_gallery_assets(str(tmp_path))

    assert reads == ["a.png", "a.png"]
    assert assets[0].prompt == "A tower."


def test_a_file_deleted_outside_the_app_is_never_shown_and_its_row_is_pruned(tmp_path):
    gone = _aged_png(tmp_path / "gone.png")
    _aged_png(tmp_path / "kept.png")
    list_gallery_assets(str(tmp_path))
    assert _stored_rel_paths() == ["gone.png", "kept.png"]

    gone.unlink()

    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == ["kept.png"]
    assert _stored_rel_paths() == ["kept.png"]


def test_a_missing_root_lists_nothing_and_keeps_its_rows(tmp_path, reads):
    root = tmp_path / "out"
    root.mkdir()
    _aged_png(root / "a.png")
    _aged_png(root / "b.png")
    list_gallery_assets(str(root))
    reads.clear()

    root.rename(tmp_path / "away")
    assert list_gallery_assets(str(root)) == []
    assert _stored_rel_paths() == ["a.png", "b.png"]

    (tmp_path / "away").rename(root)
    assert sorted(asset.id for asset in list_gallery_assets(str(root))) == ["a.png", "b.png"]
    assert reads == []


def test_an_emptied_root_keeps_its_rows(tmp_path):
    first = _aged_png(tmp_path / "a.png")
    second = _aged_png(tmp_path / "b.png")
    list_gallery_assets(str(tmp_path))

    first.unlink()
    second.unlink()

    assert list_gallery_assets(str(tmp_path)) == []
    assert _stored_rel_paths() == ["a.png", "b.png"]


def test_a_walk_error_marks_the_scan_incomplete_and_prunes_nothing(tmp_path, monkeypatch):
    _aged_png(tmp_path / "a.png")
    _aged_png(tmp_path / "b.png")
    list_gallery_assets(str(tmp_path))

    def failing_walk(top, onerror=None, **_kwargs):
        onerror(PermissionError("denied"))
        yield str(top), [], ["a.png"]

    monkeypatch.setattr("zvisiongenerator.web.gallery.os.walk", failing_walk)
    files, walk_complete = _scan_output_root(tmp_path)
    assets = list_gallery_assets(str(tmp_path))

    assert walk_complete is False
    assert [file.rel_path for file in files] == ["a.png"]
    assert [asset.id for asset in assets] == ["a.png"]
    assert _stored_rel_paths() == ["a.png", "b.png"]


def test_a_file_that_vanishes_during_the_walk_is_skipped(tmp_path, monkeypatch):
    _make_plain_png(tmp_path / "real.png")

    def walk_with_ghost(top, **_kwargs):
        yield str(top), [], ["ghost.png", "real.png"]

    monkeypatch.setattr("zvisiongenerator.web.gallery.os.walk", walk_with_ghost)

    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == ["real.png"]


def test_an_upscale_source_id_follows_the_source_without_reading_again(tmp_path, reads):
    source = _aged_png(tmp_path / "src.png", _CONFIG)
    upscale_config = {**_CONFIG, "source": {"path": str(source), "width": 64, "height": 64, "workflow": "txt2img"}}
    _aged_png(tmp_path / "up.png", upscale_config)

    def source_id() -> str | None:
        return _find_asset(list_gallery_assets(str(tmp_path)), "up.png").source["id"]

    assert source_id() == "src.png"
    reads.clear()

    source.unlink()
    assert source_id() is None
    _aged_png(source, _CONFIG)
    assert source_id() == "src.png"
    assert "up.png" not in reads


def test_an_extension_flags_a_deleted_source_video_without_reading_again(tmp_path, monkeypatch, reads):
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)
    _backdate(source, clip)
    config = {"schema": "zvisiongenerator.config.v1", **_extend_config(source), "model": "ltx-8"}
    with patch("zvisiongenerator.web.gallery.ffprobe_json", return_value={}), patch("zvisiongenerator.web.gallery.mp4_config_from_probe", return_value=config):
        list_gallery_assets(str(tmp_path))
    reads.clear()

    def fallback_reasons() -> list[str]:
        asset = _find_asset(list_gallery_assets(str(tmp_path)), "clip.mp4")
        return gallery_asset_to_json(asset, _reuse_web_config())["reuse_state"]["fallback_reasons"]

    assert "missing_source_video" not in fallback_reasons()
    source.unlink()
    assert "missing_source_video" in fallback_reasons()
    assert reads == []


def test_deleting_an_asset_forgets_its_row_before_any_listing(tmp_path):
    _aged_png(tmp_path / "a.png")
    _aged_png(tmp_path / "b.png")
    list_gallery_assets(str(tmp_path))

    delete_gallery_assets(str(tmp_path), ["a.png"])

    assert not (tmp_path / "a.png").exists()
    assert _stored_rel_paths() == ["b.png"]


def test_deleting_an_asset_succeeds_when_the_database_is_unusable(tmp_path, monkeypatch):
    _aged_png(tmp_path / "a.png")
    _use_unusable_index(monkeypatch, tmp_path)

    with pytest.warns(UserWarning):
        delete_gallery_assets(str(tmp_path), ["a.png"])

    assert not (tmp_path / "a.png").exists()


def test_an_unusable_database_falls_back_to_reading_the_files(tmp_path, monkeypatch):
    _aged_png(tmp_path / "a.png", _CONFIG)
    _use_unusable_index(monkeypatch, tmp_path)

    with pytest.warns(UserWarning):
        assets = list_gallery_assets(str(tmp_path))

    assert [(asset.id, asset.prompt) for asset in assets] == [("a.png", "A castle.")]


def test_indexed_assets_equal_freshly_read_ones_except_the_age_label(tmp_path, monkeypatch, reads):
    _aged_png(tmp_path / "configured.png", _CONFIG)
    _aged_png(tmp_path / "plain_512x768_steps4_seed9.png")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)
    _backdate(clip)
    video_config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "model": "ltx-8", "scene": "Rain.", "frame_count": 49}
    with patch("zvisiongenerator.web.gallery.ffprobe_json", return_value={}), patch("zvisiongenerator.web.gallery.mp4_config_from_probe", return_value=video_config):
        cold = list_gallery_assets(str(tmp_path))
    reads.clear()
    _restart_index(monkeypatch)

    warm = list_gallery_assets(str(tmp_path))

    assert reads == []
    assert len(cold) == len(warm) == 3

    def comparable(assets: list[GalleryAsset]) -> list[dict]:
        return [{key: value for key, value in asdict(asset).items() if key != "modified_label"} for asset in assets]

    assert comparable(warm) == comparable(cold)
    assert all(asset.modified_at == (tmp_path / asset.id).stat().st_mtime for asset in warm)


def test_assets_with_equal_times_list_by_descending_id(tmp_path):
    names = ("a.png", "c.png", "b.png")
    for name in names:
        _make_plain_png(tmp_path / name)
    stamp = time.time() - 60
    for name in names:
        os.utime(tmp_path / name, ns=(int(stamp * 1e9),) * 2)

    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == ["c.png", "b.png", "a.png"]


def test_building_one_output_path_does_not_create_the_database(tmp_path):
    png = _aged_png(tmp_path / "a.png", _CONFIG)

    asset = gallery_asset_for_output_path(str(tmp_path), str(png))

    assert asset is not None
    assert asset.prompt == "A castle."
    assert not app_db.app_db_path().exists()


def test_a_hidden_rewrite_temp_is_not_listed_served_or_deletable(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 8)
    temp = tmp_path / ".ziv-tmp-clip.mp4"
    temp.write_bytes(b"\x00" * 8)

    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == ["clip.mp4"]
    assert gallery_asset_for_output_path(str(tmp_path), str(temp)) is None
    assert resolve_output_asset_path(tmp_path, temp.name) is None
    assert normalize_asset_id(temp.name) is None
    delete_gallery_assets(str(tmp_path), [temp.name])
    assert temp.exists()


def test_the_walk_never_enters_or_stats_staging_directories(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 8)
    for directory in (".ziv-tmp-x", ".splice-x"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "inside.mp4").write_bytes(b"\x00" * 8)
    visited: list[str] = []
    statted: list[str] = []
    real_walk = os.walk
    real_scanned_file = gallery._scanned_file

    def recording_walk(top, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, **kwargs):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    def recording_scanned_file(root, path):
        statted.append(path.name)
        return real_scanned_file(root, path)

    monkeypatch.setattr("zvisiongenerator.web.gallery.os.walk", recording_walk)
    monkeypatch.setattr("zvisiongenerator.web.gallery._scanned_file", recording_scanned_file)

    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == ["clip.mp4"]
    assert visited == [str(tmp_path)]
    assert statted == ["clip.mp4"]


def test_a_failed_config_read_is_listed_but_not_stored(tmp_path):
    _aged_png(tmp_path / "broken.png", _CONFIG)

    with patch("zvisiongenerator.web.gallery.read_png_config", side_effect=Exception("corrupt")) as reader:
        first = list_gallery_assets(str(tmp_path))
        second = list_gallery_assets(str(tmp_path))

    assert [(asset.id, asset.prompt) for asset in first] == [("broken.png", "broken")]
    assert [asset.id for asset in second] == ["broken.png"]
    assert reader.call_count == 2
    assert _stored_rel_paths() == []


# ---------------------------------------------------------------------------
# Videos upscaled from a file: recorded settings, Reuse and Extend
# ---------------------------------------------------------------------------

_VIDEO_UPSCALE_RECORD = {"method": "seedvr2", "factor": 2, "model": "seedvr2-7b", "quantize": 8, "softness": 0.0, "seed": 42}


def _upscaled_video_json(tmp_path, monkeypatch, source_config: dict | None) -> tuple[dict, dict[str, str]]:
    monkeypatch.setattr("zvisiongenerator.web.gallery.supports_video_extend", lambda: True)
    payload = build_video_upscale_payload(
        source_config,
        source_path=str(tmp_path / "clip.mp4"),
        source_width=512,
        source_height=256,
        width=1024,
        height=512,
        frame_count=97,
        audio=True,
        upscale=_VIDEO_UPSCALE_RECORD,
        generation_time=90.0,
    )
    (tmp_path / "clip_2x.mp4").write_bytes(b"\x00" * 8)
    with _video_probe_returning(payload):
        asset = _find_asset(list_gallery_assets(str(tmp_path)), "clip_2x.mp4")
    result = gallery_asset_to_json(asset, _reuse_web_config())
    return result, _reuse_params(result)


def test_an_upscaled_app_video_is_marked_upscaled_and_reuses_the_sources_workflow_at_its_size(tmp_path, monkeypatch):
    source = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "model": "ltx-8", "seed": 5, "steps": 8, "prompt": "A knight rides in.", "frame_count": 97, "width": 512, "height": 256}

    result, params = _upscaled_video_json(tmp_path, monkeypatch, source)

    assert result["has_reusable_config"] is True
    assert result["extend"] == {"available": False, "reason": EXTEND_UPSCALED_REASON}
    assert result["details"]["generation"]["upscale"] == _VIDEO_UPSCALE_RECORD
    assert result["reuse_state"]["requested_workflow"] == "txt2vid"
    assert (params["workflow"], params["width"], params["height"], params["prompt"]) == ("txt2vid", "512", "256", "A knight rides in.")


def test_reuse_of_an_upscaled_extension_opens_extend_with_the_original_source_and_new_frames(tmp_path, monkeypatch):
    original = tmp_path / "base.mp4"
    original.write_bytes(b"\x00")
    source = {"schema": "zvisiongenerator.config.v1", "workflow": "extend", "model": "ltx-8", "source_video": str(original), "extend_frames": 48, "frame_count": 97, "prompt": "She walks away."}

    result, params = _upscaled_video_json(tmp_path, monkeypatch, source)

    assert result["reuse_state"]["resolved_workflow"] == "extend"
    assert (params["source_video"], params["extend_frames"]) == (str(original), "48")


@pytest.mark.parametrize("source_config", [None, {}])
def test_the_upscale_of_a_clip_with_no_recorded_settings_cannot_be_reused(tmp_path, monkeypatch, source_config):
    result, _params = _upscaled_video_json(tmp_path, monkeypatch, source_config)

    assert result["has_reusable_config"] is False
    assert result["extend"] == {"available": False, "reason": EXTEND_UPSCALED_REASON}
