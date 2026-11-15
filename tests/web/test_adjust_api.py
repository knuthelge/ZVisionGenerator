"""Tests for the Web UI Adjust action: the request contract, the saved picture and the POST /api/adjust route."""

from __future__ import annotations

import io
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tests.web._helpers import make_web_config
from zvisiongenerator.processing.adjust import MIX_NAMES, LightSettings
from zvisiongenerator.processing.reframe import CropBox, Reframe
from zvisiongenerator.utils.provenance import EXIF_IMAGE_DESCRIPTION, embed_png_config, image_prompt_text, read_png_config
from zvisiongenerator.web import adjust_api
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.upload_folders import UPLOADS_FOLDER

_CONFIG = {"workflow": "txt2img", "prompt": "a lake", "model": "zit", "seed": 7, "width": 8, "height": 6}
_EXIF_ORIENTATION = 0x0112
_ORIGIN = {"Origin": "http://testserver"}
MIX_FIELDS = tuple(f"mix_{name}" for name in MIX_NAMES)


@pytest.fixture
def adjust_env(monkeypatch, tmp_path):
    web_config = make_web_config()
    root = tmp_path / "out"
    root.mkdir()
    web_config.output_dir = str(root)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: root.resolve())
    # The home folder is the parent of the output folder, so recorded paths read ``~/out/...``; the app's own data folder lands outside the output folder.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with TestClient(web_server.app) as client:
        yield SimpleNamespace(client=client, root=root, monkeypatch=monkeypatch)


def _gradient(size: tuple[int, int] = (8, 6), mode: str = "RGB") -> Image.Image:
    """Return a picture whose pixels all differ, so a wrong rotation or crop shows."""
    width, height = size
    grid = np.arange(width * height, dtype=np.uint8).reshape(height, width) * 3
    planes = [grid, grid // 2 + 40, 255 - grid]
    if mode == "RGBA":
        planes.append(np.full_like(grid, 200))
    return Image.fromarray(np.stack(planes, axis=2), mode)


def _save_png(path: Path, image: Image.Image | None = None, *, config: dict[str, Any] | None = _CONFIG, prompt: str | None = "a red lake") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    if prompt is not None:
        info.add_text("Description", prompt)
    if config is not None:
        embed_png_config(info, config)
    (image or _gradient()).save(path, pnginfo=info)
    return path


def _post(env, **body):
    body.setdefault("save", "copy")
    return env.client.post("/api/adjust", json=body, headers=_ORIGIN)


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert(image.mode)).copy()


class TestRequestContract:
    def test_a_full_body_is_parsed(self):
        body = {"asset_id": "a/b.png", "save": "replace", "rotate": 90, "flip": True, "crop": {"x": 1, "y": 2, "width": 3, "height": 4}, "exposure": 10, "warmth": -5}

        request = adjust_api.adjust_json_request(body)

        assert request.asset_id == "a/b.png"
        assert request.save == "replace"
        assert request.frame == Reframe(rotate=90, flip=True, crop=CropBox(1, 2, 3, 4))
        assert request.light == LightSettings(exposure=10, warmth=-5)

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            pytest.param([], "JSON object", id="not an object"),
            pytest.param({"save": "copy", "exposure": 1}, "'asset_id'", id="no asset id"),
            pytest.param({"asset_id": "  ", "save": "copy", "exposure": 1}, "'asset_id'", id="blank asset id"),
            pytest.param({"asset_id": "a.png", "exposure": 1}, "'save'", id="no save"),
            pytest.param({"asset_id": "a.png", "save": "move", "exposure": 1}, "'save'", id="unknown save"),
            pytest.param({"asset_id": "a.png", "save": "copy", "rotate": 45}, "'rotate'", id="rotate off the quarter turns"),
            pytest.param({"asset_id": "a.png", "save": "copy", "rotate": False}, "'rotate'", id="rotate bool"),
            pytest.param({"asset_id": "a.png", "save": "copy", "rotate": "90"}, "'rotate'", id="rotate text"),
            pytest.param({"asset_id": "a.png", "save": "copy", "flip": 1}, "'flip'", id="flip not a bool"),
            pytest.param({"asset_id": "a.png", "save": "copy", "crop": 5}, "'crop'", id="crop not an object"),
            pytest.param({"asset_id": "a.png", "save": "copy", "crop": {"x": 0, "y": 0, "width": 4}}, "'crop.height'", id="crop missing a key"),
            pytest.param({"asset_id": "a.png", "save": "copy", "crop": {"x": -1, "y": 0, "width": 4, "height": 4}}, "'crop.x'", id="crop negative x"),
            pytest.param({"asset_id": "a.png", "save": "copy", "crop": {"x": 0, "y": 0, "width": 0, "height": 4}}, "'crop.width'", id="crop zero width"),
            pytest.param({"asset_id": "a.png", "save": "copy", "crop": {"x": 0, "y": True, "width": 4, "height": 4}}, "'crop.y'", id="crop bool"),
            pytest.param({"asset_id": "a.png", "save": "copy", "crop": {"x": 0.5, "y": 0, "width": 4, "height": 4}}, "'crop.x'", id="crop float"),
            pytest.param({"asset_id": "a.png", "save": "copy", "exposure": 101}, "'exposure'", id="slider over"),
            pytest.param({"asset_id": "a.png", "save": "copy", "warmth": -101}, "'warmth'", id="slider under"),
            pytest.param({"asset_id": "a.png", "save": "copy", "contrast": True}, "'contrast'", id="slider bool"),
            pytest.param({"asset_id": "a.png", "save": "copy", "shadows": 1.5}, "'shadows'", id="slider float"),
            pytest.param({"asset_id": "a.png", "save": "copy", "vibrance": 101}, "'vibrance'", id="vibrance over"),
            pytest.param({"asset_id": "a.png", "save": "copy", "vibrance": -101}, "'vibrance'", id="vibrance under"),
            pytest.param({"asset_id": "a.png", "save": "copy", "vibrance": True}, "'vibrance'", id="vibrance bool"),
            pytest.param({"asset_id": "a.png", "save": "copy", "vibrance": "10"}, "'vibrance'", id="vibrance text"),
            pytest.param({"asset_id": "a.png", "save": "copy", "mix_blue": 101}, "'mix_blue'", id="mix over"),
            pytest.param({"asset_id": "a.png", "save": "copy", "mix_red": -101}, "'mix_red'", id="mix under"),
            pytest.param({"asset_id": "a.png", "save": "copy", "mix_blue": True}, "'mix_blue'", id="mix bool"),
            pytest.param({"asset_id": "a.png", "save": "copy", "mix_blue": "5"}, "'mix_blue'", id="mix text"),
            pytest.param({"asset_id": "a.png", "save": "copy", "mix_magenta": 2.5}, "'mix_magenta'", id="mix float"),
            pytest.param({"asset_id": "a.png", "save": "copy", **dict.fromkeys(MIX_FIELDS, 0)}, "Nothing to save", id="all eight bands neutral"),
            pytest.param({"asset_id": "a.png", "save": "copy"}, "Nothing to save: no adjustment is set.", id="nothing set"),
            pytest.param({"asset_id": "a.png", "save": "copy", "rotate": 0, "flip": False, "crop": None, "exposure": 0}, "Nothing to save", id="all neutral"),
            pytest.param(
                {"asset_id": "a.png", "save": "copy", "exposure": 0, "contrast": 0, "highlights": 0, "shadows": 0, "saturation": 0, "vibrance": 0, "warmth": 0},
                "Nothing to save",
                id="all seven neutral",
            ),
            pytest.param(
                {"asset_id": "a.png", "save": "copy", "exposure": 0, "vibrance": 0, "warmth": 0, **dict.fromkeys(MIX_FIELDS, 0)},
                "Nothing to save",
                id="all fifteen neutral",
            ),
        ],
    )
    def test_a_bad_body_is_refused_naming_the_field(self, body, message):
        with pytest.raises(ValueError, match=message):
            adjust_api.adjust_json_request(body)


class TestVibranceField:
    def test_vibrance_alone_is_a_request_and_defaults_to_0_when_absent(self):
        assert adjust_api.adjust_json_request({"asset_id": "a.png", "save": "copy", "vibrance": 20}).light == LightSettings(vibrance=20)
        assert adjust_api.adjust_json_request({"asset_id": "a.png", "save": "copy", "saturation": 5}).light == LightSettings(saturation=5)

    def test_vibrance_alone_saves_a_changed_copy(self, adjust_env):
        source = _save_png(adjust_env.root / "muted.png", image=Image.new("RGB", (4, 4), (150, 130, 120)))

        response = _post(adjust_env, asset_id="muted.png", vibrance=100)

        assert response.status_code == 200
        assert _pixels(adjust_env.root / "muted_adjusted.png")[0, 0].tolist() == [161, 127, 108]
        assert _pixels(source)[0, 0].tolist() == [150, 130, 120]


class TestColourMixerFields:
    def test_each_band_alone_is_a_request_and_defaults_to_0_when_absent(self):
        for name in MIX_NAMES:
            light = adjust_api.adjust_json_request({"asset_id": "a.png", "save": "copy", f"mix_{name}": -20}).light
            assert light == LightSettings(**{f"mix_{name}": -20})
        assert adjust_api.adjust_json_request({"asset_id": "a.png", "save": "copy", "exposure": 5}).light.mix == (0,) * 8

    def test_a_mix_alone_saves_a_changed_copy(self, adjust_env):
        source = _save_png(adjust_env.root / "warm.png", image=Image.new("RGB", (4, 4), (184, 92, 80)))

        response = _post(adjust_env, asset_id="warm.png", mix_red=100)

        assert response.status_code == 200
        assert _pixels(adjust_env.root / "warm_adjusted.png")[0, 0].tolist() == [218, 63, 7]
        assert _pixels(source)[0, 0].tolist() == [184, 92, 80]


class TestSaveAsCopy:
    def test_the_copy_is_named_for_the_source_and_counted_when_taken(self, adjust_env):
        source = _save_png(adjust_env.root / "set" / "lake.png")

        first = _post(adjust_env, asset_id="set/lake.png", exposure=10)
        second = _post(adjust_env, asset_id="set/lake.png", exposure=10)
        of_the_copy = _post(adjust_env, asset_id="set/lake_adjusted.png", exposure=10)

        assert first.status_code == 200
        assert [r.json()["asset"]["id"] for r in (first, second, of_the_copy)] == ["set/lake_adjusted.png", "set/lake_adjusted_2.png", "set/lake_adjusted_3.png"]
        assert source.read_bytes() == source.read_bytes()
        assert sorted(p.name for p in source.parent.iterdir()) == ["lake.png", "lake_adjusted.png", "lake_adjusted_2.png", "lake_adjusted_3.png"]

    def test_a_name_taken_after_it_was_chosen_is_not_overwritten(self, adjust_env):
        _save_png(adjust_env.root / "lake.png")
        taken = adjust_env.root / "lake_adjusted.png"
        real_write = adjust_api.write_new_file

        def racing_write(path, data, *, stem):
            taken.write_bytes(b"someone else")
            return real_write(path, data, stem=stem)

        adjust_env.monkeypatch.setattr(adjust_api, "write_new_file", racing_write)

        response = _post(adjust_env, asset_id="lake.png", exposure=10)

        assert response.json()["asset"]["id"] == "lake_adjusted_2.png"
        assert taken.read_bytes() == b"someone else"

    def test_the_answer_is_the_gallery_asset_of_the_copy(self, adjust_env):
        _save_png(adjust_env.root / "set" / "lake.png")

        asset = _post(adjust_env, asset_id="set/lake.png", rotate=90).json()["asset"]

        assert asset["id"] == "set/lake_adjusted.png"
        assert (asset["width"], asset["height"]) == (6, 8)
        assert asset["url"].startswith("/media/set/lake_adjusted.png?v=")
        assert asset["thumbnail_url"].startswith("/thumbs/set/lake_adjusted.png?v=")
        assert asset["details"]["source"]["id"] == "set/lake.png"

    def test_the_copy_records_where_it_came_from_with_the_home_folder_as_a_tilde(self, adjust_env):
        source = _save_png(adjust_env.root / "set" / "lake.png")

        _post(adjust_env, asset_id="set/lake.png", exposure=10)

        written = source.parent / "lake_adjusted.png"
        assert read_png_config(written) == {**_CONFIG, "copied_from": "~/out/set/lake.png"}
        assert str(adjust_env.root) not in written.read_bytes().decode("latin-1")

    def test_the_prompt_is_kept_in_the_text_chunk_and_in_exif(self, adjust_env):
        _save_png(adjust_env.root / "lake.png", prompt="a red lake")

        _post(adjust_env, asset_id="lake.png", exposure=10)

        with Image.open(adjust_env.root / "lake_adjusted.png") as copy:
            assert image_prompt_text(copy) == "a red lake"
            assert copy.info["Description"] == "a red lake"
            assert copy.getexif()[EXIF_IMAGE_DESCRIPTION] == "a red lake"

    def test_a_picture_without_a_prompt_or_settings_gets_neither(self, adjust_env):
        _save_png(adjust_env.root / "plain.png", config=None, prompt=None)

        _post(adjust_env, asset_id="plain.png", exposure=10)

        with Image.open(adjust_env.root / "plain_adjusted.png") as copy:
            assert "Description" not in copy.info
            assert "zvisiongenerator.config" not in copy.info
            assert EXIF_IMAGE_DESCRIPTION not in copy.getexif()

    def test_a_jpeg_with_no_settings_gives_a_png_copy_with_none(self, adjust_env):
        Image.new("RGB", (8, 6), "teal").save(adjust_env.root / "photo.jpg")

        response = _post(adjust_env, asset_id="photo.jpg", contrast=20)

        assert response.json()["asset"]["id"] == "photo_adjusted.png"
        assert read_png_config(adjust_env.root / "photo_adjusted.png") is None

    def test_a_webp_source_gives_a_copy(self, adjust_env):
        _gradient().save(adjust_env.root / "pic.webp", lossless=True)

        response = _post(adjust_env, asset_id="pic.webp", flip=True)

        assert response.json()["asset"]["id"] == "pic_adjusted.png"

    def test_a_crop_only_copy_has_exactly_the_cropped_pixels(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")

        _post(adjust_env, asset_id="lake.png", crop={"x": 2, "y": 1, "width": 4, "height": 3})

        assert np.array_equal(_pixels(adjust_env.root / "lake_adjusted.png"), _pixels(source)[1:4, 2:6])

    def test_rotate_then_flip_then_crop_follow_the_documented_order(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")
        expected = np.fliplr(np.rot90(_pixels(source), k=-1))[1:4, 0:2]

        _post(adjust_env, asset_id="lake.png", rotate=90, flip=True, crop={"x": 0, "y": 1, "width": 2, "height": 3})

        assert np.array_equal(_pixels(adjust_env.root / "lake_adjusted.png"), expected)

    def test_a_colour_only_copy_changes_the_colour_and_keeps_the_size(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")

        _post(adjust_env, asset_id="lake.png", exposure=50)

        copy = _pixels(adjust_env.root / "lake_adjusted.png")
        assert copy.shape == _pixels(source).shape
        assert copy.sum() > _pixels(source).sum()

    def test_transparency_is_kept_and_alpha_is_untouched_by_colour(self, adjust_env):
        source = _save_png(adjust_env.root / "cut.png", _gradient(mode="RGBA"))
        alpha = _pixels(source)[..., 3]
        alpha[::2, ::2] = 0
        Image.fromarray(np.dstack([_pixels(source)[..., :3], alpha]), "RGBA").save(source)

        _post(adjust_env, asset_id="cut.png", exposure=40, saturation=-30)

        with Image.open(adjust_env.root / "cut_adjusted.png") as copy:
            assert copy.mode == "RGBA"
            assert np.array_equal(np.asarray(copy)[..., 3], alpha)

    @pytest.mark.parametrize(("mode", "suffix"), [("P", "png"), ("LA", "png"), ("1", "png"), ("L", "png"), ("CMYK", "jpg")])
    def test_modes_are_normalised_to_rgb_or_rgba(self, adjust_env, mode, suffix):
        has_alpha = mode == "LA"
        picture = Image.new("RGBA" if has_alpha else "RGB", (8, 6), (200, 120, 40, 90) if has_alpha else (200, 120, 40)).convert(mode)
        picture.save(adjust_env.root / f"{mode}.{suffix}")

        response = _post(adjust_env, asset_id=f"{mode}.{suffix}", exposure=20)

        assert response.status_code == 200
        with Image.open(adjust_env.root / f"{mode}_adjusted.png") as copy:
            assert copy.mode == ("RGBA" if has_alpha else "RGB")

    def test_a_palette_picture_with_transparency_keeps_it(self, adjust_env):
        picture = Image.new("P", (8, 6), 0)
        picture.putpalette([255, 0, 0] * 256)
        picture.info["transparency"] = 0
        picture.save(adjust_env.root / "pal.png")

        _post(adjust_env, asset_id="pal.png", flip=True)

        with Image.open(adjust_env.root / "pal_adjusted.png") as copy:
            assert copy.mode == "RGBA"
            assert copy.getpixel((0, 0))[3] == 0

    def test_exif_orientation_is_applied_before_the_adjustment_and_not_kept(self, adjust_env):
        exif = Image.Exif()
        exif[_EXIF_ORIENTATION] = 6
        _gradient((8, 6)).save(adjust_env.root / "phone.jpg", exif=exif.tobytes(), quality=100)
        with Image.open(adjust_env.root / "phone.jpg") as opened:
            upright = opened.copy()
        assert upright.size == (8, 6)

        response = _post(adjust_env, asset_id="phone.jpg", crop={"x": 0, "y": 0, "width": 6, "height": 8})

        assert response.status_code == 200
        with Image.open(adjust_env.root / "phone_adjusted.png") as copy:
            assert copy.size == (6, 8)
            assert _EXIF_ORIENTATION not in copy.getexif()

    def test_a_pixel_format_with_more_than_eight_bits_is_refused(self, adjust_env):
        Image.new("I;16", (8, 6)).save(adjust_env.root / "deep.png")

        response = _post(adjust_env, asset_id="deep.png", exposure=20)

        assert response.status_code == 422
        assert response.json()["detail"] == "deep.png can't be adjusted: its pixel format (I;16) isn't supported."
        assert [p.name for p in adjust_env.root.iterdir()] == ["deep.png"]


class TestReplace:
    def test_replace_keeps_the_id_and_changes_the_url_and_the_pixels(self, adjust_env):
        source = _save_png(adjust_env.root / "set" / "lake.png")
        os.utime(source, ns=(1_000_000_000, 1_000_000_000))
        before = _post_listing(adjust_env, "set/lake.png")

        response = _post(adjust_env, asset_id="set/lake.png", save="replace", rotate=90)

        asset = response.json()["asset"]
        assert asset["id"] == "set/lake.png"
        assert asset["url"] != before["url"]
        assert (asset["width"], asset["height"]) == (6, 8)
        assert [p.name for p in source.parent.iterdir()] == ["lake.png"]

    def test_replace_writes_the_settings_unchanged(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")

        _post(adjust_env, asset_id="lake.png", save="replace", exposure=10)

        assert read_png_config(source) == _CONFIG

    @pytest.mark.parametrize("name", ["photo.jpg", "photo.webp"])
    def test_only_a_png_can_be_replaced(self, adjust_env, name):
        path = adjust_env.root / name
        Image.new("RGB", (8, 6), "teal").save(path)
        before = path.read_bytes()

        response = _post(adjust_env, asset_id=name, save="replace", exposure=10)

        assert response.status_code == 422
        assert response.json()["detail"] == "Only PNG files can be replaced; use Save as copy."
        assert path.read_bytes() == before

    def test_copy_then_replace_then_copy_record_the_right_sources(self, adjust_env):
        original = _save_png(adjust_env.root / "lake.png")

        _post(adjust_env, asset_id="lake.png", exposure=10)
        copy = adjust_env.root / "lake_adjusted.png"
        copied = read_png_config(copy)
        _post(adjust_env, asset_id="lake_adjusted.png", save="replace", contrast=10)
        replaced = read_png_config(copy)
        _post(adjust_env, asset_id="lake_adjusted.png", exposure=10)
        copy_of_copy = read_png_config(adjust_env.root / "lake_adjusted_2.png")

        assert read_png_config(original) == _CONFIG
        assert copied == {**_CONFIG, "copied_from": "~/out/lake.png"}
        assert replaced == copied
        assert copy_of_copy == {**_CONFIG, "copied_from": "~/out/lake_adjusted.png"}

    def test_a_failing_encoder_leaves_the_original_and_no_temp_file(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")
        before = source.read_bytes()
        real_save = Image.Image.save

        def failing_save(self, fp, format=None, **params):
            if isinstance(fp, io.BytesIO):
                raise OSError(28, "No space left on device")
            return real_save(self, fp, format, **params)

        adjust_env.monkeypatch.setattr(Image.Image, "save", failing_save)

        response = _post(adjust_env, asset_id="lake.png", save="replace", exposure=10)

        assert response.status_code == 500
        assert response.json()["detail"] == "Could not save lake.png: No space left on device"
        assert source.read_bytes() == before
        assert [p.name for p in adjust_env.root.iterdir()] == ["lake.png"]

    def test_a_failing_replace_leaves_the_original_and_no_temp_file(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")
        before = source.read_bytes()

        def refuse(self, target):
            raise PermissionError(13, "Permission denied")

        adjust_env.monkeypatch.setattr(Path, "replace", refuse)

        response = _post(adjust_env, asset_id="lake.png", save="replace", exposure=10)

        assert response.status_code == 500
        assert response.json()["detail"] == "Could not save lake.png: Permission denied"
        assert source.read_bytes() == before
        assert sorted(p.name for p in adjust_env.root.iterdir()) == ["lake.png"]

    def test_the_source_is_closed_before_the_replacement_is_written(self, adjust_env):
        source = _save_png(adjust_env.root / "lake.png")
        events: list[str] = []
        real_open = Image.open

        @contextmanager
        def recording_open(*args, **kwargs):
            with real_open(*args, **kwargs) as image:
                events.append("opened")
                yield image
            events.append("closed")

        open_at_write: list[int] = []

        def recording_write(path, data):
            events.append("write")
            open_at_write.append(events.count("opened") - events.count("closed"))

        adjust_env.monkeypatch.setattr(adjust_api.Image, "open", recording_open)
        adjust_env.monkeypatch.setattr(adjust_api, "write_bytes_atomic", recording_write)

        response = _post(adjust_env, asset_id="lake.png", save="replace", rotate=180)

        assert response.status_code == 200
        assert "opened" in events[: events.index("write")]
        assert open_at_write == [0]
        assert read_png_config(source) == _CONFIG


def _post_listing(env, asset_id: str) -> dict[str, Any]:
    return next(asset for asset in env.client.get("/api/gallery", params={"page": 1}).json()["assets"] if asset["id"] == asset_id)


class TestRefusals:
    @pytest.mark.parametrize(
        ("body", "status", "detail"),
        [
            pytest.param({"asset_id": "../outside.png", "exposure": 5}, 422, "The file must be inside the output folder.", id="outside the root"),
            pytest.param({"asset_id": "/etc/passwd", "exposure": 5}, 422, "The file must be inside the output folder.", id="absolute"),
            pytest.param({"asset_id": "missing.png", "exposure": 5}, 404, "Unknown image: missing.png", id="no such file"),
            pytest.param({"asset_id": "folder", "exposure": 5}, 404, "Unknown image: folder", id="a folder"),
            pytest.param({"asset_id": "clip.mp4", "exposure": 5}, 404, "Unknown image: clip.mp4", id="not an image"),
            pytest.param({"asset_id": f"{UPLOADS_FOLDER}/up.png", "exposure": 5}, 422, "The file must be inside the output folder.", id="an upload"),
            pytest.param({"asset_id": "lake.png"}, 422, "Nothing to save: no adjustment is set.", id="nothing set"),
            pytest.param({"asset_id": "lake.png", "exposure": 500}, 422, "Field 'exposure' must be an integer from -100 to 100.", id="slider out of range"),
            pytest.param({"asset_id": "lake.png", "save": "other", "exposure": 5}, 422, 'Field \'save\' must be "copy" or "replace".', id="bad save"),
            pytest.param({"asset_id": "lake.png", "crop": {"x": 4, "y": 0, "width": 8, "height": 6}}, 422, "The crop is outside the picture.", id="crop past the edge"),
            pytest.param({"asset_id": "lake.png", "rotate": 90, "crop": {"x": 0, "y": 0, "width": 8, "height": 6}}, 422, "The crop is outside the picture.", id="crop of the unrotated frame"),
            pytest.param({"asset_id": "broken.png", "exposure": 5}, 422, "broken.png can't be adjusted: the file isn't a readable image.", id="unreadable"),
        ],
    )
    def test_each_refusal_has_its_status_and_message_and_writes_nothing(self, adjust_env, body, status, detail):
        _save_png(adjust_env.root / "lake.png")
        (adjust_env.root / "folder.png").mkdir()
        (adjust_env.root / "folder").mkdir()
        (adjust_env.root / "clip.mp4").write_bytes(b"\x00" * 16)
        (adjust_env.root / "broken.png").write_bytes(b"not a picture")
        _save_png(adjust_env.root / UPLOADS_FOLDER / "up.png")
        before = sorted(str(p.relative_to(adjust_env.root)) for p in adjust_env.root.rglob("*"))

        response = _post(adjust_env, **body)

        assert response.status_code == status
        assert response.json()["detail"] == detail
        assert sorted(str(p.relative_to(adjust_env.root)) for p in adjust_env.root.rglob("*")) == before

    def test_a_body_that_is_not_json_is_refused(self, adjust_env):
        response = adjust_env.client.post("/api/adjust", content=b"nope", headers=_ORIGIN)

        assert response.status_code == 422
        assert response.json()["detail"] == "Request body must be JSON."

    def test_a_json_body_that_is_not_an_object_is_refused(self, adjust_env):
        response = adjust_env.client.post("/api/adjust", json=[1], headers=_ORIGIN)

        assert response.status_code == 422
        assert response.json()["detail"] == "Request body must be a JSON object."

    def test_a_file_that_disappears_before_it_is_read_is_unknown(self, adjust_env):
        _save_png(adjust_env.root / "lake.png")
        adjust_env.monkeypatch.setattr(web_server, "save_adjusted", MagicMock(side_effect=FileNotFoundError(2, "gone", "/private/lake.png")))

        response = _post(adjust_env, asset_id="lake.png", exposure=5)

        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown image: lake.png"

    def test_a_folder_that_cannot_be_written_is_a_server_error_without_the_path(self, adjust_env):
        _save_png(adjust_env.root / "lake.png")
        adjust_env.monkeypatch.setattr(adjust_api, "write_new_file", MagicMock(side_effect=PermissionError(13, "Permission denied", str(adjust_env.root / "x"))))

        response = _post(adjust_env, asset_id="lake.png", exposure=5)

        assert response.status_code == 500
        assert response.json()["detail"] == "Could not save lake.png: Permission denied"

    def test_a_saved_file_the_gallery_does_not_list_is_a_server_error(self, adjust_env):
        _save_png(adjust_env.root / "lake.png")
        adjust_env.monkeypatch.setattr(web_server, "gallery_asset_for_output_path", lambda *_: None)

        response = _post(adjust_env, asset_id="lake.png", exposure=5)

        assert response.status_code == 500
        assert response.json()["detail"] == "Could not save lake.png: the saved file is not listed."

    def test_the_picture_is_adjusted_off_the_event_loop(self, adjust_env):
        _save_png(adjust_env.root / "lake.png")
        seen: list[Any] = []
        real = web_server.run_in_threadpool

        async def spy(func, *args, **kwargs):
            seen.append(func)
            return await real(func, *args, **kwargs)

        adjust_env.monkeypatch.setattr(web_server, "run_in_threadpool", spy)

        assert _post(adjust_env, asset_id="lake.png", exposure=5).status_code == 200
        assert web_server._adjust_from_body in seen


class TestRouteGuards:
    def test_the_route_is_in_the_openapi_contract(self, adjust_env):
        paths = adjust_env.client.get("/openapi.json").json()["paths"]

        assert "post" in paths["/api/adjust"]

    @pytest.mark.parametrize("headers", [{"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}])
    def test_a_cross_site_post_is_refused_and_writes_nothing(self, adjust_env, headers):
        _save_png(adjust_env.root / "lake.png")

        response = adjust_env.client.post("/api/adjust", json={"asset_id": "lake.png", "save": "copy", "exposure": 5}, headers=headers)

        assert response.status_code == 403
        assert [p.name for p in adjust_env.root.iterdir()] == ["lake.png"]
