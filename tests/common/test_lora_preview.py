"""Tests for shrinking a preview image to the picture kept with a downloaded LoRA."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, PngImagePlugin

from tests._images import SIDEWAYS_UPRIGHT_SIZE, is_bluish, is_reddish, write_cutout, write_sideways_photo
from zvisiongenerator.utils.lora_preview import LARGE_PREVIEW_EDGE, PREVIEW_EDGE, make_preview


def _png(size: tuple[int, int], mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, (200, 30, 30, 120) if mode == "RGBA" else (200, 30, 30)).save(out, format="PNG")
    return out.getvalue()


def test_a_large_image_is_shrunk_to_the_edge_and_kept_as_webp():
    content_type, data = make_preview(_png((1200, 800)))

    assert content_type == "image/webp"
    with Image.open(io.BytesIO(data)) as image:
        assert (image.format, max(image.size)) == ("WEBP", PREVIEW_EDGE)
        assert image.size == (PREVIEW_EDGE, round(800 * PREVIEW_EDGE / 1200))


def test_the_edge_is_chosen_by_the_caller():
    _, data = make_preview(_png((2000, 1000)), LARGE_PREVIEW_EDGE)

    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (LARGE_PREVIEW_EDGE, LARGE_PREVIEW_EDGE // 2)


def test_a_small_image_is_not_enlarged_and_keeps_its_transparency():
    _, data = make_preview(_png((40, 30), "RGBA"))

    with Image.open(io.BytesIO(data)) as image:
        assert (image.size, "A" in image.getbands()) == ((40, 30), True)


def test_bytes_that_are_not_an_image_give_nothing():
    assert make_preview(b"not an image") is None


def test_a_path_is_read_like_its_bytes(tmp_path):
    path = tmp_path / "wide.png"
    path.write_bytes(_png((1200, 800)))

    content_type, data = make_preview(path)

    assert content_type == "image/webp"
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (PREVIEW_EDGE, round(800 * PREVIEW_EDGE / 1200))


def test_a_missing_path_gives_nothing(tmp_path):
    assert make_preview(tmp_path / "missing.png") is None


def test_a_path_with_too_many_pixels_is_refused_before_the_file_is_read(tmp_path, monkeypatch):
    path = tmp_path / "big.png"
    path.write_bytes(_png((20, 20)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    def _no_decode(*_args, **_kwargs):
        raise AssertionError("the pixels were decoded")

    monkeypatch.setattr(Image.Image, "load", _no_decode)

    assert make_preview(path) is None


def test_a_photo_taken_sideways_comes_out_upright(tmp_path):
    _, data = make_preview(Path(write_sideways_photo(tmp_path / "phone.jpg")))

    with Image.open(io.BytesIO(data)) as image:
        assert image.size == SIDEWAYS_UPRIGHT_SIZE
        assert is_reddish(image.getpixel((10, 5)))
        assert is_bluish(image.getpixel((10, 35)))
        assert not image.getexif()


def test_a_large_jpeg_is_shrunk_to_the_edge(tmp_path):
    path = tmp_path / "large.jpg"
    Image.new("RGB", (2400, 1600), (10, 200, 10)).save(path, quality=90)

    _, data = make_preview(path)

    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (PREVIEW_EDGE, round(1600 * PREVIEW_EDGE / 2400))


def test_a_jpeg_with_a_malformed_exif_block_still_previews():
    out = io.BytesIO()
    Image.new("RGB", (800, 600), (200, 30, 30)).save(out, format="JPEG")
    jpeg = out.getvalue()
    payload = b"Exif\x00\x00" + b"not a tiff header"
    exif_segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload

    result = make_preview(jpeg[:2] + exif_segment + jpeg[2:])

    assert result is not None
    with Image.open(io.BytesIO(result[1])) as image:
        assert image.size == (PREVIEW_EDGE, round(600 * PREVIEW_EDGE / 800))


def test_a_transparent_png_keeps_its_alpha(tmp_path):
    _, data = make_preview(Path(write_cutout(tmp_path / "cutout.png")))

    with Image.open(io.BytesIO(data)) as image:
        assert "A" in image.getbands()
        assert image.convert("RGBA").getpixel((0, 0))[3] == 255
        assert image.convert("RGBA").getpixel((image.width - 1, 0))[3] == 0


def test_the_text_a_png_embeds_is_not_kept(tmp_path):
    path = tmp_path / "labelled.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a private prompt")
    Image.new("RGB", (40, 30)).save(path, pnginfo=info)

    _, data = make_preview(path)

    assert b"a private prompt" not in data


def test_a_png_of_many_pixels_under_pillow_s_limit_previews(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("L", (8000, 5200), 90).save(path, compress_level=1)
    assert 8000 * 5200 < Image.MAX_IMAGE_PIXELS

    content_type, data = make_preview(path)

    assert content_type == "image/webp"
    with Image.open(io.BytesIO(data)) as image:
        assert max(image.size) == PREVIEW_EDGE


def test_an_image_over_pillow_s_limit_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "big.png"
    path.write_bytes(_png((20, 20)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 399)

    assert make_preview(path) is None


def test_a_palette_picture_is_shrunk_smoothly(tmp_path):
    path = tmp_path / "gradient.png"
    stripes = Image.new("L", (1200, 800))
    stripes.putdata([255 if (x // 3) % 2 else 0 for _ in range(800) for x in range(1200)])
    stripes.convert("P").save(path)

    _, data = make_preview(path)

    with Image.open(io.BytesIO(data)) as image:
        # Blended stripes land on grey; a nearest-neighbour shrink would keep only black and white.
        assert any(60 < image.convert("L").getpixel((x, 10)) < 200 for x in range(image.width))


def test_a_sideways_palette_png_is_turned_upright_after_it_is_shrunk(tmp_path):
    path = tmp_path / "sideways.png"
    photo = Image.new("RGB", (1200, 600), (255, 0, 0))
    photo.paste((0, 0, 255), (600, 0, 1200, 600))
    exif = Image.Exif()
    exif[0x0112] = 6
    photo.convert("P").save(path, exif=exif)

    _, data = make_preview(path)

    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (PREVIEW_EDGE // 2, PREVIEW_EDGE)
        assert is_reddish(image.getpixel((10, 5)))
        assert is_bluish(image.getpixel((10, image.height - 5)))
