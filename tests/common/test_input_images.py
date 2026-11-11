"""Tests for opening an input image upright and on white."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tests._images import SIDEWAYS_UPRIGHT_SIZE, cmyk_profile, display_p3_profile, is_bluish, is_reddish, write_cutout, write_sideways_photo
from zvisiongenerator.utils.input_images import flatten_alpha, open_image_upright, to_srgb, upright_image_copies, upright_size


def test_an_input_image_is_opened_upright_and_keeps_its_size(tmp_path):
    loaded = open_image_upright(write_sideways_photo(tmp_path / "phone.jpg"))

    assert loaded.mode == "RGB"
    assert loaded.size == SIDEWAYS_UPRIGHT_SIZE
    assert is_reddish(loaded.getpixel((10, 2))) and is_bluish(loaded.getpixel((10, 37)))


def test_the_upright_size_swaps_a_quarter_turned_picture_and_leaves_the_rest(tmp_path):
    Image.new("RGB", (30, 20)).save(tmp_path / "plain.png")

    assert upright_size(write_sideways_photo(tmp_path / "phone.jpg")) == SIDEWAYS_UPRIGHT_SIZE
    assert upright_size(tmp_path / "plain.png") == (30, 20)


def test_an_input_image_with_transparency_goes_on_white(tmp_path):
    loaded = open_image_upright(write_cutout(tmp_path / "cutout.png"))

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 0, 0)
    assert loaded.getpixel((7, 3)) == (255, 255, 255)


def test_a_palette_input_image_with_transparency_goes_on_white(tmp_path):
    path = tmp_path / "sticker.png"
    sticker = Image.new("P", (2, 1))
    sticker.putpalette([0, 0, 255, 255, 0, 0])
    sticker.putpixel((1, 0), 1)
    sticker.save(path, transparency=0)

    loaded = open_image_upright(path)

    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((1, 0)) == (255, 0, 0)


def test_upright_copies_replace_only_the_images_that_need_it(tmp_path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (6, 4), "green").save(plain)
    sideways = write_sideways_photo(tmp_path / "phone.jpg")
    cutout = write_cutout(tmp_path / "cutout.png")

    with upright_image_copies(str(plain), sideways, None, cutout) as (kept, turned, none, flattened):
        assert kept == str(plain) and none is None
        with Image.open(turned) as image:
            assert turned != sideways and image.size == SIDEWAYS_UPRIGHT_SIZE and not image.getexif()
        with Image.open(flattened) as image:
            assert image.mode == "RGB" and image.getpixel((7, 3)) == (255, 255, 255)

    assert not Path(turned).exists() and not Path(flattened).exists()
    assert Path(sideways).exists() and Path(cutout).exists()


def test_flatten_alpha_composites_over_the_given_colour():
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))

    flattened = flatten_alpha(image, (10, 20, 30))

    assert flattened.mode == "RGB"
    assert flattened.getpixel((0, 0)) == (10, 20, 30)
    assert flattened.getpixel((1, 0)) == (255, 0, 0)


def test_flatten_alpha_defaults_to_white():
    assert flatten_alpha(Image.new("RGBA", (1, 1), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)


def test_a_picture_with_a_colour_profile_is_converted_to_srgb_and_loses_the_profile():
    tagged = Image.new("RGBA", (4, 4), (60, 180, 90, 128))
    tagged.info["icc_profile"] = display_p3_profile()

    converted = to_srgb(tagged)

    assert "icc_profile" not in converted.info
    assert converted.mode == "RGBA" and converted.getpixel((0, 0))[3] == 128
    assert converted.getpixel((0, 0))[:3] != (60, 180, 90)


def test_a_cmyk_picture_is_converted_through_its_own_profile(tmp_path):
    path = tmp_path / "print.jpg"
    Image.new("CMYK", (4, 4), (200, 20, 20, 10)).save(path, icc_profile=cmyk_profile())

    with Image.open(path) as opened:
        converted = to_srgb(opened)
        naive = opened.convert("RGB").getpixel((0, 0))

    assert converted.mode == "RGB" and "icc_profile" not in converted.info
    assert max(abs(a - b) for a, b in zip(converted.getpixel((0, 0)), naive)) > 20


def test_a_damaged_profile_is_dropped_and_the_picture_kept():
    tagged = Image.new("RGB", (4, 4), (60, 180, 90))
    tagged.info["icc_profile"] = display_p3_profile()[:200]

    result = to_srgb(tagged)

    assert "icc_profile" not in result.info and result.getpixel((0, 0)) == (60, 180, 90)


def test_a_picture_without_a_profile_is_returned_as_it_is():
    plain = Image.new("RGB", (4, 4), (10, 200, 30))

    assert to_srgb(plain) is plain
