"""Unit tests for rotating, flipping and cropping an upright image."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageOps

from zvisiongenerator.processing.adjust import LightSettings, adjust_light
from zvisiongenerator.processing.reframe import CropBox, Reframe, framed_size, reframe

_ORIENTATION = 0x0112


def _labelled(mode: str = "RGB") -> Image.Image:
    """Return a 3 wide, 2 high image whose six pixels are all different."""
    image = Image.new(mode, (3, 2))
    for index in range(6):
        x, y = index % 3, index // 3
        colour = (index * 40 + 10, index * 40 + 20, index * 40 + 30)
        image.putpixel((x, y), colour + (index * 50 + 5,) if mode == "RGBA" else colour)
    return image


def _expected(image: Image.Image, rotate: int, flip: bool) -> np.ndarray:
    pixels = np.asarray(image)
    pixels = np.rot90(pixels, k=-(rotate // 90))
    return np.fliplr(pixels) if flip else pixels


_STATES = [(rotate, flip) for rotate in (0, 90, 180, 270) for flip in (False, True)]


class TestReframe:
    @pytest.mark.parametrize("mode", ["RGB", "RGBA"])
    @pytest.mark.parametrize(("rotate", "flip"), _STATES)
    def test_all_eight_states_match_a_clockwise_turn_then_a_flip(self, mode, rotate, flip):
        image = _labelled(mode)
        out = reframe(image, Reframe(rotate=rotate, flip=flip))
        assert out.mode == mode
        assert np.array_equal(np.asarray(out), _expected(image, rotate, flip))

    def test_a_quarter_turn_moves_the_top_left_pixel_to_the_top_right(self):
        image = _labelled()
        out = reframe(image, Reframe(rotate=90))
        assert out.size == (2, 3)
        assert out.getpixel((1, 0)) == image.getpixel((0, 0))

    def test_the_identity_returns_a_new_image(self):
        image = _labelled()
        out = reframe(image, Reframe())
        assert out is not image
        assert out.tobytes() == image.tobytes()

    @pytest.mark.parametrize(("rotate", "flip"), _STATES)
    def test_a_crop_is_taken_from_the_rotated_and_flipped_frame(self, rotate, flip):
        image = _labelled("RGBA")
        frame = _expected(image, rotate, flip)
        height, width = frame.shape[:2]
        crop = CropBox(x=1, y=0, width=width - 1, height=height - 1)
        out = reframe(image, Reframe(rotate=rotate, flip=flip, crop=crop))
        assert np.array_equal(np.asarray(out), frame[0 : height - 1, 1:width])

    def test_a_crop_that_is_the_whole_frame_changes_nothing(self):
        image = _labelled()
        out = reframe(image, Reframe(rotate=90, crop=CropBox(0, 0, 2, 3)))
        assert np.array_equal(np.asarray(out), _expected(image, 90, False))

    def test_a_single_pixel_crop_is_allowed(self):
        out = reframe(_labelled(), Reframe(crop=CropBox(2, 1, 1, 1)))
        assert out.size == (1, 1)
        assert out.getpixel((0, 0)) == _labelled().getpixel((2, 1))

    @pytest.mark.parametrize(
        "crop",
        [
            CropBox(0, 0, 4, 2),
            CropBox(0, 0, 3, 3),
            CropBox(1, 0, 3, 2),
            CropBox(-1, 0, 2, 2),
            CropBox(0, -1, 2, 2),
            CropBox(0, 0, 0, 1),
            CropBox(0, 0, 1, 0),
        ],
    )
    def test_a_crop_outside_the_picture_raises(self, crop):
        with pytest.raises(ValueError, match="The crop is outside the picture."):
            reframe(_labelled(), Reframe(crop=crop))

    def test_the_crop_is_checked_against_the_rotated_size(self):
        with pytest.raises(ValueError, match="outside the picture"):
            reframe(_labelled(), Reframe(rotate=90, crop=CropBox(0, 0, 3, 2)))
        assert reframe(_labelled(), Reframe(rotate=90, crop=CropBox(0, 0, 2, 3))).size == (2, 3)

    @pytest.mark.parametrize("rotate", [45, 360, -90])
    def test_other_rotations_raise(self, rotate):
        with pytest.raises(ValueError, match="rotate"):
            Reframe(rotate=rotate)

    @pytest.mark.parametrize(("rotate", "size"), [(0, (6, 4)), (90, (4, 6)), (180, (6, 4)), (270, (4, 6))])
    def test_framed_size_swaps_for_odd_quarter_turns(self, rotate, size):
        assert framed_size(6, 4, rotate) == size


class TestOrientedSource:
    def test_an_exif_oriented_picture_is_framed_after_the_transpose(self):
        stored = Image.new("RGB", (4, 2), (0, 0, 0))
        stored.putpixel((0, 0), (255, 0, 0))
        exif = Image.Exif()
        exif[_ORIENTATION] = 6
        buffer = io.BytesIO()
        stored.save(buffer, format="PNG", exif=exif)
        buffer.seek(0)
        upright = ImageOps.exif_transpose(Image.open(buffer))
        assert upright.size == (2, 4)
        assert upright.getpixel((1, 0)) == (255, 0, 0)
        out = reframe(upright, Reframe(rotate=90, crop=CropBox(2, 0, 2, 2)))
        assert out.size == (2, 2)
        assert out.getpixel((1, 1)) == (255, 0, 0)


class TestAlphaThroughTheEditor:
    def test_light_settings_keep_every_alpha_value(self):
        alpha = np.array([[0, 1, 128, 255]], dtype=np.uint8)
        rgba = np.concatenate([np.full((1, 4, 3), 90, dtype=np.uint8), alpha[..., None]], axis=-1)
        settings = LightSettings(exposure=40, contrast=-30, highlights=20, shadows=-20, saturation=50, warmth=-60)
        adjusted = adjust_light(rgba[..., :3], settings)
        out = np.concatenate([adjusted, rgba[..., 3:]], axis=-1)
        assert np.array_equal(out[..., 3], alpha[0][None])
        assert not np.array_equal(out[..., :3], rgba[..., :3])

    @pytest.mark.parametrize(("rotate", "flip"), _STATES)
    def test_geometry_moves_alpha_with_the_colour(self, rotate, flip):
        image = _labelled("RGBA")
        out = reframe(image, Reframe(rotate=rotate, flip=flip))
        assert np.array_equal(np.asarray(out)[..., 3], _expected(image, rotate, flip)[..., 3])
        assert sorted(np.asarray(out)[..., 3].ravel()) == sorted(np.asarray(image)[..., 3].ravel())
