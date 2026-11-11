"""Test the shared parsing and refusals of an image edit's inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.utils.edit_request import check_edit_inputs, edit_mode, load_paint, parse_extend, source_is_cutout


def _save_mask(path: Path, painted: np.ndarray, *, mode: str = "RGBA") -> Path:
    height, width = painted.shape
    if mode == "RGBA":
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[..., 3] = painted * 255  # colour is ignored when there is alpha
        pixels[..., :3] = 255
    elif mode == "RGB":
        pixels = np.repeat((painted * 255).astype(np.uint8)[..., None], 3, axis=2)
    elif mode == "L":
        pixels = (painted * 255).astype(np.uint8)
    else:
        raise AssertionError(mode)
    Image.fromarray(pixels, mode).save(path)
    return path


def _block(height: int, width: int) -> np.ndarray:
    painted = np.zeros((height, width), dtype=bool)
    painted[height // 4 : height // 2, width // 4 : width // 2] = True
    return painted


class TestParseExtend:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("top=256", ("top", 256)),
            ("Right=128", ("right", 128)),
            (" bottom = 512 ", ("bottom", 512)),
            (("left", 144), ("left", 144)),
            (("left", "144"), ("left", 144)),
            (["top", 256], ("top", 256)),
        ],
    )
    def test_a_valid_request_parses(self, value, expected):
        assert parse_extend(value) == expected

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("top", "side and an amount"),
            ("", "side and an amount"),
            ("up=256", "top, right, bottom or left"),
            ("=256", "top, right, bottom or left"),
            ("top=", "whole number"),
            ("top=abc", "whole number"),
            ("top=256.5", "whole number"),
            ("top=127", "128 to 512"),
            ("top=64", "128 to 512"),
            ("top=528", "128 to 512"),
            ("top=130", "steps of 16"),
            ("top=-256", "128 to 512"),
            (("top",), "side and an amount"),
            (("top", 256, 1), "side and an amount"),
            (("top", True), "whole number"),
            (("top", 256.0), "whole number"),
            (("top", None), "whole number"),
        ],
    )
    def test_every_invalid_request_is_refused(self, value, message):
        with pytest.raises(ValueError, match=message):
            parse_extend(value)

    def test_the_limits_are_included(self):
        assert parse_extend("top=128")[1] == 128
        assert parse_extend("top=512")[1] == 512


class TestLoadPaint:
    @pytest.mark.parametrize("mode", ["RGBA", "RGB", "L"])
    def test_a_mask_of_the_image_size_loads_as_its_painted_pixels(self, tmp_path, mode):
        painted = _block(120, 160)
        path = _save_mask(tmp_path / "mask.png", painted, mode=mode)

        result = load_paint(str(path), 160, 120)

        assert result.dtype == bool and np.array_equal(result, painted)

    def test_with_alpha_the_colour_is_ignored_and_without_it_any_channel_paints(self, tmp_path):
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[5:10, 5:10] = (0, 0, 0, 1)  # black but not transparent: painted
        rgba[12:16, 12:16] = (255, 255, 255, 0)  # white but transparent: not painted
        Image.fromarray(rgba, "RGBA").save(tmp_path / "alpha.png")
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[3:6, 3:6, 2] = 1  # one channel above zero
        Image.fromarray(rgb, "RGB").save(tmp_path / "rgb.png")

        with_alpha = load_paint(str(tmp_path / "alpha.png"), 20, 20)
        without = load_paint(str(tmp_path / "rgb.png"), 20, 20)

        assert with_alpha[5:10, 5:10].all() and with_alpha.sum() == 25
        assert without[3:6, 3:6].all() and without.sum() == 9

    def test_an_opaque_rgba_mask_reads_by_colour_like_any_opaque_png(self, tmp_path):
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[5:10, 5:10, :3] = 255  # a white square on opaque black
        Image.fromarray(rgba, "RGBA").save(tmp_path / "opaque.png")

        result = load_paint(str(tmp_path / "opaque.png"), 20, 20)

        assert result[5:10, 5:10].all() and result.sum() == 25

    def test_a_fully_opaque_white_rgba_mask_is_fully_painted(self, tmp_path):
        Image.fromarray(np.full((20, 30, 4), 255, dtype=np.uint8), "RGBA").save(tmp_path / "full.png")

        assert load_paint(str(tmp_path / "full.png"), 30, 20).all()

    def test_an_opaque_black_rgba_mask_has_nothing_painted(self, tmp_path):
        black = np.zeros((20, 20, 4), dtype=np.uint8)
        black[..., 3] = 255
        Image.fromarray(black, "RGBA").save(tmp_path / "black.png")

        with pytest.raises(ValueError, match="no painted area"):
            load_paint(str(tmp_path / "black.png"), 20, 20)

    def test_one_transparent_pixel_makes_alpha_the_paint_whatever_the_colour(self, tmp_path):
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[..., 3] = 255  # opaque black everywhere ...
        rgba[0, 0, 3] = 0  # ... but one transparent pixel, so alpha decides
        Image.fromarray(rgba, "RGBA").save(tmp_path / "alpha.png")

        result = load_paint(str(tmp_path / "alpha.png"), 20, 20)

        assert result.sum() == 399 and not result[0, 0]

    def test_a_sixteen_bit_mask_loads(self, tmp_path):
        pixels = np.zeros((20, 20), dtype=np.uint16)
        pixels[4:8, 4:8] = 300
        Image.fromarray(pixels).save(tmp_path / "deep.png")

        assert load_paint(str(tmp_path / "deep.png"), 20, 20).sum() == 16

    def test_scales_a_smaller_mask_with_nearest_neighbour(self, tmp_path):
        small = _block(60, 80)
        path = _save_mask(tmp_path / "mask.png", small)

        result = load_paint(str(path), 160, 120)

        assert result.shape == (120, 160)
        assert np.array_equal(result, np.repeat(np.repeat(small, 2, axis=0), 2, axis=1))

    def test_scales_a_larger_mask_down(self, tmp_path):
        large = _block(240, 320)
        path = _save_mask(tmp_path / "mask.png", large)

        result = load_paint(str(path), 160, 120)

        assert result.shape == (120, 160) and result.any()

    @pytest.mark.parametrize(
        ("size", "mask_size"),
        [
            ((2000, 1500), (1000, 750)),
            # what the browser sends: both sides of the picture scaled to 4 MP and rounded on their own
            ((2637, 5134), (1433, 2791)),
            ((1440, 3200), (1342, 2981)),
            ((1000, 750), (2000, 1501)),  # a mask larger than the picture, rounded on the picture's side
        ],
    )
    def test_a_mask_scaled_by_one_factor_and_rounded_per_side_loads(self, tmp_path, size, mask_size):
        path = _save_mask(tmp_path / "mask.png", _block(mask_size[1], mask_size[0]))

        assert load_paint(str(path), *size).shape == (size[1], size[0])

    @pytest.mark.parametrize("mask_size", [(1000, 748), (1000, 752), (500, 500), (1433, 2800)])
    def test_a_mask_of_another_shape_is_refused_with_both_sizes(self, tmp_path, mask_size):
        width, height = mask_size
        path = _save_mask(tmp_path / "mask.png", _block(height, width))

        with pytest.raises(ValueError, match=rf"{width}x{height}.*shape"):
            load_paint(str(path), 2000, 1500)

    def test_a_mask_with_nothing_painted_is_refused(self, tmp_path):
        path = _save_mask(tmp_path / "mask.png", np.zeros((60, 80), dtype=bool))

        with pytest.raises(ValueError, match="no painted area"):
            load_paint(str(path), 80, 60)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_paint(str(tmp_path / "gone.png"), 80, 60)

    def test_a_file_that_is_not_an_image_is_refused(self, tmp_path):
        (tmp_path / "mask.png").write_bytes(b"not a picture")

        with pytest.raises(ValueError, match="not a readable PNG"):
            load_paint(str(tmp_path / "mask.png"), 80, 60)

    def test_an_image_in_another_format_is_refused(self, tmp_path):
        Image.new("RGB", (80, 60), (255, 255, 255)).save(tmp_path / "mask.jpg")

        with pytest.raises(ValueError, match="must be a PNG"):
            load_paint(str(tmp_path / "mask.jpg"), 80, 60)


class TestEditMode:
    @pytest.mark.parametrize(
        ("has_mask", "extend", "expected"),
        [
            (True, None, "paint"),
            (False, ("top", 256), "extend"),
            (True, ("top", 256), "both"),
            (False, None, "whole"),
            (False, ("left", 128), "extend"),
        ],
    )
    def test_the_mode_follows_the_paint_and_the_extension(self, has_mask, extend, expected):
        assert edit_mode(has_mask, extend) == expected


class TestSourceIsCutout:
    def test_an_opaque_picture_is_not_a_cutout_whatever_its_mode(self, tmp_path):
        Image.new("RGB", (40, 40), (1, 2, 3)).save(tmp_path / "rgb.png")
        Image.new("RGBA", (40, 40), (1, 2, 3, 255)).save(tmp_path / "rgba.png")
        Image.new("RGB", (40, 40), (1, 2, 3)).save(tmp_path / "photo.jpg")

        assert [source_is_cutout(str(tmp_path / name)) for name in ("rgb.png", "rgba.png", "photo.jpg")] == [False, False, False]

    def test_one_transparent_pixel_makes_a_cutout(self, tmp_path):
        image = Image.new("RGBA", (40, 40), (1, 2, 3, 255))
        image.putpixel((3, 3), (1, 2, 3, 249))
        image.save(tmp_path / "cut.png")

        assert source_is_cutout(str(tmp_path / "cut.png")) is True

    def test_alpha_of_250_or_more_counts_as_opaque(self, tmp_path):
        Image.new("RGBA", (40, 40), (1, 2, 3, 250)).save(tmp_path / "soft.png")

        assert source_is_cutout(str(tmp_path / "soft.png")) is False

    def test_a_palette_picture_with_a_transparent_colour_is_a_cutout(self, tmp_path):
        palette = Image.new("P", (40, 40), 0)
        palette.putpalette([0, 0, 0, 255, 0, 0])
        palette.putpixel((1, 1), 1)
        palette.save(tmp_path / "pal.png", transparency=0)

        assert source_is_cutout(str(tmp_path / "pal.png")) is True


class TestCheckEditInputs:
    EXTEND = ("top", 256)

    def test_every_valid_combination_passes(self):
        check_edit_inputs("paint", None, False, 0, True)
        check_edit_inputs("paint", None, True, 0, True)
        check_edit_inputs("extend", self.EXTEND, False, 0, True)
        check_edit_inputs("both", self.EXTEND, False, 0, True)
        check_edit_inputs("whole", None, False, 4, True)

    def test_extend_on_a_cutout_is_refused(self):
        for mode in ("extend", "both"):
            with pytest.raises(ValueError, match="transparent background"):
                check_edit_inputs(mode, self.EXTEND, True, 4, True)

    def test_no_paint_and_no_extend_on_a_model_without_references_is_refused(self):
        with pytest.raises(ValueError, match="Paint the area to change"):
            check_edit_inputs("whole", None, False, 0, True)

    def test_no_paint_on_a_cutout_is_refused_even_on_a_model_with_references(self):
        with pytest.raises(ValueError, match="Paint the area to change"):
            check_edit_inputs("whole", None, True, 4, True)

    def test_a_model_without_mask_editing_is_refused(self):
        for mode, extend in (("paint", None), ("extend", self.EXTEND), ("whole", None)):
            with pytest.raises(ValueError, match="cannot edit"):
                check_edit_inputs(mode, extend, False, 4, False)
