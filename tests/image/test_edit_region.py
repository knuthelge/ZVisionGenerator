"""Test the pure image side of mask editing: crop planning, the Extend canvas and the paste-back."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests._images import noise_picture, painted_box
from zvisiongenerator.processing.edit_region import (
    BIG_MASK_FRACTION,
    PAINT_FADE,
    Plan,
    extend_canvas,
    blurred_canvas,
    edit_output_size,
    extend_draw_size,
    grid_size,
    match_join,
    paste_paint,
    paste_strip,
    fade_alpha,
    plan_region,
    soft_cell_mask,
    target_size,
)


def test_target_size_is_one_megapixel_aligned_to_16():
    assert target_size(1200, 800) == (1232, 816)
    assert target_size(300, 300) == (992, 992)
    width, height = target_size(4000, 1500)
    assert width % 16 == height % 16 == 0
    assert width * height == pytest.approx(1_000_000, rel=0.06)


@pytest.mark.parametrize(("size", "expected"), [((1920, 1088), (1920, 1088)), ((1921, 1081), (1920, 1088)), ((1930, 1075), (1936, 1072)), ((5, 9), (16, 16))])
def test_grid_size_rounds_each_side_to_the_nearest_16_with_no_upper_cap(size, expected):
    assert grid_size(*size) == expected


@pytest.mark.parametrize(("extend", "expected"), [(None, (1921, 1081)), (("right", 128), (2049, 1081)), (("top", 64), (1921, 1145))])
def test_edit_output_size_keeps_the_pictures_size_or_adds_the_extend(extend, expected):
    assert edit_output_size(1921, 1081, extend=extend) == expected


class TestExtendDrawSize:
    @pytest.mark.parametrize(
        ("size", "side", "amount", "expected"),
        [
            ((1000, 750), "right", 512, ((992, 736), 512)),  # under a megapixel: its own size, snapped down to cells
            ((320, 240), "bottom", 512, ((320, 240), 512)),
            ((64, 64), "right", 512, ((64, 64), 512)),
            ((6000, 4000), "left", 128, ((1232, 816), 32)),  # a large picture: about a megapixel, the strip at the same scale
        ],
    )
    def test_the_source_is_drawn_at_about_a_megapixel_and_never_enlarged_with_a_strip_of_whole_cells(self, size, side, amount, expected):
        drawn, strip = extend_draw_size(size, side, amount)

        assert (drawn, strip) == expected
        assert drawn[0] % 16 == drawn[1] % 16 == strip % 16 == 0
        assert drawn[0] <= size[0] and drawn[1] <= size[1]


class TestPlanRegion:
    def test_a_mask_over_half_the_canvas_takes_the_whole_canvas(self):
        paint = painted_box(800, 1200, (0, 0, 1200, 401))
        assert paint.mean() > BIG_MASK_FRACTION

        assert plan_region(paint) == Plan((0, 0, 1200, 800), (1232, 816))

    def test_a_mask_of_exactly_half_still_gets_a_crop(self):
        paint = painted_box(1000, 4000, (0, 0, 2000, 1000))

        assert paint.mean() == BIG_MASK_FRACTION
        assert plan_region(paint).box != (0, 0, 4000, 1000)

    @pytest.mark.parametrize("where", ["top-left", "top-right", "bottom-left", "bottom-right", "left", "top", "centre"])
    @pytest.mark.parametrize("size", [(24, 24), (300, 40), (40, 300)])
    def test_the_crop_is_aligned_inside_the_canvas_and_between_one_to_two_and_two_to_one(self, where, size):
        width, height = 1500, 1000
        w, h = size
        x = {"top-left": 0, "bottom-left": 0, "left": 0, "top-right": width - w, "bottom-right": width - w, "top": 300, "centre": (width - w) // 2}[where]
        y = {"top-left": 0, "top-right": 0, "top": 0, "bottom-left": height - h, "bottom-right": height - h, "left": 300, "centre": (height - h) // 2}[where]
        paint = painted_box(height, width, (x, y, x + w, y + h))

        plan = plan_region(paint)
        x0, y0, x1, y1 = plan.box

        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert plan.box != (0, 0, width, height)
        assert paint[y0:y1, x0:x1].sum() == paint.sum()
        assert plan.target[0] % 16 == plan.target[1] % 16 == 0
        assert 0.5 <= plan.target[0] / plan.target[1] <= 2.0
        assert 0.5 <= (x1 - x0) / (y1 - y0) <= 2.0
        assert x1 - x0 >= 384 and y1 - y0 >= 384

    def test_a_small_canvas_caps_the_crop_at_the_canvas(self):
        plan = plan_region(painted_box(300, 300, (100, 100, 120, 120)))

        assert plan.box == (0, 0, 300, 300)
        assert plan.target == (992, 992)


class TestFadeAlpha:
    def test_alpha_is_one_inside_falls_outward_and_is_zero_from_the_fade_width_out(self):
        paint = painted_box(300, 300, (100, 100, 200, 200))

        alpha = fade_alpha(paint, 12)

        assert alpha.dtype == np.float32
        assert (alpha[paint] == 1.0).all()
        column = alpha[150, 100 - 13 : 100][::-1]  # 1, 2, ... 13 px left of the paint
        assert column[0] > 0.98 and (np.diff(column[:12]) < 0).all()
        assert (column[11:] == 0.0).all()  # 12 px out and beyond
        assert (alpha[: 100 - 12, :] == 0.0).all()

    def test_a_thin_stroke_is_fully_generated(self):
        alpha = fade_alpha(painted_box(100, 200, (40, 30, 54, 70)), 12)

        assert (alpha[30:70, 40:54] == 1.0).all()

    def test_the_canvas_border_is_not_a_paint_edge(self):
        alpha = fade_alpha(painted_box(100, 200, (0, 0, 100, 100)), 12)

        assert alpha[50, 0] == 1.0 and alpha[0, 50] == 1.0 and alpha[99, 50] == 1.0
        assert 0.0 < alpha[50, 105] < 1.0

    def test_a_fully_painted_array_is_one_everywhere(self):
        alpha = fade_alpha(np.ones((40, 60), dtype=bool), 12)

        assert alpha.dtype == np.float32 and (alpha == 1.0).all()


class TestPastePaint:
    def test_inside_the_paint_the_result_is_generated_and_from_12_px_out_the_source_is_byte_exact(self):
        canvas = noise_picture(640, 480, seed=1)
        paint = painted_box(480, 640, (200, 150, 320, 164))  # a thin stroke
        plan = plan_region(paint)
        generated = noise_picture(1000, 700, seed=3)

        result = np.asarray(paste_paint(canvas, generated, plan, paint, PAINT_FADE))

        x0, y0, x1, y1 = plan.box
        fresh = np.asarray(generated.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS))
        source = np.asarray(canvas)
        inside = paint[y0:y1, x0:x1]
        assert np.abs(result[y0:y1, x0:x1][inside].astype(int) - fresh[inside].astype(int)).max() <= 1
        far = np.ones(paint.shape, dtype=bool)
        far[150 - 12 : 164 + 12, 200 - 12 : 320 + 12] = False
        assert np.array_equal(result[far], source[far])
        ring = np.zeros(paint.shape, dtype=bool)
        ring[150 - 11 : 164 + 11, 200 - 11 : 320 + 11] = True
        ring &= ~paint
        assert (result[ring] != source[ring]).any()
        assert result.shape == source.shape

    def test_only_the_crop_box_is_touched(self):
        canvas = noise_picture(2000, 2000, seed=4)
        paint = painted_box(2000, 2000, (1000, 1000, 1040, 1040))
        plan = plan_region(paint)
        x0, y0, x1, y1 = plan.box

        result = np.asarray(paste_paint(canvas, noise_picture(900, 900, seed=5), plan, paint, PAINT_FADE))
        source = np.asarray(canvas)

        assert (x1 - x0) < 2000
        assert np.array_equal(result[:y0], source[:y0]) and np.array_equal(result[y1:], source[y1:])
        assert np.array_equal(result[:, :x0], source[:, :x0]) and np.array_equal(result[:, x1:], source[:, x1:])

    def test_the_fade_stops_at_the_crop_box(self):
        canvas = Image.new("RGB", (400, 400), (0, 0, 0))
        paint = painted_box(400, 400, (100, 100, 300, 300))
        plan = Plan((96, 96, 304, 304), (208, 208))  # the box ends 4 px beyond the paint

        result = np.asarray(paste_paint(canvas, Image.new("RGB", (208, 208), (255, 255, 255)), plan, paint, PAINT_FADE))

        assert (result[96:100, 150] > 0).all() and (result[:96] == 0).all() and (result[304:] == 0).all()

    def test_painting_everything_replaces_everything(self):
        canvas = noise_picture(300, 200, seed=34)
        generated = noise_picture(512, 384, seed=35)
        paint = np.ones((200, 300), dtype=bool)

        result = paste_paint(canvas, generated, plan_region(paint), paint, PAINT_FADE)

        assert np.array_equal(np.asarray(result), np.asarray(generated.resize((300, 200), Image.Resampling.LANCZOS)))


class TestPasteStrip:
    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    def test_the_new_area_is_generated_right_up_to_a_hard_join_and_the_source_is_exact(self, side):
        source = noise_picture(160, 120, seed=31)
        canvas, paint, (left, top) = extend_canvas(source, side, 128)
        generated = noise_picture(canvas.width, canvas.height, seed=32)

        result = np.asarray(paste_strip(canvas, generated, (left, top, left + 160, top + 120), (left, top, left + 160, top + 120), canvas.size))

        assert np.array_equal(result[paint], np.asarray(generated)[paint])
        assert np.array_equal(result[~paint], np.asarray(canvas)[~paint])


class TestPasteStripDrawnSmaller:
    @pytest.mark.parametrize("side", ["left", "top"])
    def test_the_strip_and_the_source_are_mapped_apart_from_the_drawn_crop(self, side):
        # The source is drawn at half scale and the strip at 32 px, where the real amount 128 would give 64: the strip is not the canvas scaled.
        source = noise_picture(160, 120, seed=41)
        canvas, paint, (left, top) = extend_canvas(source, side, 128)
        horizontal = side == "left"
        drawn_size = (112, 60) if horizontal else (80, 92)
        drawn_source = (32, 0, 112, 60) if horizontal else (0, 32, 80, 92)
        generated = Image.new("RGB", drawn_size, (0, 0, 255))
        generated.paste((255, 0, 0), (0, 0, 32, 60) if horizontal else (0, 0, 80, 32))  # the strip is red

        result = np.asarray(paste_strip(canvas, generated, (left, top, left + 160, top + 120), drawn_source, drawn_size))

        far = result[:, :100] if horizontal else result[:100]  # the resampling filter may blend a few pixels at the join itself
        assert (far == (255, 0, 0)).all()  # the real strip shows the drawn strip, not part of the drawn source
        assert np.array_equal(result[~paint], np.asarray(canvas)[~paint])


class TestBlurredCanvas:
    @pytest.mark.parametrize(("side", "grown"), [("left", (160 + 64, 120)), ("right", (160 + 64, 120)), ("top", (160, 120 + 64)), ("bottom", (160, 120 + 64))])
    def test_the_source_is_in_place_and_the_new_strip_is_blurred_with_no_texture_to_copy(self, side, grown):
        source = noise_picture(160, 120, seed=51)
        canvas, strip, (left, top) = extend_canvas(source, side, 64)

        reference = blurred_canvas(source, side, 64)

        pixels = np.asarray(reference)
        assert reference.size == grown == canvas.size
        assert np.array_equal(pixels[~strip], np.asarray(canvas)[~strip])  # the source, untouched
        assert np.asarray(canvas)[strip].std() > 20 and pixels[strip].std() < 10  # noise in the edge fill, an even blur in the reference


class TestMatchJoin:
    @staticmethod
    def _canvas() -> tuple[np.ndarray, np.ndarray]:
        strip = np.zeros((120, 200), dtype=bool)
        strip[:, :80] = True
        ramp = np.linspace(40, 200, 200, dtype=np.float32)[None, :, None] * np.ones((120, 1, 3), dtype=np.float32)
        return np.rint(ramp).astype(np.uint8), strip

    def test_a_drift_of_the_model_over_the_source_is_taken_off_the_strip_only(self):
        canvas, strip = self._canvas()
        drifted = np.clip(canvas.astype(np.float32) * 0.9 + 10, 0, 255).astype(np.uint8)  # what the model drew: the whole frame, a little off

        matched = match_join(drifted, drifted, canvas, strip)

        assert np.array_equal(matched[~strip], drifted[~strip])  # the source part is not touched
        before = np.abs(drifted[:, 80:120].astype(int) - canvas[:, 80:120].astype(int)).mean()
        assert np.abs(matched[:, 40:80].astype(int) - canvas[:, 40:80].astype(int)).mean() < np.abs(drifted[:, 40:80].astype(int) - canvas[:, 40:80].astype(int)).mean()
        assert before > 2  # the drift was there to find

    def test_the_slope_is_limited(self):
        canvas, strip = self._canvas()
        squashed = np.clip(canvas.astype(np.float32) * 0.3 + 100, 0, 255).astype(np.uint8)  # a slope of about 3 would be needed

        matched = match_join(squashed, squashed, canvas, strip)

        steepest = np.abs(np.diff(matched[60, :80].astype(float).mean(axis=-1)) / np.maximum(np.diff(squashed[60, :80].astype(float).mean(axis=-1)), 1e-6)).max()
        assert steepest < 2.0  # not the 3x the fit alone would ask

    def test_a_flat_band_changes_the_strip_by_an_offset_only(self):
        canvas = np.full((120, 200, 3), 100, dtype=np.uint8)
        strip = np.zeros((120, 200), dtype=bool)
        strip[:, :80] = True
        view = np.full((120, 200, 3), 90, dtype=np.uint8)
        generated = noise_picture(200, 120, seed=52)
        generated_array = np.asarray(generated)

        matched = match_join(generated_array, view, canvas, strip)

        assert np.array_equal(matched[~strip], generated_array[~strip])
        assert matched[strip].mean() > generated_array[strip].mean()  # lifted by the offset, with the texture kept
        assert np.corrcoef(matched[strip].ravel(), generated_array[strip].ravel())[0, 1] > 0.99


class TestSoftCellMask:
    def test_each_cell_holds_the_painted_fraction(self):
        paint = np.zeros((32, 32), dtype=bool)
        paint[0:8, 0:16] = True  # the top half of cell (0, 0)
        paint[16:32, 16:32] = True  # the whole of cell (1, 1)
        paint[16:20, 0:4] = True  # a quarter of a quarter of cell (1, 0)

        cells = soft_cell_mask(paint)

        assert cells.dtype == np.float32
        assert cells.tolist() == [[0.5, 0.0], [4 * 4 / 256, 1.0]]

    def test_a_mask_that_is_not_whole_cells_is_refused(self):
        with pytest.raises(ValueError, match="cells"):
            soft_cell_mask(np.zeros((30, 32), dtype=bool))


class TestExtendCanvas:
    @pytest.mark.parametrize(
        ("side", "offset", "grown"),
        [("top", (0, 128), (160, 248)), ("bottom", (0, 0), (160, 248)), ("left", (128, 0), (288, 120)), ("right", (0, 0), (288, 120))],
    )
    def test_the_source_stays_byte_exact_at_the_offset_and_the_new_area_is_the_paint(self, side, offset, grown):
        source = noise_picture(160, 120, seed=7)

        canvas, paint, at = extend_canvas(source, side, 128)

        left, top = at
        assert at == offset and canvas.size == grown
        assert np.array_equal(np.asarray(canvas)[top : top + 120, left : left + 160], np.asarray(source))
        assert paint.shape == (grown[1], grown[0])
        assert not paint[top : top + 120, left : left + 160].any()
        assert paint.sum() == grown[0] * grown[1] - 160 * 120

    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    def test_a_flat_picture_extends_in_its_own_colour(self, side):
        canvas, paint, _ = extend_canvas(Image.new("RGB", (200, 200), (90, 120, 150)), side, 128)

        assert np.unique(np.asarray(canvas)[paint], axis=0).tolist() == [[90, 120, 150]]

    def test_the_new_area_repeats_the_edge_pixels(self):
        source = noise_picture(160, 120, seed=9)

        canvas, paint, _ = extend_canvas(source, "right", 128)

        new_area = np.asarray(canvas)[paint].reshape(120, 128, 3)
        assert (new_area == np.asarray(source)[:, -1:, :]).all()
