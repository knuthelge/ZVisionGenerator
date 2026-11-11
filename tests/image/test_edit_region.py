"""Test the pure image side of mask editing: crop planning, the Extend canvas and the paste-back."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.processing.edit_region import (
    BIG_MASK_FRACTION,
    PAINT_FADE,
    Plan,
    combined_paint,
    composite,
    extend_canvas,
    extend_megapixels,
    fade_alpha,
    plan_region,
    soft_cell_mask,
    target_size,
)


def _paint(height: int, width: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    paint = np.zeros((height, width), dtype=bool)
    paint[y : y + h, x : x + w] = True
    return paint


def _noise(width: int, height: int, seed: int = 0) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_target_size_is_one_megapixel_aligned_to_16():
    assert target_size(1200, 800) == (1232, 816)
    assert target_size(300, 300) == (992, 992)
    width, height = target_size(4000, 1500)
    assert width % 16 == height % 16 == 0
    assert width * height == pytest.approx(1_000_000, rel=0.06)


def test_an_extend_draws_a_canvas_at_its_own_size_up_to_two_megapixels_and_never_below_one():
    assert extend_megapixels(2000, 1500) == 2.0
    assert extend_megapixels(5984, 3648) == 2.0
    assert extend_megapixels(1600, 1000) == pytest.approx(1.6)
    assert extend_megapixels(1000, 1000) == extend_megapixels(800, 600) == 1.0
    width, height = target_size(5984, 3648, extend_megapixels(5984, 3648))
    assert width % 16 == height % 16 == 0
    assert width * height == pytest.approx(2_000_000, rel=0.06)


class TestPlanRegion:
    @pytest.mark.parametrize("whole", [False, True])
    def test_a_mask_over_half_the_canvas_and_whole_take_the_whole_canvas(self, whole):
        paint = _paint(800, 1200, 0, 0, 1200, 401) if not whole else _paint(800, 1200, 500, 300, 20, 20)
        assert paint.mean() > BIG_MASK_FRACTION or whole

        plan = plan_region(paint, whole=whole)

        assert plan == Plan((0, 0, 1200, 800), (1232, 816))

    def test_a_big_mask_stays_at_one_megapixel_unless_the_caller_asks_for_more(self):
        paint = _paint(2000, 3000, 0, 0, 3000, 1500)

        assert plan_region(paint).target == target_size(3000, 2000)
        assert plan_region(paint, whole_megapixels=2.0).target == target_size(3000, 2000, 2.0)

    def test_a_mask_of_exactly_half_still_gets_a_crop(self):
        paint = _paint(1000, 4000, 0, 0, 2000, 1000)

        assert paint.mean() == BIG_MASK_FRACTION
        assert plan_region(paint).box != (0, 0, 4000, 1000)

    @pytest.mark.parametrize("canvas", [(1024, 1024), (1500, 1000), (900, 1400)])
    @pytest.mark.parametrize("where", ["top-left", "top-right", "bottom-left", "bottom-right", "left", "top", "centre"])
    @pytest.mark.parametrize("size", [(24, 24), (300, 40), (40, 300)])
    def test_the_crop_is_aligned_inside_the_canvas_and_between_one_to_two_and_two_to_one(self, canvas, where, size):
        width, height = canvas
        w, h = size
        x = {"top-left": 0, "bottom-left": 0, "left": 0, "top-right": width - w, "bottom-right": width - w, "top": 300, "centre": (width - w) // 2}[where]
        y = {"top-left": 0, "top-right": 0, "top": 0, "bottom-left": height - h, "bottom-right": height - h, "left": 300, "centre": (height - h) // 2}[where]
        paint = _paint(height, width, x, y, w, h)

        plan = plan_region(paint)
        x0, y0, x1, y1 = plan.box

        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert plan.box != (0, 0, width, height)
        assert paint[y0:y1, x0:x1].sum() == paint.sum()
        assert plan.target[0] % 16 == plan.target[1] % 16 == 0
        assert 0.5 <= plan.target[0] / plan.target[1] <= 2.0
        assert 0.5 <= (x1 - x0) / (y1 - y0) <= 2.0
        assert x1 - x0 >= 384 and y1 - y0 >= 384

    def test_the_box_adds_the_pad_around_the_painted_area_and_snaps_outward(self):
        paint = _paint(2000, 2000, 1000, 1000, 400, 400)

        x0, y0, x1, y1 = plan_region(paint).box

        pad = round(0.35 * 400)
        assert (x0, y0) == ((1000 - pad) // 16 * 16, (1000 - pad) // 16 * 16)
        assert (x1, y1) == (-(-(1400 + pad) // 16) * 16, -(-(1400 + pad) // 16) * 16)

    def test_a_small_canvas_caps_the_crop_at_the_canvas(self):
        plan = plan_region(_paint(300, 300, 100, 100, 20, 20))

        assert plan.box == (0, 0, 300, 300)
        assert plan.target == (992, 992)


class TestFadeAlpha:
    def test_alpha_is_one_inside_falls_outward_and_is_zero_from_the_fade_width_out(self):
        paint = _paint(300, 300, 100, 100, 100, 100)

        alpha = fade_alpha(paint, 12)

        assert alpha.dtype == np.float32
        assert (alpha[paint] == 1.0).all()
        column = alpha[150, 100 - 13 : 100][::-1]  # 1, 2, ... 13 px left of the paint
        assert column[0] > 0.98 and (np.diff(column[:12]) < 0).all()
        assert (column[11:] == 0.0).all()  # 12 px out and beyond
        assert (alpha[: 100 - 12, :] == 0.0).all()

    def test_a_thin_stroke_is_fully_generated(self):
        alpha = fade_alpha(_paint(100, 200, 40, 30, 14, 40), 12)

        assert (alpha[30:70, 40:54] == 1.0).all()

    def test_the_canvas_border_is_not_a_paint_edge(self):
        alpha = fade_alpha(_paint(100, 200, 0, 0, 100, 100), 12)

        assert alpha[50, 0] == 1.0 and alpha[0, 50] == 1.0 and alpha[99, 50] == 1.0
        assert 0.0 < alpha[50, 105] < 1.0

    def test_a_fully_painted_array_is_one_everywhere(self):
        alpha = fade_alpha(np.ones((40, 60), dtype=bool), 12)

        assert alpha.dtype == np.float32 and (alpha == 1.0).all()


class TestComposite:
    def test_inside_the_paint_the_result_is_generated_and_from_12_px_out_the_source_is_byte_exact(self):
        canvas = _noise(640, 480, seed=1)
        paint = _paint(480, 640, 200, 150, 120, 14)  # a thin stroke
        plan = plan_region(paint)
        generated = _noise(1000, 700, seed=3)

        result = np.asarray(composite(canvas, generated, plan, paint, PAINT_FADE))

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
        canvas = _noise(2000, 2000, seed=4)
        paint = _paint(2000, 2000, 1000, 1000, 40, 40)
        plan = plan_region(paint)
        x0, y0, x1, y1 = plan.box

        result = np.asarray(composite(canvas, _noise(900, 900, seed=5), plan, paint, PAINT_FADE))
        source = np.asarray(canvas)

        assert (x1 - x0) < 2000
        assert np.array_equal(result[:y0], source[:y0]) and np.array_equal(result[y1:], source[y1:])
        assert np.array_equal(result[:, :x0], source[:, :x0]) and np.array_equal(result[:, x1:], source[:, x1:])

    def test_the_fade_stops_at_the_crop_box(self):
        canvas = Image.new("RGB", (400, 400), (0, 0, 0))
        paint = _paint(400, 400, 100, 100, 200, 200)
        plan = Plan((96, 96, 304, 304), (208, 208))  # the box ends 4 px beyond the paint

        result = np.asarray(composite(canvas, Image.new("RGB", (208, 208), (255, 255, 255)), plan, paint, PAINT_FADE))

        assert (result[96:100, 150] > 0).all() and (result[:96] == 0).all() and (result[304:] == 0).all()

    def test_painting_everything_replaces_everything(self):
        canvas = _noise(300, 200, seed=34)
        generated = _noise(512, 384, seed=35)
        paint = np.ones((200, 300), dtype=bool)

        result = composite(canvas, generated, plan_region(paint, whole=True), paint, PAINT_FADE)

        assert np.array_equal(np.asarray(result), np.asarray(generated.resize((300, 200), Image.Resampling.LANCZOS)))


class TestCompositeAnExtend:
    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    def test_the_new_area_is_generated_right_up_to_a_hard_join_and_the_source_is_exact(self, side):
        source = _noise(160, 120, seed=31)
        canvas, paint, (left, top) = extend_canvas(source, side, 128)
        plan = Plan((0, 0, canvas.width, canvas.height), (canvas.width, canvas.height))
        generated = _noise(canvas.width, canvas.height, seed=32)

        result = np.asarray(composite(canvas, generated, plan, paint, PAINT_FADE, (left, top, left + 160, top + 120)))

        assert np.array_equal(result[paint], np.asarray(generated)[paint])
        assert np.array_equal(result[~paint], np.asarray(canvas)[~paint])

    def test_paint_fades_into_the_source_but_not_into_the_new_area(self):
        source = _noise(200, 160, seed=33)
        canvas, extension, (left, top) = extend_canvas(source, "left", 128)
        source_paint = _paint(160, 200, 0, 40, 100, 80)  # touches the join on the left
        paint = combined_paint(source_paint, extension, (left, top))
        plan = Plan((0, 0, canvas.width, canvas.height), (canvas.width, canvas.height))
        generated = Image.new("RGB", canvas.size, (255, 255, 255))
        base = Image.new("RGB", canvas.size, (0, 0, 0))

        result = np.asarray(composite(base, generated, plan, paint, PAINT_FADE, (left, top, left + 200, top + 160)))[..., 0]

        assert (result[:, :128] == 255).all()  # the new area
        assert (result[40:120, 128 : 128 + 100] == 255).all()  # the paint, up to the join
        assert 0 < result[80, 128 + 100 + 5] < 255 and result[80, 128 + 100 + 12] == 0  # fading into the source on the right
        assert 0 < result[40 - 5, 128 + 50] < 255 and result[40 - 12, 128 + 50] == 0  # and above
        assert (result[:, 128 + 100 + 12 :] == 0).all()
        assert result[10, 128 + 1] == 0  # no ring from the new area into the unpainted source beside the join


class TestSoftCellMask:
    def test_each_cell_holds_the_painted_fraction(self):
        paint = np.zeros((32, 32), dtype=bool)
        paint[0:8, 0:16] = True  # the top half of cell (0, 0)
        paint[16:32, 16:32] = True  # the whole of cell (1, 1)
        paint[16:20, 0:4] = True  # a quarter of a quarter of cell (1, 0)

        cells = soft_cell_mask(paint, 16)

        assert cells.dtype == np.float32
        assert cells.tolist() == [[0.5, 0.0], [4 * 4 / 256, 1.0]]

    def test_a_cell_of_eight(self):
        paint = np.zeros((16, 24), dtype=bool)
        paint[0:8, 0:4] = True

        cells = soft_cell_mask(paint, 8)

        assert cells.shape == (2, 3)
        assert cells[0, 0] == 0.5 and cells.sum() == 0.5

    def test_a_mask_that_is_not_whole_cells_is_refused(self):
        with pytest.raises(ValueError, match="cells"):
            soft_cell_mask(np.zeros((30, 32), dtype=bool), 16)


class TestExtendCanvas:
    @pytest.mark.parametrize(
        ("side", "offset", "grown"),
        [("top", (0, 128), (160, 248)), ("bottom", (0, 0), (160, 248)), ("left", (128, 0), (288, 120)), ("right", (0, 0), (288, 120))],
    )
    def test_the_source_stays_byte_exact_at_the_offset_and_the_new_area_is_the_paint(self, side, offset, grown):
        source = _noise(160, 120, seed=7)

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
        source = _noise(160, 120, seed=9)

        canvas, paint, _ = extend_canvas(source, "right", 128)

        new_area = np.asarray(canvas)[paint].reshape(120, 128, 3)
        assert (new_area == np.asarray(source)[:, -1:, :]).all()


class TestCombinedPaint:
    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    def test_the_source_paint_lands_at_the_offset_and_the_new_area_is_painted(self, side):
        source = _noise(160, 120, seed=8)
        source_paint = _paint(120, 160, 20, 30, 40, 10)
        _, extend_paint, offset = extend_canvas(source, side, 128)

        paint = combined_paint(source_paint, extend_paint, offset)

        left, top = offset
        assert paint.shape == extend_paint.shape
        assert np.array_equal(paint[top : top + 120, left : left + 160], source_paint)
        assert paint[extend_paint].all()
        assert paint.sum() == extend_paint.sum() + source_paint.sum()
