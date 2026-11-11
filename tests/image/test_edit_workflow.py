"""Test the edit workflow's stages with a mock backend: the pixels outside the paint stay the source's own."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageOps

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.processing.edit_region import target_size
from zvisiongenerator.utils.filename import edit_output_stem
from zvisiongenerator.workflows import build_edit_workflow


def _noise(width: int, height: int, seed: int = 1) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


def _photo(path: Path, width: int = 480, height: int = 320) -> Path:
    _noise(width, height, seed=7).save(path)
    return path


def _sideways_photo(path: Path) -> Path:
    """Write a 200 x 120 JPEG tagged to show upright as 120 x 200."""
    exif = Image.Exif()
    exif[0x0112] = 6
    _noise(200, 120, seed=3).save(path, quality=95, exif=exif)
    return path


def _cutout(path: Path, width: int = 480, height: int = 320) -> Path:
    """Write an RGBA picture with a transparent left, a soft middle and an opaque right, each with its own colours."""
    pixels = np.random.default_rng(5).integers(0, 256, (height, width, 4), dtype=np.uint8)
    pixels[:, : width // 3, 3] = 0
    pixels[:, width // 3 : 2 * width // 3, 3] = np.linspace(1, 254, width // 3, dtype=np.uint8)
    pixels[:, 2 * width // 3 :, 3] = 255
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def _mask(width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Return the painted area, painted inside *box* ``(x0, y0, x1, y1)``."""
    painted = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    painted[y0:y1, x0:x1] = True
    return painted


def _beyond_the_fade(width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Return the pixels at least 12 px from *box*: the ones an edit must leave exactly as they were."""
    x0, y0, x1, y1 = box
    beyond = np.ones((height, width), dtype=bool)
    beyond[max(0, y0 - 12) : y1 + 12, max(0, x0 - 12) : x1 + 12] = False
    return beyond


def _backend(*, size: tuple[int, int] = (777, 555)) -> MagicMock:
    """Return a backend mock whose model returns noise at an unrelated size."""
    backend = MagicMock(spec=ImageBackend)
    backend.edit_region.side_effect = lambda **kwargs: _noise(*size, seed=11)
    backend.text_to_image.side_effect = lambda **kwargs: _noise(kwargs["width"], kwargs["height"], seed=12)
    return backend


def _run(tmp_path: Path, source: Path, backend: MagicMock, **overrides) -> tuple[StageOutcome, ImageWorkingArtifacts, ImageGenerationRequest]:
    values = dict(
        backend=backend,
        model=object(),
        prompt="a red door",
        edit_source=str(source),
        output_dir=str(tmp_path / "out"),
        filename_base="x",
        steps=4,
        seed=9,
        max_reference_images=0,
    )
    filename = overrides.pop("filename", f"{source.stem}_edit")
    values.update(overrides)
    request = ImageGenerationRequest(**values)
    artifacts = ImageWorkingArtifacts(filename=filename)
    outcome = build_edit_workflow().run(request, artifacts)
    return outcome, artifacts, request


def _saved(artifacts: ImageWorkingArtifacts) -> Image.Image:
    assert artifacts.filepath is not None
    with Image.open(artifacts.filepath) as image:
        image.load()
        return image.copy()


class TestBeyondThePaintFadeIsTheSource:
    def test_a_normal_source(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        box = (150, 100, 260, 200)
        mask = _mask(480, 320, box)

        outcome, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        saved = np.asarray(_saved(artifacts))
        original = np.asarray(Image.open(source).convert("RGB"))
        outside = _beyond_the_fade(480, 320, box)
        assert outcome is StageOutcome.success
        assert saved.shape == original.shape
        assert np.array_equal(saved[outside], original[outside])
        assert not np.array_equal(saved[100:200, 150:260], original[100:200, 150:260])

    def test_an_exif_rotated_jpeg_is_edited_upright(self, tmp_path):
        source = _sideways_photo(tmp_path / "phone.jpg")
        mask = _mask(120, 200, (30, 60, 80, 120))

        outcome, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        saved = np.asarray(_saved(artifacts))
        with Image.open(source) as opened:
            upright = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        outside = _beyond_the_fade(120, 200, (30, 60, 80, 120))
        assert outcome is StageOutcome.success
        assert saved.shape == (200, 120, 3)
        assert np.array_equal(saved[outside], upright[outside])

    def test_a_cutout_keeps_its_alpha_and_its_own_colours_outside_the_paint(self, tmp_path):
        source = _cutout(tmp_path / "cutout.png")
        box = (170, 100, 330, 220)  # straddles the soft middle
        mask = _mask(480, 320, box)

        outcome, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        saved = _saved(artifacts)
        original = np.asarray(Image.open(source))
        outside = _beyond_the_fade(480, 320, box)
        assert outcome is StageOutcome.success
        assert saved.mode == "RGBA"
        assert np.array_equal(np.asarray(saved)[..., 3], original[..., 3])
        assert np.array_equal(np.asarray(saved)[outside], original[outside])

    def test_a_name_taken_gets_a_counter(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = _mask(480, 320, (150, 100, 260, 200))
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "photo_edit.png").write_bytes(b"taken")

        _, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        assert Path(artifacts.filepath).name == "photo_edit_2.png"

    @pytest.mark.parametrize(
        ("source_name", "taken", "saved"),
        [("photo.png", [], "photo_edit.png"), ("photo_edit.png", ["photo_edit.png"], "photo_edit_2.png"), ("photo_edit_2.png", ["photo_edit.png", "photo_edit_2.png"], "photo_edit_3.png")],
    )
    def test_an_edit_of_an_edit_is_named_by_the_counter_alone(self, tmp_path, source_name, taken, saved):
        source = _photo(tmp_path / source_name)
        (tmp_path / "out").mkdir()
        for name in taken:
            (tmp_path / "out" / name).write_bytes(b"taken")

        _, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=_mask(480, 320, (150, 100, 260, 200)), filename=edit_output_stem(source))

        assert Path(artifacts.filepath).name == saved


class TestPaint:
    def test_the_model_gets_the_crop_and_its_paint_at_the_target_size(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = _mask(480, 320, (150, 100, 260, 200))
        backend = _backend()

        _run(tmp_path, source, backend, edit_paint=mask, scheduler="sched")

        call = backend.edit_region.call_args.kwargs
        assert call["image"].size == (call["paint"].shape[1], call["paint"].shape[0])
        assert call["image"].width % 16 == call["image"].height % 16 == 0
        assert call["paint"].dtype == bool and call["paint"].any()
        assert (call["prompt"], call["seed"], call["steps"], call["scheduler"]) == ("a red door", 9, 4, "sched")
        backend.text_to_image.assert_not_called()

    def test_the_generation_prompt_is_the_resolved_one(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = _mask(480, 320, (150, 100, 260, 200))
        backend = _backend()

        _run(tmp_path, source, backend, edit_paint=mask, prompt="a {red|blue} door", resolved_prompt="a blue door")

        assert backend.edit_region.call_args.kwargs["prompt"] == "a blue door"

    def test_progress_steps_are_labelled_as_generation(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = _mask(480, 320, (150, 100, 260, 200))
        backend = _backend()
        events: list[dict] = []

        _run(tmp_path, source, backend, edit_paint=mask, step_callback=events.append)
        backend.edit_region.call_args.kwargs["step_callback"]({"current_step": 2, "total_steps": 4})

        assert events == [{"phase": "image_generate", "current_step": 2, "total_steps": 4}]

    def test_a_mask_of_another_size_is_refused_with_both_sizes(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = _mask(240, 160, (10, 10, 50, 50))

        with pytest.raises(ValueError, match=r"240x160.*480x320"):
            _run(tmp_path, source, _backend(), edit_paint=mask)

    def test_a_missing_source_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="gone.png"):
            _run(tmp_path, tmp_path / "gone.png", _backend())


class TestWholePicture:
    def test_a_model_with_references_redraws_from_the_crop_with_no_inpainting(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        backend = _backend()

        outcome, artifacts, _ = _run(tmp_path, source, backend, max_reference_images=4, supports_negative_prompt=False)

        target = target_size(480, 320)
        call = backend.text_to_image.call_args.kwargs
        backend.edit_region.assert_not_called()
        assert outcome is StageOutcome.success
        assert (call["width"], call["height"]) == target
        assert len(call["reference_images"]) == 1 and call["reference_images"][0].size == target
        assert _saved(artifacts).size == target

    def test_a_model_without_references_is_told_to_paint(self, tmp_path):
        with pytest.raises(ValueError, match="Paint the area to change"):
            _run(tmp_path, _photo(tmp_path / "photo.png"), _backend(), max_reference_images=0)

    def test_a_cutout_is_told_to_paint_even_with_references(self, tmp_path):
        with pytest.raises(ValueError, match="Paint the area to change"):
            _run(tmp_path, _cutout(tmp_path / "cutout.png"), _backend(), max_reference_images=4)


class TestExtend:
    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    def test_the_picture_grows_on_that_side_and_the_source_is_exact(self, tmp_path, side):
        source = _photo(tmp_path / "photo.png", 320, 240)
        backend = _backend()

        outcome, artifacts, _ = _run(tmp_path, source, backend, edit_extend=(side, 128))

        saved = np.asarray(_saved(artifacts))
        original = np.asarray(Image.open(source).convert("RGB"))
        left, top = {"top": (0, 128), "bottom": (0, 0), "left": (128, 0), "right": (0, 0)}[side]
        width, height = 320 + (128 if side in ("left", "right") else 0), 240 + (128 if side in ("top", "bottom") else 0)
        assert outcome is StageOutcome.success
        assert saved.shape == (height, width, 3)
        assert np.array_equal(saved[top : top + 240, left : left + 320], original)
        assert artifacts.edit.plan.box == (0, 0, width, height)

    @pytest.mark.parametrize(("size", "megapixels"), [((1600, 1300), 2.0), ((1200, 1000), 1.328), ((600, 400), 1.0)])
    def test_the_whole_canvas_is_drawn_at_its_own_size_up_to_two_megapixels_and_not_below_one(self, tmp_path, size, megapixels):
        source = _photo(tmp_path / "photo.png", *size)
        backend = _backend()

        _run(tmp_path, source, backend, edit_extend=("right", 128))

        drawn = backend.edit_region.call_args.kwargs["image"].size
        assert drawn == target_size(size[0] + 128, size[1], megapixels)

    def test_a_cutout_cannot_be_extended(self, tmp_path):
        with pytest.raises(ValueError, match="transparent background"):
            _run(tmp_path, _cutout(tmp_path / "cutout.png"), _backend(), edit_extend=("top", 128))


class TestPaintAndExtend:
    def test_the_source_size_mask_is_one_mask_with_the_new_area_and_the_whole_canvas(self, tmp_path):
        source = _photo(tmp_path / "photo.png", 320, 240)
        mask = _mask(320, 240, (100, 60, 200, 160))
        backend = _backend()

        outcome, artifacts, _ = _run(tmp_path, source, backend, edit_paint=mask, edit_extend=("left", 128))

        saved = np.asarray(_saved(artifacts))
        original = np.asarray(Image.open(source).convert("RGB"))
        keep = _beyond_the_fade(320, 240, (100, 60, 200, 160))
        edit = artifacts.edit
        assert outcome is StageOutcome.success
        assert edit.plan.box == (0, 0, 448, 240)
        assert saved.shape == (240, 448, 3)
        assert np.array_equal(saved[:, 128:][keep], original[keep])
        assert edit.paint[:, :128].all() and edit.paint[60:160, 128 + 100 : 128 + 200].all() and not edit.paint[:, 128:][keep].any()
        assert backend.edit_region.call_args.kwargs["image"].size == target_size(448, 240)

    def test_a_mask_at_the_extended_size_is_refused(self, tmp_path):
        source = _photo(tmp_path / "photo.png", 320, 240)
        mask = _mask(448, 240, (10, 10, 50, 50))

        with pytest.raises(ValueError, match=r"448x240.*320x240"):
            _run(tmp_path, source, _backend(), edit_paint=mask, edit_extend=("left", 128))


class TestSkipped:
    def test_a_stopped_edit_is_skipped_and_nothing_is_saved(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = _mask(480, 320, (150, 100, 260, 200))
        backend = _backend()
        backend.edit_region.side_effect = None
        backend.edit_region.return_value = None

        outcome, artifacts, _ = _run(tmp_path, source, backend, edit_paint=mask)

        assert outcome is StageOutcome.skipped
        assert artifacts.image is None
        assert list((tmp_path / "out").iterdir()) == []
