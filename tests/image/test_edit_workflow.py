"""Test the edit workflow's stages with a mock backend: the pixels outside the paint stay the source's own."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageOps

from tests._images import display_p3_profile, noise_picture, painted_box, write_sideways_photo
from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.progress_events import run_workflow_with_progress
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.processing.edit_region import extend_draw_size
from zvisiongenerator.utils.input_images import to_srgb
from zvisiongenerator.workflows import build_edit_workflow


def _photo(path: Path, width: int = 480, height: int = 320) -> Path:
    noise_picture(width, height, seed=7).save(path)
    return path


def _cutout(path: Path, width: int = 480, height: int = 320) -> Path:
    """Write an RGBA picture with a transparent left, a soft middle and an opaque right, each with its own colours."""
    pixels = np.random.default_rng(5).integers(0, 256, (height, width, 4), dtype=np.uint8)
    pixels[:, : width // 3, 3] = 0
    pixels[:, width // 3 : 2 * width // 3, 3] = np.linspace(1, 254, width // 3, dtype=np.uint8)
    pixels[:, 2 * width // 3 :, 3] = 255
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def _beyond_the_fade(width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Return the pixels at least 12 px from *box*: the ones an edit must leave exactly as they were."""
    x0, y0, x1, y1 = box
    beyond = np.ones((height, width), dtype=bool)
    beyond[max(0, y0 - 12) : y1 + 12, max(0, x0 - 12) : x1 + 12] = False
    return beyond


def _backend(*, size: tuple[int, int] = (777, 555)) -> MagicMock:
    """Return a backend mock whose model returns noise at an unrelated size."""
    backend = MagicMock(spec=ImageBackend)
    backend.edit_region.side_effect = lambda **kwargs: noise_picture(*size, seed=11)
    backend.text_to_image.side_effect = lambda **kwargs: noise_picture(kwargs["width"], kwargs["height"], seed=12)
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
        mask = painted_box(320, 480, box)

        outcome, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        saved = np.asarray(_saved(artifacts))
        original = np.asarray(Image.open(source).convert("RGB"))
        outside = _beyond_the_fade(480, 320, box)
        assert outcome is StageOutcome.success
        assert saved.shape == original.shape
        assert np.array_equal(saved[outside], original[outside])
        assert not np.array_equal(saved[100:200, 150:260], original[100:200, 150:260])

    def test_an_exif_rotated_jpeg_is_edited_upright(self, tmp_path):
        source = Path(write_sideways_photo(tmp_path / "phone.jpg"))  # stored 40 x 20, shown 20 x 40
        box = (0, 36, 20, 40)
        mask = painted_box(40, 20, box)

        outcome, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        saved = np.asarray(_saved(artifacts))
        with Image.open(source) as opened:
            upright = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        outside = _beyond_the_fade(20, 40, box)
        assert outcome is StageOutcome.success
        assert saved.shape == (40, 20, 3)
        assert outside.any()
        assert np.array_equal(saved[outside], upright[outside])

    def test_a_cutout_keeps_its_alpha_and_its_own_colours_outside_the_paint(self, tmp_path):
        source = _cutout(tmp_path / "cutout.png")
        box = (170, 100, 330, 220)  # straddles the soft middle
        mask = painted_box(320, 480, box)

        outcome, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=mask)

        saved = _saved(artifacts)
        original = np.asarray(Image.open(source))
        outside = _beyond_the_fade(480, 320, box)
        assert outcome is StageOutcome.success
        assert saved.mode == "RGBA"
        assert np.array_equal(np.asarray(saved)[..., 3], original[..., 3])
        assert np.array_equal(np.asarray(saved)[outside], original[outside])

    def test_a_source_with_a_colour_profile_is_edited_and_saved_in_srgb(self, tmp_path):
        source = tmp_path / "p3.png"
        noise_picture(480, 320, seed=2).save(source, icc_profile=display_p3_profile())
        box = (150, 100, 260, 200)

        _, artifacts, _ = _run(tmp_path, source, _backend(), edit_paint=painted_box(320, 480, box))

        saved = _saved(artifacts)
        with Image.open(source) as opened:
            expected = np.asarray(to_srgb(opened).convert("RGB"))
        outside = _beyond_the_fade(480, 320, box)
        assert "icc_profile" not in saved.info
        assert np.array_equal(np.asarray(saved)[outside], expected[outside])
        assert not np.array_equal(expected, np.asarray(Image.open(source).convert("RGB")))


class TestPaint:
    def test_the_model_gets_the_crop_and_its_paint_at_the_target_size(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        mask = painted_box(320, 480, (150, 100, 260, 200))
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
        mask = painted_box(320, 480, (150, 100, 260, 200))
        backend = _backend()

        _run(tmp_path, source, backend, edit_paint=mask, prompt="a {red|blue} door", resolved_prompt="a blue door")

        assert backend.edit_region.call_args.kwargs["prompt"] == "a blue door"

    def test_a_missing_source_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="gone.png"):
            _run(tmp_path, tmp_path / "gone.png", _backend())


class TestWholePicture:
    @pytest.mark.parametrize(
        ("size", "drawn"),
        [((480, 320), (480, 320)), ((1921, 1081), (1920, 1088)), ((1930, 1075), (1936, 1072)), ((10, 7), (16, 16))],
        ids=["on the grid", "off the grid", "rounds to the nearest", "tiny"],
    )
    def test_the_model_draws_at_the_sources_own_size_on_the_grid_and_the_saved_picture_is_the_sources_size(self, tmp_path, size, drawn):
        source = _photo(tmp_path / "photo.png", *size)
        backend = _backend()

        outcome, artifacts, _ = _run(tmp_path, source, backend, max_reference_images=4, supports_negative_prompt=False)

        call = backend.text_to_image.call_args.kwargs
        backend.edit_region.assert_not_called()
        assert outcome is StageOutcome.success
        assert (call["width"], call["height"]) == drawn
        assert len(call["reference_images"]) == 1 and call["reference_images"][0].size == drawn
        assert _saved(artifacts).size == size

    def test_a_picture_already_at_the_sources_size_is_saved_as_the_model_drew_it(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        backend = _backend()

        _, artifacts, _ = _run(tmp_path, source, backend, max_reference_images=4)

        assert np.array_equal(np.asarray(_saved(artifacts)), np.asarray(noise_picture(480, 320, seed=12)))


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

    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    def test_the_model_draws_the_source_unenlarged_with_a_strip_of_whole_cells(self, tmp_path, side):
        source = _photo(tmp_path / "photo.png", 320, 240)
        backend = _backend()

        _, artifacts, _ = _run(tmp_path, source, backend, edit_extend=(side, 128))

        call = backend.edit_region.call_args.kwargs
        drawn, strip = extend_draw_size((320, 240), side, 128)
        horizontal = side in ("left", "right")
        assert strip % 16 == 0 and drawn == (320, 240)  # a source under a megapixel keeps its size
        assert call["reference"].size == call["image"].size  # one frame
        assert call["image"].size == ((drawn[0] + strip, drawn[1]) if horizontal else (drawn[0], drawn[1] + strip))
        assert call["paint"].shape == call["image"].size[::-1]
        cells = call["paint"].reshape(call["paint"].shape[0] // 16, 16, call["paint"].shape[1] // 16, 16).mean(axis=(1, 3))
        assert set(np.unique(cells)) == {0.0, 1.0}  # the join sits on a cell edge: no half-painted cell
        left, top, right, bottom = artifacts.edit.drawn_source_box
        assert call["image"].crop((left, top, right, bottom)).size == drawn


class TestTheReference:
    @pytest.mark.parametrize("side", ["left", "top", "right", "bottom"])
    def test_an_extends_reference_is_the_drawn_canvas_with_a_blurred_strip(self, tmp_path, side):
        source = _photo(tmp_path / "photo.png", 320, 240)
        backend = _backend()

        _, artifacts, _ = _run(tmp_path, source, backend, edit_extend=(side, 128))

        call = backend.edit_region.call_args.kwargs
        reference, crop, strip = np.asarray(call["reference"]), np.asarray(call["image"]), call["paint"]
        assert call["reference"].size == call["image"].size  # the same frame, so the source sits where it will be drawn
        assert np.array_equal(reference[~strip], crop[~strip])  # the source pixels, untouched
        assert reference[strip].std() < 10 < crop[strip].std()  # an even blur in the strip, noise in the edge fill: nothing to copy
        assert call["reference"].width % 16 == call["reference"].height % 16 == 0

    def test_a_paint_only_edits_reference_is_the_crop_and_it_asks_for_no_join_fix(self, tmp_path):
        source = _photo(tmp_path / "photo.png")
        backend = _backend()

        _run(tmp_path, source, backend, edit_paint=painted_box(320, 480, (150, 100, 260, 200)), prompt="a vase")

        call = backend.edit_region.call_args.kwargs
        assert np.array_equal(np.asarray(call["reference"]), np.asarray(call["image"]))
        assert call["fix_join"] is False and call["prompt"] == "a vase"


class TestTheExtendPrompt:
    def test_an_extend_asks_for_the_join_fix(self, tmp_path):
        backend = _backend()

        _run(tmp_path, _photo(tmp_path / "photo.png", 320, 240), backend, edit_extend=("left", 128), prompt="more of the garden")

        assert backend.edit_region.call_args.kwargs["fix_join"] is True

    def test_the_saved_file_records_the_users_own_prompt(self, tmp_path):
        from zvisiongenerator.utils.provenance import read_png_config

        _, artifacts, _ = _run(tmp_path, _photo(tmp_path / "photo.png", 320, 240), _backend(), edit_extend=("left", 128), prompt="more of the garden")

        recorded = read_png_config(artifacts.filepath)
        assert recorded["prompt"] == "more of the garden"


class TestProgressEvents:
    """The Web UI tells the drawing steps from the finishing up by the stage name each event carries."""

    @staticmethod
    def _events(tmp_path: Path, **overrides) -> list[dict]:
        def step(**kwargs):
            kwargs["step_callback"]({"current_step": 2, "total_steps": 4})

        backend = _backend()
        backend.edit_region.side_effect = lambda **kwargs: (step(**kwargs), noise_picture(777, 555))[1]
        backend.text_to_image.side_effect = lambda **kwargs: (step(**kwargs), noise_picture(kwargs["width"], kwargs["height"]))[1]
        events: list[dict] = []
        request = ImageGenerationRequest(
            backend=backend,
            model=object(),
            prompt="a red door",
            edit_source=str(_photo(tmp_path / "photo.png")),
            output_dir=str(tmp_path / "out"),
            filename_base="x",
            steps=4,
            max_reference_images=0,
            step_callback=events.append,
            **overrides,
        )
        run_workflow_with_progress(build_edit_workflow(), request, ImageWorkingArtifacts(filename="photo_edit"), progress_callback=events.append, event_context={})
        return events

    @pytest.mark.parametrize("paint", [painted_box(320, 480, (150, 100, 260, 200)), None], ids=["paint", "whole picture"])
    def test_the_steps_belong_to_the_edit_stage(self, tmp_path, paint):
        events = self._events(tmp_path, edit_paint=paint)

        started = next(event for event in events if event.get("type") == "workflow_started")
        stepping = [event for event in events if "current_step" in event]
        assert [event["workflow_stage_name"] for event in stepping] == ["edit"]
        assert {event["phase"] for event in stepping} == {"image_generate"}
        assert started["stage_names"] == ["resolve_prompt", "load_edit_source", "prepare_edit", "edit", "composite", "save_image"]
