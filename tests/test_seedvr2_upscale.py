"""Tests for the SeedVR2 upscale workflow: stage order, the engine call, progress, quit handling and the saved file."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.progress_events import _stage_label
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.provenance import read_png_config
from zvisiongenerator.workflows import build_seedvr2_upscale_workflow
from zvisiongenerator.workflows.seedvr2_stages import upscale_stage


def _source(tmp_path, size=(32, 16)):
    path = tmp_path / "portrait.png"
    Image.new("RGB", size, color="purple").save(path, pnginfo=PngInfo())
    return path


def _request(tmp_path, source, **overrides):
    backend = MagicMock()
    backend.upscale_image.side_effect = lambda model, path, *, width, height, **_kw: Image.new("RGB", (width, height), color="white")
    defaults = dict(
        backend=backend,
        model=MagicMock(name="seedvr2_model"),
        prompt="{a|b} barn",
        resolved_prompt="a barn",
        seed=7,
        upscale_factor=2,
        upscale_method="seedvr2",
        upscale_model="seedvr2-7b",
        upscale_quantize=8,
        upscale_softness=0.75,
        sharpen=False,
        output_dir=str(tmp_path),
        upscale_source=str(source),
    )
    defaults.update(overrides)
    return ImageGenerationRequest(**defaults)


def _artifacts(size=(32, 16)):
    return ImageWorkingArtifacts(image=Image.new("RGB", size))


def test_workflow_stages_are_prompt_load_upscale_save():
    stages = build_seedvr2_upscale_workflow().stages

    assert [stage.__name__ for stage in stages] == ["resolve_prompt_stage", "load_source_stage", "upscale_stage", "save_image_stage"]
    assert stages[2] is upscale_stage


def test_stage_label_stays_upscale():
    assert _stage_label(upscale_stage) == "upscale"


def test_stage_calls_the_engine_with_the_target_size(tmp_path):
    request = _request(tmp_path, _source(tmp_path))
    artifacts = _artifacts((33, 17))

    outcome = upscale_stage(request, artifacts)

    assert outcome is StageOutcome.success
    request.backend.upscale_image.assert_called_once_with(request.model, request.upscale_source, width=66, height=34, seed=7, softness=0.75)
    assert artifacts.image.size == (66, 34)
    assert artifacts.was_upscaled is True
    assert artifacts.generation_time > 0


def test_stage_without_softness_passes_zero(tmp_path):
    request = _request(tmp_path, _source(tmp_path), upscale_softness=None)

    upscale_stage(request, _artifacts())

    assert request.backend.upscale_image.call_args.kwargs["softness"] == 0.0


def test_stage_sends_one_complete_step_with_a_callback(tmp_path):
    events: list[dict] = []
    request = _request(tmp_path, _source(tmp_path), step_callback=events.append)

    upscale_stage(request, _artifacts())

    assert events == [{"phase": "image_upscale", "current_step": 1, "total_steps": 1}]


def test_stage_without_a_callback_sends_nothing(tmp_path):
    request = _request(tmp_path, _source(tmp_path))

    assert upscale_stage(request, _artifacts()) is StageOutcome.success


def test_stage_quit_after_the_call_is_skipped_and_keeps_the_source(tmp_path):
    events: list[dict] = []
    skip_signal = MagicMock()
    skip_signal.pending.return_value = "quit"
    request = _request(tmp_path, _source(tmp_path), skip_signal=skip_signal, step_callback=events.append)
    artifacts = _artifacts()
    source_image = artifacts.image

    outcome = upscale_stage(request, artifacts)

    assert outcome is StageOutcome.skipped
    request.backend.upscale_image.assert_called_once()
    assert artifacts.image is source_image
    assert artifacts.was_upscaled is False
    assert events == []


def test_stage_without_a_pending_quit_succeeds(tmp_path):
    skip_signal = MagicMock()
    skip_signal.pending.return_value = None
    request = _request(tmp_path, _source(tmp_path), skip_signal=skip_signal)

    assert upscale_stage(request, _artifacts()) is StageOutcome.success


@pytest.mark.parametrize("missing", ["upscale_source", "upscale_factor"])
def test_stage_requires_source_and_factor(tmp_path, missing):
    request = _request(tmp_path, _source(tmp_path), **{missing: None})

    with pytest.raises(ValueError, match=missing):
        upscale_stage(request, _artifacts())


def test_stage_requires_a_loaded_image(tmp_path):
    request = _request(tmp_path, _source(tmp_path))

    with pytest.raises(ValueError, match="loaded"):
        upscale_stage(request, ImageWorkingArtifacts())


def test_workflow_saves_the_rendered_prompt_and_seedvr2_record(tmp_path):
    source = _source(tmp_path)
    request = _request(tmp_path, source, prompt="{a|b} barn", resolved_prompt=None)
    artifacts = ImageWorkingArtifacts()

    outcome = build_seedvr2_upscale_workflow().run(request, artifacts)

    assert outcome is StageOutcome.success
    assert artifacts.filepath == str(tmp_path / "portrait_2x.png")
    with Image.open(artifacts.filepath) as saved:
        assert saved.size == (64, 32)
        # resolve_prompt_stage expands the {a|b} choice; the template itself is never saved as the description.
        assert saved.info["Description"] in {"a barn", "b barn"}
    payload = read_png_config(artifacts.filepath)
    assert payload["workflow"] == "upscale"
    assert payload["generation"]["upscale"] == {"method": "seedvr2", "factor": 2, "model": "seedvr2-7b", "quantize": 8, "softness": 0.75, "seed": 7}
    assert "sharpen" not in payload["generation"]


def test_workflow_planned_prompt_is_saved_as_is(tmp_path):
    request = _request(tmp_path, _source(tmp_path), resolved_prompt="a barn")
    artifacts = ImageWorkingArtifacts()

    build_seedvr2_upscale_workflow().run(request, artifacts)

    with Image.open(artifacts.filepath) as saved:
        assert saved.info["Description"] == "a barn"


def test_workflow_quit_after_the_call_saves_nothing(tmp_path):
    skip_signal = MagicMock()
    skip_signal.pending.return_value = "quit"
    request = _request(tmp_path, _source(tmp_path), skip_signal=skip_signal)
    artifacts = ImageWorkingArtifacts()

    outcome = build_seedvr2_upscale_workflow().run(request, artifacts)

    assert outcome is StageOutcome.skipped
    assert not (tmp_path / "portrait_2x.png").exists()
