"""Tests for the SeedVR2 upscale workflow: stage order, the engine call, progress, quit handling and the saved file."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.upscale_runner import read_upscale_source
from zvisiongenerator.utils.provenance import embed_png_config, read_png_config
from zvisiongenerator.utils.seedvr2 import MLX_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
from zvisiongenerator.web.upscale_api import plan_seedvr2_upscale
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


def _planned_request(tmp_path, source_path, factor=2):
    """Plan a SeedVR2 upscale of a real file the way the web UI does, then attach a mock engine."""
    tier = SeedVR2Tier("seedvr2-3b", weights_gb=4.0, activation_floor_gb=5.0, activation_base_gb=1.0, activation_gb_per_mp=0.5)
    setup = SeedVR2Setup(
        settings=SeedVR2Settings(model="auto", softness=0.5, video_softness=0.0, reference_megapixels=16, tiers=(tier,)), tier=tier, budget_bytes=None, downloaded=True, weights=MLX_WEIGHTS
    )
    request = plan_seedvr2_upscale(read_upscale_source(source_path), factor, setup=setup).request
    backend = MagicMock()
    backend.upscale_image.side_effect = lambda model, path, *, width, height, **_kw: Image.new("RGB", (width, height))
    return backend, ImageGenerationRequest(**{**request.__dict__, "backend": backend, "model": MagicMock()})


def test_planned_upscale_of_an_image_without_a_record_saves_null_sampling_values(tmp_path):
    source = _source(tmp_path)
    _backend, request = _planned_request(tmp_path, source)
    artifacts = ImageWorkingArtifacts()

    assert build_seedvr2_upscale_workflow().run(request, artifacts) is StageOutcome.success

    payload = read_png_config(artifacts.filepath)
    assert (payload["seed"], payload["steps"], payload["guidance"]) == (None, None, None)
    assert payload["generation"]["upscale"] == {"method": "seedvr2", "factor": 2, "model": "seedvr2-3b", "quantize": 8, "softness": 0.5, "seed": 0}


def test_planned_upscale_keeps_the_source_record_and_saves_its_rendered_prompt(tmp_path):
    source = tmp_path / "recorded.png"
    info = PngInfo()
    embed_png_config(info, {"prompt": "{red|blue} barn", "model": "zit", "seed": 11, "steps": 8, "guidance": 1.5, "scheduler": "beta", "generation": {"quantize": 4}})
    info.add_text("Description", "red barn")
    Image.new("RGB", (32, 16)).save(source, pnginfo=info)
    _backend, request = _planned_request(tmp_path, source)
    artifacts = ImageWorkingArtifacts()

    assert build_seedvr2_upscale_workflow().run(request, artifacts) is StageOutcome.success

    payload = read_png_config(artifacts.filepath)
    assert (payload["prompt"], payload["seed"], payload["steps"], payload["guidance"], payload["scheduler"]) == ("{red|blue} barn", 11, 8, 1.5, "beta")
    assert payload["generation"]["quantize"] == 4
    assert payload["generation"]["upscale"]["seed"] == 11
    with Image.open(artifacts.filepath) as saved:
        assert saved.info["Description"] == "red barn"


def test_upscaling_a_seedvr2_output_again_reads_it_as_a_source_without_sampling_values(tmp_path):
    first = _source(tmp_path)
    _backend, request = _planned_request(tmp_path, first)
    first_artifacts = ImageWorkingArtifacts()
    build_seedvr2_upscale_workflow().run(request, first_artifacts)

    _backend, again = _planned_request(tmp_path, first_artifacts.filepath)

    assert again.source_settings_known is False
    assert again.upscale_method == "seedvr2"
