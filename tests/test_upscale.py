"""Tests for upscaling an existing image: limits, source reading, stages, workflow and runner events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.upscale_runner import read_upscale_source, run_upscale
from zvisiongenerator.utils.provenance import embed_png_config, read_png_config
from zvisiongenerator.utils.upscale import default_upscale_denoise, existing_image_denoise, max_megapixels, upscale_options, upscale_output_size
from zvisiongenerator.workflows import build_upscale_workflow
from zvisiongenerator.workflows.image_stages import load_source_stage, save_image_stage


def _write_png(path, size=(64, 48), *, config=None, description=None):
    info = PngInfo()
    if description is not None:
        info.add_text("Description", description)
    if config is not None:
        embed_png_config(info, config)
    Image.new("RGB", size, color="purple").save(path, pnginfo=info)
    return path


# ── Sizes and limits ────────────────────────────────────────────────────────


def test_output_size_is_aligned_to_16():
    assert upscale_output_size(832, 1216, 2) == (1664, 2432)
    assert upscale_output_size(100, 50, 2) == (208, 112)


def test_options_list_each_factor_with_size():
    options = upscale_options(832, 1216, {}, {"upscale": {"max_megapixels": 20}})

    assert [(o.factor, o.width, o.height, o.allowed) for o in options] == [(2, 1664, 2432, True), (4, 3328, 4864, True)]


def test_option_over_megapixel_cap_is_disallowed():
    options = upscale_options(832, 1216, {}, {"upscale": {"max_megapixels": 16}})

    assert options[0].allowed is True
    assert options[1].allowed is False
    assert "16-megapixel" in options[1].reason


def test_option_over_model_dimension_max_is_disallowed():
    options = upscale_options(832, 1216, {"dimension_max": 2048}, {})

    assert [o.allowed for o in options] == [False, False]
    assert "2048" in options[0].reason


@pytest.mark.parametrize("capabilities", [{"supports_upscale": False}, {"supports_img2img": False}])
def test_models_that_cannot_refine_disallow_every_factor(capabilities):
    options = upscale_options(256, 256, capabilities, {})

    assert all(not o.allowed and o.reason for o in options)


def test_max_megapixels_defaults_to_20():
    assert max_megapixels({}) == 20.0


@pytest.mark.parametrize("value", [0, -1, "big", True])
def test_max_megapixels_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_megapixels"):
        max_megapixels({"upscale": {"max_megapixels": value}})


def test_default_denoise_reads_config_per_factor():
    config = {"upscale": {"default_denoise_2x": 0.25, "default_denoise_4x": 0.45}}

    assert default_upscale_denoise(config, 2) == 0.25
    assert default_upscale_denoise(config, 4) == 0.45


def test_existing_image_denoise_is_picked_by_output_size():
    # xs → 2×: about 1 MP, the small-output denoise.
    assert existing_image_denoise({}, 1344, 768) == 0.4
    # A 2× of that 2× and a direct 4× both land at about 4 MP and get the same light touch.
    assert existing_image_denoise({}, 2688, 1536) == 0.2


def test_existing_image_denoise_reads_config():
    config = {"upscale": {"existing_denoise_small": 0.35, "existing_denoise_large": 0.15, "existing_large_megapixels": 1}}

    assert existing_image_denoise(config, 1000, 1000) == 0.35
    assert existing_image_denoise(config, 1000, 1001) == 0.15


# ── Reading the source ──────────────────────────────────────────────────────


def test_read_source_carries_recorded_settings(tmp_path):
    config = {
        "schema": "zvisiongenerator.config.v1",
        "prompt": "a {red|blue} barn",
        "negative_prompt": "blurry",
        "model": "zit",
        "seed": 42,
        "steps": 9,
        "guidance": 1.5,
        "scheduler": "beta",
        "ratio": "4:3",
        "size": "s",
        "lora": "/loras/style.safetensors:0.8",
        "generation": {"quantize": 8},
    }
    path = _write_png(tmp_path / "barn.png", config=config, description="a red barn")

    source = read_upscale_source(path)

    settings = source.settings
    assert (source.width, source.height) == (64, 48)
    assert source.rendered_prompt == "a red barn"
    assert settings.prompt == "a {red|blue} barn"
    assert settings.negative_prompt == "blurry"
    assert (settings.model, settings.seed, settings.steps, settings.guidance, settings.scheduler) == ("zit", 42, 9, 1.5, "beta")
    assert settings.lora == "/loras/style.safetensors:0.8"
    assert settings.generation == {"quantize": 8}


def test_read_source_without_settings_has_none(tmp_path):
    path = _write_png(tmp_path / "imported.png")

    source = read_upscale_source(path)

    assert source.settings is None
    assert source.rendered_prompt is None


def test_read_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_upscale_source(tmp_path / "gone.png")


def test_read_source_rejects_non_image(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Cannot read"):
        read_upscale_source(path)


# ── Stages and workflow ─────────────────────────────────────────────────────


def _request(tmp_path, source, **overrides):
    defaults = dict(
        backend=MagicMock(),
        model=MagicMock(),
        prompt="a barn",
        model_name="zit",
        upscale_factor=2,
        upscale_denoise=0.4,
        upscale_steps=3,
        output_dir=str(tmp_path),
        upscale_source=str(source),
    )
    defaults.update(overrides)
    return ImageGenerationRequest(**defaults)


def test_load_source_keeps_native_size_and_names_output(tmp_path):
    source = _write_png(tmp_path / "portrait_42.png", size=(100, 60))
    artifacts = ImageWorkingArtifacts()

    outcome = load_source_stage(_request(tmp_path, source), artifacts)

    assert outcome is StageOutcome.success
    assert artifacts.image.size == (100, 60)
    assert artifacts.filepath == str(tmp_path / "portrait_42_2x.png")
    assert artifacts.metadata["upscale_source"] == {"path": str(source), "width": 100, "height": 60}


def test_load_source_adds_counter_when_output_exists(tmp_path):
    source = _write_png(tmp_path / "portrait_42.png")
    (tmp_path / "portrait_42_4x.png").write_bytes(b"")
    artifacts = ImageWorkingArtifacts()

    load_source_stage(_request(tmp_path, source, upscale_factor=4), artifacts)

    assert artifacts.filepath == str(tmp_path / "portrait_42_4x_2.png")


def test_upscaling_an_upscale_appends_another_factor(tmp_path):
    source = _write_png(tmp_path / "portrait_42_2x.png")
    artifacts = ImageWorkingArtifacts()

    load_source_stage(_request(tmp_path, source), artifacts)

    assert artifacts.filepath == str(tmp_path / "portrait_42_2x_2x.png")


def test_load_source_is_noop_without_source(tmp_path):
    artifacts = ImageWorkingArtifacts()

    assert load_source_stage(_request(tmp_path, "x", upscale_source=None), artifacts) is StageOutcome.success
    assert artifacts.image is None


def test_save_never_overwrites_a_file_taken_after_naming(tmp_path):
    request = ImageGenerationRequest(backend=None, model=None, prompt="p", output_dir=str(tmp_path))
    artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (8, 8)), filename="shot", filepath=str(tmp_path / "shot.png"))
    (tmp_path / "shot.png").write_bytes(b"existing")

    save_image_stage(request, artifacts)

    assert artifacts.filepath == str(tmp_path / "shot_2.png")
    assert (tmp_path / "shot.png").read_bytes() == b"existing"


def test_upscale_workflow_stages():
    names = [stage.__name__ for stage in build_upscale_workflow().stages]

    assert names == ["resolve_prompt_stage", "suppress_negative_stage", "load_source_stage", "upscale_stage", "sharpen_stage", "save_image_stage"]


def test_upscale_workflow_without_sharpen():
    names = [stage.__name__ for stage in build_upscale_workflow(sharpen=False).stages]

    assert "sharpen_stage" not in names


def test_upscale_workflow_saves_upscaled_image_with_provenance(tmp_path):
    source = _write_png(tmp_path / "portrait.png", size=(32, 16))
    backend = MagicMock()
    backend.image_to_image.side_effect = lambda **kwargs: kwargs["image"]
    request = _request(tmp_path, source, backend=backend, resolved_prompt="a red barn", sharpen=False)
    artifacts = ImageWorkingArtifacts()

    outcome = build_upscale_workflow(sharpen=False).run(request, artifacts)

    assert outcome is StageOutcome.success
    assert artifacts.filepath == str(tmp_path / "portrait_2x.png")
    with Image.open(artifacts.filepath) as saved:
        assert saved.size == (64, 32)
    payload = read_png_config(artifacts.filepath)
    assert payload["workflow"] == "upscale"
    assert payload["source"] == {"path": str(source), "width": 32, "height": 16, "workflow": "txt2img"}
    assert payload["generation"]["upscale"]["factor"] == 2
    assert backend.image_to_image.call_args.kwargs["prompt"] == "a red barn"


# ── Runner events ───────────────────────────────────────────────────────────


def _event_types(callback):
    return [call.args[0]["type"] for call in callback.call_args_list]


def test_run_upscale_emits_one_image_batch(tmp_path):
    output = tmp_path / "out_2x.png"

    def _stage(request, artifacts):
        artifacts.filepath = str(output)
        return StageOutcome.success

    callback = MagicMock()
    request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_source=str(tmp_path / "out.png"))

    outcome = run_upscale(MagicMock(), MagicMock(), request, GenerationWorkflow(name="t", stages=[_stage]), progress_callback=callback)

    assert outcome is StageOutcome.success
    types = _event_types(callback)
    assert types[:3] == ["batch_started", "prompt_started", "generation_started"]
    assert types[-1] == "batch_completed"
    finished = next(call.args[0] for call in callback.call_args_list if call.args[0]["type"] == "generation_finished")
    assert finished["status"] == "success"
    assert finished["output_path"] == str(output)


def test_run_upscale_reports_stop_as_cancelled():
    callback = MagicMock()
    request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_source="/x.png")
    workflow = GenerationWorkflow(name="t", stages=[lambda _r, _a: StageOutcome.skipped])

    outcome = run_upscale(MagicMock(), MagicMock(), request, workflow, progress_callback=callback)

    assert outcome is StageOutcome.skipped
    assert _event_types(callback)[-1] == "batch_cancelled"


def test_run_upscale_reports_failed_stage():
    callback = MagicMock()
    request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_source="/x.png")
    workflow = GenerationWorkflow(name="t", stages=[lambda _r, _a: StageOutcome.failed])

    run_upscale(MagicMock(), MagicMock(), request, workflow, progress_callback=callback)

    assert _event_types(callback)[-1] == "batch_failed"
