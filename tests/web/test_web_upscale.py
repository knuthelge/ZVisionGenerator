"""Tests for the Web UI upscale action: job planning, the /api/upscale contract, and gallery asset fields."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import json
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit


from tests._factories import make_extend_source, make_probe, packaged_metadata_copy, use_model_metadata

from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.upscale_runner import UpscaleSource
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.provenance import RecordedSettings, embed_png_config
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
from zvisiongenerator.utils.upscale import SEEDVR2_MEMORY_WARNING
from zvisiongenerator.utils.video_extend import ExtendInspection, ExtendProblem, extend_source_problems
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import upscale_api
from zvisiongenerator import video_file_upscale_runner
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.gallery import gallery_asset_to_json, list_gallery_assets
from zvisiongenerator.web.model_inventory import ImageInventoryEntry
from zvisiongenerator.web.web_runner import JobConflictError
from zvisiongenerator.workflows import build_seedvr2_upscale_workflow, build_upscale_workflow
from zvisiongenerator.workflows.image_stages import load_references_stage

_DEFAULTS = {
    "steps": 10,
    "guidance": 0.0,
    "scheduler": "beta",
    "upscale_steps": 3,
    "supports_negative_prompt": True,
    "supports_img2img": True,
    "supports_upscale": True,
    "supports_quantize": True,
    "dimension_max": None,
}


_LTX_REFS = {"ltx-4": "dgrauet/ltx-2.3-mlx-q4@a", "ltx-8": "dgrauet/ltx-2.3-mlx-q8@b", "ltx25-4": "MarceloFSO/ltx-2.5-mlx-q4-fast@c", "h3": "MiniMax/H3@d"}


@pytest.fixture(autouse=True)
def _ltx_aliases(monkeypatch, tmp_path):
    """Serve the four video aliases with test refs and ``ltx-4`` with a download size."""
    data = packaged_metadata_copy()
    for name, ref in _LTX_REFS.items():
        data["models"][name]["downloads"] = [{"ref": ref}]
    data["models"]["ltx-4"]["download_size"] = {"darwin": "26 GB"}
    use_model_metadata(monkeypatch, tmp_path, data)


def _web_config(output_dir, **overrides):
    app_config = {
        "upscale": {"default_denoise_2x": 0.4, "default_denoise_4x": 0.4, "max_megapixels": 16},
        "sharpening": {"normal": 1.0, "upscaled": 1.2, "pre_upscale": 0.8},
    }
    values = dict(
        app_config=app_config,
        output_dir=str(output_dir),
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
        image_model_options=("zit", "other"),
        video_model_options=("ltx-8",),
        quantize_options=(4, 8),
        seedvr2=None,
        image_inventory=(
            ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="zit"),
            ImageInventoryEntry(name="ideo", family="ideogram4", size=None, source="alias", resolved_path="ideo"),
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_model_resolution(monkeypatch, *, family="zimage", defaults=None):
    monkeypatch.setattr(upscale_api, "resolve_model_path", lambda name, **_: f"/models/{name}")
    monkeypatch.setattr(upscale_api, "detect_image_model", lambda _ref: ImageModelInfo(family=family, is_distilled=False, size=None))
    monkeypatch.setattr(upscale_api, "resolve_defaults", lambda _info, _config, _overrides, _backend: dict(defaults or _DEFAULTS))


_FILE_FIELDS = ("path", "width", "height", "rendered_prompt")


def _source(**overrides):
    """Build an UpscaleSource; file fields and RecordedSettings fields can both be overridden."""
    file = {"path": "/out/portrait.png", "width": 832, "height": 1216, "rendered_prompt": "a red barn"}
    file.update({key: overrides.pop(key) for key in _FILE_FIELDS if key in overrides})
    settings = dict(
        prompt="a barn",
        negative_prompt="blurry",
        model="zit",
        seed=42,
        steps=8,
        guidance=1.5,
        scheduler="beta",
        generation={"time": 10.0, "quantize": 8, "sharpen": 1.0},
    )
    settings.update(overrides)
    return UpscaleSource(**file, settings=RecordedSettings(**settings))


def _write_png(path, size=(832, 1216), config=None):
    info = PngInfo()
    if config is not None:
        embed_png_config(info, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path, pnginfo=info)
    return path


# ── Planning ────────────────────────────────────────────────────────────────


def test_plan_carries_recorded_settings(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(), 2, _web_config(tmp_path), backend_name="mflux")

    request = plan.request
    assert plan.model_ref == "/models/zit"
    assert plan.request.quantize == 8
    assert plan.notices == ()
    assert (request.model_name, request.seed, request.steps, request.guidance) == ("zit", 42, 8, 1.5)
    assert request.scheduler == "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
    assert request.scheduler_name == "beta"
    assert request.negative_prompt == "blurry"
    assert request.resolved_prompt == "a red barn"
    # 832×1216 → 1664×2432 is about 4 MP, so the large-output denoise applies.
    assert (request.upscale_factor, request.upscale_denoise, request.upscale_steps) == (2, 0.2, 3)
    assert request.upscale_source == "/out/portrait.png"
    assert request.output_dir == "/out"
    assert (plan.option.width, plan.option.height) == (1664, 2432)


def test_plan_of_an_edit_refines_without_its_prompt(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    request = upscale_api.plan_upscale(_source(workflow="edit", prompt="make the door red"), 2, _web_config(tmp_path), backend_name="mflux").request

    assert (request.prompt, request.resolved_prompt) == ("", None)


@pytest.mark.parametrize(
    ("generation", "expected"),
    [({"time": 1.0, "transparent": True}, True), ({"time": 1.0}, False), ({"time": 1.0, "transparent": False}, False)],
)
def test_plan_inherits_transparency_from_the_recorded_generation(monkeypatch, tmp_path, generation, expected):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(generation=generation), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.transparent is expected


@pytest.mark.parametrize(
    ("generation", "expected"),
    [({"time": 1.0, "transparent": True}, True), ({"time": 1.0}, False)],
)
def test_seedvr2_plan_inherits_transparency_from_the_recorded_generation(monkeypatch, generation, expected):
    _forbid_image_model_detection(monkeypatch)

    plan = upscale_api.plan_seedvr2_upscale(_source(generation=generation), 2, setup=_seedvr2_setup(), platform_key="darwin")

    assert plan.request.transparent is expected


def test_plan_without_settings_is_not_transparent(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    source = UpscaleSource(path="/out/imported.png", width=512, height=512)

    assert upscale_api.plan_upscale(source, 2, _web_config(tmp_path), backend_name="mflux").request.transparent is False
    assert upscale_api.plan_seedvr2_upscale(source, 2, setup=_seedvr2_setup(), platform_key="darwin").request.transparent is False


def test_plan_picks_denoise_by_output_size_not_the_in_run_default(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    web_config = _web_config(tmp_path)
    web_config.app_config["upscale"].update(existing_denoise_small=0.35, existing_denoise_large=0.15)

    small = upscale_api.plan_upscale(_source(width=512, height=512), 2, web_config, backend_name="mflux")  # 1 MP
    large = upscale_api.plan_upscale(_source(width=512, height=512), 4, web_config, backend_name="mflux")  # 4.2 MP
    large_2x = upscale_api.plan_upscale(_source(width=1024, height=1024), 2, web_config, backend_name="mflux")  # 4.2 MP

    assert small.request.upscale_denoise == 0.35
    assert large.request.upscale_denoise == 0.15
    assert large_2x.request.upscale_denoise == 0.15


@pytest.mark.parametrize(
    ("viewer_keys", "source", "expected"),
    [
        pytest.param(
            {}, {}, {"upscale_sharpen": False, "sharpen": True, "sharpen_amount_override": None, "sharpen_amount_upscaled": 1.2}, id="skips the pre-sharpen and sharpens at the upscaled amount"
        ),
        pytest.param(
            {"existing_upscaled": 1.3, "existing_pre_upscale": 0.6}, {}, {"upscale_sharpen": True, "sharpen_amount_pre_upscale": 0.6, "sharpen_amount_upscaled": 1.3}, id="uses the viewer sharpen keys"
        ),
        pytest.param({}, {"generation": {"time": 10.0, "sharpen": 0.5}}, {"sharpen_amount_override": None, "sharpen_amount_upscaled": 1.2}, id="ignores the source sharpen amount"),
        pytest.param({}, {"generation": {"time": 10.0}}, {"sharpen": False}, id="does not sharpen a source recorded without sharpening"),
        pytest.param({}, {"generation": {}}, {"sharpen": True}, id="sharpens files without a record"),
    ],
)
def test_plan_sharpening(monkeypatch, tmp_path, viewer_keys, source, expected):
    """The final pass always uses the configured viewer amount, so editing the config never reinterprets old files."""
    _patch_model_resolution(monkeypatch)
    web_config = _web_config(tmp_path)
    web_config.app_config["sharpening"].update(viewer_keys)

    request = upscale_api.plan_upscale(_source(**source), 2, web_config, backend_name="mflux").request

    assert {name: getattr(request, name) for name in expected} == expected


def test_plan_keeps_how_an_enhanced_source_was_enhanced(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    record = {"settings": {"style": "film"}, "model": "org/fast", "tier": "fast", "format": "text"}

    request = upscale_api.plan_upscale(_source(prompt_input="a barn", enhance=record), 2, _web_config(tmp_path), backend_name="mflux").request

    assert (request.prompt_input, request.enhance_record) == ("a barn", record)


def test_plan_falls_back_to_default_model_with_its_defaults(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(model="gone-model", lora="/l.safetensors:1"), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.model_name == "zit"
    assert plan.request.steps == _DEFAULTS["steps"]
    assert plan.request.lora_paths is None
    assert plan.request.quantize is None
    assert plan.notices == (upscale_api.NOTICE_MODEL_NOT_CONFIGURED,)


def test_plan_and_menu_use_the_same_model_without_a_configured_default(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    _write_png(tmp_path / "plain.png", size=(256, 256))
    web_config = _web_config(tmp_path, default_models=WebUiDefaultModels(image=None, video="ltx-8"))

    menu = _asset_json(tmp_path, "plain.png", web_config)["upscale"]
    plan = upscale_api.plan_upscale(UpscaleSource(path=str(tmp_path / "plain.png"), width=256, height=256), 2, web_config, backend_name="mflux")

    # Both fall back to the first configured model (zit), so the menu offers what the server accepts.
    assert plan.request.model_name == "zit"
    assert all(factor["allowed"] for factor in menu["factors"])


def test_plan_carries_the_source_reference_image(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(image_path="/out/ref.png", image_strength=0.6), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.image_path == "/out/ref.png"
    assert plan.request.image_strength == 0.6


def test_plan_carries_the_source_references_for_reuse_only(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(reference_images=("/out/a.png", "/out/b.png")), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.reference_paths == ("/out/a.png", "/out/b.png")
    assert load_references_stage not in build_upscale_workflow().stages


def test_plan_without_settings_uses_default_model_and_notes_it(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    source = UpscaleSource(path="/out/imported.png", width=512, height=512)

    plan = upscale_api.plan_upscale(source, 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.model_name == "zit"
    assert plan.request.prompt == ""
    assert plan.request.seed == 0
    assert plan.notices == (upscale_api.NOTICE_SETTINGS_UNKNOWN,)


def test_plan_drops_missing_local_loras(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    present = tmp_path / "style.safetensors"
    present.write_bytes(b"")
    source = _source(lora=f"{present}:0.8,{tmp_path / 'gone.safetensors'}:0.5,owner/repo:1")

    plan = upscale_api.plan_upscale(source, 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.lora_paths == [str(present), "owner/repo"]
    assert plan.request.lora_weights == [0.8, 1.0]
    assert upscale_api.NOTICE_LORA_MISSING in plan.notices


def test_plan_rejects_output_over_the_cap(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match="megapixel"):
        upscale_api.plan_upscale(_source(), 4, _web_config(tmp_path), backend_name="mflux")


def test_plan_rejects_model_without_upscale(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch, defaults={**_DEFAULTS, "supports_upscale": False})

    with pytest.raises(ValueError, match="does not support upscaling"):
        upscale_api.plan_upscale(_source(), 2, _web_config(tmp_path), backend_name="mflux")


def test_plan_rejects_unknown_factor(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="factor"):
        upscale_api.plan_upscale(_source(), 3, _web_config(tmp_path), backend_name="mflux")


# ── SeedVR2 planning ────────────────────────────────────────────────────────

_SEEDVR2_TIER = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)


def _seedvr2_setup(*, budget_bytes=None, tier=_SEEDVR2_TIER):
    settings = SeedVR2Settings(model="auto", softness=0.75, video_softness=0.0, reference_megapixels=16, tiers=(tier,))
    return SeedVR2Setup(settings=settings, tier=tier, budget_bytes=budget_bytes, downloaded=False, weights=MLX_WEIGHTS)


def _forbid_image_model_detection(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("SeedVR2 planning must not touch the image model")

    monkeypatch.setattr(upscale_api, "detect_image_model", _fail)
    monkeypatch.setattr(upscale_api, "resolve_defaults", _fail)


def test_seedvr2_plan_carries_recorded_settings_without_an_image_model(monkeypatch):
    _forbid_image_model_detection(monkeypatch)
    source = _source(model_family="zimage", ratio="4:3", size="m", image_path="/out/ref.png", image_strength=0.6, lora="/gone.safetensors:0.8,style:1")

    plan = upscale_api.plan_seedvr2_upscale(source, 2, setup=_seedvr2_setup(), platform_key="darwin")

    request = plan.request
    assert plan.model_ref == "numz/SeedVR2_comfyUI"
    assert (plan.option.factor, plan.option.width, plan.option.height) == (2, 1664, 2432)
    assert plan.notices == ()
    assert (request.upscale_method, request.upscale_model, request.upscale_quantize, request.upscale_softness, request.upscale_factor) == ("seedvr2", "seedvr2-7b", 8, 0.75, 2)
    assert (request.prompt, request.resolved_prompt, request.negative_prompt, request.supports_negative_prompt) == ("a barn", "a red barn", "blurry", True)
    assert (request.model_name, request.model_family, request.quantize) == ("zit", "zimage", 8)
    assert (request.seed, request.steps, request.guidance, request.source_settings_known) == (42, 8, 1.5, True)
    assert (request.scheduler, request.scheduler_name) == (None, "beta")
    # Missing LoRA files stay in the record.
    assert (request.lora_paths, request.lora_weights) == (["/gone.safetensors", "style"], [0.8, 1.0])
    assert (request.ratio, request.size, request.image_path, request.image_strength) == ("4:3", "m", "/out/ref.png", 0.6)
    assert (request.width, request.height, request.output_dir, request.upscale_source, request.sharpen) == (832, 1216, "/out", "/out/portrait.png", False)


def test_seedvr2_plan_without_settings_saves_no_invented_values(monkeypatch):
    _forbid_image_model_detection(monkeypatch)
    source = UpscaleSource(path="/out/imported.png", width=512, height=512)

    plan = upscale_api.plan_seedvr2_upscale(source, 4, setup=_seedvr2_setup(), platform_key="darwin")

    request = plan.request
    assert plan.notices == (upscale_api.NOTICE_SETTINGS_UNKNOWN,)
    assert (request.prompt, request.resolved_prompt, request.model_name, request.model_family) == ("", None, None, "unknown")
    assert (request.seed, request.source_settings_known) == (0, False)
    assert (request.negative_prompt, request.supports_negative_prompt, request.scheduler_name) == (None, False, None)
    assert (request.lora_paths, request.lora_weights, request.quantize) == (None, None, None)


def test_seedvr2_plan_with_a_partial_record_marks_settings_unknown():
    source = _source(seed=None)

    request = upscale_api.plan_seedvr2_upscale(source, 2, setup=_seedvr2_setup(), platform_key="darwin").request

    assert request.source_settings_known is False
    assert (request.seed, request.steps, request.guidance) == (ImageGenerationRequest.seed, ImageGenerationRequest.steps, ImageGenerationRequest.guidance)
    assert request.prompt == "a barn"


def test_seedvr2_plan_prompt_falls_back_to_the_recorded_template():
    request = upscale_api.plan_seedvr2_upscale(_source(rendered_prompt=None, prompt="a {red|blue} barn"), 2, setup=_seedvr2_setup(), platform_key="darwin").request

    assert request.resolved_prompt == "a {red|blue} barn"


def test_seedvr2_plan_ignores_an_unparseable_lora_record():
    request = upscale_api.plan_seedvr2_upscale(_source(lora="a,,b"), 2, setup=_seedvr2_setup(), platform_key="darwin").request

    assert (request.lora_paths, request.lora_weights) == (None, None)


def test_seedvr2_plan_warns_instead_of_refusing_a_large_output():
    plan = upscale_api.plan_seedvr2_upscale(_source(), 4, setup=_seedvr2_setup(budget_bytes=int(20e9)), platform_key="darwin")

    assert plan.option.allowed is True
    assert plan.option.warning == SEEDVR2_MEMORY_WARNING
    assert upscale_api.plan_seedvr2_upscale(_source(), 2, setup=_seedvr2_setup(), platform_key="darwin").option.warning is None


def test_seedvr2_plan_rejects_unknown_factor():
    with pytest.raises(ValueError, match="factor"):
        upscale_api.plan_seedvr2_upscale(_source(), 3, setup=_seedvr2_setup(), platform_key="darwin")


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {"factor": 2},
        {"asset_id": "a.png"},
        {"asset_id": "a.png", "factor": 3},
        {"asset_id": "a.png", "factor": True},
        {"asset_id": "a.png", "factor": 2, "method": "bogus"},
        {"asset_id": "a.png", "factor": 2, "method": None},
    ],
)
def test_request_body_validation(body):
    with pytest.raises(ValueError):
        upscale_api.upscale_json_request(body)


@pytest.mark.parametrize(("extra", "method"), [({}, "creative"), ({"method": "creative"}, "creative"), ({"method": "seedvr2"}, "seedvr2")])
def test_request_body_method(extra, method):
    assert upscale_api.upscale_json_request({"asset_id": "a.png", "factor": 4, **extra}) == ("a.png", 4, method)


# ── Endpoint ────────────────────────────────────────────────────────────────


@pytest.fixture
def upscale_client(monkeypatch, tmp_path):
    web_config = _web_config(tmp_path)
    submitted: list[dict] = []
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())
    monkeypatch.setattr(web_server, "get_backend_name", lambda: "mflux")
    _patch_model_resolution(monkeypatch)

    def _submit(**kwargs):
        submitted.append(kwargs)
        return "job-up"

    monkeypatch.setattr(web_server.web_runner, "submit_upscale_job", _submit)
    with TestClient(web_server.app) as client:
        yield SimpleNamespace(client=client, submitted=submitted, root=tmp_path, monkeypatch=monkeypatch, web_config=web_config)


def test_upscale_route_queues_job(upscale_client):
    _write_png(upscale_client.root / "set" / "portrait.png", config={"prompt": "a barn", "model": "zit", "seed": 3})

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "set/portrait.png", "factor": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-up"
    assert payload["workflow"] == "upscale"
    assert payload["job_type"] == "Upscale"
    assert payload["supported_controls"] == ["quit"]
    assert payload["runs"] == 1
    assert payload["events_url"] == "/jobs/job-up/events"
    assert payload["meta"].startswith("2× creative → 1664×2432")
    submitted = upscale_client.submitted[0]
    assert submitted["request"].upscale_source == str((upscale_client.root / "set" / "portrait.png").resolve())
    assert [stage.__name__ for stage in submitted["workflow"].stages][2] == "load_source_stage"
    assert submitted["config"] is upscale_client.web_config.app_config
    assert submitted["context"]["workflow"] == "upscale"
    assert submitted["context"]["settings"] == {}
    assert submitted["context"]["meta"].startswith("2× creative → 1664×2432")
    assert submitted["context"]["save_project"] == "set"
    assert payload["queue_position"] is None


def test_upscale_route_names_no_project_for_a_source_outside_every_project(upscale_client):
    _write_png(upscale_client.root / "portrait.png", config={"prompt": "a barn", "model": "zit", "seed": 3})

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "portrait.png", "factor": 2})

    assert response.status_code == 200
    assert upscale_client.submitted[0]["context"]["save_project"] is None
    assert response.json()["save_project"] is None


def _busy_runner(client):
    def _busy(**_kwargs):
        raise JobConflictError("busy")

    client.monkeypatch.setattr(web_server.web_runner, "submit_upscale_job", _busy)


@pytest.mark.parametrize(
    ("body", "seedvr2", "defaults", "busy", "status", "detail"),
    [
        pytest.param({"asset_id": "missing.png", "factor": 2}, False, None, False, 404, None, id="unknown asset"),
        pytest.param({"asset_id": "../outside.png", "factor": 2}, False, None, False, 422, None, id="path outside the output folder"),
        pytest.param({"asset_id": "a.png", "factor": 3}, False, None, False, 422, None, id="bad factor"),
        pytest.param({"asset_id": "a.png", "factor": 2}, False, {**_DEFAULTS, "supports_img2img": False}, False, 422, None, id="model without img2img"),
        pytest.param({"asset_id": "a.png", "factor": 4}, False, None, False, 422, "megapixel", id="output over the cap"),
        pytest.param({"asset_id": "a.png", "factor": 2}, False, None, True, 409, None, id="busy"),
        pytest.param({"asset_id": "a.png", "factor": 2, "method": "bogus"}, False, None, False, 422, None, id="unknown method"),
        pytest.param({"asset_id": "a.png", "factor": 2, "method": "seedvr2"}, False, None, False, 422, None, id="seedvr2 where it is unsupported"),
        pytest.param({"asset_id": "missing.png", "factor": 2, "method": "seedvr2"}, True, None, False, 404, None, id="seedvr2 unknown asset"),
        pytest.param({"asset_id": "a.png", "factor": 2, "method": "seedvr2"}, True, None, True, 409, None, id="seedvr2 busy"),
    ],
)
def test_upscale_route_refusals(upscale_client, body, seedvr2, defaults, busy, status, detail):
    if seedvr2:
        upscale_client.web_config.seedvr2 = _seedvr2_setup()
    if defaults:
        _patch_model_resolution(upscale_client.monkeypatch, defaults=defaults)
    if busy:
        _busy_runner(upscale_client)
    _write_png(upscale_client.root / "a.png")

    response = upscale_client.client.post("/api/upscale", json=body)

    assert response.status_code == status
    assert detail is None or detail in response.json()["detail"]
    assert upscale_client.submitted == []


def test_upscale_route_defaults_to_the_creative_plan_even_with_seedvr2_on(upscale_client):
    upscale_client.web_config.seedvr2 = _seedvr2_setup()
    _write_png(upscale_client.root / "a.png", config={"prompt": "a barn", "model": "zit"})

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 2})

    assert response.status_code == 200
    submitted = upscale_client.submitted[0]
    assert submitted["request"].upscale_method == "creative"
    assert submitted["model_ref"] == "/models/zit"
    assert submitted["context"]["model"] == "zit"


def test_upscale_route_queues_a_seedvr2_job(upscale_client):
    upscale_client.web_config.seedvr2 = _seedvr2_setup()
    _write_png(upscale_client.root / "set" / "portrait.png", config={"prompt": "a barn", "model": "zit", "seed": 3})

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "set/portrait.png", "factor": 2, "method": "seedvr2"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-up"
    assert payload["workflow"] == "upscale"
    assert payload["model"] == "SeedVR2 7B q8"
    assert payload["meta"] == "2× → 1664×2432"
    submitted = upscale_client.submitted[0]
    assert submitted["request"].upscale_method == "seedvr2"
    assert submitted["request"].upscale_model == "seedvr2-7b"
    assert submitted["workflow"].stages == build_seedvr2_upscale_workflow().stages
    assert submitted["context"]["model"] == "SeedVR2 7B q8"
    assert submitted["context"]["meta"] == "2× → 1664×2432"
    assert submitted["context"]["settings"] == {}
    assert submitted["context"]["save_project"] == "set"


@pytest.mark.parametrize("method", ["creative", "seedvr2"])
def test_upscale_route_cuts_out_the_result_of_a_transparent_source(upscale_client, method):
    upscale_client.web_config.seedvr2 = _seedvr2_setup()
    _write_png(upscale_client.root / "cut.png", config={"prompt": "a barn", "model": "zit", "seed": 3, "generation": {"transparent": True}})
    _write_png(upscale_client.root / "plain.png", config={"prompt": "a barn", "model": "zit", "seed": 3})

    for name in ("cut.png", "plain.png"):
        response = upscale_client.client.post("/api/upscale", json={"asset_id": name, "factor": 2, "method": method})
        assert response.status_code == 200

    transparent, opaque = upscale_client.submitted
    assert transparent["request"].transparent is True
    assert [stage.__name__ for stage in transparent["workflow"].stages][-2] == "matte_stage"
    assert opaque["request"].transparent is False
    assert "matte_stage" not in [stage.__name__ for stage in opaque["workflow"].stages]


def test_upscale_route_queues_a_seedvr2_job_for_a_model_that_cannot_upscale(upscale_client):
    upscale_client.web_config.seedvr2 = _seedvr2_setup()
    _write_png(upscale_client.root / "plain.png")
    _patch_model_resolution(upscale_client.monkeypatch, defaults={**_DEFAULTS, "supports_upscale": False})

    creative = upscale_client.client.post("/api/upscale", json={"asset_id": "plain.png", "factor": 2})
    plain = upscale_client.client.post("/api/upscale", json={"asset_id": "plain.png", "factor": 2, "method": "seedvr2"})

    assert (creative.status_code, plain.status_code) == (422, 200)


def test_upscale_route_queues_a_warned_seedvr2_size_that_creative_refuses(upscale_client):
    upscale_client.web_config.seedvr2 = _seedvr2_setup(budget_bytes=int(20e9))
    _write_png(upscale_client.root / "a.png")

    refused = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 4})
    queued = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 4, "method": "seedvr2"})

    assert refused.status_code == 422
    assert queued.status_code == 200
    assert upscale_client.submitted[0]["request"].upscale_factor == 4
    assert queued.json()["meta"] == "4× → 3328×4864"


# ── Gallery asset fields ────────────────────────────────────────────────────


def _asset_json(tmp_path, name, web_config=None):
    web_config = web_config or _web_config(tmp_path, image_model_options=("zit", "ideo"))
    asset = next(item for item in list_gallery_assets(str(tmp_path)) if item.name == name)
    return gallery_asset_to_json(asset, web_config)


def test_gallery_upscale_field_lists_factors(tmp_path):
    _write_png(tmp_path / "a.png", config={"prompt": "p", "model": "zit"})

    upscale = _asset_json(tmp_path, "a.png")["upscale"]

    assert upscale["factors"] == [
        {"factor": 2, "width": 1664, "height": 2432, "allowed": True, "reason": None, "warning": None, "warning_code": None},
        {"factor": 4, "width": 3328, "height": 4864, "allowed": False, "reason": upscale["factors"][1]["reason"], "warning": None, "warning_code": None},
    ]
    assert "megapixel" in upscale["factors"][1]["reason"]


def test_gallery_upscale_field_has_no_seedvr2_where_it_is_unsupported(tmp_path):
    _write_png(tmp_path / "a.png", config={"prompt": "p", "model": "zit"})

    assert _asset_json(tmp_path, "a.png")["upscale"]["seedvr2"] is None


def test_gallery_upscale_field_lists_seedvr2_options(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")  # the size label is keyed by platform
    _write_png(tmp_path / "a.png", size=(1024, 1536), config={"prompt": "p", "model": "zit"})
    web_config = _web_config(tmp_path, seedvr2=_seedvr2_setup())

    seedvr2 = _asset_json(tmp_path, "a.png", web_config)["upscale"]["seedvr2"]

    assert seedvr2["options"] == [
        {"factor": 2, "width": 2048, "height": 3072, "allowed": True, "reason": None, "warning": None, "warning_code": None},
        {"factor": 4, "width": 4096, "height": 6144, "allowed": True, "reason": None, "warning": None, "warning_code": None},
    ]
    assert seedvr2["model"] == "SeedVR2 7B q8"
    assert seedvr2["downloaded"] is False
    assert seedvr2["download_size_label"] == "17.0 GB"


def test_gallery_upscale_field_seedvr2_warns_only_over_the_budget(tmp_path):
    _write_png(tmp_path / "a.png", size=(1024, 1536))
    web_config = _web_config(tmp_path, seedvr2=_seedvr2_setup(budget_bytes=int(21e9)))

    options = _asset_json(tmp_path, "a.png", web_config)["upscale"]["seedvr2"]["options"]

    assert [(option["factor"], option["allowed"], option["warning"]) for option in options] == [(2, True, None), (4, True, SEEDVR2_MEMORY_WARNING)]


def test_gallery_upscale_field_seedvr2_download_label_follows_the_stored_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")  # the size label is keyed by platform
    _write_png(tmp_path / "a.png")
    downloaded = replace(_seedvr2_setup(), downloaded=True)
    three_b = SeedVR2Tier("seedvr2-3b", weights_gb=4.0, activation_floor_gb=5.0, activation_base_gb=1.0, activation_gb_per_mp=0.5)

    ready = _asset_json(tmp_path, "a.png", _web_config(tmp_path, seedvr2=downloaded))["upscale"]["seedvr2"]
    small = _asset_json(tmp_path, "a.png", _web_config(tmp_path, seedvr2=_seedvr2_setup(tier=three_b)))["upscale"]["seedvr2"]

    assert (ready["downloaded"], ready["download_size_label"]) == (True, None)
    assert (small["model"], small["download_size_label"]) == ("SeedVR2 3B q8", "7.3 GB")


def test_gallery_upscale_field_seedvr2_does_not_depend_on_the_image_model(tmp_path):
    _write_png(tmp_path / "ideo.png", size=(256, 256), config={"prompt": "p", "model": "ideo"})
    _write_png(tmp_path / "plain.png", size=(256, 256))
    web_config = _web_config(tmp_path, image_model_options=("zit", "ideo"), seedvr2=_seedvr2_setup())
    no_models = _web_config(tmp_path, image_model_options=(), default_models=WebUiDefaultModels(), seedvr2=_seedvr2_setup())

    ideo = _asset_json(tmp_path, "ideo.png", web_config)["upscale"]
    plain = _asset_json(tmp_path, "plain.png", no_models)["upscale"]

    for upscale in (ideo, plain):
        assert [factor["allowed"] for factor in upscale["factors"]] == [False, False]
        assert all(option["allowed"] for option in upscale["seedvr2"]["options"])


def test_gallery_upscale_field_disallows_models_without_upscale(tmp_path):
    _write_png(tmp_path / "a.png", size=(256, 256), config={"prompt": "p", "model": "ideo"})

    factors = _asset_json(tmp_path, "a.png")["upscale"]["factors"]

    assert [f["allowed"] for f in factors] == [False, False]


def test_gallery_upscale_field_uses_default_model_without_metadata(tmp_path):
    _write_png(tmp_path / "plain.png", size=(256, 256))

    upscale = _asset_json(tmp_path, "plain.png")["upscale"]

    assert all(f["allowed"] for f in upscale["factors"])


def test_gallery_details_expose_recorded_settings(tmp_path):
    config = {
        "prompt": "p",
        "model": "zit",
        "model_family": "zimage",
        "negative_prompt": "blurry",
        "scheduler": "beta",
        "generation": {"time": 12.3, "quantize": 8, "upscale": {"factor": 2, "denoise": 0.4, "steps": 3, "pre_sharpen": 0.6}, "sharpen": 1.2, "bogus": "x"},
    }
    _write_png(tmp_path / "a.png", config=config)

    details = _asset_json(tmp_path, "a.png")["details"]

    assert details["negative_prompt"] == "blurry"
    assert details["scheduler"] == "beta"
    assert details["model_family"] == "zimage"
    # A file that predates the method reads as creative.
    assert details["generation"] == {"time": 12.3, "quantize": 8, "upscale": {"method": "creative", "factor": 2, "denoise": 0.4, "steps": 3, "pre_sharpen": 0.6}, "sharpen": 1.2}
    assert details["source"] is None


def test_gallery_reuse_carries_negative_prompt_and_scheduler(tmp_path):
    _write_png(tmp_path / "a.png", config={"workflow": "txt2img", "prompt": "p", "model": "zit", "negative_prompt": "blurry", "scheduler": "beta"})

    url = _asset_json(tmp_path, "a.png")["reuse_workspace_url"]

    assert "negative_prompt=blurry" in url
    assert "scheduler=beta" in url


def test_gallery_upscaled_image_links_source_and_reuses_source_size(tmp_path):
    source = _write_png(tmp_path / "portrait.png", size=(64, 32))
    config = {
        "workflow": "upscale",
        "prompt": "p",
        "model": "zit",
        "width": 128,
        "height": 64,
        "source": {"path": str(source), "width": 64, "height": 32},
    }
    _write_png(tmp_path / "portrait_2x.png", size=(128, 64), config=config)

    payload = _asset_json(tmp_path, "portrait_2x.png")

    assert payload["workflow"] == "txt2img"
    assert payload["details"]["recorded_workflow"] == "upscale"
    assert payload["details"]["source"] == {"path": str(source), "width": 64, "height": 32, "workflow": None, "id": "portrait.png"}
    assert "width=64" in payload["reuse_workspace_url"]
    assert "height=32" in payload["reuse_workspace_url"]


def test_an_upscale_whose_config_records_no_prompt_shows_and_reuses_no_prompt(tmp_path):
    source = _write_png(tmp_path / "portrait.png", size=(64, 32))
    config = {"workflow": "upscale", "model": "zit", "width": 128, "height": 64, "source": {"path": str(source), "width": 64, "height": 32}}
    _write_png(tmp_path / "portrait_2x.png", size=(128, 64), config=config)

    payload = _asset_json(tmp_path, "portrait_2x.png")

    assert payload["prompt"] == ""
    assert "prompt" not in parse_qs(urlsplit(payload["reuse_workspace_url"]).query)


def test_an_image_with_no_recorded_settings_shows_its_file_name_as_display_text(tmp_path):
    _write_png(tmp_path / "wet_street.png")

    payload = _asset_json(tmp_path, "wet_street.png")

    assert payload["prompt"] == "wet street"


def test_gallery_reuses_an_upscaled_img2img_image_as_img2img_with_its_reference(tmp_path):
    source = _write_png(tmp_path / "portrait.png", size=(64, 32))
    config = {
        "workflow": "upscale",
        "prompt": "p",
        "model": "zit",
        "image_path": "/out/ref.png",
        "image_strength": 0.6,
        "source": {"path": str(source), "width": 64, "height": 32, "workflow": "img2img"},
    }
    _write_png(tmp_path / "portrait_2x.png", size=(128, 64), config=config)

    payload = _asset_json(tmp_path, "portrait_2x.png")

    assert payload["workflow"] == "img2img"
    assert "workflow=img2img" in payload["reuse_workspace_url"]
    assert "image_path=%2Fout%2Fref.png" in payload["reuse_workspace_url"]


def test_gallery_source_outside_output_root_has_no_id(tmp_path):
    root = tmp_path / "out"
    _write_png(root / "u.png", config={"prompt": "p", "source": {"path": str(tmp_path / "elsewhere.png"), "width": 1, "height": 1}})

    source = _asset_json(root, "u.png", _web_config(root))["details"]["source"]

    assert source["id"] is None


def test_gallery_video_has_no_upscale_field_without_a_seedvr2_setup(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"")

    with patch("zvisiongenerator.web.gallery.ffprobe_json", side_effect=RuntimeError("ffprobe could not read the file")):
        assert _asset_json(tmp_path, "clip.mp4")["upscale"] is None


# ── Videos ──────────────────────────────────────────────────────────────────

_STREAMED_7B = SeedVR2Tier("seedvr2-7b", weights_gb=1.6, activation_floor_gb=5.0, activation_base_gb=0.21, activation_gb_per_mp=0.556, system_gb=22)
_CARD_10GB = 10240 * 1024 * 1024


def _video_setup(**overrides):
    setup = _seedvr2_setup(budget_bytes=_CARD_10GB, tier=_STREAMED_7B)
    return replace(setup, weights=FP8_WEIGHTS, **overrides)


def _ffprobe_data(config=None, *, size=None, rotation=None):
    """Return the JSON the gallery's one ffprobe call would give for a video with *config* embedded and a stream of *size*."""
    tags = {"zvisiongenerator.config": json.dumps(config)} if config is not None else {}
    stream = {"codec_type": "video", "codec_name": "h264", "width": size[0], "height": size[1], "r_frame_rate": "24/1", "avg_frame_rate": "24/1"} if size else None
    if stream and rotation is not None:
        stream["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": rotation}]
    return {"streams": [stream] if stream else [], "format": {"tags": tags}}


@pytest.fixture
def video_gallery(monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.video_file_upscale_method", lambda: "seedvr2")


def _video_asset_json(tmp_path, name, data, web_config=None):
    (tmp_path / name).write_bytes(b"\x00" * 8)
    with patch("zvisiongenerator.web.gallery.ffprobe_json", return_value=data):
        return _asset_json(tmp_path, name, web_config or _web_config(tmp_path, seedvr2=_video_setup()))


def test_gallery_video_offers_one_2x_option_at_the_recorded_size(tmp_path, video_gallery, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.web.gallery.sys.platform", "linux")
    config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "width": 768, "height": 448, "frame_count": 121}

    payload = _video_asset_json(tmp_path, "clip.mp4", _ffprobe_data(config, size=(1536, 896)))

    seedvr2 = payload["upscale"]["seedvr2"]
    assert payload["upscale"]["factors"] == []
    assert seedvr2["options"] == [{"factor": 2, "width": 1536, "height": 896, "allowed": True, "reason": None, "warning": None, "warning_code": None}]
    assert (seedvr2["model"], seedvr2["downloaded"], seedvr2["download_size_label"]) == ("SeedVR2 7B fp8", False, "9.0 GB")


def test_gallery_video_uses_the_displayed_size_of_a_rotated_foreign_clip(tmp_path, video_gallery):
    payload = _video_asset_json(tmp_path, "phone.mov", _ffprobe_data(size=(1920, 1080), rotation=-90))

    option = payload["upscale"]["seedvr2"]["options"][0]
    assert (option["width"], option["height"]) == (2160, 3840)
    assert (payload["width"], payload["height"]) == (1080, 1920)


@pytest.mark.parametrize(("size", "output", "code"), [((2560, 1440), (5120, 2880), "memory"), ((1920, 1080), (3840, 2160), "one_frame_batch"), ((768, 448), (1536, 896), None)])
def test_gallery_video_carries_a_warning_code_without_disabling_the_option(tmp_path, video_gallery, size, output, code):
    payload = _video_asset_json(tmp_path, "big.webm", _ffprobe_data(size=size))

    option = payload["upscale"]["seedvr2"]["options"][0]
    assert (option["width"], option["height"], option["allowed"], option["warning_code"]) == (*output, True, code)
    assert (option["warning"] is not None) == (code is not None)


@pytest.mark.parametrize("case", ["unsupported", "no_setup", "no_size"])
def test_gallery_video_has_no_upscale_menu_without_support_a_setup_or_a_size(tmp_path, monkeypatch, case):
    monkeypatch.setattr("zvisiongenerator.web.gallery.video_file_upscale_method", lambda: None if case == "unsupported" else "seedvr2")
    web_config = _web_config(tmp_path, seedvr2=None if case == "no_setup" else _video_setup())
    data = _ffprobe_data() if case == "no_size" else _ffprobe_data(size=(768, 448))

    assert _video_asset_json(tmp_path, "clip.mp4", data, web_config)["upscale"] is None


def test_gallery_runs_ffprobe_once_per_video_whatever_its_container(tmp_path, video_gallery):
    (tmp_path / "app.mp4").write_bytes(b"\x00")
    (tmp_path / "foreign.webm").write_bytes(b"\x00")

    with patch("zvisiongenerator.web.gallery.ffprobe_json", return_value=_ffprobe_data(size=(640, 360))) as probe:
        assets = list_gallery_assets(str(tmp_path))

    assert len(assets) == 2
    assert probe.call_count == 2
    assert all(call.kwargs == {"count_packets": False} for call in probe.call_args_list)


@pytest.fixture
def video_upscale_client(upscale_client, monkeypatch):
    submitted: list[dict] = []
    upscale_client.web_config.seedvr2 = _video_setup()
    source = make_extend_source(path=upscale_client.root / "clip.mp4", width=768, height=448, frame_count=121, scene=None, prompt=None)
    monkeypatch.setattr(web_server, "video_file_upscale_method", lambda: "seedvr2")
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server, "read_video_upscale_source", lambda path: (source, None))

    def _submit(**kwargs):
        submitted.append(kwargs)
        return "job-video"

    monkeypatch.setattr(web_server.web_runner, "submit_video_file_upscale_job", _submit)
    (upscale_client.root / "clip.mp4").write_bytes(b"\x00")
    return SimpleNamespace(
        client=upscale_client.client, root=upscale_client.root, monkeypatch=monkeypatch, submitted=submitted, image_submitted=upscale_client.submitted, web_config=upscale_client.web_config
    )


def _post_video(client, **body):
    return client.client.post("/api/upscale", json={"asset_id": "clip.mp4", "factor": 2, "method": "seedvr2", **body})


def test_video_upscale_route_queues_the_job(video_upscale_client):
    response = _post_video(video_upscale_client)

    assert response.status_code == 200
    payload = response.json()
    assert (payload["job_id"], payload["workflow"], payload["job_type"], payload["media_type"]) == ("job-video", "upscale", "Upscale", "video")
    assert (payload["meta"], payload["supported_controls"], payload["runs"]) == ("2× → 1536×896", ["quit"], 1)
    assert payload["notices"] == [upscale_api.NOTICE_SETTINGS_UNKNOWN]
    submitted = video_upscale_client.submitted[0]
    request = submitted["request"]
    assert (request.width, request.height, request.batch_size, request.upscale_model, request.softness) == (1536, 896, 29, "seedvr2-7b", 0.0)
    assert Path(request.output_dir) == video_upscale_client.root
    assert (submitted["context"]["media_type"], payload["upscale_method"]) == ("video", "seedvr2")
    assert submitted["context"]["settings"] == {}
    assert submitted["context"]["save_project"] is None
    assert video_upscale_client.image_submitted == []


def test_video_upscale_route_names_the_project_the_clip_is_in(video_upscale_client):
    clip = video_upscale_client.root / "Reel" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"\x00")
    source = make_extend_source(path=clip, width=768, height=448, frame_count=121, scene=None, prompt=None)
    video_upscale_client.monkeypatch.setattr(web_server, "read_video_upscale_source", lambda path: (source, None))

    response = _post_video(video_upscale_client, asset_id="Reel/clip.mp4")

    assert response.status_code == 200
    assert video_upscale_client.submitted[0]["context"]["save_project"] == "Reel"
    assert response.json()["save_project"] == "Reel"


def _video_unsupported(client):
    client.monkeypatch.setattr(web_server, "video_file_upscale_method", lambda: None)


def _video_without_setup(client):
    client.web_config.seedvr2 = None


def _video_without_ffmpeg(client):
    def _missing():
        raise RuntimeError("ffmpeg is not installed.")

    client.monkeypatch.setattr(web_server, "require_ffmpeg", _missing)


def _video_unreadable(client):
    problem = ExtendProblem("unreadable", "ffprobe could not read the file")
    client.monkeypatch.setattr(web_server, "read_video_upscale_source", lambda path: (_ for _ in ()).throw(ValueError("clip.mp4 can't be upscaled: " + problem.message)))


def _video_with_the_real_reader(client):
    client.monkeypatch.setattr(web_server, "read_video_upscale_source", video_file_upscale_runner.read_video_upscale_source)


def _video_busy(client):
    def _busy(**_kwargs):
        raise JobConflictError("busy")

    client.monkeypatch.setattr(web_server.web_runner, "submit_video_file_upscale_job", _busy)


@pytest.mark.parametrize(
    ("body", "setup", "status"),
    [
        pytest.param({"method": "creative"}, None, 422, id="creative method"),
        pytest.param({"method": None}, None, 422, id="no method"),
        pytest.param({"factor": 4}, None, 422, id="4x"),
        pytest.param({}, _video_unsupported, 422, id="unsupported platform"),
        pytest.param({}, _video_without_setup, 422, id="no seedvr2 setup"),
        pytest.param({}, _video_without_ffmpeg, 422, id="no ffmpeg"),
        pytest.param({}, _video_unreadable, 422, id="unreadable file"),
        pytest.param({"asset_id": "gone.mp4"}, _video_with_the_real_reader, 404, id="missing file"),
        pytest.param({"asset_id": "../clip.mp4"}, _video_with_the_real_reader, 422, id="outside the output folder"),
        pytest.param({}, _video_busy, 409, id="busy"),
    ],
)
def test_video_upscale_route_refusals(video_upscale_client, body, setup, status):
    if setup:
        setup(video_upscale_client)
    body = {key: value for key, value in {"asset_id": "clip.mp4", "factor": 2, "method": "seedvr2", **body}.items() if value is not None}

    response = video_upscale_client.client.post("/api/upscale", json=body)

    assert response.status_code == status
    assert video_upscale_client.submitted == []


@pytest.mark.parametrize(
    ("overrides", "audio_ok"),
    [({"color_trc": "smpte2084"}, True), ({"avg_frame_rate": "2400/101"}, True), ({"has_audio": True}, False)],
    ids=["hdr", "variable frame rate", "unreadable sound"],
)
def test_video_upscale_route_refuses_a_clip_the_upscale_would_get_wrong(video_upscale_client, monkeypatch, overrides, audio_ok):
    probe = make_probe(**{"frame_count": 72, "color_space": "bt709", "color_primaries": "bt709", "color_trc": "bt709", **overrides})
    problems = tuple(extend_source_problems(probe, None, audio_ok=audio_ok))
    monkeypatch.setattr(web_server, "read_video_upscale_source", video_file_upscale_runner.read_video_upscale_source)
    monkeypatch.setattr(video_file_upscale_runner, "inspect_extend_source", lambda path: ExtendInspection(probe, None, problems, None))

    response = _post_video(video_upscale_client)

    assert response.status_code == 422
    assert video_upscale_client.submitted == []


def test_video_upscale_plan_carries_the_clips_settings_and_softness():
    source = make_extend_source(width=1280, height=704, frame_count=60)
    setup = _video_setup()
    setup = replace(setup, settings=replace(setup.settings, video_softness=0.25))

    plan = upscale_api.plan_video_upscale(source, {"workflow": "txt2vid"}, setup=setup)

    assert (plan.request.width, plan.request.height) == (2560, 1408)
    assert (plan.request.batch_size, plan.request.softness, plan.request.recorded_config) == (9, 0.25, {"workflow": "txt2vid"})
    assert plan.notices == ()


# --- macOS: LTX upscale of a gallery video ---


def _ltx_web_config(tmp_path, *, video_model_options=("ltx-8", "ltx-4", "ltx25-4", "h3"), default_video="ltx-8", steps=2):
    web_config = _web_config(tmp_path, video_model_options=video_model_options, default_models=WebUiDefaultModels(image="zit", video=default_video))
    web_config.app_config = {
        **web_config.app_config,
        "video_model_presets": {"ltx": {"upscale": {"existing_video_steps": steps}, "versions": {"ltx25": {"aliases": ["ltx25-4"]}}}},
    }
    return web_config


def _ltx_source(tmp_path, **overrides):
    values = dict(path=tmp_path / "clip.mp4", width=768, height=448, frame_count=121, model="ltx-4", scene="A harbour at dawn.", prompt="A man walks his bicycle.")
    return make_extend_source(**{**values, **overrides})


def test_the_ltx_upscale_job_names_the_project_the_clip_is_in(tmp_path):
    source = _ltx_source(tmp_path, path=tmp_path / "Reel" / "clip.mp4")

    parsed = web_server._parse_ltx_video_upscale(source, {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

    assert parsed.context["save_project"] == "Reel"


class TestLtxVideoUpscalePlan:
    def test_the_clip_is_refined_with_its_own_model_prompt_and_seed_at_twice_the_size(self, tmp_path):
        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path), {"workflow": "txt2vid", "seed": 11}, _ltx_web_config(tmp_path))

        request = plan.request
        assert (request.width, request.height, request.model_name, request.model_ref) == (1536, 896, "ltx-4", "dgrauet/ltx-2.3-mlx-q4@a")
        assert (request.prompt, request.seed, request.steps) == ("A harbour at dawn. A man walks his bicycle.", 11, 2)
        assert (Path(request.output_dir), request.recorded_config) == (tmp_path, {"workflow": "txt2vid", "seed": 11})
        assert plan.notices == ()

    def test_the_configured_steps_are_used(self, tmp_path):
        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path), {"workflow": "txt2vid"}, _ltx_web_config(tmp_path, steps=1))

        assert plan.request.steps == 1

    def test_a_clip_without_recorded_settings_gets_the_default_model_an_empty_prompt_and_the_fixed_seed(self, tmp_path):
        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path, model=None, scene=None, prompt=None), None, _ltx_web_config(tmp_path))

        assert (plan.request.model_name, plan.request.prompt, plan.request.seed) == ("ltx-8", "", 42)
        assert plan.notices == (upscale_api.NOTICE_SETTINGS_UNKNOWN,)

    @pytest.mark.parametrize(
        ("recorded_model", "notice"), [("ltx25-4", upscale_api.NOTICE_MODEL_NOT_LTX23), ("h3", upscale_api.NOTICE_MODEL_NOT_LTX23), ("gone", upscale_api.NOTICE_MODEL_NOT_CONFIGURED)]
    )
    def test_a_clip_made_with_another_model_is_refined_with_the_default_ltx_2_3_model(self, tmp_path, recorded_model, notice):
        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path, model=recorded_model), {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.model_name == "ltx-8"
        assert plan.notices == (notice,)

    def test_an_extended_clip_is_refined_with_its_scene_alone(self, tmp_path):
        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path), {"workflow": "extend"}, _ltx_web_config(tmp_path))

        assert plan.request.prompt == "A harbour at dawn."

    def test_an_upscale_of_an_extended_clip_is_refined_with_its_scene_alone(self, tmp_path):
        recorded = {"workflow": "upscale", "source": {"workflow": "extend"}}

        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path), recorded, _ltx_web_config(tmp_path))

        assert plan.request.prompt == "A harbour at dawn."

    def test_without_an_ltx_2_3_default_the_first_ltx_2_3_model_is_used(self, tmp_path):
        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path, model=None), {"workflow": "txt2vid"}, _ltx_web_config(tmp_path, default_video="h3"))

        assert plan.request.model_name == "ltx-8"

    def test_without_any_ltx_2_3_model_it_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="LTX-2.3"):
            upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path), {"workflow": "txt2vid"}, _ltx_web_config(tmp_path, video_model_options=("ltx25-4", "h3"), default_video="h3"))

    def test_the_clip_keeps_its_loras_on_its_own_model_and_missing_ones_are_left_out(self, tmp_path):
        kept = tmp_path / "style.safetensors"
        kept.write_bytes(b"")
        source = _ltx_source(tmp_path, lora=f"{kept}:0.8,{tmp_path / 'gone.safetensors'}:1")

        plan = upscale_api.plan_ltx_video_upscale(source, {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.loras == ((str(kept), 0.8),)
        assert plan.notices == (upscale_api.NOTICE_LORA_MISSING,)

    def test_a_clip_of_another_model_leaves_its_loras_out_with_a_notice(self, tmp_path):
        kept = tmp_path / "style.safetensors"
        kept.write_bytes(b"")
        source = _ltx_source(tmp_path, model="ltx25-4", lora=f"{kept}:0.8")

        plan = upscale_api.plan_ltx_video_upscale(source, {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.loras == ()
        assert plan.notices == (upscale_api.NOTICE_MODEL_NOT_LTX23, upscale_api.NOTICE_LORA_OTHER_MODEL)

    def test_a_clip_of_an_unconfigured_ltx_2_3_model_keeps_its_loras(self, tmp_path):
        kept = tmp_path / "style.safetensors"
        kept.write_bytes(b"")
        source = _ltx_source(tmp_path, model="dgrauet/ltx-2.3-mlx-q4", lora=f"{kept}:0.8")

        plan = upscale_api.plan_ltx_video_upscale(source, {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.model_name == "ltx-8"
        assert plan.request.loras == ((str(kept), 0.8),)

    def test_a_clip_of_a_removed_ltx_alias_leaves_its_loras_out_since_its_version_is_unknown(self, tmp_path):
        kept = tmp_path / "style.safetensors"
        kept.write_bytes(b"")
        source = _ltx_source(tmp_path, model="ltx25-gone", lora=f"{kept}:0.8")

        plan = upscale_api.plan_ltx_video_upscale(source, {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.loras == ()
        assert plan.notices == (upscale_api.NOTICE_MODEL_NOT_CONFIGURED, upscale_api.NOTICE_LORA_OTHER_MODEL)

    def test_a_bare_lora_name_missing_from_the_lora_folder_is_left_out_whatever_the_working_folder_holds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "style").write_bytes(b"")
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / "data")

        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path, lora="style:0.8"), {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.loras == ()
        assert plan.notices == (upscale_api.NOTICE_LORA_MISSING,)

    def test_a_lora_folder_is_left_out(self, tmp_path):
        folder = tmp_path / "style"
        folder.mkdir()

        plan = upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path, lora=f"{folder}:0.8"), {"workflow": "txt2vid"}, _ltx_web_config(tmp_path))

        assert plan.request.loras == ()

    @pytest.mark.parametrize(("width", "frames"), [(770, 121), (768, 124)])
    def test_a_clip_off_the_ltx_grid_is_refused(self, tmp_path, width, frames):
        with pytest.raises(ValueError, match="can't be upscaled"):
            upscale_api.plan_ltx_video_upscale(_ltx_source(tmp_path, width=width, frame_count=frames), None, _ltx_web_config(tmp_path))


@pytest.fixture
def ltx_gallery(monkeypatch):
    state = SimpleNamespace(downloaded=True, fit=None)
    monkeypatch.setattr("zvisiongenerator.web.gallery.video_file_upscale_method", lambda: "ltx")
    monkeypatch.setattr("zvisiongenerator.web.gallery.video_model_downloaded", lambda *_args, **_kwargs: state.downloaded)
    monkeypatch.setattr("zvisiongenerator.web.gallery.accelerator_memory_budget_or_none", lambda: object())
    monkeypatch.setattr("zvisiongenerator.utils.model_memory.clip_memory_fit", lambda *_args, **_kwargs: state.fit)
    return state


_LTX_CLIP = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "model": "ltx-4", "width": 768, "height": 448, "frame_count": 121}


def test_gallery_video_offers_the_ltx_2x_option_on_macos(tmp_path, ltx_gallery):
    upscale = _video_asset_json(tmp_path, "clip.mp4", _ffprobe_data(_LTX_CLIP), _ltx_web_config(tmp_path))["upscale"]

    assert (upscale["factors"], upscale["seedvr2"]) == ([], None)
    assert upscale["ltx"] == {
        "options": [{"factor": 2, "width": 1536, "height": 896, "allowed": True, "reason": None, "warning": None, "warning_code": None}],
        "model": "LTX-2.3 4-bit",
        "downloaded": True,
        "download_size_label": None,
    }


def test_gallery_ltx_option_names_the_first_use_download_while_the_model_is_missing(tmp_path, ltx_gallery, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")  # the size label is keyed by platform
    ltx_gallery.downloaded = False

    ltx = _video_asset_json(tmp_path, "clip.mp4", _ffprobe_data(_LTX_CLIP), _ltx_web_config(tmp_path))["upscale"]["ltx"]

    assert (ltx["downloaded"], ltx["download_size_label"]) == (False, "26 GB")


@pytest.mark.parametrize("status", ["tight", "too_large"])
def test_gallery_ltx_option_is_warned_when_the_clip_may_not_fit(tmp_path, ltx_gallery, status):
    ltx_gallery.fit = {"status": status, "required_gb": 27.0, "budget_gb": 25.0}

    option = _video_asset_json(tmp_path, "clip.mp4", _ffprobe_data(_LTX_CLIP), _ltx_web_config(tmp_path))["upscale"]["ltx"]["options"][0]

    assert (option["warning"], option["warning_code"], option["allowed"]) == (SEEDVR2_MEMORY_WARNING, "memory", True)


@pytest.mark.parametrize(
    ("config", "web_config_overrides"),
    [
        ({**_LTX_CLIP, "frame_count": 124}, {}),
        ({**_LTX_CLIP, "width": 770}, {}),
        (_LTX_CLIP, {"video_model_options": ("ltx25-4", "h3"), "default_video": "h3"}),
    ],
    ids=["off-grid frames", "off-grid size", "no ltx-2.3 model"],
)
def test_gallery_video_has_no_ltx_option_when_it_cannot_run(tmp_path, ltx_gallery, config, web_config_overrides):
    upscale = _video_asset_json(tmp_path, "clip.mp4", _ffprobe_data(config), _ltx_web_config(tmp_path, **web_config_overrides))["upscale"]

    assert upscale is None


@pytest.fixture
def ltx_upscale_client(upscale_client, monkeypatch):
    submitted: list[dict] = []
    web_config = _ltx_web_config(upscale_client.root)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    source = _ltx_source(upscale_client.root)
    monkeypatch.setattr(web_server, "video_file_upscale_method", lambda: "ltx")
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server, "read_video_upscale_source", lambda path: (source, {"workflow": "txt2vid", "seed": 5}))

    def _submit(**kwargs):
        submitted.append(kwargs)
        return "job-ltx"

    monkeypatch.setattr(web_server.web_runner, "submit_ltx_video_upscale_job", _submit)
    monkeypatch.setattr(web_server.web_runner, "submit_video_file_upscale_job", lambda **_kwargs: pytest.fail("the SeedVR2 job must not be submitted on macOS"))
    (upscale_client.root / "clip.mp4").write_bytes(b"\x00")
    return SimpleNamespace(client=upscale_client.client, root=upscale_client.root, monkeypatch=monkeypatch, submitted=submitted)


def test_ltx_video_upscale_route_queues_the_ltx_job(ltx_upscale_client):
    response = ltx_upscale_client.client.post("/api/upscale", json={"asset_id": "clip.mp4", "factor": 2, "method": "ltx"})

    assert response.status_code == 200
    payload = response.json()
    assert (payload["job_id"], payload["workflow"], payload["media_type"], payload["meta"], payload["supported_controls"]) == ("job-ltx", "upscale", "video", "2× → 1536×896", ["quit"])
    assert (payload["model"], payload["model_label"], payload["upscale_method"]) == ("ltx-4", "LTX-2.3 4-bit", "ltx")
    request = ltx_upscale_client.submitted[0]["request"]
    assert (request.model_name, request.steps, request.seed) == ("ltx-4", 2, 5)


@pytest.mark.parametrize(
    "body",
    [{"method": "seedvr2"}, {"method": "creative"}, {"factor": 4}],
    ids=["seedvr2 on macos", "creative", "4x"],
)
def test_ltx_video_upscale_route_refusals(ltx_upscale_client, body):
    response = ltx_upscale_client.client.post("/api/upscale", json={"asset_id": "clip.mp4", "factor": 2, "method": "ltx", **body})

    assert response.status_code == 422
    assert ltx_upscale_client.submitted == []


def test_ltx_video_upscale_route_refuses_ltx_on_a_platform_that_upscales_with_seedvr2(video_upscale_client):
    response = _post_video(video_upscale_client, method="ltx")

    assert response.status_code == 422
    assert video_upscale_client.submitted == []


def test_the_ltx_method_is_refused_for_an_image(upscale_client):
    _write_png(upscale_client.root / "a.png", config={"prompt": "a barn", "model": "zit"})

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 2, "method": "ltx"})

    assert response.status_code == 422
    assert upscale_client.submitted == []
