"""Tests for the Web UI upscale action: job planning, the /api/upscale contract, and gallery asset fields."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.upscale_runner import UpscaleSource
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.provenance import RecordedSettings, embed_png_config
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import upscale_api
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.gallery import gallery_asset_to_json, list_gallery_assets
from zvisiongenerator.web.model_inventory import ImageInventoryEntry
from zvisiongenerator.web.web_runner import JobConflictError

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


def _web_config(output_dir, **overrides):
    app_config = {
        "upscale": {"default_denoise_2x": 0.4, "default_denoise_4x": 0.4, "max_megapixels": 16},
        "sharpening": {"normal": 1.0, "upscaled": 1.2, "pre_upscale": 0.8},
        "schedulers": {"beta": {"mflux_class": "pkg.BetaScheduler"}},
        "model_presets": {"zimage": {}, "ideogram4": {"supports_img2img": False, "supports_upscale": False, "dimension_max": 2048}},
    }
    values = dict(
        app_config=app_config,
        output_dir=str(output_dir),
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
        image_model_options=("zit", "other"),
        video_model_options=("ltx-8",),
        quantize_options=(4, 8),
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
    assert request.scheduler == "pkg.BetaScheduler"
    assert request.scheduler_name == "beta"
    assert request.negative_prompt == "blurry"
    assert request.resolved_prompt == "a red barn"
    # 832×1216 → 1664×2432 is about 4 MP, so the large-output denoise applies.
    assert (request.upscale_factor, request.upscale_denoise, request.upscale_steps) == (2, 0.2, 3)
    assert request.upscale_source == "/out/portrait.png"
    assert request.output_dir == "/out"
    assert (plan.option.width, plan.option.height) == (1664, 2432)


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


def test_plan_skips_pre_sharpen_and_sharpens_at_the_upscaled_amount(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.upscale_sharpen is False
    assert plan.request.sharpen is True
    assert plan.request.sharpen_amount_override is None
    assert plan.request.sharpen_amount_upscaled == 1.2


def test_plan_uses_the_viewer_sharpen_keys(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)
    web_config = _web_config(tmp_path)
    web_config.app_config["sharpening"].update(existing_upscaled=1.3, existing_pre_upscale=0.6)

    plan = upscale_api.plan_upscale(_source(), 2, web_config, backend_name="mflux")

    assert plan.request.upscale_sharpen is True
    assert plan.request.sharpen_amount_pre_upscale == 0.6
    assert plan.request.sharpen_amount_upscaled == 1.3


def test_plan_ignores_the_source_sharpen_amount(monkeypatch, tmp_path):
    # The final pass always uses the configured viewer amount, so editing the config never reinterprets old files.
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(generation={"time": 10.0, "sharpen": 0.5}), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.sharpen_amount_override is None
    assert plan.request.sharpen_amount_upscaled == 1.2


def test_plan_does_not_sharpen_a_source_recorded_without_sharpening(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(generation={"time": 10.0}), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.sharpen is False


def test_plan_sharpens_older_files_without_a_record(monkeypatch, tmp_path):
    _patch_model_resolution(monkeypatch)

    plan = upscale_api.plan_upscale(_source(generation={}), 2, _web_config(tmp_path), backend_name="mflux")

    assert plan.request.sharpen is True


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


@pytest.mark.parametrize("body", [None, [], {"factor": 2}, {"asset_id": "a.png"}, {"asset_id": "a.png", "factor": 3}, {"asset_id": "a.png", "factor": True}])
def test_request_body_validation(body):
    with pytest.raises(ValueError):
        upscale_api.upscale_json_request(body)


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
        yield SimpleNamespace(client=client, submitted=submitted, root=tmp_path, monkeypatch=monkeypatch)


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
    assert payload["meta"].startswith("2× → 1664×2432")
    submitted = upscale_client.submitted[0]
    assert submitted["request"].upscale_source == str((upscale_client.root / "set" / "portrait.png").resolve())
    assert [stage.__name__ for stage in submitted["workflow"].stages][2] == "load_source_stage"
    assert submitted["context"]["workflow"] == "upscale"
    assert submitted["context"]["settings"] == {}
    assert submitted["context"]["meta"].startswith("2× → 1664×2432")
    assert payload["queue_position"] is None


def test_upscale_route_unknown_asset_is_404(upscale_client):
    response = upscale_client.client.post("/api/upscale", json={"asset_id": "missing.png", "factor": 2})

    assert response.status_code == 404


def test_upscale_route_rejects_path_outside_output_dir(upscale_client):
    response = upscale_client.client.post("/api/upscale", json={"asset_id": "../outside.png", "factor": 2})

    assert response.status_code == 422


def test_upscale_route_rejects_bad_factor(upscale_client):
    _write_png(upscale_client.root / "a.png")

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 3})

    assert response.status_code == 422


def test_upscale_route_rejects_unsupported_model(upscale_client):
    _patch_model_resolution(upscale_client.monkeypatch, defaults={**_DEFAULTS, "supports_img2img": False})
    _write_png(upscale_client.root / "a.png")

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 2})

    assert response.status_code == 422


def test_upscale_route_rejects_output_over_cap(upscale_client):
    _write_png(upscale_client.root / "a.png")

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 4})

    assert response.status_code == 422
    assert "megapixel" in response.json()["detail"]


def test_upscale_route_rejects_video(upscale_client):
    (upscale_client.root / "clip.mp4").write_bytes(b"")

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "clip.mp4", "factor": 2})

    assert response.status_code == 404


def test_upscale_route_conflict_while_busy(upscale_client):
    _write_png(upscale_client.root / "a.png")

    def _busy(**_kwargs):
        raise JobConflictError("busy")

    upscale_client.monkeypatch.setattr(web_server.web_runner, "submit_upscale_job", _busy)

    response = upscale_client.client.post("/api/upscale", json={"asset_id": "a.png", "factor": 2})

    assert response.status_code == 409


# ── Gallery asset fields ────────────────────────────────────────────────────


def _asset_json(tmp_path, name, web_config=None):
    web_config = web_config or _web_config(tmp_path, image_model_options=("zit", "ideo"))
    asset = next(item for item in list_gallery_assets(str(tmp_path)) if item.name == name)
    return gallery_asset_to_json(asset, web_config)


def test_gallery_upscale_field_lists_factors(tmp_path):
    _write_png(tmp_path / "a.png", config={"prompt": "p", "model": "zit"})

    upscale = _asset_json(tmp_path, "a.png")["upscale"]

    assert upscale["factors"] == [
        {"factor": 2, "width": 1664, "height": 2432, "allowed": True, "reason": None},
        {"factor": 4, "width": 3328, "height": 4864, "allowed": False, "reason": upscale["factors"][1]["reason"]},
    ]
    assert "megapixel" in upscale["factors"][1]["reason"]


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
    assert details["generation"] == {"time": 12.3, "quantize": 8, "upscale": {"factor": 2, "denoise": 0.4, "steps": 3, "pre_sharpen": 0.6}, "sharpen": 1.2}
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


def test_gallery_video_has_no_upscale_field(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"")

    assert _asset_json(tmp_path, "clip.mp4")["upscale"] is None
