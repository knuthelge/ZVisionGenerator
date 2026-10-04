"""Focused route tests for the Phase A web backend contracts."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest

from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import model_inventory as model_inventory_module
from zvisiongenerator.web import path_picker as path_picker_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.gallery import list_gallery_assets
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.provenance import embed_png_config


def _make_web_config() -> SimpleNamespace:
    app_config = {
        "generation": {
            "default_ratio": "2:3",
            "default_size": "m",
        },
        "sizes": {
            "2:3": {
                "m": {"width": 832, "height": 1216},
                "l": {"width": 1024, "height": 1536},
            },
        },
    }
    return SimpleNamespace(
        app_config=app_config,
        startup_view="config",
        gallery_page_size=12,
        data_dir="/tmp/.ziv",
        output_dir="/tmp/outputs",
        models_dir="/tmp/models",
        loras_dir="/tmp/loras",
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
        image_model_options=("zit", "local-image"),
        video_model_options=("ltx-8",),
        lora_options=("style",),
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m", "l")},
        image_size_dimensions={"2:3": {"m": (832, 1216), "l": (1152, 1728)}},
        video_size_dimensions={"16:9": {"m": (704, 448)}},
        video_ratios=("16:9",),
        video_size_options={"16:9": ("m",)},
        scheduler_options=("beta",),
        quantize_options=(4, 8),
        image_inventory=(),
        video_inventory=(),
    )


def _make_workspace_bootstrap_view() -> dict[str, object]:
    image_defaults = {
        "ratio": "2:3",
        "size": "m",
        "steps": 10,
        "guidance": 3.5,
        "width": 832,
        "height": 1216,
        "scheduler": None,
        "supports_negative_prompt": True,
        "supports_quantize": True,
        "supports_img2img": True,
        "supports_upscale": True,
        "supports_json_prompt": False,
        "supports_first_sigma": False,
        "dimension_min": 16,
        "dimension_max": None,
        "dimension_step": 16,
        "quantize": None,
        "image_strength": 0.5,
        "postprocess": {"sharpen": 0.8, "contrast": False, "saturation": False},
        "upscale": {"enabled": False, "factor": None, "denoise": None, "steps": None, "guidance": None, "sharpen": True, "save_pre": False},
    }
    video_defaults = {
        "ratio": "16:9",
        "size": "m",
        "steps": 8,
        "width": 704,
        "height": 448,
        "frame_count": 49,
        "audio": True,
        "low_memory": True,
        "supports_i2v": True,
        "supports_quantize": False,
        "quantize": None,
        "max_steps": 8,
        "fps": 24,
        "upscale": {"enabled": False, "factor": 2, "steps": None},
    }
    return {
        "image_default_model": "zit",
        "video_default_model": "ltx-8",
        "image_model_defaults": {"zit": image_defaults, "local-image": image_defaults},
        "video_model_defaults": {"ltx-8": video_defaults},
    }


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "teal").save(path)


def _write_png_with_config(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    embed_png_config(info, payload)
    Image.new("RGB", (8, 8), "teal").save(path, pnginfo=info)


def _write_png_with_description(path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    info.add_text("Description", description)
    Image.new("RGB", (8, 8), "teal").save(path, pnginfo=info)


def _png_upload_bytes() -> io.BytesIO:
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), "teal").save(payload, format="PNG")
    payload.seek(0)
    return payload


def _assert_non_empty_string(value: object) -> None:
    assert isinstance(value, str)
    assert value


def _make_resolved_image_defaults(
    *,
    supports_negative_prompt: bool = True,
    supports_quantize: bool = True,
    supports_img2img: bool = True,
    supports_upscale: bool = True,
    supports_json_prompt: bool = False,
    supports_first_sigma: bool = False,
    dimension_min: int = 16,
    dimension_max: int | None = None,
    dimension_step: int = 16,
) -> dict[str, object]:
    return {
        "steps": 10,
        "guidance": 3.5,
        "scheduler": None,
        "supports_negative_prompt": supports_negative_prompt,
        "supports_quantize": supports_quantize,
        "supports_img2img": supports_img2img,
        "supports_upscale": supports_upscale,
        "supports_json_prompt": supports_json_prompt,
        "supports_first_sigma": supports_first_sigma,
        "dimension_min": dimension_min,
        "dimension_max": dimension_max,
        "dimension_step": dimension_step,
    }


def _patch_image_submit_dependencies(
    monkeypatch,
    *,
    model_info: ImageModelInfo,
    defaults: dict[str, object],
    submitted: list[dict[str, object]] | None = None,
) -> None:
    monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(web_server, "detect_image_model", lambda _model: model_info)
    monkeypatch.setattr(web_server, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(web_server, "validate_scheduler", lambda _scheduler, _config: None)
    monkeypatch.setattr(web_server, "resolve_defaults", lambda _model_info, _config, _cli_overrides, _backend_name: defaults)

    def _capture_submit(**kwargs):
        if submitted is not None:
            submitted.append(kwargs)
        return "job-123"

    monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", _capture_submit)


def test_phase_a_routes_share_config_and_path_authority(monkeypatch):
    """Config, workspace, and models routes should expose the same backend-owned authority."""
    web_config = _make_web_config()
    active_job_snapshot = {
        "id": "job-live",
        "job_id": "job-live",
        "workflow": "txt2img",
        "job_type": "Text to Image",
        "status": "running",
        "created_at": "2026-04-30T09:00:00Z",
        "completed_at": None,
        "event_count": 2,
        "last_event": {"type": "step_progress", "current_step": 1, "total_steps": 10},
        "supported_controls": ["next", "pause", "resume", "repeat", "quit"],
        "supports_controls": ["next", "pause", "resume", "repeat", "quit"],
        "paused": False,
        "result_path": None,
        "prompt": "prompt",
        "model": "zit",
        "runs": 1,
    }
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "list_gallery_assets", lambda _output_dir: [])
    monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: _make_workspace_bootstrap_view())
    monkeypatch.setattr(web_server, "huggingface_token_env_var", lambda: "HF_TOKEN")
    monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: active_job_snapshot)
    monkeypatch.setattr(
        config_contract_module,
        "read_user_config_override",
        lambda: {
            "ui": {
                "default_models": {"image": "zit", "video": "ltx-8"},
                "output_dir": "/persisted/outputs",
            },
            "generation": {"default_size": "l"},
        },
    )

    with TestClient(web_server.app) as client:
        config_response = client.get("/api/config")
        workspace_response = client.get("/api/workspace")
        models_response = client.get("/api/models")

    assert config_response.status_code == 200
    assert workspace_response.status_code == 200
    assert models_response.status_code == 200

    config_payload = config_response.json()
    workspace_payload = workspace_response.json()
    models_payload = models_response.json()
    schema_fields = {field["key"]: field for field in config_payload["writable_config"]["fields"]}
    prompt_file = workspace_payload["prompt_file"]

    assert config_payload["output_dir"] == "/tmp/outputs"
    assert workspace_payload["output_dir"] == "/tmp/outputs"
    assert config_payload["ui"]["loras_dir"] == "/tmp/loras"
    assert models_payload["loras_dir"] == "/tmp/loras"
    assert workspace_payload["loras"][0]["path"] == "/tmp/loras/style.safetensors"
    assert workspace_payload["image_size_dimensions"]["2:3"]["m"] == [832, 1216]
    assert workspace_payload["video_size_dimensions"]["16:9"]["m"] == [704, 448]

    assert sorted(schema_fields) == [
        "generation.default_size",
        "prompt_enhancer.user_model",
        "ui.default_models.image",
        "ui.default_models.video",
        "ui.output_dir",
    ]
    assert schema_fields["ui.output_dir"]["omitted"] == "unchanged"
    assert schema_fields["ui.output_dir"]["null"] == "clear"
    assert schema_fields["ui.output_dir"]["empty_string"] == "clear"
    assert schema_fields["ui.output_dir"]["persisted_value"] == "/persisted/outputs"
    assert schema_fields["ui.output_dir"]["effective_value"] == "/tmp/outputs"
    assert "default_source" in schema_fields["ui.output_dir"]
    _assert_non_empty_string(schema_fields["ui.output_dir"]["default_source"])
    assert schema_fields["ui.output_dir"]["owning_consumer"]

    assert "legacy_aliases" not in workspace_payload["workflow_contract"]
    assert workspace_payload["workflow_contract"]["definitions"]["txt2img"]["visible_controls"]
    assert workspace_payload["active_job"] == active_job_snapshot
    assert prompt_file["accepted_extensions"] == [".yaml", ".yml"]
    assert prompt_file["browse_kind"] == "existing_file"
    assert prompt_file["selection_required"] is True
    assert prompt_file["trust_boundary"]["scope"] == "server_host_only"
    assert prompt_file["trust_boundary"]["manual_entry"] == "submitted_value_kept_until_backend_validation"
    assert set(prompt_file["help"]) == {
        "path",
        "editor",
        "option_required",
        "option_optional",
        "empty_options",
        "stale_selection",
        "loaded",
        "saved",
        "ignored_negative_video",
        "ignored_negative_unsupported",
    }
    assert all(isinstance(value, str) and value for value in prompt_file["help"].values())


def test_workspace_route_can_skip_history_asset_serialization(monkeypatch):
    """Initial workspace hydration should be able to avoid gallery inventory work."""
    web_config = _make_web_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: _make_workspace_bootstrap_view())
    monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)

    def _fail_list_gallery_assets(_output_dir: str):
        raise AssertionError("workspace core hydration should not list gallery assets")

    monkeypatch.setattr(web_server, "list_gallery_assets", _fail_list_gallery_assets)

    with TestClient(web_server.app) as client:
        response = client.get("/api/workspace?include_history=false")

    assert response.status_code == 200
    payload = response.json()
    assert payload["history_assets"] == []
    assert payload["active_job"] is None


def test_docs_asset_rejects_path_traversal():
    """Docs assets should remain confined to the docs/assets directory."""
    with TestClient(web_server.app) as client:
        response = client.get("/docs/assets/..%2F..%2FREADME.md")

    assert response.status_code == 404


def test_submit_image_job_uses_backend_registry_name(monkeypatch, tmp_path):
    """Image job defaults should get their backend name from the backend registry helper."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(web_server, "detect_image_model", lambda _model: ImageModelInfo(family="zimage", is_distilled=False, size="xl"))
    monkeypatch.setattr(web_server, "get_backend_name", lambda: "registry-owned")
    monkeypatch.setattr(web_server, "validate_scheduler", lambda _scheduler, _config: None)
    monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", lambda **_: "job-123")

    captured: dict[str, str] = {}

    def _fake_resolve_defaults(model_info, config, cli_overrides, backend_name):
        captured["backend_name"] = backend_name
        return {
            "steps": 11,
            "guidance": 4.0,
            "scheduler": None,
            "supports_negative_prompt": True,
        }

    monkeypatch.setattr(web_server, "resolve_defaults", _fake_resolve_defaults)

    response = web_server._submit_image_job(
        {"prompt": "hello world", "output": str(tmp_path / "stale-browser-output")},
        web_config,
    )

    assert response["job_id"] == "job-123"
    assert response["output_dir"] == str(tmp_path)
    assert captured["backend_name"] == "registry-owned"


def test_workspace_bootstrap_uses_backend_registry_name(monkeypatch):
    """Workspace bootstrap defaults should use the backend registry helper for schedulers."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda _model: ImageModelInfo(family="zimage", is_distilled=False, size="xl"))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "registry-owned")

    captured: dict[str, str] = {}

    def _fake_resolve_defaults(model_info, config, cli_overrides, backend_name):
        captured["backend_name"] = backend_name
        return {
            "steps": 10,
            "guidance": 3.5,
            "scheduler": "beta",
            "supports_negative_prompt": True,
        }

    monkeypatch.setattr(workspace_api_module, "resolve_defaults", _fake_resolve_defaults)

    defaults = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)

    assert captured["backend_name"] == "registry-owned"
    assert defaults["scheduler"] == "beta"


def test_workspace_bootstrap_defaults_include_ideogram_capability_flags(monkeypatch):
    """Bootstrap defaults should surface the declared-family capability contract."""
    web_config = _make_web_config()
    web_config.app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
        "zit": "Tongyi-MAI/Z-Image-Turbo",
    }
    web_config.app_config["model_alias_families"] = {"ideo": "ideogram4"}
    web_config.image_model_options = ("ideo", "zit")

    detect_calls: list[str] = []

    def _fake_detect_image_model(value: object) -> ImageModelInfo:
        detect_calls.append(str(value))
        return ImageModelInfo(family="zimage", is_distilled=False, size="xl")

    def _fake_resolve_defaults(model_info, _config, _cli_overrides, _backend_name):
        if model_info.family == "ideogram4":
            return _make_resolved_image_defaults(
                supports_negative_prompt=False,
                supports_quantize=False,
                supports_img2img=False,
                supports_upscale=False,
                supports_json_prompt=True,
                supports_first_sigma=True,
                dimension_min=256,
                dimension_max=2048,
                dimension_step=16,
            )
        return _make_resolved_image_defaults()

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "detect_image_model", _fake_detect_image_model)
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", _fake_resolve_defaults)

    ideogram_defaults = workspace_api_module._build_image_bootstrap_defaults("ideo", web_config)
    zimage_defaults = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)

    assert ideogram_defaults["supports_img2img"] is False
    assert ideogram_defaults["supports_upscale"] is False
    assert ideogram_defaults["supports_quantize"] is False
    assert ideogram_defaults["supports_json_prompt"] is True
    assert ideogram_defaults["supports_first_sigma"] is True
    assert ideogram_defaults["dimension_min"] == 256
    assert ideogram_defaults["dimension_max"] == 2048
    assert ideogram_defaults["dimension_step"] == 16
    assert zimage_defaults["supports_img2img"] is True
    assert zimage_defaults["supports_upscale"] is True
    assert zimage_defaults["supports_quantize"] is True
    assert zimage_defaults["supports_json_prompt"] is False
    assert zimage_defaults["supports_first_sigma"] is False
    assert zimage_defaults["dimension_min"] == 16
    assert zimage_defaults["dimension_max"] is None
    assert zimage_defaults["dimension_step"] == 16
    assert detect_calls == ["Tongyi-MAI/Z-Image-Turbo"]


def test_workspace_bootstrap_defaults_fall_back_to_supported_ideogram_preset(monkeypatch):
    web_config = _make_web_config()
    web_config.app_config["generation"] = {
        "default_ratio": "16:9",
        "default_size": "xl",
    }
    web_config.app_config["sizes"] = {
        "16:9": {
            "m": {"width": 1344, "height": 768},
            "l": {"width": 1888, "height": 1056},
            "xl": {"width": 2112, "height": 1184},
        },
        "2:3": {
            "m": {"width": 832, "height": 1216},
        },
    }
    web_config.image_ratios = ("16:9", "2:3")
    web_config.image_size_options = {
        "16:9": ("m", "l", "xl"),
        "2:3": ("m",),
    }
    web_config.app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
    }
    web_config.app_config["model_alias_families"] = {"ideo": "ideogram4"}

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", lambda *_args, **_kwargs: _make_resolved_image_defaults(dimension_min=256, dimension_max=2048, dimension_step=16))

    defaults = workspace_api_module._build_image_bootstrap_defaults("ideo", web_config)

    assert defaults["ratio"] == "16:9"
    assert defaults["size"] == "l"
    assert defaults["width"] == 1888
    assert defaults["height"] == 1056


def test_workspace_bootstrap_defaults_disable_quantize_when_globally_unavailable(monkeypatch):
    web_config = _make_web_config()
    web_config.app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
        "zit": "Tongyi-MAI/Z-Image-Turbo",
    }
    web_config.app_config["model_alias_families"] = {"ideo": "ideogram4"}
    web_config.image_model_options = ("ideo", "zit")
    web_config.quantize_options = ()

    def _fake_detect_image_model(value: object) -> ImageModelInfo:
        return ImageModelInfo(family="zimage", is_distilled=False, size="xl")

    def _fake_resolve_defaults(model_info, _config, _cli_overrides, _backend_name):
        if model_info.family == "ideogram4":
            return _make_resolved_image_defaults(supports_negative_prompt=False, supports_quantize=False)
        return _make_resolved_image_defaults(supports_quantize=True)

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "detect_image_model", _fake_detect_image_model)
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", _fake_resolve_defaults)

    ideogram_defaults = workspace_api_module._build_image_bootstrap_defaults("ideo", web_config)
    zimage_defaults = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)

    assert ideogram_defaults["supports_quantize"] is False
    assert zimage_defaults["supports_quantize"] is False


def test_submit_image_job_threads_json_prompt_and_first_sigma_to_authoritative_args(monkeypatch, tmp_path):
    """JSON-only submissions should thread their authoritative args and prompt payloads."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.default_models = WebUiDefaultModels(image="ideo", video="ltx-8")
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(
            supports_negative_prompt=False,
            supports_img2img=False,
            supports_upscale=False,
            supports_json_prompt=True,
            supports_first_sigma=True,
            dimension_min=256,
            dimension_max=2048,
            dimension_step=16,
        ),
        submitted=submitted,
    )

    response = web_server._submit_image_job(
        {
            "model": "ideo",
            "json_prompt": '{"high_level_description":"x"}',
            "first_sigma": "1.005",
        },
        web_config,
    )

    assert response["job_id"] == "job-123"
    assert len(submitted) == 1
    assert submitted[0]["args"].first_sigma == 1.005
    assert submitted[0]["args"].json_prompt_enabled is True
    assert submitted[0]["prompts_data"] == {"prompt": [('{"high_level_description":"x"}', None)]}
    assert submitted[0]["request"].first_sigma == 1.005
    assert submitted[0]["request"].json_prompt is True
    assert submitted[0]["request"].prompt == '{"high_level_description":"x"}'


@pytest.mark.parametrize(
    ("form", "expected_steps", "expected_guidance", "expected_steps_explicit", "expected_guidance_explicit"),
    [
        ({"model": "ideo", "prompt": "hello", "steps": "28", "guidance": "6.0"}, 28, 6.0, True, True),
        ({"model": "ideo", "prompt": "hello"}, 10, 3.5, False, False),
    ],
)
def test_submit_image_job_threads_steps_and_guidance_explicit_flags_to_authoritative_args(
    monkeypatch,
    tmp_path,
    form,
    expected_steps,
    expected_guidance,
    expected_steps_explicit,
    expected_guidance_explicit,
):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    defaults = _make_resolved_image_defaults(
        supports_negative_prompt=False,
        supports_img2img=False,
        supports_upscale=False,
        supports_json_prompt=True,
        supports_first_sigma=True,
        dimension_min=256,
        dimension_max=2048,
        dimension_step=16,
    )
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=defaults,
        submitted=submitted,
    )

    def _resolve_defaults(_model_info, _config, cli_overrides, _backend_name):
        resolved = dict(defaults)
        resolved["steps"] = cli_overrides.get("steps", defaults["steps"])
        resolved["guidance"] = cli_overrides.get("guidance", defaults["guidance"])
        resolved["scheduler"] = cli_overrides.get("scheduler", defaults["scheduler"])
        return resolved

    monkeypatch.setattr(web_server, "resolve_defaults", _resolve_defaults)

    response = web_server._submit_image_job(form, web_config)

    assert response["job_id"] == "job-123"
    assert len(submitted) == 1
    assert submitted[0]["args"].steps == expected_steps
    assert submitted[0]["args"].guidance == expected_guidance
    assert submitted[0]["args"].steps_explicit is expected_steps_explicit
    assert submitted[0]["args"].guidance_explicit is expected_guidance_explicit
    assert submitted[0]["request"].steps == expected_steps
    assert submitted[0]["request"].guidance == expected_guidance
    assert submitted[0]["request"].steps_explicit is expected_steps_explicit
    assert submitted[0]["request"].guidance_explicit is expected_guidance_explicit


@pytest.mark.parametrize(
    ("form", "expected_substring"),
    [
        (
            {
                "model": "ideo",
                "prompt": "plain prompt",
                "json_prompt": '{"high_level_description":"x"}',
            },
            "Provide either a prompt or a structured JSON caption, not both.",
        ),
        (
            {
                "model": "ideo",
                "prompt_source": "file",
                "json_prompt": '{"high_level_description":"x"}',
            },
            "A structured JSON caption cannot be combined with prompt-file mode.",
        ),
    ],
)
def test_submit_image_job_rejects_json_prompt_mutual_exclusion_cases(monkeypatch, tmp_path, form, expected_substring):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_json_prompt=True, supports_first_sigma=True),
    )

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job(form, web_config)

    assert expected_substring in str(exc_info.value)


@pytest.mark.parametrize("value", ["0", "2.5", "-1"])
def test_submit_image_job_rejects_out_of_band_first_sigma(monkeypatch, tmp_path, value):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_json_prompt=True, supports_first_sigma=True),
    )

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job({"model": "ideo", "prompt": "hello", "first_sigma": value}, web_config)

    assert "first_sigma must be in (0.0, 2.0]" in str(exc_info.value)


@pytest.mark.parametrize("json_prompt_value", ["not json", "[]", '"caption"'])
def test_submit_image_job_rejects_invalid_json_prompt_values(monkeypatch, tmp_path, json_prompt_value):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_json_prompt=True, supports_first_sigma=True),
    )

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job({"model": "ideo", "json_prompt": json_prompt_value}, web_config)

    assert "json_prompt must be a JSON object" in str(exc_info.value)


@pytest.mark.parametrize(
    ("form", "expected_substring"),
    [
        ({"model": "zit", "json_prompt": '{"high_level_description":"x"}'}, "This model does not support structured JSON captions."),
        ({"model": "zit", "prompt": "hello", "first_sigma": "1.005"}, "This model does not support the first-step sigma control."),
    ],
)
def test_submit_image_job_rejects_unsupported_model_json_prompt_and_first_sigma(monkeypatch, tmp_path, form, expected_substring):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_json_prompt=False, supports_first_sigma=False),
    )

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job(form, web_config)

    assert expected_substring in str(exc_info.value)


def test_submit_image_job_rejects_ideogram_capability_violations_before_queue(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.app_config["sizes"]["16:9"] = {"xl": {"width": 2112, "height": 1184}}
    web_config.image_ratios = ("2:3", "16:9")
    web_config.image_size_options["16:9"] = ("xl",)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(
            supports_quantize=False,
            supports_img2img=False,
            supports_upscale=False,
            supports_json_prompt=True,
            supports_first_sigma=True,
            dimension_min=256,
            dimension_max=2048,
            dimension_step=16,
        ),
        submitted=submitted,
    )

    image_path = tmp_path / "reference.png"
    _write_png(image_path)

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job({"model": "ideo", "prompt": "hello", "image_path": str(image_path)}, web_config)
    assert "does not support reference-image (img2img) steering." in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job({"model": "ideo", "prompt": "hello", "upscale": "2"}, web_config)
    assert "does not support upscaling." in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job({"model": "ideo", "prompt": "hello", "quantize": "4"}, web_config)
    assert "This model does not support quantization." in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        web_server._submit_image_job({"model": "ideo", "prompt": "hello", "ratio": "16:9", "size": "xl"}, web_config)
    error_text = str(exc_info.value)
    assert "must be between" in error_text
    assert "multiple of" in error_text
    assert submitted == []


def test_submit_image_job_accepts_quantize_for_supported_models(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_negative_prompt=True, supports_quantize=True),
        submitted=submitted,
    )

    response = web_server._submit_image_job(
        {
            "model": "zit",
            "prompt": "hello world",
            "quantize": "4",
        },
        web_config,
    )

    assert response["job_id"] == "job-123"
    assert len(submitted) == 1
    assert submitted[0]["args"].quantize == 4
    assert submitted[0]["request"].prompt == "hello world"


def test_submit_image_job_standard_prompt_path_remains_unchanged_for_non_ideogram(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_negative_prompt=True),
        submitted=submitted,
    )

    response = web_server._submit_image_job(
        {
            "model": "zit",
            "prompt": "hello world",
            "negative_prompt": "avoid blur",
        },
        web_config,
    )

    assert response["job_id"] == "job-123"
    assert len(submitted) == 1
    assert submitted[0]["args"].json_prompt_enabled is False
    assert submitted[0]["args"].first_sigma is None
    assert submitted[0]["prompts_data"] == {"web": [("hello world", "avoid blur")]}
    assert submitted[0]["request"].prompt == "hello world"


def test_picker_route_rejects_unknown_or_mismatched_host_local_purpose():
    """Picker requests should stay inside explicit backend-owned host-local trust buckets."""
    with TestClient(web_server.app) as client:
        unknown_response = client.post("/api/picker", json={"kind": "directory", "purpose": "unknown", "initial_path": None})
        mismatch_response = client.post("/api/picker", json={"kind": "directory", "purpose": "prompt_file", "initial_path": None})

    assert unknown_response.status_code == 422
    unknown_payload = unknown_response.json()
    assert set(unknown_payload) == {"detail"}
    assert "unknown" in unknown_payload["detail"]
    assert mismatch_response.status_code == 422
    mismatch_payload = mismatch_response.json()
    assert set(mismatch_payload) == {"detail"}
    assert "prompt_file" in mismatch_payload["detail"]
    assert "existing_file" in mismatch_payload["detail"]


def test_picker_route_accepts_model_file_purposes(monkeypatch):
    """Model-management Browse buttons should use backend-owned picker purposes."""
    captured_requests: list[tuple[str, str, str | None]] = []

    def _fake_pick_path(kind: str, *, purpose: str, initial_path: str | None = None):
        captured_requests.append((kind, purpose, initial_path))
        return SimpleNamespace(to_payload=lambda: {"status": "cancelled", "path": None, "message": None})

    monkeypatch.setattr(web_server, "pick_path", _fake_pick_path)

    with TestClient(web_server.app) as client:
        checkpoint_response = client.post("/api/picker", json={"kind": "existing_file", "purpose": "checkpoint_file", "initial_path": "/models/model.safetensors"})
        lora_response = client.post("/api/picker", json={"kind": "existing_file", "purpose": "lora_file", "initial_path": "/loras/style.safetensors"})

    assert checkpoint_response.status_code == 200
    assert lora_response.status_code == 200
    assert captured_requests == [
        ("existing_file", "checkpoint_file", "/models/model.safetensors"),
        ("existing_file", "lora_file", "/loras/style.safetensors"),
    ]


def test_model_picker_purposes_require_existing_safetensors_files(tmp_path, monkeypatch):
    """Checkpoint and LoRA picker purposes should accept only host-local safetensors files."""
    checkpoint = tmp_path / "checkpoint.safetensors"
    lora = tmp_path / "style.safetensors"
    wrong_extension = tmp_path / "notes.txt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    lora.write_text("lora", encoding="utf-8")
    wrong_extension.write_text("text", encoding="utf-8")
    selected_paths = iter([str(checkpoint), str(lora), str(wrong_extension), str(tmp_path / "missing.safetensors")])

    monkeypatch.setattr(path_picker_module.sys, "platform", "linux")
    monkeypatch.setattr(path_picker_module, "_pick_tk", lambda kind, initial_path, picker_purpose: next(selected_paths))

    checkpoint_result = path_picker_module.pick_path("existing_file", purpose="checkpoint_file")
    lora_result = path_picker_module.pick_path("existing_file", purpose="lora_file")
    wrong_extension_result = path_picker_module.pick_path("existing_file", purpose="checkpoint_file")
    missing_result = path_picker_module.pick_path("existing_file", purpose="lora_file")

    assert checkpoint_result.status == "selected"
    assert checkpoint_result.path == str(checkpoint)
    assert lora_result.status == "selected"
    assert lora_result.path == str(lora)
    assert wrong_extension_result.status == "error"
    assert ".safetensors" in (wrong_extension_result.message or "")
    assert missing_result.status == "error"
    assert "existing file" in (missing_result.message or "")


def test_prompt_file_routes_reject_non_local_or_wrong_extension_paths(tmp_path):
    """Prompt-file routes should reject non-host-local URLs and non-YAML files visibly."""
    text_file = tmp_path / "prompts.txt"
    text_file.write_text("prompts: []\n", encoding="utf-8")

    with TestClient(web_server.app) as client:
        remote_response = client.post("/api/prompt-files/inspect", json={"path": "https://example.com/prompts.yaml"})
        extension_response = client.post("/api/prompt-files/inspect", json={"path": str(text_file)})

    assert remote_response.status_code == 422
    remote_payload = remote_response.json()
    assert set(remote_payload) == {"detail"}
    assert isinstance(remote_payload["detail"], str)
    assert extension_response.status_code == 422
    extension_payload = extension_response.json()
    assert set(extension_payload) == {"detail"}
    assert ".yaml" in extension_payload["detail"]
    assert ".yml" in extension_payload["detail"]


def test_manual_prompt_file_submission_rejects_missing_host_local_path():
    """Manual prompt-file submissions should fail visibly instead of silently falling back."""
    with pytest.raises(ValueError):
        web_server._resolve_prompt_submission_with_enhance(
            {
                "prompt_source": "file",
                "prompts_file": "/missing/prompts.yaml",
                "prompt_option_id": "portrait:0",
            }
        )


def test_packaged_spa_serves_packaged_logo_asset() -> None:
    """The SPA should reference and serve logo assets from the packaged app static tree."""
    with TestClient(web_server.app) as client:
        app_response = client.get("/app")
        logo_response = client.get("/app-static/ziv-icon.png")

    assert app_response.status_code == 200
    assert "/docs/assets/" not in app_response.text
    assert logo_response.status_code == 200
    assert logo_response.headers["content-type"] == "image/png"


def test_generate_route_returns_requested_runs_from_job_context(monkeypatch):
    """The public generate response should return the queued job's requested runs value."""
    monkeypatch.setattr(web_server, "load_web_config", _make_web_config)
    monkeypatch.setattr(
        web_server,
        "_submit_image_job",
        lambda _form, _web_config: {
            "job_id": "job-123",
            "job_type": "txt2img",
            "title": "zit",
            "prompt": "prompt",
            "events_url": "/jobs/job-123/events",
            "status_url": "/jobs/job-123",
            "supported_controls": ("next", "pause"),
            "runs": 7,
            "meta": "2:3 · m · 10 steps",
        },
    )

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "image"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-123"
    assert payload["runs"] == 7
    assert payload["supported_controls"] == ["next", "pause"]


def test_video_submission_missing_ffmpeg_returns_422_before_job_registration(monkeypatch):
    """The Web UI rejects missing ffmpeg as validation without entering a worker/install flow."""
    web_config = _make_web_config()
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        web_server,
        "detect_video_model",
        lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=64, frame_alignment=8),
    )
    monkeypatch.setattr(web_server, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(web_server, "_normalize_video_args", lambda *_args: None)
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg is missing; install it and retry")))
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "video", "model": "ltx-8", "prompt": "a lake"})

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert "ffmpeg" in response.json()["detail"].lower()
    assert "install" in response.json()["detail"].lower()
    assert submitted == []


def test_image_submission_does_not_require_ffmpeg(monkeypatch):
    """Image submissions retain their existing path when the video prerequisite is unavailable."""
    monkeypatch.setattr(web_server, "load_web_config", _make_web_config)
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: (_ for _ in ()).throw(AssertionError("image submissions must not check ffmpeg")))
    monkeypatch.setattr(
        web_server,
        "_submit_image_job",
        lambda _form, _web_config: {
            "job_id": "job-123",
            "job_type": "txt2img",
            "title": "zit",
            "prompt": "prompt",
            "events_url": "/jobs/job-123/events",
            "status_url": "/jobs/job-123",
            "supported_controls": (),
            "runs": 1,
            "meta": "",
        },
    )

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "image"})

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-123"


def test_generate_route_rejects_unsupported_workflow_alias(monkeypatch):
    """Generate submissions should accept canonical workflow values only."""
    monkeypatch.setattr(web_server, "load_web_config", _make_web_config)

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "image", "workflow": "image"})

    assert response.status_code == 422
    assert "Unknown workflow 'image'" in response.json()["detail"]


def test_dummy_job_route_is_not_exposed_by_production_app(monkeypatch):
    """The production Web API should not accept test-only dummy job submissions."""
    submitted_dummy_jobs: list[tuple[int, float]] = []
    monkeypatch.setattr(web_server.web_runner, "submit_dummy_job", lambda *, total_steps, delay_seconds: submitted_dummy_jobs.append((total_steps, delay_seconds)))

    with TestClient(web_server.app) as client:
        response = client.post("/jobs/dummy", params={"steps": 999999, "delay_seconds": 999999})
        openapi_response = client.get("/openapi.json")

    assert response.status_code == 405
    assert submitted_dummy_jobs == []
    assert "/jobs/dummy" not in openapi_response.json()["paths"]


def test_cancel_route_returns_terminal_status_for_finished_jobs(monkeypatch):
    """Cancelling a terminal job should report the real terminal state without queuing controls."""
    queued_controls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        web_server.web_runner,
        "get_job_snapshot",
        lambda job_id: {
            "id": job_id,
            "job_id": job_id,
            "workflow": "txt2img",
            "job_type": "Text to Image",
            "status": "completed",
            "created_at": "2026-04-30T09:00:00Z",
            "completed_at": "2026-04-30T09:00:05Z",
            "event_count": 4,
            "last_event": {"type": "job_completed"},
            "supported_controls": [],
            "paused": False,
            "result_path": "/tmp/output.png",
            "prompt": "done",
            "model": "zit",
            "runs": 1,
        },
    )
    monkeypatch.setattr(web_server.web_runner, "queue_job_control", lambda job_id, action: queued_controls.append((job_id, action)))

    with TestClient(web_server.app) as client:
        response = client.post("/api/jobs/job-done/cancel")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-done", "status": "completed"}
    assert queued_controls == []


def test_cancel_route_rejects_running_jobs_without_cancel_support(monkeypatch):
    """Cancelling an uncancellable running job should fail instead of lying about success."""
    queued_controls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        web_server.web_runner,
        "get_job_snapshot",
        lambda job_id: {
            "id": job_id,
            "job_id": job_id,
            "workflow": "txt2vid",
            "job_type": "Text to Video",
            "status": "running",
            "created_at": "2026-04-30T09:00:00Z",
            "completed_at": None,
            "event_count": 2,
            "last_event": {"type": "step_progress", "current_step": 1, "total_steps": 8},
            "supported_controls": [],
            "paused": False,
            "result_path": None,
            "prompt": "video",
            "model": "ltx-8",
            "runs": 1,
        },
    )
    monkeypatch.setattr(web_server.web_runner, "queue_job_control", lambda job_id, action: queued_controls.append((job_id, action)))

    with TestClient(web_server.app) as client:
        response = client.post("/api/jobs/job-video/cancel")

    assert response.status_code == 409
    assert set(response.json()) == {"detail"}
    assert queued_controls == []


def test_models_route_uses_shared_alias_inventory(monkeypatch, tmp_path):
    """Models route should expose the same alias-backed inventory authority as config loading."""
    web_config = _make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.app_config["model_aliases"] = {
        "alias-image": str(tmp_path / "models" / "alias-image"),
        "alias-video": str(tmp_path / "models" / "alias-video"),
    }
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(
        model_inventory_module,
        "list_models",
        lambda _data_dir: [SimpleNamespace(name="local-image", family="zimage", size="m"), SimpleNamespace(name="local-image@q8", family="zimage", size="m")],
    )
    monkeypatch.setattr(
        model_inventory_module,
        "list_video_models",
        lambda _data_dir: [SimpleNamespace(name="local-video", family="ltx", supports_i2v=True)],
    )
    monkeypatch.setattr(
        model_inventory_module,
        "resolve_model_path",
        lambda name, **_: web_config.app_config["model_aliases"].get(name, name),
    )
    monkeypatch.setattr(
        model_inventory_module,
        "detect_image_model",
        lambda value: ImageModelInfo(family="zimage" if "alias-image" in str(value) else "unknown", is_distilled=False, size="xl"),
    )
    monkeypatch.setattr(
        model_inventory_module,
        "detect_video_model",
        lambda value: SimpleNamespace(family="ltx" if "alias-video" in str(value) else "unknown", supports_i2v=False),
    )
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
    # load_web_config discovers the inventory once; the route must serve that instead of discovering again.
    web_config.image_inventory = model_inventory_module.discover_image_inventory(web_config.app_config, tmp_path)
    web_config.video_inventory = model_inventory_module.discover_video_inventory(web_config.app_config, tmp_path)
    monkeypatch.setattr(model_inventory_module, "list_models", lambda _data_dir: pytest.fail("models route rediscovered the inventory"))

    with TestClient(web_server.app) as client:
        response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    image_models = {entry["name"]: entry for entry in payload["image_models"]}
    video_models = {entry["name"]: entry for entry in payload["video_models"]}

    assert image_models["local-image"]["source"] == "installed"
    assert image_models["alias-image"]["family"] == "zimage"
    assert image_models["alias-image"]["source"] == "alias"
    assert image_models["local-image"]["size_label"] == "m"
    assert image_models["alias-image"]["size_label"] == "xl"
    assert "size" not in image_models["alias-image"]
    assert video_models["local-video"]["source"] == "installed"
    assert video_models["alias-video"]["family"] == "ltx"
    assert video_models["alias-video"]["source"] == "alias"
    assert video_models["alias-video"]["supports_i2v"] is False
    assert image_models["local-image@q8"]["stored_quant"] == {"base_model": "local-image", "bits": 8}
    assert image_models["local-image"]["stored_quant"] is None
    assert image_models["alias-image"]["stored_quant"] is None
    assert isinstance(payload["stored_quants_supported"], bool)


def test_submit_video_job_surfaces_platform_alias_mismatch(monkeypatch):
    """Video submissions should return the alias mismatch message from platform-aware resolution."""
    web_config = _make_web_config()
    web_config.app_config = {
        "model_aliases": {
            "ltx-2.3": {
                "darwin": {"message": "Alias 'ltx-2.3' is available on Windows and Linux only. On macOS, use 'ltx-4' or 'ltx-8'."},
                "win32": "dg845/LTX-2.3-Diffusers",
                "linux": "dg845/LTX-2.3-Diffusers",
            }
        },
        "video_generation": {"default_ratio": "16:9", "default_size": "m"},
        "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
        "video_model_presets": {"ltx": {"default_steps": 8}},
    }
    web_config.default_models = WebUiDefaultModels(image="zit", video="ltx-2.3")
    web_config.video_model_options = ("ltx-2.3",)

    monkeypatch.setattr(web_server.sys, "platform", "darwin")

    with pytest.raises(ValueError, match="available on Windows and Linux only"):
        web_server._submit_video_job({"model": "ltx-2.3", "prompt": "a lake"}, web_config)


def test_gallery_asset_id_contract_serves_and_deletes_relative_assets(monkeypatch, tmp_path):
    """Gallery list, media, and delete should share output-root-relative POSIX IDs."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())

    asset_path = tmp_path / "nested" / "asset one.png"
    _write_png(asset_path)
    sidecar_path = asset_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps({"prompt": "ignored relative asset", "workflow": "txt2img", "model": "zit"}), encoding="utf-8")

    asset_id = "nested/asset one.png"
    with TestClient(web_server.app) as client:
        gallery_response = client.get("/api/gallery")
        media_response = client.get(f"/media/{quote(asset_id, safe='/')}")
        absolute_delete_response = client.delete(f"/api/gallery/{quote(str(asset_path), safe='')}")
        delete_response = client.delete(f"/api/gallery/{quote(asset_id, safe='')}")

    assert gallery_response.status_code == 200
    payload = gallery_response.json()
    assert payload["total_count"] == 1
    listed_asset = payload["assets"][0]
    assert listed_asset["id"] == asset_id
    assert "path" not in listed_asset
    assert listed_asset["url"] == "/media/nested/asset%20one.png"
    assert not listed_asset["id"].startswith("/")

    assert media_response.status_code == 200
    assert absolute_delete_response.status_code == 404
    assert delete_response.status_code == 200
    assert not asset_path.exists()
    assert sidecar_path.exists()


def test_gallery_assets_without_model_metadata_do_not_reuse_media_kind_as_model(monkeypatch, tmp_path):
    """Missing model provenance should stay unavailable instead of falling back to media type labels."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

    image_path = tmp_path / "image-only.png"
    video_path = tmp_path / "video-only.mp4"
    _write_png(image_path)
    video_path.write_bytes(b"placeholder")
    image_path.with_suffix(".json").write_text(json.dumps({"prompt": "ignored", "model": "zit", "seed": 42, "steps": 12}), encoding="utf-8")

    with TestClient(web_server.app) as client:
        response = client.get("/api/gallery")

    assert response.status_code == 200
    payload = response.json()
    assets_by_id = {asset["id"]: asset for asset in payload["assets"]}
    assert set(assets_by_id) == {"image-only.png", "video-only.mp4"}

    for asset in assets_by_id.values():
        assert asset["model"] == "Unavailable"
        assert asset["reuse_state"]["requested_model"] is None
        assert asset["reuse_state"]["resolved_model"] is None
        assert asset["reuse_state"]["model_available"] is True
        assert "model_not_configured" not in asset["reuse_state"]["fallback_reasons"]
        assert "model=" not in asset["reuse_workspace_url"]
        assert "prompt=" not in asset["reuse_workspace_url"]
        assert "seed=" not in asset["reuse_workspace_url"]
        assert "steps=" not in asset["reuse_workspace_url"]


def test_gallery_plain_asset_display_prompt_does_not_reuse_generation_settings(monkeypatch, tmp_path):
    """Display-only prompt metadata should not be serialized as reusable generation config."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

    asset_path = tmp_path / "plain.png"
    _write_png_with_description(asset_path, "Display-only prompt")

    with TestClient(web_server.app) as client:
        response = client.get("/api/gallery")

    assert response.status_code == 200
    asset = response.json()["assets"][0]

    assert asset["prompt"] == "Display-only prompt"
    assert asset["has_reusable_config"] is False
    assert asset["reuse_workspace_url"] == "#/workspace?workflow=txt2img"
    assert "prompt=" not in asset["reuse_workspace_url"]


def test_gallery_assets_with_real_model_metadata_preserve_reuse_model(monkeypatch, tmp_path):
    """Configured model provenance should remain available and reusable."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

    asset_path = tmp_path / "with-model.png"
    _write_png_with_config(
        asset_path,
        {
            "schema": "zvisiongenerator.config.v1",
            "prompt": "with metadata",
            "workflow": "txt2img",
            "model": "zit",
        },
    )

    with TestClient(web_server.app) as client:
        response = client.get("/api/gallery")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 1
    asset = payload["assets"][0]

    assert asset["model"] == "zit"
    assert asset["has_reusable_config"] is True
    assert asset["reuse_state"]["requested_model"] == "zit"
    assert asset["reuse_state"]["resolved_model"] == "zit"
    assert asset["reuse_state"]["model_available"] is True
    assert "model_not_configured" not in asset["reuse_state"]["fallback_reasons"]
    assert "model=zit" in asset["reuse_workspace_url"]


def test_gallery_reuse_reads_embedded_png_config(monkeypatch, tmp_path):
    """Embedded PNG config should drive Gallery details and reuse URLs."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

    asset_path = tmp_path / "generated.png"
    _write_png_with_config(
        asset_path,
        {
            "schema": "zvisiongenerator.config.v1",
            "workflow": "img2img",
            "prompt": "original prompt",
            "model": "zit",
            "seed": 1234,
            "steps": 9,
            "guidance": 2.5,
            "width": 640,
            "height": 480,
            "ratio": "4:3",
            "size": "custom",
            "image_path": "/input/reference.png",
            "lora": "style.safetensors:0.8",
        },
    )

    with TestClient(web_server.app) as client:
        response = client.get("/api/gallery")

    assert response.status_code == 200
    asset = response.json()["assets"][0]

    assert asset["prompt"] == "original prompt"
    assert asset["has_reusable_config"] is True
    assert asset["model"] == "zit"
    assert asset["width"] == 640
    assert asset["height"] == 480
    assert asset["ratio"] == "4:3"
    assert asset["size"] == "custom"
    assert asset["image_path"] == "/input/reference.png"
    assert asset["file_path"] == str(asset_path.resolve())
    assert asset["seed"] == 1234
    assert asset["steps"] == 9
    assert asset["guidance"] == 2.5
    assert asset["lora"] == "style.safetensors:0.8"
    assert asset["reuse_state"]["resolved_model"] == "zit"
    assert "workflow=img2img" in asset["reuse_workspace_url"]
    assert "prompt=original+prompt" in asset["reuse_workspace_url"]
    assert "model=zit" in asset["reuse_workspace_url"]
    assert "seed=1234" in asset["reuse_workspace_url"]
    assert "steps=9" in asset["reuse_workspace_url"]
    assert "guidance=2.5" in asset["reuse_workspace_url"]
    assert "image_path=%2Finput%2Freference.png" in asset["reuse_workspace_url"]


def test_media_and_delete_reject_invalid_asset_ids(monkeypatch, tmp_path):
    """Media and delete routes should reject traversal, absolute, and staging asset IDs."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())

    valid_asset = tmp_path / "nested" / "asset.png"
    _write_png(valid_asset)

    invalid_ids = [
        "../nested/asset.png",
        "/nested/asset.png",
        "nested\\asset.png",
        "C:/nested/asset.png",
        ".web_uploads/reference.png",
    ]

    with TestClient(web_server.app) as client:
        for asset_id in invalid_ids:
            media_response = client.get(f"/media/{quote(asset_id, safe='')}")
            delete_response = client.delete(f"/api/gallery/{quote(asset_id, safe='')}")

            assert media_response.status_code == 404, asset_id
            assert delete_response.status_code == 404, asset_id

    assert valid_asset.exists()


def test_gallery_and_history_exclude_reference_upload_staging(monkeypatch, tmp_path):
    """Temporary Web upload staging should not appear in user-visible gallery/history inventory."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: _make_workspace_bootstrap_view())

    visible_asset = tmp_path / "published.png"
    staged_asset = tmp_path / ".web_uploads" / "reference.png"
    _write_png(visible_asset)
    _write_png(staged_asset)

    with TestClient(web_server.app) as client:
        gallery_response = client.get("/api/gallery")
        history_response = client.get("/api/history")
        workspace_response = client.get("/api/workspace")
        staged_media_response = client.get("/media/.web_uploads/reference.png")

    assert gallery_response.status_code == 200
    assert history_response.status_code == 200
    assert workspace_response.status_code == 200
    assert staged_media_response.status_code == 404

    gallery_ids = [asset["id"] for asset in gallery_response.json()["assets"]]
    history_ids = [asset["id"] for asset in history_response.json()["assets"]]
    workspace_history_ids = [asset["id"] for asset in workspace_response.json()["history_assets"]]
    assert gallery_ids == ["published.png"]
    assert history_ids == ["published.png"]
    assert workspace_history_ids == ["published.png"]
    assert [asset.id for asset in list_gallery_assets(str(tmp_path))] == ["published.png"]


def test_reference_upload_staging_still_accepts_generation_uploads(tmp_path):
    """Reference uploads should still be staged for generation while staying out of inventory."""
    uploaded_file = SimpleNamespace(filename="reference.png", file=_png_upload_bytes())

    staged_path = Path(web_server._save_uploaded_reference_image(uploaded_file, str(tmp_path)))

    assert staged_path.is_file()
    assert staged_path.parent.name == ".web_uploads"
    assert list_gallery_assets(str(tmp_path)) == []


@pytest.mark.parametrize("selected", [["portrait:1", "landscape:0", "portrait:0", "portrait:1"], ["portrait:0"]])
def test_prompt_file_submission_batches_checked_prompts_in_file_order(tmp_path, selected):
    from starlette.datastructures import FormData

    path = tmp_path / "prompts.yaml"
    path.write_text("portrait:\n  - prompt: first\n    negative: blur\n  - prompt: second\nlandscape:\n  - prompt: third\n  - prompt: unchecked\n", encoding="utf-8")
    form = FormData([("prompt_source", "file"), ("prompts_file", str(path)), *[("prompt_option_id", option_id) for option_id in selected]])
    source, prompt, negative, batch, _enhance = web_server._resolve_prompt_submission_with_enhance(form)
    assert source == "file"
    assert prompt == "first"
    assert negative == "blur"
    expected = {"portrait": [("first", "blur")]}
    if len(selected) > 1:
        expected["portrait"].append(("second", None))
        expected["landscape"] = [("third", None)]
    assert batch == expected


@pytest.mark.parametrize("selected", [[], ["portrait:1"], ["portrait:0", "missing:0"]])
def test_prompt_file_submission_rejects_empty_inactive_or_stale_selection(tmp_path, selected):
    from starlette.datastructures import FormData

    path = tmp_path / "prompts.yaml"
    path.write_text("portrait:\n  - prompt: first\n  - prompt: inactive\n    active: false\n", encoding="utf-8")
    form = FormData([("prompt_source", "file"), ("prompts_file", str(path)), *[("prompt_option_id", option_id) for option_id in selected]])
    with pytest.raises(ValueError, match="Select at least one|missing or inactive"):
        web_server._resolve_prompt_submission_with_enhance(form)


@pytest.mark.parametrize(("preset_upscale_steps", "expected"), [(3, 3), (None, 5)])
def test_submit_image_job_uses_preset_upscale_steps_default(monkeypatch, tmp_path, preset_upscale_steps, expected):
    """Blank upscale steps should follow the preset default like the CLI, falling back to steps // 2."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    defaults = _make_resolved_image_defaults()
    defaults["upscale_steps"] = preset_upscale_steps
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=defaults, submitted=submitted)

    web_server._submit_image_job({"model": "zit", "prompt": "hello", "upscale": "2"}, web_config)

    assert submitted[0]["args"].upscale_steps == expected


@pytest.mark.parametrize(
    ("lora", "message"),
    [
        ("some-org/some-lora:0.8", "Remote HuggingFace LoRA references are not supported"),
        ("/definitely/missing/lora.safetensors", "LoRA file not found"),
        ("unknown-bare-name", "LoRA file not found"),
    ],
)
def test_submit_image_job_rejects_unloadable_loras(monkeypatch, tmp_path, lora, message):
    """Web submits should reject remote and missing LoRAs up front, matching the CLI."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=_make_resolved_image_defaults(), submitted=submitted)

    with pytest.raises(ValueError, match=message):
        web_server._submit_image_job({"model": "zit", "prompt": "hello", "lora": lora}, web_config)

    assert submitted == []


def test_submit_image_job_accepts_existing_lora_file(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    lora_file = tmp_path / "style.safetensors"
    lora_file.write_bytes(b"")
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=_make_resolved_image_defaults(), submitted=submitted)

    web_server._submit_image_job({"model": "zit", "prompt": "hello", "lora": f"{lora_file}:0.7"}, web_config)

    assert submitted[0]["args"].lora_paths == [str(lora_file)]
    assert submitted[0]["args"].lora_weights == [0.7]


def test_media_only_serves_gallery_media_types(monkeypatch, tmp_path):
    """/media must not serve arbitrary files under the output root (e.g. keys or configs)."""
    (tmp_path / "id_rsa").write_text("secret", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    _write_png(tmp_path / "ok.png")
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())

    with TestClient(web_server.app) as client:
        assert client.get("/media/id_rsa").status_code == 404
        assert client.get("/media/notes.json").status_code == 404
        assert client.get("/media/ok.png").status_code == 200


def test_picker_macos_passes_initial_path_as_argv_not_script_source(monkeypatch, tmp_path):
    """A hostile directory name must reach osascript as data, never as AppleScript source."""
    hostile = tmp_path / 'a\\" & (do shell script "echo INJECTED") --'
    hostile.mkdir()
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=str(hostile) + "/\n", stderr="")

    monkeypatch.setattr(path_picker_module.subprocess, "run", _fake_run)

    selected = path_picker_module._pick_macos("directory", str(hostile), path_picker_module._PICKER_PURPOSES["output_directory"])

    assert selected == str(hostile) + "/"
    cmd = calls[0]
    script = cmd[2]
    assert "INJECTED" not in script
    assert "item 1 of argv" in script
    assert cmd[3:] == [str(hostile.resolve())]


def test_media_output_root_is_cached_until_user_config_changes(monkeypatch, tmp_path):
    import yaml

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("ZIV_DATA_DIR", str(data_dir))
    monkeypatch.setattr(web_server, "_media_output_root_cache", None)
    (data_dir / "config.yaml").write_text(yaml.safe_dump({"ui": {"output_dir": str(tmp_path / "first")}}), encoding="utf-8")
    loads: list[int] = []
    real_load_config = web_server.load_config
    monkeypatch.setattr(web_server, "load_config", lambda: loads.append(1) or real_load_config())

    assert web_server._media_output_root() == (tmp_path / "first").resolve()
    assert web_server._media_output_root() == (tmp_path / "first").resolve()
    assert len(loads) == 1

    config_contract_module.write_user_config_override({"ui": {"output_dir": str(tmp_path / "second")}})
    assert web_server._media_output_root() == (tmp_path / "second").resolve()
    assert len(loads) == 2


def test_picker_allows_only_one_dialog_at_a_time(monkeypatch):
    """Concurrent picker requests (worker threads) must not open a second native dialog; Tk is not thread-safe."""
    import threading

    opened = threading.Event()
    release = threading.Event()

    def _slow_tk(kind, initial_path, picker_purpose):
        opened.set()
        release.wait(timeout=2.0)
        return None

    monkeypatch.setattr(path_picker_module.sys, "platform", "linux")
    monkeypatch.setattr(path_picker_module, "_pick_tk", _slow_tk)
    results: list[object] = []
    first = threading.Thread(target=lambda: results.append(path_picker_module.pick_path("directory", purpose="output_directory")))
    first.start()
    assert opened.wait(timeout=2.0)

    second = path_picker_module.pick_path("directory", purpose="output_directory")
    release.set()
    first.join(timeout=2.0)

    assert second.status == "error"
    assert "already open" in (second.message or "")
    assert results[0].status == "cancelled"
    assert path_picker_module.pick_path("directory", purpose="output_directory").status == "cancelled"


@pytest.mark.parametrize(
    ("completed", "expected_status", "expected_path"),
    [
        (SimpleNamespace(returncode=0, stdout="/picked/dir\n", stderr=""), "selected", "/picked/dir"),
        (SimpleNamespace(returncode=0, stdout="", stderr=""), "cancelled", None),
        (SimpleNamespace(returncode=1, stdout="", stderr="Traceback...\nModuleNotFoundError: No module named 'tkinter'"), "unsupported", None),
        (SimpleNamespace(returncode=1, stdout="", stderr="Traceback...\n_tkinter.TclError: no display name and no $DISPLAY environment variable"), "error", None),
    ],
)
def test_tk_picker_runs_in_child_process(monkeypatch, tmp_path, completed, expected_status, expected_path):
    """On Linux/Windows the Tk dialog runs in a child Python so Tk owns that process's main thread."""
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return completed

    monkeypatch.setattr(path_picker_module.sys, "platform", "linux")
    monkeypatch.setattr(path_picker_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(path_picker_module, "_validate_selected_path", lambda _path, _purpose: None)

    result = path_picker_module.pick_path("directory", purpose="output_directory", initial_path=str(tmp_path))

    cmd = calls[0]
    assert cmd[0] == path_picker_module.sys.executable and cmd[1] == "-c"
    assert cmd[3:5] == ["directory", str(tmp_path.resolve())]
    assert result.status == expected_status
    if expected_path is not None:
        assert result.path == str(Path(expected_path).resolve())
    if expected_status == "error":
        assert "no display" in (result.message or "")


def test_model_listings_carry_download_and_memory_status(monkeypatch, tmp_path):
    """Workspace and models payloads expose per-model download and memory-fit status from one source."""
    web_config = _make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.image_inventory = (
        ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),
        ImageInventoryEntry(name="local-image", family="ideogram4", size=None, source="installed", resolved_path=str(tmp_path / "local-image")),
    )
    web_config.video_inventory = (VideoInventoryEntry(name="ltx-8", family="ltx", supports_i2v=True, source="alias", resolved_path="owner/ltx"),)
    web_config.app_config["model_presets"] = {"ideogram4": {"supports_quantize": False}}
    calls: list[tuple[str, str, tuple[int, ...]]] = []

    def _fake_status(resolved_path, *, kind, quantize_options=(), budget_bytes=None, find_local_dir=None):
        calls.append((resolved_path, kind, quantize_options))
        return {"downloaded": resolved_path != "owner/ltx", "memory_fit": None}

    monkeypatch.setattr(workspace_api_module, "describe_model_status", _fake_status)
    monkeypatch.setattr(workspace_api_module, "memory_budget_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
    bootstrap_view = _make_workspace_bootstrap_view()
    bootstrap_view["image_model_defaults"]["zit"]["supports_quantize"] = False  # the picker hides quantize for zit

    workspace = workspace_api_module.build_workspace_response(
        web_config,
        [],
        active_job=None,
        prompt_sources=["inline"],
        default_prompt_source="inline",
        prompt_file_contract={},
        workflow_contract={},
        build_bootstrap_view=lambda _cfg: bootstrap_view,
    )
    workspace_calls = list(calls)
    models = workspace_api_module.build_models_response(
        web_config,
        token_var=None,
        image_defaults_for=lambda name, _cfg: bootstrap_view["image_model_defaults"].get(name, {}),
    )

    workspace_images = {entry["id"]: entry for entry in workspace["image_models"]}
    assert workspace_images["zit"]["downloaded"] is True
    assert workspace["video_models"][0]["downloaded"] is False
    assert {entry["name"]: entry["downloaded"] for entry in models["video_models"]} == {"ltx-8": False}
    assert all("memory_fit" in entry for entry in [*workspace["image_models"], *models["image_models"]])
    # Both pages estimate exactly the quantize levels the workspace picker offers (from the bootstrap defaults).
    assert ("owner/zit", "image", ()) in workspace_calls
    assert calls[len(workspace_calls) :] == workspace_calls
    assert ("owner/ltx", "video", ()) in calls


def test_workspace_models_without_inventory_entry_report_unknown_status(monkeypatch):
    """Names with no inventory entry (e.g. stale config) report unknown rather than guessing."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "memory_budget_bytes", lambda: None)

    workspace = workspace_api_module.build_workspace_response(
        web_config,
        [],
        active_job=None,
        prompt_sources=["inline"],
        default_prompt_source="inline",
        prompt_file_contract={},
        workflow_contract={},
        build_bootstrap_view=lambda _cfg: _make_workspace_bootstrap_view(),
    )

    assert {(entry["downloaded"], entry["memory_fit"]) for entry in workspace["image_models"]} == {(None, None)}


def test_one_failing_model_status_does_not_break_the_listing(monkeypatch):
    """A model whose files cannot be inspected reports unknown status instead of failing the whole page."""
    web_config = _make_web_config()
    web_config.image_inventory = (
        ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),
        ImageInventoryEntry(name="local-image", family="zimage", size=None, source="installed", resolved_path="/broken"),
    )

    def _fake_status(resolved_path, **_):
        if resolved_path == "/broken":
            raise AttributeError("corrupt header")
        return {"downloaded": True, "memory_fit": None}

    monkeypatch.setattr(workspace_api_module, "describe_model_status", _fake_status)
    monkeypatch.setattr(workspace_api_module, "memory_budget_bytes", lambda: None)

    with pytest.warns(UserWarning, match="local-image"):
        workspace = workspace_api_module.build_workspace_response(
            web_config,
            [],
            active_job=None,
            prompt_sources=["inline"],
            default_prompt_source="inline",
            prompt_file_contract={},
            workflow_contract={},
            build_bootstrap_view=lambda _cfg: _make_workspace_bootstrap_view(),
        )

    statuses = {entry["id"]: entry["downloaded"] for entry in workspace["image_models"]}
    assert statuses == {"zit": True, "local-image": None}


def test_models_page_skips_quantize_resolution_without_a_memory_budget(monkeypatch):
    """Off macOS there is no budget, so per-model defaults (and their detection lookups) are not resolved."""
    web_config = _make_web_config()
    web_config.image_inventory = (ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),)
    monkeypatch.setattr(workspace_api_module, "memory_budget_bytes", lambda: None)
    monkeypatch.setattr(workspace_api_module, "describe_model_status", lambda resolved_path, **_: {"downloaded": True, "memory_fit": None})
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])

    models = workspace_api_module.build_models_response(
        web_config,
        token_var=None,
        image_defaults_for=lambda name, _cfg: pytest.fail("resolved image defaults without a memory budget"),
    )

    assert models["image_models"][0]["downloaded"] is True


@pytest.mark.parametrize(("quantize", "expected_tail"), [("", []), ("8", ["--quantize", "8"])])
def test_convert_route_forwards_an_optional_quantized_copy(monkeypatch, tmp_path, quantize, expected_tail):
    """The converter route passes the quantized-copy level to ziv-model only when one is chosen."""
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    captured: list[list[str]] = []
    monkeypatch.setattr(web_server, "_run_model_management_command", lambda args: captured.append(args) or "done")

    with TestClient(web_server.app) as client:
        response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "flux2-klein-9b", "quantize": quantize})

    assert response.status_code == 200
    assert captured[0] == ["model", "--input", str(checkpoint.resolve()), "--model-type", "flux2-klein-9b", *expected_tail]


def test_convert_route_rejects_unsupported_quantize_levels(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    monkeypatch.setattr(web_server, "_run_model_management_command", lambda args: pytest.fail("converter should not run"))

    with TestClient(web_server.app) as client:
        response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "flux2-klein-9b", "quantize": "6"})

    assert response.status_code in (400, 422)
