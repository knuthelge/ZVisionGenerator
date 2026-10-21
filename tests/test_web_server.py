"""Focused route tests for the Phase A web backend contracts."""

from __future__ import annotations

import contextlib
import functools
from dataclasses import replace
import inspect
import io
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.testclient import TestClient
import huggingface_hub
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest
from conftest import make_probe

from zvisiongenerator.downloads.space import SpaceNeed
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import model_download_plan as download_plan_module
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.model_download_plan import plan_summary as real_plan_summary
from zvisiongenerator.web import extend_api as extend_api_module
from zvisiongenerator.web import model_inventory as model_inventory_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import video_clip as video_clip_module
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.gallery import list_gallery_assets
from zvisiongenerator.web.job_contract import public_job_snapshot
from zvisiongenerator.web.job_log import JOB_LOG_STATUS_FILTERS, JobLog
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.ltx_converted import write_converted_ltx
from zvisiongenerator.utils import video_extend as video_extend_module
from zvisiongenerator.seedvr2_loader import seedvr2_weights_dir
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.model_memory import MemoryBudget, clip_memory_fit as real_clip_memory_fit
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
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
        seedvr2=None,
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
        "postprocess": {"sharpen": True, "contrast": False, "saturation": False},
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
        "frame_alignment": 8,
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
    supports_scheduler: bool = True,
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
        "supports_scheduler": supports_scheduler,
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
    monkeypatch.setattr(web_server.web_runner, "list_jobs", lambda: {"active_job": active_job_snapshot, "queued_jobs": []})
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
    assert workspace_payload["queued_jobs"] == []
    assert prompt_file["accepted_extensions"] == [".yaml", ".yml"]
    assert prompt_file["selection_required"] is True
    assert prompt_file["trust_boundary"]["scope"] == "server_host_only"
    assert prompt_file["trust_boundary"]["manual_entry"] == "submitted_value_kept_until_backend_validation"
    assert prompt_file["trust_boundary"]["read_write"] == "yaml_files_only"
    assert set(prompt_file["help"]) == {
        "path",
        "option_required",
        "option_optional",
        "empty_options",
        "stale_selection",
        "loaded",
        "ignored_negative_video",
        "ignored_negative_unsupported",
        "ignored_scene_image",
    }
    assert all(isinstance(value, str) and value for value in prompt_file["help"].values())


class TestHuggingFaceTokenPayload:
    """``/api/config`` and ``/api/models`` report the token ``huggingface_hub`` finds and where it comes from."""

    @staticmethod
    def _payloads(monkeypatch):
        web_config = _make_web_config()
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        with TestClient(web_server.app) as client:
            config = client.get("/api/config")
            models = client.get("/api/models")
        assert (config.status_code, models.status_code) == (200, 200)
        return config.json()["ui"], models.json()

    def test_no_token_is_not_configured(self, monkeypatch):
        ui, models = self._payloads(monkeypatch)

        assert (ui["huggingface_token_configured"], ui["huggingface_token_source"]) == (False, None)
        assert (models["huggingface_configured"], models["huggingface_token_source"]) == (False, None)

    def test_a_saved_token_is_configured_with_the_login_source(self, monkeypatch):
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "saved")

        ui, models = self._payloads(monkeypatch)

        assert (ui["huggingface_token_configured"], ui["huggingface_token_source"]) == (True, "hf auth login")
        assert (models["huggingface_configured"], models["huggingface_token_source"]) == (True, "hf auth login")

    def test_an_environment_token_names_its_variable(self, monkeypatch):
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "from-env")
        monkeypatch.setenv("HF_TOKEN", "from-env")

        ui, models = self._payloads(monkeypatch)

        assert ui["huggingface_token_source"] == models["huggingface_token_source"] == "HF_TOKEN"

    def test_a_token_reader_that_raises_reads_as_not_configured(self, monkeypatch):
        def broken() -> str:
            raise RuntimeError("OIDC unavailable")

        monkeypatch.setattr(huggingface_hub, "get_token", broken)

        ui, models = self._payloads(monkeypatch)

        assert ui["huggingface_token_configured"] is False
        assert models["huggingface_configured"] is False


def test_workspace_route_can_skip_history_asset_serialization(monkeypatch):
    """Initial workspace hydration should be able to avoid gallery inventory work."""
    web_config = _make_web_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: _make_workspace_bootstrap_view())
    monkeypatch.setattr(web_server.web_runner, "list_jobs", lambda: {"active_job": None, "queued_jobs": []})

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
    assert response["context"]["output_dir"] == str(tmp_path)
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


@pytest.mark.parametrize(("family", "alignment"), [("ltx", 8), ("unknown", 1)])
def test_video_bootstrap_defaults_include_frame_alignment(monkeypatch, family, alignment):
    """Video bootstrap defaults should expose the model's frame alignment so the UI can snap frame counts."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        workspace_api_module,
        "detect_video_model",
        lambda _model: VideoModelInfo(family=family, backend=family, supports_i2v=True, default_fps=24, frame_alignment=alignment, resolution_alignment=32),
    )

    defaults = workspace_api_module._build_video_bootstrap_defaults("ltx-8", web_config)

    assert defaults["frame_alignment"] == alignment


@pytest.mark.parametrize(("detected", "expected"), [("ltx", "ltx"), ("unknown", None)])
def test_video_bootstrap_defaults_include_detected_family(monkeypatch, detected, expected):
    """Video defaults name the detected family so the Workspace can remember LoRAs per family."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        workspace_api_module,
        "detect_video_model",
        lambda _model: VideoModelInfo(family=detected, backend=detected, supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
    )

    assert workspace_api_module._build_video_bootstrap_defaults("ltx-8", web_config)["family"] == expected


def test_image_bootstrap_defaults_include_detected_family(monkeypatch):
    """Image defaults name the detected family, and none when detection fails."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", lambda *_: _make_resolved_image_defaults())
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda _model: ImageModelInfo(family="zimage", is_distilled=True, size=None))

    assert workspace_api_module._build_image_bootstrap_defaults("zit", web_config)["family"] == "zimage"

    def _fail(_model):
        raise RuntimeError("no model")

    monkeypatch.setattr(workspace_api_module, "detect_image_model", _fail)

    assert workspace_api_module._build_image_bootstrap_defaults("missing", web_config)["family"] is None


def test_video_bootstrap_defaults_fall_back_to_unaligned_frames_when_detection_fails(monkeypatch):
    """A model that cannot be detected should leave frame counts unconstrained."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)

    def _fail(_model):
        raise RuntimeError("no model")

    monkeypatch.setattr(workspace_api_module, "detect_video_model", _fail)

    defaults = workspace_api_module._build_video_bootstrap_defaults("missing", web_config)

    assert defaults["frame_alignment"] == 1


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


def test_workspace_bootstrap_defaults_include_scheduler_capability(monkeypatch):
    web_config = _make_web_config()
    web_config.app_config["model_aliases"] = {"krea2": "unsloth/Krea-2-Turbo", "zit": "Tongyi-MAI/Z-Image-Turbo"}
    web_config.image_model_options = ("krea2", "zit")
    families = {"unsloth/Krea-2-Turbo": "krea2", "Tongyi-MAI/Z-Image-Turbo": "zimage"}

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda value: ImageModelInfo(family=families[str(value)], is_distilled=True, size=None))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(
        workspace_api_module,
        "resolve_defaults",
        lambda model_info, *_: _make_resolved_image_defaults(supports_scheduler=model_info.family != "krea2"),
    )

    assert workspace_api_module._build_image_bootstrap_defaults("krea2", web_config)["supports_scheduler"] is False
    assert workspace_api_module._build_image_bootstrap_defaults("zit", web_config)["supports_scheduler"] is True


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


def _submit_enhanced_image(monkeypatch, tmp_path, record: str) -> list[dict[str, object]]:
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=True, size=None), defaults=_make_resolved_image_defaults(), submitted=submitted)
    form = {"model": "zit", "prompt": "A red fox curls up in deep snow.", "prompt_input": "a fox in snow", "enhance_record": record}
    web_server._submit_image_job(form, web_config)
    return submitted


def test_submit_image_job_records_the_web_rewrite(monkeypatch, tmp_path):
    record = json.dumps({"settings": {"style": "photo"}, "model": "someone/custom-model", "tier": "smart", "format": "text"})
    args = _submit_enhanced_image(monkeypatch, tmp_path, record)[0]["args"]
    assert args.prompt_input == "a fox in snow"
    assert args.enhance_record["settings"]["style"] == "photo"
    assert args.enhance_record["tier"] == "custom"  # worked out from the model, not taken from the browser


@pytest.mark.parametrize("record", ['{"settings": {"style": "nope"}, "model": "org/fast"}', "{broken"])
def test_submit_image_job_drops_an_invalid_enhance_record(monkeypatch, tmp_path, record):
    with pytest.warns(UserWarning, match="enhance record"):
        args = _submit_enhanced_image(monkeypatch, tmp_path, record)[0]["args"]
    assert args.enhance_record is None
    assert args.prompt_input == "a fox in snow"


def test_submit_image_job_compacts_a_pretty_json_prompt(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_json_prompt=True, supports_first_sigma=True),
        submitted=submitted,
    )
    web_server._submit_image_job({"model": "ideo", "json_prompt": '{\n  "high_level_description": "Café"\n}'}, web_config)
    assert submitted[0]["prompts_data"] == {"prompt": [('{"high_level_description":"Café"}', None)]}


def test_submit_image_job_rejects_json_enhancement_for_models_without_json_captions(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_json_prompt=False),
    )
    with pytest.raises(ValueError, match="does not support structured JSON captions"):
        web_server._submit_image_job({"model": "zimage", "prompt": "a fox", "enhance_auto": "true", "enhance_format": "json"}, web_config)


@pytest.mark.parametrize(
    ("form", "expected_steps", "expected_guidance", "expected_steps_explicit", "expected_guidance_explicit"),
    [
        ({"model": "ideo", "prompt": "hello", "steps": "28", "guidance": "6.0"}, 28, 6.0, True, True),
        ({"model": "ideo", "prompt": "hello"}, 10, 3.5, False, False),
        ({"model": "ideo", "prompt": "hello", "steps": "10", "guidance": "3.5"}, 10, 3.5, False, False),
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


def test_submit_image_job_drops_scheduler_for_models_with_their_own_sampler(monkeypatch, tmp_path):
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="krea2", is_distilled=False, size=None),
        defaults=_make_resolved_image_defaults(supports_scheduler=False),
        submitted=submitted,
    )
    monkeypatch.setattr(web_server, "resolve_defaults", lambda _info, _config, cli_overrides, _backend: {**_make_resolved_image_defaults(supports_scheduler=False), **cli_overrides})

    web_server._submit_image_job({"model": "krea2", "prompt": "hello", "scheduler": "beta"}, web_config)

    assert submitted[0]["args"].scheduler is None
    assert submitted[0]["request"].scheduler is None


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


def _public_context(*, runs: int = 1, meta: str = "") -> dict[str, object]:
    return {
        "workflow": "txt2img",
        "job_type": "Text to Image",
        "prompt": "prompt",
        "model": "zit",
        "runs": runs,
        "meta": meta,
        "created_at": "2026-10-04T10:00:00+00:00",
        "output_dir": "/tmp/outputs",
        "settings": {},
    }


def test_generate_route_returns_requested_runs_from_job_context(monkeypatch):
    """The public generate response should return the queued job's requested runs value."""
    monkeypatch.setattr(web_server, "load_web_config", _make_web_config)
    monkeypatch.setattr(
        web_server,
        "_submit_image_job",
        lambda _form, _web_config: {
            "job_id": "job-123",
            "context": _public_context(runs=7, meta="2:3 · m · 10 steps"),
            "supported_controls": ("next", "pause"),
        },
    )

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "image"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-123"
    assert payload["runs"] == 7
    assert payload["supported_controls"] == ["next", "pause"]
    assert payload["meta"] == "2:3 · m · 10 steps"
    assert "queue_position" in payload


def test_video_submission_missing_ffmpeg_returns_422_before_job_registration(monkeypatch):
    """The Web UI rejects missing ffmpeg as validation without entering a worker/install flow."""
    web_config = _make_web_config()
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(video_clip_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        video_clip_module,
        "detect_video_model",
        lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=64, frame_alignment=8, default_fps=24),
    )
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg is missing; install it and retry")))
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "video", "model": "ltx-8", "prompt": "a lake"})

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert "ffmpeg" in response.json()["detail"].lower()
    assert "install" in response.json()["detail"].lower()
    assert submitted == []


def _stub_video_submission(monkeypatch, tmp_path) -> list[dict[str, object]]:
    """Stub model detection and the runner so a video submission reaches ``submit_video_request_job`` and records its kwargs."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(video_clip_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        video_clip_module,
        "detect_video_model",
        lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=64, frame_alignment=8, default_fps=24),
    )
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")
    return submitted


def _post_video(data: dict[str, object]):
    with TestClient(web_server.app) as client:
        return client.post("/api/generate", data={"mode": "video", "model": "ltx-8", **data})


def test_video_submission_threads_scene_into_args_and_prompts(monkeypatch, tmp_path):
    """A scene and prompt reach the job args, with the user's inputs recorded when sent."""
    submitted = _stub_video_submission(monkeypatch, tmp_path)

    response = _post_video({"scene": " A castle at dusk. ", "prompt": "A knight rides in.", "scene_input": "castle", "prompt_input": "knight"})

    assert response.status_code == 200
    (job,) = submitted
    args = job["args"]
    assert (args.scene, args.prompt, args.scene_input, args.prompt_input) == ("A castle at dusk.", "A knight rides in.", "castle", "knight")
    assert job["prompts_data"] == {"web": [("A knight rides in.", None)]}
    assert job["scene_by_set"] is None


def test_text_to_video_submission_without_scene_keeps_the_scene_and_extend_args_empty(monkeypatch, tmp_path):
    submitted = _stub_video_submission(monkeypatch, tmp_path)

    response = _post_video({"prompt": "A knight rides in."})

    assert response.status_code == 200
    args = submitted[0]["args"]
    assert (args.scene, args.scene_input, args.prompt_input) == (None, None, None)
    assert not hasattr(args, "extend_source")
    assert submitted[0]["context"]["job_type"] == "Text to Video"


def test_video_submission_accepts_a_scene_without_a_prompt(monkeypatch, tmp_path):
    """Scene OR prompt is enough: a scene-only inline submission plans one empty-prompt entry."""
    submitted = _stub_video_submission(monkeypatch, tmp_path)

    response = _post_video({"scene": "A castle at dusk."})

    assert response.status_code == 200
    (job,) = submitted
    assert job["args"].scene == "A castle at dusk."
    assert job["args"].prompt == ""
    assert job["prompts_data"] == {"web": [("", None)]}
    assert job["context"]["prompt"] == "A castle at dusk."


@pytest.mark.parametrize("data", [{}, {"scene": "   ", "prompt": ""}, {"prompt": "  "}])
def test_video_submission_refuses_when_scene_and_prompt_are_both_empty(monkeypatch, tmp_path, data):
    submitted = _stub_video_submission(monkeypatch, tmp_path)

    response = _post_video(data)

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert "prompt" in response.json()["detail"]
    assert submitted == []


def test_image_submission_ignores_the_scene_field(monkeypatch, tmp_path):
    """Images keep the prompt requirement and never read a scene."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(web_server, "detect_image_model", lambda _model: ImageModelInfo(family="zimage", is_distilled=False, size="xl"))
    monkeypatch.setattr(web_server, "get_backend_name", lambda: "registry-owned")
    monkeypatch.setattr(web_server, "validate_scheduler", lambda _scheduler, _config: None)
    monkeypatch.setattr(web_server, "resolve_defaults", lambda *_args: {"steps": 11, "guidance": 4.0, "scheduler": None, "supports_negative_prompt": True})
    monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")

    with pytest.raises(ValueError, match="prompt"):
        web_server._submit_image_job({"scene": "A castle at dusk."}, web_config)
    web_server._submit_image_job({"prompt": "hello", "scene": "A castle at dusk."}, web_config)

    (job,) = submitted
    assert not hasattr(job["args"], "scene")
    assert job["prompts_data"] == {"web": [("hello", None)]}


def test_prompt_file_video_submission_returns_scene_by_set_and_job_scene_default(monkeypatch, tmp_path):
    """File entries keep their own ``scene:``; the form scene is the default for the others."""
    submitted = _stub_video_submission(monkeypatch, tmp_path)
    path = tmp_path / "prompts.yaml"
    path.write_text("shots:\n  - scene: A castle.\n    prompt: first\n  - prompt: second\n", encoding="utf-8")

    response = _post_video({"prompt_source": "file", "prompts_file": str(path), "prompt_option_id": ["shots:0", "shots:1"], "scene": "Night."})

    assert response.status_code == 200
    (job,) = submitted
    assert job["scene_by_set"] == {"shots": ["A castle.", None]}
    assert job["args"].scene == "Night."
    assert job["prompts_data"] == {"shots": [("first", None), ("second", None)]}


def test_prompt_file_options_payload_carries_scene_preview_and_flag(tmp_path):
    """The options payload adds ``scene_preview`` and ``has_scene`` per option and keeps its existing keys."""
    path = tmp_path / "prompts.yaml"
    path.write_text("shots:\n  - scene: A castle.\n    prompt: first\n  - prompt: second\n", encoding="utf-8")

    with TestClient(web_server.app) as client:
        response = client.post("/api/prompt-files/inspect", json={"path": str(path)})

    assert response.status_code == 200
    with_scene, without_scene = response.json()["options"]
    assert (with_scene["scene_preview"], with_scene["has_scene"]) == ("A castle.", True)
    assert (without_scene["scene_preview"], without_scene["has_scene"]) == (None, False)
    assert {"id", "set_name", "source_index", "label", "prompt_preview", "negative_preview", "enhance"} <= set(with_scene)
    assert with_scene["prompt_preview"] == "first"


def test_image_submission_does_not_require_ffmpeg(monkeypatch):
    """Image submissions retain their existing path when the video prerequisite is unavailable."""
    monkeypatch.setattr(web_server, "load_web_config", _make_web_config)
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: (_ for _ in ()).throw(AssertionError("image submissions must not check ffmpeg")))
    monkeypatch.setattr(
        web_server,
        "_submit_image_job",
        lambda _form, _web_config: {
            "job_id": "job-123",
            "context": _public_context(),
            "supported_controls": (),
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
    web_config.app_config["model_labels"] = {"alias-image": {"name": "Alias Image"}, "alias-video": {"name": "Alias Video"}}
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(
        model_inventory_module,
        "list_models",
        lambda _data_dir: [SimpleNamespace(name="local-image", family="zimage", size="m"), SimpleNamespace(name="local-image@q8", family="zimage", size="m")],
    )
    monkeypatch.setattr(
        model_inventory_module,
        "list_video_models",
        lambda _data_dir: [SimpleNamespace(name="local-video", family="ltx", supports_i2v=True), SimpleNamespace(name="converted-video", family="ltx", supports_i2v=True)],
    )
    converted = tmp_path / "models" / "converted-video"
    converted.mkdir(parents=True)
    write_converted_ltx(converted, {"bits": 8, "storage": "mlx-affine-g64", "block_matrices": 1, "transformer_rest": 1})
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
    assert image_models["alias-image"]["label"] == "Alias Image"
    assert image_models["local-image"]["label"] == "local-image"
    assert video_models["alias-video"]["label"] == "Alias Video"
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
    assert video_models["converted-video"]["converted_bits"] == 8
    assert video_models["local-video"]["converted_bits"] is None
    assert video_models["alias-video"]["converted_bits"] is None
    assert isinstance(payload["stored_quants_supported"], bool)


_SEEDVR2_7B = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)


def _models_route_with_seedvr2(monkeypatch, tmp_path, *, copy_files: dict[str, bytes] | None):
    """Serve ``GET /api/models`` with SeedVR2 on, its 7B copy holding *copy_files* (or absent when ``None``)."""
    copy = seedvr2_weights_dir("seedvr2-7b", tmp_path / "models", MLX_WEIGHTS)
    if copy_files is not None:
        for name, content in copy_files.items():
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            (copy / name).write_bytes(content)
    settings = SeedVR2Settings(model="auto", softness=0.75, reference_megapixels=16, tiers=(_SEEDVR2_7B,))
    web_config = _make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.models_dir = str(tmp_path / "models")
    web_config.seedvr2 = SeedVR2Setup(settings=settings, tier=_SEEDVR2_7B, budget_bytes=None, downloaded=copy_files is not None, weights=MLX_WEIGHTS)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
    return web_config


def test_models_route_lists_the_downloaded_seedvr2_tier_with_its_disk_use(monkeypatch, tmp_path):
    _models_route_with_seedvr2(monkeypatch, tmp_path, copy_files={"transformer/w.safetensors": b"x" * 2_000_000_000, "vae/v.safetensors": b"y" * 500_000_000})

    with TestClient(web_server.app) as client:
        rows = client.get("/api/models").json()["image_models"]

    assert rows == [
        {
            "name": "seedvr2-7b@q8",
            "label": "seedvr2-7b@q8",
            "family": "seedvr2",
            "size_label": "7b",
            "source": "seedvr2",
            "stored_quant": None,
            "downloaded": True,
            "memory_fit": None,
            "delete": {"kind": "installed", "repo_id": None, "linked_by": [], "stored_quants": []},
            "disk_size_label": "2.5 GB",
            "download_size_label": "17.0 GB",
            "download_needed": False,
        }
    ]


def test_models_route_lists_the_fp8_tier_with_its_folder_disk_use(monkeypatch, tmp_path):
    web_config = _models_route_with_seedvr2(monkeypatch, tmp_path, copy_files=None)
    folder = seedvr2_weights_dir("seedvr2-7b", tmp_path / "models", FP8_WEIGHTS)
    folder.mkdir(parents=True)
    (folder / "ema_vae_fp16.safetensors").write_bytes(b"x" * 1_500_000_000)
    web_config.seedvr2 = replace(web_config.seedvr2, downloaded=True, weights=FP8_WEIGHTS)

    with TestClient(web_server.app) as client:
        (row,) = client.get("/api/models").json()["image_models"]

    assert row["name"] == "seedvr2-7b@fp8"
    assert row["delete"] == {"kind": "installed", "repo_id": None, "linked_by": [], "stored_quants": []}
    assert (row["disk_size_label"], row["download_size_label"]) == ("1.5 GB", "9.0 GB")


def test_models_route_lists_the_seedvr2_tier_as_not_downloaded_without_a_copy(monkeypatch, tmp_path):
    _models_route_with_seedvr2(monkeypatch, tmp_path, copy_files=None)

    with TestClient(web_server.app) as client:
        (row,) = client.get("/api/models").json()["image_models"]

    assert (row["name"], row["downloaded"], row["delete"], row["disk_size_label"], row["download_size_label"]) == ("seedvr2-7b@q8", False, None, None, "17.0 GB")


def test_models_route_reads_not_downloaded_after_the_copy_is_deleted(monkeypatch, tmp_path):
    """The row is rebuilt from disk on every request, so deleting the copy flips it without a restart."""
    from zvisiongenerator.seedvr2_loader import SEEDVR2_STORED_FORMAT, seedvr2_downloaded
    from zvisiongenerator.utils.stored_quant import write_manifest

    models_dir = tmp_path / "models"
    copy = seedvr2_weights_dir("seedvr2-7b", models_dir, MLX_WEIGHTS)
    (copy / "w.safetensors").parent.mkdir(parents=True)
    (copy / "w.safetensors").write_bytes(b"x" * 1000)
    write_manifest(copy, {"bits": 8, "backend_format": SEEDVR2_STORED_FORMAT, "model": "seedvr2-7b"})
    web_config = _make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.models_dir = str(models_dir)
    settings = SeedVR2Settings(model="auto", softness=0.75, reference_megapixels=16, tiers=(_SEEDVR2_7B,))

    def load() -> SimpleNamespace:
        web_config.seedvr2 = SeedVR2Setup(settings=settings, tier=_SEEDVR2_7B, budget_bytes=None, downloaded=seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS), weights=MLX_WEIGHTS)
        return web_config

    monkeypatch.setattr(web_server, "load_web_config", load)
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
    monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)

    with TestClient(web_server.app) as client:
        (before,) = client.get("/api/models").json()["image_models"]
        assert client.delete("/api/models/seedvr2-7b@q8").status_code == 200
        (after,) = client.get("/api/models").json()["image_models"]

    assert (before["downloaded"], before["delete"] is not None, before["disk_size_label"]) == (True, True, "0.0 GB")
    assert (after["downloaded"], after["delete"], after["disk_size_label"], after["download_size_label"]) == (False, None, None, "17.0 GB")
    assert not copy.exists()


def test_models_route_has_no_seedvr2_row_where_seedvr2_is_off(monkeypatch, tmp_path):
    web_config = _models_route_with_seedvr2(monkeypatch, tmp_path, copy_files={"w.safetensors": b"x"})
    web_config.seedvr2 = None

    with TestClient(web_server.app) as client:
        rows = client.get("/api/models").json()["image_models"]

    assert rows == []


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
    source, prompt, negative, batch, _enhance, _scenes = web_server._resolve_prompt_submission_with_enhance(form)
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
    ("form", "review", "controls"),
    [
        ({"upscale": "2", "review": "on"}, True, ["next", "pause", "resume", "repeat", "approve", "discard", "quit"]),
        ({"upscale": "2"}, False, ["next", "pause", "resume", "repeat", "quit"]),
        ({"review": "on"}, False, ["next", "pause", "resume", "repeat", "quit"]),
        ({"upscale": "2", "review": "off"}, False, ["next", "pause", "resume", "repeat", "quit"]),
    ],
)
def test_submit_image_job_review_needs_upscale(monkeypatch, tmp_path, form, review, controls):
    """Review is carried only with upscale on, and only then do approve and discard become controls."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=_make_resolved_image_defaults(), submitted=submitted)

    response = web_server._submit_image_job({"model": "zit", "prompt": "hello", **form}, web_config)

    assert submitted[0]["args"].review is review
    assert list(response["supported_controls"]) == controls


def test_workspace_contract_offers_the_image_review_control_for_image_workflows_only():
    definitions = web_server._build_workflow_contract()["definitions"]

    assert "image_upscale_review" in definitions["txt2img"]["visible_controls"]
    assert "image_upscale_review" in definitions["img2img"]["visible_controls"]
    assert "image_upscale_review" not in definitions["txt2vid"]["visible_controls"]
    assert "image_upscale_review" not in definitions["img2vid"]["visible_controls"]


def _patch_video_submit_dependencies(monkeypatch, *, review_supported: bool, submitted: list[dict[str, object]]) -> None:
    monkeypatch.setattr(video_clip_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        video_clip_module,
        "detect_video_model",
        lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=64, frame_alignment=8, default_fps=24),
    )
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server, "supports_video_upscale_review", lambda: review_supported)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")


@pytest.mark.parametrize(
    ("form", "review", "controls"),
    [
        ({"upscale": "2", "review": "on"}, True, ["approve", "discard", "quit"]),
        ({"upscale": "2"}, False, ["quit"]),
        ({"review": "on"}, False, ["quit"]),
        ({"upscale": "2", "review": "off"}, False, ["quit"]),
    ],
)
def test_submit_video_job_review_needs_upscale(monkeypatch, tmp_path, form, review, controls):
    """A video review is carried only with upscale on, and only then do approve and discard become controls."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=True, submitted=submitted)

    response = web_server._submit_video_job({"model": "ltx-8", "prompt": "a lake", **form}, web_config)

    assert submitted[0]["args"].review is review
    assert list(response["supported_controls"]) == controls


def test_submit_video_job_refuses_review_where_the_platform_cannot_pause(monkeypatch, tmp_path):
    """A review request for video on a platform without support is a 400 and queues nothing; without upscale it is ignored."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=False, submitted=submitted)

    with pytest.raises(HTTPException) as refused:
        web_server._submit_video_job({"model": "ltx-8", "prompt": "a lake", "upscale": "2", "review": "on"}, web_config)
    assert refused.value.status_code == 400
    assert submitted == []

    web_server._submit_video_job({"model": "ltx-8", "prompt": "a lake", "review": "on"}, web_config)
    assert submitted[0]["args"].review is False


@pytest.mark.parametrize("supported", [True, False])
def test_workspace_contract_hides_the_video_review_control_where_unsupported(monkeypatch, supported):
    monkeypatch.setattr(web_server, "supports_video_upscale_review", lambda: supported)
    definitions = web_server._build_workflow_contract()["definitions"]

    for workflow in ("txt2vid", "img2vid"):
        assert ("video_upscale_review" in definitions[workflow]["visible_controls"]) is supported
        assert "video_upscale_enabled" in definitions[workflow]["visible_controls"]
    for workflow in ("txt2img", "img2img"):
        assert "video_upscale_review" not in definitions[workflow]["visible_controls"]
        assert "image_upscale_review" in definitions[workflow]["visible_controls"]


class TestJobReviewRoute:
    """GET /jobs/{id}/review serves the recorded small render of a pending video review."""

    @staticmethod
    def _patch_review_paths(monkeypatch, paths: dict[str, str | None]) -> None:
        def _get_job_review_path(job_id):
            if job_id not in paths:
                raise KeyError(job_id)
            return paths[job_id]

        monkeypatch.setattr(web_server.web_runner, "get_job_review_path", _get_job_review_path)

    def test_serves_the_recorded_file_as_uncached_mp4(self, monkeypatch, tmp_path):
        render = tmp_path / "small.mp4"
        render.write_bytes(b"0123456789")
        self._patch_review_paths(monkeypatch, {"job-1": str(render)})

        with TestClient(web_server.app) as client:
            ok = client.get("/jobs/job-1/review?v=1")
            ranged = client.get("/jobs/job-1/review?v=1", headers={"Range": "bytes=2-5"})

        assert ok.status_code == 200
        assert ok.headers["content-type"] == "video/mp4"
        assert ok.headers["cache-control"] == "no-store"
        assert ok.content == b"0123456789"
        assert ranged.status_code == 206
        assert ranged.content == b"2345"
        assert ranged.headers["cache-control"] == "no-store"

    def test_never_takes_the_path_from_the_request(self, monkeypatch, tmp_path):
        render = tmp_path / "small.mp4"
        render.write_bytes(b"recorded")
        secret = tmp_path / "secret.txt"
        secret.write_bytes(b"secret")
        self._patch_review_paths(monkeypatch, {"job-1": str(render)})

        with TestClient(web_server.app) as client:
            response = client.get("/jobs/job-1/review", params={"v": "1", "path": str(secret), "review_path": str(secret)})

        assert response.content == b"recorded"

    def test_is_404_without_a_pending_review_a_file_or_a_job(self, monkeypatch, tmp_path):
        self._patch_review_paths(monkeypatch, {"idle": None, "gone": str(tmp_path / "gone.mp4")})

        with TestClient(web_server.app) as client:
            responses = [client.get(f"/jobs/{job_id}/review?v=1") for job_id in ("idle", "gone", "unknown")]

        assert [response.status_code for response in responses] == [404, 404, 404]


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
    video_names: list[tuple[str | None, bool]] = []

    def _fake_status(resolved_path, *, kind, quantize_options=(), budget=None, find_local_dir=None, model_name=None, app_config=None):
        calls.append((resolved_path, kind, quantize_options))
        if kind == "video":
            video_names.append((model_name, app_config is web_config.app_config))
        return {"downloaded": resolved_path != "owner/ltx", "memory_fit": None}

    monkeypatch.setattr(workspace_api_module, "describe_model_status", _fake_status)
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: MemoryBudget(10 * 1024**3))
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
        token_source=None,
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
    assert video_names and set(video_names) == {("ltx-8", True)}  # a video model's level comes from its name and the config


def test_workspace_options_carry_the_configured_name_as_their_label(monkeypatch):
    """AC4.3: a named alias is labelled by its name, an unnamed model by its id; the id never changes."""
    web_config = _make_web_config()
    web_config.app_config["model_labels"] = {"zit": {"name": "Z-Image Turbo"}, "ltx-8": {"name": "LTX-2.3 8-bit"}}
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)

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

    assert {(entry["id"], entry["label"]) for entry in workspace["image_models"]} == {("zit", "Z-Image Turbo"), ("local-image", "local-image")}
    assert [(entry["id"], entry["label"]) for entry in workspace["video_models"]] == [("ltx-8", "LTX-2.3 8-bit")]


def test_workspace_models_without_inventory_entry_report_unknown_status(monkeypatch):
    """Names with no inventory entry (e.g. stale config) report unknown rather than guessing."""
    web_config = _make_web_config()
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)

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
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)

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
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)
    monkeypatch.setattr(workspace_api_module, "describe_model_status", lambda resolved_path, **_: {"downloaded": True, "memory_fit": None})
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])

    models = workspace_api_module.build_models_response(
        web_config,
        token_source=None,
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


@pytest.mark.parametrize(("quantize", "expected_status"), [("8", 409), ("", 200)])
def test_convert_route_refuses_a_quantized_copy_while_a_job_runs(monkeypatch, tmp_path, quantize, expected_status):
    """Saving a quantized copy loads a full model, so it waits for a running job; a plain convert does not."""
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    ran: list[list[str]] = []
    monkeypatch.setattr(web_server, "_run_model_management_command", lambda args: ran.append(args) or "done")
    monkeypatch.setattr(web_server.web_runner, "_find_active_exclusive_job_id", lambda: "busy")

    with TestClient(web_server.app) as client:
        response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "flux2-klein-9b", "quantize": quantize})

    assert response.status_code == expected_status
    assert len(ran) == (1 if expected_status == 200 else 0)


def test_jobs_are_refused_while_a_model_is_being_quantized(monkeypatch, tmp_path):
    """While a quantized-copy conversion runs, job admission is refused; afterwards it is allowed again."""
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    admission: list[str] = []

    def _convert(args):
        try:
            web_server._reject_while_busy()
            admission.append("admitted")
        except web_server.JobConflictError:
            admission.append("refused")
        return "done"

    monkeypatch.setattr(web_server, "_run_model_management_command", _convert)

    with TestClient(web_server.app) as client:
        response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "flux2-klein-9b", "quantize": "8"})

    assert response.status_code == 200
    assert admission == ["refused"]
    web_server._reject_while_busy()  # released after the conversion


def test_quantize_slot_is_released_when_the_conversion_fails(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")

    def _fail(args):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(web_server, "_run_model_management_command", _fail)

    with TestClient(web_server.app) as client:
        response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "flux2-klein-9b", "quantize": "8"})

    assert response.status_code == 400
    web_server._reject_while_busy()


class TestJobQueueRoutes:
    """Queue routes on a real runner whose worker is held by a running job."""

    @pytest.fixture()
    def busy_runner(self, monkeypatch):
        runner = web_server.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        monkeypatch.setattr(web_server, "web_runner", runner)
        started = threading.Event()
        release = threading.Event()

        def _block(_cb):
            started.set()
            release.wait(timeout=2.0)

        running = runner._submit_job(job_type="image", exclusive=True, supported_controls=("quit",), control_signal=MagicMock(), target_factory=_block)
        assert started.wait(timeout=1.0)
        yield runner, running
        release.set()
        runner.shutdown()

    @pytest.mark.parametrize("controls", [("quit",), ()], ids=["image", "video"])
    def test_cancel_route_removes_a_queued_job(self, busy_runner, controls):
        runner, _running = busy_runner
        queued = runner._submit_job(job_type="test", exclusive=True, supported_controls=controls, target_factory=lambda _cb: None)

        with TestClient(web_server.app) as client:
            response = client.post(f"/api/jobs/{queued}/cancel")

        assert response.status_code == 200
        assert response.json() == {"job_id": queued, "status": "cancelled"}
        assert runner.get_job_snapshot(queued)["status"] == "cancelled"

    def test_cancel_route_still_quits_a_running_image_job(self, busy_runner):
        _runner, running = busy_runner
        with TestClient(web_server.app) as client:
            response = client.post(f"/api/jobs/{running}/cancel")

        assert response.status_code == 200
        assert response.json()["action"] == "quit"

    def test_list_and_clear_routes(self, monkeypatch, busy_runner):
        runner, running = busy_runner
        monkeypatch.setattr(web_server, "load_web_config", _make_web_config)
        monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: _make_workspace_bootstrap_view())
        queued = [runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None) for _ in range(2)]

        with TestClient(web_server.app) as client:
            listed = client.get("/api/jobs").json()
            workspace_queue = client.get("/api/workspace", params={"include_history": "false"}).json()["queued_jobs"]
            cleared = client.delete("/api/jobs/queue").json()
            after = client.get("/api/jobs").json()

        assert listed["active_job"]["job_id"] == running
        assert [job["job_id"] for job in listed["queued_jobs"]] == queued
        assert [job["queue_position"] for job in listed["queued_jobs"]] == [1, 2]
        assert [job["job_id"] for job in workspace_queue] == queued
        assert cleared == {"cancelled": queued}
        assert after["active_job"]["job_id"] == running
        assert after["queued_jobs"] == []

    def test_deleting_models_is_refused_while_jobs_are_queued(self, busy_runner):
        with TestClient(web_server.app) as client:
            response = client.delete("/api/loras/style")

        assert response.status_code == 409
        assert "all jobs" in response.json()["detail"]


def test_submitted_settings_keep_every_text_field_and_replace_an_upload_with_its_path():
    """Load settings restores the form from these values, including repeated fields."""
    from starlette.datastructures import FormData, UploadFile

    upload = UploadFile(file=io.BytesIO(b"png"), filename="ref.png")
    form = FormData([("prompt", "a lake"), ("prompt_option_id", "a:0"), ("prompt_option_id", "a:1"), ("steps", "8"), ("image_file", upload)])

    settings = web_server._submitted_settings(form, image_path="/out/.web_uploads/ref.png")

    assert settings == {"prompt": "a lake", "prompt_option_id": ["a:0", "a:1"], "steps": "8", "image_path": "/out/.web_uploads/ref.png"}


def test_submit_image_job_queues_with_its_public_context(monkeypatch, tmp_path):
    """The job carries its prompt, model, size label and settings from the moment it is queued."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=_make_resolved_image_defaults(), submitted=submitted)

    response = web_server._submit_image_job({"model": "zit", "prompt": "hello", "steps": "9", "workflow": "txt2img"}, web_config)

    context = submitted[0]["context"]
    assert context == response["context"]
    assert context["prompt"] == "hello"
    assert context["model"] == "zit"
    assert context["model_label"] == "zit"
    assert context["workflow"] == "txt2img"
    assert "steps" in context["meta"]
    assert context["settings"] == {"model": "zit", "prompt": "hello", "steps": "9", "workflow": "txt2img"}


def test_a_submitted_job_carries_the_models_configured_name_beside_its_id(monkeypatch, tmp_path):
    """AC4.5: the context and the response say the name; the model id is unchanged."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.app_config["model_labels"] = {"zit": {"name": "Z-Image Turbo"}}
    submitted: list[dict[str, object]] = []
    _patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=_make_resolved_image_defaults(), submitted=submitted)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server.web_runner, "get_job_snapshot", lambda _job_id: {"queue_position": 1})

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data={"mode": "image", "model": "zit", "prompt": "hello", "workflow": "txt2img"})

    assert response.status_code == 200
    body = response.json()
    assert (body["model"], body["model_label"]) == ("zit", "Z-Image Turbo")
    assert (submitted[0]["context"]["model"], submitted[0]["context"]["model_label"]) == ("zit", "Z-Image Turbo")


def test_a_job_snapshot_carries_the_model_label_and_falls_back_to_the_model():
    """AC4.5: ``model_label`` is in the snapshot; a context without one reads as its model id."""
    base = {
        "job_id": "j",
        "status": "queued",
        "workflow": "txt2img",
        "supported_controls": (),
        "created_at": 0.0,
        "completed_at": None,
        "event_count": 0,
        "last_event": None,
        "paused": False,
        "result_path": None,
    }

    named = public_job_snapshot(context={"model": "zit", "model_label": "Z-Image Turbo"}, **base)
    unnamed = public_job_snapshot(context={"model": "zit"}, **base)

    assert (named["model"], named["model_label"]) == ("zit", "Z-Image Turbo")
    assert (unnamed["model"], unnamed["model_label"]) == ("zit", "zit")


@pytest.mark.parametrize("amount", [-0.1, 1.6])
def test_generate_rejects_sharpen_amounts_outside_the_safe_range(amount):
    from conftest import _make_args

    with pytest.raises(ValueError, match="Sharpen amount must be between 0 and 1.5"):
        web_server._validate_image_args(_make_args(sharpen=amount, image_strength=0.5, quantize=None, runs=1))


def test_generate_endpoint_answers_400_for_a_video_review_the_platform_cannot_pause(monkeypatch, tmp_path):
    """The 400 reaches the HTTP client of POST /api/generate, and nothing is queued."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=False, submitted=submitted)

    with TestClient(web_server.app) as client:
        refused = client.post("/api/generate", data={"mode": "video", "model": "ltx-8", "prompt": "a lake", "upscale": "2", "review": "true"})
        ignored = client.post("/api/generate", data={"mode": "video", "model": "ltx-8", "prompt": "a lake", "review": "true"})

    assert refused.status_code == 400
    assert ignored.status_code == 200
    assert ignored.json()["supported_controls"] == ["quit"]
    assert len(submitted) == 1


def test_review_route_is_404_when_the_file_vanishes_before_it_is_served(monkeypatch, tmp_path):
    render = tmp_path / "small.mp4"
    render.write_bytes(b"0123456789")

    def _review_path_then_vanish(_job_id):
        path = str(render)
        render.unlink()  # the job resumed and its temporary directory was cleaned up right after the lookup
        return path

    monkeypatch.setattr(web_server.web_runner, "get_job_review_path", _review_path_then_vanish)

    with TestClient(web_server.app, raise_server_exceptions=False) as client:
        assert client.get("/jobs/job-1/review?v=1").status_code == 404


def test_review_route_is_404_when_the_path_is_not_a_regular_file(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server.web_runner, "get_job_review_path", lambda _job_id: str(tmp_path))

    with TestClient(web_server.app, raise_server_exceptions=False) as client:
        assert client.get("/jobs/job-1/review?v=1").status_code == 404


# ── Extend: source inspection, picker purpose and submission ────────────────

_EXTEND_SCENE = "A woman in a red raincoat on a rainy street at night."
_EXTEND_PROMPT = "She opens a black umbrella and walks away."


# A clean 512x256, 24 fps, 97-frame clip with audio and BT.709 colour tags.
_clip_probe = functools.partial(make_probe, frame_count=97, duration=4.04, color_space="bt709", color_primaries="bt709", color_trc="bt709")


@pytest.fixture
def clip(monkeypatch, tmp_path):
    """Create a source clip file whose probe and embedded settings come from the returned, mutable ``spec``."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 16)
    spec = SimpleNamespace(path=path, probe={}, config={"scene": _EXTEND_SCENE, "prompt": _EXTEND_PROMPT, "model": "ltx-8"}, audio_ok=True)
    extend_api_module._inspect_by_identity.cache_clear()
    spec.probed = []
    monkeypatch.setattr(video_extend_module, "probe_video", lambda path: spec.probed.append(path) or _clip_probe(**spec.probe))
    monkeypatch.setattr(video_extend_module, "read_mp4_config", lambda _path: spec.config)
    monkeypatch.setattr(video_extend_module, "audio_decodes", lambda *_args, **_kwargs: spec.audio_ok)
    for module in (web_server, extend_api_module):
        monkeypatch.setattr(module, "supports_video_extend", lambda: True)
        monkeypatch.setattr(module, "supports_video_audio", lambda: True)
    monkeypatch.setattr(web_server, "load_web_config", _make_web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path)
    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: None)
    for module in (web_server, extend_api_module):
        monkeypatch.setattr(module, "require_ffmpeg", lambda: None)
    return spec


def _inspect(body):
    with TestClient(web_server.app) as client:
        return client.post("/api/extend/source", json=body)


def test_a_clip_that_has_not_changed_is_probed_once(clip):
    first = extend_api_module._inspect_source(clip.path)
    second = extend_api_module._inspect_source(clip.path)

    assert second is first
    assert len(clip.probed) == 1


@pytest.mark.parametrize("change", ["modified", "resized"])
def test_a_clip_that_was_edited_is_probed_again(clip, change):
    extend_api_module._inspect_source(clip.path)
    stat = clip.path.stat()
    if change == "modified":
        os.utime(clip.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        clip.path.write_bytes(b"\x00" * 99)
        os.utime(clip.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same time, only the size differs

    extend_api_module._inspect_source(clip.path)

    assert len(clip.probed) == 2


def test_a_missing_clip_is_not_probed(clip):
    with pytest.raises(FileNotFoundError):
        extend_api_module._inspect_source(clip.path.with_name("gone.mp4"))

    assert clip.probed == []


def test_a_source_is_probed_once_across_the_inspect_route_and_the_job_checks(clip):
    body = {"path": str(clip.path)}

    assert _inspect(body).status_code == 200
    first = extend_api_module.load_source_video(str(clip.path), key="source_video")
    second = extend_api_module.load_source_video(str(clip.path), key="context.source_video")

    assert second is first
    assert len(clip.probed) == 1


def test_a_source_that_changed_since_it_was_inspected_is_checked_again(clip):
    assert _inspect({"path": str(clip.path)}).status_code == 200
    clip.path.write_bytes(b"\x00" * 32)
    clip.probe = {"frame_count": 5}

    with pytest.raises(ValueError, match="cannot be extended"):
        extend_api_module.load_source_video(str(clip.path), key="source_video")

    assert len(clip.probed) == 2


def test_inspect_route_reports_the_facts_recorded_prompts_and_defaults_of_a_good_source(clip):
    """The response is the shape the frontend types are checked against: ``frontend/src/lib/state/extend.test.ts`` loads this same file as an ``ExtendSourceInfo``; change both together."""
    fixture = json.loads((Path(__file__).parent / "fixtures" / "extend_source_response.json").read_text(encoding="utf-8"))

    payload = _inspect({"path": str(clip.path)}).json()

    assert list(payload) == list(fixture)
    # Only where the temporary clip lives differs from the file.
    assert payload == {**fixture, "path": str(clip.path.resolve())}


def test_inspect_route_gives_no_media_url_outside_the_output_root(clip, monkeypatch, tmp_path):
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    monkeypatch.setattr(web_server, "_media_output_root", lambda: other_root)

    assert _inspect({"path": str(clip.path)}).json()["media_url"] is None


def test_inspect_route_shows_the_displayed_size_of_a_rotated_source(clip):
    clip.probe = {"coded_width": 512, "coded_height": 256, "width": 256, "height": 512, "rotation": 90}

    payload = _inspect({"path": str(clip.path)}).json()

    assert (payload["width"], payload["height"], payload["rotation"]) == (256, 512, 90)
    assert payload["problems"] == []


@pytest.mark.parametrize(
    ("probe", "config", "audio_ok", "code"),
    [
        ({"width": 500, "coded_width": 500}, None, True, "off_grid"),
        ({}, {"generation": {"upscale": {"factor": 2}}}, True, "upscaled"),
        ({}, None, False, "unreadable_audio"),
    ],
)
def test_inspect_route_surfaces_a_problem_code_with_a_message(clip, probe, config, audio_ok, code):
    clip.probe = probe
    clip.audio_ok = audio_ok
    if config is not None:
        clip.config = {**clip.config, **config}

    payload = _inspect({"path": str(clip.path)}).json()

    assert [problem["code"] for problem in payload["problems"]] == [code]
    assert all(set(problem) == {"code", "message"} and problem["message"] for problem in payload["problems"])
    assert payload["width"] is not None
    assert payload["memory_fit"] is None


def test_inspect_route_reports_a_file_that_is_not_a_video_without_facts(clip, monkeypatch):
    def _no_video(_path):
        raise ValueError(video_extend_module.NO_VIDEO_STREAM)

    monkeypatch.setattr(video_extend_module, "probe_video", _no_video)

    payload = _inspect({"path": str(clip.path)}).json()

    assert [problem["code"] for problem in payload["problems"]] == ["no_video_stream"]
    assert all(payload[key] is None for key in ("width", "height", "rotation", "fps", "frame_count", "duration_seconds", "has_audio", "extend_frames", "window_frames"))
    assert payload["memory_fit"] is None


def test_inspect_route_flags_a_single_prompt_source_as_legacy(clip):
    clip.config = {"prompt": 'A woman stands in the rain and says "Hello".', "model": "ltx-8"}

    payload = _inspect({"path": str(clip.path)}).json()

    assert (payload["scene"], payload["prompt"], payload["legacy"]) == (None, 'A woman stands in the rain and says "Hello".', True)


def test_inspect_route_handles_a_source_without_embedded_settings(clip):
    clip.config = None

    payload = _inspect({"path": str(clip.path)}).json()

    assert (payload["scene"], payload["prompt"], payload["legacy"], payload["model"], payload["lora"]) == (None, None, False, None, None)
    assert payload["problems"] == []
    assert payload["extend_frames"] == 48


def test_inspect_route_resolves_the_recorded_model_against_the_video_models(clip):
    clip.config = {**clip.config, "model": "some-removed-model", "lora": "/loras/style.safetensors:0.8"}

    payload = _inspect({"path": str(clip.path)}).json()

    assert payload["model"] is None
    assert payload["lora"] == "/loras/style.safetensors:0.8"


def test_inspect_route_reports_no_loras_as_blank_for_a_source_with_settings(clip):
    """Blank, not null: the source recorded its LoRAs and had none, so extending it clears the chosen ones."""
    clip.config = {"prompt": "A walk.", "model": "ltx-8"}

    assert _inspect({"path": str(clip.path)}).json()["lora"] == ""


@pytest.mark.parametrize(("requested", "expected"), [(50, 48), (1, 8)])
def test_inspect_route_snaps_the_length_to_whole_latent_frames(clip, requested, expected):
    payload = _inspect({"path": str(clip.path), "extend_frames": requested}).json()

    assert (payload["extend_frames"], payload["window_frames"]) == (expected, 25 + expected)


def test_inspect_route_defaults_to_two_seconds_at_the_source_frame_rate(clip):
    clip.probe = {"r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001"}

    assert _inspect({"path": str(clip.path)}).json()["extend_frames"] == 56


def test_inspect_route_adds_the_memory_fit_of_the_window(clip, monkeypatch):
    calls: list[dict[str, object]] = []

    def _fit(model, **kwargs):
        calls.append({"model": model, **kwargs})
        return {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}

    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: f"/models/{model}")
    monkeypatch.setattr(extend_api_module, "clip_memory_fit", _fit)

    payload = _inspect({"path": str(clip.path), "extend_frames": 48, "low_memory": False}).json()

    assert payload["memory_fit"] == {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}
    assert len(calls) == 1
    assert {key: value for key, value in calls[0].items() if key != "app_config"} == {
        "model": "/models/ltx-8",
        "width": 512,
        "height": 256,
        "frames": 73,
        "low_memory": False,
        "budget": MemoryBudget(24 * 2**30),
        "model_name": "ltx-8",
    }


def test_inspect_route_uses_the_requested_model_and_defaults_low_memory_on(clip, monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: f"/models/{model}")
    monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append({"model": model, **kwargs}))

    payload = _inspect({"path": str(clip.path), "model": "other"}).json()

    assert payload["memory_fit"] is None
    assert (calls[0]["model"], calls[0]["low_memory"]) == ("/models/other", True)


def test_inspect_route_skips_the_memory_fit_for_a_source_with_problems(clip, monkeypatch):
    clip.probe = {"field_order": "tt"}
    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
    monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda *_args, **_kwargs: pytest.fail("memory fit asked for a source with problems"))

    assert _inspect({"path": str(clip.path)}).json()["memory_fit"] is None


def test_inspect_route_returns_404_for_a_missing_file(clip, tmp_path):
    response = _inspect({"path": str(tmp_path / "gone.mp4")})

    assert response.status_code == 404
    assert set(response.json()) == {"detail"}


def _no_ffmpeg() -> None:
    raise RuntimeError("ffmpeg is missing")


def test_inspect_route_without_ffmpeg_is_a_422_not_a_missing_file(clip, monkeypatch):
    monkeypatch.setattr(extend_api_module, "require_ffmpeg", _no_ffmpeg)

    response = _inspect({"path": str(clip.path)})

    assert response.status_code == 422
    assert clip.probed == []


def test_extend_submission_without_ffmpeg_is_refused_before_the_source_is_read(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    monkeypatch.setattr(extend_api_module, "require_ffmpeg", _no_ffmpeg)

    response = _post_extend({"source_video": str(clip.path), "prompt": "She walks away."})

    assert response.status_code == 422
    assert clip.probed == []
    assert submitted == []


@pytest.mark.parametrize(
    "body",
    [
        [1, 2],
        {},
        {"path": ""},
        {"path": 5},
        {"path": "https://example.com/clip.mp4"},
        {"path": "file:///tmp/clip.mp4"},
        {"path": "ftp://host/clip.mp4"},
        {"path": "clip.txt"},
        {"path": "CLIP.PNG"},
        {"path": "clip.mp4", "extend_frames": True},
        {"path": "clip.mp4", "extend_frames": 0},
        {"path": "clip.mp4", "extend_frames": "48"},
        {"path": "clip.mp4", "model": 5},
        {"path": "clip.mp4", "low_memory": "yes"},
    ],
)
def test_inspect_route_rejects_invalid_bodies(clip, body):
    if isinstance(body, dict) and body.get("path") == "clip.mp4":
        body = {**body, "path": str(clip.path)}

    response = _inspect(body)

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}


def test_inspect_route_rejects_a_body_that_is_not_json(clip):
    with TestClient(web_server.app) as client:
        response = client.post("/api/extend/source", content=b"not json", headers={"content-type": "application/json"})

    assert response.status_code == 422


def test_inspect_route_accepts_every_source_container(clip, tmp_path):
    for suffix in (".mov", ".m4v", ".mkv", ".webm", ".MP4"):
        path = tmp_path / f"clip{suffix}"
        path.write_bytes(b"\x00")
        assert _inspect({"path": str(path)}).status_code == 200


def test_workspace_contract_carries_extend_only_with_the_capability(monkeypatch):
    monkeypatch.setattr(web_server, "supports_video_extend", lambda: True)
    with_extend = web_server._build_workflow_contract()
    monkeypatch.setattr(web_server, "supports_video_extend", lambda: False)
    without_extend = web_server._build_workflow_contract()

    assert "extend" in with_extend["values"] and "extend" in with_extend["definitions"]
    assert "extend" not in without_extend["values"] and "extend" not in without_extend["definitions"]


def _post_extend(data: dict[str, object]):
    return _post_video({"workflow": "extend", **data})


def _stub_extend_submission(monkeypatch, tmp_path, clip) -> list[dict[str, object]]:
    submitted = _stub_video_submission(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "resolve_video_clip", lambda *_args, **_kwargs: pytest.fail("an extension must not be aligned"))
    return submitted


def test_extend_submission_sets_the_args_the_workflow_reads(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), "extend_frames": "48", "prompt": "She walks away.", "audio": "on"})

    assert response.status_code == 200
    (job,) = submitted
    args = job["args"]
    source = args.extend_source
    assert (source.path, source.width, source.height, source.fps, source.frame_count) == (clip.path.resolve(), 512, 256, 24.0, 97)
    assert (args.extend_frames, args.width, args.height, args.fps, args.num_frames) == (48, 512, 256, 24.0, 73)
    assert (args.image_path, args.upscale, args.upscale_steps, args.frames) == (None, None, None, None)
    assert (args.no_audio, args.audio) == (False, True)
    assert (args.scene, args.prompt) == (_EXTEND_SCENE, "She walks away.")
    assert args.steps == 8
    request = job["request"]
    assert (request.width, request.height, request.num_frames, request.image_path, request.upscale) == (512, 256, 73, None, None)
    assert job["prompts_data"] == {"web": [("She walks away.", None)]}
    assert job["context"]["workflow"] == "extend"
    assert job["context"]["job_type"] == "Extend video"
    assert job["context"]["meta"] == "512x256 · +48 frames · 8 steps"
    assert job["context"]["settings"]["source_video"] == str(clip.path)
    assert response.json()["workflow"] == "extend"


def test_extend_submission_turns_audio_off_when_unchecked_or_unsupported(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    _post_extend({"source_video": str(clip.path), "audio": "off"})
    monkeypatch.setattr(web_server, "supports_video_audio", lambda: False)
    _post_extend({"source_video": str(clip.path), "audio": "on"})

    assert [job["args"].no_audio for job in submitted] == [True, True]


def test_extend_submission_reads_the_length_as_frames_or_seconds_at_the_source_frame_rate(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    base = {"source_video": str(clip.path)}

    _post_extend({**base, "extend_frames": "16"})
    _post_extend({**base, "extend_seconds": "1"})
    _post_extend({**base, "extend_seconds": "2.5", "extend_frames": "8"})
    _post_extend(base)

    assert [job["args"].extend_frames for job in submitted] == [16, 24, 8, 48]
    assert [job["args"].num_frames for job in submitted] == [41, 49, 33, 73]


@pytest.mark.parametrize(
    "data",
    [
        {"extend_frames": "12"},
        {"extend_frames": "0"},
        {"extend_frames": "x"},
        {"extend_seconds": "0"},
        {"extend_seconds": "-1"},
        {"extend_seconds": "nan"},
        {"extend_seconds": "inf"},
        {"extend_seconds": "1e999"},
    ],
)
def test_extend_submission_rejects_an_invalid_length(monkeypatch, tmp_path, clip, data):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), **data})

    assert response.status_code == 422
    assert "extend_" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_defaults_the_scene_to_the_sources_and_keeps_an_explicit_one(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    base = {"source_video": str(clip.path), "prompt": "She walks away."}

    _post_extend(base)
    _post_extend({**base, "scene": "A night street."})
    _post_extend({**base, "scene": "  "})

    assert [job["args"].scene for job in submitted] == [_EXTEND_SCENE, "A night street.", None]


def test_extend_submission_of_a_legacy_source_has_no_default_scene(monkeypatch, tmp_path, clip):
    clip.config = {"prompt": "A woman stands in the rain.", "model": "ltx-8"}
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), "prompt": "A woman stands in the rain."})

    assert response.status_code == 200
    assert submitted[0]["args"].scene is None
    assert submitted[0]["args"].extend_source.legacy is True


def test_extend_submission_allows_an_empty_continuation(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    nothing = _post_extend({"source_video": str(clip.path)})
    blank = _post_extend({"source_video": str(clip.path), "prompt": "  ", "scene": ""})

    assert nothing.status_code == blank.status_code == 200
    assert [job["prompts_data"] for job in submitted] == [{"web": [("", None)]}, {"web": [("", None)]}]
    assert [job["context"]["prompt"] for job in submitted] == [_EXTEND_SCENE, ""]


def test_extend_submission_ignores_size_reference_and_upscale_fields(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    ignored = {"ratio": "bogus", "size": "bogus", "width": "640", "height": "352", "frames": "121", "image_path": "/missing.png", "upscale": "2"}

    response = _post_extend({"source_video": str(clip.path), "prompt": "She walks away.", **ignored})

    assert response.status_code == 200
    args = submitted[0]["args"]
    assert (args.ratio, args.size, args.image_path, args.upscale) == (None, None, None, None)
    assert (args.width, args.height, args.num_frames) == (512, 256, 73)


def test_extend_submission_checks_the_source_again(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    base = {"source_video": str(clip.path), "prompt": "She walks away."}

    clip.probe = {"field_order": "tt"}
    clip.path.write_bytes(b"\x00" * 32)  # an edited file has another size or time, so the earlier inspection is not reused
    changed = _post_extend(base)
    clip.probe = {}
    clip.path.unlink()
    moved = _post_extend(base)

    assert changed.status_code == 422 and "cannot be extended" in changed.json()["detail"]
    assert moved.status_code == 422 and "not found" in moved.json()["detail"]
    assert submitted == []


@pytest.mark.parametrize("source", [None, "", "https://example.com/clip.mp4", "clip.png", "clip.txt"])
def test_extend_submission_requires_a_video_source_path(monkeypatch, tmp_path, clip, source):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    data = {"prompt": "She walks away."} if source is None else {"prompt": "She walks away.", "source_video": source}

    response = _post_extend(data)

    assert response.status_code == 422
    assert "source_video" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_refuses_a_link_to_a_file_that_is_not_a_video(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    target = tmp_path / "notes.txt"
    target.write_text("not a video")
    link = tmp_path / "looks-like-a-video.mp4"
    link.symlink_to(target)

    response = _post_extend({"source_video": str(link), "prompt": "She walks away."})

    assert response.status_code == 422
    assert "must be a video file" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_is_refused_without_the_capability(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    monkeypatch.setattr(web_server, "supports_video_extend", lambda: False)

    response = _post_extend({"source_video": str(clip.path), "prompt": "She walks away."})

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert submitted == []


def test_extend_submission_with_a_prompt_file_threads_entry_scenes(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    path = tmp_path / "prompts.yaml"
    path.write_text("next:\n  - scene: A night street.\n    prompt: she leaves\n  - prompt: he arrives\n", encoding="utf-8")

    response = _post_extend({"source_video": str(clip.path), "prompt_source": "file", "prompts_file": str(path), "prompt_option_id": ["next:0", "next:1"]})

    assert response.status_code == 200
    (job,) = submitted
    assert job["scene_by_set"] == {"next": ["A night street.", None]}
    assert job["args"].scene == _EXTEND_SCENE
    assert job["prompts_data"] == {"next": [("she leaves", None), ("he arrives", None)]}


_NO_OP_ENHANCE = {"enhance_auto": "on", "enhance_settings": json.dumps({"style": "keep", "mood": "keep", "details": [], "length": "same", "motion": [], "sound": []}), "prompt": "She walks away."}


def test_extend_submission_never_treats_enhance_settings_as_a_no_op(monkeypatch, tmp_path, clip):
    """An extension always writes a new action beside the kept scene, so even unchanged settings are accepted."""
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), **_NO_OP_ENHANCE})

    assert response.status_code == 200
    assert submitted[0]["args"].enhance is not None


def test_video_submission_with_a_scene_accepts_keep_everything_enhance_settings(monkeypatch, tmp_path):
    submitted = _stub_video_submission(monkeypatch, tmp_path)

    response = _post_video({"scene": "A night street.", **_NO_OP_ENHANCE})

    assert response.status_code == 200
    assert len(submitted) == 1


def test_video_submission_accepts_action_only_settings_that_keep_everything(monkeypatch, tmp_path):
    """``parts=action`` rewrites the action to fit the scene, so unchanged style, mood and length are not a no-op."""
    submitted = _stub_video_submission(monkeypatch, tmp_path)
    settings = json.dumps({"style": "keep", "mood": "keep", "details": [], "length": "same", "motion": [], "sound": [], "parts": "action"})

    response = _post_video({"scene": "A night street.", "enhance_auto": "on", "enhance_settings": settings, "prompt": "She walks away."})

    assert response.status_code == 200
    assert submitted[0]["args"].enhance.parts == "action"


def test_video_submission_refuses_an_unknown_parts_value(monkeypatch, tmp_path):
    submitted = _stub_video_submission(monkeypatch, tmp_path)

    response = _post_video({"scene": "A night street.", "enhance_auto": "on", "enhance_settings": json.dumps({"parts": "scene"}), "prompt": "She walks away."})

    assert response.status_code == 422 and "both, action" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_with_a_prompt_file_keeps_no_op_entries_for_the_enhancer(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    path = tmp_path / "prompts.yaml"
    path.write_text("next:\n  - prompt: she leaves\n    enhance:\n      style: keep\n", encoding="utf-8")

    response = _post_extend({"source_video": str(clip.path), "prompt_source": "file", "prompts_file": str(path), "prompt_option_id": ["next:0"]})

    assert response.status_code == 200
    assert submitted[0]["enhance_by_set"] is not None
    assert submitted[0]["enhance_by_set"]["next"][0] is not None


def test_inspect_route_treats_a_directory_with_a_video_name_as_missing(clip, tmp_path):
    folder = tmp_path / "folder.mp4"
    folder.mkdir()

    assert _inspect({"path": str(folder)}).status_code == 404


def test_inspect_route_resolves_the_path_it_reports(clip, tmp_path):
    (tmp_path / "sub").mkdir()
    payload = _inspect({"path": str(tmp_path / "sub" / ".." / "clip.mp4")}).json()

    assert payload["path"] == str(clip.path.resolve())
    assert payload["media_url"] == "/media/clip.mp4"


# ─── POST /api/video/memory-fit ──────────────────────────────────────────────

_UNALIGNED_CLIP = {"steps": 8, "width": 720, "height": 450, "num_frames": 52}


@pytest.fixture
def clip_fit(monkeypatch, tmp_path):
    """Stub the model and budget behind ``POST /api/video/memory-fit`` and record what the memory check receives."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    calls: list[dict[str, object]] = []

    def _fit(model, **kwargs):
        calls.append({"model": model, **kwargs})
        return {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}

    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(video_clip_module, "resolve_model_path", lambda model, **_: f"/models/{model}")
    monkeypatch.setattr(video_clip_module, "detect_video_model", lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=32, frame_alignment=8, default_fps=24))
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: dict(_UNALIGNED_CLIP))
    monkeypatch.setattr(web_server, "memory_budget", lambda: MemoryBudget(24 * 2**30, unified_bytes=32 * 2**30))
    monkeypatch.setattr(web_server, "clip_memory_fit", _fit)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **_kwargs: pytest.fail("the memory check queued a job"))
    return SimpleNamespace(calls=calls, web_config=web_config)


def _memory_fit(body):
    with TestClient(web_server.app) as client:
        return client.post("/api/video/memory-fit", json=body)


def test_memory_fit_route_checks_the_aligned_clip_against_the_budget(clip_fit):
    response = _memory_fit({"model": "ltx-8", "ratio": "16:9", "size": "m", "frames": 52, "low_memory": False, "upscale": None})

    assert response.status_code == 200
    assert response.json() == {"memory_fit": {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}, "width": 736, "height": 448, "frames": 49}
    assert len(clip_fit.calls) == 1
    assert {key: value for key, value in clip_fit.calls[0].items() if key != "app_config"} == {
        "model": "/models/ltx-8",
        "width": 736,
        "height": 448,
        "frames": 49,
        "low_memory": False,
        "upscale": False,
        "budget": MemoryBudget(24 * 2**30, unified_bytes=32 * 2**30),
        "model_name": "ltx-8",
    }


def test_memory_fit_route_defaults_the_model_and_turns_low_memory_on(clip_fit):
    assert _memory_fit({}).status_code == 200

    assert (clip_fit.calls[0]["model"], clip_fit.calls[0]["low_memory"]) == ("/models/ltx-8", True)


def test_memory_fit_route_checks_an_upscaled_clip_as_upscaled(clip_fit):
    _memory_fit({"upscale": 2})

    assert clip_fit.calls[0]["upscale"] is True


def test_memory_fit_route_sends_the_form_fields_to_the_defaults_like_a_job(clip_fit, monkeypatch):
    seen: list[dict[str, object]] = []
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda family, _config, overrides: seen.append({"family": family, **overrides}) or dict(_UNALIGNED_CLIP))

    _memory_fit({"ratio": "16:9", "size": "m", "width": 640, "height": 384, "frames": 97})

    assert seen == [{"family": "ltx", "ratio": "16:9", "size": "m", "width": 640, "height": 384, "num_frames": 97}]


@pytest.mark.filterwarnings("ignore:.*Adjusted")
@pytest.mark.parametrize("upscale", [None, 2])
def test_memory_fit_route_resolves_the_size_and_length_a_job_gets(clip_fit, monkeypatch, upscale):
    """The route and ``POST /api/generate`` align the same request the same way."""
    jobs: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: jobs.append(kwargs) or "job-123")
    fields = {"ratio": "16:9", "size": "m", "frames": 52, **({"upscale": upscale} if upscale else {})}

    with TestClient(web_server.app) as client:
        form = {"mode": "video", "model": "ltx-8", "prompt": "a lake", **{key: str(value) for key, value in fields.items()}}
        assert client.post("/api/generate", data=form).status_code == 200
    payload = _memory_fit(fields).json()

    (job,) = jobs
    assert (payload["width"], payload["height"], payload["frames"]) == (job["args"].width, job["args"].height, job["args"].num_frames)
    assert (payload["width"], payload["height"], payload["frames"]) == ((704, 448, 49) if upscale else (736, 448, 49))


def test_memory_fit_route_gives_no_fit_without_a_budget(clip_fit, monkeypatch):
    monkeypatch.setattr(web_server, "memory_budget", lambda: None)
    monkeypatch.setattr(web_server, "clip_memory_fit", real_clip_memory_fit)

    payload = _memory_fit({"frames": 97}).json()

    assert payload["memory_fit"] is None
    assert (payload["width"], payload["height"], payload["frames"]) == (736, 448, 49)


def test_memory_fit_route_gives_no_fit_for_a_model_that_cannot_be_estimated(clip_fit, monkeypatch):
    """Not LTX MLX, not downloaded, or the text encoder missing: the memory check finds nothing to estimate."""
    monkeypatch.setattr(web_server, "clip_memory_fit", lambda *_args, **_kwargs: None)

    response = _memory_fit({})

    assert response.status_code == 200
    assert response.json()["memory_fit"] is None


def test_memory_fit_route_answers_while_a_job_runs(clip_fit, monkeypatch):
    monkeypatch.setattr(web_server.web_runner, "has_active_jobs", lambda: True)

    assert _memory_fit({}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        [1, 2],
        {"model": 5},
        {"ratio": 5},
        {"ratio": "3:1"},
        {"size": "huge"},
        {"width": "704"},
        {"width": 0},
        {"width": 7.5},
        {"height": True},
        {"frames": -9},
        {"low_memory": "yes"},
        {"upscale": 4},
        {"upscale": True},
    ],
)
def test_memory_fit_route_refuses_a_malformed_body(clip_fit, body):
    response = _memory_fit(body)

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert clip_fit.calls == []


def test_memory_fit_route_refuses_a_body_that_is_not_json(clip_fit):
    with TestClient(web_server.app) as client:
        response = client.post("/api/video/memory-fit", content=b"not json", headers={"content-type": "application/json"})

    assert response.status_code == 422


def test_memory_fit_route_refuses_a_size_that_is_too_small(clip_fit, monkeypatch):
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 16, "height": 448, "num_frames": 49})

    assert _memory_fit({}).status_code == 422


class TestLtxConversion:
    """LTX checkpoints convert as a background job that holds the quantize slot until it ends."""

    def _hold(self, monkeypatch):
        """Record the job instead of running it; return the started (args, slot) pairs."""
        started: list[tuple[list[str], object]] = []
        monkeypatch.setattr("zvisiongenerator.web.ltx_conversion.start_conversion", lambda args, name, slot: started.append((args, slot)))
        monkeypatch.setattr(web_server, "_run_model_management_command", lambda args: pytest.fail("LTX never runs as a blocking convert"))
        return started

    def test_the_conversion_starts_in_the_background_with_its_level_and_distill(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "ltx-2.3", "name": "myft", "quantize": "8", "distill": True})

        assert response.status_code == 200
        assert response.json()["background"] is True
        assert started[0][0] == ["model", "--input", str(checkpoint.resolve()), "--model-type", "ltx-2.3", "--name", "myft", "--quantize", "8", "--distill"]
        with pytest.raises(web_server.JobConflictError):
            web_server._reject_while_busy()  # the slot stays held while the job runs
        started[0][1].close()
        web_server._reject_while_busy()

    def test_a_quantize_level_is_required(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "ltx-2.3", "quantize": ""})

        assert response.status_code == 400
        assert started == []

    def test_it_waits_for_running_jobs(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)
        monkeypatch.setattr(web_server.web_runner, "_find_active_exclusive_job_id", lambda: "busy")

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "ltx-2.3", "quantize": "4"})

        assert response.status_code == 409
        assert started == []

    def test_the_job_releases_the_slot_and_reports_the_converters_error(self, monkeypatch):
        from contextlib import ExitStack

        from zvisiongenerator.web import ltx_conversion

        class _Process:
            stdout = iter(["Converting: Writing the transformer (3/10)\n", "Error: A model named 'myft' already exists.\n"])

            def wait(self):
                return 1

        monkeypatch.setattr(ltx_conversion.subprocess, "Popen", lambda *a, **k: _Process())
        released: list[bool] = []
        slot = ExitStack()
        slot.callback(lambda: released.append(True))

        ltx_conversion.start_conversion(["model"], "myft", slot)
        for _ in range(100):
            if released:
                break
            threading.Event().wait(0.01)

        status = ltx_conversion.conversion_status()
        assert released == [True]
        assert status["state"] == "failed"
        assert status["message"] == "A model named 'myft' already exists."
        assert (status["stage"], status["done"], status["total"]) == ("Writing the transformer", 3, 10)

    def test_a_killed_conversion_says_how_it_stopped_not_its_last_progress_line(self):
        from zvisiongenerator.web import ltx_conversion

        progress = ["Converting: Writing the transformer (1200/4186)"]

        assert "signal 9" in ltx_conversion._failure_message(progress, -9)
        assert ltx_conversion._failure_message([*progress, "RuntimeError: CUDA error"], 1) == "RuntimeError: CUDA error"

    def test_inspect_returns_what_the_form_preselects(self, monkeypatch, tmp_path):
        from zvisiongenerator.converters.ltx_convert import LtxInspection, distill_advice

        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        found = LtxInspection(checkpoint=SimpleNamespace(fp8_weights=440, bundles_components=True), distillation="dev", advice="Use --distill.")
        monkeypatch.setattr("zvisiongenerator.web.ltx_conversion.inspect_ltx", lambda path: found)
        monkeypatch.setattr("zvisiongenerator.web.ltx_conversion.recommended_ltx_quantize", lambda config: 8)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert/inspect", json={"input_path": str(checkpoint)})

        assert response.status_code == 200
        assert response.json() == {"layout": "full", "fp8_weights": 440, "distillation": "dev", "advice": distill_advice("dev", "Distill on"), "recommend_distill": True, "recommended_quantize": 8}

    def test_inspect_refuses_a_file_that_is_not_ltx(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert/inspect", json={"input_path": str(checkpoint)})

        assert response.status_code == 400


@pytest.mark.parametrize(("unified_gib", "expected"), [(32, 8), (64, 8), (24, 4)])
def test_a_mac_converts_ltx_at_q8_from_32_gb_of_unified_memory(monkeypatch, unified_gib, expected):
    from zvisiongenerator.utils.model_memory import MemoryBudget
    from zvisiongenerator.web import ltx_conversion

    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(unified_gib * 1024**3 * 3 // 4, unified_bytes=unified_gib * 1024**3))

    assert ltx_conversion.recommended_ltx_quantize({}) == expected


def test_converter_subprocesses_print_in_utf8(monkeypatch):
    from zvisiongenerator.web import ltx_conversion

    seen: list[dict] = []

    class _Process:
        stdout = iter([])

        def wait(self):
            return 0

    def _popen(*_args, **kwargs):
        seen.append(kwargs)
        return _Process()

    def _run(*_args, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(ltx_conversion.subprocess, "Popen", _popen)
    monkeypatch.setattr(web_server.subprocess, "run", _run)
    ltx_conversion._run(["model"], contextlib.nullcontext())
    web_server._run_model_management_command(["model"])

    assert [(kwargs["encoding"], kwargs["env"]["PYTHONIOENCODING"]) for kwargs in seen] == [("utf-8", "utf-8")] * 2


class TestModelDownloadRoutes:
    """The Models-page download lane routes: plan, queue, poll and cancel."""

    _FAKE_PARTS = (DownloadTarget("owner/zit"),)

    @pytest.fixture
    def lane(self, monkeypatch, tmp_path):
        """A fresh lane whose ensure step is a fake that waits for ``release`` and honours cancel."""
        release = threading.Event()
        calls: list[tuple[DownloadTarget, ...]] = []

        def _ensure(targets, *, flights, report, cancelled, **_):
            calls.append(tuple(targets))
            while not release.wait(0.01):
                if cancelled():
                    return "stopped"
            return "ready"

        lane = DownloadLane(flights=web_server.download_flights, ensure=_ensure)
        web_config = _make_web_config()
        web_config.data_dir = str(tmp_path)
        web_config.image_inventory = (ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),)
        web_config.video_inventory = (VideoInventoryEntry(name="ltx-8", family="ltx", supports_i2v=True, source="alias", resolved_path="owner/ltx"),)
        web_config.app_config["prompt_enhancer"] = {"model": "owner/enhancer"}
        web_config.app_config["model_labels"] = {"zit": {"name": "Z-Image Turbo"}, "ltx-8": {"name": "LTX-2.3 8-bit"}}
        monkeypatch.setattr(web_server, "download_lane", lane)
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(download_plan_module, "image_plan", lambda *a, **k: self._FAKE_PARTS)
        monkeypatch.setattr(download_plan_module, "video_plan", lambda *a, **k: (DownloadTarget("owner/ltx"),))
        monkeypatch.setattr(download_plan_module, "enhancer_plan", lambda *a, **k: (DownloadTarget("owner/enhancer"),))
        lane.release = release
        lane.calls = calls
        yield lane
        release.set()
        lane.shutdown(join_timeout=5.0)

    @pytest.fixture
    def client(self, lane):
        with TestClient(web_server.app) as client:
            yield client

    @staticmethod
    def _wait_for(client, item_id: str, state: str) -> dict:
        for _ in range(500):
            items = client.get("/api/models/downloads").json()["items"]
            item = next((entry for entry in items if entry["id"] == item_id), None)
            if item is not None and item["state"] == state:
                return item
            threading.Event().wait(0.01)
        raise AssertionError(f"{item_id} never reached {state}")

    def test_the_lane_is_empty_to_begin_with_and_the_route_is_not_taken_by_the_delete_pattern(self, client):
        response = client.get("/api/models/downloads")

        assert response.status_code == 200
        assert response.json() == {"items": []}

    @pytest.mark.parametrize(("kind", "name", "label"), [("model", "zit", "Z-Image Turbo"), ("model", "ltx-8", "LTX-2.3 8-bit"), ("enhancer", "fast", None)])
    def test_posting_a_download_queues_an_item_keyed_by_kind_and_name(self, client, lane, kind, name, label):
        response = client.post("/api/models/download", json={"kind": kind, "name": name})

        assert response.status_code == 200
        item = response.json()
        assert (item["kind"], item["name"], item["state"]) == (kind, name, item["state"])
        assert item["state"] in {"queued", "running"}
        assert item["id"]
        assert item["message"] is None
        assert item["label"] == (label or item["label"])
        assert client.get("/api/models/downloads").json()["items"][0]["id"] == item["id"]

    def test_the_lane_runs_the_rows_plan_and_finishes_the_item(self, client, lane):
        item = client.post("/api/models/download", json={"kind": "model", "name": "zit"}).json()
        lane.release.set()

        done = self._wait_for(client, item["id"], "done")

        assert lane.calls == [self._FAKE_PARTS]
        assert done["message"] is None

    def test_posting_the_same_row_again_returns_the_existing_item(self, client, lane):
        first = client.post("/api/models/download", json={"kind": "model", "name": "zit"}).json()
        second = client.post("/api/models/download", json={"kind": "model", "name": "zit"}).json()
        other = client.post("/api/models/download", json={"kind": "model", "name": "ltx-8"}).json()

        assert second["id"] == first["id"]
        assert other["id"] != first["id"]
        assert other["state"] == "queued"
        assert other["queue_position"] == 1

    def test_unknown_rows_and_kinds_are_404_and_bad_bodies_422(self, client):
        assert client.post("/api/models/download", json={"kind": "model", "name": "nope"}).status_code == 404
        assert client.post("/api/models/download", json={"kind": "enhancer", "name": "nope"}).status_code == 404
        assert client.post("/api/models/download", json={"kind": "lora", "name": "zit"}).status_code == 404
        assert client.post("/api/models/download", json={"kind": "model"}).status_code == 422
        assert client.post("/api/models/download", json=["model", "zit"]).status_code == 422
        assert client.get("/api/models/downloads").json() == {"items": []}

    def test_cancelling_a_queued_item_cancels_it_and_a_running_one_stops(self, client, lane):
        running = client.post("/api/models/download", json={"kind": "model", "name": "zit"}).json()
        queued = client.post("/api/models/download", json={"kind": "model", "name": "ltx-8"}).json()

        cancelled_queued = client.post(f"/api/models/downloads/{queued['id']}/cancel")
        cancelled_running = client.post(f"/api/models/downloads/{running['id']}/cancel")

        assert cancelled_queued.status_code == 200
        assert cancelled_queued.json()["state"] == "cancelled"
        assert cancelled_running.status_code == 200
        self._wait_for(client, running["id"], "cancelled")
        assert lane.calls == [self._FAKE_PARTS]

    def test_cancelling_an_unknown_item_is_404(self, client):
        assert client.post("/api/models/downloads/dl-999/cancel").status_code == 404

    def test_a_shut_down_lane_refuses_new_downloads_with_a_retryable_status(self, client, lane):
        lane.shutdown()

        assert client.post("/api/models/download", json={"kind": "model", "name": "zit"}).status_code == 503

    def test_a_running_download_does_not_block_generation_jobs(self, client, lane, monkeypatch):
        client.post("/api/models/download", json={"kind": "model", "name": "zit"})
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)

        web_server._reject_while_busy()

        assert client.get("/api/jobs").json()["active_job"] is None
        assert web_server.web_runner.has_active_jobs() is False

    def test_the_lifespan_shuts_the_lane_down(self, lane):
        with TestClient(web_server.app):
            assert lane.snapshot() == {"items": []}

        with pytest.raises(RuntimeError):
            lane.submit(kind="model", name="zit", label="zit", plan=lambda: ())


class TestModelDownloadPlanRoute:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        web_config = _make_web_config()
        web_config.data_dir = str(tmp_path)
        web_config.image_inventory = (ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),)
        web_config.app_config["prompt_enhancer"] = {"model": "owner/enhancer"}
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(download_plan_module, "image_plan", lambda *a, **k: (DownloadTarget("owner/zit"), DownloadTarget("owner/vae")))
        monkeypatch.setattr(download_plan_module, "enhancer_plan", lambda *a, **k: (DownloadTarget("owner/enhancer"),))
        monkeypatch.setattr(download_plan_module, "target_present_offline", lambda target: False)
        self.checks: dict[str, object] = {"owner/zit": TargetCheck(3, 3_000), "owner/vae": TargetCheck(1, 500), "owner/enhancer": TargetCheck(2, 40)}
        monkeypatch.setattr(
            web_server,
            "plan_summary",
            functools.partial(
                real_plan_summary,
                check=lambda target: self._check(target),
                present=lambda target: False,
                space=lambda items, *, cache_dir: SpaceNeed(cache_dir, sum(check.missing_bytes or 0 for _, check in items), 1_000),
                cache_dir=lambda: tmp_path / "hub",
            ),
        )
        with TestClient(web_server.app) as client:
            yield client

    def _check(self, target):
        outcome = self.checks[target.repo_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def test_returns_the_parts_total_and_space_for_a_model_row(self, client, tmp_path):
        response = client.get("/api/models/download/plan", params={"kind": "model", "name": "zit"})

        assert response.status_code == 200
        assert response.json() == {
            "kind": "model",
            "name": "zit",
            "label": "zit",
            "parts": [{"repo": "owner/zit", "files": 3, "bytes": 3_000}, {"repo": "owner/vae", "files": 1, "bytes": 500}],
            "total_bytes": 3_500,
            "space": {"path": str(tmp_path / "hub"), "needed_bytes": 3_500, "free_bytes": 1_000, "enough": False},
        }

    def test_an_enhancer_row_is_looked_up_by_tier(self, client):
        payload = client.get("/api/models/download/plan", params={"kind": "enhancer", "name": "fast"}).json()

        assert (payload["kind"], payload["name"], payload["total_bytes"]) == ("enhancer", "fast", 40)

    def test_a_row_with_nothing_missing_has_no_parts_and_no_space(self, client):
        self.checks.update({"owner/zit": TargetCheck(0, 0), "owner/vae": TargetCheck(0, 0)})

        payload = client.get("/api/models/download/plan", params={"kind": "model", "name": "zit"}).json()

        assert (payload["parts"], payload["total_bytes"], payload["space"]) == ([], 0, None)

    def test_unknown_row_is_404(self, client):
        assert client.get("/api/models/download/plan", params={"kind": "model", "name": "nope"}).status_code == 404
        assert client.get("/api/models/download/plan", params={"kind": "enhancer", "name": "smart"}).status_code == 404
        assert client.get("/api/models/download/plan", params={"kind": "bogus", "name": "zit"}).status_code == 404

    def test_a_missing_query_field_is_422(self, client):
        assert client.get("/api/models/download/plan", params={"kind": "model"}).status_code == 422

    def test_an_unreachable_hub_is_503(self, client):
        self.checks["owner/vae"] = None

        response = client.get("/api/models/download/plan", params={"kind": "model", "name": "zit"})

        assert response.status_code == 503
        assert response.json()["detail"]

    def test_a_plan_that_cannot_be_made_is_503(self, client, monkeypatch):
        monkeypatch.setattr(download_plan_module, "image_plan", lambda *a, **k: None)

        assert client.get("/api/models/download/plan", params={"kind": "model", "name": "zit"}).status_code == 503

    def test_a_hub_refusal_is_424_and_names_the_repo(self, client):
        self.checks["owner/vae"] = RuntimeError("Could not download owner/vae: gated")

        response = client.get("/api/models/download/plan", params={"kind": "model", "name": "zit"})

        assert response.status_code == 424
        assert "owner/vae" in response.json()["detail"]

    def test_a_refusal_while_planning_the_row_is_424_on_both_routes(self, client, monkeypatch):
        line = "owner/zit needs a Hugging Face token: accept its license at https://huggingface.co/owner/zit, then run `hf auth login`."

        def refuse(*_args, **_kwargs):
            raise RuntimeError(line)

        monkeypatch.setattr(download_plan_module, "image_plan", refuse)

        plan = client.get("/api/models/download/plan", params={"kind": "model", "name": "zit"})
        queued = client.post("/api/models/download", json={"kind": "model", "name": "zit"})

        assert (plan.status_code, queued.status_code) == (424, 424)
        assert plan.json()["detail"] == line
        assert queued.json()["detail"] == line


class TestJobLogRoutes:
    """The job log routes: paging and filters, single delete, clear, and store failures."""

    @pytest.fixture()
    def path(self, tmp_path, monkeypatch) -> Path:
        log_path = tmp_path / "jobs.jsonl"
        monkeypatch.setattr(web_server, "job_log", JobLog(lambda: log_path))
        return log_path

    @staticmethod
    def _seed(entries: list[dict]) -> None:
        for entry in entries:
            web_server.job_log.append(entry)

    @staticmethod
    def _entry(job_id: str, status: str = "completed", failed_runs: int = 0) -> dict:
        return {"job_id": job_id, "status": status, "failed_runs": failed_runs}

    def test_route_status_values_match_the_filter_constant(self):
        schema = web_server.app.openapi()["paths"]["/api/jobs/log"]["get"]
        status_schema = next(param["schema"] for param in schema["parameters"] if param["name"] == "status")

        assert tuple(status_schema["enum"]) == JOB_LOG_STATUS_FILTERS

    @pytest.mark.parametrize("route", [web_server.api_job_log, web_server.api_delete_job_log_entry, web_server.api_clear_job_log])
    def test_log_routes_run_in_the_threadpool(self, route):
        assert not inspect.iscoroutinefunction(route)

    def test_listing_returns_the_page_keys_newest_first(self, path):
        self._seed([self._entry("a"), self._entry("b"), self._entry("c")])

        with TestClient(web_server.app) as client:
            response = client.get("/api/jobs/log")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"entries", "page", "total_pages", "total_count", "page_size"}
        assert [entry["job_id"] for entry in body["entries"]] == ["c", "b", "a"]
        assert (body["page"], body["total_pages"], body["total_count"], body["page_size"]) == (1, 1, 3, 50)

    def test_empty_log_is_one_empty_page(self, path):
        with TestClient(web_server.app) as client:
            body = client.get("/api/jobs/log").json()

        assert body == {"entries": [], "page": 1, "total_pages": 1, "total_count": 0, "page_size": 50}

    def test_paging_and_page_clamp(self, path):
        self._seed([self._entry(f"j{i:03d}") for i in range(120)])

        with TestClient(web_server.app) as client:
            second = client.get("/api/jobs/log", params={"page": 2}).json()
            clamped = client.get("/api/jobs/log", params={"page": 99}).json()

        assert second["page"] == 2
        assert second["total_pages"] == 3
        assert [entry["job_id"] for entry in second["entries"]][0] == "j069"
        assert len(second["entries"]) == 50
        assert clamped["page"] == 3
        assert [entry["job_id"] for entry in clamped["entries"]] == [f"j{i:03d}" for i in range(19, -1, -1)]

    def test_status_filters_include_partial_failures(self, path):
        self._seed(
            [
                self._entry("done"),
                self._entry("failed", status="failed"),
                self._entry("partial", failed_runs=2),
                self._entry("cancelled", status="cancelled"),
                self._entry("cancelled-partial", status="cancelled", failed_runs=1),
            ]
        )

        with TestClient(web_server.app) as client:
            ids = {status: [entry["job_id"] for entry in client.get("/api/jobs/log", params={"status": status}).json()["entries"]] for status in ("all", "failed", "completed", "cancelled")}
            counted = client.get("/api/jobs/log", params={"status": "failed"}).json()

        assert ids["failed"] == ["cancelled-partial", "partial", "failed"]
        assert ids["completed"] == ["partial", "done"]
        assert ids["cancelled"] == ["cancelled-partial", "cancelled"]
        assert len(ids["all"]) == 5
        assert counted["total_count"] == 3

    @pytest.mark.parametrize("params", [{"status": "bogus"}, {"page": 0}, {"page": -1}, {"page": "x"}])
    def test_invalid_query_is_422(self, path, params):
        with TestClient(web_server.app) as client:
            assert client.get("/api/jobs/log", params=params).status_code == 422

    def test_delete_one_entry(self, path):
        self._seed([self._entry("a"), self._entry("b")])

        with TestClient(web_server.app) as client:
            deleted = client.delete("/api/jobs/log/a")
            again = client.delete("/api/jobs/log/a")
            remaining = client.get("/api/jobs/log").json()

        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": "a"}
        assert again.status_code == 404
        assert again.json()["detail"] == "Unknown log entry: a"
        assert [entry["job_id"] for entry in remaining["entries"]] == ["b"]

    def test_delete_on_a_missing_file_is_404(self, path):
        with TestClient(web_server.app) as client:
            assert client.delete("/api/jobs/log/nope").status_code == 404

    def test_clear_returns_the_removed_count(self, path):
        self._seed([self._entry("a"), self._entry("b")])

        with TestClient(web_server.app) as client:
            first = client.delete("/api/jobs/log")
            second = client.delete("/api/jobs/log")

        assert first.json() == {"deleted_count": 2}
        assert second.json() == {"deleted_count": 0}

    def test_clear_on_a_missing_file_is_zero(self, path):
        with TestClient(web_server.app) as client:
            response = client.delete("/api/jobs/log")

        assert response.status_code == 200
        assert response.json() == {"deleted_count": 0}
        assert not path.exists()

    def test_log_routes_are_not_swallowed_by_job_id_routes(self, path):
        with TestClient(web_server.app) as client:
            assert client.get("/api/jobs/log").json()["total_count"] == 0
            assert client.get("/api/jobs").json() == {"active_job": None, "queued_jobs": []}

    @pytest.mark.parametrize(("method", "url"), [("get", "/api/jobs/log"), ("delete", "/api/jobs/log/a"), ("delete", "/api/jobs/log")])
    def test_store_os_errors_become_500_with_a_detail(self, monkeypatch, method, url):
        def _broken(*_args, **_kwargs):
            raise PermissionError(13, "Permission denied")

        for name in ("page", "delete", "clear"):
            monkeypatch.setattr(web_server.job_log, name, _broken)

        with TestClient(web_server.app) as client:
            response = getattr(client, method)(url)

        assert response.status_code == 500
        assert response.json()["detail"] == "Could not read or write the job log: Permission denied"

    def test_log_that_is_a_directory_is_500(self, tmp_path, monkeypatch):
        monkeypatch.setattr(web_server, "job_log", JobLog(lambda: tmp_path))

        with TestClient(web_server.app) as client:
            assert client.get("/api/jobs/log").status_code == 500

    def test_the_server_runner_writes_to_the_server_job_log(self):
        assert web_server.web_runner._job_log is web_server.job_log
