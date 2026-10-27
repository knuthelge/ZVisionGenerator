"""Focused route tests for the Phase A web backend contracts."""

from __future__ import annotations

import contextlib
import functools
from dataclasses import replace
import inspect
import io
import json
import os
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import quote

from fastapi.testclient import TestClient
import huggingface_hub
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest


from tests._server_helpers import post_video, stub_video_submission, make_resolved_image_defaults, make_web_config, make_workspace_bootstrap_view, patch_image_submit_dependencies, write_png

from zvisiongenerator.downloads.space import SpaceNeed
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import model_download_plan as download_plan_module
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.model_download_plan import plan_summary as real_plan_summary
from zvisiongenerator.web import model_inventory as model_inventory_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import video_clip as video_clip_module
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.gallery import list_gallery_assets
from zvisiongenerator.web.job_contract import public_job_snapshot
from zvisiongenerator.web.job_log import JOB_LOG_STATUS_FILTERS, JobLog, JobLogError, build_job_log_entry
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.h3_converted import write_converted_h3
from zvisiongenerator.utils.ltx_converted import write_converted_ltx
from zvisiongenerator.utils import app_db
from zvisiongenerator.seedvr2_loader import seedvr2_weights_dir
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
from zvisiongenerator.utils.provenance import embed_png_config


@pytest.fixture(autouse=True)
def _gemma_repo(monkeypatch):
    """Give the minimal test configs the macOS LTX text encoder reference the Models page reads."""
    monkeypatch.setattr(workspace_api_module, "ltx_mlx_text_encoder_repo_id", lambda config: "owner/gemma")


def _write_png_with_config(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    embed_png_config(info, payload)
    Image.new("RGB", (8, 8), "teal").save(path, pnginfo=info)


def _png_upload_bytes() -> io.BytesIO:
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), "teal").save(payload, format="PNG")
    payload.seek(0)
    return payload


def test_phase_a_routes_share_config_and_path_authority(monkeypatch):
    """Config, workspace, and models routes should expose the same backend-owned authority."""
    web_config = make_web_config()
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
    monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: make_workspace_bootstrap_view())
    monkeypatch.setattr(web_server.web_runner, "list_jobs", lambda: {"active_job": active_job_snapshot, "queued_jobs": []})

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

    assert schema_fields["ui.output_dir"]["effective_value"] == "/tmp/outputs"

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


class TestHuggingFaceTokenPayload:
    """``/api/config`` and ``/api/models`` report the token ``huggingface_hub`` finds and where it comes from."""

    @staticmethod
    def _payloads(monkeypatch):
        web_config = make_web_config()
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
    web_config = make_web_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: make_workspace_bootstrap_view())
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


def test_submit_image_job_threads_json_prompt_and_first_sigma_to_authoritative_args(monkeypatch, tmp_path):
    """JSON-only submissions should thread their authoritative args and prompt payloads."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.default_models = WebUiDefaultModels(image="ideo", video="ltx-8")
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=make_resolved_image_defaults(
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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=True, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)
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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=make_resolved_image_defaults(supports_json_prompt=True, supports_first_sigma=True),
        submitted=submitted,
    )
    web_server._submit_image_job({"model": "ideo", "json_prompt": '{\n  "high_level_description": "Café"\n}'}, web_config)
    assert submitted[0]["prompts_data"] == {"prompt": [('{"high_level_description":"Café"}', None)]}


_IDEOGRAM = (
    "ideogram4",
    {
        "supports_quantize": False,
        "supports_img2img": False,
        "supports_upscale": False,
        "supports_json_prompt": True,
        "supports_first_sigma": True,
        "dimension_min": 256,
        "dimension_max": 2048,
        "dimension_step": 16,
    },
)
_ZIMAGE = ("zimage", {"supports_json_prompt": False, "supports_first_sigma": False})
_JSON_CAPTION = '{"high_level_description":"x"}'


@pytest.mark.parametrize(
    ("model", "form", "message"),
    [
        (_IDEOGRAM, {"prompt": "plain prompt", "json_prompt": _JSON_CAPTION}, "Provide either a prompt or a structured JSON caption, not both."),
        (_IDEOGRAM, {"prompt_source": "file", "json_prompt": _JSON_CAPTION}, "A structured JSON caption cannot be combined with prompt-file mode."),
        (_IDEOGRAM, {"json_prompt": "not json"}, "json_prompt must be a JSON object"),
        (_IDEOGRAM, {"json_prompt": "[]"}, "json_prompt must be a JSON object"),
        (_IDEOGRAM, {"prompt": "hello", "first_sigma": "0"}, "first_sigma must be in (0.0, 2.0]"),
        (_IDEOGRAM, {"prompt": "hello", "first_sigma": "2.5"}, "first_sigma must be in (0.0, 2.0]"),
        (_ZIMAGE, {"json_prompt": _JSON_CAPTION}, "This model does not support structured JSON captions."),
        (_ZIMAGE, {"prompt": "hello", "first_sigma": "1.005"}, "This model does not support the first-step sigma control."),
        (_ZIMAGE, {"prompt": "a fox", "enhance_auto": "true", "enhance_format": "json"}, "does not support structured JSON captions"),
        (_IDEOGRAM, {"prompt": "hello", "image_path": "<reference>"}, "does not support reference-image (img2img) steering."),
        (_IDEOGRAM, {"prompt": "hello", "upscale": "2"}, "does not support upscaling."),
        (_IDEOGRAM, {"prompt": "hello", "quantize": "4"}, "This model does not support quantization."),
        (_IDEOGRAM, {"prompt": "hello", "ratio": "16:9", "size": "xl"}, "must be between"),
    ],
)
def test_submit_image_job_refuses_an_invalid_request_before_queueing(monkeypatch, tmp_path, model, form, message):
    """Each refusal names its cause and queues nothing."""
    family, capabilities = model
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.app_config["sizes"]["16:9"] = {"xl": {"width": 2112, "height": 1184}}
    web_config.image_ratios = ("2:3", "16:9")
    web_config.image_size_options["16:9"] = ("xl",)
    reference = tmp_path / "reference.png"
    write_png(reference)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family=family, is_distilled=False, size=None), defaults=make_resolved_image_defaults(**capabilities), submitted=submitted)
    form = {key: str(reference) if value == "<reference>" else value for key, value in form.items()}

    with pytest.raises(ValueError) as refused:
        web_server._submit_image_job({"model": "ideo", **form}, web_config)

    assert message in str(refused.value)
    assert submitted == []


def test_submit_image_job_queues_a_plain_job_with_its_public_context(monkeypatch, tmp_path):
    """The job carries its prompt, model, settings and authoritative args from the moment it is queued, whatever output folder the browser sent."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)
    form = {"model": "zit", "prompt": "hello", "negative_prompt": "avoid blur", "quantize": "4", "steps": "9", "workflow": "txt2img", "output": str(tmp_path / "stale-browser-output")}

    response = web_server._submit_image_job(form, web_config)

    (job,) = submitted
    context = job["context"]
    assert response["job_id"] == "job-123"
    assert context == response["context"]
    assert (context["prompt"], context["model"], context["model_label"], context["workflow"]) == ("hello", "zit", "zit", "txt2img")
    assert "steps" in context["meta"]
    assert context["output_dir"] == str(tmp_path)
    assert context["settings"] == form
    assert job["args"].quantize == 4
    assert job["args"].json_prompt_enabled is False
    assert job["args"].first_sigma is None
    assert job["prompts_data"] == {"web": [("hello", "avoid blur")]}
    assert job["request"].prompt == "hello"


@pytest.mark.parametrize(
    ("form", "expected_steps", "expected_guidance", "expected_steps_explicit", "expected_guidance_explicit"),
    [
        ({"model": "ideo", "prompt": "hello", "steps": "28", "guidance": "6.0"}, 28, 6.0, True, True),
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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    defaults = make_resolved_image_defaults(
        supports_negative_prompt=False,
        supports_img2img=False,
        supports_upscale=False,
        supports_json_prompt=True,
        supports_first_sigma=True,
        dimension_min=256,
        dimension_max=2048,
        dimension_step=16,
    )
    patch_image_submit_dependencies(
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


def test_submit_image_job_drops_scheduler_for_models_with_their_own_sampler(monkeypatch, tmp_path):
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="krea2", is_distilled=False, size=None),
        defaults=make_resolved_image_defaults(supports_scheduler=False),
        submitted=submitted,
    )
    monkeypatch.setattr(web_server, "resolve_defaults", lambda _info, _config, cli_overrides, _backend: {**make_resolved_image_defaults(supports_scheduler=False), **cli_overrides})

    web_server._submit_image_job({"model": "krea2", "prompt": "hello", "scheduler": "beta"}, web_config)

    assert submitted[0]["args"].scheduler is None
    assert submitted[0]["request"].scheduler is None


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
    monkeypatch.setattr(web_server, "load_web_config", make_web_config)
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
    web_config = make_web_config()
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


def test_video_submission_threads_scene_into_args_and_prompts(monkeypatch, tmp_path):
    """A scene and prompt reach the job args, with the user's inputs recorded when sent."""
    submitted = stub_video_submission(monkeypatch, tmp_path)

    response = post_video({"scene": " A castle at dusk. ", "prompt": "A knight rides in.", "scene_input": "castle", "prompt_input": "knight"})

    assert response.status_code == 200
    (job,) = submitted
    args = job["args"]
    assert (args.scene, args.prompt, args.scene_input, args.prompt_input) == ("A castle at dusk.", "A knight rides in.", "castle", "knight")
    assert job["prompts_data"] == {"web": [("A knight rides in.", None)]}
    assert job["scene_by_set"] is None


def test_text_to_video_submission_without_scene_keeps_the_scene_and_extend_args_empty(monkeypatch, tmp_path):
    submitted = stub_video_submission(monkeypatch, tmp_path)

    response = post_video({"prompt": "A knight rides in."})

    assert response.status_code == 200
    args = submitted[0]["args"]
    assert (args.scene, args.scene_input, args.prompt_input) == (None, None, None)
    assert not hasattr(args, "extend_source")
    assert submitted[0]["context"]["job_type"] == "Text to Video"


def test_video_submission_accepts_a_scene_without_a_prompt(monkeypatch, tmp_path):
    """Scene OR prompt is enough: a scene-only inline submission plans one empty-prompt entry."""
    submitted = stub_video_submission(monkeypatch, tmp_path)

    response = post_video({"scene": "A castle at dusk."})

    assert response.status_code == 200
    (job,) = submitted
    assert job["args"].scene == "A castle at dusk."
    assert job["args"].prompt == ""
    assert job["prompts_data"] == {"web": [("", None)]}
    assert job["context"]["prompt"] == "A castle at dusk."


@pytest.mark.parametrize("data", [{}, {"scene": "   ", "prompt": ""}, {"prompt": "  "}])
def test_video_submission_refuses_when_scene_and_prompt_are_both_empty(monkeypatch, tmp_path, data):
    submitted = stub_video_submission(monkeypatch, tmp_path)

    response = post_video(data)

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert "prompt" in response.json()["detail"]
    assert submitted == []


def test_video_submission_accepts_action_only_settings_that_keep_everything(monkeypatch, tmp_path):
    """``parts=action`` rewrites the action to fit the scene, so unchanged style, mood and length are not a no-op."""
    submitted = stub_video_submission(monkeypatch, tmp_path)
    settings = json.dumps({"style": "keep", "mood": "keep", "details": [], "length": "same", "motion": [], "sound": [], "parts": "action"})

    response = post_video({"scene": "A night street.", "enhance_auto": "on", "enhance_settings": settings, "prompt": "She walks away."})

    assert response.status_code == 200
    assert submitted[0]["args"].enhance.parts == "action"


def test_video_submission_refuses_an_unknown_parts_value(monkeypatch, tmp_path):
    submitted = stub_video_submission(monkeypatch, tmp_path)

    response = post_video({"scene": "A night street.", "enhance_auto": "on", "enhance_settings": json.dumps({"parts": "scene"}), "prompt": "She walks away."})

    assert response.status_code == 422 and "both, action" in response.json()["detail"]
    assert submitted == []


def test_image_submission_ignores_the_scene_field(monkeypatch, tmp_path):
    """Images keep the prompt requirement and never read a scene."""
    web_config = make_web_config()
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
    submitted = stub_video_submission(monkeypatch, tmp_path)
    path = tmp_path / "prompts.yaml"
    path.write_text("shots:\n  - scene: A castle.\n    prompt: first\n  - prompt: second\n", encoding="utf-8")

    response = post_video({"prompt_source": "file", "prompts_file": str(path), "prompt_option_id": ["shots:0", "shots:1"], "scene": "Night."})

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
    monkeypatch.setattr(web_server, "load_web_config", make_web_config)
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
    monkeypatch.setattr(web_server, "load_web_config", make_web_config)

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


@pytest.mark.usefixtures("hub_offline")
def test_models_route_uses_shared_alias_inventory(monkeypatch, tmp_path):
    """Models route should expose the same alias-backed inventory authority as config loading."""
    web_config = make_web_config()
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
        lambda _data_dir: [
            SimpleNamespace(name="local-video", family="ltx", supports_i2v=True),
            SimpleNamespace(name="converted-video", family="ltx", supports_i2v=True),
            SimpleNamespace(name="converted-h3", family="ltx", supports_i2v=True),  # the family plays no part in the level
        ],
    )
    converted = tmp_path / "models" / "converted-video"
    converted.mkdir(parents=True)
    write_converted_ltx(converted, {"bits": 8, "storage": "mlx-affine-g64", "block_matrices": 1, "transformer_rest": 1})
    converted_h3 = tmp_path / "models" / "converted-h3"
    converted_h3.mkdir(parents=True)
    write_converted_h3(converted_h3, {"storage": "nf4-folded", "turbo": "merged"})
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
    assert video_models["converted-h3"]["converted_bits"] == 4
    assert video_models["local-video"]["converted_bits"] is None
    assert video_models["alias-video"]["converted_bits"] is None
    assert isinstance(payload["stored_quants_supported"], bool)


_SEEDVR2_7B = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)


def _models_route_with_seedvr2(monkeypatch, tmp_path, *, copy_files: dict[str, int] | None):
    """Serve ``GET /api/models`` with SeedVR2 on, its 7B copy holding sparse *copy_files* of the given sizes (or absent when ``None``)."""
    copy = seedvr2_weights_dir("seedvr2-7b", tmp_path / "models", MLX_WEIGHTS)
    if copy_files is not None:
        for name, size in copy_files.items():
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            (copy / name).touch()
            os.truncate(copy / name, size)
    settings = SeedVR2Settings(model="auto", softness=0.75, video_softness=0.0, reference_megapixels=16, tiers=(_SEEDVR2_7B,))
    web_config = make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.models_dir = str(tmp_path / "models")
    web_config.seedvr2 = SeedVR2Setup(settings=settings, tier=_SEEDVR2_7B, budget_bytes=None, downloaded=copy_files is not None, weights=MLX_WEIGHTS)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
    return web_config


def test_models_route_lists_the_downloaded_seedvr2_tier_with_its_disk_use(monkeypatch, tmp_path):
    _models_route_with_seedvr2(monkeypatch, tmp_path, copy_files={"transformer/w.safetensors": 2_000_000_000, "vae/v.safetensors": 500_000_000})

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
    (folder / "ema_vae_fp16.safetensors").touch()
    os.truncate(folder / "ema_vae_fp16.safetensors", 1_500_000_000)
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
    web_config = make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.models_dir = str(models_dir)
    settings = SeedVR2Settings(model="auto", softness=0.75, video_softness=0.0, reference_megapixels=16, tiers=(_SEEDVR2_7B,))

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
    web_config = _models_route_with_seedvr2(monkeypatch, tmp_path, copy_files={"w.safetensors": 1})
    web_config.seedvr2 = None

    with TestClient(web_server.app) as client:
        rows = client.get("/api/models").json()["image_models"]

    assert rows == []


def test_submit_video_job_surfaces_platform_alias_mismatch(monkeypatch):
    """Video submissions should return the alias mismatch message from platform-aware resolution."""
    web_config = make_web_config()
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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())

    asset_path = tmp_path / "nested" / "asset one.png"
    write_png(asset_path)
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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

    image_path = tmp_path / "image-only.png"
    video_path = tmp_path / "video-only.mp4"
    write_png(image_path)
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


def test_gallery_reuse_reads_embedded_png_config(monkeypatch, tmp_path):
    """Embedded PNG config should drive Gallery details and reuse URLs."""
    web_config = make_web_config()
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
    assert asset["reuse_state"]["requested_model"] == "zit"
    assert asset["reuse_state"]["resolved_model"] == "zit"
    assert asset["reuse_state"]["model_available"] is True
    assert "workflow=img2img" in asset["reuse_workspace_url"]
    assert "prompt=original+prompt" in asset["reuse_workspace_url"]
    assert "model=zit" in asset["reuse_workspace_url"]
    assert "seed=1234" in asset["reuse_workspace_url"]
    assert "steps=9" in asset["reuse_workspace_url"]
    assert "guidance=2.5" in asset["reuse_workspace_url"]
    assert "image_path=%2Finput%2Freference.png" in asset["reuse_workspace_url"]


def test_media_and_delete_reject_invalid_asset_ids(monkeypatch, tmp_path):
    """Media and delete routes should reject traversal, absolute, and staging asset IDs."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())

    valid_asset = tmp_path / "nested" / "asset.png"
    write_png(valid_asset)

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


def test_staged_reference_uploads_are_not_served(monkeypatch, tmp_path):
    """Temporary Web upload staging is not a gallery asset, so /media refuses it."""
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path.resolve())
    write_png(tmp_path / ".web_uploads" / "reference.png")

    with TestClient(web_server.app) as client:
        assert client.get("/media/.web_uploads/reference.png").status_code == 404


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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    defaults = make_resolved_image_defaults()
    defaults["upscale_steps"] = preset_upscale_steps
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=defaults, submitted=submitted)

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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)

    response = web_server._submit_image_job({"model": "zit", "prompt": "hello", **form}, web_config)

    assert submitted[0]["args"].review is review
    assert list(response["supported_controls"]) == controls


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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=True, submitted=submitted)

    response = web_server._submit_video_job({"model": "ltx-8", "prompt": "a lake", **form}, web_config)

    assert submitted[0]["args"].review is review
    assert list(response["supported_controls"]) == controls


@pytest.mark.parametrize(
    ("capability", "form"),
    [
        ("upscale_review", {"upscale": "2", "review": "on"}),
        ("i2v", {"image_path": "ref.png"}),
        ("end_frame", {"image_path": "ref.png", "end_image_path": "end.png"}),
    ],
)
def test_submit_video_job_refuses_what_the_model_family_cannot_do(monkeypatch, tmp_path, capability, form):
    """A family that lacks a capability refuses the workflow that needs it, whatever the platform supports, and queues nothing."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    ltx = load_config()["video_model_presets"]["ltx"]
    web_config.app_config["video_model_presets"] = {"ltx": {**ltx, "capabilities": {**ltx["capabilities"], capability: False}}}
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=True, submitted=submitted)
    monkeypatch.setattr(web_server, "_resolve_reference_image", lambda _form, _output_dir, path_key="image_path": form.get(path_key))

    with pytest.raises(ValueError):
        web_server._submit_video_job({"model": "ltx-8", "prompt": "a lake", **form}, web_config)

    assert submitted == []


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

    @pytest.mark.parametrize("case", ["no_review", "missing_file", "unknown_job", "vanishes_before_it_is_served", "not_a_regular_file"])
    def test_is_404_when_there_is_nothing_to_serve(self, monkeypatch, tmp_path, case):
        render = tmp_path / "small.mp4"
        render.write_bytes(b"0123456789")

        def _review_path(_job_id):
            if case == "no_review":
                return None
            if case == "missing_file":
                return str(tmp_path / "gone.mp4")
            if case == "unknown_job":
                raise KeyError(_job_id)
            if case == "not_a_regular_file":
                return str(tmp_path)
            path = str(render)
            render.unlink()  # the job resumed and its temporary directory was cleaned up right after the lookup
            return path

        monkeypatch.setattr(web_server.web_runner, "get_job_review_path", _review_path)

        with TestClient(web_server.app, raise_server_exceptions=False) as client:
            assert client.get("/jobs/job-1/review?v=1").status_code == 404


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
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)

    with pytest.raises(ValueError, match=message):
        web_server._submit_image_job({"model": "zit", "prompt": "hello", "lora": lora}, web_config)

    assert submitted == []


def test_submit_image_job_accepts_existing_lora_file(monkeypatch, tmp_path):
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    lora_file = tmp_path / "style.safetensors"
    lora_file.write_bytes(b"")
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)

    web_server._submit_image_job({"model": "zit", "prompt": "hello", "lora": f"{lora_file}:0.7"}, web_config)

    assert submitted[0]["args"].lora_paths == [str(lora_file)]
    assert submitted[0]["args"].lora_weights == [0.7]


def test_media_only_serves_gallery_media_types(monkeypatch, tmp_path):
    """/media must not serve arbitrary files under the output root (e.g. keys or configs)."""
    (tmp_path / "id_rsa").write_text("secret", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    write_png(tmp_path / "ok.png")
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
        monkeypatch.setattr(web_server, "load_web_config", make_web_config)
        monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: make_workspace_bootstrap_view())
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

    settings = web_server._submitted_settings(form, uploads={"image_path": "/out/.web_uploads/ref.png"})

    assert settings == {"prompt": "a lake", "prompt_option_id": ["a:0", "a:1"], "steps": "8", "image_path": "/out/.web_uploads/ref.png"}


def test_submitted_settings_record_each_upload_under_its_own_path_setting():
    """A first and a last frame uploaded together are each restored by Load settings and Retry."""
    from starlette.datastructures import FormData, UploadFile

    first = UploadFile(file=io.BytesIO(b"png"), filename="a.png")
    last = UploadFile(file=io.BytesIO(b"png"), filename="b.png")
    form = FormData([("prompt", "a lake"), ("image_file", first), ("end_image_file", last), ("end_image_path", "/typed/end.png")])

    settings = web_server._submitted_settings(form, uploads={"image_path": "/out/a.png", "end_image_path": "/out/b.png"})

    assert settings == {"prompt": "a lake", "end_image_path": "/out/b.png", "image_path": "/out/a.png"}


def test_submitted_settings_keep_a_typed_last_frame_path_when_nothing_was_uploaded():
    settings = web_server._submitted_settings({"image_path": "/in/a.png", "end_image_path": "/in/b.png"}, uploads={"image_path": "/in/a.png", "end_image_path": "/in/b.png"})

    assert settings == {"image_path": "/in/a.png", "end_image_path": "/in/b.png"}


def _h3_like_config(web_config) -> None:
    ltx = load_config()["video_model_presets"]["ltx"]
    web_config.app_config["video_model_presets"] = {"ltx": {**ltx, "capabilities": {**ltx["capabilities"], "end_frame": True}}}


def test_submit_video_job_takes_a_typed_last_frame_path(monkeypatch, tmp_path):
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    _h3_like_config(web_config)
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    write_png(first)
    write_png(last)
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=True, submitted=submitted)

    response = web_server._submit_video_job({"workflow": "img2vid", "model": "ltx-8", "prompt": "a lake", "image_path": str(first), "end_image_path": str(last)}, web_config)

    assert submitted[0]["args"].end_image_path == str(last.resolve())
    assert submitted[0]["request"].end_image_path == str(last.resolve())
    assert response["context"]["settings"]["end_image_path"] == str(last)


def test_submit_video_job_saves_an_uploaded_last_frame_and_records_it_in_the_job_settings(monkeypatch, tmp_path):
    """An end_image_file wins over a typed path, and the job's settings name the saved file so Retry works."""
    from starlette.datastructures import FormData, UploadFile

    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    _h3_like_config(web_config)
    first = tmp_path / "first.png"
    write_png(first)
    submitted: list[dict[str, object]] = []
    _patch_video_submit_dependencies(monkeypatch, review_supported=True, submitted=submitted)
    upload = UploadFile(file=io.BytesIO(first.read_bytes()), filename="last.png")
    form = FormData([("workflow", "img2vid"), ("model", "ltx-8"), ("prompt", "a lake"), ("image_path", str(first)), ("end_image_path", "/not/used.png"), ("end_image_file", upload)])

    response = web_server._submit_video_job(form, web_config)

    saved = submitted[0]["args"].end_image_path
    assert Path(saved).parent == tmp_path / "out" / ".web_uploads"
    assert Path(saved).is_file()
    assert response["context"]["settings"]["end_image_path"] == saved
    assert response["context"]["settings"]["image_path"] == str(first)


def test_submit_video_job_refuses_a_last_frame_without_a_first_frame(monkeypatch, tmp_path):
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    _h3_like_config(web_config)
    last = tmp_path / "last.png"
    write_png(last)
    _patch_video_submit_dependencies(monkeypatch, review_supported=True, submitted=[])

    with pytest.raises(ValueError, match="needs a first-frame reference image"):
        web_server._submit_video_job({"workflow": "txt2vid", "model": "ltx-8", "prompt": "a lake", "end_image_path": str(last)}, web_config)


def test_submit_image_job_with_a_custom_size_carries_that_size(monkeypatch, tmp_path):
    """A custom size sets the job's size and label, not the ratio preset the form still holds."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)

    response = web_server._submit_image_job({"model": "zit", "prompt": "hello", "steps": "9", "width": "1600", "height": "896", "workflow": "txt2img"}, web_config)

    context = response["context"]
    assert (context["width"], context["height"]) == (1600, 896)
    assert context["meta"].startswith("1600x896 · ")


def test_a_submitted_job_carries_the_models_configured_name_beside_its_id(monkeypatch, tmp_path):
    """AC4.5: the context and the response say the name; the model id is unchanged."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.app_config["model_labels"] = {"zit": {"name": "Z-Image Turbo"}}
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=submitted)
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
    from tests._factories import _make_args

    with pytest.raises(ValueError, match="Sharpen amount must be between 0 and 1.5"):
        web_server._validate_image_args(_make_args(sharpen=amount, image_strength=0.5, quantize=None, runs=1))


def test_generate_endpoint_answers_400_for_a_video_review_the_platform_cannot_pause(monkeypatch, tmp_path):
    """The 400 reaches the HTTP client of POST /api/generate, and nothing is queued."""
    web_config = make_web_config()
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


# ── Extend: source inspection, picker purpose and submission ────────────────

# ─── POST /api/video/memory-fit ──────────────────────────────────────────────


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
        web_config = make_web_config()
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

    @pytest.mark.parametrize(("kind", "name", "label"), [("model", "zit", "Z-Image Turbo"), ("enhancer", "fast", None)])
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

    def test_unknown_rows_and_kinds_are_404_and_bad_bodies_422(self, client):
        assert client.post("/api/models/download", json={"kind": "model", "name": "nope"}).status_code == 404
        assert client.post("/api/models/download", json={"kind": "enhancer", "name": "nope"}).status_code == 404
        assert client.post("/api/models/download", json={"kind": "lora", "name": "zit"}).status_code == 404
        assert client.post("/api/models/download", json={"kind": "model"}).status_code == 422
        assert client.post("/api/models/download", json=["model", "zit"]).status_code == 422
        assert client.get("/api/models/downloads").json() == {"items": []}

    def test_cancelling_an_unknown_item_is_404(self, client):
        assert client.post("/api/models/downloads/dl-999/cancel").status_code == 404

    def test_a_shut_down_lane_refuses_new_downloads_with_a_retryable_status(self, client, lane):
        lane.shutdown()

        assert client.post("/api/models/download", json={"kind": "model", "name": "zit"}).status_code == 503

    def test_the_lifespan_shuts_the_lane_down(self, lane):
        with TestClient(web_server.app):
            assert lane.snapshot() == {"items": []}

        with pytest.raises(RuntimeError):
            lane.submit(kind="model", name="zit", label="zit", plan=lambda: ())


class TestModelDownloadPlanRoute:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        web_config = make_web_config()
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
        db_path = tmp_path / "ziv.sqlite3"
        monkeypatch.setattr(web_server, "job_log", JobLog(lambda: db_path))
        return db_path

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

    def test_the_page_query_selects_the_page(self, path):
        self._seed([self._entry(f"j{i:03d}") for i in range(120)])

        with TestClient(web_server.app) as client:
            second = client.get("/api/jobs/log", params={"page": 2}).json()

        assert (second["page"], second["total_pages"], len(second["entries"])) == (2, 3, 50)
        assert second["entries"][0]["job_id"] == "j069"

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

    def test_clear_returns_the_removed_count(self, path):
        self._seed([self._entry("a"), self._entry("b")])

        with TestClient(web_server.app) as client:
            first = client.delete("/api/jobs/log")
            second = client.delete("/api/jobs/log")

        assert first.json() == {"deleted_count": 2}
        assert second.json() == {"deleted_count": 0}

    def test_listing_keeps_the_written_key_order_and_float_forms(self, path):
        entry = build_job_log_entry(
            job_id="abc",
            job_type="image",
            context={"prompt": "a lake", "model": "m1", "workflow": "txt2img", "settings": {"steps": 4}},
            status="completed",
            created_at=100.0,
            started_at=110.0,
            completed_at=112.5,
            outputs=[{"id": "a.png"}],
            error=None,
            failed_runs=0,
        )
        self._seed([entry])

        with TestClient(web_server.app) as client:
            response = client.get("/api/jobs/log")

        assert b'"created_at":100.0' in response.content
        assert list(response.json()["entries"][0]) == list(entry)

    def test_log_routes_are_not_swallowed_by_job_id_routes(self, path):
        with TestClient(web_server.app) as client:
            assert client.get("/api/jobs/log").json()["total_count"] == 0
            assert client.get("/api/jobs").json() == {"active_job": None, "queued_jobs": []}

    @pytest.mark.parametrize(("method", "url"), [("get", "/api/jobs/log"), ("delete", "/api/jobs/log/a"), ("delete", "/api/jobs/log")])
    def test_store_errors_become_500_with_a_detail(self, monkeypatch, method, url):
        def _broken(*_args, **_kwargs):
            raise JobLogError("Permission denied")

        for name in ("page", "delete", "clear"):
            monkeypatch.setattr(web_server.job_log, name, _broken)

        with TestClient(web_server.app) as client:
            response = getattr(client, method)(url)

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert isinstance(detail, str)
        assert detail

    def test_the_server_runner_writes_to_the_server_job_log(self):
        assert web_server.web_runner._job_log is web_server.job_log

    def test_the_server_job_log_lives_in_the_app_database(self):
        web_server.job_log.append({"job_id": "a"})

        with contextlib.closing(sqlite3.connect(app_db.app_db_path())) as conn:
            assert conn.execute("SELECT COUNT(*) FROM jobs_entries_v1").fetchone() == (1,)


class TestLoraFamilies:
    """/api/workspace lists each LoRA with its detected family."""

    @staticmethod
    def _write_lora(path: Path) -> Path:
        import struct

        header = {"layer.lora_A.weight": {"dtype": "F16", "shape": [4, 8], "data_offsets": [0, 0]}, "__metadata__": {"ss_base_model_version": "klein_4b"}}
        payload = json.dumps(header).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(struct.pack("<Q", len(payload)) + payload)
        return path

    @pytest.fixture
    def workspace_client(self, tmp_path, monkeypatch):
        web_config = make_web_config()
        web_config.loras_dir = str(tmp_path / "loras")
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(web_server, "_build_workspace_bootstrap_view", lambda _cfg: make_workspace_bootstrap_view())
        monkeypatch.setattr(web_server.web_runner, "list_jobs", lambda: {"active_job": None, "queued_jobs": []})
        with TestClient(web_server.app) as client:
            yield client

    def test_workspace_rows_carry_the_detected_family(self, tmp_path, workspace_client):
        path = self._write_lora(tmp_path / "loras" / "style.safetensors")

        rows = workspace_client.get("/api/workspace", params={"include_history": "false"}).json()["loras"]

        assert rows == [{"name": "style", "path": str(path), "family": "flux2_klein", "size_variant": "4b"}]

    def test_a_lora_without_a_file_has_no_family(self, workspace_client):
        row = workspace_client.get("/api/workspace", params={"include_history": "false"}).json()["loras"][0]

        assert (row["family"], row["size_variant"]) == (None, None)
