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
        video_ratios=("16:9",),
        video_size_options={"16:9": ("m",)},
        scheduler_options=("beta",),
        quantize_options=(4, 8),
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

    assert sorted(schema_fields) == [
        "generation.default_size",
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

    response = web_server._submit_image_job({"prompt": "hello world"}, web_config)

    assert response["job_id"] == "job-123"
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
        web_server._resolve_prompt_submission(
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
        logo_response = client.get("/app-static/zvision-white.png")

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
        lambda _data_dir: [SimpleNamespace(name="local-image", family="zimage", size="m")],
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


def test_gallery_asset_id_contract_serves_and_deletes_relative_assets(monkeypatch, tmp_path):
    """Gallery list, media, and delete should share output-root-relative POSIX IDs."""
    web_config = _make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

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
