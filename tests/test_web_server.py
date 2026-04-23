"""Tests for the surviving Svelte/FastAPI web surface."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import yaml

from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.config import WebUiConfig, WebUiDefaultModels


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOVED_ROUTE_PATHS = {
    "/workspace",
    "/gallery",
    "/config",
    "/models",
    "/models/convert",
    "/models/import-lora/local",
    "/models/import-lora/hf",
    "/gallery/delete",
    "/api/gallery/page/{page}",
    "/jobs/{job_id}/preview",
    "/api/workspace/asset",
    "/api/workspace/history",
}


def _make_web_config(output_dir: Path) -> WebUiConfig:
    app_config = {
        "generation": {
            "default_ratio": "2:3",
            "default_size": "m",
            "default_steps": 10,
            "default_guidance": 3.5,
        },
        "video_generation": {
            "default_ratio": "16:9",
            "default_size": "m",
        },
        "video_model_presets": {
            "ltx": {
                "default_steps": 8,
            },
        },
        "sizes": {
            "2:3": {
                "m": {"width": 832, "height": 1216},
            },
        },
        "video_sizes": {
            "ltx": {
                "16:9": {
                    "m": {"width": 704, "height": 448, "frames": 49},
                },
            },
        },
        "schedulers": {
            "beta": {},
        },
    }
    return WebUiConfig(
        app_config=app_config,
        port=8080,
        theme="dark",
        startup_view="workspace",
        gallery_page_size=2,
        output_dir=str(output_dir),
        visible_sections=("image_generation", "video_generation", "lora_management", "gallery_summary"),
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
        image_model_options=("zit",),
        video_model_options=("ltx-8",),
        lora_options=("detail",),
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m",)},
        video_ratios=("16:9",),
        video_size_options={"16:9": ("m",)},
        scheduler_options=("beta",),
    )


def _write_asset(asset_path: Path, *, prompt: str, model: str, seed: int, steps: int, guidance: float, modified_at: int) -> None:
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    if asset_path.suffix.lower() == ".mp4":
        asset_path.write_bytes(b"mp4")
    else:
        Image.new("RGB", (8, 8), color="white").save(asset_path)
    asset_path.with_suffix(".json").write_text(
        yaml.safe_dump(
            {
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
                "width": 832,
                "height": 1216,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    os.utime(asset_path, (modified_at, modified_at))


class TestWebServerSurface:
    def test_root_redirects_to_svelte_spa(self):
        with TestClient(web_server.app, follow_redirects=False) as client:
            response = client.get("/")

        assert response.status_code == 302
        assert response.headers["location"] == "/app"

    def test_spa_and_docs_asset_routes_serve_static_files(self):
        with TestClient(web_server.app) as client:
            app_response = client.get("/app")
            asset_response = client.get("/docs/assets/zvision-white.png")

        assert app_response.status_code == 200
        assert app_response.headers["content-type"].startswith("text/html")
        assert asset_response.status_code == 200
        assert asset_response.headers["content-type"].startswith("image/png")

    def test_media_route_serves_output_files_and_rejects_escape_attempts(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        image_path = output_dir / "nested" / "still.png"
        image_path.parent.mkdir()
        Image.new("RGB", (8, 8), color="white").save(image_path)
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))

        with TestClient(web_server.app) as client:
            ok_response = client.get("/media/nested/still.png")
            bad_response = client.get("/media/../outside.png")

        assert ok_response.status_code == 200
        assert ok_response.headers["content-type"].startswith("image/png")
        assert bad_response.status_code == 404

    def test_removed_legacy_routes_are_unregistered_and_return_404(self):
        route_paths = {route.path for route in web_server.app.routes}
        assert REMOVED_ROUTE_PATHS.isdisjoint(route_paths)

        requests = [
            ("get", "/workspace"),
            ("get", "/gallery"),
            ("get", "/config"),
            ("get", "/models"),
            ("post", "/models/convert"),
            ("post", "/models/import-lora/local"),
            ("post", "/models/import-lora/hf"),
            ("post", "/config"),
            ("post", "/gallery/delete"),
            ("get", "/api/gallery/page/1"),
            ("get", "/jobs/job-1/preview"),
            ("get", "/api/workspace/asset?selected=still.png"),
            ("get", "/api/workspace/history"),
        ]

        with TestClient(web_server.app) as client:
            for method, path in requests:
                response = getattr(client, method)(path)
                assert response.status_code in {404, 405}, f"Expected removed route {method.upper()} {path} to be unavailable"

    def test_api_workspace_returns_bootstrap_context(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        _write_asset(output_dir / "latest.png", prompt="Northern lights", model="zit", seed=11, steps=12, guidance=4.5, modified_at=2_000)
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))
        monkeypatch.setattr(web_server, "detect_video_model", lambda _model: MagicMock(family="ltx", supports_i2v=True))
        monkeypatch.setattr(web_server, "resolve_defaults", lambda *args, **kwargs: {"steps": 27, "guidance": 6.2, "scheduler": "beta", "supports_negative_prompt": True})
        monkeypatch.setattr(web_server, "resolve_video_defaults", lambda *args, **kwargs: {"steps": 7, "width": 704, "height": 448, "num_frames": 33})

        with TestClient(web_server.app) as client:
            response = client.get("/api/workspace")

        assert response.status_code == 200
        payload = response.json()
        for key in (
            "image_model_defaults",
            "video_model_defaults",
        ):
            assert key in payload
        assert payload["image_models"] == [{"id": "zit", "label": "zit", "type": "image"}]
        assert payload["video_models"] == [{"id": "ltx-8", "label": "ltx-8", "type": "video"}]
        assert payload["defaults"] == {"steps": 27, "guidance": 6.2, "width": 832, "height": 1216}
        assert payload["video_defaults"] == {"steps": 7, "width": 704, "height": 448, "frame_count": 33, "fps": 24}
        assert payload["image_model_defaults"] == {"zit": {"steps": 27, "guidance": 6.2, "width": 832, "height": 1216}}
        assert payload["video_model_defaults"] == {"ltx-8": {"steps": 7, "guidance": 0, "width": 704, "height": 448}}
        assert payload["history_assets"][0]["filename"] == "latest.png"
        assert payload["config"]["startup_view"] == "workspace"

    def test_api_history_and_gallery_return_paginated_json(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        _write_asset(output_dir / "oldest.png", prompt="Old", model="zit", seed=1, steps=10, guidance=3.5, modified_at=1_000)
        _write_asset(output_dir / "middle.mp4", prompt="Middle", model="ltx-8", seed=2, steps=8, guidance=0.0, modified_at=2_000)
        _write_asset(output_dir / "newest.png", prompt="New", model="zit", seed=3, steps=11, guidance=4.0, modified_at=3_000)
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))

        with TestClient(web_server.app) as client:
            history_response = client.get("/api/history", params={"page": 1, "media_filter": "all", "sort_order": "newest"})
            gallery_response = client.get("/api/gallery", params={"page": 1, "filter": "video", "sort_order": "oldest"})

        assert history_response.status_code == 200
        history_payload = history_response.json()
        assert history_payload["page"] == 1
        assert history_payload["total_count"] == 3
        assert [asset["filename"] for asset in history_payload["assets"]] == ["newest.png", "middle.mp4"]

        assert gallery_response.status_code == 200
        gallery_payload = gallery_response.json()
        assert gallery_payload["total_count"] == 1
        assert [asset["filename"] for asset in gallery_payload["assets"]] == ["middle.mp4"]

    def test_api_config_get_returns_nested_web_ui_payload(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))

        with TestClient(web_server.app) as client:
            response = client.get("/api/config")

        assert response.status_code == 200
        payload = response.json()
        assert payload["output_dir"] == str(tmp_path / "outputs")
        assert payload["ui"]["visible_sections"] == ["image_generation", "video_generation", "lora_management", "gallery_summary"]
        assert payload["ui"]["default_models"] == {"image": "zit", "video": "ltx-8"}
        assert payload["ui"]["default_image_size"] == "m"

    def test_api_config_post_persists_supported_fields_and_returns_updated_payload(self, monkeypatch, tmp_path):
        data_dir = tmp_path / ".ziv"
        output_dir = tmp_path / "custom-outputs"
        data_dir.mkdir()

        from zvisiongenerator.utils import config as config_mod
        from zvisiongenerator.web import config as web_config_mod

        monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: data_dir)
        monkeypatch.setattr(config_mod, "get_ziv_data_dir", lambda: data_dir)
        monkeypatch.setattr(web_config_mod, "get_ziv_data_dir", lambda: data_dir)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/config",
                json={
                    "ui.default_models.image": "zit",
                    "ui.default_models.video": "ltx-8",
                    "generation.default_size": "m",
                    "ui.output_dir": str(output_dir),
                    "auth.huggingface_token": "should-not-save",
                    "paths.huggingface_cache": "/tmp/hf",
                    "paths.lora_dir": "/tmp/loras",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["ui"]["output_dir"] == str(output_dir)
        assert payload["ui"]["default_models"] == {"image": "zit", "video": "ltx-8"}

        user_config = yaml.safe_load((data_dir / "config.yaml").read_text(encoding="utf-8"))
        assert user_config == {
            "ui": {
                "default_models": {"image": "zit", "video": "ltx-8"},
                "output_dir": str(output_dir),
            },
            "generation": {"default_size": "m"},
        }

    def test_api_generate_returns_json_without_accept_header(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))
        monkeypatch.setattr(
            web_server,
            "_submit_image_job",
            MagicMock(
                return_value={
                    "job_id": "job-json",
                    "job_type": "Image",
                    "title": "zit",
                    "prompt": "Northern lights",
                    "events_url": "/jobs/job-json/events",
                    "status_url": "/jobs/job-json",
                    "supported_controls": ("pause", "quit"),
                    "meta": "2:3 · m · 10 steps",
                }
            ),
        )
        monkeypatch.setattr(web_server, "_submit_video_job", MagicMock())

        with TestClient(web_server.app) as client:
            response = client.post("/api/generate", data={"mode": "image", "prompt": "Northern lights"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        payload = response.json()
        assert payload["job_id"] == "job-json"
        assert payload["workflow"] == "Image"
        assert payload["prompt"] == "Northern lights"
        assert payload["model"] == "zit"
        assert payload["runs"] == 1
        assert payload["events_url"] == "/jobs/job-json/events"
        assert payload["status_url"] == "/jobs/job-json"
        assert "preview_url" not in payload
        assert payload["supported_controls"] == ["pause", "quit"]
        assert payload["meta"] == "2:3 · m · 10 steps"
        assert payload["created_at"]

    def test_api_generate_preserves_conflict_and_validation_status_codes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))

        with TestClient(web_server.app) as client:
            monkeypatch.setattr(web_server, "_submit_image_job", MagicMock(side_effect=web_server.JobConflictError("Another generation job is already running.")))
            conflict_response = client.post("/api/generate", data={"mode": "image", "prompt": "Busy"})

            monkeypatch.setattr(web_server, "_submit_image_job", MagicMock(side_effect=ValueError("Field 'prompt' is required.")))
            validation_response = client.post("/api/generate", data={"mode": "image"})

        assert conflict_response.status_code == 409
        assert conflict_response.json() == {"detail": "Another generation job is already running."}
        assert validation_response.status_code == 422
        assert validation_response.json() == {"detail": "Field 'prompt' is required."}

    def test_api_generate_accepts_uploaded_reference_images(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        upload_bytes = BytesIO()
        Image.new("RGB", (8, 8), color="white").save(upload_bytes, format="PNG")
        upload_bytes.seek(0)

        submitted: dict[str, object] = {}

        def _store_request(**kwargs: object) -> str:
            submitted["request"] = kwargs["request"]
            return "job-image"

        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))
        monkeypatch.setattr(web_server, "resolve_defaults", lambda *args, **kwargs: {"steps": 10, "guidance": 3.5, "scheduler": "beta", "supports_negative_prompt": True})
        monkeypatch.setattr(web_server, "validate_scheduler", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            web_server.web_runner,
            "submit_image_request_job",
            _store_request,
        )

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={"mode": "image", "workflow": "i2i", "prompt": "Refine this", "model": "zit", "ratio": "2:3", "size": "m"},
                files={"image_file": ("reference.png", upload_bytes.getvalue(), "image/png")},
            )

        assert response.status_code == 200
        request = submitted["request"]
        assert getattr(request, "image_path") is not None
        assert Path(getattr(request, "image_path")).is_file()
        assert response.json()["workflow"] == "Image to Image"

    def test_job_snapshot_events_and_controls_routes_still_work(self, monkeypatch):
        snapshot = {
            "job_id": "job-1",
            "status": "running",
            "prompt": "Northern lights",
        }
        monkeypatch.setattr(web_server.web_runner, "get_job_snapshot", lambda job_id: snapshot)
        monkeypatch.setattr(web_server.web_runner, "stream_job_events", lambda job_id: iter(['data: {"type":"tick"}\n\n']))
        monkeypatch.setattr(web_server.web_runner, "queue_job_control", lambda job_id, action: {"job_id": job_id, "action": action, "queued": True})

        with TestClient(web_server.app) as client:
            snapshot_response = client.get("/jobs/job-1")
            events_response = client.get("/jobs/job-1/events")
            control_response = client.post("/jobs/job-1/controls/pause")

        assert snapshot_response.status_code == 200
        assert snapshot_response.json() == snapshot
        assert events_response.status_code == 200
        assert events_response.headers["content-type"].startswith("text/event-stream")
        assert 'data: {"type":"tick"}' in events_response.text
        assert control_response.status_code == 200
        assert control_response.json() == {"job_id": "job-1", "action": "pause", "queued": True}

    def test_api_cancel_job_returns_cancelled_status(self, monkeypatch):
        queued: list[tuple[str, str]] = []
        monkeypatch.setattr(web_server.web_runner, "get_job_snapshot", lambda job_id: {"job_id": job_id})
        monkeypatch.setattr(web_server.web_runner, "queue_job_control", lambda job_id, action: queued.append((job_id, action)))

        with TestClient(web_server.app) as client:
            response = client.post("/api/jobs/job-1/cancel")

        assert response.status_code == 200
        assert response.json() == {"status": "cancelled"}
        assert queued == [("job-1", "quit")]

    def test_api_models_routes_return_json_inventory_and_operation_status(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))
        monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_server, "list_models", lambda _: [SimpleNamespace(name="flux-dev", family="flux", size="23 GB")])
        monkeypatch.setattr(web_server, "list_video_models", lambda _: [SimpleNamespace(name="ltx-8", family="ltx-video", supports_i2v=True)])
        monkeypatch.setattr(web_server, "list_loras", lambda _: [SimpleNamespace(name="portrait-cinematic", file_size_mb=123)])
        monkeypatch.setattr(web_server, "_convert_model_from_form", lambda _form: {"tone": "success", "message": "Converted.", "detail": "done"})
        monkeypatch.setattr(web_server, "_import_local_lora_from_form", lambda _form: {"tone": "success", "message": "Imported local.", "detail": "done"})
        monkeypatch.setattr(web_server, "_import_hf_lora_from_form", lambda _form: {"tone": "success", "message": "Imported hf.", "detail": "done"})
        monkeypatch.setenv("HF_TOKEN", "token")

        with TestClient(web_server.app) as client:
            models_response = client.get("/api/models")
            convert_response = client.post("/api/models/convert", json={"input_path": "/tmp/model.safetensors", "model_type": "zimage"})
            local_response = client.post("/api/models/import-lora/local", json={"source_path": "/tmp/detail.safetensors"})
            hf_response = client.post("/api/models/import-lora/hf", json={"repo_id": "org/lora"})

        assert models_response.status_code == 200
        assert models_response.json()["image_models"] == [{"name": "flux-dev", "family": "flux", "size": "23 GB"}]
        assert models_response.json()["video_models"] == [{"name": "ltx-8", "family": "ltx-video", "supports_i2v": True}]
        assert models_response.json()["loras"] == [{"name": "portrait-cinematic", "file_size_mb": 123, "size_label": "123 MB"}]
        assert models_response.json()["huggingface_token_env_var"] == "HF_TOKEN"
        assert convert_response.json() == {"status": "ok", "tone": "success", "message": "Converted."}
        assert local_response.json() == {"status": "ok", "tone": "success", "message": "Imported local."}
        assert hf_response.json() == {"status": "ok", "tone": "success", "message": "Imported hf."}

    def test_api_pick_directory_returns_selected_path(self, monkeypatch):
        monkeypatch.setattr(web_server, "_pick_directory", lambda initial_dir: "/tmp/picked")

        with TestClient(web_server.app) as client:
            response = client.post("/api/pick-directory", json={"initial_dir": "/tmp"})

        assert response.status_code == 200
        assert response.json() == {"path": "/tmp/picked"}

    def test_api_delete_gallery_asset_removes_asset_and_sidecar(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        asset_path = output_dir / "delete-me.png"
        _write_asset(asset_path, prompt="Delete", model="zit", seed=4, steps=9, guidance=2.5, modified_at=1_000)
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))

        with TestClient(web_server.app) as client:
            response = client.delete("/api/gallery/delete-me.png")

        assert response.status_code == 200
        assert response.json() == {"status": "deleted"}
        assert not asset_path.exists()
        assert not asset_path.with_suffix(".json").exists()

    def test_legacy_template_tree_is_removed_from_repository(self):
        assert not (REPO_ROOT / "zvisiongenerator" / "web" / "templates").exists()


@pytest.mark.parametrize(
    ("route_path", "method"),
    [
        ("/api/models/convert", "post"),
        ("/api/models/import-lora/local", "post"),
        ("/api/models/import-lora/hf", "post"),
    ],
)
def test_json_model_routes_stay_mounted(route_path: str, method: str) -> None:
    mounted = {(route.path, next(iter(route.methods - {"HEAD", "OPTIONS"}), None)) for route in web_server.app.routes if hasattr(route, "methods")}
    assert (route_path, method.upper()) in mounted
