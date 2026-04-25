"""Tests for the surviving Svelte/FastAPI web surface."""

from __future__ import annotations

from io import BytesIO
import json
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


def _write_asset(
    asset_path: Path,
    *,
    prompt: str,
    model: str,
    seed: int,
    steps: int,
    guidance: float,
    modified_at: int,
    metadata_overrides: dict[str, object] | None = None,
) -> None:
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    if asset_path.suffix.lower() == ".mp4":
        asset_path.write_bytes(b"mp4")
    else:
        Image.new("RGB", (8, 8), color="white").save(asset_path)
    metadata = {
        "prompt": prompt,
        "model": model,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "width": 832,
        "height": 1216,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    asset_path.with_suffix(".json").write_text(
        json.dumps(metadata),
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

    def test_app_route_serves_packaged_gallery_bundle_with_semantic_viewer_and_caption_track(self):
        with TestClient(web_server.app) as client:
            app_response = client.get("/app")
            captions_response = client.get("/app-static/empty.vtt")

            assert app_response.status_code == 200
            assert captions_response.status_code == 200
            assert "WEBVTT" in captions_response.text

            script_prefix = '<script type="module" crossorigin src="'
            script_start = app_response.text.find(script_prefix)
            assert script_start != -1
            bundle_path = app_response.text[script_start + len(script_prefix) :].split('"', 1)[0]

            bundle_response = client.get(bundle_path)

        assert bundle_response.status_code == 200
        bundle_text = bundle_response.text
        assert 'role="button"' in bundle_text
        assert "Asset:" in bundle_text
        assert 'aria-label="Fullscreen viewer"' in bundle_text
        assert 'aria-label="Close fullscreen viewer"' in bundle_text
        # Overlay pointer-events split: the hover overlay container must be pointer-events-none
        # and every action button/link inside it must opt back in with pointer-events-auto.
        # These combined class assertions fail if the split is missing or reverted.
        assert "pointer-events-none" in bundle_text
        assert "surface-overlay-action pointer-events-auto" in bundle_text
        assert "surface-overlay-action-primary pointer-events-auto" in bundle_text
        assert "surface-overlay-action-danger pointer-events-auto" in bundle_text
        assert bundle_text.count("surface-select") >= 2
        assert "panel-shell-right" in bundle_text
        # Active-vs-selected card state split: all three class strings must be present so
        # the bundle encodes the three distinct visual states (unselected, active-unselected,
        # selected).  A stale bundle missing the active ring or the ternary would fail here.
        assert "surface-gallery-card" in bundle_text
        assert "surface-gallery-card-selected" in bundle_text
        assert "ring-2 ring-white/30" in bundle_text
        assert "/app-static/empty.vtt" in bundle_text

    def test_app_route_serves_workspace_toolbar_bundle_with_rounded_md_radius(self):
        with TestClient(web_server.app) as client:
            app_response = client.get("/app")

            assert app_response.status_code == 200

            script_prefix = '<script type="module" crossorigin src="'
            script_start = app_response.text.find(script_prefix)
            assert script_start != -1
            bundle_path = app_response.text[script_start + len(script_prefix) :].split('"', 1)[0]

            bundle_response = client.get(bundle_path)

        assert bundle_response.status_code == 200
        bundle_text = bundle_response.text
        assert 'testId:"model-shell"' in bundle_text
        assert 'testId:"quantize-shell"' in bundle_text
        assert "data-hovered" in bundle_text
        assert "data-focused" in bundle_text
        assert "appearance-none opacity-0 cursor-pointer focus:outline-none" in bundle_text
        assert "transition-colors text-text-primary" in bundle_text
        assert "panel-toolbar" in bundle_text
        assert "surface-dropzone" in bundle_text
        assert "surface-button-primary" in bundle_text
        assert "bg-bg-surface-hover border-border-subtle" in bundle_text
        assert "bg-bg-surface border-primary-main ring-4 ring-primary-main" in bundle_text
        assert '"mouseenter"' in bundle_text
        assert '"focus"' in bundle_text
        assert "bg-zinc-950 border border-zinc-700 hover:border-zinc-600 text-sm rounded-md px-3 py-1.5 flex items-center justify-between transition text-zinc-200" not in bundle_text
        assert "bg-zinc-950 border border-zinc-700 hover:border-zinc-600 text-sm rounded-md px-3 py-1.5 flex items-center justify-between transition text-zinc-200 font-mono" not in bundle_text
        assert "bg-bg-surface border border-border-subtle hover:border-zinc-600 text-sm rounded px-3 py-1.5 flex items-center justify-between transition text-zinc-200" not in bundle_text
        assert "group-hover:bg-bg-surface-hover" not in bundle_text
        assert "group-focus-within:ring-4" not in bundle_text

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
        assert payload["defaults"] == {
            "ratio": "2:3",
            "size": "m",
            "width": 832,
            "height": 1216,
            "steps": 27,
            "guidance": 6.2,
            "scheduler": "beta",
            "supports_negative_prompt": True,
            "supports_quantize": True,
            "quantize": None,
            "image_strength": 0.5,
            "postprocess": {"sharpen": 0.8, "contrast": False, "saturation": False},
            "upscale": {"enabled": False, "factor": None, "denoise": None, "steps": None, "guidance": None, "sharpen": True, "save_pre": False},
        }
        assert payload["video_defaults"] == {
            "ratio": "16:9",
            "size": "m",
            "width": 704,
            "height": 448,
            "frame_count": 33,
            "steps": 7,
            "audio": True,
            "low_memory": True,
            "supports_i2v": True,
            "supports_quantize": False,
            "quantize": None,
            "max_steps": 8,
            "fps": 24,
            "upscale": {"enabled": False, "factor": 2, "steps": None},
        }
        assert payload["image_model_defaults"] == {
            "zit": {
                "ratio": "2:3",
                "size": "m",
                "width": 832,
                "height": 1216,
                "steps": 27,
                "guidance": 6.2,
                "scheduler": "beta",
                "supports_negative_prompt": True,
                "supports_quantize": True,
                "quantize": None,
                "image_strength": 0.5,
                "postprocess": {"sharpen": 0.8, "contrast": False, "saturation": False},
                "upscale": {"enabled": False, "factor": None, "denoise": None, "steps": None, "guidance": None, "sharpen": True, "save_pre": False},
            }
        }
        assert payload["video_model_defaults"] == {
            "ltx-8": {
                "ratio": "16:9",
                "size": "m",
                "width": 704,
                "height": 448,
                "frame_count": 33,
                "steps": 7,
                "audio": True,
                "low_memory": True,
                "supports_i2v": True,
                "supports_quantize": False,
                "quantize": None,
                "max_steps": 8,
                "fps": 24,
                "upscale": {"enabled": False, "factor": 2, "steps": None},
            }
        }
        assert payload["history_assets"][0]["filename"] == "latest.png"
        assert payload["history_assets"][0]["workflow"] == "txt2img"
        assert payload["output_dir"] == str(output_dir)
        assert payload["image_ratios"] == ["2:3"]
        assert payload["video_ratios"] == ["16:9"]
        assert payload["quantize_options"] == [4, 8]
        assert payload["prompt_sources"] == ["inline", "file"]
        assert payload["default_prompt_source"] == "inline"
        assert payload["prompt_file"] == {
            "accepted_extensions": [".yaml", ".yml"],
            "browse_kind": "existing_file",
            "selection_required": True,
        }
        assert payload["workflow_contract"]["values"] == ["txt2img", "img2img", "txt2vid", "img2vid"]
        assert payload["workflow_contract"]["definitions"]["txt2img"]["visible_controls"] == [
            "workflow",
            "model",
            "quantize",
            "loras",
            "prompt_source",
            "prompt_inline",
            "negative_prompt",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "steps",
            "guidance",
            "seed",
            "scheduler",
            "postprocess_sharpen",
            "postprocess_contrast",
            "postprocess_saturation",
            "image_upscale_enabled",
            "image_upscale_factor",
            "image_upscale_denoise",
            "image_upscale_steps",
            "image_upscale_guidance",
            "image_upscale_sharpen",
        ]
        assert payload["workflow_contract"]["definitions"]["img2img"]["visible_controls"] == [
            "workflow",
            "model",
            "quantize",
            "loras",
            "prompt_source",
            "prompt_inline",
            "negative_prompt",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "reference_image",
            "reference_image_path",
            "reference_image_clear",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "steps",
            "guidance",
            "image_strength",
            "seed",
            "scheduler",
            "postprocess_sharpen",
            "postprocess_contrast",
            "postprocess_saturation",
            "image_upscale_enabled",
            "image_upscale_factor",
            "image_upscale_denoise",
            "image_upscale_steps",
            "image_upscale_guidance",
            "image_upscale_sharpen",
        ]
        assert payload["workflow_contract"]["definitions"]["txt2vid"]["visible_controls"] == [
            "workflow",
            "model",
            "loras",
            "prompt_source",
            "prompt_inline",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "frame_count",
            "steps",
            "seed",
            "audio",
            "low_memory",
            "video_upscale_enabled",
            "video_upscale_factor",
        ]
        assert payload["workflow_contract"]["definitions"]["img2vid"]["visible_controls"] == [
            "workflow",
            "model",
            "loras",
            "prompt_source",
            "prompt_inline",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "reference_image",
            "reference_image_path",
            "reference_image_clear",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "frame_count",
            "steps",
            "seed",
            "audio",
            "low_memory",
            "video_upscale_enabled",
            "video_upscale_factor",
        ]
        assert payload["config"]["startup_view"] == "workspace"

    def test_api_workspace_uses_config_backed_video_bootstrap_fallbacks_when_detection_fails(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_server, "resolve_model_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("missing model")))
        monkeypatch.setattr(web_server, "resolve_defaults", lambda *args, **kwargs: {"steps": 27, "guidance": 6.2, "scheduler": "beta", "supports_negative_prompt": True})

        with TestClient(web_server.app) as client:
            response = client.get("/api/workspace")

        assert response.status_code == 200
        payload = response.json()
        assert payload["video_defaults"] == {
            "ratio": "16:9",
            "size": "m",
            "width": 704,
            "height": 448,
            "frame_count": 49,
            "steps": 8,
            "audio": True,
            "low_memory": True,
            "supports_i2v": False,
            "supports_quantize": False,
            "quantize": None,
            "max_steps": 8,
            "fps": 24,
            "upscale": {"enabled": False, "factor": 2, "steps": None},
        }
        assert payload["video_model_defaults"]["ltx-8"] == payload["video_defaults"]

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

    def test_workspace_history_and_gallery_normalize_reuse_fallbacks_consistently(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        _write_asset(
            output_dir / "reuse-source.mp4",
            prompt="Reuse this clip",
            model="missing-video-model",
            seed=21,
            steps=9,
            guidance=0.0,
            modified_at=4_000,
            metadata_overrides={
                "workflow": "i2v",
                "ratio": "16:9",
                "size": "m",
                "frame_count": 49,
                "lora": [{"name": "detail", "weight": 0.8}],
            },
        )
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: tmp_path)

        with TestClient(web_server.app) as client:
            workspace_response = client.get("/api/workspace")
            history_response = client.get("/api/history", params={"page": 1, "media_filter": "all", "sort_order": "newest"})
            gallery_response = client.get("/api/gallery", params={"page": 1, "filter": "video", "sort_order": "newest"})

        assert workspace_response.status_code == 200
        assert history_response.status_code == 200
        assert gallery_response.status_code == 200

        workspace_asset = workspace_response.json()["history_assets"][0]
        history_asset = history_response.json()["assets"][0]
        gallery_asset = gallery_response.json()["assets"][0]

        expected_reuse_state = {
            "requested_workflow": "img2vid",
            "resolved_workflow": "txt2vid",
            "workflow_available": False,
            "requested_model": "missing-video-model",
            "resolved_model": "ltx-8",
            "model_available": False,
            "fallback_reasons": ["missing_reference_image", "model_not_configured"],
        }
        expected_reuse_url = "#/workspace?workflow=txt2vid&prompt=Reuse+this+clip&model=ltx-8&lora=detail%3A0.8&steps=9&seed=21&ratio=16%3A9&size=m&width=832&height=1216&frames=49"

        for asset in (workspace_asset, history_asset, gallery_asset):
            assert asset["workflow"] == "img2vid"
            assert asset["media_type"] == "video"
            assert asset["ratio"] == "16:9"
            assert asset["size"] == "m"
            assert asset["frame_count"] == 49
            assert asset["image_path"] is None
            assert asset["reuse_state"] == expected_reuse_state
            assert asset["reuse_workspace_url"] == expected_reuse_url

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
        assert payload["workflow"] == "txt2img"
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

    def test_api_generate_file_mode_resolves_selected_prompt_option(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        prompt_file = tmp_path / "prompts.yaml"
        prompt_file.write_text(
            """
portrait:
  - prompt: "hero portrait"
    negative: "blurry"
""".strip(),
            encoding="utf-8",
        )
        submitted: dict[str, object] = {}

        def _store_request(**kwargs: object) -> str:
            submitted.update(kwargs)
            return "job-image"

        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))
        monkeypatch.setattr(
            web_server,
            "resolve_defaults",
            lambda *args, **kwargs: {"steps": 13, "guidance": 4.4, "scheduler": "beta", "supports_negative_prompt": True},
        )
        monkeypatch.setattr(web_server, "validate_scheduler", lambda *args, **kwargs: None)
        monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", _store_request)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "image",
                    "workflow": "txt2img",
                    "prompt_source": "file",
                    "prompts_file": str(prompt_file),
                    "prompt_option_id": "portrait:0",
                    "model": "zit",
                    "ratio": "2:3",
                    "size": "m",
                },
            )

        assert response.status_code == 200
        request = submitted["request"]
        assert getattr(request, "prompt") == "hero portrait"
        assert getattr(request, "negative_prompt") == "blurry"
        assert submitted["prompts_data"] == {"portrait": [("hero portrait", "blurry")]}
        assert response.json()["prompt"] == "hero portrait"

    def test_api_generate_file_mode_rejects_stale_prompt_option(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        prompt_file = tmp_path / "prompts.yaml"
        prompt_file.write_text("portrait:\n  - prompt: hero\n", encoding="utf-8")
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))
        monkeypatch.setattr(
            web_server,
            "resolve_defaults",
            lambda *args, **kwargs: {"steps": 13, "guidance": 4.4, "scheduler": "beta", "supports_negative_prompt": True},
        )
        monkeypatch.setattr(web_server, "validate_scheduler", lambda *args, **kwargs: None)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "image",
                    "workflow": "txt2img",
                    "prompt_source": "file",
                    "prompts_file": str(prompt_file),
                    "prompt_option_id": "portrait:99",
                    "model": "zit",
                    "ratio": "2:3",
                    "size": "m",
                },
            )

        assert response.status_code == 422
        assert response.json() == {"detail": "Prompt option 'portrait:99' is missing or inactive."}

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
        assert response.json()["workflow"] == "img2img"

    def test_api_generate_applies_cli_resolved_image_defaults_when_fields_are_omitted(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        submitted: dict[str, object] = {}

        def _store_request(**kwargs: object) -> str:
            submitted.update(kwargs)
            return "job-image"

        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))
        monkeypatch.setattr(
            web_server,
            "resolve_defaults",
            lambda *args, **kwargs: {"steps": 13, "guidance": 4.4, "scheduler": "beta", "supports_negative_prompt": True},
        )
        monkeypatch.setattr(web_server, "validate_scheduler", lambda *args, **kwargs: None)
        monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", _store_request)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "image",
                    "workflow": "txt2img",
                    "prompt": "Aurora over mountains",
                    "model": "zit",
                    "ratio": "2:3",
                    "size": "m",
                },
            )

        assert response.status_code == 200
        request = submitted["request"]
        args = submitted["args"]
        assert getattr(request, "steps") == 13
        assert getattr(request, "guidance") == 4.4
        assert getattr(request, "scheduler") == "beta"
        assert getattr(args, "steps") == 13
        assert getattr(args, "guidance") == 4.4
        assert getattr(args, "scheduler") == "beta"
        assert response.json()["workflow"] == "txt2img"

    def test_api_generate_preserves_explicit_zero_image_strength(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        reference_path = tmp_path / "reference.png"
        Image.new("RGB", (8, 8), color="white").save(reference_path)
        submitted: dict[str, object] = {}

        def _store_request(**kwargs: object) -> str:
            submitted.update(kwargs)
            return "job-image"

        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(output_dir))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))
        monkeypatch.setattr(
            web_server,
            "resolve_defaults",
            lambda *args, **kwargs: {"steps": 13, "guidance": 4.4, "scheduler": "beta", "supports_negative_prompt": True},
        )
        monkeypatch.setattr(web_server, "validate_scheduler", lambda *args, **kwargs: None)
        monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", _store_request)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "image",
                    "workflow": "img2img",
                    "prompt": "Lock the composition",
                    "model": "zit",
                    "ratio": "2:3",
                    "size": "m",
                    "image_path": str(reference_path),
                    "image_strength": "0.0",
                },
            )

        assert response.status_code == 200
        request = submitted["request"]
        args = submitted["args"]
        assert getattr(request, "image_strength") == 0.0
        assert getattr(args, "image_strength") == 0.0

    def test_resolve_numeric_toggle_parses_enabled_amount_and_disabled_state(self):
        form = {
            "sharpen_enabled": "true",
            "sharpen_amount": "0.75",
            "contrast_enabled": "false",
            "saturation_enabled": "true",
            "saturation_amount": "1.2",
        }

        assert web_server._resolve_numeric_toggle(form, "sharpen_enabled", "sharpen_amount", default_enabled=True) == 0.75
        assert web_server._resolve_numeric_toggle(form, "contrast_enabled", "contrast_amount", default_enabled=False) is False
        assert web_server._resolve_numeric_toggle(form, "saturation_enabled", "saturation_amount", default_enabled=False) == 1.2

    def test_api_generate_rejects_unknown_image_scheduler(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))

        def _resolve_defaults(_model_info, _config, cli_overrides, _backend_name):
            return {
                "steps": 10,
                "guidance": 3.5,
                "scheduler": cli_overrides.get("scheduler"),
                "supports_negative_prompt": True,
            }

        monkeypatch.setattr(web_server, "resolve_defaults", _resolve_defaults)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "image",
                    "workflow": "txt2img",
                    "prompt": "Aurora over mountains",
                    "model": "zit",
                    "ratio": "2:3",
                    "size": "m",
                    "scheduler": "unknown-scheduler",
                },
            )

        assert response.status_code == 422
        assert response.json()["detail"] == "Unknown scheduler 'unknown-scheduler'. Valid options: ['beta']"

    def test_api_generate_rejects_image_upscale_dimension_drift(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(web_server, "detect_image_model", lambda _model: MagicMock(family="flux"))

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "image",
                    "workflow": "txt2img",
                    "prompt": "Northern lights",
                    "model": "zit",
                    "ratio": "2:3",
                    "size": "m",
                    "width": "800",
                    "upscale": "4",
                },
            )

        assert response.status_code == 422
        assert "Width 800 is not compatible with upscale 4" in response.json()["detail"]

    def test_api_generate_rejects_canonical_img2vid_without_reference_image(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "load_web_config", lambda: _make_web_config(tmp_path / "outputs"))

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "video",
                    "workflow": "img2vid",
                    "prompt": "Animate this",
                    "model": "ltx-8",
                    "ratio": "16:9",
                    "size": "m",
                },
            )

        assert response.status_code == 422
        assert response.json() == {"detail": "Image-to-video requires a reference image."}

    def test_api_generate_normalizes_video_submission_like_cli(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        web_config = _make_web_config(output_dir)
        web_config.app_config["video_model_presets"]["ltx"]["upscale"] = {"default_upscale_steps": 12}
        submitted: dict[str, object] = {}

        def _store_request(**kwargs: object) -> str:
            submitted.update(kwargs)
            return "job-video"

        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(
            web_server,
            "detect_video_model",
            lambda _model: SimpleNamespace(
                family="ltx",
                backend="ltx",
                supports_i2v=True,
                default_fps=24,
                frame_alignment=8,
                resolution_alignment=32,
            ),
        )
        monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", _store_request)

        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/generate",
                data={
                    "mode": "video",
                    "workflow": "txt2vid",
                    "prompt": "Orbiting satellite",
                    "model": "ltx-8",
                    "ratio": "16:9",
                    "size": "m",
                    "width": "705",
                    "height": "447",
                    "frames": "50",
                    "upscale": "2",
                    "audio": "false",
                    "low_memory": "false",
                },
            )

        assert response.status_code == 200
        request = submitted["request"]
        args = submitted["args"]
        assert getattr(request, "width") == 704
        assert getattr(request, "height") == 448
        assert getattr(request, "num_frames") == 49
        assert getattr(request, "steps") == 8
        assert getattr(request, "upscale_steps") == 8
        assert getattr(request, "no_audio") is True
        assert getattr(args, "low_memory") is False
        assert getattr(args, "audio") is False
        assert getattr(args, "upscale_steps") == 8
        assert response.json()["workflow"] == "txt2vid"

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
        assert models_response.json() == {
            "models_dir": str(tmp_path / "models"),
            "loras_dir": str(tmp_path / "loras"),
            "image_models": [{"name": "flux-dev", "family": "flux", "size": "23 GB"}],
            "video_models": [{"name": "ltx-8", "family": "ltx-video", "supports_i2v": True}],
            "loras": [{"name": "portrait-cinematic", "file_size_mb": 123, "size_label": "123 MB"}],
            "huggingface_configured": True,
            "huggingface_token_env_var": "HF_TOKEN",
        }
        assert convert_response.json() == {"status": "ok", "tone": "success", "message": "Converted."}
        assert local_response.json() == {"status": "ok", "tone": "success", "message": "Imported local."}
        assert hf_response.json() == {"status": "ok", "tone": "success", "message": "Imported hf."}

    def test_api_pick_directory_returns_selected_path(self, monkeypatch):
        monkeypatch.setattr(web_server, "_pick_directory", lambda initial_dir: "/tmp/picked")

        with TestClient(web_server.app) as client:
            response = client.post("/api/pick-directory", json={"initial_dir": "/tmp"})

        assert response.status_code == 200
        assert response.json() == {"path": "/tmp/picked"}

    def test_api_picker_returns_status_payload_for_existing_file(self, monkeypatch):
        monkeypatch.setattr(
            web_server,
            "pick_path",
            lambda *_args, **_kwargs: SimpleNamespace(to_payload=lambda: {"status": "selected", "path": "/tmp/prompts.yaml", "message": None}),
        )

        with TestClient(web_server.app) as client:
            response = client.post("/api/picker", json={"kind": "existing_file", "purpose": "prompt_file"})

        assert response.status_code == 200
        assert response.json() == {"status": "selected", "path": "/tmp/prompts.yaml", "message": None}

    def test_api_prompt_files_inspect_normalizes_path_and_returns_active_options(self, monkeypatch, tmp_path):
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        prompt_file = home_dir / "prompts.yaml"
        prompt_file.write_text(
            """
snippets:
  mood: cinematic lighting
portrait:
  - prompt: ["hero", $mood]
  - prompt: "inactive"
    active: false
  - prompt: "second active"
    negative: "blurry"
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(home_dir))

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt-files/inspect", json={"path": "~/prompts.yaml"})

        assert response.status_code == 200
        assert response.json() == {
            "path": str(prompt_file.resolve()),
            "options": [
                {
                    "id": "portrait:0",
                    "set_name": "portrait",
                    "source_index": 0,
                    "label": "portrait #1 · hero. cinematic lighting",
                    "prompt_preview": "hero. cinematic lighting",
                    "negative_preview": None,
                },
                {
                    "id": "portrait:2",
                    "set_name": "portrait",
                    "source_index": 2,
                    "label": "portrait #3 · second active",
                    "prompt_preview": "second active",
                    "negative_preview": "blurry",
                },
            ],
        }

    def test_api_prompt_files_read_normalizes_path_and_returns_raw_text_and_active_options(self, monkeypatch, tmp_path):
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        prompt_file = home_dir / "prompts.yaml"
        raw_text = """
portrait:
  - prompt: first active
  - prompt: inactive
    active: false
  - prompt: second active
    negative: noisy
""".strip()
        prompt_file.write_text(raw_text, encoding="utf-8")
        monkeypatch.setenv("HOME", str(home_dir))

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt-files/read", json={"path": "~/prompts.yaml"})

        assert response.status_code == 200
        assert response.json() == {
            "path": str(prompt_file.resolve()),
            "raw_text": raw_text,
            "options": [
                {
                    "id": "portrait:0",
                    "set_name": "portrait",
                    "source_index": 0,
                    "label": "portrait #1 · first active",
                    "prompt_preview": "first active",
                    "negative_preview": None,
                },
                {
                    "id": "portrait:2",
                    "set_name": "portrait",
                    "source_index": 2,
                    "label": "portrait #3 · second active",
                    "prompt_preview": "second active",
                    "negative_preview": "noisy",
                },
            ],
        }

    def test_api_prompt_files_write_rejects_invalid_yaml_without_mutating_file(self, tmp_path):
        prompt_file = tmp_path / "prompts.yaml"
        original_text = "portrait:\n  - prompt: hello\n"
        prompt_file.write_text(original_text, encoding="utf-8")

        with TestClient(web_server.app) as client:
            response = client.put(
                "/api/prompt-files/write",
                json={"path": str(prompt_file), "raw_text": "portrait:\n  - prompt: [unterminated"},
            )

        assert response.status_code == 422
        assert "Failed to parse prompts file" in response.json()["detail"]
        assert prompt_file.read_text(encoding="utf-8") == original_text

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
