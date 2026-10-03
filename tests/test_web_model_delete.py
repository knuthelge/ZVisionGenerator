"""Tests for deleting installed models, LoRAs, and HuggingFace downloads from the Web UI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from fastapi.testclient import TestClient

from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.model_delete import delete_lora, delete_model, installed_models_linking_to, model_delete_target
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _image_entry(name: str, source: str, resolved_path: str) -> ImageInventoryEntry:
    return ImageInventoryEntry(name=name, family="flux2_klein", size=None, source=source, resolved_path=resolved_path)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A data dir with a converted model linking into a cached HuggingFace base repo, plus a LoRA."""
    hub = tmp_path / "hub"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    base_snapshot = hub / "models--org--base" / "snapshots" / "abc"
    _touch(base_snapshot / "text_encoder" / "model.safetensors")
    _touch(hub / "models--org--other" / "snapshots" / "def" / "model.safetensors")

    data_dir = tmp_path / "ziv"
    converted = data_dir / "models" / "custom"
    _touch(converted / "transformer" / "model.safetensors")
    (converted / "text_encoder").symlink_to(base_snapshot / "text_encoder", target_is_directory=True)
    _touch(data_dir / "loras" / "style.safetensors")
    return SimpleNamespace(data_dir=data_dir, models_dir=data_dir / "models", hub=hub, base_snapshot=base_snapshot)


class TestModelDeleteTarget:
    def test_installed_model_deletes_its_folder(self, layout):
        target = model_delete_target(_image_entry("custom", "installed", str(layout.models_dir / "custom")), layout.models_dir)

        assert target.kind == "installed"
        assert target.path == layout.models_dir / "custom"

    def test_alias_repo_deletes_its_cache_download(self, layout):
        target = model_delete_target(_image_entry("base", "alias", "org/base"), layout.models_dir)

        assert target.kind == "huggingface"
        assert target.repo_id == "org/base"
        assert target.path == layout.hub / "models--org--base"

    def test_alias_without_complete_weights_is_not_deletable(self, layout):
        _touch(layout.hub / "models--org--partial" / "snapshots" / "abc" / "model_index.json")

        assert model_delete_target(_image_entry("partial", "alias", "org/partial"), layout.models_dir) is None

    def test_alias_to_local_directory_is_not_deletable(self, layout, tmp_path):
        assert model_delete_target(_image_entry("local", "alias", str(tmp_path / "elsewhere")), layout.models_dir) is None

    @pytest.mark.parametrize("name", ["..", "a/b"])
    def test_installed_names_cannot_escape_the_models_dir(self, layout, name):
        assert model_delete_target(_image_entry(name, "installed", ""), layout.models_dir) is None


def test_deleting_a_converted_model_keeps_the_linked_base_files(layout):
    delete_model(model_delete_target(_image_entry("custom", "installed", ""), layout.models_dir))

    assert not (layout.models_dir / "custom").exists()
    assert (layout.base_snapshot / "text_encoder" / "model.safetensors").is_file()


def test_deleting_a_huggingface_download_leaves_other_repos(layout):
    delete_model(model_delete_target(_image_entry("base", "alias", "org/base"), layout.models_dir))

    assert not (layout.hub / "models--org--base").exists()
    assert (layout.hub / "models--org--other").is_dir()


def test_converted_models_linking_into_a_download_are_reported(layout):
    assert installed_models_linking_to(layout.hub / "models--org--base", layout.models_dir) == ("custom",)
    assert installed_models_linking_to(layout.hub / "models--org--other", layout.models_dir) == ()


def test_delete_lora_removes_only_the_named_file(layout):
    loras_dir = layout.data_dir / "loras"

    delete_lora(loras_dir, "style")

    assert not (loras_dir / "style.safetensors").exists()
    with pytest.raises(FileNotFoundError):
        delete_lora(loras_dir, "style")
    with pytest.raises(FileNotFoundError):
        delete_lora(loras_dir, "../models/custom/transformer/model")


class TestDeleteRoutes:
    @pytest.fixture
    def client(self, layout, monkeypatch):
        web_config = SimpleNamespace(
            data_dir=str(layout.data_dir),
            image_inventory=(_image_entry("custom", "installed", str(layout.models_dir / "custom")), _image_entry("base", "alias", "org/base")),
            video_inventory=(VideoInventoryEntry(name="ltx", family="ltx", supports_i2v=True, source="alias", resolved_path="org/missing"),),
        )
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)
        with TestClient(web_server.app) as client:
            yield client

    def test_delete_installed_model(self, client, layout):
        response = client.delete("/api/models/custom")

        assert response.status_code == 200
        assert not (layout.models_dir / "custom").exists()

    def test_delete_huggingface_download(self, client, layout):
        response = client.delete("/api/models/base")

        assert response.status_code == 200
        assert "org/base" in response.json()["message"]
        assert not (layout.hub / "models--org--base").exists()

    def test_unknown_or_undownloaded_model_is_404(self, client):
        assert client.delete("/api/models/nope").status_code == 404
        assert client.delete("/api/models/ltx").status_code == 404

    def test_delete_lora(self, client, layout):
        assert client.delete("/api/loras/style").status_code == 200
        assert not (layout.data_dir / "loras" / "style.safetensors").exists()
        assert client.delete("/api/loras/style").status_code == 404

    def test_deletes_are_refused_while_a_job_runs(self, client, layout, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j1"})

        assert client.delete("/api/models/custom").status_code == 409
        assert client.delete("/api/loras/style").status_code == 409
        assert (layout.models_dir / "custom").is_dir()


def test_models_payload_describes_what_each_delete_removes(layout):
    from zvisiongenerator.web.workspace_api import _delete_info_resolver

    delete_info = _delete_info_resolver(layout.models_dir)

    assert delete_info(_image_entry("custom", "installed", "")) == {"kind": "installed", "repo_id": None, "linked_by": []}
    assert delete_info(_image_entry("base", "alias", "org/base")) == {"kind": "huggingface", "repo_id": "org/base", "linked_by": ["custom"]}
    assert delete_info(_image_entry("missing", "alias", "org/missing")) is None
