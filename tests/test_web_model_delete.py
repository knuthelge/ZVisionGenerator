"""Tests for deleting installed models, LoRAs, and HuggingFace downloads from the Web UI."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from fastapi.testclient import TestClient

from zvisiongenerator.utils.seedvr2 import SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
from zvisiongenerator.web import server as web_server
from zvisiongenerator.seedvr2_loader import seedvr2_copy_dir
from zvisiongenerator.web.model_delete import delete_lora, delete_model, delete_scope, installed_models_linking_to, model_delete_target, seedvr2_delete_target
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


def test_deleting_a_model_removes_its_stored_quants(layout):
    for name in ("custom@q8", "custom@q4", "other@q8"):
        _touch(layout.models_dir / name / "transformer" / "0.safetensors")

    target = model_delete_target(_image_entry("custom", "installed", ""), layout.models_dir)
    assert {path.name for path in target.stored_quants} == {"custom@q8", "custom@q4"}
    delete_model(target)

    assert not (layout.models_dir / "custom@q8").exists()
    assert not (layout.models_dir / "custom@q4").exists()
    assert (layout.models_dir / "other@q8").is_dir()


def test_deleting_a_huggingface_download_removes_the_alias_stored_quants(layout):
    _touch(layout.models_dir / "base@q8" / "transformer" / "0.safetensors")

    delete_model(model_delete_target(_image_entry("base", "alias", "org/base"), layout.models_dir))

    assert not (layout.models_dir / "base@q8").exists()


class TestSharedDownload:
    """ltx-4 and ltx-8 on Windows/Linux load from one download plus a separate transformer checkpoint repo."""

    _CONFIG = {"video_model_presets": {"ltx": {"diffusers": {"default_repo": "org/base", "transformer": {"repo": "org/ckpt", "filename": "t.safetensors"}}}}}
    _INVENTORY = (_image_entry("ltx-4", "alias", "org/base"), _image_entry("ltx-8", "alias", "org/base"))

    def _target(self, layout, name):
        entry = next(entry for entry in self._INVENTORY if entry.name == name)
        return model_delete_target(entry, layout.models_dir, **delete_scope(entry, self._INVENTORY, self._CONFIG))

    def test_while_the_other_alias_has_its_copy_only_this_aliases_copy_goes(self, layout):
        _touch(layout.models_dir / "ltx-4@q4" / "transformer" / "0.safetensors")
        _touch(layout.models_dir / "ltx-8@q8" / "transformer" / "0.safetensors")

        target = self._target(layout, "ltx-4")
        assert (target.kind, target.shared_with) == ("copy", ("ltx-8",))
        delete_model(target)

        assert not (layout.models_dir / "ltx-4@q4").exists()
        assert (layout.models_dir / "ltx-8@q8").is_dir()
        assert (layout.hub / "models--org--base").is_dir()

    def test_the_last_alias_removes_the_download_and_the_checkpoint_repo(self, layout):
        _touch(layout.models_dir / "ltx-8@q8" / "transformer" / "0.safetensors")
        _touch(layout.hub / "models--org--ckpt" / "snapshots" / "abc" / "t.safetensors")

        delete_model(self._target(layout, "ltx-8"))

        assert not (layout.models_dir / "ltx-8@q8").exists()
        assert not (layout.hub / "models--org--base").exists()
        assert not (layout.hub / "models--org--ckpt").exists()
        assert (layout.hub / "models--org--other").is_dir()

    def test_an_alias_without_a_copy_while_the_other_has_one_is_not_deletable(self, layout):
        _touch(layout.models_dir / "ltx-8@q8" / "transformer" / "0.safetensors")

        assert self._target(layout, "ltx-4") is None


def test_a_stored_quant_deletes_only_itself(layout):
    _touch(layout.models_dir / "custom@q8" / "transformer" / "0.safetensors")

    target = model_delete_target(_image_entry("custom@q8", "installed", ""), layout.models_dir)

    assert target.stored_quants == ()
    delete_model(target)
    assert not (layout.models_dir / "custom@q8").exists()
    assert (layout.models_dir / "custom").is_dir()


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
            models_dir=str(layout.models_dir),
            seedvr2=None,
            app_config={},
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

    assert delete_info(_image_entry("custom", "installed", "")) == {"kind": "installed", "repo_id": None, "linked_by": [], "stored_quants": []}
    assert delete_info(_image_entry("base", "alias", "org/base")) == {"kind": "huggingface", "repo_id": "org/base", "linked_by": ["custom"], "stored_quants": []}
    assert delete_info(_image_entry("missing", "alias", "org/missing")) is None


class TestSeedvr2DeleteTarget:
    def test_a_stored_copy_is_the_target(self, layout):
        copy = seedvr2_copy_dir("seedvr2-7b", layout.models_dir)
        _touch(copy / "transformer" / "weights.safetensors")

        target = seedvr2_delete_target("seedvr2-7b", layout.models_dir)

        assert target is not None
        assert (target.kind, target.path, target.stored_quants) == ("installed", copy, ())

    def test_no_copy_means_no_target(self, layout):
        assert seedvr2_delete_target("seedvr2-7b", layout.models_dir) is None

    def test_deleting_the_copy_leaves_the_huggingface_cache_and_other_models(self, layout):
        copy = seedvr2_copy_dir("seedvr2-7b", layout.models_dir)
        _touch(copy / "transformer" / "weights.safetensors")

        delete_model(seedvr2_delete_target("seedvr2-7b", layout.models_dir))

        assert not copy.exists()
        assert (layout.models_dir / "custom").is_dir()
        assert layout.base_snapshot.is_dir()


class TestSeedvr2DeleteRoute:
    @pytest.fixture
    def client(self, layout, monkeypatch):
        tier = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)
        settings = SeedVR2Settings(model="auto", softness=0.75, reference_megapixels=16, tiers=(tier,))
        self.copy = seedvr2_copy_dir("seedvr2-7b", layout.models_dir)
        _touch(self.copy / "transformer" / "weights.safetensors")
        self.web_config = SimpleNamespace(
            data_dir=str(layout.data_dir),
            models_dir=str(layout.models_dir),
            image_inventory=(_image_entry("custom", "installed", str(layout.models_dir / "custom")),),
            video_inventory=(),
            seedvr2=SeedVR2Setup(settings=settings, tier=tier, budget_bytes=None, downloaded=True),
        )
        monkeypatch.setattr(web_server, "load_web_config", lambda: self.web_config)
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)
        with TestClient(web_server.app) as client:
            yield client

    def test_delete_removes_only_the_stored_copy(self, client, layout):
        response = client.delete("/api/models/seedvr2-7b@q8")

        assert response.status_code == 200
        assert not self.copy.exists()
        assert (layout.models_dir / "custom").is_dir()
        assert layout.base_snapshot.is_dir()

    def test_a_second_delete_finds_nothing(self, client):
        assert client.delete("/api/models/seedvr2-7b@q8").status_code == 200
        assert client.delete("/api/models/seedvr2-7b@q8").status_code == 404

    def test_other_names_and_the_other_tier_are_404(self, client):
        assert client.delete("/api/models/seedvr2-3b@q8").status_code == 404
        assert client.delete("/api/models/seedvr2-7b").status_code == 404
        assert self.copy.is_dir()

    def test_delete_is_404_where_seedvr2_is_off(self, client):
        self.web_config.seedvr2 = None

        assert client.delete("/api/models/seedvr2-7b@q8").status_code == 404
        assert self.copy.is_dir()

    def test_delete_is_refused_while_a_job_runs(self, client, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j1"})

        assert client.delete("/api/models/seedvr2-7b@q8").status_code == 409
        assert self.copy.is_dir()

    def test_names_that_climb_out_of_the_copy_folder_are_404(self, client, layout):
        sibling = layout.models_dir / "custom"
        for name in ("..", ".seedvr2", "%2e%2e%2fcustom", "..%2Fcustom", "seedvr2-7b@q8%2F..%2F..%2Fcustom", "SEEDVR2-7B@Q8"):
            assert client.delete(f"/api/models/{name}").status_code in {404, 405}, name
        assert self.copy.is_dir()
        assert sibling.is_dir()
        assert layout.base_snapshot.is_dir()

    def test_a_symlinked_copy_is_unlinked_and_its_target_survives(self, client, layout, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        _touch(elsewhere / "weights.safetensors")
        shutil.rmtree(self.copy)
        self.copy.symlink_to(elsewhere, target_is_directory=True)

        assert client.delete("/api/models/seedvr2-7b@q8").status_code == 200

        assert not self.copy.is_symlink()
        assert (elsewhere / "weights.safetensors").is_file()
