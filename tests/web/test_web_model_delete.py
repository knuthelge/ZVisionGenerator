"""Tests for deleting installed models, LoRAs, and HuggingFace downloads from the Web UI."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from fastapi.testclient import TestClient

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.built_model import write_record
from zvisiongenerator.utils.config import enhancer_tier_models
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
from zvisiongenerator.web import server as web_server
from zvisiongenerator.seedvr2_loader import seedvr2_weights_dir
from zvisiongenerator.web.model_delete import delete_lora, delete_model, delete_scope, enhancer_delete_target, installed_models_linking_to, model_delete_target, seedvr2_delete_target
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry


_SCOPE = {"models_dir": Path("/nonexistent/models"), "platform_key": "linux"}


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _image_entry(name: str, source: str, resolved_path: str) -> ImageInventoryEntry:
    return ImageInventoryEntry(name=name, family="flux2_klein", size=None, source=source, resolved_path=resolved_path)


def _ltx_entry(name: str, source: str, resolved_path: str) -> VideoInventoryEntry:
    return VideoInventoryEntry(name=name, family="ltx", supports_i2v=True, source=source, resolved_path=resolved_path)


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


@pytest.fixture
def downloading(monkeypatch):
    """Register a running download in the server's registry (as a Models-page lane item would); returns the claim function."""
    claimed = []

    def _claim(target: DownloadTarget) -> None:
        flight, owner = web_server.download_flights.claim(target)
        assert owner
        claimed.append(flight)

    yield _claim
    for flight in claimed:
        web_server.download_flights.finish(flight, "stopped")


@pytest.fixture
def web_client(monkeypatch):
    """Return a factory that serves *web_config* from the app with no job running, and closes the client afterwards."""
    clients: list[TestClient] = []

    def _make(web_config) -> TestClient:
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)
        client = TestClient(web_server.app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


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

    @pytest.mark.parametrize("make_entry", [_image_entry, _ltx_entry], ids=["image", "ltx"])
    def test_alias_with_a_partial_download_is_deletable(self, layout, make_entry):
        _touch(layout.hub / "models--org--partial" / "snapshots" / "abc" / "model_index.json")

        target = model_delete_target(make_entry("partial", "alias", "org/partial"), layout.models_dir)

        assert (target.kind, target.repo_id, target.path) == ("huggingface", "org/partial", layout.hub / "models--org--partial")

    def test_a_partial_ltx_download_is_removed_from_the_cache(self, layout):
        partial = _touch(layout.hub / "models--org--partial" / "blobs" / "model.safetensors.incomplete").parent.parent
        target = model_delete_target(_ltx_entry("partial", "alias", "org/partial"), layout.models_dir)

        delete_model(target)

        assert not partial.exists()
        assert (layout.hub / "models--org--other").is_dir()

    def test_alias_never_downloaded_is_not_deletable(self, layout):
        assert model_delete_target(_ltx_entry("missing", "alias", "org/missing"), layout.models_dir) is None

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
    _INVENTORY = (_ltx_entry("ltx-4", "alias", "org/base"), _ltx_entry("ltx-8", "alias", "org/base"))

    def _target(self, layout, name):
        entry = next(entry for entry in self._INVENTORY if entry.name == name)
        return model_delete_target(entry, layout.models_dir, **delete_scope(entry, self._INVENTORY, self._CONFIG, **_SCOPE))

    def test_only_a_video_family_with_companions_lists_them(self):
        ltx, other = self._INVENTORY[0], _image_entry("other", "alias", "org/base")

        assert delete_scope(ltx, self._INVENTORY, self._CONFIG, **_SCOPE)["companion_repos"] == ("org/ckpt",)
        assert delete_scope(other, self._INVENTORY, self._CONFIG, **_SCOPE)["companion_repos"] == ()

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
    def client(self, layout, web_client):
        return web_client(
            SimpleNamespace(
                data_dir=str(layout.data_dir),
                models_dir=str(layout.models_dir),
                seedvr2=None,
                app_config={},
                image_inventory=(_image_entry("custom", "installed", str(layout.models_dir / "custom")), _image_entry("base", "alias", "org/base")),
                video_inventory=(VideoInventoryEntry(name="ltx", family="ltx", supports_i2v=True, source="alias", resolved_path="org/missing"),),
            )
        )

    def test_delete_installed_model(self, client, layout):
        response = client.delete("/api/models/custom")

        assert response.status_code == 200
        assert not (layout.models_dir / "custom").exists()

    def test_delete_huggingface_download(self, client, layout):
        response = client.delete("/api/models/base")

        assert response.status_code == 200
        assert not (layout.hub / "models--org--base").exists()

    def test_unknown_or_undownloaded_model_is_404(self, client):
        assert client.delete("/api/models/nope").status_code == 404
        assert client.delete("/api/models/ltx").status_code == 404

    def test_delete_lora(self, client, layout):
        assert client.delete("/api/loras/style").status_code == 200
        assert not (layout.data_dir / "loras" / "style.safetensors").exists()
        assert client.delete("/api/loras/style").status_code == 404

    def test_deleting_a_huggingface_download_is_refused_while_its_repo_downloads(self, client, layout, downloading):
        downloading(DownloadTarget("Org/Base"))

        response = client.delete("/api/models/base")

        assert response.status_code == 409
        assert (layout.hub / "models--org--base").is_dir()

    def test_deleting_an_installed_model_is_refused_while_a_download_writes_inside_it(self, client, layout, downloading):
        downloading(DownloadTarget("org/part", local_dir=str(layout.models_dir / "custom" / ".part")))

        assert client.delete("/api/models/custom").status_code == 409
        assert (layout.models_dir / "custom").is_dir()

    def test_deletes_go_ahead_when_the_download_is_of_something_else(self, client, layout, downloading):
        downloading(DownloadTarget("org/other-repo"))

        assert client.delete("/api/models/custom").status_code == 200
        assert client.delete("/api/models/base").status_code == 200
        assert not (layout.hub / "models--org--base").exists()

    def test_deleting_a_lora_ignores_running_downloads(self, client, layout, downloading):
        downloading(DownloadTarget("org/base"))

        assert client.delete("/api/loras/style").status_code == 200

    def test_deletes_are_refused_while_a_job_runs(self, client, layout, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j1"})

        assert client.delete("/api/models/custom").status_code == 409
        assert client.delete("/api/loras/style").status_code == 409
        assert (layout.models_dir / "custom").is_dir()


def test_models_payload_describes_what_each_delete_removes(layout):
    from zvisiongenerator.web.workspace_api import _delete_info_resolver

    delete_info = _delete_info_resolver(layout.models_dir)

    assert delete_info(_image_entry("custom", "installed", "")) == {"kind": "installed", "repo_id": None, "linked_by": [], "stored_quants": [], "source_repos": []}
    assert delete_info(_image_entry("base", "alias", "org/base")) == {"kind": "huggingface", "repo_id": "org/base", "linked_by": ["custom"], "stored_quants": [], "source_repos": []}
    assert delete_info(_image_entry("missing", "alias", "org/missing")) is None
    _touch(layout.hub / "models--org--partial" / "snapshots" / "abc" / "model_index.json")
    assert delete_info(_ltx_entry("partial", "alias", "org/partial")) == {"kind": "huggingface", "repo_id": "org/partial", "linked_by": [], "stored_quants": [], "source_repos": []}


class TestMacH3Delete:
    """MiniMax H3 on macOS downloads parts of three repositories and keeps a text-encoder copy beside the model."""

    _REF = "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"

    @pytest.fixture
    def h3(self, layout, monkeypatch):
        from zvisiongenerator.utils.config import load_config

        monkeypatch.setattr("sys.platform", "darwin")
        for repo in ("MiniMaxAI--MiniMax-H3", "water1234--MiniMax-H3-MLX-Argus-Calibrated-INT8", "lightx2v--Minimax-h3-Turbo"):
            _touch(layout.hub / f"models--{repo}" / "snapshots" / "abc" / "part.json")
        _touch(layout.models_dir / "h3@mlx-te4" / "model-00001-of-00001.safetensors")
        _touch(layout.models_dir / "h3@mlx-te4" / "ziv-quant.json")
        entry = VideoInventoryEntry(name="h3", family="h3", supports_i2v=True, source="alias", resolved_path=self._REF)
        layout.entry, layout.config = entry, load_config()
        return layout

    def test_deleting_h3_removes_its_downloads_and_the_text_encoder_copy(self, h3):
        scope = delete_scope(h3.entry, (h3.entry,), h3.config, models_dir=h3.models_dir, platform_key="darwin")
        target = model_delete_target(h3.entry, h3.models_dir, **scope)

        assert target is not None and target.kind == "huggingface"
        assert [path.name for path in target.stored_quants] == ["h3@mlx-te4"]
        assert set(target.companion_repos) == {"water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "lightx2v/Minimax-h3-Turbo"}

        delete_model(target)

        assert not (h3.models_dir / "h3@mlx-te4").exists()
        assert not (h3.hub / "models--MiniMaxAI--MiniMax-H3").exists()
        assert not (h3.hub / "models--water1234--MiniMax-H3-MLX-Argus-Calibrated-INT8").exists()
        assert not (h3.hub / "models--lightx2v--Minimax-h3-Turbo").exists()
        assert (h3.hub / "models--org--other").is_dir()
        assert (h3.models_dir / "custom").is_dir()

    def test_the_models_payload_names_the_copy_among_what_is_removed(self, h3):
        from zvisiongenerator.web.workspace_api import _delete_info_resolver

        delete_info = _delete_info_resolver(h3.models_dir, inventory=(h3.entry,), app_config=h3.config)

        info = delete_info(h3.entry)
        assert info is not None
        assert (info["kind"], info["stored_quants"]) == ("huggingface", ["h3@mlx-te4"])

    def test_the_copy_counts_towards_the_disk_size_unless_it_is_a_row_of_its_own(self, h3):
        from zvisiongenerator.web.workspace_api import _disk_size_resolver

        os.truncate(h3.models_dir / "h3@mlx-te4" / "model-00001-of-00001.safetensors", 14_000_000_000)

        assert _disk_size_resolver(h3.models_dir, listed={"h3"}, app_config=h3.config)(h3.entry) == "14.0 GB"
        assert _disk_size_resolver(h3.models_dir, listed={"h3", "h3@mlx-te4"}, app_config=h3.config)(h3.entry) == "1 KB"

    def test_a_model_never_downloaded_is_not_deletable(self, h3):
        shutil.rmtree(h3.hub / "models--MiniMaxAI--MiniMax-H3")

        assert model_delete_target(h3.entry, h3.models_dir, **delete_scope(h3.entry, (h3.entry,), h3.config, models_dir=h3.models_dir, platform_key="darwin")) is None

    def test_an_installed_h3_folder_is_removed_with_its_copy(self, h3):
        _touch(h3.models_dir / "h3-local" / "FL2VA" / "model_index.json")
        _touch(h3.models_dir / "h3-local@mlx-te4" / "ziv-quant.json")
        entry = VideoInventoryEntry(name="h3-local", family="h3", supports_i2v=True, source="installed", resolved_path=str(h3.models_dir / "h3-local"))

        target = model_delete_target(entry, h3.models_dir)
        assert target is not None
        delete_model(target)

        assert not (h3.models_dir / "h3-local").exists()
        assert not (h3.models_dir / "h3-local@mlx-te4").exists()
        assert (h3.models_dir / "h3@mlx-te4").is_dir()

    def test_the_copy_itself_has_no_stored_quants_of_its_own(self, h3):
        entry = VideoInventoryEntry(name="h3@mlx-te4", family="h3", supports_i2v=True, source="installed", resolved_path=str(h3.models_dir / "h3@mlx-te4"))

        target = model_delete_target(entry, h3.models_dir)

        assert target is not None and target.stored_quants == ()


class TestSeedvr2DeleteTarget:
    def test_a_stored_copy_is_the_target(self, layout):
        copy = seedvr2_weights_dir("seedvr2-7b", layout.models_dir, MLX_WEIGHTS)
        _touch(copy / "transformer" / "weights.safetensors")

        target = seedvr2_delete_target("seedvr2-7b", layout.models_dir, MLX_WEIGHTS)

        assert target is not None
        assert (target.kind, target.path, target.stored_quants) == ("installed", copy, ())

    def test_no_copy_means_no_target(self, layout):
        assert seedvr2_delete_target("seedvr2-7b", layout.models_dir, MLX_WEIGHTS) is None

    def test_deleting_the_copy_leaves_the_huggingface_cache_and_other_models(self, layout):
        copy = seedvr2_weights_dir("seedvr2-7b", layout.models_dir, MLX_WEIGHTS)
        _touch(copy / "transformer" / "weights.safetensors")

        delete_model(seedvr2_delete_target("seedvr2-7b", layout.models_dir, MLX_WEIGHTS))

        assert not copy.exists()
        assert (layout.models_dir / "custom").is_dir()
        assert layout.base_snapshot.is_dir()


class TestSeedvr2Fp8DeleteTarget:
    def test_the_downloaded_files_folder_is_the_target(self, layout):
        folder = seedvr2_weights_dir("seedvr2-7b", layout.models_dir, FP8_WEIGHTS)
        _touch(folder / "ema_vae_fp16.safetensors")

        target = seedvr2_delete_target("seedvr2-7b", layout.models_dir, FP8_WEIGHTS)

        assert target is not None
        assert (target.kind, target.path) == ("installed", folder)

    def test_a_macos_copy_is_not_the_target(self, layout):
        _touch(seedvr2_weights_dir("seedvr2-7b", layout.models_dir, MLX_WEIGHTS) / "transformer" / "weights.safetensors")

        assert seedvr2_delete_target("seedvr2-7b", layout.models_dir, FP8_WEIGHTS) is None

    def test_deleting_it_leaves_the_huggingface_cache(self, layout):
        folder = seedvr2_weights_dir("seedvr2-7b", layout.models_dir, FP8_WEIGHTS)
        _touch(folder / "ema_vae_fp16.safetensors")
        cached = layout.hub / "models--numz--SeedVR2_comfyUI" / "snapshots" / "rev" / "ema_vae_fp16.safetensors"
        _touch(cached)

        delete_model(seedvr2_delete_target("seedvr2-7b", layout.models_dir, FP8_WEIGHTS))

        assert not folder.exists()
        assert cached.is_file()


@pytest.fixture(params=[MLX_WEIGHTS, FP8_WEIGHTS], ids=["mlx", "fp8"])
def seedvr2_route(request, layout, web_client):
    """Serve a SeedVR2 row for each weights engine, with its stored weights folder on disk."""
    weights = request.param
    tier = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)
    settings = SeedVR2Settings(model="auto", softness=0.75, video_softness=0.0, reference_megapixels=16, tiers=(tier,))
    folder = seedvr2_weights_dir("seedvr2-7b", layout.models_dir, weights)
    _touch(folder / "weights.safetensors")
    web_config = SimpleNamespace(
        data_dir=str(layout.data_dir),
        models_dir=str(layout.models_dir),
        image_inventory=(_image_entry("custom", "installed", str(layout.models_dir / "custom")),),
        video_inventory=(),
        seedvr2=SeedVR2Setup(settings=settings, tier=tier, budget_bytes=None, downloaded=True, weights=weights),
    )
    own, other = ("seedvr2-7b@q8", "seedvr2-7b@fp8") if weights is MLX_WEIGHTS else ("seedvr2-7b@fp8", "seedvr2-7b@q8")
    return SimpleNamespace(client=web_client(web_config), web_config=web_config, folder=folder, own=own, other=other)


class TestSeedvr2DeleteRoute:
    def test_delete_removes_only_the_weights_folder(self, seedvr2_route, layout):
        response = seedvr2_route.client.delete(f"/api/models/{seedvr2_route.own}")

        assert response.status_code == 200
        assert not seedvr2_route.folder.exists()
        assert (layout.models_dir / "custom").is_dir()
        assert layout.base_snapshot.is_dir()

    def test_a_second_delete_finds_nothing(self, seedvr2_route):
        assert seedvr2_route.client.delete(f"/api/models/{seedvr2_route.own}").status_code == 200
        assert seedvr2_route.client.delete(f"/api/models/{seedvr2_route.own}").status_code == 404

    def test_other_names_and_the_other_engines_row_are_404(self, seedvr2_route):
        for name in (seedvr2_route.other, "seedvr2-3b@q8", "seedvr2-7b"):
            assert seedvr2_route.client.delete(f"/api/models/{name}").status_code == 404, name
        assert seedvr2_route.folder.is_dir()

    def test_delete_is_404_where_seedvr2_is_off(self, seedvr2_route):
        seedvr2_route.web_config.seedvr2 = None

        assert seedvr2_route.client.delete(f"/api/models/{seedvr2_route.own}").status_code == 404
        assert seedvr2_route.folder.is_dir()

    def test_delete_is_refused_while_a_job_runs(self, seedvr2_route, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j1"})

        assert seedvr2_route.client.delete(f"/api/models/{seedvr2_route.own}").status_code == 409
        assert seedvr2_route.folder.is_dir()

    def test_names_that_climb_out_of_the_copy_folder_are_404(self, seedvr2_route, layout):
        sibling = layout.models_dir / "custom"
        for name in ("..", ".seedvr2", "%2e%2e%2fcustom", "..%2Fcustom", "seedvr2-7b@q8%2F..%2F..%2Fcustom", "SEEDVR2-7B@Q8"):
            assert seedvr2_route.client.delete(f"/api/models/{name}").status_code in {404, 405}, name
        assert seedvr2_route.folder.is_dir()
        assert sibling.is_dir()
        assert layout.base_snapshot.is_dir()

    def test_a_symlinked_copy_is_unlinked_and_its_target_survives(self, seedvr2_route, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        _touch(elsewhere / "weights.safetensors")
        shutil.rmtree(seedvr2_route.folder)
        seedvr2_route.folder.symlink_to(elsewhere, target_is_directory=True)

        assert seedvr2_route.client.delete(f"/api/models/{seedvr2_route.own}").status_code == 200

        assert not seedvr2_route.folder.is_symlink()
        assert (elsewhere / "weights.safetensors").is_file()


def _shared_blob_repo(hub: Path, repo: str, xet_hash: str) -> Path:
    """Cache *repo* the way huggingface_hub stores Xet downloads: its blob links into the cache-wide shared store."""
    store = hub / "blobs"
    store.mkdir(parents=True, exist_ok=True)
    (store / ".huggingface-shared-blobs").write_text("1\n")
    payload = _touch(store / xet_hash[:2] / xet_hash)
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    blob = repo_dir / "blobs" / f"etag-{xet_hash[:8]}"
    blob.parent.mkdir(parents=True)
    blob.symlink_to(Path("..", "..", "blobs", xet_hash[:2], xet_hash))
    manifest = payload.with_name(f"{xet_hash}.refs")
    manifest.write_text("".join(f"{line}\n" for line in [*(manifest.read_text().splitlines() if manifest.exists() else []), blob.relative_to(hub).as_posix()]))
    snapshot = repo_dir / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(Path("..", "..", "blobs", blob.name))
    _touch(repo_dir / "refs" / "main").write_text("a" * 40)
    return payload


def test_repo_size_counts_files_in_the_shared_blob_store(layout):
    from zvisiongenerator.utils.hf_cache import repo_disk_bytes

    payload = _shared_blob_repo(layout.hub, "org/enh", "ef" * 32)
    payload.write_bytes(b"x" * 1000)
    _touch(layout.hub / "models--org--enh" / "blobs" / "small").write_bytes(b"y" * 10)

    assert repo_disk_bytes(layout.hub / "models--org--enh") == 1000 + 10 + 40  # shared payload, local blob, refs/main


def test_disk_size_label_picks_a_readable_unit():
    from zvisiongenerator.utils.model_files import disk_size_label

    assert [disk_size_label(n) for n in (34_720_000_000, 92_400_000, 4_200, 12)] == ["34.7 GB", "92 MB", "4 KB", "1 KB"]


def test_disk_size_counts_own_files_and_unlisted_stored_quants(layout):
    from zvisiongenerator.web.workspace_api import _disk_size_resolver

    # Sparse files: the sizes count, without writing gigabytes.
    os.truncate(_touch(layout.models_dir / "base@q8" / "weights"), 2_000_000_000)
    os.truncate(layout.models_dir / "custom" / "transformer" / "model.safetensors", 1_000_000_000)
    disk_size = _disk_size_resolver(layout.models_dir, listed={"custom", "base"}, app_config={})

    # The converted model counts its own folder, not the base files it links to.
    assert disk_size(_image_entry("custom", "installed", "")) == "1.0 GB"
    # The alias counts its download and its stored quant, which has no row of its own here...
    assert disk_size(_image_entry("base", "alias", "org/base")) == "2.0 GB"
    # ...but not one listed as its own row.
    assert _disk_size_resolver(layout.models_dir, listed={"base", "base@q8"}, app_config={})(_image_entry("base", "alias", "org/base")) == "1 KB"
    assert disk_size(_image_entry("missing", "alias", "org/missing")) is None


class TestSharedBlobDelete:
    def test_deleting_a_download_frees_its_shared_blob(self, layout):
        payload = _shared_blob_repo(layout.hub, "org/enh", "ab" * 32)

        delete_model(enhancer_delete_target("org/enh"))

        assert not (layout.hub / "models--org--enh").exists()
        assert not payload.exists()

    def test_a_blob_another_repo_still_uses_stays(self, layout):
        payload = _shared_blob_repo(layout.hub, "org/enh", "cd" * 32)
        _shared_blob_repo(layout.hub, "org/mirror", "cd" * 32)

        delete_model(enhancer_delete_target("org/enh"))

        assert not (layout.hub / "models--org--enh").exists()
        assert payload.is_file()
        assert (layout.hub / "models--org--mirror" / "snapshots" / ("a" * 40) / "model.safetensors").is_file()


_ENHANCER_CONFIG = {
    "prompt_enhancer": {
        "model": "org/fast",
        "download_size_label": dict.fromkeys(("darwin", "win32", "linux"), "9 GB"),
        "smart": {"model": "org/smart", "download_size_label": dict.fromkeys(("darwin", "win32", "linux"), "19 GB"), "min_memory_gb": 9},
    }
}


def test_enhancer_tier_models_lists_fast_smart_and_custom():
    assert enhancer_tier_models(_ENHANCER_CONFIG, platform_key="linux") == {"fast": ("org/fast", None), "smart": ("org/smart", None)}
    custom = {"prompt_enhancer": {**_ENHANCER_CONFIG["prompt_enhancer"], "user_model": "org/custom@abc123"}}
    assert enhancer_tier_models(custom, platform_key="linux")["custom"] == ("org/custom", "abc123")
    assert enhancer_tier_models({}, platform_key="linux") == {}


def test_enhancer_delete_target_needs_a_cached_hub_repo(layout, tmp_path):
    target = enhancer_delete_target("org/base@abc")

    assert (target.kind, target.path, target.repo_id) == ("huggingface", layout.hub / "models--org--base", "org/base")
    assert enhancer_delete_target("org/never-downloaded") is None
    assert enhancer_delete_target(str(tmp_path)) is None


def test_models_payload_lists_enhancer_tiers(layout):
    from zvisiongenerator.web.workspace_api import _enhancer_model_rows

    _touch(layout.hub / "models--org--fast" / "blobs" / "weights")
    rows = _enhancer_model_rows(_ENHANCER_CONFIG, downloaded=lambda repo, _revision: repo == "org/fast", budget_bytes=lambda: 8 * 2**30)

    fast, smart = rows
    assert (fast["tier"], fast["repo"], fast["downloaded"], fast["deletable"], fast["unavailable"]) == ("fast", "org/fast", True, True, None)
    assert fast["disk_size_label"] is not None
    assert (smart["tier"], smart["downloaded"], smart["deletable"], smart["disk_size_label"], smart["download_size_label"]) == ("smart", False, False, None, "19 GB")
    assert smart["unavailable"] is not None


class TestEnhancerDeleteRoute:
    @pytest.fixture
    def session(self):
        return PromptEnhancerSession(lambda repo, revision: None, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)

    @pytest.fixture
    def client(self, layout, monkeypatch, session):
        _touch(layout.hub / "models--org--fast" / "snapshots" / "abc" / "model.safetensors")
        monkeypatch.setattr(web_server, "load_web_config", lambda: SimpleNamespace(app_config=_ENHANCER_CONFIG))
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: session)
        with TestClient(web_server.app) as client:
            yield client

    def test_deletes_the_tier_download_and_frees_the_slot(self, client, layout, session):
        response = client.delete("/api/enhancer-models/fast")

        assert response.status_code == 200
        assert not (layout.hub / "models--org--fast").exists()
        assert not session.busy()

    def test_unknown_or_undownloaded_tier_is_404(self, client):
        assert client.delete("/api/enhancer-models/smart").status_code == 404
        assert client.delete("/api/enhancer-models/org--fast").status_code == 404

    def test_refused_while_the_tier_model_downloads_and_the_slot_stays_free(self, client, layout, session, downloading):
        downloading(DownloadTarget("org/fast"))

        response = client.delete("/api/enhancer-models/fast")

        assert response.status_code == 409
        assert (layout.hub / "models--org--fast").is_dir()
        assert not session.busy()

    def test_refused_while_an_enhancement_runs(self, client, layout, session):
        session.reserve()

        assert client.delete("/api/enhancer-models/fast").status_code == 409
        assert (layout.hub / "models--org--fast").is_dir()

    def test_refused_while_a_job_runs(self, client, layout, monkeypatch):
        def _busy(admit, *, busy_message):
            raise web_server.JobConflictError(busy_message)

        monkeypatch.setattr(web_server.web_runner, "admit_exclusive", _busy)

        assert client.delete("/api/enhancer-models/fast").status_code == 409
        assert (layout.hub / "models--org--fast").is_dir()


class TestBuiltModelDelete:
    """MOSS-Image and Ideogram 4 Instant folders hard-link their cache sources: deleting a row frees the sources nothing else needs."""

    _CONFIG = {
        "moss_image": {"sft": "org/moss-sft@" + "d" * 40, "turbo_lora": {"repo": "org/moss-turbo", "revision": "e" * 40, "filename": "turbo.safetensors"}},
        "ideogram4_instant": {
            "transformer": {"repo": "org/instant-transformer", "revision": "a" * 40, "filename": "t.safetensors"},
            "components": {"linux": "org/ideo-nf4@" + "c" * 40},
            "transformer_format": {"linux": "bf16"},
        },
    }

    @pytest.fixture
    def built(self, layout):
        """Cache repos for every source; a helper that makes a row built (an installed folder) or leaves it unbuilt (an alias)."""
        for repo in ("org--moss-sft", "org--moss-turbo", "org--ideo-nf4"):
            _touch(layout.hub / f"models--{repo}" / "snapshots" / "abc" / "model.safetensors")
        layout.inventory = {}

        def _row(name: str, *, built: bool) -> ImageInventoryEntry:
            path = layout.models_dir / name
            if built:
                _touch(path / "transformer" / "model.safetensors")
                write_record(path, {"version": 1})
            entry = _image_entry(name, "installed" if built else "alias", str(path))
            layout.inventory[name] = entry
            return entry

        layout.row = _row
        return layout

    def _target(self, built, entry):
        inventory = tuple(built.inventory.values())
        scope = delete_scope(entry, inventory, self._CONFIG, models_dir=built.models_dir, platform_key="linux")
        return model_delete_target(entry, built.models_dir, **scope)

    def _cached(self, built) -> set[str]:
        return {path.name.removeprefix("models--") for path in built.hub.glob("models--*")}

    def test_deleting_the_only_built_moss_removes_the_folder_and_both_sources(self, built):
        moss = built.row("moss", built=True)

        target = self._target(built, moss)
        assert target is not None and target.kind == "installed"
        assert target.companion_repos == ("org/moss-sft", "org/moss-turbo")
        delete_model(target)

        assert not (built.models_dir / "moss").exists()
        assert self._cached(built) == {"org--ideo-nf4", "org--base", "org--other"}

    def test_deleting_moss_keeps_the_sft_repo_while_moss_sft_is_built(self, built):
        moss, _sft = built.row("moss", built=True), built.row("moss-sft", built=True)

        target = self._target(built, moss)
        assert target is not None and target.companion_repos == ("org/moss-turbo",)
        delete_model(target)

        assert "org--moss-sft" in self._cached(built) and "org--moss-turbo" not in self._cached(built)

    def test_deleting_moss_sft_keeps_everything_while_moss_is_built(self, built):
        built.row("moss", built=True)
        sft = built.row("moss-sft", built=True)

        target = self._target(built, sft)
        assert target is not None and target.companion_repos == ()
        delete_model(target)

        assert not (built.models_dir / "moss-sft").exists()
        assert {"org--moss-sft", "org--moss-turbo"} <= self._cached(built)

    def test_deleting_the_sole_built_moss_frees_both_repos_even_with_moss_sft_unbuilt(self, built):
        moss = built.row("moss", built=True)
        built.row("moss-sft", built=False)

        target = self._target(built, moss)
        assert target is not None and target.companion_repos == ("org/moss-sft", "org/moss-turbo")
        delete_model(target)

        assert self._cached(built) == {"org--ideo-nf4", "org--base", "org--other"}

    def test_an_unbuilt_moss_row_deletes_its_downloaded_sources(self, built):
        moss = built.row("moss", built=False)

        target = self._target(built, moss)
        assert target is not None and target.kind == "installed"
        assert target.companion_repos == ("org/moss-sft", "org/moss-turbo")
        delete_model(target)

        assert self._cached(built) == {"org--ideo-nf4", "org--base", "org--other"}

    def test_an_unbuilt_moss_row_keeps_only_the_download_a_built_row_needs(self, built):
        moss = built.row("moss", built=False)
        built.row("moss-sft", built=True)

        target = self._target(built, moss)
        assert target is not None and target.companion_repos == ("org/moss-turbo",)
        delete_model(target)

        assert "org--moss-sft" in self._cached(built) and "org--moss-turbo" not in self._cached(built)

    def test_an_unbuilt_row_with_nothing_downloaded_is_not_deletable(self, built):
        shutil.rmtree(built.hub / "models--org--moss-sft")
        shutil.rmtree(built.hub / "models--org--moss-turbo")

        assert self._target(built, built.row("moss", built=False)) is None

    def test_deleting_ideo_removes_the_components_repo(self, built):
        ideo = built.row("ideo", built=True)

        target = self._target(built, ideo)
        assert target is not None and target.companion_repos == ("org/ideo-nf4",)
        delete_model(target)

        assert "org--ideo-nf4" not in self._cached(built) and "org--moss-sft" in self._cached(built)

    def test_the_models_payload_names_the_sources_and_sizes_an_unbuilt_row_by_them(self, built):
        from zvisiongenerator.web.workspace_api import _delete_info_resolver, _disk_size_resolver

        moss, sft = built.row("moss", built=True), built.row("moss-sft", built=False)
        inventory = (moss, sft)

        delete_info = _delete_info_resolver(built.models_dir, inventory=inventory, app_config=self._CONFIG)
        disk_size = _disk_size_resolver(built.models_dir, listed={"moss", "moss-sft"}, app_config=self._CONFIG, inventory=inventory)

        # The unbuilt moss-sft row does not pin the SFT repo, so deleting the built moss frees both.
        assert delete_info(moss)["source_repos"] == ["org/moss-sft", "org/moss-turbo"]
        # moss-sft is downloaded but not built, and moss is built: the SFT repo is moss's, so the unbuilt row frees nothing.
        assert delete_info(sft) is None and disk_size(sft) is None

    def _hidden_download(self, built) -> Path:
        return _touch(built.models_dir / ".ideo.download" / "t.safetensors").parent

    def test_an_unbuilt_ideo_row_deletes_its_transformer_download_with_the_components(self, built):
        hidden = self._hidden_download(built)
        ideo = built.row("ideo", built=False)

        target = self._target(built, ideo)
        assert target is not None and target.build_downloads == (hidden,)
        delete_model(target)

        assert not hidden.exists() and "org--ideo-nf4" not in self._cached(built)

    def test_deleting_a_built_ideo_row_also_removes_a_download_left_by_an_interrupted_rebuild(self, built):
        hidden = self._hidden_download(built)

        target = self._target(built, built.row("ideo", built=True))
        assert target is not None
        delete_model(target)

        assert not hidden.exists() and not (built.models_dir / "ideo").exists()

    def test_the_unbuilt_ideo_row_sizes_the_transformer_download_too(self, built):
        from zvisiongenerator.web.workspace_api import _disk_size_resolver

        shutil.rmtree(built.hub / "models--org--ideo-nf4")
        self._hidden_download(built)
        ideo = built.row("ideo", built=False)
        size = _disk_size_resolver(built.models_dir, listed={"ideo"}, app_config=self._CONFIG, inventory=(ideo,))

        assert size(ideo) == "1 KB"

    @pytest.fixture
    def client(self, built, web_client, monkeypatch):
        moss = built.row("moss", built=True)
        monkeypatch.setattr("sys.platform", "linux")
        return web_client(SimpleNamespace(data_dir=str(built.data_dir), models_dir=str(built.models_dir), seedvr2=None, app_config=self._CONFIG, image_inventory=(moss,), video_inventory=()))

    def test_an_unbuilt_row_shows_the_disk_its_deletable_sources_use(self, built):
        from zvisiongenerator.web.workspace_api import _disk_size_resolver

        moss = built.row("moss", built=False)
        size = _disk_size_resolver(built.models_dir, listed={"moss"}, app_config=self._CONFIG, inventory=(moss,))

        assert size(moss) == "1 KB"
        shutil.rmtree(built.hub / "models--org--moss-sft")
        shutil.rmtree(built.hub / "models--org--moss-turbo")
        assert size(moss) is None

    def test_the_route_deletes_the_row_with_its_sources(self, client, built):
        assert client.delete("/api/models/moss").status_code == 200

        assert not (built.models_dir / "moss").exists()
        assert not {"org--moss-sft", "org--moss-turbo"} & self._cached(built)

    def test_the_route_refuses_while_the_transformer_downloads(self, client, built, downloading, monkeypatch):
        hidden = self._hidden_download(built)
        ideo = built.row("ideo", built=False)
        web_config = SimpleNamespace(data_dir=str(built.data_dir), models_dir=str(built.models_dir), seedvr2=None, app_config=self._CONFIG, image_inventory=(ideo,), video_inventory=())
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        downloading(DownloadTarget("org/instant-transformer", allow_patterns=("t.safetensors",), local_dir=str(hidden)))

        assert client.delete("/api/models/ideo").status_code == 409
        assert hidden.is_dir()

    def test_the_route_refuses_while_a_source_downloads(self, client, built, downloading):
        downloading(DownloadTarget("org/moss-turbo", allow_patterns=("*.safetensors",)))

        assert client.delete("/api/models/moss").status_code == 409
        assert (built.models_dir / "moss").is_dir() and "org--moss-turbo" in self._cached(built)
