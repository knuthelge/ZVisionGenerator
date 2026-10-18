"""Tests for reading the HuggingFace cache and finding the downloads nothing uses."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from fastapi.testclient import TestClient

from zvisiongenerator.utils.hf_cache import cached_model_repos, free_stranded_shared_files, stranded_shared_files
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.model_delete import cache_repos_linked_from, installed_models_linking_to, used_repo_ids
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.web.workspace_api import _other_downloads_payload, find_other_downloads

_SNAPSHOT = "a" * 40


def _touch(path: Path, size: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _repo(hub: Path, repo_id: str, *, shared: dict[str, int] | None = None, local: int = 0) -> Path:
    """Cache *repo_id* with blobs linked into the shared store (*shared*: xet hash to size) and a local blob of *local* bytes."""
    repo_dir = hub / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "snapshots" / _SNAPSHOT).mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    if local:
        _touch(repo_dir / "blobs" / "local", local)
    store = hub / "blobs"
    for xet_hash, size in (shared or {}).items():
        store.mkdir(exist_ok=True)
        (store / ".huggingface-shared-blobs").write_text("1\n")
        payload = store / xet_hash[:2] / xet_hash
        if not payload.exists():
            _touch(payload, size)
        blob = repo_dir / "blobs" / f"etag-{xet_hash[:8]}"
        blob.symlink_to(Path("..", "..", "blobs", xet_hash[:2], xet_hash))
        with payload.with_name(f"{xet_hash}.refs").open("a") as refs:
            refs.write(f"{blob.relative_to(hub).as_posix()}\n")
    return repo_dir


def _alias(name: str, repo: str) -> ImageInventoryEntry:
    return ImageInventoryEntry(name=name, family="flux2_klein", size=None, source="alias", resolved_path=repo)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    return hub


class TestCachedModelRepos:
    def test_a_shared_file_counts_only_for_the_repo_that_alone_links_to_it(self, hub):
        _repo(hub, "org/old", shared={"aa" * 32: 500, "bb" * 32: 300}, local=7)
        _repo(hub, "org/mirror", shared={"aa" * 32: 500})

        repos = {repo.repo_id: repo for repo in cached_model_repos(hub)}

        assert repos["org/old"].freed_bytes == 300 + 7
        assert repos["org/mirror"].freed_bytes == 0

    def test_a_shared_file_a_dataset_also_uses_is_not_counted(self, hub):
        xet_hash = "ff" * 32
        _repo(hub, "org/model", shared={xet_hash: 900})
        dataset_blob = hub / "datasets--org--data" / "blobs" / "etag-data"
        dataset_blob.parent.mkdir(parents=True)
        dataset_blob.symlink_to(Path("..", "..", "blobs", xet_hash[:2], xet_hash))
        with (hub / "blobs" / xet_hash[:2] / f"{xet_hash}.refs").open("a") as refs:
            refs.write(f"{dataset_blob.relative_to(hub).as_posix()}\n")

        assert [repo.freed_bytes for repo in cached_model_repos(hub)] == [0]

    def test_lists_only_model_repos(self, hub):
        _repo(hub, "org/model")
        (hub / "datasets--org--data").mkdir()

        assert [repo.repo_id for repo in cached_model_repos(hub)] == ["org/model"]

    def test_a_missing_cache_has_no_repos(self, tmp_path):
        assert cached_model_repos(tmp_path / "missing") == ()


class TestStrandedSharedFiles:
    def test_files_left_by_a_removed_repo_folder_are_stranded_and_freed(self, hub):
        shutil.rmtree(_repo(hub, "org/gone", shared={"cc" * 32: 400}))
        _repo(hub, "org/kept", shared={"dd" * 32: 200})

        assert list(stranded_shared_files(hub).values()) == [400]
        assert free_stranded_shared_files(hub) == 400
        assert stranded_shared_files(hub) == {}
        assert (hub / "blobs" / "dd" / ("dd" * 32)).is_file()

    def test_no_shared_store_means_nothing_stranded(self, hub):
        _repo(hub, "org/plain", local=10)

        assert stranded_shared_files(hub) == {}


def test_used_repo_ids_cover_aliases_companions_enhancers_and_extras():
    config = {
        "prompt_enhancer": {"model": "org/Fast", "user_model": "/local/folder"},
        "video_model_presets": {"ltx": {"diffusers": {"default_repo": "org/ltx", "transformer": {"repo": "org/ltx-single", "filename": "t.safetensors"}}}},
    }
    inventory = [
        _alias("klein", "Org/Klein@abc123"),
        VideoInventoryEntry(name="ltx", family="ltx", supports_i2v=True, source="alias", resolved_path="org/ltx"),
        ImageInventoryEntry(name="custom", family="zimage", size=None, source="installed", resolved_path="/models/custom"),
    ]

    used = used_repo_ids(inventory, config, platform_key="linux", extra=["org/SeedVR2"])

    assert used == {"org/klein", "org/ltx", "org/ltx-single", "org/fast", "org/seedvr2"}


def test_cache_repos_linked_from_installed_models(hub, tmp_path):
    base = _repo(hub, "org/base")
    _touch(base / "snapshots" / _SNAPSHOT / "text_encoder" / "model.safetensors")
    models_dir = tmp_path / "models"
    converted = models_dir / "custom"
    _touch(converted / "transformer" / "model.safetensors")
    (converted / "text_encoder").symlink_to(base / "snapshots" / _SNAPSHOT / "text_encoder", target_is_directory=True)
    whole = _repo(hub, "org/whole")
    (models_dir / "linked-folder").symlink_to(whole / "snapshots" / _SNAPSHOT, target_is_directory=True)
    _repo(hub, "org/unlinked")

    assert cache_repos_linked_from(models_dir, hub) == {"models--org--base", "models--org--whole"}


def test_a_file_link_into_a_xet_repo_names_the_repo_not_the_shared_store(hub, tmp_path):
    xet = _repo(hub, "org/xet", shared={"ab" * 32: 100})
    (xet / "snapshots" / _SNAPSHOT / "model.safetensors").symlink_to(Path("..", "..", "blobs", f"etag-{'ab' * 4}"))
    models_dir = tmp_path / "models"
    (models_dir / "custom").mkdir(parents=True)
    (models_dir / "custom" / "model.safetensors").symlink_to(xet / "snapshots" / _SNAPSHOT / "model.safetensors")

    assert cache_repos_linked_from(models_dir, hub) == {"models--org--xet"}
    assert installed_models_linking_to(xet, models_dir) == ("custom",)


@pytest.fixture
def web_config(hub, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _repo(hub, "org/used", local=5)
    _repo(hub, "org/old", local=1_500)
    return SimpleNamespace(
        data_dir=str(tmp_path),
        models_dir=str(models_dir),
        seedvr2=None,
        app_config={},
        image_inventory=(_alias("used", "org/used"),),
        video_inventory=(),
    )


def test_other_downloads_are_the_repos_nothing_uses(web_config, hub):
    assert [repo.repo_id for repo in find_other_downloads(web_config, cache_dir=hub)] == ["org/old"]


def test_models_payload_lists_other_downloads(web_config, hub):
    payload = _other_downloads_payload(web_config)

    assert payload["cache_dir"] == os.fspath(hub)
    assert payload["repos"] == [{"repo_id": "org/old", "size_label": "2 KB"}]
    assert payload["stranded"] is None


class TestOtherDownloadRoutes:
    @pytest.fixture
    def client(self, web_config, monkeypatch):
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)
        with TestClient(web_server.app) as client:
            yield client

    def test_deletes_an_unused_download(self, client, hub):
        response = client.delete("/api/hf-cache/org/old")

        assert response.status_code == 200
        assert not (hub / "models--org--old").exists()

    def test_a_download_in_use_or_an_odd_name_is_404_and_stays(self, client, hub):
        assert client.delete("/api/hf-cache/org/used").status_code == 404
        assert client.delete("/api/hf-cache/..%2F..%2Fetc").status_code == 404
        assert (hub / "models--org--used").is_dir()

    def test_frees_stranded_shared_files(self, client, hub):
        shutil.rmtree(_repo(hub, "org/gone", shared={"ee" * 32: 2_000_000}))

        response = client.delete("/api/hf-cache/stranded")

        assert response.status_code == 200
        assert not (hub / "blobs" / "ee" / ("ee" * 32)).exists()

    def test_refused_while_a_job_runs(self, client, hub, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j1"})

        assert client.delete("/api/hf-cache/org/old").status_code == 409
        assert client.delete("/api/hf-cache/stranded").status_code == 409
        assert (hub / "models--org--old").is_dir()
