"""Tests for the audio rows of the Models page: the inventory payload, the download plan, the delete and the cache listing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import HIGGS_CODEC_REPO, HIGGS_REPO, YUE2_REPO, audio_config
from tests._factories import load_packaged_config
from tests._licences import use_licences

from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.hf_snapshot import cached_snapshot
from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.hf_cache import repo_disk_bytes
from zvisiongenerator.utils.licences import load_licences
from zvisiongenerator.utils.model_files import disk_size_label
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.audio_inventory import audio_inventory_rows, audio_repo_ids
from zvisiongenerator.web.model_delete import audio_delete_target, delete_model
from zvisiongenerator.web.model_download_plan import parts_summary, row_plan

_BUDGET = MemoryBudget(10 * GIB, system_bytes=64 * GIB)
_LICENCES = {
    "yue2-nc": {
        "name": "CC BY-NC 4.0 with individual creator permission",
        "supplier": "M-A-P",
        "summary": "Non-commercial; individual creators may monetise their outputs, companies need a licence",
        "url": "https://github.com/multimodal-art-projection/YuE/blob/3acdec8cf1b97603bb696e4c6c4560348ffbb0b7/MODEL_LICENSE",
        "models": ["yue2"],
    }
}


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _cache(hub: Path, repo_id: str) -> Path:
    return _touch(hub / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc" / "model.safetensors")


@pytest.fixture
def env(monkeypatch, tmp_path):
    """An isolated data dir and Hub cache, on a CUDA machine."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: "cuda")
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: _BUDGET)
    use_licences(monkeypatch, tmp_path, _LICENCES)
    return SimpleNamespace(data_dir=tmp_path, models_dir=tmp_path / "models", hub=hub)


def _web_config(env, config=None) -> SimpleNamespace:
    return SimpleNamespace(
        app_config=audio_config() if config is None else config,
        data_dir=str(env.data_dir),
        models_dir=str(env.models_dir),
        image_inventory=(),
        video_inventory=(),
        seedvr2=None,
        quantize_options=(4, 8),
        loras_dir=str(env.data_dir / "loras"),
    )


def _rows(env, config=None, **kwargs) -> dict[str, dict]:
    config = audio_config() if config is None else config
    return {row["name"]: row for row in audio_inventory_rows(config, _BUDGET, **kwargs)}


class TestInventoryRows:
    def test_each_runnable_preset_is_a_row_with_the_fields_the_page_reads(self, env):
        rows = _rows(env, target_present=lambda target: target.repo_id.startswith("m-a-p/"))

        assert list(rows) == ["yue2", "higgs3", "sa3"]
        assert rows["yue2"] == {
            "name": "yue2",
            "label": "YuE2 3B",
            "kind": "music",
            "licences": [
                {
                    "name": "CC BY-NC 4.0 with individual creator permission",
                    "supplier": "M-A-P",
                    "summary": "Non-commercial; individual creators may monetise their outputs, companies need a licence",
                    "url": _LICENCES["yue2-nc"]["url"],
                    "url_label": "Licence",
                    "attribution": None,
                }
            ],
            "downloaded": True,
            "download_needed": False,
            "memory_fit": {"kind": "discrete", "budget_gb": 10.0, "by_quantize": {"none": {"status": "fits", "required_gb": 6.7}}},
            "delete": None,
            "disk_size_label": None,
        }
        assert [rows[name]["kind"] for name in rows] == ["music", "speech", "sound"]
        assert rows["sa3"]["downloaded"] is False and rows["sa3"]["download_needed"] is True
        assert rows["sa3"]["licences"] == []

    def test_a_model_with_files_on_disk_reports_their_size_and_what_deleting_removes(self, env):
        _cache(env.hub, YUE2_REPO.split("@")[0])

        row = _rows(env, target_present=lambda _target: False)["yue2"]

        assert row["delete"] == {"kind": "huggingface", "repo_id": "m-a-p/YuE2-3B", "linked_by": [], "stored_quants": [], "source_repos": []}
        assert row["disk_size_label"] == "1 KB"
        assert row["downloaded"] is False and row["download_needed"] is True

    def test_one_unreadable_model_leaves_unknown_status_and_the_other_rows(self, env):
        def broken(target: DownloadTarget) -> bool:
            if target.repo_id.startswith("m-a-p/"):
                raise OSError("cache unreadable")
            return True

        with pytest.warns(UserWarning, match="Could not determine status for audio model 'yue2'"):
            rows = _rows(env, target_present=broken)

        assert (rows["yue2"]["downloaded"], rows["yue2"]["memory_fit"]) == (None, None)
        assert rows["yue2"]["download_needed"] is False
        assert rows["sa3"]["downloaded"] is True

    def test_a_config_without_presets_has_no_rows_so_the_panel_stays_hidden(self, env):
        assert audio_inventory_rows({**load_packaged_config(), "audio_model_presets": {}}, _BUDGET) == []

    def test_the_packaged_config_has_a_row_per_model(self, env):
        rows = audio_inventory_rows(load_packaged_config(), _BUDGET)

        assert [row["name"] for row in rows] == ["yue2", "higgs3", "sa3"]

    def test_the_models_payload_lists_the_rows_after_the_video_models(self, env, monkeypatch):
        monkeypatch.setattr(audio_models_module, "target_present_offline", lambda _target: True)
        monkeypatch.setattr(workspace_api_module, "find_other_downloads", lambda *_a, **_k: ())

        models = workspace_api_module.build_models_response(_web_config(env), token_source=None, image_defaults_for=lambda *_a: {})

        assert [row["name"] for row in models["audio_models"]] == ["yue2", "higgs3", "sa3"]
        assert list(models).index("audio_models") == list(models).index("video_models") + 1


class TestHiggs:
    """Higgs TTS 3 has two repositories and nothing to prepare after them."""

    def _cache_blobs(self, env, repo: str, size: int) -> Path:
        """Cache *repo* as the Hub lays it out (a blob and a snapshot symlink) with a weights file of *size* bytes."""
        repo_dir = env.hub / f"models--{repo.split('@')[0].replace('/', '--')}"
        (repo_dir / "blobs").mkdir(parents=True)
        (repo_dir / "snapshots" / "abc").mkdir(parents=True)
        weights = repo_dir / "blobs" / "sha-weights"
        weights.write_bytes(b"w" * size)
        (repo_dir / "snapshots" / "abc" / "model.safetensors").symlink_to(weights)
        return repo_dir

    def test_it_is_downloaded_as_soon_as_both_repositories_are_present(self, env):
        row = _rows(env, target_present=lambda _target: True)["higgs3"]

        assert (row["downloaded"], row["download_needed"]) == (True, False)

    def test_a_missing_repository_needs_a_download(self, env):
        row = _rows(env, target_present=lambda target: target.repo_id != HIGGS_CODEC_REPO.split("@")[0])["higgs3"]

        assert (row["downloaded"], row["download_needed"]) == (False, True)

    def test_deleting_it_targets_exactly_its_two_repositories(self, env):
        for repo in (HIGGS_REPO, HIGGS_CODEC_REPO):
            _cache(env.hub, repo.split("@")[0])
        preset = {item.id: item for item in parse_audio_presets(audio_config())}["higgs3"]

        target = audio_delete_target(preset)

        assert target.kind == "huggingface"
        assert target.repo_ids == ("bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v2-tokenizer")
        delete_model(target)
        assert not any(env.hub.iterdir())

    def test_the_row_reports_what_deleting_removes(self, env):
        for repo in (HIGGS_REPO, HIGGS_CODEC_REPO):
            _cache(env.hub, repo.split("@")[0])

        row = _rows(env, target_present=lambda _target: True)["higgs3"]

        assert row["delete"] == {"kind": "huggingface", "repo_id": "bosonai/higgs-tts-3-4b", "linked_by": [], "stored_quants": [], "source_repos": []}

    def _snapshot_every_listed_file(self, env, files) -> Path:
        """Lay *files* out in the fake Hub cache at their pinned revision, as a finished download does."""
        for repo in files:
            for name in repo.patterns:
                _touch(env.hub / f"models--{repo.repo_id.replace('/', '--')}" / "snapshots" / repo.revision / name)
        return env.hub

    def test_the_plan_the_presence_check_and_the_loader_agree_on_the_packaged_files(self, env):
        """The row, the offline presence check and the backend's cache lookup all read the same file list."""
        config = load_packaged_config()
        preset = {item.id: item for item in parse_audio_presets(config)}["higgs3"]
        files = preset.files
        targets = row_plan("audio", "higgs3", _web_config(env, config)).targets

        assert [item.repo_id for item in files] == [target.repo_id for target in targets] == ["bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v2-tokenizer"]
        assert [set(item.patterns) for item in files] == [set(target.allow_patterns) for target in targets]
        assert not any(target_present_offline(target) for target in targets)
        assert _rows(env, config)["higgs3"]["downloaded"] is False

        self._snapshot_every_listed_file(env, files)

        assert all(target_present_offline(target) for target in targets)
        assert all(cached_snapshot(HubFiles(item.repo_id, item.revision, item.patterns)) is not None for item in files)
        assert [snapshot_dir(item).name for item in files] == [item.revision for item in files]
        assert _rows(env, config)["higgs3"]["downloaded"] is True

    @pytest.mark.parametrize("index", [0, 1])
    def test_a_listed_file_missing_from_either_repository_fails_all_three_alike(self, env, index):
        config = load_packaged_config()
        preset = {item.id: item for item in parse_audio_presets(config)}["higgs3"]
        files = preset.files
        self._snapshot_every_listed_file(env, files)
        broken = files[index]
        (env.hub / f"models--{broken.repo_id.replace('/', '--')}" / "snapshots" / broken.revision / broken.patterns[-1]).unlink()
        target = row_plan("audio", "higgs3", _web_config(env, config)).targets[index]

        assert target_present_offline(target) is False
        assert cached_snapshot(HubFiles(broken.repo_id, broken.revision, broken.patterns)) is None
        with pytest.raises(FileNotFoundError, match=broken.repo_id):
            snapshot_dir(broken)
        assert _rows(env, config)["higgs3"]["downloaded"] is False

    def test_the_size_is_the_sum_of_both_repositories(self, env):
        official = self._cache_blobs(env, HIGGS_REPO, 6_000_000)
        codec = self._cache_blobs(env, HIGGS_CODEC_REPO, 1_000)

        row = _rows(env, target_present=lambda _target: True)["higgs3"]

        assert row["disk_size_label"] == disk_size_label(repo_disk_bytes(official) + repo_disk_bytes(codec))
        assert row["disk_size_label"] == disk_size_label(6_001_000)


class TestDownloadPlan:
    def test_the_plan_is_the_presets_pinned_repositories(self, env):
        plan = row_plan("audio", "yue2", _web_config(env))

        assert (plan.model_id, plan.label) == ("yue2", "YuE2 3B")
        assert [(target.repo_id, target.revision) for target in plan.targets] == [
            ("m-a-p/YuE2-3B", "8f24312f187763b9854adeec87e437e2b03bbff1"),
            ("m-a-p/YuE2-Vae", "152733a19ad43aa67e367f9b5503ef8075bb5126"),
        ]

    def test_a_model_that_is_not_an_audio_row_is_refused(self, env):
        with pytest.raises(ValueError, match="Unknown audio model 'zit'"):
            row_plan("audio", "zit", _web_config(env))

    def test_the_summary_lists_the_missing_model_with_the_licence_lines_of_the_model(self, env):
        plan = row_plan("audio", "yue2", _web_config(env))

        summary = parts_summary(
            (plan,),
            check=lambda target: TargetCheck(missing_files=2, missing_bytes=5_000_000_000 if target.repo_id.endswith("3B") else 100),
            present=lambda _target: False,
            space=lambda *_a, **_k: None,
            cache_dir=lambda: env.hub,
            licences=load_licences(),
        )

        assert summary["parts"] == [{"label": "YuE2 3B", "model": "yue2", "bytes": 5_000_000_100}]
        assert summary["total_bytes"] == 5_000_000_100
        assert [(record["model"], record["name"]) for record in summary["licences"]] == [("yue2", "CC BY-NC 4.0 with individual creator permission")]

    def test_an_unreachable_hub_leaves_the_plan_unchecked(self, env):
        plan = row_plan("audio", "sa3", _web_config(env))

        assert parts_summary((plan,), check=lambda _target: None, present=lambda _target: False, cache_dir=lambda: env.hub)["unchecked"] is True


class TestDeleteRoute:
    @pytest.fixture
    def client(self, env, monkeypatch):
        web_config = _web_config(env)
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)
        with TestClient(web_server.app) as client:
            yield client

    def test_deletes_every_repository_of_the_model_and_nothing_else(self, client, env):
        for repo in (YUE2_REPO, "m-a-p/YuE2-Vae@x"):
            _cache(env.hub, repo.split("@")[0])
        _cache(env.hub, "org/other")

        response = client.delete("/api/audio-models/yue2")

        assert response.status_code == 200
        assert not (env.hub / "models--m-a-p--YuE2-3B").exists()
        assert not (env.hub / "models--m-a-p--YuE2-Vae").exists()
        assert (env.hub / "models--org--other").is_dir()

    def test_is_404_for_a_model_with_nothing_on_disk_and_for_an_unknown_one(self, client):
        assert client.delete("/api/audio-models/yue2").status_code == 404
        assert client.delete("/api/audio-models/zit").status_code == 404

    def test_is_refused_while_a_job_runs(self, client, env, monkeypatch):
        _cache(env.hub, YUE2_REPO.split("@")[0])
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "job-1"})

        assert client.delete("/api/audio-models/yue2").status_code == 409
        assert (env.hub / "models--m-a-p--YuE2-3B").is_dir()

    def test_is_refused_while_one_of_its_repositories_downloads(self, client, env):
        _cache(env.hub, YUE2_REPO.split("@")[0])
        flight, owner = web_server.download_flights.claim(DownloadTarget("m-a-p/YuE2-Vae"))
        assert owner
        try:
            assert client.delete("/api/audio-models/yue2").status_code == 409
        finally:
            web_server.download_flights.finish(flight, "stopped")
        assert (env.hub / "models--m-a-p--YuE2-3B").is_dir()


class TestCacheListing:
    def test_the_repositories_of_every_preset_are_used_on_every_platform(self):
        used = audio_repo_ids(audio_config())

        assert {"m-a-p/YuE2-3B", "m-a-p/YuE2-Vae", "bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v2-tokenizer", "Comfy-Org/stable-audio-3"} <= used

    def test_a_cached_audio_repository_is_not_offered_as_an_unused_download(self, env):
        _cache(env.hub, "m-a-p/YuE2-3B")
        _cache(env.hub, "org/unused")

        web_config = _web_config(env)
        web_config.app_config["video_model_presets"] = {"ltx": {"mlx_text_encoder": "owner/gemma"}}

        unused = workspace_api_module.find_other_downloads(web_config, cache_dir=env.hub)

        assert [repo.repo_id for repo in unused] == ["org/unused"]
