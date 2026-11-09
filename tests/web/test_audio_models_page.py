"""Tests for the audio rows of the Models page: the inventory payload, the download plan, the delete and the cache listing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import (
    QWEN_BASE_REPO,
    QWEN_CUSTOM_REPO,
    QWEN_DESIGN_REPO,
    YUE2_REPO,
    audio_config,
    audio_entries,
    no_audio_models,
    qwen3_tts_preset,
    sa3_preset,
    sa3_two_kind_preset,
    use_audio_models,
    yue2_preset,
)
from tests._factories import load_packaged_config, packaged_metadata_copy
from tests._licences import use_licences

from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.licences import load_licences
from zvisiongenerator.utils.model_files import disk_size_label
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.audio_inventory import audio_inventory_rows
from zvisiongenerator.web.model_delete import audio_delete_target, delete_model, used_repo_ids
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
    use_licences(monkeypatch, tmp_path, _LICENCES, models={**audio_entries(), **{model_id: packaged_metadata_copy()["models"][model_id] for model_id in ("matte", "enhancer-fast", "enhancer-smart")}})
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

        assert list(rows) == ["yue2", "qwen3_tts", "sa3"]
        assert rows["yue2"] == {
            "name": "yue2",
            "label": "YuE2 3B",
            "kinds": ["music"],
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
        assert [rows[name]["kinds"] for name in rows] == [["music"], ["speech"], ["sound"]]
        assert not any("kind" in row for row in rows.values())
        assert rows["sa3"]["downloaded"] is False and rows["sa3"]["download_needed"] is True
        assert rows["sa3"]["licences"] == []

    def test_a_model_of_two_kinds_is_one_row_that_lists_both(self, env, monkeypatch, tmp_path):
        use_audio_models(monkeypatch, tmp_path, sa3=sa3_two_kind_preset())
        rows = _rows(env, audio_config())

        assert list(rows) == ["sa3"]
        assert rows["sa3"]["kinds"] == ["sound", "music"]

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

    def test_metadata_without_audio_models_has_no_rows_so_the_panel_stays_hidden(self, env, monkeypatch, tmp_path):
        no_audio_models(monkeypatch, tmp_path)

        assert audio_inventory_rows(audio_config(), _BUDGET) == []

    def test_one_unreadable_speech_model_leaves_both_voice_downloads_unknown(self, env):
        def broken(target: DownloadTarget) -> bool:
            if target.repo_id.startswith("Qwen/"):
                raise OSError("cache unreadable")
            return True

        with pytest.warns(UserWarning, match="Could not determine status for audio model 'qwen3_tts'"):
            rows = _rows(env, audio_config(), target_present=broken)

        assert (rows["qwen3_tts"]["downloaded"], rows["qwen3_tts"]["voice_download"], rows["qwen3_tts"]["can_design"]) == (None, None, True)
        assert rows["yue2"]["downloaded"] is True and "can_design" not in rows["yue2"]

    def test_the_packaged_config_has_a_row_per_model(self, env):
        rows = audio_inventory_rows(load_packaged_config(), _BUDGET)

        assert [row["name"] for row in rows] == ["yue2", "qwen3_tts", "sa3"]

    def test_the_models_payload_lists_the_rows_after_the_video_models(self, env, monkeypatch):
        monkeypatch.setattr(audio_models_module, "target_present_offline", lambda _target: True)
        monkeypatch.setattr(workspace_api_module, "find_other_downloads", lambda *_a, **_k: ())

        models = workspace_api_module.build_models_response(_web_config(env), token_source=None, image_defaults_for=lambda *_a: {})

        assert [row["name"] for row in models["audio_models"]] == ["yue2", "qwen3_tts", "sa3"]
        assert list(models).index("audio_models") == list(models).index("video_models") + 1


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

    def test_is_refused_while_the_voicedesign_repository_downloads(self, client, env):
        _cache(env.hub, QWEN_CUSTOM_REPO.split("@")[0])
        flight, owner = web_server.download_flights.claim(DownloadTarget("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
        assert owner
        try:
            assert client.delete("/api/audio-models/qwen3_tts").status_code == 409
        finally:
            web_server.download_flights.finish(flight, "stopped")
        assert (env.hub / "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice").is_dir()


class TestCacheListing:
    def test_the_repositories_of_every_preset_are_used(self, env):
        used = used_repo_ids([], {}, platform_key=sys.platform)

        assert {"m-a-p/yue2-3b", "m-a-p/yue2-vae", "qwen/qwen3-tts-12hz-1.7b-customvoice", "comfy-org/stable-audio-3"} <= used

    def test_a_cached_audio_repository_is_not_offered_as_an_unused_download(self, env):
        _cache(env.hub, "m-a-p/YuE2-3B")
        _cache(env.hub, "org/unused")

        web_config = _web_config(env)

        unused = workspace_api_module.find_other_downloads(web_config, cache_dir=env.hub)

        assert [repo.repo_id for repo in unused] == ["org/unused"]


class TestQwenCloningFiles:
    """Qwen3-TTS keeps the cloning checkpoint apart: the row's download covers CustomVoice, ``voice_download`` covers Base, and delete and the cache cover both."""

    def _config(self):
        return audio_config(qwen3_tts=qwen3_tts_preset())

    def test_the_row_is_downloaded_with_customvoice_alone_and_reports_the_cloning_files_apart(self, env):
        row = _rows(env, self._config(), target_present=lambda target: target.repo_id == QWEN_CUSTOM_REPO.split("@")[0])["qwen3_tts"]

        assert (row["downloaded"], row["download_needed"]) == (True, False)
        assert row["voice_download"] == {"downloaded": False}

    def test_the_cloning_files_count_as_downloaded_when_present(self, env):
        row = _rows(env, self._config(), target_present=lambda _target: True)["qwen3_tts"]

        assert row["voice_download"] == {"downloaded": True}

    def test_a_speech_model_without_cloning_files_has_a_null_voice_download_and_other_kinds_have_no_key(self, env, monkeypatch, tmp_path):
        preset = qwen3_tts_preset()
        del preset["voice_files"]
        use_audio_models(monkeypatch, tmp_path, qwen3_tts=preset, yue2=yue2_preset(), sa3=sa3_preset())
        rows = _rows(env, audio_config(), target_present=lambda _target: True)

        assert rows["qwen3_tts"]["voice_download"] is None
        assert "voice_download" not in rows["yue2"] and "voice_download" not in rows["sa3"]

    def test_deleting_removes_every_checkpoint(self, env):
        for repo in (QWEN_CUSTOM_REPO, QWEN_BASE_REPO, QWEN_DESIGN_REPO):
            _cache(env.hub, repo.split("@")[0])
        preset = next(item for item in parse_audio_presets(self._config()) if item.id == "qwen3_tts")

        target = audio_delete_target(preset, platform_key=sys.platform)

        assert target.repo_ids == ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        delete_model(target)
        assert not any(env.hub.iterdir())

    def test_deleting_with_only_the_cloning_files_on_disk_still_finds_them(self, env):
        _cache(env.hub, QWEN_BASE_REPO.split("@")[0])
        preset = next(item for item in parse_audio_presets(self._config()) if item.id == "qwen3_tts")

        assert audio_delete_target(preset, platform_key=sys.platform).repo_ids == ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",)

    def test_the_disk_size_adds_every_checkpoint(self, env):
        for repo in (QWEN_CUSTOM_REPO, QWEN_BASE_REPO, QWEN_DESIGN_REPO):
            _cache(env.hub, repo.split("@")[0])

        row = _rows(env, self._config(), target_present=lambda _target: True)["qwen3_tts"]

        assert row["disk_size_label"] == disk_size_label(3)

    def test_the_cache_listing_uses_every_repository(self, env):
        assert {"qwen/qwen3-tts-12hz-1.7b-customvoice", "qwen/qwen3-tts-12hz-1.7b-base", "qwen/qwen3-tts-12hz-1.7b-voicedesign"} <= used_repo_ids([], {}, platform_key=sys.platform)

    def test_deleting_with_only_the_design_files_on_disk_still_finds_them(self, env):
        _cache(env.hub, QWEN_DESIGN_REPO.split("@")[0])
        preset = next(item for item in parse_audio_presets(self._config()) if item.id == "qwen3_tts")

        assert audio_delete_target(preset, platform_key=sys.platform).repo_ids == ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",)

    def test_the_design_files_do_not_count_toward_the_row_download(self, env):
        row = _rows(env, self._config(), target_present=lambda target: target.repo_id != "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")["qwen3_tts"]

        assert (row["downloaded"], row["voice_download"], row["can_design"]) == (True, {"downloaded": True}, True)

    def test_a_speech_model_without_design_files_cannot_design_and_other_kinds_have_no_key(self, env, monkeypatch, tmp_path):
        preset = qwen3_tts_preset()
        del preset["design_files"]
        use_audio_models(monkeypatch, tmp_path, qwen3_tts=preset, yue2=yue2_preset(), sa3=sa3_preset())
        rows = _rows(env, audio_config(), target_present=lambda _target: True)

        assert rows["qwen3_tts"]["can_design"] is False
        assert "can_design" not in rows["yue2"] and "can_design" not in rows["sa3"]

    def test_the_models_page_download_plan_stays_on_the_main_files(self, env):
        part = row_plan("audio", "qwen3_tts", _web_config(env, self._config()))

        assert [target.repo_id for target in part.targets] == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]
