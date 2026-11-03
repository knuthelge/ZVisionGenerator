"""Tests for audio in the gallery: listing, the asset JSON and its Reuse settings, the filter, the index and ``/media``."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import audio_config, yue2_preset
from tests._media import REAL
from tests.audio._helpers import sine
from tests.web._helpers import make_web_config

from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.audio_save import save_flac
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import gallery
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.gallery import (
    AUDIO_EXTENSIONS,
    GALLERY_MEDIA_EXTENSIONS,
    build_gallery_page_json,
    filter_and_sort_assets,
    gallery_asset_for_output_path,
    gallery_asset_to_json,
    list_gallery_assets,
    output_media_url,
)
from zvisiongenerator.web.gallery_index import GALLERY_INDEX_VERSION

_SONG = {
    "schema": "zvisiongenerator.config.v1",
    "workflow": "music",
    "kind": "music",
    "model": "yue2",
    "prompt": "upbeat synth-pop",
    "lyrics": "[verse]\nHello there",
    "seed": 7,
    "generation": {"time": 12.5, "duration": 31.2},
}
_ANY = {"schema": "zvisiongenerator.config.v1", "workflow": "speech", "kind": "speech", "model": "higgs3", "prompt": "Welcome back.", "seed": 4, "generation": {"parts": 1}}
_CLONED = {**_ANY, "prompt": "More words.", "voice_path": "~/voices/me.wav", "voice_text": "Hello, this is me."}
_SOUND = {"schema": "zvisiongenerator.config.v1", "workflow": "sound", "kind": "sound", "model": "sa3", "prompt": "rain on a roof", "seconds": 12.5, "seed": 3, "generation": {"cut": True}}


@pytest.fixture(autouse=True)
def _cuda(monkeypatch):
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: "cuda")


def _web_config(**presets) -> SimpleNamespace:
    config = make_web_config()
    config.app_config.update(audio_config(**presets) if presets else audio_config())
    return config


def _take(root: Path, name: str, config: dict | None, *, seconds: float = 1.0, rate: int = 24000) -> Path:
    """Write a real FLAC under *root*, with *config* embedded when given."""
    path = root / name
    if config is None:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=duration={seconds}", "-c:a", "flac", str(path)], check=True)
    else:
        save_flac(sine(seconds, rate=rate), rate, path, config)
    return path


def _asset(root: Path, name: str):
    return next(asset for asset in list_gallery_assets(str(root)) if asset.name == name)


def _json(root: Path, name: str, web_config=None) -> dict:
    return gallery_asset_to_json(_asset(root, name), web_config or _web_config())


def _params(payload: dict) -> dict[str, str]:
    query = urlsplit(payload["reuse_workspace_url"].removeprefix("#")).query
    return {key: values[0] for key, values in parse_qs(query, keep_blank_values=True).items()}


class TestListing:
    def test_audio_files_of_every_extension_list_as_audio(self, tmp_path):
        names = [f"take{suffix}" for suffix in sorted(AUDIO_EXTENSIONS)]
        for name in names:
            (tmp_path / name).write_bytes(b"not decodable")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "take.aiff").write_bytes(b"x")

        assets = list_gallery_assets(str(tmp_path))

        assert sorted(asset.name for asset in assets) == sorted(names)
        assert {asset.kind for asset in assets} == {"audio"}
        assert AUDIO_EXTENSIONS == {".flac", ".wav", ".mp3", ".m4a", ".ogg"} and AUDIO_EXTENSIONS <= GALLERY_MEDIA_EXTENSIONS

    def test_a_file_that_cannot_be_decoded_still_lists_with_no_facts(self, tmp_path):
        (tmp_path / "broken.flac").write_bytes(b"not audio")

        payload = _json(tmp_path, "broken.flac")

        assert payload["media_type"] == "audio" and payload["duration"] is None and payload["peaks"] == [] and payload["has_reusable_config"] is False
        assert payload["file_bytes"] == len(b"not audio")

    def test_the_media_url_of_an_audio_file_is_served(self, tmp_path):
        (tmp_path / "sub").mkdir()
        take = tmp_path / "sub" / "a take.flac"
        take.write_bytes(b"x")

        assert output_media_url(tmp_path, take) == "/media/sub/a%20take.flac"

    def test_the_staging_folder_for_uploads_is_not_listed(self, tmp_path):
        (tmp_path / ".web_uploads" / "abc").mkdir(parents=True)
        (tmp_path / ".web_uploads" / "abc" / "voice.flac").write_bytes(b"x")

        assert list_gallery_assets(str(tmp_path)) == []


@REAL
class TestAssetJson:
    def test_the_model_is_named_by_its_preset_label_while_model_stays_the_id(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG)
        _take(tmp_path, "talk.flac", _ANY)

        song, talk = _json(tmp_path, "song.flac"), _json(tmp_path, "talk.flac")

        assert (song["model"], song["model_label"]) == ("yue2", "YuE2 3B")
        assert (talk["model"], talk["model_label"]) == ("higgs3", "Higgs TTS 3 4B")

    def test_a_song_carries_its_facts_and_the_settings_to_reuse(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG, seconds=2.0, rate=48000)

        payload = _json(tmp_path, "song.flac")

        assert (payload["media_type"], payload["kind"], payload["workflow"], payload["filename"]) == ("audio", "music", "music", "song.flac")
        assert payload["duration"] == pytest.approx(2.0, abs=0.05) and (payload["channels"], payload["sample_rate"]) == (1, 48000)
        assert len(payload["peaks"]) == 100 and all(0 <= peak <= 255 for peak in payload["peaks"])
        assert (payload["has_reference_voice"], payload["lyrics"], payload["cut"], payload["parts"]) == (False, "[verse]\nHello there", False, None)
        assert (payload["prompt"], payload["model"], payload["seed"], payload["has_reusable_config"]) == ("upbeat synth-pop", "yue2", 7, True)
        assert payload["url"] == "/media/song.flac" and payload["file_path"] == str(tmp_path / "song.flac")
        assert _params(payload) == {"workflow": "music", "model": "yue2", "prompt": "upbeat synth-pop", "lyrics": "[verse]\nHello there", "seed": "7"}
        assert payload["reuse_state"] == {
            "requested_workflow": "music",
            "resolved_workflow": "music",
            "workflow_available": True,
            "requested_model": "yue2",
            "resolved_model": "yue2",
            "model_available": True,
            "fallback_reasons": [],
        }

    def test_there_are_no_upscale_or_extend_blocks_and_no_size(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG)

        payload = _json(tmp_path, "song.flac")

        assert payload["upscale"] is None and payload["extend"] is None and payload["enhance"] is None
        assert (payload["width"], payload["height"], payload["frame_count"], payload["image_path"]) == (None, None, None, None)

    def test_a_take_in_the_models_own_voice_has_no_reference_voice(self, tmp_path):
        _take(tmp_path, "any.flac", _ANY)

        payload = _json(tmp_path, "any.flac")

        assert (payload["kind"], payload["has_reference_voice"]) == ("speech", False)
        assert _params(payload) == {"workflow": "speech", "model": "higgs3", "prompt": "Welcome back.", "seed": "4"}

    def test_a_cloned_take_reuses_its_voice_with_the_home_folder_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        _take(tmp_path, "cloned.flac", _CLONED)

        payload = _json(tmp_path, "cloned.flac")

        assert payload["has_reference_voice"] is True
        params = _params(payload)
        assert (params["voice_path"], params["voice_text"]) == (str(tmp_path / "home" / "voices" / "me.wav"), "Hello, this is me.")

    def test_the_length_asked_for_and_the_cut_badge_come_from_the_settings(self, tmp_path):
        _take(tmp_path, "rain.flac", _SOUND)

        payload = _json(tmp_path, "rain.flac")

        assert _params(payload)["seconds"] == "12.5" and payload["cut"] is True

    def test_consent_is_never_part_of_a_reuse_even_when_a_file_carries_it(self, tmp_path):
        _take(tmp_path, "cloned.flac", {**_CLONED, "voice_consent": True, "consent": True})

        payload = _json(tmp_path, "cloned.flac")

        assert not [key for key in _params(payload) if "consent" in key] and "consent" not in payload["reuse_workspace_url"]
        assert not [key for key in payload if "consent" in key]

    def test_a_file_without_this_apps_settings_cannot_be_reused(self, tmp_path):
        _take(tmp_path, "plain.flac", None)

        payload = _json(tmp_path, "plain.flac")

        assert payload["has_reusable_config"] is False and payload["kind"] is None and payload["has_reference_voice"] is False
        assert payload["duration"] == pytest.approx(1.0, abs=0.05) and len(payload["peaks"]) == 100
        assert _params(payload) == {"workflow": "music"}
        assert payload["prompt"] == "plain"  # the file name stands in as display text

    def test_a_model_that_no_longer_runs_is_not_carried_and_says_why(self, tmp_path):
        _take(tmp_path, "song.flac", {**_SONG, "model": "retired"})

        payload = _json(tmp_path, "song.flac")

        assert "model" not in _params(payload)
        assert payload["reuse_state"]["resolved_model"] is None and payload["reuse_state"]["fallback_reasons"] == ["model_not_configured"]

    def test_a_kind_with_no_runnable_model_is_an_unavailable_workflow(self, tmp_path):
        _take(tmp_path, "rain.flac", _SOUND)

        payload = _json(tmp_path, "rain.flac", _web_config(yue2=yue2_preset()))

        assert payload["reuse_state"]["workflow_available"] is False and "no_audio_model" in payload["reuse_state"]["fallback_reasons"]

    def test_the_machine_without_audio_models_still_lists_and_plays_a_take(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG)
        web_config = make_web_config()  # no audio_model_presets at all

        payload = _json(tmp_path, "song.flac", web_config)

        assert payload["media_type"] == "audio" and payload["reuse_state"]["workflow_available"] is False

    def test_a_page_of_the_gallery_holds_audio_next_to_images(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG)
        (tmp_path / "pic.png").write_bytes(b"\x89PNG")
        web_config = _web_config()
        web_config.gallery_page_size = 12

        page = build_gallery_page_json(list_gallery_assets(str(tmp_path)), web_config, page=1, page_size=12)

        assert sorted(asset["media_type"] for asset in page["assets"]) == ["audio", "image"]

    def test_a_take_just_made_is_built_from_its_path_for_the_job_event(self, tmp_path):
        path = _take(tmp_path, "song.flac", _SONG)

        asset = gallery_asset_for_output_path(str(tmp_path), str(path))

        assert asset is not None and gallery_asset_to_json(asset, _web_config())["kind"] == "music"


class TestFilter:
    @pytest.fixture
    def assets(self, tmp_path):
        for name in ("a.png", "b.mp4", "c.flac", "d.wav"):
            (tmp_path / name).write_bytes(b"x")
        return list_gallery_assets(str(tmp_path))

    @pytest.mark.parametrize(
        ("media_filter", "names"),
        [
            ("audio", {"c.flac", "d.wav"}),
            ("image", {"a.png"}),
            ("video", {"b.mp4"}),
            ("all", {"a.png", "b.mp4", "c.flac", "d.wav"}),
        ],
    )
    def test_each_media_filter_lists_only_its_kind(self, assets, media_filter, names):
        assert {asset.name for asset in filter_and_sort_assets(assets, media_filter=media_filter, sort_order="newest")} == names

    def test_the_gallery_route_takes_the_audio_filter(self, assets, tmp_path, monkeypatch):
        web_config = _web_config()
        web_config.output_dir = str(tmp_path)
        web_config.gallery_page_size = 12
        monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)

        with TestClient(web_server.app) as client:
            body = client.get("/api/gallery", params={"filter": "audio"}).json()

        assert {asset["filename"] for asset in body["assets"]} == {"c.flac", "d.wav"} and body["total_count"] == 2


@REAL
class TestIndex:
    def test_an_audio_files_facts_are_stored_and_not_read_again(self, tmp_path, monkeypatch):
        path = _take(tmp_path, "song.flac", _SONG)
        os.utime(path, (time.time() - 60, time.time() - 60))
        reads: list[str] = []
        real = gallery._read_audio_facts
        monkeypatch.setattr(gallery, "_read_audio_facts", lambda p: reads.append(p.name) or real(p))

        first = _json(tmp_path, "song.flac")
        second = _json(tmp_path, "song.flac")

        assert reads == ["song.flac"]
        assert second == first and len(second["peaks"]) == 100
        with closing(sqlite3.connect(app_db.app_db_path())) as conn:
            rows = conn.execute(f"SELECT rel_path FROM gallery_assets_v{GALLERY_INDEX_VERSION}").fetchall()
        assert rows == [("song.flac",)]

    def test_a_file_whose_facts_could_not_be_read_is_read_again_next_time(self, tmp_path, monkeypatch):
        path = tmp_path / "bad.flac"
        path.write_bytes(b"not audio")
        os.utime(path, (time.time() - 60, time.time() - 60))
        reads: list[str] = []
        real = gallery._read_audio_facts
        monkeypatch.setattr(gallery, "_read_audio_facts", lambda p: reads.append(p.name) or real(p))

        list_gallery_assets(str(tmp_path))
        list_gallery_assets(str(tmp_path))

        assert reads == ["bad.flac", "bad.flac"]

    def test_peaks_that_cannot_be_decoded_leave_the_rest_and_mark_the_read_incomplete(self, tmp_path, monkeypatch):
        path = save_flac(sine(0.2, rate=24000), 24000, tmp_path / "t.flac", {"kind": "sound"})
        monkeypatch.setattr(gallery, "read_waveform_peaks", lambda _path: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")))

        read = gallery._read_audio_facts(path)

        assert not read.complete and "peaks" not in read.facts["audio"] and read.facts["audio"]["sample_rate"] == 24000
        assert read.facts["config"] == {"kind": "sound"}


class TestMedia:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        root = tmp_path / "outputs"
        root.mkdir()
        (root / "sub").mkdir()
        (root / "take.flac").write_bytes(b"fLaC-audio-bytes")
        (root / "sub" / "voice.wav").write_bytes(b"RIFF")
        (root / "notes.txt").write_text("secret")
        (root / ".web_uploads").mkdir()
        (root / ".web_uploads" / "v.flac").write_bytes(b"x")
        (tmp_path / "outside.flac").write_bytes(b"outside")
        monkeypatch.setattr(web_server, "_media_output_root", lambda: root)
        with TestClient(web_server.app) as client:
            yield client

    def test_an_audio_file_under_the_output_root_is_served(self, client):
        response = client.get("/media/take.flac")

        assert response.status_code == 200 and response.content == b"fLaC-audio-bytes" and response.headers["content-type"].startswith("audio/")
        assert client.get("/media/sub/voice.wav").status_code == 200

    def test_a_range_of_an_audio_file_is_served_so_the_player_can_seek(self, client):
        response = client.get("/media/take.flac", headers={"Range": "bytes=5-9"})

        assert response.status_code == 206 and response.content == b"audio"

    def test_other_suffixes_and_paths_outside_the_root_are_still_refused(self, client):
        assert client.get("/media/notes.txt").status_code == 404
        assert client.get("/media/../outside.flac").status_code == 404
        assert client.get("/media/%2e%2e/outside.flac").status_code == 404
        assert client.get("/media/.web_uploads/v.flac").status_code == 404
        assert client.get("/media/missing.flac").status_code == 404
