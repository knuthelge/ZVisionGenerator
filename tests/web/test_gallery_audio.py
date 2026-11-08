"""Tests for audio in the gallery: listing, the asset JSON and its Reuse settings, the filter, the index and ``/media``."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import audio_config, no_audio_models, qwen3_tts_preset, sa3_preset, sa3_two_kind_preset, use_audio_models, yue2_preset
from tests._fake_audio import install_fake_audio, write_fake_audio
from tests._paths import FIXTURES
from tests.audio._helpers import clip, sine
from tests.web._helpers import make_web_config

from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.provenance import build_audio_config_payload
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
    move_gallery_assets,
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
_ANY = {"schema": "zvisiongenerator.config.v1", "workflow": "speech", "kind": "speech", "model": "qwen3_tts", "prompt": "Welcome back.", "seed": 4, "generation": {"parts": 1}}
_VOICED = {**_ANY, "speaker": "vivian"}
_CLONED = {**_ANY, "prompt": "More words.", "voice_path": "~/voices/me.wav", "voice_text": "Hello, this is me."}
SETTINGS_CASES = json.loads((FIXTURES / "audio_settings_cases.json").read_text(encoding="utf-8"))["cases"]
_SOUND = {
    "schema": "zvisiongenerator.config.v1",
    "workflow": "sound",
    "kind": "sound",
    "model": "sa3",
    "prompt": "rain on a roof",
    "seconds": 12.5,
    "requested_seconds": 12.5,
    "seed": 3,
    "generation": {"cut": True},
}


@pytest.fixture(autouse=True)
def _cuda(monkeypatch):
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: "cuda")


@pytest.fixture(autouse=True)
def _fake_audio_tools(monkeypatch):
    """Read the files this module writes through the fakes in ``tests/_fake_audio.py``; ffmpeg and ffprobe never start."""
    install_fake_audio(monkeypatch)


@pytest.fixture(autouse=True)
def _fixture_audio_models(monkeypatch, tmp_path):
    """Serve the three fixture audio models (``sa3`` making sound only); a test that needs other models serves them itself."""
    use_audio_models(monkeypatch, tmp_path)


def _web_config() -> SimpleNamespace:
    config = make_web_config()
    config.app_config.update(audio_config())
    return config


def _take(root: Path, name: str, config: dict | None, *, seconds: float = 1.0, rate: int = 24000) -> Path:
    """Write a fake take under *root*, with *config* embedded when given (a file without this app's settings when ``None``)."""
    path = root / name
    write_fake_audio(sine(seconds, rate=rate), rate, path, config)
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

        assert payload["media_type"] == "audio" and payload["duration"] is None and payload["peaks"] is None and payload["has_reusable_config"] is False
        assert payload["file_bytes"] == len(b"not audio")

    def test_the_staging_folder_for_uploads_is_not_listed(self, tmp_path):
        (tmp_path / ".web_uploads" / "abc").mkdir(parents=True)
        (tmp_path / ".web_uploads" / "abc" / "voice.flac").write_bytes(b"x")

        assert list_gallery_assets(str(tmp_path)) == []


class TestReuseOfSettings:
    """The Reuse leg of the round trip: the request, the payload the runner saves, and the params Reuse restores (the Vitest half carries them on to the request)."""

    @pytest.fixture
    def config(self, monkeypatch, tmp_path):
        use_audio_models(monkeypatch, tmp_path, yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3=sa3_two_kind_preset())
        return _web_config()

    @pytest.mark.parametrize("case", SETTINGS_CASES, ids=[case["name"] for case in SETTINGS_CASES])
    def test_a_take_that_sets_every_value_saves_and_reuses_them(self, tmp_path, config, case):
        item = case["request"]
        request = AudioGenerationRequest(
            **{key: value for key, value in item.items() if key not in ("voice", "delivery")},
            voice=VoiceRef(**item["voice"]) if item["voice"] else None,
            delivery=SpeechDelivery(**item["delivery"]) if item["delivery"] else None,
            seed=case["seed"],
        )

        payload = build_audio_config_payload(request, clip(0.1), parts=None, generation_time=1.0)
        _take(tmp_path, "take.flac", payload)

        assert {key: value for key, value in payload.items() if key not in ("schema", "generation")} == case["payload"]
        assert _params(_json(tmp_path, "take.flac", config)) == case["reuse"]

    def test_an_enhanced_take_whose_own_text_is_empty_reuses_that_empty_text(self, tmp_path, config):
        record = {"settings": {}, "model": "fast", "tier": "fast", "format": "text", "task": "style"}
        _take(tmp_path, "fill.flac", {**_SONG, "prompt": "dreamy lo-fi", "prompt_input": "", "enhance": record})

        params = _params(_json(tmp_path, "fill.flac", config))

        assert (params["prompt"], params["enhanced_prompt"]) == ("", "dreamy lo-fi")

    def test_an_enhanced_take_reuses_how_its_text_was_written(self, tmp_path, config):
        record = {"settings": {"genre": "folk"}, "model": "org/fast", "tier": "fast", "format": "text", "task": "style"}
        _take(tmp_path, "fill.flac", {**_SONG, "prompt": "Folk, piano", "prompt_input": "sad folk", "enhance": record})
        _take(tmp_path, "typed.flac", {**_SONG, "prompt": "Folk, piano"})

        params, plain = _params(_json(tmp_path, "fill.flac", config)), _params(_json(tmp_path, "typed.flac", config))

        assert json.loads(params["enhance"]) == record
        assert _asset(tmp_path, "fill.flac").audio["enhance"] == record
        assert "enhance" not in plain and _asset(tmp_path, "typed.flac").audio["enhance"] is None

    def test_an_own_text_without_a_record_is_not_an_enhanced_take(self, tmp_path, config):
        _take(tmp_path, "typed.flac", {**_SONG, "prompt": "dreamy lo-fi", "prompt_input": "lo-fi"})

        params = _params(_json(tmp_path, "typed.flac", config))

        assert params["prompt"] == "dreamy lo-fi" and "enhanced_prompt" not in params

    def test_a_take_without_any_new_key_reuses_as_auto_and_the_defaults(self, tmp_path, config):
        _take(tmp_path, "old-song.flac", _SONG)
        _take(tmp_path, "old-voice.flac", _VOICED)
        _take(tmp_path, "old-rain.flac", _SOUND)

        songs, voiced, rain = (_params(_json(tmp_path, name, config)) for name in ("old-song.flac", "old-voice.flac", "old-rain.flac"))

        new = {"tempo", "key", "emotion", "manner", "pace", "pitch", "expressiveness", "sound_type", "enhanced_prompt"}
        assert not new & (songs.keys() | voiced.keys() | rain.keys())
        assert rain["seconds"] == "12.5"

    def test_the_facts_of_a_take_hold_its_new_settings(self, tmp_path, config):
        _take(tmp_path, "song.flac", {**_SONG, "tempo": 92, "key": "F#m"})

        asset = _asset(tmp_path, "song.flac")

        assert (asset.audio["tempo"], asset.audio["key"], asset.audio["enhanced"], asset.audio["prompt_input"]) == (92, "F#m", False, None)


class TestAssetJson:
    def test_the_model_is_named_by_its_preset_label_while_model_stays_the_id(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG)
        _take(tmp_path, "talk.flac", _ANY)

        song, talk = _json(tmp_path, "song.flac"), _json(tmp_path, "talk.flac")

        assert (song["model"], song["model_label"]) == ("yue2", "YuE2 3B")
        assert (talk["model"], talk["model_label"]) == ("qwen3_tts", "Qwen3-TTS 1.7B")

    def test_a_song_carries_its_facts_and_the_settings_to_reuse(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG, seconds=2.0, rate=48000)

        payload = _json(tmp_path, "song.flac")

        assert (payload["media_type"], payload["kind"], payload["workflow"], payload["filename"]) == ("audio", "music", "music", "song.flac")
        assert payload["duration"] == pytest.approx(2.0, abs=0.05) and (payload["channels"], payload["sample_rate"]) == (1, 48000)
        assert payload["peaks"] is None  # the waveform is computed on the first request for it, not by the listing
        assert (payload["has_reference_voice"], payload["lyrics"], payload["cut"]) == (False, "[verse]\nHello there", False)
        assert (payload["prompt"], payload["model"], payload["seed"], payload["has_reusable_config"]) == ("upbeat synth-pop", "yue2", 7, True)
        assert payload["url"] == payload["thumbnail_url"] == "/media/song.flac"
        assert payload["file_path"] == str(tmp_path / "song.flac")
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

        assert not {"upscale", "extend", "width", "height", "frame_count", "image_path"} & payload.keys()

    def test_a_take_carries_the_tempo_key_delivery_and_sound_type_it_recorded(self, tmp_path):
        _take(tmp_path, "song.flac", {**_SONG, "tempo": 92, "key": "Am"})
        _take(tmp_path, "talk.flac", {**_VOICED, "emotion": "sad", "manner": "whisper", "pace": "slow", "pitch": "low", "expressiveness": "high"})
        _take(tmp_path, "rain.flac", {**_SOUND, "sound_type": "ambience"})

        song, talk, rain = (_json(tmp_path, name) for name in ("song.flac", "talk.flac", "rain.flac"))

        assert (song["tempo"], song["key"]) == (92, "Am")
        assert {field: talk[field] for field in ("emotion", "manner", "pace", "pitch", "expressiveness")} == {
            "emotion": "sad",
            "manner": "whisper",
            "pace": "slow",
            "pitch": "low",
            "expressiveness": "high",
        }
        assert rain["sound_type"] == "ambience"

    def test_what_a_take_did_not_record_is_null(self, tmp_path):
        _take(tmp_path, "song.flac", _SONG)

        payload = _json(tmp_path, "song.flac")

        assert all(payload[field] is None for field in ("tempo", "key", "emotion", "manner", "pace", "pitch", "expressiveness", "sound_type", "enhance"))

    def test_an_enhanced_take_carries_how_its_text_was_written_and_the_users_own_text(self, tmp_path):
        record = {"settings": {}, "model": "org/fast", "tier": "fast", "format": "text", "task": "style"}
        _take(tmp_path, "styled.flac", {**_SONG, "prompt": "Folk, piano", "prompt_input": "sad folk", "enhance": record})

        styled = _json(tmp_path, "styled.flac")["enhance"]

        assert (styled["tier"], styled["prompt_input"], styled["model"], styled["task"]) == ("fast", "sad folk", "org/fast", "style")

    def test_a_take_in_the_models_own_voice_has_no_reference_voice(self, tmp_path):
        _take(tmp_path, "any.flac", _ANY)

        payload = _json(tmp_path, "any.flac")

        assert (payload["kind"], payload["has_reference_voice"]) == ("speech", False)
        assert _params(payload) == {"workflow": "speech", "model": "qwen3_tts", "prompt": "Welcome back.", "seed": "4"}

    def test_a_take_in_a_built_in_voice_reuses_its_speaker(self, tmp_path):
        _take(tmp_path, "voiced.flac", _VOICED)

        payload = _json(tmp_path, "voiced.flac")

        assert payload["has_reference_voice"] is False
        assert _params(payload) == {"workflow": "speech", "model": "qwen3_tts", "prompt": "Welcome back.", "speaker": "vivian", "seed": "4"}

    def test_a_cloned_take_reuses_its_recording_and_no_speaker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        _take(tmp_path, "cloned.flac", {**_CLONED, "speaker": "vivian"})

        assert "speaker" not in _params(_json(tmp_path, "cloned.flac"))

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

    def test_the_notes_of_a_take_come_from_the_settings(self, tmp_path):
        _take(tmp_path, "noted.flac", {**_SOUND, "generation": {"notes": ["Couldn't plan.", "", 3]}})
        _take(tmp_path, "plain.flac", _SOUND)

        assert _json(tmp_path, "noted.flac")["notes"] == ["Couldn't plan."]
        assert _json(tmp_path, "plain.flac")["notes"] == []

    def test_a_length_left_on_auto_is_recorded_but_not_reused(self, tmp_path):
        _take(tmp_path, "auto.flac", {**_SOUND, "seconds": 12.0, "requested_seconds": None})

        asset = _asset(tmp_path, "auto.flac")

        assert (asset.audio["seconds"], asset.audio["requested_seconds"]) == (12.0, None)
        assert "seconds" not in _params(_json(tmp_path, "auto.flac"))

    def test_consent_is_never_part_of_a_reuse_even_when_a_file_carries_it(self, tmp_path):
        _take(tmp_path, "cloned.flac", {**_CLONED, "voice_consent": True, "consent": True})

        payload = _json(tmp_path, "cloned.flac")

        assert not [key for key in _params(payload) if "consent" in key] and "consent" not in payload["reuse_workspace_url"]
        assert not [key for key in payload if "consent" in key]

    def test_a_file_without_this_apps_settings_cannot_be_reused(self, tmp_path):
        _take(tmp_path, "plain.flac", None)

        payload = _json(tmp_path, "plain.flac")

        assert payload["has_reusable_config"] is False and payload["kind"] is None and payload["has_reference_voice"] is False
        assert payload["duration"] == pytest.approx(1.0, abs=0.05) and payload["peaks"] is None
        assert _params(payload) == {"workflow": "music"}
        assert payload["prompt"] == "plain"  # the file name stands in as display text

    def test_a_model_that_no_longer_runs_is_not_carried_and_says_why(self, tmp_path):
        _take(tmp_path, "song.flac", {**_SONG, "model": "retired"})

        payload = _json(tmp_path, "song.flac")

        assert "model" not in _params(payload)
        assert payload["reuse_state"]["resolved_model"] is None and payload["reuse_state"]["fallback_reasons"] == ["model_not_configured"]

    def test_a_kind_with_no_runnable_model_is_an_unavailable_workflow(self, tmp_path, monkeypatch):
        _take(tmp_path, "rain.flac", _SOUND)
        use_audio_models(monkeypatch, tmp_path, yue2=yue2_preset())

        payload = _json(tmp_path, "rain.flac", _web_config())

        assert payload["reuse_state"]["workflow_available"] is False and "no_audio_model" in payload["reuse_state"]["fallback_reasons"]

    def test_a_kind_is_available_when_any_kind_of_a_runnable_model_is_it(self, tmp_path, monkeypatch):
        _take(tmp_path, "song.flac", _SONG)
        _take(tmp_path, "rain.flac", _SOUND)
        use_audio_models(monkeypatch, tmp_path, sa3=sa3_two_kind_preset())
        web_config = _web_config()

        song, rain = _json(tmp_path, "song.flac", web_config), _json(tmp_path, "rain.flac", web_config)

        assert (song["reuse_state"]["workflow_available"], rain["reuse_state"]["workflow_available"]) == (True, True)
        use_audio_models(monkeypatch, tmp_path, sa3=sa3_preset())
        assert _json(tmp_path, "song.flac", _web_config())["reuse_state"]["workflow_available"] is False

    def test_a_music_take_of_a_model_of_two_kinds_reuses_that_model_and_its_length(self, tmp_path, monkeypatch):
        music = {**_SONG, "model": "sa3", "prompt": "calm piano", "lyrics": "", "seconds": 45.0, "requested_seconds": 45.0}
        _take(tmp_path, "piano.flac", music)

        use_audio_models(monkeypatch, tmp_path, sa3=sa3_two_kind_preset())
        payload = _json(tmp_path, "piano.flac", _web_config())

        state = payload["reuse_state"]
        assert (state["resolved_workflow"], state["workflow_available"], state["model_available"], state["resolved_model"]) == ("music", True, True, "sa3")
        assert state["fallback_reasons"] == []
        assert _params(payload) == {"workflow": "music", "model": "sa3", "prompt": "calm piano", "seconds": "45", "seed": "7"}

    def test_the_machine_without_audio_models_still_lists_and_plays_a_take(self, tmp_path, monkeypatch):
        _take(tmp_path, "song.flac", _SONG)
        no_audio_models(monkeypatch, tmp_path)
        web_config = _web_config()

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

    def test_a_take_in_a_project_names_it(self, tmp_path):
        (tmp_path / "Reel").mkdir()
        _take(tmp_path, "Reel/song.flac", _SONG)
        _take(tmp_path, "top.flac", _SONG)

        assert _json(tmp_path, "song.flac", _web_config())["project"] == "Reel"
        assert _json(tmp_path, "top.flac", _web_config())["project"] is None


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
        assert second == first and second["peaks"] is None
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

    def test_reading_the_facts_of_an_audio_file_decodes_nothing(self, tmp_path, monkeypatch):
        path = write_fake_audio(sine(0.2, rate=24000), 24000, tmp_path / "t.flac", {"kind": "sound"})
        monkeypatch.setattr(gallery, "read_waveform_peaks", lambda _path: pytest.fail("the scan decoded the file"))

        read = gallery._read_audio_facts(path)

        assert read.complete and "peaks" not in read.facts["audio"] and read.facts["audio"]["sample_rate"] == 24000
        assert read.facts["config"] == {"kind": "sound"}


class TestPeaks:
    @pytest.fixture
    def decodes(self, monkeypatch):
        """Count the files the gallery decodes for their waveform."""
        names: list[str] = []
        real = gallery.read_waveform_peaks
        monkeypatch.setattr(gallery, "read_waveform_peaks", lambda path: names.append(Path(path).name) or real(path))
        return names

    @staticmethod
    def _settled(tmp_path, name="song.flac", config=_SONG, **kwargs):
        path = _take(tmp_path, name, config, **kwargs)
        os.utime(path, (time.time() - 60, time.time() - 60))
        return path

    def test_a_scan_lists_audio_files_without_decoding_them(self, tmp_path, decodes):
        self._settled(tmp_path)
        self._settled(tmp_path, "talk.flac", _ANY)

        assets = list_gallery_assets(str(tmp_path))
        payloads = [gallery_asset_to_json(asset, _web_config()) for asset in assets]

        assert decodes == [] and len(payloads) == 2
        assert all(payload["peaks"] is None and payload["duration"] is not None for payload in payloads)

    def test_the_first_request_decodes_and_stores_the_peaks_and_the_second_does_not(self, tmp_path, decodes):
        self._settled(tmp_path, seconds=2.0)
        list_gallery_assets(str(tmp_path))

        first = gallery.audio_peaks(str(tmp_path), "song.flac")
        second = gallery.audio_peaks(str(tmp_path), "song.flac")

        assert first is not None and len(first) == 100 and all(0 <= peak <= 255 for peak in first)
        assert second == first and decodes == ["song.flac"]

    def test_a_listing_after_the_request_carries_the_stored_peaks_and_keeps_the_other_facts(self, tmp_path, decodes):
        self._settled(tmp_path, seconds=2.0, rate=48000)
        list_gallery_assets(str(tmp_path))
        peaks = gallery.audio_peaks(str(tmp_path), "song.flac")

        payload = _json(tmp_path, "song.flac")

        assert payload["peaks"] == peaks and decodes == ["song.flac"]
        assert payload["duration"] == pytest.approx(2.0, abs=0.05) and (payload["channels"], payload["sample_rate"]) == (1, 48000)

    def test_a_changed_file_is_decoded_again(self, tmp_path, decodes):
        path = self._settled(tmp_path, seconds=1.0)
        list_gallery_assets(str(tmp_path))
        gallery.audio_peaks(str(tmp_path), "song.flac")
        write_fake_audio(sine(3.0, rate=24000, amplitude=0.9), 24000, path, _SONG)
        os.utime(path, (time.time() - 30, time.time() - 30))
        list_gallery_assets(str(tmp_path))

        gallery.audio_peaks(str(tmp_path), "song.flac")

        assert decodes == ["song.flac", "song.flac"]

    def test_a_file_the_index_has_not_stored_is_decoded_each_time_until_it_is(self, tmp_path, decodes):
        _take(tmp_path, "fresh.flac", _SONG)  # newer than the settle time: the scan does not store it

        gallery.audio_peaks(str(tmp_path), "fresh.flac")
        gallery.audio_peaks(str(tmp_path), "fresh.flac")

        assert decodes == ["fresh.flac", "fresh.flac"]

    def test_a_moved_take_keeps_its_stored_peaks_without_decoding_again(self, tmp_path, decodes):
        self._settled(tmp_path, seconds=2.0, rate=48000)
        list_gallery_assets(str(tmp_path))
        peaks = gallery.audio_peaks(str(tmp_path), "song.flac")
        move_gallery_assets(str(tmp_path), ["song.flac"], "Songs")
        moved = _json(tmp_path, "song.flac")

        assert moved["id"] == "Songs/song.flac"
        assert gallery.audio_peaks(str(tmp_path), "Songs/song.flac") == peaks
        assert moved["peaks"] == peaks
        assert moved["duration"] == pytest.approx(2.0, abs=0.05) and (moved["channels"], moved["sample_rate"]) == (1, 48000)
        assert decodes == ["song.flac"]

    def test_a_file_that_cannot_be_decoded_gives_no_peaks_and_stores_nothing(self, tmp_path, monkeypatch):
        self._settled(tmp_path)
        list_gallery_assets(str(tmp_path))
        monkeypatch.setattr(gallery, "read_waveform_peaks", lambda _path: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")))

        assert gallery.audio_peaks(str(tmp_path), "song.flac") == []
        assert _json(tmp_path, "song.flac")["peaks"] is None

    @pytest.mark.parametrize("asset_id", ["missing.flac", "../outside.flac", "pic.png", ".web_uploads/v.flac", "/etc/passwd"])
    def test_only_an_audio_file_under_the_output_root_has_peaks(self, tmp_path, asset_id):
        root = tmp_path / "out"
        (root / ".web_uploads").mkdir(parents=True)
        (root / ".web_uploads" / "v.flac").write_bytes(b"x")
        (root / "pic.png").write_bytes(b"\x89PNG")
        (tmp_path / "outside.flac").write_bytes(b"x")

        assert gallery.audio_peaks(str(root), asset_id) is None

    def test_the_route_returns_the_peaks_of_an_asset_and_404s_for_anything_else(self, tmp_path, monkeypatch, decodes):
        root = tmp_path / "out"
        root.mkdir()
        self._settled(root)
        list_gallery_assets(str(root))
        config = _web_config()
        config.output_dir = str(root)
        monkeypatch.setattr(web_server, "load_web_config", lambda: config)
        with TestClient(web_server.app) as client:
            found = client.get("/api/gallery-peaks", params={"id": "song.flac"})
            again = client.get("/api/gallery-peaks", params={"id": "song.flac"})
            missing = client.get("/api/gallery-peaks", params={"id": "nope.flac"})

        assert found.status_code == 200 and found.json()["id"] == "song.flac" and len(found.json()["peaks"]) == 100
        assert again.json() == found.json() and decodes == ["song.flac"]
        assert missing.status_code == 404

    def test_a_take_just_made_reaches_the_job_card_with_its_peaks(self, tmp_path, monkeypatch):
        from zvisiongenerator.web import web_runner

        path = _take(tmp_path, "song.flac", _SONG)
        monkeypatch.setattr(web_runner, "load_web_config", lambda: _web_config())

        payload = web_runner._output_asset_payload({"output_dir": str(tmp_path)}, str(path))

        assert payload is not None and len(payload["peaks"]) == 100


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
