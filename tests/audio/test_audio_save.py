"""Tests for saving a take as FLAC with its settings, the settings payload, and reading them back."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from tests._media import REAL
from tests.audio._helpers import clip, decode, sine

from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils.audio_save import save_flac
from zvisiongenerator.utils.audio_voice import is_any_voice_take, voice_needs_consent
from zvisiongenerator.utils.provenance import (
    AUDIO_CONFIG_SCHEMA,
    build_audio_config_payload,
    config_from_probe,
    embed_config,
    read_config,
)
from zvisiongenerator.utils.video_probe import ffprobe_json


class TestFlacRoundTrip:
    @REAL
    def test_a_mono_take_reads_back_with_its_audio_rate_and_settings(self, tmp_path):
        payload = {"schema": AUDIO_CONFIG_SCHEMA, "kind": "speech", "prompt": "Hello", "seed": 7}
        samples = sine(0.5, rate=24000, amplitude=0.5)

        path = save_flac(samples, 24000, tmp_path / "take.flac", payload)

        decoded, rate = decode(path)
        assert path == tmp_path / "take.flac" and rate == 24000 and decoded.shape[0] == 1
        assert decoded.shape[1] == samples.shape[1]
        np.testing.assert_allclose(decoded, samples, atol=2 / 32768)
        assert read_config(path) == payload

    @REAL
    def test_a_stereo_take_keeps_its_rate_and_channels(self, tmp_path):
        samples = np.stack([sine(0.25, rate=48000, amplitude=0.4)[0], sine(0.25, rate=48000, amplitude=0.2, frequency=330)[0]])

        path = save_flac(samples, 48000, tmp_path / "song.flac", {"kind": "music"})

        decoded, rate = decode(path)
        data = ffprobe_json(path, count_packets=False)
        assert rate == 48000 and decoded.shape == samples.shape
        assert data["streams"][0]["codec_name"] == "flac" and data["streams"][0]["bits_per_raw_sample"] == "16"
        np.testing.assert_allclose(decoded, samples, atol=2 / 32768)

    @REAL
    def test_lyrics_with_new_lines_and_non_latin_text_survive(self, tmp_path):
        payload = {"kind": "music", "prompt": "synth-pop", "lyrics": '[verse]\nHei, jeg elsker deg ÆØÅ\n[chorus]\n歌 "quoted" = ; # \\'}

        path = save_flac(sine(0.1), 24000, tmp_path / "s.flac", payload)

        assert read_config(path) == payload

    @REAL
    def test_a_clip_beyond_full_scale_is_clipped_not_wrapped(self, tmp_path):
        loud = sine(0.1, amplitude=2.0)

        decoded, _ = decode(save_flac(loud, 24000, tmp_path / "loud.flac", {}))

        assert decoded.max() <= 1.0 and decoded.min() >= -1.0 and decoded.max() > 0.99

    @REAL
    def test_no_scratch_file_is_left_beside_the_result(self, tmp_path):
        save_flac(sine(0.1), 24000, tmp_path / "take.flac", {})

        assert [item.name for item in tmp_path.iterdir()] == ["take.flac"]

    def test_a_failed_encode_raises_and_leaves_nothing_behind(self, tmp_path):
        failed = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"boom")

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", return_value=failed):
            with pytest.raises(RuntimeError, match="could not write the audio file"):
                save_flac(sine(0.1), 24000, tmp_path / "take.flac", {})

        assert list(tmp_path.iterdir()) == []


class TestConfigInAnyContainer:
    def test_the_tag_is_found_whatever_its_case(self):
        assert config_from_probe({"format": {"tags": {"ZVISIONGENERATOR.CONFIG": json.dumps({"a": 1})}}}) == {"a": 1}

    @REAL
    def test_embed_config_adds_the_settings_to_a_flac_without_changing_its_audio(self, tmp_path):
        path = tmp_path / "plain.flac"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=0.2", "-c:a", "flac", str(path)], check=True)
        assert read_config(path) is None
        before, _ = decode(path)

        embed_config(path, {"kind": "sound", "seed": 4})

        after, _ = decode(path)
        np.testing.assert_array_equal(before, after)
        assert read_config(path) == {"kind": "sound", "seed": 4}
        assert [item.name for item in tmp_path.iterdir()] == ["plain.flac"]


class TestPayload:
    def test_a_speech_take_records_the_full_text_and_its_part_count(self):
        text = "Sentence one. " * 100
        request = AudioGenerationRequest(preset="higgs3", kind="speech", prompt=text, seed=5)

        payload = build_audio_config_payload(request, clip(2.0), parts=3, generation_time=12.34)

        assert payload["prompt"] == text
        assert payload["generation"] == {"time": 12.3, "duration": 2.0, "parts": 3}
        assert (payload["schema"], payload["workflow"], payload["kind"], payload["model"], payload["seed"]) == (AUDIO_CONFIG_SCHEMA, "speech", "speech", "higgs3", 5)
        assert "voice_path" not in payload and "voice_text" not in payload and "lyrics" not in payload

    def test_a_cloned_voice_records_its_path_and_text_but_never_consent(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/someone")
        monkeypatch.setenv("USERPROFILE", "/home/someone")
        request = AudioGenerationRequest(preset="higgs3", kind="speech", prompt="Hi", voice=VoiceRef("/home/someone/me.wav", "Hello, this is me."), seed=1)

        payload = build_audio_config_payload(request, clip(), parts=1, generation_time=1.0)

        assert payload["voice_path"] == "~/me.wav" and payload["voice_text"] == "Hello, this is me."
        assert not any("consent" in key for key in payload)

    def test_a_song_records_style_lyrics_and_a_cut(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="synth-pop", lyrics="[verse]\nla", seed=2)

        payload = build_audio_config_payload(request, clip(1.0, rate=48000, cut=True, channels=2), parts=None, generation_time=0.0)

        assert (payload["prompt"], payload["lyrics"]) == ("synth-pop", "[verse]\nla")
        assert payload["generation"] == {"duration": 1.0, "cut": True}

    def test_a_sound_records_the_length_asked_for_and_the_length_made(self):
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="rain", seconds=2.0, seed=9)

        payload = build_audio_config_payload(request, clip(1.5, rate=44100), parts=None, generation_time=3.0)

        assert payload["seconds"] == 2.0 and payload["generation"]["duration"] == 1.5
        assert "parts" not in payload["generation"] and "cut" not in payload["generation"]


class TestAnyVoiceTake:
    def test_a_speech_take_without_a_recorded_voice_is_an_any_take(self):
        assert is_any_voice_take({"kind": "speech", "prompt": "Hi"})

    @pytest.mark.parametrize("config", [None, {}, {"kind": "speech", "voice_path": "~/me.wav"}, {"kind": "music"}, {"kind": "sound"}])
    def test_everything_else_is_not(self, config):
        assert not is_any_voice_take(config)

    @REAL
    def test_a_file_needs_consent_unless_its_settings_show_an_any_take(self, tmp_path):
        any_take = save_flac(sine(0.1), 24000, tmp_path / "any.flac", {"kind": "speech", "prompt": "Hi"})
        cloned = save_flac(sine(0.1), 24000, tmp_path / "cloned.flac", {"kind": "speech", "prompt": "Hi", "voice_path": "~/me.wav"})
        plain = tmp_path / "plain.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=0.1", str(plain)], check=True)

        assert not voice_needs_consent(any_take)
        assert voice_needs_consent(cloned)
        assert voice_needs_consent(plain)
        assert voice_needs_consent(tmp_path / "missing.wav")
