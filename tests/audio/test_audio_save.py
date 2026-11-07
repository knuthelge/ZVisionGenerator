"""Tests for saving a take as FLAC with its settings, the settings payload, and reading them back."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.audio._helpers import clip, sine

from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.utils.audio_save import save_flac
from zvisiongenerator.utils import audio_voice
from zvisiongenerator.utils.audio_voice import voice_needs_consent
from zvisiongenerator.utils.provenance import (
    AUDIO_CONFIG_SCHEMA,
    build_audio_config_payload,
    config_from_probe,
    read_config,
)


def _ffmpeg_that_writes_its_output(seen: dict):
    """Return a stand-in for ``subprocess.run`` that notes the encode command and the WAV it was given, and writes the output file it names."""

    def run(command, **_kwargs):
        wav = command[command.index("-i") + 1]
        with wave.open(wav, "rb") as reader:
            seen.update(
                command=command,
                wav_rate=reader.getframerate(),
                wav_channels=reader.getnchannels(),
                wav_pcm=np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16),
                destination_when_encoding=Path(command[-1]),
            )
        Path(command[-1]).write_bytes(b"encoded")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    return run


def _tag_of(command: list[str]) -> str:
    """Return the value of the ``zvisiongenerator.config`` tag in an encode command."""
    values = [command[index + 1] for index, word in enumerate(command) if word == "-metadata"]
    (value,) = [item.removeprefix("zvisiongenerator.config=") for item in values if item.startswith("zvisiongenerator.config=")]
    return value


class TestSaveFlac:
    @pytest.mark.parametrize(("rate", "channels"), [(24000, 1), (48000, 2)])
    def test_the_clip_is_encoded_from_a_wav_at_its_own_rate_and_channel_count(self, tmp_path, rate, channels):
        seen: dict = {}
        samples = sine(0.25, rate=rate, channels=channels)

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", side_effect=_ffmpeg_that_writes_its_output(seen)):
            save_flac(samples, rate, tmp_path / "take.flac", {"kind": "music"})

        assert (seen["wav_rate"], seen["wav_channels"]) == (rate, channels)
        assert seen["wav_pcm"].shape[0] == samples.shape[0] * samples.shape[1]

    def test_the_file_is_16_bit_flac_with_the_inputs_own_tags_dropped(self, tmp_path):
        seen: dict = {}

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", side_effect=_ffmpeg_that_writes_its_output(seen)):
            save_flac(sine(0.1), 24000, tmp_path / "take.flac", {})

        command = seen["command"]
        assert command[0] == "ffmpeg" and command[command.index("-c:a") + 1] == "flac" and command[command.index("-sample_fmt") + 1] == "s16"
        assert command[command.index("-map_metadata") + 1] == "-1"

    @pytest.mark.parametrize(
        "payload",
        [
            {"schema": AUDIO_CONFIG_SCHEMA, "kind": "speech", "prompt": "Hello", "seed": 7},
            {"kind": "music", "prompt": "synth-pop", "lyrics": '[verse]\nHei, jeg elsker deg ÆØÅ\n[chorus]\n歌 "quoted" = ; # \\'},
        ],
        ids=["plain", "new lines and non-latin text"],
    )
    def test_the_settings_go_into_the_tag_the_reader_looks_for(self, tmp_path, payload):
        seen: dict = {}

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", side_effect=_ffmpeg_that_writes_its_output(seen)):
            save_flac(sine(0.1), 24000, tmp_path / "take.flac", payload)

        assert config_from_probe({"format": {"tags": {"zvisiongenerator.config": _tag_of(seen["command"])}}}) == payload

    def test_a_clip_beyond_full_scale_is_clipped_not_wrapped(self, tmp_path):
        seen: dict = {}

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", side_effect=_ffmpeg_that_writes_its_output(seen)):
            save_flac(sine(0.1, amplitude=2.0), 24000, tmp_path / "loud.flac", {})

        pcm = seen["wav_pcm"]
        assert pcm.max() == 32767 and pcm.min() == -32767 and (np.abs(pcm) > 30000).sum() > 0.4 * pcm.size

    def test_the_result_appears_only_when_complete_and_no_scratch_file_is_left_beside_it(self, tmp_path):
        seen: dict = {}
        path = tmp_path / "take.flac"
        path.write_bytes(b"older take")

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", side_effect=_ffmpeg_that_writes_its_output(seen)):
            save_flac(sine(0.1), 24000, path, {})

        assert seen["destination_when_encoding"] != path and seen["destination_when_encoding"].parent == tmp_path
        assert path.read_bytes() == b"encoded" and [item.name for item in tmp_path.iterdir()] == ["take.flac"]

    def test_a_failed_encode_raises_and_leaves_nothing_behind(self, tmp_path):
        failed = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"boom")

        with patch("zvisiongenerator.utils.video_splice.subprocess.run", return_value=failed):
            with pytest.raises(RuntimeError, match="could not write the audio file"):
                save_flac(sine(0.1), 24000, tmp_path / "take.flac", {})

        assert list(tmp_path.iterdir()) == []


class TestConfigInAnyContainer:
    def test_the_tag_is_found_whatever_its_case(self):
        assert config_from_probe({"format": {"tags": {"ZVISIONGENERATOR.CONFIG": json.dumps({"a": 1})}}}) == {"a": 1}


class TestReadConfig:
    @staticmethod
    def _probe_prints(stdout: str, returncode: int = 0):
        def run(command, **_kwargs):
            if returncode:
                raise subprocess.CalledProcessError(returncode, command, stderr="bad")
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        return patch("zvisiongenerator.utils.provenance.subprocess.run", side_effect=run)

    def test_the_settings_are_read_from_the_format_tags_ffprobe_prints(self):
        printed = json.dumps({"format": {"tags": {"zvisiongenerator.config": json.dumps({"kind": "speech", "seed": 3})}}})

        with self._probe_prints(printed) as probe:
            config = read_config("take.flac")

        assert config == {"kind": "speech", "seed": 3}
        assert probe.call_args.args[0][0] == "ffprobe" and "-show_format" in probe.call_args.args[0]

    def test_a_file_without_the_tag_has_no_settings(self):
        with self._probe_prints(json.dumps({"format": {"tags": {"title": "x"}}})):
            assert read_config("plain.flac") is None

    def test_a_file_ffprobe_cannot_read_raises(self):
        with self._probe_prints("", returncode=1), pytest.raises(subprocess.CalledProcessError):
            read_config("broken.flac")


class TestPayload:
    def test_a_speech_take_records_the_full_text_and_its_part_count(self):
        text = "Sentence one. " * 100
        request = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt=text, seed=5)

        payload = build_audio_config_payload(request, clip(2.0), parts=3, generation_time=12.34)

        assert payload["prompt"] == text
        assert payload["generation"] == {"time": 12.3, "duration": 2.0, "parts": 3}
        assert (payload["schema"], payload["workflow"], payload["kind"], payload["model"], payload["seed"]) == (AUDIO_CONFIG_SCHEMA, "speech", "speech", "qwen3_tts", 5)
        assert "voice_path" not in payload and "voice_text" not in payload and "lyrics" not in payload

    def test_a_built_in_voice_take_records_its_speaker(self):
        request = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hi", seed=1, speaker="vivian")

        payload = build_audio_config_payload(request, clip(), parts=1, generation_time=1.0)

        assert payload["speaker"] == "vivian"
        assert "voice_path" not in payload and "voice_text" not in payload

    def test_a_take_with_a_recording_records_no_speaker(self):
        request = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hi", voice=VoiceRef("/home/someone/me.wav", "Hello"), seed=1)

        assert "speaker" not in build_audio_config_payload(request, clip(), parts=1, generation_time=1.0)

    def test_a_cloned_voice_records_its_path_and_text_but_never_consent(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/someone")
        monkeypatch.setenv("USERPROFILE", "/home/someone")
        request = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hi", voice=VoiceRef("/home/someone/me.wav", "Hello, this is me."), seed=1)

        payload = build_audio_config_payload(request, clip(), parts=1, generation_time=1.0)

        assert payload["voice_path"] == "~/me.wav" and payload["voice_text"] == "Hello, this is me."
        assert not any("consent" in key for key in payload)

    def test_a_song_records_style_lyrics_and_a_cut(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="synth-pop", lyrics="[verse]\nla", seed=2)

        payload = build_audio_config_payload(request, clip(1.0, rate=48000, cut=True, channels=2), parts=None, generation_time=0.0)

        assert (payload["prompt"], payload["lyrics"]) == ("synth-pop", "[verse]\nla")
        assert payload["generation"] == {"duration": 1.0, "cut": True}

    def test_the_notes_of_a_take_are_recorded_and_left_out_when_there_are_none(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="synth-pop", seed=2)

        noted = build_audio_config_payload(request, clip(1.0, notes=("Couldn't plan.", "Voice stays.")), parts=None, generation_time=0.0)
        plain = build_audio_config_payload(request, clip(1.0), parts=None, generation_time=0.0)

        assert noted["generation"]["notes"] == ["Couldn't plan.", "Voice stays."]
        assert "notes" not in plain["generation"]

    def test_a_sound_records_the_length_asked_for_and_the_length_made(self):
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="rain", seconds=2.0, seed=9)

        payload = build_audio_config_payload(request, clip(1.5, rate=44100), parts=None, generation_time=3.0)

        assert payload["seconds"] == 2.0 and payload["generation"]["duration"] == 1.5
        assert "parts" not in payload["generation"] and "cut" not in payload["generation"]

    def test_a_song_records_its_tempo_and_key_only_when_set(self):
        set_ = AudioGenerationRequest(preset="yue2", kind="music", prompt="pop", tempo=92, key="F#m", seed=2)
        auto = AudioGenerationRequest(preset="yue2", kind="music", prompt="pop", seed=2)

        payload = build_audio_config_payload(set_, clip(), parts=None, generation_time=1.0)

        assert (payload["tempo"], payload["key"]) == (92, "F#m")
        assert not {"tempo", "key"} & build_audio_config_payload(auto, clip(), parts=None, generation_time=1.0).keys()

    def test_music_of_a_model_that_sets_its_length_records_the_length_it_was_made_at(self):
        request = AudioGenerationRequest(preset="sa3", kind="music", prompt="pad", seconds=30.0, seed=2)

        assert build_audio_config_payload(request, clip(), parts=None, generation_time=1.0)["seconds"] == 30.0

    def test_a_sound_records_its_type_and_the_resolved_length(self):
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="door", seconds=5.0, sound_type="effect", seed=9)

        payload = build_audio_config_payload(request, clip(), parts=None, generation_time=1.0)

        assert (payload["sound_type"], payload["seconds"]) == ("effect", 5.0)

    def test_speech_in_a_built_in_voice_records_all_five_delivery_values_and_a_recording_none(self):
        delivery = SpeechDelivery(emotion="sad", manner="whisper", pace="slow", pitch="low", expressiveness="high")
        built_in = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hi", speaker="vivian", delivery=delivery, seed=1)
        neutral = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hi", speaker="ryan", delivery=SpeechDelivery(), seed=1)
        recording = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hi", voice=VoiceRef("/a/me.wav", "Hello"), seed=1)

        payload = build_audio_config_payload(built_in, clip(), parts=1, generation_time=1.0)
        fields = ("emotion", "manner", "pace", "pitch", "expressiveness")

        assert [payload[name] for name in fields] == ["sad", "whisper", "slow", "low", "high"]
        assert [build_audio_config_payload(neutral, clip(), parts=1, generation_time=1.0)[name] for name in fields] == ["neutral", "natural", "normal", "normal", "normal"]
        assert not set(fields) & build_audio_config_payload(recording, clip(), parts=1, generation_time=1.0).keys()

    def test_an_enhanced_take_records_how_and_the_own_text_even_when_empty(self):
        record = {"settings": {"note": ""}, "model": "fast", "tier": "fast", "format": "text", "task": "style"}
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="dreamy lo-fi", prompt_input="", enhance_record=record, seed=2)

        payload = build_audio_config_payload(request, clip(), parts=None, generation_time=1.0)

        assert (payload["enhance"], payload["prompt_input"], payload["prompt"]) == (record, "", "dreamy lo-fi")

    def test_a_take_the_user_wrote_records_neither_enhance_nor_own_text(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="dreamy lo-fi", prompt_input="lo-fi", seed=2)

        assert not {"enhance", "prompt_input"} & build_audio_config_payload(request, clip(), parts=None, generation_time=1.0).keys()


class TestAnyVoiceTake:
    """A file needs the speaker's consent unless its embedded settings show a speech take made without a reference voice."""

    @pytest.mark.parametrize(
        ("config", "needs_consent"),
        [
            ({"kind": "speech", "prompt": "Hi"}, False),
            ({"kind": "speech", "prompt": "Hi", "voice_path": "~/me.wav"}, True),
            ({"kind": "music", "prompt": "synth-pop"}, True),
            (None, True),
        ],
        ids=["any take", "cloned take", "not speech", "no settings"],
    )
    def test_only_a_speech_take_without_a_reference_voice_is_free(self, monkeypatch, config, needs_consent):
        monkeypatch.setattr(audio_voice, "read_config", lambda path: config)

        assert voice_needs_consent("voice.flac") is needs_consent

    @pytest.mark.parametrize(
        "error", [subprocess.CalledProcessError(1, ["ffprobe"]), json.JSONDecodeError("bad", "", 0), FileNotFoundError("missing")], ids=["ffprobe fails", "bad settings", "missing"]
    )
    def test_a_file_whose_settings_cannot_be_read_needs_consent(self, monkeypatch, error):
        def unreadable(path):
            raise error

        monkeypatch.setattr(audio_voice, "read_config", unreadable)

        assert voice_needs_consent("voice.wav")
