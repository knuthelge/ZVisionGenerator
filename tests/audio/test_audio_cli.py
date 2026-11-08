"""Tests for ``ziv audio``: the flags, each refusal, the licence notice, and the commands Copy CLI builds."""

from __future__ import annotations

import functools
import json
import shlex
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from tests._audio_presets import audio_config, audio_entry, qwen3_tts_preset, sa3_preset, sa3_two_kind_preset, use_audio_models, yue2_preset
from tests._factories import load_packaged_config, use_enhancer_models
from tests._fake_audio import fake_read_config as read_config
from tests._licences import use_licences
from tests._paths import REPO_ROOT
from tests.audio._helpers import case_request, clip, fake_backend, patch_runner, sine, voice_folder

from zvisiongenerator import audio_cli
from zvisiongenerator.audio_cli import _build_audio_parser
from zvisiongenerator.backends import prompt_enhancer_session
from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.downloads import require_checked
from zvisiongenerator.utils.audio_settings import VIBES
from zvisiongenerator.utils.audio_track import TRACK_UNREADABLE
from zvisiongenerator.utils.enhance_record import parse_enhance_record
from zvisiongenerator.utils.library_noise import MUTED_ERROR_TEXT
from zvisiongenerator.utils import audio_request as audio_request_module
from zvisiongenerator.utils.prompt_enhance_audio import SPACE_AXIS
from zvisiongenerator.utils.provenance import build_audio_config_payload
from zvisiongenerator.utils.video_splice import write_pcm16_wav
from zvisiongenerator.web.audio_job import AudioFields, audio_request_from_fields
from zvisiongenerator.web.gallery_audio import audio_asset_facts

CASES = json.loads((REPO_ROOT / "tests" / "fixtures" / "audio_cli_cases.json").read_text(encoding="utf-8"))


@pytest.fixture
def voices(tmp_path):
    """Return a folder with a recording (``me.wav``), a take in a built-in voice (``any.flac``) and a cloned take (``cloned.flac``), each 4 s."""
    return voice_folder(tmp_path)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Run ``main`` with the three fixture presets on a CUDA machine and a backend that makes a short clip; return the backend and the patched pieces."""
    backend = fake_backend(clip(0.2))
    patched = patch_runner(monkeypatch, backend)
    patched.ensure = MagicMock(return_value="ready")
    monkeypatch.setattr(audio_cli, "ensure_downloaded", patched.ensure)
    use_audio_models(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_cli, "load_config", lambda: audio_config())
    monkeypatch.setattr(audio_cli, "ensure_ffmpeg", lambda: None)
    patched.backend = backend
    patched.out = tmp_path / "out"
    return patched


def run_cli(monkeypatch, *argv: str, prog: str = "ziv audio") -> None:
    monkeypatch.setattr(sys, "argv", [prog, *argv])
    audio_cli.main(prog=prog)


def refuses(monkeypatch, capsys, *argv: str) -> str:
    """Run the CLI, expecting it to exit with a usage error, and return the error text."""
    with pytest.raises(SystemExit) as exit_:
        run_cli(monkeypatch, *argv)
    assert exit_.value.code == 2
    return capsys.readouterr().err


def make(env, *argv: str, monkeypatch) -> list[AudioGenerationRequest]:
    run_cli(monkeypatch, *argv, "-o", str(env.out))
    return [call.args[0] for call in env.backend.generate.call_args_list]


class TestQuit:
    @pytest.mark.parametrize("outcome", ["stopped", "detached"])
    def test_a_quit_during_the_download_says_so_on_its_own_row_and_makes_nothing(self, env, monkeypatch, capsys, outcome):
        env.ensure.return_value = outcome

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "-o", str(env.out))

        out = capsys.readouterr().out
        assert out.endswith("⏹ Quitting batch...\n") and not out.endswith("\n\n")
        env.backend.load.assert_not_called()

    def test_the_commands_line_is_followed_by_no_blank_row(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "stdin_is_interactive", lambda: True)
        monkeypatch.setattr(audio_cli.SkipSignal, "start", lambda self: None)

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "-o", str(env.out))

        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("Commands:") and lines[1] != ""


class TestConsoleNoise:
    def test_warnings_and_log_lines_are_routed_through_the_live_line_with_the_known_noise_muted(self, env, monkeypatch):
        route = MagicMock()
        monkeypatch.setattr(audio_cli, "route_stderr_through_live_line", route)

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        route.assert_called_once_with(mute=MUTED_ERROR_TEXT)


class TestFlags:
    def test_lyrics_and_a_lyrics_file_cannot_be_given_together(self, capsys):
        with pytest.raises(SystemExit):
            _build_audio_parser().parse_args(["--lyrics", "a", "--lyrics-file", "b.txt"])

        assert "not allowed with argument" in capsys.readouterr().err

    def test_a_song_is_made_from_a_style_and_lyrics_with_the_defaults_for_the_rest(self, env, monkeypatch, tmp_path):
        lyrics = tmp_path / "song.txt"
        lyrics.write_text("[verse]\nla la\n", encoding="utf-8")

        requests = make(env, "-m", "yue2", "--prompt", "synth-pop", "--lyrics-file", str(lyrics), "--seed", "4", monkeypatch=monkeypatch)

        assert requests == [AudioGenerationRequest(preset="yue2", kind="music", prompt="synth-pop", lyrics="[verse]\nla la\n", seconds=None, voice=None, seed=4)]
        assert len(list(env.out.glob("music_*.flac"))) == 1

    @pytest.mark.parametrize(("lyrics", "kind"), [(None, "instrumental"), ("[intro] [verse] [outro]", "instrumental"), ("[verse]\nwords here", "song")])
    def test_no_lyrics_or_only_tags_make_an_instrumental(self, env, monkeypatch, capsys, lyrics, kind):
        extra = ["--lyrics", lyrics] if lyrics is not None else []

        make(env, "-m", "yue2", "--prompt", "calm", *extra, monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert ("Making an instrumental" in out) == (kind == "instrumental") and ("Making a song" in out) == (kind == "song")

    @pytest.mark.parametrize(("lyrics", "sections"), [(None, 8), ("[Intro] [Instrumental] [Solo] [Outro]", 4)])
    def test_the_shipped_config_gives_an_empty_or_tags_only_lyrics_box_its_sections(self, env, monkeypatch, capsys, lyrics, sections):
        """The request keeps the lyrics as given (the backend swaps in the default list); the announcement counts the packaged list."""
        monkeypatch.setattr(audio_cli, "load_config", load_packaged_config)
        extra = ["--lyrics", lyrics] if lyrics is not None else []

        (request,) = make(env, "-m", "yue2", "--prompt", "calm piano, slow", *extra, monkeypatch=monkeypatch)

        assert (request.prompt, request.lyrics) == ("calm piano, slow", lyrics or "")
        assert f"Making an instrumental of {sections} sections." in capsys.readouterr().out

    def test_a_sound_takes_twelve_seconds_unless_told(self, env, monkeypatch):
        assert make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)[0].seconds == 12.0

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["-m", "sa3", "--prompt", "rain"], {"sound_type": "auto", "delivery": None, "tempo": None, "key": None}),
            (["-m", "qwen3_tts", "--prompt", "Hi"], {"sound_type": None, "delivery": SpeechDelivery()}),
            (["-m", "yue2", "--prompt", "pop"], {"sound_type": None, "delivery": None, "tempo": None, "key": None}),
        ],
        ids=["sound", "speech in a built-in voice", "music"],
    )
    def test_the_request_carries_the_settings_the_form_defaults_to(self, env, monkeypatch, argv, expected):
        (request,) = make(env, *argv, monkeypatch=monkeypatch)

        assert {name: getattr(request, name) for name in expected} == expected

    def test_a_recording_has_no_delivery(self, env, monkeypatch, voices):
        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Hi", "--voice", str(voices / "any.flac"), "--voice-text", "Welcome back.", monkeypatch=monkeypatch)

        assert request.delivery is None

    @pytest.fixture
    def music_first(self, env, monkeypatch, tmp_path):
        """Make ``sa3`` a model of two kinds with music first: music without lyrics, whose length is chosen."""
        use_audio_models(monkeypatch, tmp_path, yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3={**sa3_preset(), "kinds": ["music", "sound"]})

    def test_a_model_is_made_as_its_first_kind_and_that_kind_decides_the_flags(self, env, monkeypatch, music_first):
        (request,) = make(env, "-m", "sa3", "--prompt", "calm piano", monkeypatch=monkeypatch)

        assert (request.kind, request.lyrics, request.seconds) == ("music", "", 30.0)

    def test_music_without_lyrics_takes_a_length_and_refuses_lyrics(self, env, monkeypatch, capsys, music_first):
        (request,) = make(env, "-m", "sa3", "--prompt", "calm piano", "--seconds", "45", monkeypatch=monkeypatch)

        assert request.seconds == 45.0
        assert "takes no lyrics" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--lyrics", "[verse]")

    def test_the_length_of_music_is_checked_against_the_models_limit(self, env, monkeypatch, capsys, music_first):
        assert "makes songs of that length" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--seconds", "400")

    def test_the_runs_are_taken_with_one_load(self, env, monkeypatch):
        requests = make(env, "-m", "sa3", "--prompt", "rain", "--seconds", "5", "-r", "3", monkeypatch=monkeypatch)

        assert len(requests) == 3 and env.backend.load.call_count == 1

    def test_ziv_dispatches_audio_to_this_command(self, env, monkeypatch):
        from zvisiongenerator import cli

        monkeypatch.setattr(sys, "argv", ["ziv", "audio", "-m", "sa3", "--prompt", "rain", "-o", str(env.out)])
        cli.main()

        assert env.backend.generate.call_count == 1


class TestRefusals:
    def test_a_flag_the_model_does_not_take_is_refused_by_name(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "--seconds", "30")

        assert "--seconds" in error and "YuE2 3B" in error

    @pytest.mark.parametrize(("flag", "model"), [("--lyrics", "sa3"), ("--lyrics", "qwen3_tts")])
    def test_lyrics_are_refused_by_a_model_without_them(self, env, monkeypatch, capsys, flag, model):
        assert flag in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", flag, "[verse]")

    @pytest.mark.parametrize("model", ["yue2", "sa3"])
    def test_a_voice_is_refused_by_a_model_that_cannot_follow_one(self, env, monkeypatch, capsys, model):
        assert "--voice" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--voice", "me.wav", "--voice-text", "x", "--voice-consent")

    def test_a_voice_without_its_text_is_refused(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-consent")

        assert "--voice-text" in error
        assert env.backend.load.call_count == 0

    def test_a_voice_without_consent_is_refused(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-text", "Hello")

        assert "--voice-consent" in error
        assert env.ensure.call_count == 0 and env.backend.load.call_count == 0

    def test_a_cloned_take_needs_consent_too(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(voices / "cloned.flac"), "--voice-text", "Hello")

        assert "--voice-consent" in error

    def test_a_take_in_a_built_in_voice_needs_no_consent(self, env, monkeypatch, voices):
        requests = make(env, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(voices / "any.flac"), "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert requests[0].voice == VoiceRef(str(voices / "any.flac"), "Hello")

    def test_a_relative_voice_path_is_recorded_in_full(self, env, monkeypatch, voices):
        monkeypatch.chdir(voices)

        requests = make(env, "-m", "qwen3_tts", "--prompt", "x", "--voice", "any.flac", "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert requests[0].voice == VoiceRef(str((voices / "any.flac").resolve()), "Hello")

    def test_ffmpeg_is_asked_for_before_the_recording_is_read(self, env, monkeypatch, capsys, voices):
        """A machine without ffmpeg is offered the install, not a consent refusal or a probe failure from the missing tool."""
        probes: list[str] = []

        def declined() -> None:
            raise SystemExit(1)

        monkeypatch.setattr(audio_cli, "ensure_ffmpeg", declined)
        monkeypatch.setattr(audio_cli, "voice_needs_consent", lambda path: probes.append(str(path)) or True)
        monkeypatch.setattr(audio_request_module, "probe_track_seconds", lambda path: probes.append(str(path)) or 4.0)

        with pytest.raises(SystemExit) as exit_:
            run_cli(monkeypatch, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-text", "Hello")

        assert exit_.value.code == 1
        assert probes == []
        assert "--voice-consent" not in capsys.readouterr().err

    def test_speech_without_a_speaker_is_read_in_the_models_default_voice(self, env, monkeypatch):
        requests = make(env, "-m", "qwen3_tts", "--prompt", "x", monkeypatch=monkeypatch)

        assert (requests[0].speaker, requests[0].voice) == ("ryan", None)

    def test_a_speaker_is_the_voice_of_the_take(self, env, monkeypatch):
        requests = make(env, "-m", "qwen3_tts", "--prompt", "x", "--speaker", "vivian", monkeypatch=monkeypatch)

        assert requests[0].speaker == "vivian"

    def test_a_speaker_the_model_lacks_is_refused_with_the_ones_it_has(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--speaker", "nobody")

        assert "nobody" in error and "ryan, aiden, vivian" in error
        assert env.ensure.call_count == 0

    def test_a_speaker_with_a_voice_recording_is_refused(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--speaker", "ryan", "--voice", str(voices / "me.wav"), "--voice-text", "Hello", "--voice-consent")

        assert "--speaker picks a built-in voice; drop it to follow --voice" in error

    @pytest.mark.parametrize("model", ["yue2", "sa3"])
    def test_a_speaker_is_refused_by_a_model_without_built_in_voices(self, env, monkeypatch, capsys, model):
        assert "--speaker" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--speaker", "ryan")

    def test_a_take_with_a_recording_has_no_speaker(self, env, monkeypatch, voices):
        requests = make(env, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(voices / "any.flac"), "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert requests[0].speaker is None

    def test_the_help_lists_the_voices_of_each_speech_model(self, env, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["ziv audio", "--help"])

        with pytest.raises(SystemExit):
            audio_cli.main(prog="ziv audio")

        text = " ".join(capsys.readouterr().out.split())
        assert "--speaker ID" in text and "qwen3_tts: ryan, aiden, vivian, serena, uncle_fu, dylan, eric, ono_anna, sohee" in text

    def test_a_voice_that_is_not_there_is_refused(self, env, monkeypatch, capsys, tmp_path):
        assert "gone.wav" in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(tmp_path / "gone.wav"), "--voice-text", "x", "--voice-consent")

    def test_a_voice_that_is_not_an_audio_file_is_refused_by_name(self, env, monkeypatch, capsys, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("not audio", encoding="utf-8")

        def unreadable(path):
            raise ValueError(TRACK_UNREADABLE)

        monkeypatch.setattr(audio_request_module, "probe_track_seconds", unreadable)

        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(notes), "--voice-text", "x", "--voice-consent")

        assert f"{TRACK_UNREADABLE} in {notes}" in error and env.ensure.call_count == 0

    @pytest.mark.parametrize("flag", [["--voice-text", "x"], ["--voice-consent"]])
    def test_the_voice_flags_need_a_voice(self, env, monkeypatch, capsys, flag):
        assert f"{flag[0]} needs --voice" in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", *flag)

    def test_a_recording_outside_the_allowed_length_is_refused_before_anything_loads(self, env, monkeypatch, capsys, tmp_path):
        short = tmp_path / "short.wav"
        write_pcm16_wav(sine(1.0, rate=24000), short, 24000)

        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--voice", str(short), "--voice-text", "x", "--voice-consent")

        assert "3" in error and "30" in error and env.ensure.call_count == 0

    @pytest.mark.parametrize("model", ["yue2", "qwen3_tts", "sa3"])
    def test_a_missing_or_blank_prompt_is_refused_by_flag_name(self, env, monkeypatch, capsys, model):
        assert "--prompt" in refuses(monkeypatch, capsys, "-m", model)
        assert "--prompt" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "   ")

    def test_speech_over_10000_characters_is_refused(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "word " * 2001)

        assert "10,000" in error  # the limit is the config's, shown with a thousands separator
        assert env.ensure.call_count == 0

    @pytest.mark.parametrize("seconds", ["0.5", "381", "0", "-3", "nan"])
    def test_a_length_outside_the_models_range_is_refused(self, env, monkeypatch, capsys, seconds):
        error = refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--seconds", seconds)

        assert "--seconds" in error and "380" in error

    def test_a_lyrics_file_that_cannot_be_read_is_refused(self, env, monkeypatch, capsys, tmp_path):
        assert "gone.txt" in refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "--lyrics-file", str(tmp_path / "gone.txt"))

    @pytest.mark.parametrize("seed", ["-1", str(2**63)])
    def test_a_seed_no_model_takes_is_refused(self, env, monkeypatch, capsys, seed):
        assert "--seed" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--seed", seed)
        env.backend.load.assert_not_called()

    def test_zero_runs_are_refused(self, env, monkeypatch, capsys):
        assert "--runs" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "-r", "0")


class TestSteeringFlags:
    """``--kind``, ``--tempo``, ``--key``, the delivery flags and ``--sound-type``: each builds the request of the form and each is refused where it does not apply."""

    @pytest.fixture
    def multi(self, env, monkeypatch, tmp_path):
        """Make ``sa3`` a model of two kinds (sound first, then music without lyrics) next to the speech and song models."""
        use_audio_models(monkeypatch, tmp_path, yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3=sa3_two_kind_preset())
        return env

    def test_a_song_takes_a_tempo_and_a_key(self, env, monkeypatch):
        (request,) = make(env, "-m", "yue2", "--prompt", "pop", "--tempo", "96", "--key", "F#m", monkeypatch=monkeypatch)

        assert (request.tempo, request.key) == (96, "F#m")

    def test_the_help_names_the_auto_lengths_and_every_choice_of_the_shared_vocabulary(self):
        from zvisiongenerator.utils.audio_settings import AUTO_SOUND_SECONDS, EMOTIONS, LEVELS, MANNERS, MUSIC_KEYS, PACES, SOUND_TYPES

        text = "".join(_build_audio_parser().format_help().split())  # help wraps at spaces and hyphens, so compare without whitespace

        for name, seconds in AUTO_SOUND_SECONDS.items():
            if name != "auto":
                assert f"{name}{seconds:g}" in text
        for choices in (MUSIC_KEYS, SOUND_TYPES, EMOTIONS, MANNERS, PACES, LEVELS):
            assert ",".join(choices) in text

    @pytest.mark.parametrize(
        ("flag", "value", "message"),
        [("--tempo", "fast", "Tempo must be 40 to 220"), ("--tempo", "300", "Tempo must be 40 to 220"), ("--tempo", "90.5", "Tempo must be 40 to 220"), ("--key", "H", "Key must be one of")],
    )
    def test_a_tempo_or_key_out_of_range_is_refused_before_anything_loads(self, env, monkeypatch, capsys, flag, value, message):
        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "pop", flag, value)

        assert f"{flag}: {message}" in error
        assert env.ensure.call_count == 0

    @pytest.mark.parametrize("flag", ["--tempo", "--key"])
    @pytest.mark.parametrize("model", ["qwen3_tts", "sa3"])
    def test_a_tempo_or_key_is_refused_by_a_model_that_makes_no_song(self, env, monkeypatch, capsys, flag, model):
        value = "100" if flag == "--tempo" else "C"

        error = refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", flag, value)

        assert error.count(f"has no {flag[2:]}; drop {flag}") == 1

    def test_the_refusal_of_a_tempo_names_the_model(self, env, monkeypatch, capsys):
        assert "Qwen3-TTS 1.7B has no tempo; drop --tempo" in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--tempo", "100")

    def test_every_delivery_flag_sets_its_part_of_the_delivery(self, env, monkeypatch):
        argv = ["--emotion", "sad", "--manner", "whisper", "--pace", "very-slow", "--pitch", "low", "--expressiveness", "high"]

        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Hi", *argv, monkeypatch=monkeypatch)

        assert request.delivery == SpeechDelivery("sad", "whisper", "very-slow", "low", "high")

    def test_a_delivery_flag_leaves_the_others_at_their_defaults(self, env, monkeypatch):
        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Hi", "--pace", "fast", monkeypatch=monkeypatch)

        assert request.delivery == SpeechDelivery(pace="fast")

    @pytest.mark.parametrize(
        ("flag", "message"),
        [
            ("--emotion", "Emotion must be one of"),
            ("--manner", "Manner must be one of"),
            ("--pace", "Pace must be one of"),
            ("--pitch", "Pitch must be one of"),
            ("--expressiveness", "Expressiveness must be one of"),
        ],
    )
    def test_a_delivery_value_that_is_not_a_choice_is_refused(self, env, monkeypatch, capsys, flag, message):
        assert message in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "Hi", flag, "nope")

    def test_delivery_is_refused_with_a_recording_and_says_to_drop_it(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--emotion", "sad", "--voice", str(voices / "any.flac"), "--voice-text", "Hello")

        assert "Delivery applies to built-in voices; drop --emotion to follow --voice" in error

    @pytest.mark.parametrize("model", ["yue2", "sa3"])
    def test_delivery_is_refused_by_a_model_that_does_not_speak(self, env, monkeypatch, capsys, model):
        assert "has no delivery; drop --pace" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--pace", "fast")

    def test_a_vibe_sets_the_five_delivery_values_and_the_take_keeps_those_not_the_name(self, env, monkeypatch):
        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Hi", "--vibe", "whispered-secret", monkeypatch=monkeypatch)

        assert request.delivery == VIBES["whispered-secret"] == SpeechDelivery("neutral", "whisper", "slow", "low", "low")
        payload = build_audio_config_payload(request, clip(0.1), parts=1, generation_time=1.0)
        assert {field: payload[field] for field in ("emotion", "manner", "pace", "pitch", "expressiveness")} == {
            "emotion": "neutral",
            "manner": "whisper",
            "pace": "slow",
            "pitch": "low",
            "expressiveness": "low",
        }
        assert "whispered-secret" not in json.dumps(payload)

    def test_a_delivery_flag_overrides_that_one_value_of_the_vibe(self, env, monkeypatch):
        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Hi", "--pace", "very-fast", "--vibe", "storyteller", "--emotion", "sad", monkeypatch=monkeypatch)

        assert request.delivery == SpeechDelivery("sad", "natural", "very-fast", "normal", "high")

    def test_an_unknown_vibe_is_refused_with_the_ids_there_are(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "Hi", "--vibe", "villain")

        assert "--vibe: Vibe must be one of: " + ", ".join(VIBES) in error

    def test_a_vibe_is_refused_with_a_recording_and_says_to_drop_it(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--vibe", "announcer", "--voice", str(voices / "any.flac"), "--voice-text", "Hello")

        assert "Delivery applies to built-in voices; drop --vibe to follow --voice" in error

    @pytest.mark.parametrize("model", ["yue2", "sa3"])
    def test_a_vibe_is_refused_by_a_model_that_does_not_speak(self, env, monkeypatch, capsys, model):
        assert "has no delivery; drop --vibe" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--vibe", "announcer")

    def test_a_mark_in_the_text_is_part_of_the_request_as_written(self, env, monkeypatch):
        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Fine. [sad] Not fine.", monkeypatch=monkeypatch)

        assert request.prompt == "Fine. [sad] Not fine."

    @pytest.mark.parametrize(
        ("text", "way_out"),
        [("It was [sadd] odd.", "remove the brackets"), ("It was [laughs] funny.", "write haha"), ("Wait [pause] now.", "type …")],
    )
    def test_a_bracketed_word_that_is_not_a_mark_is_refused_with_the_way_out(self, env, monkeypatch, capsys, text, way_out):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", text)

        mark = text[text.index("[") : text.index("]") + 1]
        assert f"{mark} isn't a Delivery mark" in error and way_out in error and "`" not in error
        assert env.ensure.call_count == 0

    @pytest.mark.parametrize(("sound_type", "seconds"), [("auto", 12.0), ("ambience", 12.0), ("effect", 5.0), ("one-shot", 2.0)])
    def test_a_sound_type_sets_the_auto_length(self, env, monkeypatch, sound_type, seconds):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--sound-type", sound_type, monkeypatch=monkeypatch)

        assert (request.sound_type, request.seconds) == (sound_type, seconds)

    def test_a_length_given_wins_over_the_auto_length_of_the_type(self, env, monkeypatch):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--sound-type", "effect", "--seconds", "8", monkeypatch=monkeypatch)

        assert (request.sound_type, request.seconds) == ("effect", 8.0)

    def test_a_sound_type_that_is_not_a_choice_is_refused(self, env, monkeypatch, capsys):
        assert "--sound-type: Sound type must be one of: auto, ambience, effect, one-shot." in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--sound-type", "loud")

    @pytest.mark.parametrize("model", ["yue2", "qwen3_tts"])
    def test_a_sound_type_is_refused_by_a_model_that_makes_no_sound(self, env, monkeypatch, capsys, model):
        assert "has no sound type; drop --sound-type" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--sound-type", "effect")

    @pytest.mark.parametrize(
        ("argv", "fields"),
        [
            (["-m", "yue2", "--prompt", "pop", "--tempo", "96", "--key", "F#m"], {"tempo": 96, "key": "F#m"}),
            (["-m", "qwen3_tts", "--prompt", "Hi", "--emotion", "sad", "--pace", "slow", "--pitch", "high"], {"emotion": "sad", "pace": "slow", "pitch": "high"}),
            (["-m", "sa3", "--prompt", "door", "--sound-type", "effect"], {"sound_type": "effect"}),
        ],
        ids=["music", "speech", "sound"],
    )
    def test_the_command_builds_the_request_the_web_form_builds(self, env, monkeypatch, argv, fields):
        from dataclasses import replace

        from zvisiongenerator.web.audio_job import AudioFields, audio_request_from_fields

        (request,) = make(env, *argv, monkeypatch=monkeypatch)
        preset = next(p for p in audio_cli.parse_audio_presets(audio_config()) if p.id == argv[1])
        from_form = audio_request_from_fields(preset, request.kind, AudioFields(prompt=argv[3], **fields))

        assert replace(from_form, seed=request.seed) == request

    def test_a_model_of_two_kinds_makes_the_one_named(self, multi, monkeypatch):
        make(multi, "-m", "sa3", "--prompt", "calm piano", "--kind", "music", "--tempo", "80", "--key", "Am", monkeypatch=monkeypatch)
        make(multi, "-m", "sa3", "--prompt", "rain", "--kind", "sound", "--sound-type", "effect", monkeypatch=monkeypatch)
        made = [call.args[0] for call in multi.backend.generate.call_args_list]

        assert [(r.kind, r.tempo, r.key, r.sound_type, r.seconds) for r in made] == [("music", 80, "Am", None, 30.0), ("sound", None, None, "effect", 5.0)]

    def test_a_model_of_two_kinds_is_made_as_its_first_kind_without_kind(self, multi, monkeypatch):
        (request,) = make(multi, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        assert request.kind == "sound"

    def test_the_kind_decides_which_flags_apply(self, multi, monkeypatch, capsys):
        assert "Stable Audio 3 Medium has no sound type for music; drop --sound-type" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--kind", "music", "--sound-type", "effect")
        assert "Stable Audio 3 Medium has no tempo for sound; drop --tempo" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--tempo", "90")

    def test_a_kind_the_model_lacks_is_refused_with_the_kinds_it_makes(self, multi, monkeypatch, capsys):
        assert "--kind speech: Stable Audio 3 Medium makes sound and music, not speech." in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--kind", "speech")
        assert "--kind music: Qwen3-TTS 1.7B makes speech, not music." in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--kind", "music")

    def test_a_kind_that_does_not_exist_is_refused_with_the_choices(self, env, monkeypatch, capsys):
        assert "--kind must be one of: music, speech, sound." in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--kind", "video")

    def test_the_kind_of_a_model_of_one_kind_may_be_named(self, env, monkeypatch):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--kind", "sound", monkeypatch=monkeypatch)

        assert request.kind == "sound"

    def test_the_missing_prompt_is_asked_for_by_the_kind_named(self, multi, monkeypatch, capsys):
        assert "a description of the sound" in refuses(monkeypatch, capsys, "-m", "sa3")
        assert "the style of the song" in refuses(monkeypatch, capsys, "-m", "sa3", "--kind", "music")

    def test_the_speaker_of_a_non_speech_kind_is_refused(self, multi, monkeypatch, capsys, tmp_path):
        # a model with voices that makes more than one kind: speech and sound
        speech_and_sound = qwen3_tts_preset()
        multi_voices = {**speech_and_sound, "kinds": ["speech", "sound"], "limits": {**speech_and_sound["limits"], "max_seconds": 60}}
        use_audio_models(monkeypatch, tmp_path, yue2=yue2_preset(), qwen3_tts=multi_voices, sa3=sa3_preset())

        assert "has no voice for sound; drop --speaker" in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "--kind", "sound", "--speaker", "ryan")


class TestModels:
    def test_a_config_without_audio_models_says_so(self, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "audio_platform_key", lambda: "cuda")
        monkeypatch.setattr(audio_cli, "load_config", lambda: {**load_packaged_config(), "audio_model_presets": {}})

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x")

        assert "audio_model_presets" in error

    def test_a_model_is_required_and_the_choices_are_listed(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "--prompt", "x")

        assert "--model" in error and all(model in error for model in ("yue2", "qwen3_tts", "sa3"))

    def test_an_unknown_model_is_refused_with_the_choices(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "nope", "--prompt", "x")

        assert "nope" in error and all(model in error for model in ("yue2", "qwen3_tts", "sa3"))


class TestLicenceNotice:
    LICENCE = {
        "sa3-licence": {
            "name": "Stability AI Community License",
            "supplier": "Stability AI",
            "summary": "Free under $1M revenue",
            "url": "https://example.com/blob/75aad8836271cb5a5123fb58304fae5d3bb4b0f9/LICENSE",
            "models": ["sa3"],
        }
    }

    @pytest.fixture(autouse=True)
    def _licence_records(self, env, monkeypatch, tmp_path):
        models = {model_id: audio_entry(preset) for model_id, preset in (("yue2", yue2_preset()), ("qwen3_tts", qwen3_tts_preset()), ("sa3", sa3_preset()))}
        use_licences(monkeypatch, tmp_path, self.LICENCE, models=models)

    def test_the_terms_are_printed_before_a_missing_model_downloads(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", audio_config)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)
        env.ensure.side_effect = lambda *a, **k: print("DOWNLOAD STARTS") or "ready"

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert "By downloading you agree to these terms:" in out and "Stability AI Community License" in out
        assert out.index("By downloading you agree") < out.index("DOWNLOAD STARTS")

    def test_nothing_is_printed_for_a_model_that_is_already_downloaded(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", audio_config)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: True)

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        assert "By downloading" not in capsys.readouterr().out

    def test_a_model_without_a_record_prints_nothing(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", audio_config)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)

        make(env, "-m", "yue2", "--prompt", "calm", monkeypatch=monkeypatch)

        assert "By downloading" not in capsys.readouterr().out


class TestQwenCloningFiles:
    """``--voice`` makes the CLI fetch Qwen3-TTS's cloning checkpoint as well; a built-in voice take fetches CustomVoice alone."""

    CUSTOM, BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    @pytest.fixture(autouse=True)
    def _qwen(self, env, monkeypatch):
        monkeypatch.setattr(audio_cli, "load_config", lambda: audio_config(qwen3_tts=qwen3_tts_preset()))

    def test_a_built_in_voice_take_downloads_customvoice_alone(self, env, monkeypatch):
        make(env, "-m", "qwen3_tts", "--prompt", "Hello there.", monkeypatch=monkeypatch)

        assert [target.repo_id for target in env.ensure.call_args.args[0]] == [self.CUSTOM]

    def test_a_take_with_a_recording_downloads_the_cloning_checkpoint_too(self, env, monkeypatch, voices):
        make(env, "-m", "qwen3_tts", "--prompt", "Hello there.", "--voice", str(voices / "any.flac"), "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert [target.repo_id for target in env.ensure.call_args.args[0]] == [self.CUSTOM, self.BASE]


class TestMemoryWarning:
    def test_a_card_that_is_too_small_is_warned_before_the_licence_notice_and_the_download(self, env, monkeypatch, capsys):
        from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

        monkeypatch.setattr(audio_cli, "accelerator_memory_budget_or_none", lambda: MemoryBudget(5 * GIB, system_bytes=64 * GIB))
        order: list[str] = []
        monkeypatch.setattr(audio_cli, "print_licence_notice", lambda *a, **k: order.append("licence"))
        env.ensure.side_effect = lambda *a, **k: order.append("download") or "ready"

        with pytest.warns(UserWarning, match="Qwen3-TTS 1.7B.*6 GB.*5 GB") as warned:
            make(env, "-m", "qwen3_tts", "--prompt", "Hello there.", monkeypatch=monkeypatch)

        assert len(warned) == 1 and order == ["licence", "download"]

    def test_a_card_that_fits_is_not_warned(self, env, monkeypatch, recwarn):
        from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

        monkeypatch.setattr(audio_cli, "accelerator_memory_budget_or_none", lambda: MemoryBudget(24 * GIB, system_bytes=64 * GIB))

        make(env, "-m", "qwen3_tts", "--prompt", "Hello there.", monkeypatch=monkeypatch)

        assert not [w for w in recwarn if "GPU" in str(w.message)]


class TestDownload:
    def test_the_models_files_are_fetched_before_it_loads(self, env, monkeypatch):
        order: list[str] = []
        env.ensure.side_effect = lambda *a, **k: order.append("download") or "ready"
        env.backend.load.side_effect = lambda *a, **k: order.append("load")

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        assert order == ["download", "load"]
        assert [target.repo_id for target in env.ensure.call_args.args[0]][0] == "Comfy-Org/stable-audio-3"

    @pytest.mark.parametrize("outcome", ["stopped", "detached"])
    def test_quitting_during_the_download_makes_nothing(self, env, monkeypatch, outcome):
        env.ensure.return_value = outcome

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "-o", str(env.out))

        env.backend.load.assert_not_called()


class TestFailures:
    def test_running_out_of_memory_names_the_model(self, env, monkeypatch, capsys):
        import torch

        env.backend.generate.side_effect = torch.OutOfMemoryError("out of memory")

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "YuE2 3B" in error and "GPU memory" in error
        env.backend.close.assert_called_once()

    def test_hugging_face_out_of_reach_ends_with_a_message_about_the_connection(self, env, monkeypatch, capsys):
        env.ensure.return_value = "unchecked"
        monkeypatch.setattr(audio_cli, "require_checked", functools.partial(require_checked, present=lambda target: False))

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "Hugging Face" in error and "YuE2 3B" in error
        env.backend.load.assert_not_called()

    def test_files_missing_after_a_download_that_was_checked_do_not_blame_the_connection(self, env, monkeypatch, capsys):
        monkeypatch.setattr("zvisiongenerator.audio_runner.target_present_offline", lambda target: False)

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "YuE2 3B is not downloaded" in error and "Hugging Face" not in error
        env.backend.load.assert_not_called()

    def test_a_model_whose_files_are_not_in_the_cache_ends_with_its_own_message(self, env, monkeypatch, capsys):
        env.backend.load.side_effect = FileNotFoundError("Qwen3-TTS needs Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, which is not downloaded.")

        assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "x", "-o", str(env.out))


class TestCopyCli:
    """The command Copy CLI builds (``frontend/src/lib/state/cliCommand.ts``, same fixture in its Vitest) makes the job the form would."""

    @pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
    def test_the_copied_command_runs_to_the_request_of_the_form(self, env, monkeypatch, voices, case):
        command = case["command"].replace("/voices/", f"{voices}/")
        assert command.startswith("ziv audio ")

        requests = make(env, *shlex.split(command)[2:], monkeypatch=monkeypatch)

        expected = [case_request(item, voices) for item in case["requests"]]
        assert requests == expected
        assert len(list(env.out.glob("*.flac"))) == len(expected)


_REPLY = "Heavy rain drumming on a tin roof inside a large hall, thunder far away."
_FAST = "org/fast"


class _Enhancer:
    """A prompt enhancer that answers with *replies* in turn (an exception is raised), and notes in *events* when it is closed."""

    def __init__(self, events: list, *replies):
        self.events = events
        self.replies = list(replies)
        self.messages: list[list[dict[str, str]]] = []
        self.seeds: list[int] = []
        self.loaded: list[tuple[str, str | None]] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.messages.append(messages)
        self.seeds.append(seed)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        yield reply

    def close(self) -> None:
        self.events.append("enhancer_closed")


class TestEnhance:
    """``--enhance`` on a sound and on a song's style: the rewrite is what is made, and the take records both texts like a Web UI take."""

    @pytest.fixture
    def enhancer(self, env, monkeypatch, tmp_path):
        """Run with a config that has an enhancer and a fake one in its place, and note in the backend's events when the memory is given back."""
        fake = _Enhancer(env.backend.events, _REPLY)
        session = PromptEnhancerSession(lambda repo, revision: fake.loaded.append((repo, revision)) or fake, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)
        use_audio_models(monkeypatch, tmp_path)
        use_enhancer_models(monkeypatch, tmp_path, fast=_FAST)
        config = audio_config()
        monkeypatch.setattr(audio_cli, "load_config", lambda: config)
        monkeypatch.setattr(prompt_enhancer_session, "get_prompt_enhancer_session", lambda: session)
        monkeypatch.setattr(prompt_enhancer_session, "ensure_available", lambda repo, revision: True)
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", lambda: env.backend.events.append("memory_released"))
        fake.config = config
        return fake

    def test_a_sound_prompt_is_rewritten_and_the_take_records_the_own_text_and_how_it_was_written(self, env, enhancer, monkeypatch):
        argv = ["-m", "sa3", "--prompt", "rain on a roof", "--enhance", "space=large-hall,distance=far", "--enhance-note", "at night", "--seed", "4"]

        (request,) = make(env, *argv, monkeypatch=monkeypatch)

        assert (request.prompt, request.prompt_input) == (_REPLY, "rain on a roof")
        assert request.enhance_record == {
            "settings": {"space": "large-hall", "distance": "far", "note": "at night"},
            "model": _FAST,
            "tier": "fast",
            "format": "text",
            "task": "sound",
        }
        system = enhancer.messages[0][0]["content"]
        assert SPACE_AXIS.option("large-hall").instruction in system and "at night" in system
        assert enhancer.seeds[0] == 4  # the same command writes the same text

    def test_a_bare_enhance_is_a_general_rewrite(self, env, enhancer, monkeypatch):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--enhance", monkeypatch=monkeypatch)

        assert request.prompt == _REPLY and request.enhance_record["settings"] == {}

    def test_every_run_makes_the_one_rewrite(self, env, enhancer, monkeypatch):
        requests = make(env, "-m", "sa3", "--prompt", "rain", "--enhance", "--runs", "3", monkeypatch=monkeypatch)

        assert [request.prompt for request in requests] == [_REPLY] * 3 and len(enhancer.messages) == 1

    def test_a_songs_style_is_rewritten_for_the_model_and_told_the_lyrics(self, env, enhancer, monkeypatch):
        enhancer.replies = ["Folk, acoustic guitar, warm female vocal"]
        argv = ["-m", "yue2", "--prompt", "sad folk", "--lyrics", "[verse]\nwalking the shore", "--enhance", "genre=folk,instruments=guitar,singer=female"]

        (request,) = make(env, *argv, monkeypatch=monkeypatch)

        assert (request.prompt, request.prompt_input, request.lyrics) == ("Folk, acoustic guitar, warm female vocal", "sad folk", "[verse]\nwalking the shore")
        assert request.enhance_record["task"] == "style"
        assert request.enhance_record["settings"] == {"genre": "folk", "instruments": ["guitar"], "singer": "female"}
        assert "walking the shore" in enhancer.messages[0][0]["content"]

    def test_a_style_enhance_is_made_for_an_instrumental_without_a_voice(self, env, enhancer, monkeypatch):
        enhancer.replies = ["Ambient, piano"]

        (request,) = make(env, "-m", "yue2", "--prompt", "calm piano", "--enhance", "genre=ambient", monkeypatch=monkeypatch)

        assert request.prompt == "Ambient, piano" and "instrumental" in enhancer.messages[0][0]["content"]

    def test_the_enhancer_is_unloaded_and_its_memory_freed_before_the_audio_model_loads(self, env, enhancer, monkeypatch):
        make(env, "-m", "sa3", "--prompt", "rain", "--enhance", monkeypatch=monkeypatch)

        events = env.backend.events
        assert events.index("enhancer_closed") < events.index("memory_released") < events.index("load")

    def test_the_saved_take_is_the_one_a_web_ui_enhanced_take_saves(self, env, enhancer, monkeypatch):
        """The record, the own text and the enhanced text round-trip through Reuse, the Web UI's own validation and the Web UI's request."""
        make(env, "-m", "sa3", "--prompt", "rain on a roof", "--enhance", "space=large-hall", "--seed", "4", monkeypatch=monkeypatch)

        saved = read_config(next(env.out.glob("sound_*.flac")))

        assert (saved["prompt"], saved["prompt_input"]) == (_REPLY, "rain on a roof")
        facts = audio_asset_facts(saved, None)
        assert (facts["enhanced"], facts["prompt_input"], facts["enhance"]) == (True, "rain on a roof", saved["enhance"])
        assert parse_enhance_record(saved["enhance"], mode="audio", config=enhancer.config, platform_key=sys.platform) == saved["enhance"]
        preset = next(item for item in audio_cli.parse_audio_presets(enhancer.config) if item.id == "sa3")
        from_form = audio_request_from_fields(preset, "sound", AudioFields(prompt=_REPLY, prompt_input="rain on a roof", enhance_record=saved["enhance"]))
        payload = build_audio_config_payload(replace(from_form, seed=4), clip(0.2), parts=None, generation_time=1.0)
        assert {key: value for key, value in saved.items() if key != "generation"} == {key: value for key, value in payload.items() if key != "generation"}

    @pytest.mark.parametrize("flag", [["--enhance"], ["--enhance", "space=forest"], ["--enhance-note", "slowly"], ["--enhance-format", "text"]])
    def test_speech_has_no_enhance_and_the_refusal_names_the_script_helper(self, env, enhancer, monkeypatch, capsys, flag):
        error = refuses(monkeypatch, capsys, "-m", "qwen3_tts", "--prompt", "Hi", *flag)

        assert f"drop {flag[0]}" in error and "Script helper" in error
        assert enhancer.messages == []

    def test_speech_can_still_say_no_enhance(self, env, enhancer, monkeypatch):
        (request,) = make(env, "-m", "qwen3_tts", "--prompt", "Hi", "--no-enhance", monkeypatch=monkeypatch)

        assert request.prompt == "Hi" and request.enhance_record is None

    def test_a_json_caption_is_refused_for_audio(self, env, enhancer, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "rain", "--enhance-format", "json")

        assert "--enhance-format json" in error and "audio text is always plain text" in error

    def test_the_plain_format_is_the_default_and_enhances_nothing_by_itself(self, env, enhancer, monkeypatch):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--enhance-format", "text", monkeypatch=monkeypatch)

        assert request.prompt == "rain" and enhancer.messages == [] and enhancer.loaded == []

    def test_no_enhance_makes_the_take_as_written_and_never_loads_the_enhancer(self, env, enhancer, monkeypatch):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--no-enhance", monkeypatch=monkeypatch)

        assert (request.prompt, request.prompt_input, request.enhance_record) == ("rain", None, None)
        assert enhancer.loaded == []

    @pytest.mark.parametrize("flag", [["--enhance"], ["--enhance", "space=forest"], ["--enhance-note", "x"]])
    def test_no_enhance_cannot_be_combined_with_enhance(self, env, enhancer, monkeypatch, capsys, flag):
        assert "cannot be combined with --no-enhance" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "rain", "--no-enhance", *flag)

    def test_an_option_of_the_other_kind_or_an_unknown_one_is_refused_under_the_flag(self, env, enhancer, monkeypatch, capsys):
        assert "--enhance: 'genre' only applies to music" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "rain", "--enhance", "genre=jazz")
        assert "--enhance: Unknown space" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "rain", "--enhance", "space=nowhere")

    def test_the_direction_cannot_go_in_the_spec(self, env, enhancer, monkeypatch, capsys):
        assert "--enhance-note" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "rain", "--enhance", "note=slowly")

    def test_an_enhancer_model_is_only_picked_for_an_enhance_that_runs(self, env, enhancer, monkeypatch, capsys):
        assert "--enhance-model picks the enhancer for --enhance" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "rain", "--enhance-model", "smart")

    def test_a_custom_enhancer_model_is_loaded_and_recorded(self, env, enhancer, monkeypatch):
        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--enhance", "--enhance-model", "me/llm@abc123", monkeypatch=monkeypatch)

        assert enhancer.loaded == [("me/llm", "abc123")]
        assert (request.enhance_record["model"], request.enhance_record["tier"]) == ("me/llm@abc123", "custom")

    def test_a_rewrite_that_fails_warns_and_the_take_is_made_from_the_own_text(self, env, enhancer, monkeypatch):
        enhancer.replies = [RuntimeError("template rejected")]

        with pytest.warns(UserWarning, match=r"Prompt enhancement failed \(template rejected\); using the original prompt"):
            (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--enhance", monkeypatch=monkeypatch)

        assert (request.prompt, request.prompt_input, request.enhance_record) == ("rain", None, None)
        assert "enhancer_closed" in env.backend.events

    def test_next_pressed_before_the_rewrite_skips_it(self, env, enhancer, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli.SkipSignal, "start", lambda self: self.queue_action("skip"))

        (request,) = make(env, "-m", "sa3", "--prompt", "rain", "--enhance", monkeypatch=monkeypatch)

        assert request.prompt == "rain" and request.enhance_record is None and enhancer.loaded == []
        assert "Skipped the rewrite." in capsys.readouterr().out

    def test_quit_pressed_before_the_rewrite_makes_nothing(self, env, enhancer, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli.SkipSignal, "start", lambda self: self.queue_action("quit"))

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "--enhance", "-o", str(env.out))

        assert capsys.readouterr().out.rstrip().endswith("⏹ Quitting batch...")
        env.backend.load.assert_not_called()
        assert enhancer.loaded == []

    def test_the_command_copied_from_an_enhanced_take_makes_the_enhanced_text_as_the_prompt_as_image_and_video_do(self, env, enhancer, monkeypatch):
        """Copy CLI of an enhanced take names the Enhanced text with ``--prompt`` (``cliCommand.test.ts`` has the same pair), so the command rewrites nothing again."""
        command = "ziv audio --model sa3 --kind sound --prompt 'Heavy rain on a tin roof' --seed 3"

        (request,) = make(env, *shlex.split(command)[2:], monkeypatch=monkeypatch)

        assert (request.prompt, request.prompt_input, request.enhance_record) == ("Heavy rain on a tin roof", None, None)
        assert enhancer.loaded == []
